// Package audio implements the pinned kokoro-onnx mono postprocessing and WAV
// encoding. Trim follows its librosa-derived trim.py; see THIRD_PARTY.md.
package audio

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
)

const SampleRate = 24000

func Validate(samples []float32) error {
	if len(samples) == 0 {
		return errors.New("empty audio")
	}
	for i, sample := range samples {
		if math.IsNaN(float64(sample)) || math.IsInf(float64(sample), 0) {
			return fmt.Errorf("non-finite audio sample %d", i)
		}
	}
	return nil
}

// TrimBounds returns the half-open interval from centered 2048-sample RMS
// frames, a 512-sample hop, and a strict -60 dB threshold relative to peak RMS.
// As in librosa, uniform digital silence remains untrimmed.
func TrimBounds(samples []float32) (int, int) {
	if len(samples) == 0 {
		return 0, 0
	}
	power := make([]float64, 1+len(samples)/512)
	peak := 1e-10
	for frame := range power {
		var sum float64
		for i := max(0, frame*512-1024); i < min(len(samples), frame*512+1024); i++ {
			sum += float64(samples[i] * samples[i])
		}
		power[frame] = sum / 2048
		peak = max(peak, power[frame])
	}
	first, last := -1, -1
	for i, p := range power {
		if 10*math.Log10(max(1e-10, p))-10*math.Log10(peak) > -60 {
			if first < 0 {
				first = i
			}
			last = i
		}
	}
	if first < 0 {
		return 0, 0
	}
	return first * 512, min(len(samples), (last+1)*512)
}

// WriteWAV writes mono PCM16 at 24 kHz. The pinned SoundFile/libsndfile path
// rounds to signed PCM32 (nearest/even), then keeps its upper 16 bits. This is
// almost floor quantization, but differs near integer boundaries. Finite
// out-of-range samples are saturated.
// The caller owns the writer and must handle any Close error itself.
func WriteWAV(w io.Writer, samples []float32) error {
	if err := Validate(samples); err != nil {
		return err
	}
	if uint64(len(samples))*2 > math.MaxUint32-36 {
		return errors.New("audio exceeds RIFF WAV size limit")
	}
	n := uint32(len(samples) * 2)
	header := make([]byte, 44)
	copy(header, "RIFF")
	binary.LittleEndian.PutUint32(header[4:], 36+n)
	copy(header[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(header[16:], 16)
	binary.LittleEndian.PutUint16(header[20:], 1)
	binary.LittleEndian.PutUint16(header[22:], 1)
	binary.LittleEndian.PutUint32(header[24:], SampleRate)
	binary.LittleEndian.PutUint32(header[28:], SampleRate*2)
	binary.LittleEndian.PutUint16(header[32:], 2)
	binary.LittleEndian.PutUint16(header[34:], 16)
	copy(header[36:], "data")
	binary.LittleEndian.PutUint32(header[40:], n)
	if err := writeFull(w, header); err != nil {
		return err
	}
	buffer := make([]byte, 8192)
	for start := 0; start < len(samples); start += len(buffer) / 2 {
		end := min(len(samples), start+len(buffer)/2)
		for i, x := range samples[start:end] {
			pcm32 := int64(max(math.MinInt32, min(math.MaxInt32, math.RoundToEven(float64(x)*2147483648))))
			value := int16(pcm32 >> 16)
			binary.LittleEndian.PutUint16(buffer[i*2:], uint16(value))
		}
		if err := writeFull(w, buffer[:(end-start)*2]); err != nil {
			return err
		}
	}
	return nil
}

func writeFull(w io.Writer, data []byte) error {
	n, err := w.Write(data)
	if err == nil && n != len(data) {
		return io.ErrShortWrite
	}
	return err
}
