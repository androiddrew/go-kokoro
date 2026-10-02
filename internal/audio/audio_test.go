package audio

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"testing"
)

func TestPCM16(t *testing.T) {
	samples := []float32{-2, -1, -0.1, -1.0 / 65536, 0, 1.0 / 65536, 0.1, 1, 2, 0.00027465802850201726}
	want := []int16{-32768, -32768, -3277, -1, 0, 0, 3276, 32767, 32767, 9}
	var b bytes.Buffer
	if err := WriteWAV(&b, samples); err != nil {
		t.Fatal(err)
	}
	data := b.Bytes()
	if len(data) != 44+len(samples)*2 || string(data[8:12]) != "WAVE" || binary.LittleEndian.Uint32(data[24:]) != 24000 || binary.LittleEndian.Uint16(data[22:]) != 1 {
		t.Fatal("invalid WAV header")
	}
	for i, v := range want {
		if int16(binary.LittleEndian.Uint16(data[44+2*i:])) != v {
			t.Fatalf("PCM %d", i)
		}
	}
	for _, bad := range [][]float32{nil, {float32(math.NaN())}, {float32(math.Inf(-1))}} {
		if err := WriteWAV(io.Discard, bad); err == nil {
			t.Fatal("accepted invalid audio")
		}
	}
	if err := WriteWAV(shortWriter{}, samples); !errors.Is(err, io.ErrShortWrite) {
		t.Fatal(err)
	}
}

type shortWriter struct{}

func (shortWriter) Write(b []byte) (int, error) { return len(b) - 1, nil }

func TestTrim(t *testing.T) {
	for _, length := range []int{1, 511, 512, 2048, 9000} {
		start, end := TrimBounds(make([]float32, length))
		if start != 0 || end != length {
			t.Fatalf("silent input was trimmed: %d:%d", start, end)
		}
	}
	samples := make([]float32, 12000)
	for i := 4096; i < 6144; i++ {
		samples[i] = 0.25
	}
	// Independent frame calculation: first overlapping center is 3584; last is
	// 6656, and librosa includes one following hop (7168).
	start, end := TrimBounds(samples)
	if start != 3584 || end != 7168 {
		t.Fatalf("trim = %d:%d", start, end)
	}
}
