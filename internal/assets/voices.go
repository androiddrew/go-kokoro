// Package assets reads the Kokoro v1.0 voice archive (a NumPy .npz file) and
// the vocabulary.
package assets

import (
	"archive/zip"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"regexp"
	"sort"
	"strings"
)

const (
	VoiceRows  = 510
	StyleWidth = 256
	voiceBytes = VoiceRows * StyleWidth * 4
)

type Voices struct{ data map[string][]float32 }

var voiceName = regexp.MustCompile(`^[a-z][a-z0-9_]*\.npy$`)

// NumPy's literal header, deliberately restricted to the pinned numeric layout.
// Pickled objects, Fortran layout and other dtypes are rejected.
var npyHeader = regexp.MustCompile(`^\s*\{\s*'descr':\s*'<f4',\s*'fortran_order':\s*False,\s*'shape':\s*\(510,\s*1,\s*256\),?\s*\}\s*$`)

func LoadVoices(path string) (*Voices, error) {
	z, err := zip.OpenReader(path)
	if err != nil {
		return nil, fmt.Errorf("open voices NPZ: %w", err)
	}
	defer z.Close()
	if len(z.File) == 0 || len(z.File) > 256 {
		return nil, errors.New("voices NPZ must contain 1..256 arrays")
	}
	v := &Voices{data: make(map[string][]float32)}
	for _, f := range z.File {
		if !voiceName.MatchString(f.Name) || f.UncompressedSize64 > voiceBytes+4096+12 {
			return nil, fmt.Errorf("invalid voice entry %q or oversized array", f.Name)
		}
		name := strings.TrimSuffix(f.Name, ".npy")
		if _, ok := v.data[name]; ok {
			return nil, fmt.Errorf("duplicate voice %q", name)
		}
		r, err := f.Open()
		if err != nil {
			return nil, err
		}
		data, err := io.ReadAll(io.LimitReader(r, voiceBytes+4096+13))
		err = errors.Join(err, r.Close())
		if err != nil {
			return nil, fmt.Errorf("read voice %s: %w", name, err)
		}
		values, err := decodeNPY(data)
		if err != nil {
			return nil, fmt.Errorf("voice %s: %w", name, err)
		}
		v.data[name] = values
	}
	return v, nil
}

func decodeNPY(data []byte) ([]float32, error) {
	if len(data) < 10 || string(data[:6]) != "\x93NUMPY" || data[7] != 0 {
		return nil, errors.New("invalid NPY header")
	}
	start, n := 10, uint32(binary.LittleEndian.Uint16(data[8:10]))
	switch data[6] {
	case 1:
	case 2, 3:
		if len(data) < 12 {
			return nil, errors.New("truncated NPY header")
		}
		start, n = 12, binary.LittleEndian.Uint32(data[8:12])
	default:
		return nil, errors.New("unsupported NPY version")
	}
	if n == 0 || n > 4096 || len(data) != start+int(n)+voiceBytes {
		return nil, errors.New("invalid NPY size; expected float32 [510,1,256]")
	}
	if !npyHeader.Match(data[start : start+int(n)]) {
		return nil, errors.New("expected C-order little-endian float32 [510,1,256] NPY array")
	}
	data = data[start+int(n):]
	values := make([]float32, VoiceRows*StyleWidth)
	for i := range values {
		values[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
		if math.IsNaN(float64(values[i])) || math.IsInf(float64(values[i]), 0) {
			return nil, errors.New("non-finite voice value")
		}
	}
	return values, nil
}

func (v *Voices) Names() []string {
	names := make([]string, 0, len(v.data))
	for name := range v.data {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func (v *Voices) Style(name string, length int) ([]float32, error) {
	data, ok := v.data[name]
	if !ok {
		return nil, fmt.Errorf("unknown voice %q; use Voices() to list available names", name)
	}
	if length < 1 || length >= VoiceRows {
		return nil, fmt.Errorf("voice style length must be 1..%d, got %d", VoiceRows-1, length)
	}
	return append([]float32(nil), data[length*StyleWidth:(length+1)*StyleWidth]...), nil
}
