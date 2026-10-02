package assets

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func npy(header string) []byte {
	b := new(bytes.Buffer)
	b.WriteString("\x93NUMPY\x01\x00")
	_ = binary.Write(b, binary.LittleEndian, uint16(len(header)))
	b.WriteString(header)
	b.Write(make([]byte, voiceBytes))
	return b.Bytes()
}

const validHeader = "{'descr': '<f4', 'fortran_order': False, 'shape': (510, 1, 256), }\n"

func TestDecodeNPY(t *testing.T) {
	good := npy(validHeader)
	binary.LittleEndian.PutUint32(good[len(good)-4:], math.Float32bits(0.25))
	values, err := decodeNPY(good)
	if err != nil || len(values) != VoiceRows*StyleWidth || values[len(values)-1] != 0.25 {
		t.Fatalf("decode: %v", err)
	}
	for name, data := range map[string][]byte{
		"truncated": good[:len(good)-1], "trailing": append(append([]byte(nil), good...), 0),
		"object":     npy(strings.Replace(validHeader, "<f4", "|O8", 1)),
		"endian":     npy(strings.Replace(validHeader, "<f4", ">f4", 1)),
		"fortran":    npy(strings.Replace(validHeader, "False", "True", 1)),
		"dimensions": npy(strings.Replace(validHeader, "256", "255", 1)),
		"not numpy":  []byte("not an array"),
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := decodeNPY(data); err == nil {
				t.Fatal("accepted malformed array")
			}
		})
	}
	binary.LittleEndian.PutUint32(good[len(good)-4:], math.Float32bits(float32(math.Inf(1))))
	if _, err := decodeNPY(good); err == nil {
		t.Fatal("accepted non-finite value")
	}
}

func TestVoiceArchive(t *testing.T) {
	for _, names := range [][]string{{"af_test.npy"}, {}, {"../af_test.npy"}, {"af_test.npy", "af_test.npy"}, {"af_test.txt"}} {
		path := filepath.Join(t.TempDir(), "voices.bin")
		f, err := os.Create(path)
		if err != nil {
			t.Fatal(err)
		}
		z := zip.NewWriter(f)
		for _, name := range names {
			w, err := z.Create(name)
			if err != nil {
				t.Fatal(err)
			}
			if _, err = w.Write(npy(validHeader)); err != nil {
				t.Fatal(err)
			}
		}
		if err = z.Close(); err != nil {
			t.Fatal(err)
		}
		if err = f.Close(); err != nil {
			t.Fatal(err)
		}
		v, err := LoadVoices(path)
		if len(names) != 1 || names[0] != "af_test.npy" {
			if err == nil {
				t.Fatalf("accepted %v", names)
			}
			continue
		}
		if err != nil {
			t.Fatal(err)
		}
		if _, err := v.Style("missing", 1); err == nil {
			t.Fatal("missing voice accepted")
		}
		for _, length := range []int{0, 510, -1} {
			if _, err := v.Style("af_test", length); err == nil {
				t.Fatal("bad style row accepted")
			}
		}
		style, err := v.Style("af_test", 509)
		if err != nil || len(style) != 256 {
			t.Fatal("last row", err)
		}
		style[0] = 1
		other, _ := v.Style("af_test", 509)
		if other[0] != 0 {
			t.Fatal("style aliases voice storage")
		}
	}
}

func TestVocabulary(t *testing.T) {
	vocab, err := LoadVocab("")
	if err != nil || len(vocab) != 114 || vocab['ˈ'] == 0 {
		t.Fatalf("embedded vocabulary: %d entries, %v", len(vocab), err)
	}
	for _, text := range []string{`{}`, `{"vocab":{" ":16,"ab":1}}`, `{"vocab":{" ":16,"a":178}}`, `{"vocab":{" ":16,"a":0}}`, `{"vocab":{" ":16,"a":16}}`} {
		path := filepath.Join(t.TempDir(), "vocab.json")
		if err := os.WriteFile(path, []byte(text), 0600); err != nil {
			t.Fatal(err)
		}
		if _, err := LoadVocab(path); err == nil {
			t.Fatalf("accepted %s", text)
		}
	}
}
