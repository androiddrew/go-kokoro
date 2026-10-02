package assets

import (
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"unicode/utf8"
)

// defaultVocab is the "vocab" table from hexgrad/Kokoro-82M config.json
// (Apache-2.0, commit f3ff3571791e39611d31c381e3a41a3af07b4987), the vocabulary
// the Kokoro v1.0 model was trained with. thewh1teagle/kokoro-onnx ships an
// identical copy. See THIRD_PARTY.md.
//
//go:embed vocab.json
var defaultVocab []byte

// LoadVocab reads a {"vocab": {phoneme: token}} file. An empty path selects the
// embedded Kokoro v1.0 vocabulary.
func LoadVocab(path string) (map[rune]int64, error) {
	if path == "" {
		return ParseVocab(defaultVocab)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("load vocabulary: %w", err)
	}
	return ParseVocab(data)
}

// ParseVocab validates a vocabulary. The pinned model has 178 embedding
// entries; zero is reserved for boundary padding.
func ParseVocab(data []byte) (map[rune]int64, error) {
	var config struct {
		Vocab map[string]int64 `json:"vocab"`
	}
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, err
	}
	if len(config.Vocab) == 0 {
		return nil, errors.New("empty vocabulary")
	}
	vocab, ids := make(map[rune]int64), make(map[int64]bool)
	for key, id := range config.Vocab {
		if !utf8.ValidString(key) || utf8.RuneCountInString(key) != 1 || id < 1 || id > 177 || ids[id] {
			return nil, fmt.Errorf("invalid or duplicate vocabulary entry %q: %d", key, id)
		}
		c, _ := utf8.DecodeRuneInString(key)
		vocab[c], ids[id] = id, true
	}
	if vocab[' '] != 16 {
		return nil, errors.New("incompatible vocabulary: expected space token 16")
	}
	return vocab, nil
}
