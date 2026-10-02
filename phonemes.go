// Package kokoro synthesizes precomputed phonemes using the pinned Kokoro v1.0
// ONNX model. It has no dependency on a CLI or pronunciation fallback.
package kokoro

import (
	"errors"
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/androiddrew/go-kokoro/internal/assets"
)

// MaxPhonemes is the longest chunk the model accepts: the voice table has 510
// rows and a chunk selects row len(tokens).
const MaxPhonemes = assets.VoiceRows - 1

// Chunk is one model input: padded tokens plus the selected style row and speed.
type Chunk struct {
	Phonemes   string    `json:"phonemes"`
	Tokens     []int64   `json:"tokens"`
	StyleIndex int       `json:"style_index"`
	Style      []float32 `json:"style"`
	Speed      float32   `json:"speed"`
}

// SplitPhonemes preserves kokoro-onnx 0.4.9 punctuation packing for safe inputs.
// Oversized spans split at the last space, then at Unicode code-point boundaries.
// Unlike upstream, it never emits empty chunks or truncates an oversized span.
func SplitPhonemes(phonemes string) ([]string, error) {
	if !utf8.ValidString(phonemes) {
		return nil, errors.New("phonemes must be valid UTF-8")
	}
	var parts []string
	start := 0
	for i, r := range phonemes {
		if strings.ContainsRune(".,!?;", r) {
			parts = append(parts, phonemes[start:i], string(r))
			start = i + 1 // these five punctuation marks are ASCII
		}
	}
	parts = append(parts, phonemes[start:])
	var chunks []string
	current := ""
	flush := func() {
		if c := strings.TrimSpace(current); c != "" {
			chunks = append(chunks, c)
		}
		current = ""
	}
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		if utf8.RuneCountInString(current)+utf8.RuneCountInString(part)+1 >= 510 {
			flush()
		}
		runes := []rune(part)
		for len(runes) > MaxPhonemes {
			cut := MaxPhonemes
			for i := cut; i > 0; i-- {
				if unicode.IsSpace(runes[i]) {
					cut = i
					break
				}
			}
			chunks = append(chunks, strings.TrimSpace(string(runes[:cut])))
			runes = []rune(strings.TrimSpace(string(runes[cut:])))
		}
		part = string(runes)
		if current != "" && !strings.Contains(".,!?;", part) {
			current += " "
		}
		current += part
	}
	flush()
	if len(chunks) == 0 {
		return nil, errors.New("phonemes are empty")
	}
	return chunks, nil
}

func prepare(phonemes, voice string, speed float32, voices *assets.Voices, vocab map[rune]int64) ([]Chunk, error) {
	if !(speed >= 0.5 && speed <= 2) {
		return nil, errors.New("speed must be finite and between 0.5 and 2.0")
	}
	parts, err := SplitPhonemes(phonemes)
	if err != nil {
		return nil, err
	}
	chunks := make([]Chunk, 0, len(parts))
	for _, part := range parts {
		ids := []int64{0}
		for _, r := range part {
			id, ok := vocab[r]
			if !ok {
				return nil, fmt.Errorf("unsupported phoneme %q (U+%04X); resolve it in the frontend or vocabulary", r, r)
			}
			ids = append(ids, id)
		}
		length := len(ids) - 1
		style, err := voices.Style(voice, length)
		if err != nil {
			return nil, err
		}
		chunks = append(chunks, Chunk{part, append(ids, 0), length, style, speed})
	}
	return chunks, nil
}
