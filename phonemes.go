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
	return prepareWithOptions(phonemes, voice, speed, voices, vocab, StreamOptions{})
}

// splitStreamPhonemes uses the original packing unless a limit is requested.
// Custom limits prefer the last punctuation boundary within the budget, then
// whitespace, then a Unicode code-point boundary. No non-whitespace is dropped.
func splitStreamPhonemes(phonemes string, options StreamOptions) ([]string, error) {
	if options.MaxChunkPhonemes < 0 || options.MaxChunkPhonemes > MaxPhonemes {
		return nil, fmt.Errorf("MaxChunkPhonemes must be between 0 and %d", MaxPhonemes)
	}
	maximum := options.MaxChunkPhonemes
	if maximum == 0 {
		maximum = MaxPhonemes
	}
	if options.FirstChunkPhonemes < 0 || options.FirstChunkPhonemes > maximum {
		return nil, fmt.Errorf("FirstChunkPhonemes must be between 0 and %d", maximum)
	}
	if options == (StreamOptions{}) {
		return SplitPhonemes(phonemes)
	}
	if !utf8.ValidString(phonemes) {
		return nil, errors.New("phonemes must be valid UTF-8")
	}
	runes := []rune(strings.TrimSpace(phonemes))
	if len(runes) == 0 {
		return nil, errors.New("phonemes are empty")
	}
	limit := maximum
	if options.FirstChunkPhonemes > 0 {
		limit = options.FirstChunkPhonemes
	}
	var chunks []string
	for len(runes) > 0 {
		cut := len(runes)
		if cut > limit {
			cut = limit
			punctuation, space := 0, 0
			for i := 0; i <= limit; i++ {
				if unicode.IsSpace(runes[i]) {
					space = i
				}
				if i < limit && strings.ContainsRune(".,!?;", runes[i]) {
					punctuation = i + 1
				}
			}
			if punctuation > 0 {
				cut = punctuation
			} else if space > 0 {
				cut = space
			}
		}
		chunks = append(chunks, strings.TrimSpace(string(runes[:cut])))
		runes = runes[cut:]
		for len(runes) > 0 && unicode.IsSpace(runes[0]) {
			runes = runes[1:]
		}
		limit = maximum
	}
	return chunks, nil
}

func prepareWithOptions(phonemes, voice string, speed float32, voices *assets.Voices, vocab map[rune]int64, options StreamOptions) ([]Chunk, error) {
	if !(speed >= 0.5 && speed <= 2) {
		return nil, errors.New("speed must be finite and between 0.5 and 2.0")
	}
	parts, err := splitStreamPhonemes(phonemes, options)
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
