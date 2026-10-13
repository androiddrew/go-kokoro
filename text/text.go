// Package text speaks US and UK English text with Kokoro: it prepares the
// text (markdown as prose, numbers, dates, URLs and the like as words), splits
// it into sentences, pronounces each with go-g2p and streams Kokoro's audio
// as each model run completes.
//
// The root kokoro package takes phonemes; this package is the text front end
// for it. It shares the process's ONNX Runtime environment with the Kokoro
// engine, so both must be given the same ORTLibrary selector.
package text

import (
	"strings"

	g2p "github.com/androiddrew/go-g2p"
	"github.com/androiddrew/go-ttsnorm/markdown"
	"github.com/androiddrew/go-ttsnorm/normalize"
)

// PrepOptions selects the text preparation steps.
type PrepOptions struct {
	Markdown  bool // read markdown as prose: markup, code blocks and link targets are dropped
	Normalize bool // read numbers, dates, times, money, units, URLs and the like as words
}

// Prepare applies the selected steps to s. Misaki's inline pronunciation
// overrides, such as [Kubernetes](/kˌubəɹnˈɛtiz/), pass through both
// unchanged.
func Prepare(s string, o PrepOptions) string {
	if o.Markdown {
		s = markdown.ToSpeech(s, markdown.KeepOverrides())
	}
	if o.Normalize {
		// Misaki reads punctuation for prosody and pronunciation, so it stays,
		// and reads capitals, not lower-case letters, as letter names.
		s = normalize.Text(s, normalize.KeepPunctuation(), normalize.UpperLetters(), normalize.ProtectOverrides())
	}
	return s
}

// VoiceDialect returns the dialect a Kokoro English voice speaks: US for a*
// voices such as af_heart, GB for b* voices such as bf_emma.
func VoiceDialect(voice string) (g2p.Dialect, bool) {
	switch {
	case strings.HasPrefix(voice, "a"):
		return g2p.US, true
	case strings.HasPrefix(voice, "b"):
		return g2p.GB, true
	}
	return "", false
}
