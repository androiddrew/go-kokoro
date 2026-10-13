package text

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"strings"
	"time"
	"unicode"

	g2p "github.com/androiddrew/go-g2p"
	"github.com/androiddrew/go-g2p/fallback"
	"github.com/androiddrew/go-g2p/fallback/espeak"
	"github.com/androiddrew/go-g2p/fallback/neural"
	"github.com/androiddrew/go-ttsnorm/sentence"

	kokoro "github.com/androiddrew/go-kokoro"
)

// SampleRate is the output rate: Kokoro's.
const SampleRate = kokoro.SampleRate

// Pronunciation fallbacks for words missing from the dictionaries.
const (
	// FallbackNeural uses go-g2p's built-in neural models. It is the default.
	FallbackNeural = neural.Name
	// FallbackESpeak runs an installed eSpeak-ng (GPL-3.0-or-later) as a
	// separate program.
	FallbackESpeak = espeak.Name
)

// FrontendConfig configures the pronunciation frontend.
type FrontendConfig struct {
	// G2P configures the dictionary and POS engine. Its Fallback is ignored:
	// NewFrontend creates the fallback below and owns it.
	G2P g2p.Config
	// Fallback is FallbackNeural (or empty) or FallbackESpeak.
	Fallback string
	// FallbackModelDir overrides the neural fallback's embedded models.
	// Usually empty.
	FallbackModelDir string
	// ESpeak is the eSpeak-ng executable for FallbackESpeak; empty selects
	// espeak-ng on PATH.
	ESpeak string
}

// Frontend is a G2P engine together with the fallback it owns.
type Frontend struct {
	*g2p.Engine
	fallback fallback.Engine
}

// NewFrontend loads the G2P engine and its fallback. Close is mandatory.
func NewFrontend(c FrontendConfig) (_ *Frontend, err error) {
	f := &Frontend{}
	defer func() {
		if err != nil {
			err = errors.Join(err, f.Close())
		}
	}()
	switch c.Fallback {
	case "", FallbackNeural:
		f.fallback, err = neural.New(neural.Config{ModelDir: c.FallbackModelDir, ORTLibrary: c.G2P.ORTLibrary})
	case FallbackESpeak:
		f.fallback, err = espeak.New(espeak.Config{Executable: c.ESpeak})
	default:
		err = fmt.Errorf("fallback %q: use %q or %q", c.Fallback, FallbackNeural, FallbackESpeak)
	}
	if err != nil {
		return nil, err
	}
	gc := c.G2P
	gc.Fallback = f.fallback
	if f.Engine, err = g2p.New(gc); err != nil {
		return nil, err
	}
	return f, nil
}

// Close shuts the engine before the fallback it uses.
func (f *Frontend) Close() error {
	var err error
	if f.Engine != nil {
		err = f.Engine.Close()
	}
	if f.fallback != nil {
		err = errors.Join(err, f.fallback.Close())
	}
	return err
}

// Config loads a whole pipeline with New.
type Config struct {
	// ORTLibrary selects the ONNX Runtime library for the frontend and Kokoro;
	// it overrides the ORTLibrary fields below. Empty means the caller owns
	// the environment, as in the kokoro and g2p packages.
	ORTLibrary string
	Frontend   FrontendConfig
	Synthesis  kokoro.Config
}

// Pipeline is a pronunciation frontend and a Kokoro engine. Each serializes
// its own calls, so a pipeline serves one stream at a time at full speed.
type Pipeline struct {
	Frontend  *Frontend
	Synthesis *kokoro.Engine
}

// New loads the frontend and the Kokoro engine. Close is mandatory.
// Applications that load them separately, to time or warm each, can build a
// Pipeline from its fields instead.
func New(c Config) (_ *Pipeline, err error) {
	p := &Pipeline{}
	defer func() {
		if err != nil {
			err = errors.Join(err, p.Close())
		}
	}()
	c.Frontend.G2P.ORTLibrary = c.ORTLibrary
	if p.Frontend, err = NewFrontend(c.Frontend); err != nil {
		return nil, err
	}
	c.Synthesis.ORTLibrary = c.ORTLibrary
	if p.Synthesis, err = kokoro.New(c.Synthesis); err != nil {
		return nil, err
	}
	return p, nil
}

// Close releases the model and then the frontend. The ONNX Runtime
// environment stays loaded.
func (p *Pipeline) Close() error {
	var err error
	if p.Synthesis != nil {
		err = p.Synthesis.Close()
		p.Synthesis = nil
	}
	if p.Frontend != nil {
		err = errors.Join(err, p.Frontend.Close())
		p.Frontend = nil
	}
	return err
}

// Request is one text to stream.
type Request struct {
	Text  string
	Voice string  // a Kokoro English voice; its prefix selects the dialect (see VoiceDialect)
	Speed float32 // 0.5 to 2.0
	Prep  PrepOptions
	Trim  bool          // trim each model run's leading and trailing silence
	Gap   time.Duration // silence before each sentence after the first
	// Chunking limits how many phonemes each model run speaks. The zero value
	// runs whole sentences, up to Kokoro's limit. Smaller runs start audio
	// sooner and let cancellation take effect sooner, since a run in progress
	// can't be interrupted, at some cost to prosody.
	Chunking kokoro.StreamOptions
	// Strict fails on an unresolved or generation-limited pronunciation.
	// Otherwise it is spoken as far as known and reported in Chunk.Diagnostics.
	Strict bool
}

// Chunk is one model run's audio: a sentence, or part of one.
type Chunk struct {
	PCM         []float32 // mono at SampleRate, including any leading gap; owned by the caller
	Text        string    // the prepared sentence the audio is from
	Phonemes    string    // the phonemes this run spoke
	Tokens      int       // phoneme tokens the model ran on
	Diagnostics []g2p.Diagnostic
}

// ErrPronunciation is returned with Strict for text the frontend can't
// fully pronounce.
var ErrPronunciation = errors.New("incomplete pronunciation")

// errNothing is returned for text with no sound to make.
var errNothing = errors.New("the text has nothing to speak")

// unknown is the phoneme Misaki writes for a sound it couldn't resolve,
// which Kokoro can't speak.
const unknown = "❓"

// splitter keeps the first sentence short, so streamed audio starts soon.
var splitter = sentence.Splitter{}

// errStopped ends a sentence's synthesis when the consumer stops iterating.
var errStopped = errors.New("stream stopped by the consumer")

// Stream prepares r's text, splits it into sentences and yields the audio of
// each model run as it is made. It stops at the first error, which it yields
// last. Cancelling ctx stops it between model runs.
//
// Each sentence's audio is delivered while the Kokoro engine is locked, so
// the loop body must not call p.Synthesis; hand the audio to a playback
// queue rather than playing it inline, so synthesis overlaps playback.
func (p *Pipeline) Stream(ctx context.Context, r Request) iter.Seq2[Chunk, error] {
	return func(yield func(Chunk, error) bool) {
		dialect, ok := VoiceDialect(r.Voice)
		if !ok {
			yield(Chunk{}, fmt.Errorf("voice %q is not a US or UK English voice", r.Voice))
			return
		}
		gap := make([]float32, int(r.Gap.Seconds()*SampleRate))
		first := true
		for _, text := range splitter.Split(Prepare(r.Text, r.Prep)) {
			if err := ctx.Err(); err != nil {
				yield(Chunk{}, err)
				return
			}
			result, err := p.Frontend.Phonemize(ctx, g2p.Request{Text: text, Dialect: dialect, AllowTruncated: !r.Strict})
			var incomplete *g2p.IncompleteError
			if errors.As(err, &incomplete) {
				if r.Strict {
					err = fmt.Errorf("%w: %s", ErrPronunciation, describe(result.Diagnostics))
				} else {
					// Speak what is known; the diagnostics say what isn't.
					result.Phonemes = strings.ReplaceAll(result.Phonemes, unknown, "")
					err = nil
				}
			}
			if err != nil {
				yield(Chunk{}, err)
				return
			}
			if !speakable(result.Phonemes) {
				continue
			}
			sentenceStart := true
			_, err = p.Synthesis.SynthesizeStream(ctx,
				kokoro.Request{Phonemes: result.Phonemes, Voice: r.Voice, Speed: r.Speed, Trim: r.Trim},
				r.Chunking,
				func(audio kokoro.ChunkAudio) error {
					c := Chunk{PCM: audio.Samples(), Text: text, Phonemes: audio.Phonemes, Tokens: len(audio.Tokens) - 2}
					if sentenceStart {
						c.Diagnostics = result.Diagnostics
						if !first {
							c.PCM = append(gap[:len(gap):len(gap)], c.PCM...)
						}
					}
					sentenceStart, first = false, false
					if !yield(c, nil) {
						return errStopped
					}
					return nil
				})
			if errors.Is(err, errStopped) {
				return
			}
			if err != nil {
				yield(Chunk{}, err)
				return
			}
		}
		if first {
			yield(Chunk{}, errNothing)
		}
	}
}

// speakable reports whether phonemes have a sound to make, rather than only
// punctuation, which the model can't run on alone.
func speakable(phonemes string) bool {
	return strings.ContainsFunc(phonemes, func(r rune) bool { return !unicode.IsPunct(r) && !unicode.IsSpace(r) })
}

func describe(diagnostics []g2p.Diagnostic) string {
	parts := make([]string, len(diagnostics))
	for i, d := range diagnostics {
		parts[i] = fmt.Sprintf("%s %q", d.Code, d.Text)
	}
	return strings.Join(parts, ", ")
}
