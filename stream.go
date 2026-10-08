package kokoro

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/androiddrew/go-kokoro/internal/audio"
)

// StreamOptions controls phoneme chunking for SynthesizeStream. The zero value
// preserves Synthesize's chunk boundaries. Smaller chunks can reduce time to
// first audio, at the cost of more native runs and potentially different prosody.
// Custom limits prefer the last punctuation boundary within the budget, then
// whitespace, then a Unicode code-point boundary. Short remaining input stays
// together; no non-whitespace phonemes are dropped.
type StreamOptions struct {
	// MaxChunkPhonemes limits each chunk's unpadded phoneme count (Unicode
	// code points, including spaces and punctuation). Zero selects MaxPhonemes.
	MaxChunkPhonemes int
	// FirstChunkPhonemes optionally sets a smaller limit for the first chunk.
	// Zero uses MaxChunkPhonemes. It must not exceed the effective maximum.
	FirstChunkPhonemes int
}

// Samples returns the mono 24 kHz audio selected by the trim bounds. It aliases
// Raw; the caller owns both and may retain or modify them after delivery.
func (c ChunkAudio) Samples() []float32 { return c.Raw[c.TrimStart:c.TrimEnd:c.TrimEnd] }

// SynthesizeStream delivers each completed chunk to yield, in order, before
// starting the next native run. It retains no aggregate waveform or delivered
// audio. All request inputs are prepared and validated before the first run.
//
// yield must be non-nil. It runs synchronously with the engine locked, so it must
// not call back into this engine. A slow callback applies backpressure; enqueue
// audio for a separate playback consumer to overlap playback with synthesis.
// The engine starts no goroutines. Callbacks that block should observe ctx.
//
// Cancellation is checked around native runs and callbacks, including after the
// final callback. An in-flight native run cannot be interrupted. A callback
// error or cancellation stops subsequent delivery; previously delivered audio
// remains owned by the caller and is not rolled back. Callback errors are
// wrapped with the zero-based chunk index and preserve errors.Is/errors.As.
// Returned timings include work completed before an error, excluding callback
// time. Requests and Close remain serialized for the entire stream.
func (e *Engine) SynthesizeStream(ctx context.Context, request Request, options StreamOptions, yield func(ChunkAudio) error) (Timings, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return Timings{}, errors.New("kokoro engine is closed")
	}
	if err := ctx.Err(); err != nil {
		return Timings{}, err
	}
	if yield == nil {
		return Timings{}, errors.New("stream callback is required")
	}
	start := time.Now()
	chunks, err := prepareWithOptions(request.Phonemes, request.Voice, request.Speed, e.voices, e.vocab, options)
	preparation := time.Since(start).Seconds()
	if err != nil {
		return Timings{PreparationSeconds: preparation}, err
	}
	timings, err := streamChunks(ctx, chunks, request.Trim, e.run, yield)
	timings.PreparationSeconds = preparation
	return timings, err
}

// streamChunks is shared by collecting and streaming synthesis. The run
// function isolates native execution from delivery, trimming and cancellation.
func streamChunks(ctx context.Context, chunks []Chunk, trim bool, run func(Chunk) ([]float32, error), yield func(ChunkAudio) error) (Timings, error) {
	var timings Timings
	for i, chunk := range chunks {
		if err := ctx.Err(); err != nil {
			return timings, err
		}
		startTime := time.Now()
		raw, err := run(chunk)
		timings.InferenceSeconds += time.Since(startTime).Seconds()
		if err != nil {
			return timings, fmt.Errorf("chunk %d: %w", i, err)
		}
		if err := ctx.Err(); err != nil {
			return timings, err
		}
		startTime = time.Now()
		start, end := 0, len(raw)
		if trim {
			start, end = audio.TrimBounds(raw)
		}
		timings.PostprocessingSeconds += time.Since(startTime).Seconds()
		if end <= start {
			return timings, fmt.Errorf("chunk %d is empty after trimming", i)
		}
		if err := ctx.Err(); err != nil {
			return timings, err
		}
		if err := yield(ChunkAudio{chunk, raw, start, end}); err != nil {
			return timings, fmt.Errorf("chunk %d callback: %w", i, err)
		}
		if err := ctx.Err(); err != nil {
			return timings, err
		}
	}
	return timings, nil
}
