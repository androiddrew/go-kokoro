package kokoro

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"sync"
	"time"

	"github.com/androiddrew/go-kokoro/internal/assets"
	"github.com/androiddrew/go-kokoro/internal/audio"
	"github.com/androiddrew/go-kokoro/internal/ortruntime"
	"github.com/androiddrew/ortenv"
	ort "github.com/yalue/onnxruntime_go"
)

// SampleRate is the output rate in Hz. Output is mono.
const SampleRate = audio.SampleRate

// ErrNotInitialized is returned by New when ORTLibrary is empty and the caller
// has not initialized the ONNX Runtime environment.
var ErrNotInitialized = ortruntime.ErrNotInitialized

// Provider selects the ONNX Runtime execution provider.
type Provider string

// Supported providers.
const (
	CPU  Provider = "cpu"
	CUDA Provider = "cuda" // appended explicitly; there is no CPU retry on failure
)

// Config locates the model assets and configures the session.
type Config struct {
	// ORTLibrary is an ONNX Runtime library path or OS-loader name. When set, the
	// engine shares the environment through an ortenv lease. When empty, the
	// caller owns the environment and must initialize it before New and destroy
	// it only after Close.
	ORTLibrary string
	ModelPath  string // kokoro-v1.0.onnx
	VoicesPath string // voices-v1.0.bin
	// VocabPath overrides the embedded Kokoro v1.0 vocabulary. Usually empty.
	VocabPath string
	Provider  Provider // empty selects CPU
	DeviceID  int      // CUDA device
	Threads   int      // intra-op threads; zero selects 1
	Verbose   bool     // write ORT node-placement diagnostics to stderr
}

// Engine synthesizes phonemes. Requests and Close are serialized.
type Engine struct {
	mu             sync.Mutex
	session        *ort.DynamicAdvancedSession
	voices         *assets.Voices
	vocab          map[rune]int64
	info           ModelInfo
	closed         bool
	lease          *ortenv.Lease
	initialization Initialization
}

// Initialization is a diagnostic breakdown of New's loading time. ModelSeconds
// includes both ONNX loads: metadata inspection and the inference session.
type Initialization struct {
	RuntimeSeconds float64 `json:"runtime_initialization_seconds"`
	VoiceSeconds   float64 `json:"voice_loading_seconds"`
	VocabSeconds   float64 `json:"vocabulary_loading_seconds"`
	ModelSeconds   float64 `json:"model_load_seconds"`
}

// Initialization returns how long New spent in each loading stage.
func (e *Engine) Initialization() Initialization { return e.initialization }

// Timings is a diagnostic breakdown of one request into disjoint stages. Inference includes tensor creation/destruction,
// synchronous Run, device-to-host retrieval, an owned copy and finite validation.
type Timings struct {
	PreparationSeconds    float64 `json:"preparation_seconds"`
	InferenceSeconds      float64 `json:"inference_seconds"`
	PostprocessingSeconds float64 `json:"postprocessing_seconds"`
}

// New validates assets and tensor metadata before accepting requests. Close is
// mandatory. Concurrent calls on one engine are serialized, including Close.
func New(c Config) (_ *Engine, err error) {
	if c.Provider != "" && c.Provider != CPU && c.Provider != CUDA {
		return nil, fmt.Errorf("unsupported provider %q; choose cpu or cuda", c.Provider)
	}
	if c.Threads < 0 || c.DeviceID < 0 {
		return nil, errors.New("threads and device ID must be nonnegative")
	}
	if c.ModelPath == "" {
		return nil, errors.New("ModelPath is required")
	}
	e := &Engine{}
	start := time.Now()
	e.voices, err = assets.LoadVoices(c.VoicesPath)
	e.initialization.VoiceSeconds = time.Since(start).Seconds()
	if err != nil {
		return nil, err
	}
	start = time.Now()
	e.vocab, err = assets.LoadVocab(c.VocabPath)
	e.initialization.VocabSeconds = time.Since(start).Seconds()
	if err != nil {
		return nil, err
	}
	start = time.Now()
	if e.lease, err = ortruntime.Acquire(c.ORTLibrary); err != nil {
		return nil, err
	}
	e.initialization.RuntimeSeconds = time.Since(start).Seconds()
	defer func() {
		if err != nil {
			err = errors.Join(err, e.lease.Close())
		}
	}()
	start = time.Now()
	defer func() { e.initialization.ModelSeconds = time.Since(start).Seconds() }()
	options, err := sessionOptions(c)
	if err != nil {
		return nil, err
	}
	// Defer ordering matters: destroy options/session before releasing the lease.
	defer func() {
		err = errors.Join(err, options.Destroy())
		if err != nil && e.session != nil {
			err = errors.Join(err, e.session.Destroy())
		}
	}()
	e.info, err = inspectModel(c.ModelPath, options)
	if err != nil {
		return nil, err
	}
	e.session, err = ort.NewDynamicAdvancedSession(c.ModelPath, []string{"tokens", "style", "speed"}, []string{"audio"}, options)
	if err != nil {
		return nil, fmt.Errorf("load Kokoro %s session: %w", c.Provider, err)
	}
	return e, nil
}

// Close destroys the session and releases any ortenv lease. It is idempotent.
func (e *Engine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil
	}
	e.closed = true
	err := e.session.Destroy()
	e.session, e.voices, e.vocab = nil, nil, nil
	return errors.Join(err, e.lease.Close())
}

// Voices returns the sorted voice names, or nil after Close.
func (e *Engine) Voices() []string {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil
	}
	return e.voices.Names()
}

// ModelInfo returns an owned copy; -1 denotes a dynamic dimension.
func (e *Engine) ModelInfo() ModelInfo {
	copyGroup := func(group []TensorInfo) []TensorInfo {
		out := append([]TensorInfo(nil), group...)
		for i := range out {
			out[i].Shape = append([]int64(nil), out[i].Shape...)
		}
		return out
	}
	return ModelInfo{copyGroup(e.info.Inputs), copyGroup(e.info.Outputs)}
}

// Prepare splits phonemes into model-sized chunks and returns the exact token,
// style and speed values Synthesize would feed the model.
func (e *Engine) Prepare(phonemes, voice string, speed float32) ([]Chunk, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil, errors.New("kokoro engine is closed")
	}
	return prepare(phonemes, voice, speed, e.voices, e.vocab)
}

// Request is one synthesis request.
type Request struct {
	Phonemes string  `json:"phonemes"` // Kokoro phonemes, e.g. from go-g2p
	Voice    string  `json:"voice"`    // a name from Voices, e.g. "af_heart"
	Speed    float32 `json:"speed"`    // explicit 0.5..2.0; zero is invalid
	Trim     bool    `json:"trim"`     // trim leading/trailing silence per chunk
}

// ChunkAudio is the diagnostic per-chunk output behind Result.Samples.
type ChunkAudio struct {
	Chunk
	Raw       []float32 `json:"-"` // untrimmed model output, owned by the caller
	TrimStart int       `json:"trim_start"`
	TrimEnd   int       `json:"trim_end"`
}

// Result is the synthesized audio.
type Result struct {
	Samples    []float32    `json:"-"` // mono samples at SampleRate
	Chunks     []ChunkAudio `json:"chunks"`
	SampleRate int          `json:"sample_rate"`
	Timings    Timings      `json:"timings"`
}

// Synthesize performs synchronous inference and output retrieval, trims each
// chunk, and concatenates the chunks with no added pauses, as kokoro-onnx does.
// Cancellation is checked between native runs; ORT v1.22's Go wrapper cannot
// interrupt an in-flight Run. An error returns no partially assembled result.
func (e *Engine) Synthesize(ctx context.Context, request Request) (Result, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return Result{}, errors.New("kokoro engine is closed")
	}
	if err := ctx.Err(); err != nil {
		return Result{}, err
	}
	start := time.Now()
	chunks, err := prepare(request.Phonemes, request.Voice, request.Speed, e.voices, e.vocab)
	if err != nil {
		return Result{}, err
	}
	preparation := time.Since(start).Seconds()
	result, err := e.synthesizeChunks(ctx, chunks, request.Trim)
	result.Timings.PreparationSeconds = preparation
	return result, err
}

// SynthesizePrepared uses the supplied tensors verbatim, independent of the
// frontend, vocabulary encoding and voice row selection. Callers must not mutate
// chunks during the call. Invalid shapes, token IDs or non-finite values fail
// before any native run. Use it to benchmark or test inference on exact inputs.
func (e *Engine) SynthesizePrepared(ctx context.Context, chunks []Chunk, trim bool) (Result, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return Result{}, errors.New("kokoro engine is closed")
	}
	start := time.Now()
	if len(chunks) == 0 {
		return Result{}, errors.New("prepared input has no chunks")
	}
	valid := map[int64]bool{}
	for _, id := range e.vocab {
		valid[id] = true
	}
	for i, c := range chunks {
		if len(c.Tokens) < 3 || len(c.Tokens) > MaxPhonemes+2 || c.Tokens[0] != 0 || c.Tokens[len(c.Tokens)-1] != 0 || len(c.Style) != assets.StyleWidth || !(c.Speed >= .5 && c.Speed <= 2) {
			return Result{}, fmt.Errorf("invalid prepared chunk %d shape, boundaries or speed", i)
		}
		for _, id := range c.Tokens[1 : len(c.Tokens)-1] {
			if !valid[id] || id == 0 {
				return Result{}, fmt.Errorf("invalid prepared token %d", id)
			}
		}
		for _, v := range c.Style {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return Result{}, errors.New("non-finite prepared style")
			}
		}
	}
	preparation := time.Since(start).Seconds()
	result, err := e.synthesizeChunks(ctx, chunks, trim)
	result.Timings.PreparationSeconds = preparation
	return result, err
}

func (e *Engine) synthesizeChunks(ctx context.Context, chunks []Chunk, trim bool) (Result, error) {
	result := Result{SampleRate: SampleRate}
	for i, chunk := range chunks {
		if err := ctx.Err(); err != nil {
			return Result{}, err
		}
		startTime := time.Now()
		raw, err := e.run(chunk)
		result.Timings.InferenceSeconds += time.Since(startTime).Seconds()
		if err != nil {
			return Result{}, fmt.Errorf("chunk %d: %w", i, err)
		}
		if err := ctx.Err(); err != nil {
			return Result{}, err
		}
		startTime = time.Now()
		start, end := 0, len(raw)
		if trim {
			start, end = audio.TrimBounds(raw)
		}
		if end <= start {
			return Result{}, fmt.Errorf("chunk %d is empty after trimming", i)
		}
		result.Samples = append(result.Samples, raw[start:end]...)
		result.Chunks = append(result.Chunks, ChunkAudio{chunk, raw, start, end})
		result.Timings.PostprocessingSeconds += time.Since(startTime).Seconds()
	}
	return result, nil
}

func (e *Engine) run(chunk Chunk) (_ []float32, err error) {
	var values []ort.Value
	defer func() {
		for _, v := range values {
			err = errors.Join(err, v.Destroy())
		}
	}()
	tokens, err := ort.NewTensor(ort.NewShape(1, int64(len(chunk.Tokens))), chunk.Tokens)
	if err != nil {
		return nil, err
	}
	values = append(values, tokens)
	style, err := ort.NewTensor(ort.NewShape(1, assets.StyleWidth), chunk.Style)
	if err != nil {
		return nil, err
	}
	values = append(values, style)
	speed, err := ort.NewTensor(ort.NewShape(1), []float32{chunk.Speed})
	if err != nil {
		return nil, err
	}
	values = append(values, speed)
	outputs := []ort.Value{nil}
	defer func() {
		for _, v := range outputs {
			if v != nil {
				err = errors.Join(err, v.Destroy())
			}
		}
	}()
	if err = e.session.Run(values, outputs); err != nil {
		return nil, err
	}
	output, ok := outputs[0].(*ort.Tensor[float32])
	if !ok || len(output.GetShape()) != 1 {
		return nil, errors.New("expected one-dimensional float32 audio output")
	}
	samples := append([]float32(nil), output.GetData()...)
	if err = audio.Validate(samples); err != nil {
		return nil, err
	}
	return samples, nil
}

// WriteWAV writes samples as a mono 16-bit PCM WAV at SampleRate. Out-of-range
// samples saturate.
func WriteWAV(w io.Writer, samples []float32) error { return audio.WriteWAV(w, samples) }
