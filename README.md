# go-kokoro

`github.com/androiddrew/go-kokoro` (package `kokoro`) synthesizes phonemes with
Kokoro v1.0 through ONNX Runtime. It is independent of text frontends and CLIs:
to go from English text to phonemes, use
[`github.com/androiddrew/go-g2p`](https://github.com/androiddrew/go-g2p).

You need an installed ONNX Runtime and two downloads, the Kokoro model (325 MB)
and voices (28 MB); see [ASSETS.md](ASSETS.md) for links and checksums. The
vocabulary is built in.

```go
import (
    "context"
    "errors"
    "os"
    kokoro "github.com/androiddrew/go-kokoro"
)

func speak(ctx context.Context) (err error) {
    engine, err := kokoro.New(kokoro.Config{
        ORTLibrary: "libonnxruntime.so", // or an explicit installed-library path
        ModelPath:  "/assets/kokoro/kokoro-v1.0.onnx",
        VoicesPath: "/assets/kokoro/voices-v1.0.bin",
        Provider:   kokoro.CPU, // or kokoro.CUDA
        Threads:    1,
    })
    if err != nil { return err }
    defer func() { err = errors.Join(err, engine.Close()) }()
    result, err := engine.Synthesize(ctx, kokoro.Request{
        Phonemes: "həlˈO wˈɜɹld!", Voice: "af_heart", Speed: 1, Trim: true,
    })
    if err != nil { return err }
    f, err := os.OpenFile("hello.wav", os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0644)
    if err != nil { return err }
    return errors.Join(kokoro.WriteWAV(f, result.Samples), f.Close())
}
```

## API and guarantees

- `ListVoices(path)` lists sorted archive names without initializing ORT.
- `Prepare` creates padded tokens, selected style rows and speed values.
- `SynthesizePrepared` consumes validated token/style/speed tensors directly.
- `SynthesizeStream` is an alternative to `Synthesize`: it delivers completed
  chunks to a callback without accumulating a whole-request waveform.
- `ModelInfo` returns inspected tensor metadata. `Initialization` and
  `Result.Timings` expose loading and preparation/inference/postprocessing stages.
- Results are owned Go audio buffers, with raw per-chunk samples and trim bounds.
  Output is mono 24 kHz; `WriteWAV` writes PCM16 with saturation, not hidden gain.
- Unknown phonemes, invalid tensors/speed, missing/unsupported assets and voices
  fail. Speed is 0.5–2.0. Effective chunk limit is **509 phonemes** because the
  pinned voice table has 510 rows and selects row `len(unpadded tokens)`.
- Long input retains non-whitespace phones, prefers whitespace boundaries, then
  uses rune boundaries. Trimmed chunks concatenate without added pauses/crossfades.
  Forced splits can change prosody; finite output can still exceed full scale.

The Kokoro v1.0 vocabulary is embedded; `VocabPath` overrides it only for a
custom model.

## Streaming audio

Use `Synthesize` for a complete waveform, or opt into `SynthesizeStream` when an
application can consume audio incrementally. Each callback receives one completed
model chunk, after optional silence trimming and before the next native run.
`chunk.Samples()` is the trimmed mono float32 audio at `kokoro.SampleRate` (24 kHz);
it aliases the caller-owned `chunk.Raw` buffer and remains valid after the
callback and synthesis return.

```go
// playback is an application-owned, bounded queue drained concurrently by an
// audio consumer. Run synthesis on a producer goroutine if playback is on the
// calling goroutine. Reuse the initialized engine across requests.
playback := make(chan []float32, 2)
// ... start the playback consumer before calling SynthesizeStream ...
timings, err := engine.SynthesizeStream(ctx, kokoro.Request{
    Phonemes: phonemes, Voice: "af_heart", Speed: 1, Trim: true,
}, kokoro.StreamOptions{
    FirstChunkPhonemes: 80,
    MaxChunkPhonemes:   160,
}, func(chunk kokoro.ChunkAudio) error {
    select {
    case playback <- chunk.Samples():
        return nil
    case <-ctx.Done():
        return ctx.Err()
    }
})
close(playback) // this producer owns the channel; the consumer drains it
// Handle err even if some chunks have already played. timings excludes callback time.
```

The example limits are starting points to tune against playback underruns and
prosody, not measured optimal values. `StreamOptions{}` preserves the exact
chunk boundaries and preparation used by `Synthesize`. With custom limits:

- `MaxChunkPhonemes` is 1–509; zero selects 509.
- `FirstChunkPhonemes` optionally imposes a smaller first-chunk limit. Zero uses
  the effective maximum; it cannot exceed that maximum.
- Limits count unpadded Unicode code points, including spaces and punctuation.
  A split prefers the last `.,!?;` boundary within the limit, then whitespace,
  then a code-point boundary. Short remaining input is kept together. Boundary
  whitespace is trimmed; all non-whitespace phonemes are retained.
- Each chunk selects its style row using its actual unpadded token count. Smaller
  chunks add native runs and can change timing/prosody. Delivery still waits for
  a complete chunk, so a callback alone does not accelerate a short one-chunk request.

The callback is synchronous and applies backpressure. Queue audio for a separate
consumer to overlap playback with the next chunk's synthesis. The engine creates
no background goroutines and retains no delivered audio or aggregate waveform;
the consumer controls how much audio it keeps. All input chunks are prepared and
validated before the first native run.

**Errors and lifetime:** a callback error stops further inference and delivery;
it is wrapped with the zero-based chunk index and supports `errors.Is`/`errors.As`.
Context cancellation is checked around native runs and callbacks, including after
the final callback. Native inference cannot be interrupted mid-run. Previously
delivered audio remains valid on any error. A blocking callback must observe the
context itself, and the playback consumer should cancel it if playback fails.
Returned timings include work completed before failure but exclude callback time.

Requests and `Close` remain serialized for the entire stream, including callbacks.
**Do not call back into the same engine from a callback**; that can deadlock.
Close the engine after streaming returns. `WriteWAV` writes a complete WAV per
call; concatenating those files is not a continuous PCM stream. Playback/network
framing belongs to the application. See `ExampleEngine_SynthesizeStream` for a
complete producer/consumer example.

## ONNX Runtime ownership

ONNX Runtime has one environment per process, shared by every model in it.

- **Set `ORTLibrary`** (simplest): the engine initializes the environment through
  [`github.com/androiddrew/ortenv`](https://github.com/androiddrew/ortenv) on first
  use. The environment stays loaded until process exit, so closing the last engine
  never unloads the runtime. Every component using ortenv in the process must use
  the same library selector, and nothing may destroy the environment ortenv owns.
- **Leave `ORTLibrary` empty** when your application already manages ONNX Runtime
  itself (directly through `onnxruntime_go` or through ortenv). Initialize the
  environment before `New` and destroy it only after `Close`; otherwise `New`
  returns `ErrNotInitialized`:

```go
ort.SetSharedLibraryPath("libonnxruntime.so")
if err := ort.InitializeEnvironment(); err != nil { return err }
defer ort.DestroyEnvironment() // runs after engine.Close

engine, err := kokoro.New(kokoro.Config{ModelPath: modelPath, VoicesPath: voicesPath})
if err != nil { return err }
defer engine.Close()
```

## Providers and lifetime

Requires Go 1.25 or newer with CGO. Tested on Linux/amd64, Go 1.25.0 and 1.26.4,
and ORT 1.22.0/C API 22, with CUDA 12.8/cuDNN
9.8 for GPU tests. These describe the known tested configuration. Native runtime
installation is user-managed and must provide the resolved binding's C API;
there is no exact 1.22.0 version-string restriction. CUDA/cuDNN requirements follow
the selected native runtime/provider build. `Provider: kokoro.CUDA` uses an explicit
device ID, `cudnn_conv_algo_search=DEFAULT` and `use_tf32=0`.
CPU uses one intra/inter-op thread by default, sequential graph
execution and full optimization. Set `Threads` for intra-op parallelism.

Close every engine. Requests and Close on an engine are serialized; native Run
cannot be interrupted mid-call. `Synthesize` retains whole-request audio;
`SynthesizeStream` delivers owned chunks without collecting the waveform.
Metadata inspection currently creates a temporary session before the inference
session, so loading timings include **two graph loads**. `Verbose` exposes node
placement; CUDA graphs still contain some CPU operators.

## Development

```bash
go test ./...
go vet ./...
go build ./cmd/inference-probe
```

Set `KOKORO_TEST_ASSETS` to the directory holding the model and voices, and
`KOKORO_TEST_ORT` to an ONNX Runtime library, then run `go test -race ./...`.
Unit tests need neither. With only `KOKORO_TEST_ASSETS`, the voice and `Prepare`
fixture tests also run; the model tests skip without `KOKORO_TEST_ORT`. Tests
that load ONNX Runtime, or need it unloaded, each run in a fresh subprocess
because ortenv keeps the environment loaded until process exit. The
fixtures in `testdata/` hold the exact token, style and speed inputs expected
for phoneme strings produced by go-g2p. CI runs everything against the official
ONNX Runtime 1.22.0 release.

`cmd/inference-probe` accepts a JSON request array, writes WAV/raw audio and
records the exact model inputs. It is an optional diagnostic tool.

See [ASSETS.md](ASSETS.md), [THIRD_PARTY.md](THIRD_PARTY.md), LICENSE and NOTICE.
