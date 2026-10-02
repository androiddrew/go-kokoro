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

## ONNX Runtime ownership

ONNX Runtime has one environment per process, shared by every model in it.

- **Set `ORTLibrary`** (simplest): the engine takes a lease from
  [`github.com/androiddrew/ortenv`](https://github.com/androiddrew/ortenv), which
  starts the environment on first use and shuts it down when the last lease closes.
  Every component using ortenv in the process must use the same library selector.
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
cannot be interrupted mid-call. Whole-request audio stays in memory.
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
fixture tests also run; the model tests skip without `KOKORO_TEST_ORT`. The
fixtures in `testdata/` hold the exact token, style and speed inputs expected
for phoneme strings produced by go-g2p. CI runs everything against the official
ONNX Runtime 1.22.0 release.

`cmd/inference-probe` accepts a JSON request array, writes WAV/raw audio and
records the exact model inputs. It is an optional diagnostic tool.

See [ASSETS.md](ASSETS.md), [THIRD_PARTY.md](THIRD_PARTY.md), LICENSE and NOTICE.
