# Kokoro assets

The engine needs the pinned float32 Kokoro v1.0 model and voice archive:

| File | Size | Download |
|---|---|---|
| `kokoro-v1.0.onnx` | 325.5 MB | https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx |
| `voices-v1.0.bin` | 28.2 MB | https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin |

[`SHA256SUMS`](SHA256SUMS) pins each file's exact contents. Its paths assume a
`kokoro/` directory under an asset root:

```bash
cd /path/to/asset-root && sha256sum -c /path/to/go-kokoro/SHA256SUMS
```

The vocabulary is embedded in the library (`internal/assets/vocab.json`, from
hexgrad/Kokoro-82M `config.json`; see THIRD_PARTY.md). The model and voices are not
in Git.
The voice `.bin` is a NumPy ZIP archive; the Go loader decodes it directly.
Expected tensors: int64 tokens `[1,N]`, float32 style `[1,256]`, float32 speed `[1]`,
and one-dimensional float32 audio output. Unsupported schemas fail at load time.

Download the pinned files, verify them with `SHA256SUMS`, and pass their paths
in `kokoro.Config`. Supply an independently installed ONNX
Runtime through ortenv by path or loader name. The current binding requests
C API 22; native 1.22.0 is a tested baseline, not an exact-version restriction.
GPU use requires the dependencies appropriate to the selected native runtime.

Model card: https://huggingface.co/hexgrad/Kokoro-82M/blob/f3ff3571791e39611d31c381e3a41a3af07b4987/README.md
Preserve its Apache-2.0 declaration and full source/training-data acknowledgments
when redistributing model/voice files. The local bundle retains the pinned card.
