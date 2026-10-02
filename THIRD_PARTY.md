# Attribution

Original code is Apache-2.0 (LICENSE/NOTICE). Retained notices are under
`third_party/licenses/`.

- The embedded vocabulary (`internal/assets/vocab.json`) is the `vocab` table from
  **hexgrad/Kokoro-82M** `config.json`, Apache-2.0, at commit
  `f3ff3571791e39611d31c381e3a41a3af07b4987`
  (https://huggingface.co/hexgrad/Kokoro-82M/blob/f3ff3571791e39611d31c381e3a41a3af07b4987/config.json).
  It is the token table the Kokoro v1.0 model was trained with; kokoro-onnx ships
  an identical copy.
- Phoneme packing and voice-row policy adapt **kokoro-onnx 0.4.9**,
  thewh1teagle, MIT (`kokoro-onnx.txt`). Oversized-span handling is extended to
  retain all phones at the effective 509-phone maximum.
- Mono trimming adapts **librosa**'s trimming algorithm under its ISC notice
  (`librosa-trim.txt`). NPZ/NPY reading and WAV encoding are implemented in Go.
- **yalue/onnxruntime_go v1.22.0** is an external MIT-licensed Go dependency.
  The native ONNX Runtime is installed by the user and is not redistributed here.
- **github.com/androiddrew/ortenv v0.1.0** is MIT licensed, Copyright (c) 2026 Drew Bednar.
- Kokoro v1.0 model/voices declare Apache-2.0 in the pinned model card. Preserve
  the card's StyleTTS 2, ISTFTNet and training-data acknowledgments, including
  Koniwa `tnc` (CC BY 3.0) and SIWIS (CC BY 4.0), with redistributed assets.
- Optional NVIDIA CUDA/cuDNN installations are also user-managed, not repo assets.

No pronunciation backend is included in this module. Model and voice binaries
are external assets rather than files relicensed by the Go code license.
