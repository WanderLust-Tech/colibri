# Third-party notices

Portions of `backend_cuda_dsv4.cu`, including the DeepSeek-V4 GPU router
selection algorithm, are adapted from `ds4_cuda.cu` in the ds4 project:
https://github.com/antirez/ds4

MIT License

Copyright (c) 2026 The ds4.c authors
Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The pinned DeepSeek-V4 reference runner in `c/tools/dsv4_vllm_reference.py`
executes an unmodified vLLM checkout (commit
`ffd46bfab2128bb84146050e98b51a617c6575ab`) as a behavioural oracle for the
native port; no vLLM code is vendored.

## Swiftlet qpack and MLX affine Metal GEMV

`c/qpack.c` and `c/qpack.h` implement the Swiftlet qpack v1 container schema
documented by `Sources/SwiftletCore/Qpack.swift` at the commit linked below.
The reader is a new portable implementation for Colibri and does not copy
Swift source code.

The MLX affine Q4/Q8 kernels in `c/backend_metal.mm` are adapted from
`gemv_affine_fast` and `gemv_affine_fast8` in Swiftlet:
https://github.com/leonickson1/Swiftlet at commit
`b3a04676748c7597800c5bcc8b80a32508f9f43d`.

Swiftlet is licensed under Apache License 2.0. The kernels were modified for
Colibri's checked descriptor, batched dispatch, buffer ownership, and fallback
contract. This repository's `LICENSE` contains the applicable Apache 2.0 text.

## DeepGEMM sm120 headers (fetched, not vendored: `c/third_party/deepgemm/`)

The DeepSeek V4 CUDA tier's DeepGEMM flavour (`make cuda-dsv4-dg-dll`,
`Makefile.deepseek-v4 DEEPGEMM=1`) compiles against headers that
`c/tools/fetch_deepgemm.sh` checks out at a pinned commit into the gitignored
`c/third_party/deepgemm/`; nothing from them is committed to this repository.
What that checkout contains, and the licences that apply when you build with it:

- DeepGEMM (MIT, Copyright (c) 2025 DeepSeek), from the community sm120 port
  https://github.com/bvolpato/DeepGEMM branch `sm120-full` @
  39fb4447a062b418fd08ce17cd308adb28559417, with
  `c/patches-deepgemm-sm120-msvc.patch` applied at fetch time. License text:
  `c/third_party/deepgemm/LICENSE` after the fetch.
- NVIDIA CUTLASS / CuTe (BSD-3-Clause), the `third-party/cutlass` submodule of
  that commit @ f3fde58372d33e9a5650ba7b80fc48b3b49d40c8. License text:
  `c/third_party/deepgemm/third-party/cutlass/LICENSE.txt` after the fetch.

## Swiftlet qpack installers (`c/tools/qpack_*install*.py`)

Swiftlet (https://github.com/leonickson1/Swiftlet) is licensed under Apache
License 2.0. This repository's `LICENSE` contains the applicable Apache 2.0
text.

`c/tools/qpack_install_policy.py` adapts the source-bound resume and
manifest-last completion policy from Swiftlet's `StreamingInstaller.swift` at
commit `86246618ba2af30334227e09ff84a6a7182c2a40`. It is a new
transport-neutral Python implementation and does not copy Swift source code.

`c/tools/qpack_http_install.py` is a new Python HTTP frontend over that policy.
It interoperates with Hugging Face-hosted Swiftlet qpack repositories but does
not copy Swiftlet or huggingface_hub source code.

`c/tools/qpack_mirror_install.py` and the static-mirror support in
`c/tools/qpack_http_install.py` interoperate with the `hashes.json` schema
emitted by Swiftlet's `scripts/verify_container.py`. They are new Python
implementations with stricter path, transport, credential, resume, and digest
validation; no Swiftlet source code is copied.

## minimp3 (`c/third_party/minimp3.h`, used by `c/acestep_audio.h`)

minimp3 (https://github.com/lieff/minimp3), the single-header MPEG-1/2/2.5 Layer I/II/III decoder, is vendored unmodified at
commit `ea99364f61c14656440e8d77e9c233ccf3124633`. It is dedicated to the public domain under CC0 1.0
(http://creativecommons.org/publicdomain/zero/1.0/): "to the extent possible under law, the author(s) have dedicated all
copyright and related and neighboring rights to this software to the public domain worldwide. This software is distributed
without any warranty." It decodes the MP3 input of the ACE-Step engine; the WAV reader, the gapless trimming and the
resampler in `c/acestep_audio.h` are new code, the resampler a port of the algorithm documented in torchaudio's
`functional.resample` (BSD-2-Clause) with no torchaudio source copied.
