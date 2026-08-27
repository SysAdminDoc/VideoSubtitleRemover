# Privacy, network access, and offline use

Video Subtitle Remover processes media on the local computer. Video frames,
images, masks, subtitle text, and generated outputs are not uploaded by the
cleanup pipeline.

The default RapidOCR, OpenCV, TBE, and local ONNX paths do not fetch model
files at runtime. Optional engines can contact model hosts after they are
enabled. The complete list is below.

## Optional model downloads

| Path | Trigger | Remote source | Local destination | Identity rule |
|---|---|---|---|---|
| PaddleOCR | PaddleOCR is selected and its assets are missing | URLs managed by the installed PaddleOCR package | PaddleOCR user cache | Package-managed. Pre-populate the cache for offline use. |
| EasyOCR | EasyOCR is selected and its assets are missing | URLs managed by the installed EasyOCR package | EasyOCR user model cache | Package-managed. Pre-populate the cache for offline use. |
| Surya | Surya is installed and `VSR_ALLOW_GPL=1` | Hugging Face repositories selected by Surya | Hugging Face user cache | Package-managed. Pre-populate the cache for offline use. |
| Florence-2 | `VSR_VLM_OCR=florence2` and an approved remote source is configured | `huggingface.co/microsoft/Florence-2-base` | Hugging Face user cache | `VSR_FLORENCE2_REVISION` must be a full commit because repository code executes. |
| Qwen2.5-VL | `VSR_VLM_OCR=qwen25vl` and an approved remote source is configured | `huggingface.co/Qwen/Qwen2.5-VL-2B-Instruct` | Hugging Face user cache | `VSR_QWEN25VL_REVISION` must be a pinned revision. |
| simple-lama-inpainting | `VSR_ENABLE_PYTORCH_LAMA=1` and no local weight is cached | URL managed by simple-lama-inpainting | Torch or simple-lama user cache | VSR checks the downloaded `big-lama.pt` against its known SHA-256. |
| Wan2.1-VACE-1.3B | `VSR_VACE_AUTO_FETCH=1` | `huggingface.co/Wan-AI/Wan2.1-VACE-1.3B` plus Hugging Face storage hosts | VSR app model cache under `models/vace` | VSR pins the repository, full commit, and every required file hash. |
| faster-whisper | Whisper fallback uses a model size instead of a local path | Hugging Face repository selected by faster-whisper | Hugging Face user cache | Package-managed. Set a local model path or pre-populate the cache for offline use. |
| MatAnyone 2 | `VSR_MATANYONE=1` and an approved remote source is configured | Configured MatAnyone 2 model repository | Hugging Face user cache | `VSR_MATANYONE_REVISION` must be a pinned revision. |
| CoTracker3 | `VSR_COTRACKER=1` and a remote repository is configured | `github.com/facebookresearch/co-tracker` through `torch.hub`; upstream code may fetch its weights | Torch Hub user cache | `VSR_COTRACKER_REF` must be a full commit. |

PaddleOCR-VL through llama.cpp, FFmpeg Whisper, VideoPainter, and FloED use
local model paths or reviewed local wrapper commands. VSR does not download
their model files.

Hugging Face downloads can use `huggingface.co` for API and redirect traffic,
then Hugging Face Xet or CDN hosts for large file content. Exact storage hosts
are controlled by Hugging Face and can change. A strict firewall should allow
them only during an intentional download window.

## VACE identity and verification

The approved VACE identity is:

- Repository: `Wan-AI/Wan2.1-VACE-1.3B`
- Commit: `574e6a744642ce3bee319afc31496b88bde8aac4`
- Required artifacts: the diffusion safetensors file, T5 checkpoint, VAE
  checkpoint, model config, and four UMT5 tokenizer files

`VSR_VACE_AUTO_FETCH=1` sends those exact file names and the full commit to
`huggingface_hub.snapshot_download`. VSR hashes all eight files after download
and before importing or constructing VACE. A missing file always fails. A
repository or commit change must still name a full commit and also requires
`VSR_ALLOW_UNVERIFIED_MODELS=1`. A hash mismatch requires the same flag.

The model provenance record contains the repository, commit, expected and
actual file hashes, cache path, verification result, and unsafe-override state.
Processed-output sidecars and batch evidence include that record. Support
bundles include the identity but replace the cache path with `<redacted>`.

The VACE checkpoint includes PyTorch `.pth` files. A matching hash proves that
the bytes are the reviewed bytes. It does not make arbitrary pickle data safe.
Do not point VSR at an untrusted checkpoint.

## Network-silent operation

For a run that must not contact the network:

1. Keep the default OCR and cleanup path, or populate every optional model
   cache while the machine is connected.
2. Prefer explicit local model paths for Florence-2, Qwen2.5-VL, Whisper,
   MatAnyone 2, and CoTracker3.
3. Set `VSR_VACE_CKPT_DIR` to a fully verified local VACE snapshot instead of
   enabling auto-fetch.
4. Leave the startup update check off. Do not configure crash reporting, or
   set `VSR_CRASH_REPORTS=0`.
5. Disconnect the machine or enforce the policy with the operating system or
   network firewall. VSR does not claim to be a firewall.

Once a VACE snapshot exists in the VSR app cache and all hashes still match,
VSR reuses it without calling `snapshot_download`.
