# Video Subtitle Remover Pro -- Roadmap

Actionable work only. Historical and completed roadmap material is archived in CHANGELOG.md; blocked work is kept in Roadmap_Blocked.md.

## Remaining work

Source research: `RESEARCH.md`, dated 2026-08-21. There is no P0.

Rejected on purpose (do not re-file): CLEAR/ROSE/EraserDiT/VOID adapters, Qt rewrite,
REST/Gradio, Mac/ROCm, re-adding GitHub Actions, Sigstore/attestations for SmartScreen,
winget/Store submission. Reasons in RESEARCH.md Rejected Ideas.

Implementer index -- drain in this order.

P3:
- RM-289 S -- PyInstaller floor 6.22.2 + onedir tripwire (advisory does not apply today)
- RM-293 S -- README Detection / Troubleshooting copy still describes the 3.8 picker
- RM-295 S -- unaudited surfaces from the 2026-08-21 pass (watch folder, Whisper, NLE, Docker)
- RM-296 M -- make the fade-in hold exact across decode-batch boundaries

### P3

- [ ] P3 -- RM-289: Raise the PyInstaller floor to 6.22.2 and guard the onedir assumption
  Why: the toolchain floor predates GHSA-9fxf-4qw3-ghmr (patched in 6.22.1; 6.22.2 is current). The advisory does not apply to this onedir/`asInvoker` build, so this is floor hygiene plus a tripwire.
  Evidence: current floor is 6.10.0 in `build_exe.bat`, `backend/release_verification.py`, and `README.md`. `VideoSubtitleRemoverPro.spec` builds onedir.
  Touches: `build_exe.bat`, `backend/release_verification.py`, `README.md`, `tests/test_release_workflow.py`.
  Acceptance: the floor reads 6.22.2 in all three places; verification names both GHSA-9w2p-rh8c-v9g5 and GHSA-9fxf-4qw3-ghmr with onedir/`asInvoker` non-applicability stated; a test fails if the spec ever stops being onedir while the floor sits below 6.22.1.
  Confidence: Verified
  Complexity: S

- [ ] P3 -- RM-293: Bring README Detection and Troubleshooting in line with the engine picker
  Why: the docs still tell users to install PaddleOCR for best accuracy and to activate VLM via env vars.
  Evidence: README pin list omits Surya/VLM; Troubleshooting still leads with PaddleOCR; `backend/detection.py` module doc still says VLM is env-only.
  Touches: `README.md` (byte-precise; mixed CRLF/LF -- do not whole-file rewrite), `backend/detection.py` module docstring. Do not hand-edit the generated CLI table.
  Acceptance: Detection pin list names every engine the picker offers; VLM section says the GUI/`--ocr-engine` path first; Troubleshooting recommends RapidOCR (default) and names PaddleOCR as opt-in; `tests/test_documentation_drift.py` still passes.
  Confidence: Verified
  Complexity: S

- [ ] P3 -- RM-295: Surfaces this audit did not execute
  Why: a pass that never opened these paths should not claim they were reviewed.
  Evidence: no live run of the watch-folder ingest, Whisper/WhisperX helpers, `--nle-input` NLE sidecar, VapourSynth `.vpy` ingest, Docker image build (Linux engine absent on this host), or a real NSIS silent uninstall. Dark theme only; there is no light theme to switch.
  Touches: `backend/watch_folder.py` (if present), Whisper helpers, `backend/nle_sidecar.py`, `backend/vapoursynth_bridge.py`, `Dockerfile`, `installer/vsr.nsi`.
  Acceptance: each named surface has a recorded smoke (command, expected result) or is moved to Roadmap_Blocked with the actual blocker.
  Confidence: Verified (gap)
  Complexity: S

- [ ] P3 -- RM-296: Make the fade-in mask hold exact across decode-batch boundaries
  Why: RM-292's fade-in reaches back only within the current decode batch, because the frames before it have already been inpainted and written by the time the batch's masks are known. A track whose first detection lands in the first few frames of a batch gets a shorter hold than requested.
  Evidence: `backend/processor.py` writes per batch (`_write_batch` inside the `while True` loop), and `state.frame_idx` advances during decode, so a carry-over would desync the checkpoint contract. Batch size is `sttn_max_load_num` (default 30) and the hold is capped at 15, so the miss is bounded but real. Fade-out is already exact via `_FrameLoopState.fade_carry`.
  Touches: `backend/processor.py` (`_decode_and_build_batch`, `_write_batch`, `_checkpoint_after_batch`), `tests/test_frame_loop.py`.
  Acceptance: a track whose first detection falls at index 0 of a decode batch still receives the full `--fade-in N` hold; the checkpoint resume contract still holds (a resumed run produces the same frames as an uninterrupted one); `N=0` stays byte-identical on the reference corpus.
  Confidence: Verified (gap)
  Complexity: M
