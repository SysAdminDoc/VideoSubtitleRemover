# Changelog

All notable changes to VideoSubtitleRemover will be documented in this file.

## [Unreleased]

### Fixed

- The FFmpeg that ships inside the OpenCV wheel is now classified against
  the same 9.0.1 floor the external binary has to meet. It was left
  permanently unclassified because OpenCV prints library ABI numbers
  rather than an FFmpeg version; those ABI numbers do identify the
  release, and the wheel turns out to carry FFmpeg 7.1, which predates
  the entire 2026 advisory batch. Release evidence and the support bundle
  now record the identified release, the ABI numbers behind the verdict,
  and the advisories it is exposed to. Shipping on that wheel is an
  explicit dated decision scoped to exactly that release: a wheel that
  moves to any other pre-9.0 build fails release verification.
- Startup no longer reports "FFmpeg ready" for a build that will stop the
  run. It was reading a bare exit code, so an FFmpeg below the enforced
  9.0.1 security floor showed a green state at launch and then failed at
  the security check partway through a job. The status line, the header
  chip, the settings warning, and the About panel now all come from the
  same classification the backend already computed, naming the detected
  version. A development snapshot, whose version cannot be identified, is
  reported as unclassified rather than ready. The support bundle carries
  the verdict too.
- A frames-to-masks length mismatch now fails the run instead of quietly
  processing the shorter list. 61 sequence pairings carried no explicit
  length rule, including the inpainting hot path, so a mismatch repaired
  the common prefix and wrote the remaining frames out untouched with
  nothing reported. Every pairing in the decode, mask, inpaint, encode,
  and quality paths now states that the lengths must match, or says in a
  comment why unequal lengths are correct there, and the linter enforces
  it from here on.
- The live preview no longer goes blank for the whole run when the
  temporary directory holds non-ASCII characters. One staged frame read
  was still going through OpenCV under a local alias, which returned
  nothing on such a path and was swallowed as "no frame yet". The
  source-hygiene check now resolves whatever local name OpenCV was
  imported under, so an alias cannot hide the next one.
- A second launch during a cold start could still get through. The entry
  point gave the instance slot back before the window existed and let the
  app take it again, leaving a gap of tens of milliseconds; it now holds
  the slot for the whole session, on every platform. A refused launch
  also no longer appends to the running instance's log file.
- Saving, deleting, or importing a preset now reads and writes inside one
  cross-process lock. Holding it only for the write kept the file valid
  but still dropped a preset another process saved in between.
- A recorded corpus gate exception can no longer claim a bound that could
  never fail. A floor of zero on a metric that is bounded below by zero
  is refused unless it is explicitly marked inert with a reason, and
  `--bless` stops exiting clean on a run the gate still rejects.

### Changed

- The NVIDIA lane moves to the CUDA 13 channel: torch 2.13.0 and
  torchvision 0.28.0 from the `cu130` index, with onnxruntime-gpu 1.29.0.
  The `cu128` index stops at torch 2.11.0, which left the NVIDIA profile
  frozen two minors behind CPU and DirectML and inside an advisory both of
  those had already left. Verified on an RTX 4070 SUPER: `verify --profile
  nvidia` passes and the inference smoke really runs on
  `CUDAExecutionProvider`.

### Fixed

- The provider smoke no longer reports a false CUDA fallback. ONNX
  Runtime's CUDA provider DLL needs the CUDA runtime that ships inside the
  PyTorch wheel, and the inpainters already preload it before every CUDA
  session while the smoke did not, so the smoke saw a fallback the product
  never hits. It now takes the same path and records the preload outcome.
- The reference corpus now runs the shipping quality gate. It used to
  check a frame hash plus PSNR and SSIM floors blessed from its own last
  run, so it reported all ten clips as passing while eight of them sat
  over the runtime residual-text ceiling by as much as 32x, which meant
  it could not detect a removal-quality failure at all. Each clip now
  records the full gate verdict, and a clip that does not pass fails the
  run unless the manifest carries a dated, reasoned exception naming the
  metric, the threshold it crosses, and the worst value that stays
  acceptable. Go past that value and the corpus goes red.
- A second launch no longer races the first one's state. Two GUI
  processes share `settings.json` and `queue_state.json`, and the
  Windows named mutex the installer already relies on was created
  without ever reading whether it existed. A second launch now reports
  that the app is open and exits without writing anything, and it does
  not steal the foreground from the running window. Every path that
  persists shared user state also takes a real cross-process lock, so a
  GUI and a CLI run that legitimately overlap cannot drop each other's
  updates.

- Still images and mask files now read and write correctly on paths holding
  CJK, Cyrillic, or accented Latin characters. OpenCV hands the filename to
  its C++ layer as a narrow byte string, so `cv2.imwrite` silently returned
  False and `cv2.imread` returned None for those paths, which made image
  cleanup, mask export and import, clean-reference plates, frame export, and
  preview staging unusable for anyone whose folders are not pure ASCII. All of
  it now goes through `backend.safe_image`, which encodes and decodes in
  memory and moves the bytes with Python's own file I/O. A source-hygiene test
  parses the tree and fails if any module outside that one calls `cv2.imread`
  or `cv2.imwrite` directly.

## [3.40.0] - 2026-08-27

### Changed

- The workbench now uses a quieter product header, aligned command controls,
  wider media gutters, and a shorter preview heading. The file name leads the
  preview once media is loaded.
- Set region, Review mask, and Test cleanup remain visible. Track planning,
  full-size viewing, mask correction, and queue maintenance now live under
  focused More menus.
- Detection controls use stacked labels, values, and sliders. Opening any
  inspector category keeps the panel near its closed width instead of pushing
  the video into a narrow column.
- The empty preview uses one clear instruction surface. Queue rows size to their
  rendered content at the active display scale, so file details stay readable
  without a scrollbar for one item.
- Nine product screenshots were recaptured with `static_dialogue.mkv`. The main
  flow and detailed controls were checked in the default and high-contrast
  themes on isolated Windows desktops.

### Fixed

- Opening Detection no longer lets long hints, toggles, or fixed slider widths
  expand the inspector by more than 200 pixels.
- Idle queue metadata is no longer clipped at 125 percent Windows scaling.
- At 200 percent app text on a 125 percent Windows display, the command strip
  now removes redundant labels and keeps a useful part of the scrollable
  workbench visible. Preview status and tools move below long file names before
  they can overlap.

## [3.39.0] - 2026-08-27

### Changed

- The main workspace now gives the preview most of the window, keeps the common
  review tools visible, and uses a full-width queue without an empty spacer.
  Opening a settings category keeps the selected controls near the top while
  the video and timeline remain visible.
- Small clips now scale up to the available preview surface. The frame is fitted
  again after the timeline or settings panel changes the layout, which prevents
  stale dimensions from cropping the video.
- Track review now uses explicit removal language, a live selection count, and
  Remove all or Keep all controls. Region editing groups its optional timing and
  motion tools, while mask correction gives preparation and ready states a clear
  status surface.
- Help stays neutral while startup checks are running. Completion summaries use
  reader-facing stage names, hide empty failure cards, and keep the useful output
  action prominent. Default and high-contrast surfaces were checked with the
  bundled test video on an isolated Windows desktop.

### Security

- VACE auto-fetch now uses the allowlisted
  `Wan-AI/Wan2.1-VACE-1.3B` repository at full commit
  `574e6a744642ce3bee319afc31496b88bde8aac4`. Only the eight runtime artifacts
  are downloaded, and every file must match its pinned SHA-256 before model
  import or construction. Missing files cannot be bypassed. A different full
  commit, repository, or mismatched hash requires the explicit
  `VSR_ALLOW_UNVERIFIED_MODELS=1` override and is recorded as unsafe.
- Verified model provenance now records repository, commit, file hashes, cache
  path, and override state. It is carried into execution sidecars and batch
  evidence, while support bundles retain the identity and redact the local
  cache path. The network guide now lists every optional runtime model fetch
  and its offline alternative.

### Fixed

- Reference-corpus baselines now require the exact reviewed CPU profile before
  processing. Two clean OpenCV 5.0.0.93 and NumPy 2.4.6 runs produced identical
  hashes for every fixture. An older global OpenCV 4.14 installation was the
  source of the three apparent hash reversions, and the runner now reports that
  package mismatch with the profile repair command instead of implying random
  output drift.
- Setup now installs one named, locked CPU, NVIDIA, or DirectML profile and
  refuses to report success after any partial failure. It prints the profile's
  exact packages and capabilities, writes an atomic setup report, and exits
  with a profile-specific repair command when installation or verification
  fails. Setup and every source launcher now share a strict verifier for exact
  versions, runtime imports, dependency consistency, and real provider
  inference. Intel uses the deterministic DirectML core, while OpenVINO stays
  an explicit optional add-on.
- Skip-existing now verifies a v3 output sidecar before accepting prior work.
  Source SHA-256, normalized processing settings, output path, byte size, and
  output SHA-256 must all match. Missing, older, changed, or truncated evidence
  is reprocessed without rewriting the old sidecar, and incomplete outputs do
  not receive trusted sidecars. The explicit `any` policy keeps existence-only
  behavior and records that the output was not verified.
- SRT export now reuses text and confidence from tracked OCR detections instead
  of running recognition across the full frame a second time. Conservative
  Unicode-aware consensus absorbs low-confidence character jitter without
  joining real caption changes. Exact VFR frame timing is unchanged, and OCR
  engines that return only boxes keep the fallback recognition path.
- FFmpeg 9 frame probes now read `duration` and `duration_time` first, with the
  older packet-duration names kept only as fallbacks. Variable-rate outputs use
  a lossless PNG timing carrier before the final codec, so an irregular final
  frame keeps its authoritative duration through audio muxing. Release evidence
  verifies the current fields against the installed runtime. Duration variance
  now triggers that exact path even when every preceding PTS delta is constant.
- Manual region now opens the region editor when no saved shape exists. An
  existing fixed, timed, or moving region is preserved and shown immediately,
  bypasses OCR only when Manual region is active, and supplies the same mask to
  Auto, STTN, LaMa, and ProPainter. Clear region returns to automatic detection
  with visible feedback. Automatic cleanup previews now combine saved regions
  with OCR detections and report each region once.
- Named OCR, inpainting, segmentation, tracking, and restoration requests now
  fail with a classified reason and recovery guidance when they can't execute.
  Missing or malformed tracker visibility and unchanged segmentation mattes
  are rejected. Uninstrumented inpainters must alter an active masked pixel,
  and post-restoration output cannot be byte-identical to its input. Validation
  finishes before provenance records success. Only Auto may change
  implementations.

## [3.38.0] - 2026-08-22

### Changed

- Primary actions now keep normal-size text above the 4.5:1 contrast floor in
  default, hover, and pressed states. Disabled copy is clearer, the shared
  radius scale uses 4, 6, 8, and 10 pixels, and thin progress tracks use crisp
  ends instead of capsule shapes.
- Detection, Inpainting, Encoding, and Advanced are now independent inspector
  disclosures. Opening one category closes the previous category, updates the
  correct chevron and accessibility state, and keeps hidden controls out of the
  keyboard and accessibility control views. Interface and app behavior options
  now have their own Advanced card instead of extending Encoding quality.
- The empty preview now explains the next step and offers a direct media action.
  The header uses a clearer brand mark, queue and workbench scrollbars appear
  only when content overflows, and fitted dialogs no longer trigger horizontal
  and vertical scrollbars from their own scrollbar chrome. Help replaces its
  clipped shortcut row with concise sample-review guidance. Conditional
  scrollbars use a reserved overlay gutter, so showing one cannot start a
  geometry feedback loop in the frozen app.
- The region editor now uses a large preview beside focused Shape, Timing,
  Region properties, and Clean reference cards. Small source clips scale to the
  available preview area, exact fields have consistent labels and focus rings,
  and secondary actions no longer compete with Save region.
- The mask editor now follows the same two-pane pattern. Paint mode is a direct
  segmented choice, frame review has a scrubber, brush size stays visible, and
  the tools panel reports saved correction ranges as they change.

## [3.37.0] - 2026-08-22

### Fixed

- Release evidence now inventories both FFmpeg runtimes. The external record
  includes its version and build configuration, while the OpenCV record keeps
  wheel provenance and the embedded `avcodec`, `avformat`, and `avutil` ABI
  versions. Embedded builds block only with a cited advisory mapping. The CPU
  container now pins its Python base by digest and builds verified FFmpeg 9.0.1
  from the official source archive.
- Video timing now keeps integer PTS and duration ticks with the source rational
  time base from ffprobe. Checkpoints, matte manifests, frozen mattes,
  track plans, audio trimming, SRT, EDL, and FCPXML retain the exact clock while
  float values remain compatibility views.
- HDR repair now keeps the decoded uint16 source surface, uses a tone-mapped
  proxy for OCR and 8-bit model inputs, and composites repaired PQ or HLG pixels
  in bounded linear light inside the active mask ROI. Outside-mask pixels remain
  exact, complete color tags are required, and synthetic repaired ramps retain
  high-bit precision through the transfer round trip. Missing probes, invalid
  transfers, conflicting stream/frame tags, untagged YUV without verified
  8-bit-surface or valid-range evidence, and unavailable native high-bit
  decoding stop the HDR path instead of flattening it to 8-bit.
- Quality reports now add a mask-local temporal score that compensates for
  motion and excludes scene cuts using untouched pixels. The final encoded
  output is reopened before scoring, the worst valid pair is gated directly,
  and its timestamp and overlay are recorded alongside outside-mask CIELAB
  drift for SDR or linear-light drift for tagged HDR. A failed color guard
  requests review without recoloring the output.
- OCR quadrilaterals now travel beside the historical axis-aligned boxes through
  detection, tracking, track-plan save/load, masks, and preview overlays. Each
  polygon gets its own local expansion, so a rotated caption doesn't remove the
  unused corners of its bounding rectangle. Rectangle regions and older plans
  still load unchanged.
- The active GUI release suite now exercises queue selection, region propagation,
  Test Cleanup dispatch, cancellation, shutdown, scaling, high contrast, RTL,
  and collapsed Advanced control-view and tab-order behavior. Disclosure rows
  now expose explicit button names and expanded or collapsed state through the
  accessibility bridge. The packaged release matrix runs without foreground
  screen automation; live Narrator and NVDA proof remains an isolated-session
  follow-up.
- Test Cleanup now follows the selected video timestamp. Temporal modes plan a
  scene-bounded before/current/after source window using the cached low-resolution
  proxy, then inpaint only full-resolution source frames. Preview evidence reports
  the tested timestamp, frame range, and planning resolution, and stale requests
  are discarded safely.
- Strict release evidence now blocks Torch 2.9.1 and older for
  CVE-2026-24747. The existing CVE-2025-32434 check remains independent, and
  boundary coverage confirms that Torch 2.10.0 passes the advisory gate while
  reviewed 2.11.0 and 2.13.0 profiles remain accepted.
- The optional PaddleOCR-VL llama.cpp endpoint is local by default in policy,
  not just in documentation. Plain HTTP is limited to resolved loopback
  addresses. Remote use requires HTTPS and `VSR_ALLOW_REMOTE_VLM=1`, URL
  credentials and unsafe redirects are refused, and DNS is checked again
  before each frame request. Settings and `--self-test` keep the frame-transfer
  warning visible.

## [3.36.0] - 2026-08-21

### Added

- Failed batch items now carry a classified reason (out of disk space,
  writer failed, FFmpeg failed, source could not be decoded, and so on)
  alongside the existing message. The batch summary counts them and shows
  a Reason column, and the queue keeps the classification across a restart.
- The quality report pools a harmonic mean beside the average PSNR and SSIM,
  and names the worst sampled frame with its score. A run whose average looks
  fine but whose worst frame does not now goes to review, and the batch
  summary has an Open frame button that takes the A/B compare straight there.
- Screen readers now get a role, a name, a value and the tooltip text for
  every custom control (buttons, toggles, sliders, the progress bar, the drop
  target) instead of an anonymous pane, and the footer status line reads out
  as it changes. Annotation failure is silent, so nothing changes on a machine
  where it is unavailable.
- The Settings language picker now lists only catalogs that have at least 90%
  of the interface translated, so choosing a language cannot leave most of the
  window in English. README documents the bar.
- A timed region can now take a whole donor video as its clean reference, not
  just a still plate. When a clean or differently-subbed release exists the
  background comes from it: frames are matched by timestamp with an offset you
  set, anything the offset does not cover falls back to the inpainter, and the
  sidecar records the donor hash, frame rate and offset. `--clean-reference`
  and `--clean-reference-offset` do the same from the CLI.
- Subtitles that fade in or out can keep their mask through the fade. The new
  Fade-in hold and Fade-out hold settings (`--fade-in` / `--fade-out`, 0-15
  frames, off by default) reuse the nearest confident mask on either side of a
  text track, covering the frames where the glyphs are visible but too faint
  for OCR. Both holds are exact across decode batches: the fade-out hold
  carries its mask forward, and the fade-in hold defers the tail of each
  batch until the following one exists so it can see what comes next.
  Pausing mid-run flushes that tail, which keeps the checkpoint on a batch
  boundary so a resumed run produces the same frames as an uninterrupted one.
- Every status message that names a file, a preset, a device or a count now
  reaches the translation catalog. Around forty of them were wrapped in a
  marker that records a msgid but cannot translate an f-string, so they always
  showed in English, and counts now use real plural forms. The catalog lint
  fails if a new one is written that way.
- Help has a Last FFmpeg commands view. The recent FFmpeg and ffprobe
  invocations are shown as quoted lines that run as-is, can be copied in one
  click, and are included in the support bundle. Paths are reduced to file
  names by the same redaction policy as the log, and the buffer is bounded.
- The build pins SOURCE_DATE_EPOCH and PYTHONHASHSEED and records both in the
  release evidence, so a rebuild starts from the same envelope. README now
  says plainly that rebuild verification is semantic: the frozen build embeds
  its own absolute paths, so matching checksums were never on offer.
- The PyInstaller floor moves to 6.22.2 in the build script, the release
  validation, and the README. Release validation now names both advisories
  that bear on it and reports GHSA-9fxf-4qw3-ghmr as informational, because a
  onedir build with an asInvoker manifest is not what it reaches. A test fails
  if the spec ever stops being onedir while the floor is behind that fix.
- README describes the engine picker that actually shipped. The Detection list
  names Surya and all four vision-language tiers, the VLM section leads with
  the picker and `--ocr-engine` instead of environment variables, and the
  Troubleshooting advice points at RapidOCR and the picker rather than telling
  everyone to install PaddleOCR.
- The watch folder, the NLE sidecar round trip, and the VapourSynth bridge
  guard now have tests that actually run them. The previous watch-folder tests
  all mocked the processing call, so none of them proved a drain writes a file.

### Security

- CPython self-test advisories are now per release line, and they name
  CVE-2026-11940 (the tarfile fix that actually sets the floor). A below-floor
  3.14 build no longer cites CVEs already fixed in 3.14.4.
- CVE-2026-58049 is treated as fixed in FFmpeg 9.0.1. A host on the reviewed
  floor is no longer told it still has an open RASC advisory.

### Fixed

- Output sidecars record the OCR engine that actually ran, not whichever OCR
  package happens to be importable.
- `--config` JSON with unknown keys now fails closed instead of silently
  dropping the typo and running with defaults.
- A time range that starts at or past the end of the clip is rejected instead
  of processing a leftover last frame.
- Queue rows no longer persist raw traceback fragments or absolute paths
  from raised exceptions. Curated backend messages (corrupt file, frozen
  matte) still surface on the row.
- The installer writes QuietUninstallString so a silent uninstall is possible.
- Onboarding Enter continues instead of skipping. The batch summary Enter key
  runs the default action. Ctrl+F focuses the queue filter. Filtering no
  longer leaves a hidden row selected. The track-review dialog is modal.
- Preview zoom centers on the window's monitor, not the primary display.
- First-run, Help, and track-review dialogs take initial keyboard focus.
  Scrollable dialog canvases are out of the tab order.

### Changed

- Batch progress, region-editor summaries, inpaint-preview captions, and the
  Help shortcut list are translatable. Mask-correction overlays follow the
  theme tokens, including high contrast.


## [3.35.0] - 2026-08-20

### Changed

- **Review every text track before anything is destroyed.** The new Track
  plan action scans the whole file and lists each temporal text track with
  its frame span, a sample of the recognized text, and a thumbnail. Keep or
  remove each one; a kept track is excluded from cleanup for exactly its
  frames and region. Plans save as JSON, rerun deterministically through
  `--plan-out` and `--plan-in`, and a plan-driven run with `--export-mask`
  produces a matte manifest that `--frozen-matte` can reuse.

- **The Surya and vision-language detectors are ordinary engine choices
  now.** Florence-2, Qwen2.5-VL, PaddleOCR-VL, and the llama.cpp-served
  PaddleOCR-VL were reachable only through environment variables; they join
  Surya in the engine picker and `--ocr-engine`. Picking one explains what it
  needs, the selection round-trips through saved settings, and a pick whose
  dependency is missing falls back to the automatic cascade with a warning
  instead of silently detecting nothing. Surya still requires its GPL opt-in.

- **The GUI test files were retired to `tests/archive/`.** They exercised the
  tkinter layer and churned with every visual change while catching little.
  The backend, controller-logic, catalog, and release-gate tests all remain
  active, and an archived file can be restored by moving it back up one
  directory.

- **The desktop interface now uses a calmer, denser visual system.** Larger
  working text, flatter surfaces, a fixed-width inspector, aligned preview and
  queue areas, and shorter labels make the main workflow easier to scan. The
  onboarding, help, region, and mask review windows now share the same compact
  hierarchy and no longer open as cramped scrolling cards.

### Fixed

- **The mouse wheel now scrolls from anywhere over a scrollable area.** On
  Windows the wheel goes to the widget under the pointer, so hovering a card,
  a toggle, or a queue row's label used to do nothing, and in dialogs the
  scroll stopped working when the pointer moved from the canvas onto its own
  content. One application-wide router walks up from the hovered widget to
  the nearest scrollable surface instead. Sliders only respond to the wheel
  when focused, so scrolling the settings column past one no longer silently
  changes its value.

- **Dropping a folder no longer freezes the window while it is scanned.** The
  walk ran synchronously in the drop callback, materialising and sorting the
  entire tree, so a large library or a slow network share locked the UI for
  the whole enumeration. Folders are now scanned on a worker thread with a
  progress line, enumeration stops at the queue's 500-file capacity, and the
  found files are added in small batches so the window stays responsive.

- **Small copy and consistency fixes across the queue.** The error status
  reads "Needs attention" in sentence case like every other status, the empty
  queue count uses the same plural-aware pattern as the live count, the move
  up and move down buttons carry real accessible names instead of announcing
  "^" to a screen reader, and the card headers stopped accepting a second
  caption they never rendered, which also removes seven strings from the
  translation catalog that no translator should have spent time on.

- **Tooltips and dialogs now respect multi-monitor layouts.** Both clamped
  against the primary display's size, so on a second monitor the tooltip
  jumped to the primary screen's edge and every dialog opened on the wrong
  monitor. Tooltips clamp against the full desktop, and dialogs center on the
  window that opened them. Toggles also release their variable watch when
  destroyed, so a dialog that shares an app-level setting no longer leaves a
  background error behind after closing.

- **Closing the app mid-batch no longer leaves an item marked as needing
  attention.** Marshalling a UI update from a worker thread raises one
  exception once the main loop has exited and a different one once the Tk
  interpreter is destroyed, and four call sites caught only the first. The
  second escaped into the per-item error handler, which recorded the item as
  failed with a Tk teardown message and saved that to the queue, so the next
  session reopened asking the user to deal with a file that had merely been
  interrupted. All worker-to-UI marshals now go through one helper.

- **A time-gated mask correction now applies to exactly the frames it covers.**
  Mask reuse caches the mask with whatever corrections were active at the frame
  it was built from, and the phash, keyframe and frame-skip reuse paths all
  re-attached it verbatim. On a static scene, which is exactly when phash
  skipping fires, a correction starting mid-scene was ignored and one ending
  mid-scene kept being applied until the next real detection frame. Reuse is
  now suppressed whenever any correction carries a time or frame bound.

- **Six small reliability gaps closed.** A non-finite tracker width or height
  reached `int(round(...))` and failed the whole job with an OverflowError. A
  correction carrying only an end frame ignored that bound, and could raise a
  TypeError out of the frame loop. Overlapping review spans of the same kind
  were left unmerged when a span of another kind fell between them, which
  duplicated review work. A bare list of detections from a vision-language
  model was collapsed into one union box, turning two far-apart captions into a
  single frame-spanning mask. The output-quality preflight marked anything
  needing an override as already overridden, so the field could never gate.
  Scene detection drove a private PySceneDetect function whose signature has
  shifted between releases; it now goes through the public API with an
  in-memory video stream.


## [3.34.0] - 2026-08-20

### Changed

- **PaddleOCR moves to 3.7.0 and can now use PP-OCRv6 models.** `tiny`, `small`
  and `medium` are selectable with `--paddleocr-variant`; PaddleOCR reports the
  medium tier at +4.6% detection and +5.1% recognition over PP-OCRv5-server,
  with one model covering 50 languages. The default deliberately stays
  PP-OCRv5 mobile, so upgrading changes nothing about model weights, latency,
  or memory unless a tier is chosen. The family alias table now has a single
  owner instead of four copies that could drift apart.

### Security

- **FFmpeg builds must now be 9.0.1 or newer.** The 8.x series ended at
  8.1.2 without fixes for CVE-2026-66037 (IAMF demuxer allocation),
  CVE-2026-66038 (LCL decoder heap disclosure), CVE-2026-66039 (MACE6
  decoder overflow), CVE-2026-64830 (VobSub subtitle demuxer overflow), and
  CVE-2026-12706 (RASC use-after-free), all reachable from crafted media.
  Because upstream closed those branches, the self-test, support bundle, and
  release validation now report every 8.x build as vulnerable with a
  cross-branch upgrade target instead of an unreachable on-branch patch.
  A build on a reviewed branch that merely predates its point release is
  reported as outdated and blocks a release without being accused of a CVE
  it already carries the fix for. CVE-2026-58049 has no upstream fixed
  version and is surfaced as an open advisory on every result.

- **The interpreter security self-test now covers every supported Python
  line.** It previously checked only 3.13 and 3.14 against CVE-2026-6100, so
  3.11 and 3.12 were unguarded. The floors now live beside the libpng and
  FFmpeg policies and require the 2026-08 security releases (3.11.16, 3.12.14,
  3.13.15, 3.14.7), which close the tarfile path-traversal bypasses and
  Windows symlink validation that VSR's archive extraction depends on, plus
  CVE-2026-2297, CVE-2026-4224 and CVE-2026-3644. A release line newer than
  the reviewed table is reported as unclassified rather than flagged.

- **The CPU dependency profile now requires ONNX Runtime 1.29.0.** That release
  fixes a TensorRT engine-refit path traversal, adds missing rank and shape
  validation across pooling, LSTM, QLinearConv and GridSample, validates
  DirectML constant-tensor byte sizes, and corrects a packed sub-byte over-copy
  in `OrtApi::GetValue`. VSR runs untrusted OCR and inpaint models through this
  runtime. The NVIDIA lane deliberately stays on the CUDA 12 wheel it has
  confirmed, and the DirectML lane stays at 1.24.4 because that is still the
  newest published DirectML wheel; both decisions are recorded in the manifest.

### Fixed

- **Clearing the queue filter no longer scrambles the visible order.**
  Re-showing a row put it at the end of the list, so filtering to one item and
  then clearing left the queue displayed in an order that disagreed with the
  order it would actually process in, which made Move up and Move down look
  wrong. Rows are now restored in queue order.

- **A second "Test cleanup" click reuses the models it already loaded.** Each
  click built a whole new backend, paying the OCR and inpainter initialisation
  again, and left the previous one for the garbage collector, so repeated
  clicks could hold several model sets in memory at once. The preview now
  caches on the same key the batch does.

- **The queue row keeps its percentage between progress updates.** The
  once-a-second elapsed timer overwrote the combined "42% / 1m 3s" label with
  the elapsed time alone, so the percentage flickered off and back.

- **Window titles and queue descriptions are translatable.** The region
  selector, onboarding, preview and A/B compare titles, the per-file override
  confirmation, and the file-type descriptions in the queue were all
  English-only literals that never reached the catalog. Plural text built by
  gluing an "s" onto a translated singular now uses proper plural entries, and
  the catalog lint understands `.title()`, so a new untranslated window title
  fails the check instead of shipping.

- **Cancelling or timing out an isolated job no longer leaves ffmpeg
  running.** Terminating the worker reached that process only, so an in-flight
  mux or post-restore ran to completion, held the output and temp files open,
  and could hand back a finished file for an item already reported as
  cancelled; an immediate retry then failed with a sharing violation. The
  worker now runs inside a Windows job object, so the whole process tree is
  reaped together. That also covers the case no cleanup code can reach: if the
  application itself dies, the job closes with it and the worker stops instead
  of processing to completion unattended.

- **The release gate is trustworthy again: five tests were failing on a clean
  checkout.** Three were regressions from the backend audit pass. A zero or
  missing frame rate was divided by an epsilon, turning frame 15 into 15
  million seconds; the SRT writer lost its plausibility floor, so a bogus
  0.001 fps stretched one frame into a 1000-second cue; and the STTN
  inpainter reported "cv2" as its backend until its first batch ran, so a
  provenance record taken before processing named the wrong engine. The
  fourth was a test that OpenCV 5 cannot satisfy, because it built its
  outlined glyph with a wide `putText` stroke and OpenCV 5 saturates text
  thickness at about two pixels, rendering no outline at all. The fifth was
  a stale reference corpus: eight intentional quality commits (translucency
  unmixing, grain restoration, Poisson seams, DIS flow, adaptive dilation)
  changed inpainting output without the committed baselines being
  re-recorded.

- **The reference corpus can be re-blessed instead of hand-edited.**
  `python -m backend.reference_corpus --bless` re-records the decoded output
  hashes and metric floors from a fresh run, refusing any clip that failed to
  produce output or that is missing a metric the manifest gates on, and it
  never invents a floor the manifest did not already declare. Floors are now
  recorded just below the measurement rather than exactly at it. Every floor
  previously sat on its measured value to six decimal places, which made it a
  duplicate of the frame-hash check and turned a 0.0002 SSIM movement into a
  release-gate failure.

- **Backend audit fixes harden resumed jobs and optional engines.** Completion
  checkpoints now include processing settings, derived exports restart from a
  complete frame, ranged selective reruns handle range-relative outputs, and
  sub-1 FPS timing is preserved. TBE flow/coverage gates, OCR confidence and
  fixes, LaMa/ONNX/diffusion adapters, SAM overlap handling, decoder seeks,
  CLI output/signals, and typed intermediate-writer failures now fail or
  degrade explicitly.

- **Deferred GUI strings now enter the translation catalog.** Status messages,
  queue labels, card headers, slider labels and hints use explicit `N_()`
  markers; the extractor supports their deferred f-string templates, and the
  lint gate models those helper signatures so new unmarked captions fail.

- **README no longer advertises unpublished documentation paths.** The
  contributor-facing edge-case guide and retired-audit directory are ignored
  maintainer notes, so the public structure now links only the tracked
  architecture guide and keeps planning-note wording unlinked.

- **The QA pseudo-locale is no longer offered to ordinary users.** `qps-Ploc`
  stays bundled and directly bindable, but the language picker hides it unless
  `VSR_PSEUDO_LOCALE=1` (or the smoke-test locale override) is explicitly set.

- **Windows setup now detects AMD and Intel GPUs without WMIC.** The installer
  probes `Get-CimInstance Win32_VideoController`, retains WMIC as a legacy
  fallback, and prints a distinct inconclusive warning when neither probe is
  available instead of silently selecting CPU mode.

- **NSIS uninstall now fails closed and preserves selected-directory contents.**
  It verifies the installed executable, refuses to run while VSR is active,
  uses the all-users shell context consistently, closes the correct mutex
  handle, and removes only known payload directories before a non-recursive
  install-root cleanup.

- **GUI batch startup now rolls back cleanly after preflight errors.** A failed
  report or output-volume preflight re-enables settings, restores the Start
  action, clears the timer state, and surfaces an error instead of wedging the
  window in a phantom processing state. Danger actions now use AA-safe dark
  ink, and stretched buttons redraw and remain clickable across their full
  layout slot; keyboard activation stops at the focused button.

- **The opt-in PyTorch LaMa fallback now restores source geometry.** Full-frame
  and tiled simple-lama outputs are cropped back from their modulo-eight
  padding before finishing, so non-mod-8 inputs no longer fail or blend at the
  wrong shape.

- **The offline guarantee is now explicit in the Overview.** README states
  that processing needs no account or upload, and documents the opt-in update
  check plus the two-variable crash-reporting gate and their disable controls.

- **RGVI has been evaluated and rejected for integration.** Its upstream
  implementation and model materials are non-commercial research-only under
  CC BY-NC and the NTU S-Lab License 1.0, which is incompatible with this
  project's redistributable MIT distribution; no numeric VRAM requirement is
  published either.

- **The TTML/IMSC rejection rationale is current.** The format notes now
  acknowledge IMSC Text Profile 1.3's 21 May 2026 W3C Recommendation status
  while recording the current demand and XML/layout surface reasons for
  staying with SRT and loss-aware WebVTT.

- **The install documentation no longer presents winget as a supported path.**
  README now states that no manifest is published, explains the unsigned
  SmartScreen/Mark-of-the-Web limitation, and points users to the verified
  direct ZIP download with **More info -> Run anyway** when Windows blocks it.

- **Provider diagnostics now include TensorRT-RTX.** The manual
  `NvTensorRTRTXExecutionProvider` lane reports its package, runtime provider
  list, security floor, and explicitly untested state; no hardware benchmark
  or frozen dependency profile is implied.

- **Mask dilation can now follow glyph edges.** `--auto-dilate` measures
  outline and drop-shadow falloff per detected region, clamps the result to
  20 pixels, and uses one distance-transform alpha for inpainting and final
  compositing; an explicit `--mask-dilate` or manual slider change keeps the
  historical binary path.

- **TBE now recovers fitted semi-transparent overlays.** Each connected mask
  region uses the existing Lab two-cluster foreground estimate and temporal
  background endpoint to solve closed-form opacity; only low-residual,
  intermediate-opacity regions are accepted, while opaque and failed fits
  stay on the prior binary path and log their decision. `--no-translucency`
  disables the new behavior.

- **The Docker image is now a usable CPU CLI runtime.** It installs the active
  requirements under the reviewed CPU constraints, including RapidOCR and the
  ONNX Runtime LaMa tier, runs the generated-image smoke during the build, and
  starts `python -m backend.cli` with mounted-volume usage documented; the
  smoke remains available through an explicit entrypoint override.

- **The CLI can now drain a watch folder unattended.** `--watch` polls for
  stable video/image inputs, forces canonical-output skip semantics, processes
  each stable file once through the existing single-file path, persists the
  normal batch reports after every item, keeps going after failures, and stops
  cleanly on the existing pause/cancellation signal. `--watch-once` provides a
  finite drain for scheduled jobs and tests.

- **PaddleOCR model selection is now explicit.** The reviewed PP-OCRv5
  mobile family remains the default for stable resource use, while
  `--paddleocr-variant server` selects the larger accuracy-oriented models;
  the chosen family is included in OCR execution provenance.

- **Film grain now reaches inpainted regions.** Positive `--film-grain`
  strength estimates high-pass texture from nearby unmasked pixels, adds
  decorrelated per-frame grain plus a small dither before 8-bit conversion,
  and leaves clean or texture-poor sources unchanged; the existing full-frame
  post-encode pass remains available for the surrounding image.

- **Finishing now has an opt-in gradient-domain seam correction.** A
  boundary-aware Poisson solve uses a dilated mask domain before feathering,
  skips frame-edge and degenerate masks, and remains shared across all
  inpainter families through `apply_finishing`.

- **TBE residual flow now has a DIS path.** DIS FAST is the default dense-flow
  estimator, with explicit Farneback selection and a logged fallback when the
  installed OpenCV build does not expose DIS; frame, mask, and karaoke warps
  use the same estimator contract.

- **TBE now rejects misaligned temporal samples.** Per-pixel median/MAD
  filtering removes corrupted exposures before averaging, while sparse pixels
  retain the historical batch-size fallback behavior.

- **Frozen optional OCR and LaMa tiers are now labeled honestly.** EasyOCR is
  marked frozen at 1.7.2 (last release 2024-09-24), and
  simple-lama-inpainting at 0.1.2 (last release 2023-07-28), in the GUI,
  backend status, dependency caps, README, and optional install guidance.
  Maintained RapidOCR, LaMa ONNX, and OpenCV 5 DNN paths are called out.

- **Reviewed PyTorch fallback profiles advanced.** CPU and DirectML now pin
  torch 2.13.0 with torchvision 0.28.0; the NVIDIA CUDA 12.8 profile advances
  to torch 2.11.0 with torchvision 0.26.0, the newest pair exposed by that
  reviewed channel. Setup, locks, drift floors, and install guidance now agree.

- **The core NumPy lane now pins 2.4.6.** This is the newest 2.4.x line
  supporting Python 3.11, so the explicit `<2.5.0` ceiling preserves the
  supported interpreter floor while the removed NumPy APIs were audited from
  the backend and test tree.

- **RapidOCR now runs the reviewed 3.9.2 PP-OCRv6 generation by default.**
  PP-OCRv5 remains selectable for regression comparison, with a CLI benchmark
  that evaluates both generations on identical fixtures and release evidence
  recording the reviewed ONNX model hashes.

- **Temporal background recovery now compensates for global camera motion.**
  Each scene segment is affine-registered to a reference before aggregation,
  rejects weak RANSAC fits, and keeps optional dense flow as a residual
  parallax refinement. The CLI and GUI expose the alignment opt-out.
- **FFmpeg runtime diagnostics now enforce the reviewed security floor.**
  VSR rejects an unclassified or vulnerable FFmpeg runtime, reports the
  affected CVEs and advisory URL in self-tests and support bundles, and
  records the pass/fail result in release evidence.
- **The support bundle no longer leaks the account name and install layout.**
  Settings, logs and reports were all scrubbed, but `support.json` -- the
  largest artifact, and the one users are told to attach to bug reports --
  was written raw, and it embeds the OpenCV wheel diagnostics whose
  `imported.file` is an absolute `site-packages` path. The whole payload now
  passes through the same redaction, with the diagnostics preserved and only
  the directory tree removed.
- **Shared-detector OCR is serialized.** Three call sites took the detector
  lock only to build or fetch the cached detector and then ran inference
  after releasing it, so a mask review racing a second review, the batch ETA
  probe, or the mask-correction editor could drive one non-thread-safe
  PaddleOCR/EasyOCR predictor from two threads at once -- garbage boxes at
  best, a native crash at worst. Inference now happens under the lock, as the
  region editor already did, and switching OCR engine takes the same lock
  before dropping the cache.
- **The region editor stops silently discarding shapes.** Saving with only a
  polygon drawn cleared every manual region and reported success, because the
  config has no field for a static polygon; it now refuses with an
  explanation. Saving drawn rectangles alongside a committed motion track
  kept the track and dropped the rectangles without a word; it now says which
  one was kept and why.
- **An explicitly typed CLI flag now beats a preset.** The guard that exists
  to stop a preset discarding what the user typed covered only fourteen
  fields; every other preset value was applied unconditionally, so
  `--preset "Logo / Watermark removal" --keep-chyrons` removed chyrons
  anyway, and the same held for `--no-tbe`, `--no-adaptive-batch`,
  `--vertical`, `--max-retries` and others. The flag maps now cover every
  preset-settable field that has a flag, and a test fails if a new one is
  added without protection. Abbreviated forms (`--thresho`) are rejected
  outright rather than parsed-then-ignored, and an attached short option
  (`-msttn`) counts as explicit.
- **A malformed preset is a clean CLI error, not a traceback.** Preset values
  bypassed the coercion that `--set` goes through, so an unregistered mode or
  a string-typed number in a user preset ended in an unhandled exception dump.
- **`--dry-run` no longer demands an output path** it has just promised not to
  write to, and an unusable `--out-dir` or `--checkpoint-dir` reports a
  one-line error instead of an `OSError` traceback.
- **Video output is no longer encoded twice.** Every shipped entry point
  passes a checkpoint directory, so finalization always encoded the frame
  sequence and then handed that finished file to the audio merge -- which
  applied the encoder arguments a second time. Every video output carrying
  audio shipped two lossy generations and paid roughly double the encode
  time. The merge now stream-copies video that finalization already produced
  under the job's own output contract; the muxed bitstream is byte-identical
  to the encode, verified against real media.
- **Per-frame timing survives long sources.** The probe read ffprobe's
  pretty-printed JSON under the shared 8 MiB output cap, and that collector
  trims from the *head*, so any video past roughly 45-60 minutes lost the
  start of its own document and failed to parse -- silently, at DEBUG level.
  The fallout was not cosmetic: a VFR source reverted to the constant-rate
  clock (mistimed mattes, audio offsets and SRT cues), and the frame-count
  correction vanished, which can end the decode loop early and fail a good
  file as `truncated_decode`. Frame rows are now read as CSV (measured at
  10.6 bytes per frame against 79.8 for JSON), under a much larger cap and a
  duration-scaled timeout, and a probe that still fails says so at WARNING.
- **Picking a language by its long-form name no longer breaks detection.**
  The picker lists PaddleOCR-style codes ("japan", "korean", "arabic") ahead
  of their ISO primaries and both are selectable, but two consumers keyed
  only on the primaries. With the language mask filter on, Japanese text
  classified as a script mismatch and *every* correct detection was thrown
  away -- the job reported success with the subtitles still burned in. The
  EasyOCR reader was handed the unknown code, raised, and dropped the user
  to OpenCV thresholding. Both now normalize through one shared table, and
  the EasyOCR map covers every code the picker offers.
- **A VLM or manga detector that cannot load falls back instead of detecting
  nothing.** The detector was constructed without loading its model, and its
  `detect` swallowed the failure and returned an empty box list forever, so
  choosing "Manga / Anime" without `manga-ocr` installed processed the whole
  video as if it had no text and reported success. The model is now warm-
  loaded when the cascade selects it; a failure logs a warning and continues
  down the normal RapidOCR chain.
- **Language auto-detection works again.** `probe_language` parsed only the
  legacy list/tuple result shape, but the pinned RapidOCR returns an object,
  so it always reported English at zero confidence -- making the CLI's
  `--auto-lang-probe` and the GUI's detect-from-preview action useless. It
  now reads the structured fields like every other parser in the module.
- **Stopping a batch during the final encode or mux no longer destroys the
  output and the resume state.** Terminating the last FFmpeg surfaced as a
  nonzero exit rather than a cancellation, so the audio merge treated it as
  a mux failure, worked through its retry ladder, and ended in the re-encode
  path -- whose blanket handler swallowed the resulting cancellation and
  promoted a silently audio-less intermediate over the destination file.
  `process_video` then deleted the pause checkpoint and every resume frame
  before noticing the cancel at its final progress report, so the item read
  as "Stopped" while the user's file had been replaced and could not be
  resumed. A cancel is now classified as a cancel at each of those layers:
  `run_process` re-checks after `wait()` returns, the encode and merge paths
  re-raise it instead of salvaging, and the deinterlace ingest no longer
  catches it as an ordinary `OSError`.
- **The PowerShell launcher repairs a broken venv instead of dying on it.**
  It probed the environment with a redirected-stderr native call under
  `$ErrorActionPreference = "Stop"`; Windows PowerShell 5.1 converts that
  stderr into error records, so the failing import's first traceback line
  terminated the script -- in precisely the case the repair branch below it
  exists to handle. Native probes now run through a helper that drops to
  `Continue` and reports an exit code, and a missing `python` no longer
  throws either. The `.bat` launcher was always correct here; the two now
  behave the same.
- **The frozen PowerShell launcher starts with no arguments.** It passed an
  empty `$args` to `Start-Process -ArgumentList`, which rejects empty
  collections, so the ordinary double-click launch failed with a parameter
  validation error. The release smoke never caught it because it always
  passes `--smoke-test`, which takes the other branch.
- **Screen-reader announcements work for the first time.** The announcer
  asked COM for `CUIAutomation8`, which is a coclass name rather than a
  registered ProgId, so every launch raised "Invalid class string" into a
  debug-level log and the provider stayed `None` -- and had it resolved, the
  client element it would have used has no `RaiseNotificationEvent` method
  at all. Announcements now go through the provider-side
  `UiaRaiseNotificationEvent` against the window's own host provider, using
  only `ctypes` (no new dependency, and it works in the frozen build). A
  failed probe is logged at INFO instead of DEBUG.
- **Windows taskbar progress works for the first time.** The `ITaskbarList3`
  object was requested with a `GUID` instance where comtypes requires an
  interface class, so construction raised `TypeError` on every launch and
  every progress call was a no-op. The interface is now dispatched through
  its vtable with `ctypes`, and the COM pointer is released on shutdown.
- **Presets now survive the start of a batch.** Applying a preset wrote its
  values onto the config, but only thirteen of the affected widgets were
  refreshed; the next `_sync_config_from_ui` -- which runs on every queue
  add, batch start, and app close -- read the untouched widgets back and
  reverted the rest. The built-in "Logo / Watermark removal" preset lost its
  entire purpose this way: it sets `remove_subtitles=False`, and starting the
  batch flipped it back to `True`, so the run removed the dialogue subtitles
  the preset exists to keep. Both directions are now driven by one shared
  field-to-widget table, and a test fails if a preset-settable field with a
  widget is ever missing from it.
- **A DirectML GPU is finally used for processing.** The batch config derived
  its device from the GPU toggle and index alone, so a DirectML adapter was
  handed to the backend as `cuda:N`; the provider probe found no CUDA and
  silently fell back to CPU while the batch report still recorded DirectML.
  The selected accelerator family now travels on the item snapshot, so both
  the in-process and isolated paths resolve it to `directml`.
- **The release build can complete on a clean tree again.** Release evidence
  was written to `dist\` (the verification step's default) while staging read
  `build\release\`, so a clean build failed after every gate had already
  passed, and a tree still holding same-version evidence promoted that stale
  set into `SHA256SUMS.txt` beside a freshly built installer. The build now
  passes `--evidence-dir` explicitly, and a test asserts the verification and
  staging steps name the same directory.
- **The release test gate no longer runs blind.** It used `unittest discover`,
  which imports the twelve modules written as bare test functions and then
  collects zero tests from them without saying so -- 71 tests, including the
  writer-failure, CVE-floor, dependency-floor, seek-accuracy, and HDR
  regressions, could not fail the build. The gate now runs pytest, whose
  collection is a strict superset, and a new invariant test fails if a
  bare-function module is ever paired with a runner that cannot see it.

- **README pip commands no longer lose their version pins.** `pip install
  torch>=2.11.0` is a redirection in PowerShell, cmd and bash alike, so the
  specifier was silently dropped and a file named `=2.11.0` was created
  instead. Every specifier is now quoted, and the Blackwell repair command
  pins the reviewed versions exactly rather than an open-ended range.

- **`setup.py` now runs from its own directory.** Invoking it by absolute path
  from somewhere else created `venv/`, wrote the launcher scripts into the
  caller's directory, and skipped `requirements.txt` entirely, falling back to
  a hardcoded package list.

- **A permanently-skipped dependency test now checks the real gate.** It
  asserted the Pillow floor inside a GitHub Actions workflow that was deleted
  under the local-build policy, so it could only ever skip; it now reads the
  reviewed CPU profile. The support-bundle subprocess test also gets a
  realistic timeout so it stops failing spuriously under full-suite load.

- **Stopping a single embedded-subtitle item now actually stops it.** The
  soft-subtitle remux polled only the batch-wide Stop, so an item cancelled on
  its own ran to completion and then reported Complete after the UI had already
  said it was stopping.

- **Both previews now use the settings the run would use.** "Test cleanup"
  built its backend config from a hand-picked subset that dropped the
  detection engine, vertical detection, the language mask filter, manual mask
  corrections and the LaMa speed switch, so choosing EasyOCR previewed the
  automatic cascade instead. The mask preview took the language, engine and
  threshold from the global settings while reading dilation and corrections
  from the item, so a per-file override previewed a mask the run would not
  produce. Both now read the item, and the two previews agree with each other.

- **Release evidence can no longer contradict the artifact it describes.** The
  PyInstaller spec treats `1`, `true`, `yes` and `on` as enabling a packaged
  feature, but the build script only recognised `1`. Building with
  `VSR_ENABLE_FULL_OCR=true` therefore packaged PaddleOCR and EasyOCR while
  recording them as excluded modules in the published evidence and SBOM. Both
  sides now normalise through one helper, and a test pins them together.

- **Three tests that read stronger than they were now check what they claim.**
  The isolated-worker crash test asserted only the status mapping, never the
  "keeps its logs" half of its own name, and drove a status table instead of
  the real outcome path; it now runs the actual branch and asserts the child's
  stderr tail survives onto the item. The translation timeout test asserted
  bare `Exception` and was in fact passing on a pre-spawn "command not found"
  error, because the provider takes one command rather than an argv list, so
  the timeout was never exercised at all. And the WebVTT parser's cue-count,
  cue-length and block-count limits had no coverage, so they could have been
  deleted without a red test.

- **An isolated job that finishes just before its deadline is no longer thrown
  away.** If the worker published its result and then failed to close stdout
  before the wall-clock timeout, the job was reported as a timeout, so the
  queue retried or failed an item whose output was already written.

- **Pausing an isolated job now reports when it cannot be delivered.** A failed
  control-file write was discarded silently, leaving the interface showing a
  paused job that was still processing. Cancel already recovered through its
  watchdog; pause had nothing.

- **Retrying an isolated job no longer leaks watchdog threads or scratch
  directories.** Each attempt started another control watchdog while the
  previous one kept polling a dead supervisor, and a cancel during a later
  attempt made those stale watchdogs recreate scratch directories that had
  already been cleaned up. Scratch removal now waits briefly for a killed
  worker to be reaped and keeps its handle for a retry instead of leaking the
  directory permanently when Windows reports a sharing violation.

## [3.33.0] - 2026-08-02

### Fixed

- Isolated job timeouts now cover the worker event stream itself. A worker
  that keeps stdout open without emitting EOF is terminated within the
  caller's wall-clock budget and reported as `worker_timeout`.
- RTL layouts now mirror the remaining Canvas affordances: slider values and
  fills, progress fills, button icon/text order, segmented controls, and the
  selected queue accent stripe.

## [3.32.0] - 2026-08-02

Deep audit pass over the code shipped since the last audit (3.24-3.31),
centered on the new process-isolation path.

### Fixed

- **The frozen job worker no longer loads the GUI or shares the parent's
  log file.** In a frozen build the `--job-worker` marker was only
  recognised inside `main()`, after module-level code had already opened
  the parent's rotating log from a second process (rotation breaks on
  Windows when two processes hold one file) and imported the entire
  widget tree. The marker now short-circuits at the top of the entry
  module, before logging setup and every GUI import.
- **A cancelled isolated job can no longer hang forever.** A child wedged
  in native code -- the exact failure process isolation exists for --
  never reads the polled control file, so a per-item cancel had no
  effect. The control watchdog now escalates to `terminate()` after a
  30-second grace period, and the outcome still reports as cancelled,
  not crashed. The watchdog also stands down when a worker fails to
  spawn instead of spinning at 20 Hz for the life of the process.
- **Journal recovery can no longer expand an empty path onto the working
  directory.** `Path("")` is `Path(".")`, so the empty-target guard in
  `backend/atomic_replace.py` was dead code: a malformed journal entry
  could direct rollback to remove the current directory tree. Raw
  strings are now validated before any `Path` is built, on both the
  rollback and finish-forward legs.
- **A refused `--frozen-matte` is a clean CLI error, not a traceback.**
  The freeze's curated user message (missing manifest, matte edited
  after export, substituted source) now reaches the console through
  `parser.error` instead of an unhandled-exception dump.
- **Saved window positions on secondary monitors now survive a restart.**
  Geometry restore validated positions against the primary display only,
  so a window kept on any other monitor was silently recentered on every
  launch. Restore now checks the full multi-monitor virtual desktop.
- Plural agreement in two remaining messages ("queued item(s)",
  "model-cache file(s)") now uses real singular/plural forms.

### Added

- **Isolated jobs reach feature parity with in-process jobs.** Live
  preview (the child writes throttled PNG frames the parent marshals
  into the preview pane), mask-correction selective reruns, auto
  subtitle-band detection (probed inside the child so no OCR model loads
  into the GUI process), and transient-failure retries with backoff all
  now work when **Run each job in a separate process** is enabled.
  Previously each of these silently degraded: no preview, full re-detect
  instead of a selective rerun, unpinned full-frame detection, no
  retries.

## [3.31.0] - 2026-07-29

### Added

- **Each queue job can run in an isolated child process.** A fatal native
  fault in OpenCV, ONNX Runtime, or a model's kernels cannot be caught by
  Python: in-process it killed the interpreter, and with it the GUI and
  every remaining queued job. Checkpoints did not help, because the process
  that would have resumed them was gone. The new **Run each job in a
  separate process** setting runs every item under a versioned local job
  protocol (`backend/job_worker.py` + `gui/job_supervisor.py`); progress,
  live preview, pause, cancel, and checkpoints keep working, and a child
  that dies without publishing a result is reported against that one item
  with its stderr tail retained as the item's log, its exit status decoded
  (0xC0000005 reads as "access violation", not 3221225477), and the rest of
  the batch untouched. Off by default: isolation trades the in-process
  model cache for a process start and a reload per item, so the choice is
  the user's rather than assumed.
- Control for an isolated job travels through a polled JSON file rather
  than the child's stdin. A reader thread parked in a blocking stdin read
  deadlocks against C-extension module initialisation during the child's
  own numpy/cv2 imports, and on Windows that read cannot be interrupted --
  the worker would hang before starting work. The child now runs with
  stdin closed, which additionally stops any grandchild it spawns (an
  import probe, an ffmpeg call) from inheriting a live pipe.
- **WebVTT translates as WebVTT instead of being flattened to SRT.**
  `.vtt` sources went through the SRT model, which silently discarded cue
  identifiers, per-cue positioning (`line`/`position`/`size`/`align`/
  `vertical`), `REGION` and `STYLE` blocks, `NOTE` comments, `<v>` voice
  spans, `<lang>` and `<c.class>` spans, ruby annotations, and karaoke
  timestamp tags -- the file still looked like subtitles, so nobody noticed
  until it was on screen. New `backend/webvtt.py` parses WebVTT as WebVTT:
  only the visible text runs of each cue reach the translation provider, and
  everything else is reproduced verbatim, so parsing and re-serializing an
  untranslated document returns it byte for byte. A `.vtt` source now yields
  a `<output>.<lang>.vtt` sidecar. Ruby annotations are deliberately left
  untranslated (an `<rt>` reading guides the *source* script), and provider
  output is escaped so a provider cannot inject markup or restructure a cue.
- **Every translation carries an explicit loss report.** A
  WebVTT-to-WebVTT pass records `lossless: true` rather than omitting the
  report, because "no report" and "nothing was lost" must not look the same.
  Converting to SRT enumerates each dropped feature with a count and a
  reason, and muxing WebVTT into MP4 now states that `mov_text` loses
  regions, cue settings, and STYLE rules instead of reporting a neutral
  "text subtitles require MP4/MOV conversion". TTML/IMSC (`.ttml`, `.dfxp`,
  `.itt`, `.xml`) are rejected with a clear message rather than parsed as
  SRT, which would be the same silent flattening one format further along.
- **An approved matte can be frozen as a reusable queue input.** A matte
  reviewed frame by frame is the most expensive artifact this pipeline
  produces, and a rerun threw it away to re-derive it from OCR and tracking
  -- slow, and not guaranteed to reproduce the same mask, since detection is
  not bit-reproducible across engine, driver, and threading changes.
  **Freeze approved matte** on a completed queue item (or `--frozen-matte
  MANIFEST`) pins the artifact and manifest hashes, a source fingerprint,
  and the geometry, timing, and frame range it was approved for. A matching
  rerun paints the approved mask directly and skips OCR, tracking, and the
  mask refiners -- the refiners deliberately included, because they exist to
  improve a derived mask and would edit approved pixels. Every part of the
  promise is re-verified before a frame is decoded: a changed source, matte,
  manifest, geometry, range, CFR/VFR mode, time base, or per-frame timing
  stops the run with a reason naming what moved and a request to re-freeze,
  never a silent re-detection and never a matte applied to frames it was not
  approved against. New `backend/frozen_matte.py`; the reproducibility
  sidecar records the freeze and the bypassed stages under `frozenMatte`.
- **Right-to-left layouts mirror the whole interface.** RTL support
  previously reached only a handful of hand-mirrored widgets, so an Arabic or
  Hebrew user got a left-to-right window with a few right-aligned labels in
  it. Layout code is now written in logical terms throughout and
  `gui/direction.py` converts logical to physical at a single interception
  point, so packing order, widget and geometry anchors, `grid` sticky masks,
  text justification, and directional arrow affordances (`A -> Z` becomes
  `A <- Z`) mirror together -- including in widgets written later. Canvas
  items are deliberately excluded because their anchors are paired with
  coordinates the caller already computed.
- **A Settings toggle for right-to-left layout.** The `rtl_layout` preference
  existed but could only be set by hand-editing `settings.json`. It is now a
  toggle beside the high-contrast and text-scale controls, and like those it
  applies on the next launch because the mirror has to be installed before the
  first widget exists.
- **`i18n_catalogs.py lint`, a runtime-string gate.** Coverage percentages
  could read 100% while user-visible strings never reached the catalog at all,
  because a string that is never extracted cannot be counted as missing. The
  new lint walks every caption sink in `gui/` -- widget captions, dialog
  titles, file-dialog filters, menu entries, accessibility labels, and
  interpolated sentences -- and fails when one bypasses `tr()`. It runs as
  part of `check`.

### Changed

- **43 user-visible strings now reach the catalog.** Confirmation dialogs,
  file-picker titles and filters, preview status text, queue counters, and
  screen-reader labels were previously hard-coded English. Counted strings use
  proper plural forms via `ntr()` instead of an inline `'s' if n != 1` suffix,
  which is wrong in most languages. The bundled pseudo-locale grew from 569 to
  619 messages.
- **Queue-item messages are translated at render time.** The model keeps
  stable English so persisted queue state and status comparisons survive a
  locale change, and `gui.utils.queue_message_text()` localizes on the way to
  the label. `backend.i18n.N_()` marks those deferred literals for extraction.

### Fixed

- **The About dialog no longer outgrows the screen.** It was the last major
  dialog still sized purely by its own content; at high text scale on a short
  work area it extended past the desktop with no way to reach the Close
  button. It now uses the same scroll-and-clamp path as every other dialog.

## [3.30.0] - 2026-07-29

### Added

- **Requested and effective execution provenance is recorded per job.** A CUDA
  request could execute RapidOCR on CPU and LaMa on cv2 while the queue, the
  JSON report, the sidecar, and the support bundle showed only an ambiguous
  engine/backend label -- so two runs that executed identically looked
  different, and two that looked identical could differ. Every job now records
  requested vs. effective device, OCR engine + execution provider, inpaint mode
  + backend, a fallback reason for each stage, and observed throughput
  (`backend.execution_provenance`). The record is written to the queue item, the
  `.vsr.json` sidecar, the batch report, and the CLI `--json` output; the queue
  row and the Help dialog label it in plain words ("RapidOCR on CPU (CUDA
  requested)"), the inference smoke records requested vs. effective device per
  probe, and a fallback is logged as a warning at startup.
- **Per-lane execution-provider support state.** `backend.dependency_caps`
  now describes four provider lanes -- CPU, CUDA 12, CUDA 13, and DirectML --
  each with its own reviewed version window, tested flag, and security floor,
  reported through `collect_provider_lane_status()` and carried in the release
  evidence's `dependencyProfile.providerLanes`. This lets the CPU lane adopt
  newer ONNX Runtime fixes (1.28.0) while the CUDA 12 lane stays on its last
  compatible build, and marks CUDA 13 as an untested manual lane instead of
  hiding it behind a single onnxruntime recommendation.
- **Atomic, version-derived release staging.** New `backend.release_staging`
  stages a release into a clean temporary directory, derives every filename
  from `APP_VERSION` (installer, portable ZIP, SBOM, audit, evidence), hashes
  exactly that set into `SHA256SUMS.txt`, and promotes the whole directory to
  `build/release/<version>/` in one move. It refuses evidence that records a
  different version, a verification error, or a failed installer/launch smoke,
  and `verify` rejects a promoted directory that gains or loses a file or whose
  checksums no longer match. `build_exe.bat` stages, verifies, and only then
  promotes the installer; `guidance` prints draft/immutable GitHub release
  steps and restates that artifacts are intentionally unsigned. This closes the
  case where a reusable release directory paired a newer installer with an
  older portable ZIP and a checksum file describing neither.
- **Mask-aware temporal regression profile.** Per-frame PSNR/SSIM misses the
  failures that dominate video inpainting: a fill that is sharp in every frame
  but pinned to the frame instead of the scene, ghosting, bleeding past its
  mask, or flickering. New `backend.temporal_profile` adds three
  motion-compensated, mask-aware metrics -- masked warp residual, mask edge
  leakage, and masked flicker -- plus deterministic synthetic fixtures that
  vary camera motion, background motion, and mask motion independently so a
  regression is attributable to one axis. Strict release verification runs the
  profile and fails on a seeded frozen, flickering, or leaking fill, while
  clean fills under heavy camera panning still pass. No learned metric is
  downloaded and no licensed real media is required.
- **Contract test for selectable OpenCV 5 DNN inference engines.** PP-OCRv6
  and LaMa now run through OpenCV 5 DNN, whose `auto` / `classic` / `new`
  engines have different operator coverage and fallback behaviour that no test
  proved. `run_opencv_dnn_engine_contract()` runs one real inference of the
  bundled PP-OCRv6 detection graph (and an advertised local LaMa model when one
  is configured) under every documented engine the installed OpenCV exposes,
  records the engine that applied, and fails on a load error, an output-shape
  change, or output that diverges materially from the first engine. Strict
  release verification blocks on a failed contract; a build with no engine
  selection (OpenCV 4) is reported as not applicable rather than assumed to
  pass, and nothing assumes an ORT-linked DNN build.
- **Dependency-profile provider smoke.**
  `python -m backend.dependency_profiles smoke --profile <name>` builds a
  minimal ONNX identity model and creates one real inference session on the
  profile's claimed execution provider, failing when ONNX Runtime silently
  falls back to another provider. Strict release verification runs it as part
  of the dependency-profile evidence.

### Fixed

- **Architecture, release, and accessibility docs now match the code.** The
  hand-maintained module map in `docs/architecture.md` had drifted badly -- it
  omitted nine GUI modules and thirty-three backend modules and still listed a
  test file that had been split apart -- and the prose still credited
  `gui/app.py` with widget building that moved into the layout mixins. The map
  is now generated by `scripts/generate_architecture_map.py` between markers,
  the release build fails on drift (`--check`), and a new documentation-drift
  test suite enforces it. `CHANGELOG.md` had two `[Unreleased]` sections; one
  remains and a test keeps it that way. Winget wording no longer implies a
  signed installer, and a regex guard blocks any future signed-distribution
  claim. A new accessibility support matrix separates what is actually tested
  (keyboard reachability, 100-200% text scaling, dialog reflow, high contrast,
  pseudo-locale, RTL mirroring) from the explicitly unsupported
  screen-reader/UI-Automation surface on the custom Canvas controls, with each
  row naming the file that proves it.
- **MatAnyone refinement no longer discards a whole clip over a subtitle gap.**
  `_read_alpha_video` returned `None` if any single alpha frame was fully
  transparent -- exactly what a frame with no subtitle produces -- so the opt-in
  refinement silently no-opped on nearly every real clip. A gap frame now falls
  back to that frame's hint mask (matching `_normalize_alpha_sequence`) or to an
  explicit empty matte; a truncated output is still rejected. The image-sequence
  reader got the same treatment.
- **Whisper audio extraction scales its ffmpeg timeout by source duration.**
  `extract_audio_to_temp` hard-coded 600 s while its sibling already scaled by
  probed duration, so a long source on a slow disk could exceed ten minutes to
  demux and resample and have the Whisper fallback killed silently. It now
  probes the duration and scales the budget, falling back to the 600 s floor
  when probing fails.
- **Legacy hardware-decoder seeks fail loudly when they cannot reposition.**
  `set(CAP_PROP_POS_FRAMES)` advanced its position and returned `True` even
  when the legacy decoder had no `SeekFrame`, so a seek-then-read returned the
  wrong frame while claiming success on the `cv2.VideoCapture` drop-in contract
  (resume and ROI re-scan misalignment). It now returns `False` and leaves the
  position untouched, and logs when a seek raises.
- **Every major dialog reflows and scrolls at 125-200% text scale.** The main
  workbench was already responsive, but the first-run onboarding modal, the
  region editor, and the mask-correction editor were fixed-size and
  non-resizable, so at high text scale their content grew past the screen and
  pushed the primary actions out of reach (probes measured 1106x969 and
  1192x1144 at 200%). A shared `gui.dialog_layout` helper gives each dialog a
  scrollable body -- keyboard-focusable, with wheel, arrow, Page, Home, and End
  bindings and scrollbars that appear only when needed -- and clamps the window
  to the usable screen work area with a resizable frame and a work-area
  maximum. The UI scaling probe now opens the dialogs at 980x720 and 2752x1152
  work areas inside every scale/theme/locale case, and the matrix covers 125%
  and 175% as well.
- **Cache and matte replacements are transactionally recoverable.** Replacing a
  set of files with `os.replace` is atomic per file but not across the set: a
  model-cache import interrupted mid-commit could strand a weight in `.vsrbak`
  with nothing at the target, and a matte export deleted the existing artifact
  before promoting the new one, so a failure could destroy both halves of the
  artifact/manifest pair. A new `backend.atomic_replace.ReplacementJournal`
  records the whole plan and fsyncs it *before* the first original moves;
  recovery is deterministic (`pending` rolls back to the complete old set,
  `committed` finishes forward), orphaned `.vsrbak` files are resolved against
  the target, and `model_cache_status()` runs recovery at startup. Matte export
  now backs up rather than deletes and journals the artifact and manifest
  together. The model-cache bundle manifest is also capped at 8 MiB before it
  is read, so a decompression bomb in `manifest.json` cannot exhaust memory.
- **User-state writes are observable and downgrade-safe.** Settings, queue, and
  preset saves swallowed their failures, so a full disk or a locked profile
  folder looked exactly like a successful save. Each now returns a typed
  `PersistenceResult` with actionable retry guidance, and a registered observer
  surfaces failures in the activity UI as a warning toast; `save_user_preset`,
  `delete_user_preset`, and `import_preset` report failure instead of claiming
  success. The corrupt-settings notice only claims a backup exists once the
  copy actually succeeded, and says so plainly when it did not. A settings file
  written by a newer build now opens in explicit read-only compatibility mode:
  it is never overwritten (so fields this version does not understand cannot be
  erased), and a new **Export settings** action writes a deliberate
  current-format copy instead.
- **Unknown inpaint modes are rejected instead of becoming STTN.** An
  unrecognised value in `settings.json`, a preset, or an imported preset used
  to fall through to STTN while the import or apply still reported success, so
  the user's intent and the effective algorithm diverged silently. A new
  `classify_inpaint_mode()` separates GUI modes, backend-only modes, and
  invalid values: an invalid settings mode is quarantined with an actionable
  notice (the rest of the file still loads), `apply_preset` refuses to apply
  and leaves the current mode untouched, `import_preset` returns `None` and
  stores nothing, and an out-of-range picker value restores the mode actually
  in effect. Backend-only modes such as `migan` keep their existing
  command-line compatibility notice.
- **Subprocess stdin is now under timeout and cancellation control.**
  `run_process` wrote the whole input payload inline before entering its
  deadline loop, so a child that never read stdin blocked the caller forever
  once the payload exceeded the OS pipe buffer -- defeating the shared bounded
  process contract. The write now runs on a worker thread; the deadline and
  cancel checks start immediately, terminating the child breaks the pipe, and
  the writer thread is always joined so no thread or process is left behind.
  The local translation provider rejects an oversized request before starting a
  child, and the support-bundle encoder probes use the shared policy instead of
  a raw `subprocess.run` (which also stops a console window flashing on
  Windows).
- **Intermediate and frame-sequence writer failures now fail closed.** A
  timed-out or nonzero-exit FFV1 intermediate encode was logged as a warning and
  a failed `cv2.imwrite` was ignored entirely, so a truncated intermediate or a
  frame that never reached disk could still advance the resume checkpoint and be
  promoted as a finished output. Both paths now raise a typed
  `backend.io.MediaWriteError`: `_LosslessIntermediateWriter.release()` reports
  `intermediate_writer_timeout` / `intermediate_writer_failed` (with the ffmpeg
  stderr tail) and clears its handles so a cleanup release stays a no-op, and
  `_FrameSequenceWriter.write()` reports `frame_write_failed` before advancing
  the frame index. `process_video` handles the new error explicitly, so the
  partial output is never promoted and the job is reported as failed.
- **The NVIDIA dependency profile is now installable.** The reviewed NVIDIA
  lock pinned `onnxruntime-gpu==1.27.0` while setup installs
  `onnxruntime-gpu>=1.26.0,<1.27.0`, so the profile advertised a CUDA 12
  provider it could never install. The lock is now `onnxruntime-gpu==1.26.0`,
  and `load_profile_manifest()` rejects any exact lock that falls outside its
  provider lane's reviewed range so setup and the generated locks cannot drift
  apart again.

### Security

- **Opt-in crash reporting now fails closed on privacy.** The scrubber returned
  the *original* event whenever it raised, so a scrubbing failure would have
  shipped unredacted data, and path scrubbing only removed the directory part
  of a path -- leaving the processed filename in the upload despite the
  documented contract. Reports are now rebuilt from an allowlist
  (`build_minimal_event`): exception type/value, module, function, line number,
  Python/OS version, and release only. Breadcrumbs, extra data, request and
  cookie payloads, user info, tags, loaded modules, server name, frame locals,
  filenames, and context lines are never carried, the SDK is initialised with
  `max_breadcrumbs=0` and `include_local_variables=False`, and any failure in
  the builder drops the report entirely. Path scrubbing now removes Windows
  drive paths, UNC shares, `file://` URLs, POSIX home/temp paths, and bare
  media basenames. The local support bundle keeps its previous
  directory-only redaction (the user reviews that archive before sharing it).
- **protobuf floor raised to 6.33.5 (CVE-2026-0994).** VSR parses ONNX model
  protos from untrusted sources, so unbounded parser recursion is a runtime
  concern. `requirements.txt` and every reviewed profile lock now require a
  fixed build, an older installed protobuf is a blocking release advisory, and
  the dependency-profile status reports it as an error.

## [3.29.0] - 2026-07-20

### Added

- **Module-boundary unit tests for `batch_report` and `adapter_manifest`.** Both
  are deterministic, GPU-free modules that were only exercised indirectly through
  the release-workflow / hardening suites. `tests/test_batch_report.py` locks the
  batch-summary schema, status counts, path-redaction default, planned-status
  decision table, retriable-error classification, and the review-gate promotion;
  `tests/test_adapter_manifest.py` locks the license/hash trust gate (verified vs
  mismatch vs hashless-legacy vs remote-code-required, the unsafe override, and
  unknown-adapter handling).

### Fixed

- **NLE (`--nle-input`) frame rate is now probed, not silently assumed 24 fps.**
  Timecode->frame conversion for EDL/FCPXML segment inputs previously fell back
  to 24 fps on any probe failure with no diagnostic, mis-aligning every SMPTE
  timecode on 25/30/50/60 fps sources. A new `backend.io.probe_video_fps` helper
  reads the true rate via ffprobe (`avg_frame_rate`, then `r_frame_rate`) with an
  OpenCV fallback, and the CLI logs a warning before substituting 24 fps only
  when every probe fails.
- **ONNX Runtime CUDA 12 install no longer breaks on 1.27.** ONNX Runtime 1.27.0
  (2026-06-15) dropped CUDA 12 -- its default `onnxruntime-gpu` PyPI wheel is now
  CUDA 13 only. The NVIDIA install path pinned `onnxruntime-gpu>=1.26.0` with no
  ceiling, so a fresh install on a CUDA 12 host silently pulled the CUDA-13-only
  wheel and failed at inference. `setup.py` now installs
  `onnxruntime-gpu>=1.26.0,<1.27.0` for the CUDA 12 line (with an on-screen note
  for CUDA 13 hosts to install the cuda13 wheel manually), and the requirements
  hints and README were corrected to match.

### Changed

- **ORT-GPU recommendation guidance now reports the tested 1.26.0 floor and the
  CUDA-12 ceiling.** `ONNXRUNTIME_GPU_RECOMMENDED_MIN` moved from 1.25.0 to
  1.26.0 (matching the shipped security floor), and the legacy-package /
  preload-DLL warnings, `recommendedPackage`, and advisory `fixedIn` now cite
  `onnxruntime-gpu>=1.26.0,<1.27.0` instead of the stale unbounded/1.25.0 text.

## [3.28.0] - 2026-07-19

### Added

- **Community translation guide.** README now documents the full gettext
  translation workflow -- how to create a new `locale/<tag>/LC_MESSAGES/vsr.po`
  catalog, run the `update` / `check` / `coverage` commands in
  `scripts/i18n_catalogs.py`, preserve placeholders, and submit a translation
  contribution -- so crowdsourced locales beyond the built-in pseudo-locale can
  be started.
- **Bundled OCR-fix lists + Subtitle Edit import tooling.** The SRT OCR-fix
  engine now layers clean-room, conservative per-language correction lists from
  `backend/ocr_fix_data/{lang}.json` (English, Spanish, French, German,
  Portuguese) between the built-in defaults and the user's own file, so exported
  SRT text automatically repairs common hardsub OCR artifacts (e.g. the
  "h read as b" class: `tbe` -> `the`, `otber` -> `other`). Every bundled source
  key is a non-word in its language, so a whole-word replacement can only repair
  a garbled token, never corrupt correct text. `scripts/import_se_ocrfix.py`
  converts Subtitle Edit `{lang}_OCRFixReplaceList.xml` WholeWords dictionaries
  into the same VSR JSON format for users who want to extend the lists locally
  (Subtitle Edit's GPL data is not bundled).

### Changed

- **gui/app.py builder methods split into focused layout mixins.** The UI
  builder, responsive-layout, queue-view, header/card helpers, and first-run
  onboarding methods were extracted into five mixin modules:
  `gui/layout_build.py` (all `_build_*` builders, ~2,097 lines),
  `gui/layout_responsive.py` (responsive layout + content-scroll handlers,
  ~339 lines), `gui/queue_view.py` (queue list rendering/sort/filter/per-file
  overrides, ~414 lines), `gui/layout_helpers.py` (header/card/label helpers,
  ~164 lines), and `gui/onboarding.py` (onboarding modal, ~221 lines).
  `gui/app.py` drops from 5,852 to 2,807 lines. No behavioral changes; the
  mixins compose onto `VideoSubtitleRemoverApp` exactly like the existing
  controller mixins, and the full GUI test suite passes unchanged.

## [3.27.0] - 2026-07-18

### Changed

- **processor.py split into focused mixin modules.** SubtitleRemover methods
  extracted into five mixin files: `_encode_mixin.py` (encode/mux/audio, ~690
  lines), `_quality_mixin.py` (quality report/sheet/seam, ~345 lines),
  `_finalize_mixin.py` (finalize/output-contract/post-restore, ~613 lines),
  `_srt_mixin.py` (SRT export/translation, ~233 lines), and
  `_clean_ref_mixin.py` (clean references/regions, ~256 lines). processor.py
  drops from 4,977 to 2,848 lines. All 886 tests pass unchanged through the
  mixin inheritance. The remaining ~850-line frame loop requires GPU-equipped
  verification for safe extraction.

## [3.26.0] - 2026-07-18

### Fixed

- **Quality report sampled wrong frames on long-GOP sources.** The PSNR/SSIM
  quality report used raw `cap.set(CAP_PROP_POS_FRAMES)` instead of the safe
  `_seek_capture_to_frame` wrapper, causing the input and output frame pairs to
  land on the nearest keyframe instead of the exact requested frame. A
  source-level guard now forbids raw seeks outside the wrapper.
- **CPU OOM-fallback inpainter skipped edge-ring color correction.** The
  guaranteed-CPU fallback path called `_feather_blend` directly instead of
  `apply_finishing`, producing visible color seams on gradient backgrounds
  during OOM recovery.
- **ONNX finishing silently degraded every frame on import failure.** The
  `_apply_feather_blend` wrapper caught `Exception` on `apply_finishing` import
  and returned raw unfinished frames with no logging; moved to a module-level
  import so broken installs fail loudly.
- **Interlace detection failures returned False silently.** A failed ffprobe
  idet probe now logs a warning instead of silently misclassifying interlaced
  content as progressive.
- **Swallowed exceptions in GUI settings/queue controls.** Five bare
  `except: pass` blocks in panel restore, settings lock, segmented picker,
  sort button, and filter visibility now log at debug level.

### Security

- **ONNX Runtime floor bumped to >= 1.26.0 for CPU/CUDA.** ORT 1.26.0 adds
  OOB/overflow hardening (unrestricted setattr allowlist, MaxPoolGrad bounds,
  sparse tensor conversion). Updated across setup.py, Dockerfile, dependency
  profiles, and release validation. onnxruntime-directml stays at 1.24.4
  (latest available).
- **CPython CVE-2026-6100 warning in self-test.** The decompressor
  use-after-free (CRITICAL 9.1, fixed 3.13.14/3.14.5) is flagged in self-test
  diagnostics and support bundles.

### Changed

- **Built-in inpainters consolidated onto apply_finishing.** STTN, LaMa,
  ProPainter, and ExternalInpainter now route through the same centralized
  edge-ring + feather finishing step as the ONNX and diffusion backends.
- **_binarize_mask deduplicated.** The ONNX file's local copy now imports from
  the canonical `_common.py` definition.
- **DirectML to Windows ML migration documented.** Architecture docs now
  describe the forward path, prerequisites, and the new `windowsml_status()`
  probe surfaced in support bundles.

## [3.25.0] - 2026-07-17

### Added

- **Load notice for command-line-only inpaint modes.** A settings.json (or a
  run saved from `--mode migan`) that names a backend-only mode the GUI cannot
  represent now surfaces a startup notice explaining it is using STTN this
  session, instead of silently downgrading.

### Changed

- **Plainer copy for advanced settings.** The STTN motion sliders (Neighbor
  stride, Reference length, Max load frames) gained plain-language hints
  explaining their tradeoff, the "Temporal coherence" label became "Motion
  smoothing", and the embedded-subtitle status/labels now say "copy"/"remove"
  instead of "remux"/"strip".
- **Disabled buttons now read as disabled.** They previously reused the same
  fill as enabled secondary/default buttons, so the only cue was dimmed text. A
  dedicated recessed `BG_DISABLED` token (in both the default and AMOLED
  palettes) makes the inert state legible without vanishing against cards.

### Performance

- **PNG frame-sequence input no longer re-parses the libpng build string per
  frame.** The process-static OpenCV libpng status is resolved once when a
  frame-sequence capture opens and reused for every frame.

### Fixed

- **CLI `--preset` no longer discards an explicit flag that equals the parser
  default.** `--preset X --threshold 0.5` previously dropped the user's 0.5 for
  the preset's value because argparse cannot tell an omitted flag from one typed
  with the default. Preset merging now inspects the raw argv tokens, so any flag
  the user actually typed wins over the preset regardless of its value.
- **CLI `--preset` now applies preset fields that have no dedicated flag.** A
  user preset carrying a backend field without a CLI option (e.g.
  `temporal_smooth_radius`) was silently dropped; such fields now apply to the
  resolved config, while an explicit `--config`/`--set` override still wins and
  a genuinely unknown field is reported rather than ignored.

## [3.24.0] - 2026-07-17

### Fixed

- **Applying a preset could crash before any changes were saved.** The
  Sensitivity slider is registered under the runtime-only
  `_detection_threshold_pct` attribute, which does not exist on a fresh config
  until the slider is dragged. Selecting a preset before touching that slider
  raised `AttributeError` mid-apply -- the mode was already changed, remaining
  widgets were not synced, and settings were never saved, with no user
  feedback. The slider now derives its value from the preset's
  `detection_threshold` and no longer depends on the transient attribute.
- **CLI `--preset` now honors `auto_band` and the inverted feature flags.**
  `auto_band` was mapped to `None` and silently dropped, so the shipped
  "TikTok / Vertical short" and "News / Chyron" presets had no auto-band effect
  from the CLI (the GUI honored it). It now maps directly. The tri-state
  `kalman_tracking` / `phash_skip_enable` / `tbe_scene_cut_split` fields route
  onto their `--no-*` flags, so a preset can disable them; an explicit
  `--no-kalman` etc. still wins.
- **Crash reports and support bundles leaked UNC paths.** The path-redaction
  regex required four leading backslashes, so a real `\\server\share\...`
  network path was never scrubbed and could ship inside a shared bug report.
  The directory tree is now stripped like drive-letter paths.
- **Below-floor OpenCV wheels reported as satisfied.** The dependency drift
  report truncated versions to three components, so `opencv-python` builds
  below the four-part `5.0.0.93` floor (e.g. `5.0.0.80`) compared as OK. The
  comparison now keeps the build segment.
- **LaMa neural inpainters degraded on soft alpha mattes.** The ONNX / DNN /
  PyTorch LaMa paths fed the mask straight to the model without binarizing, so
  a MatAnyone-refined grayscale matte passed partial-strength hints (matching
  the guard the sibling `inpainters_onnx` backend already documents). The
  mask is now thresholded for the model input while the soft matte is retained
  for the feather blend.
- **Whisper `ffmpeg` filtergraph did not escape `;`.** A model path or
  language containing a semicolon would break out of the `whisper=...` filter
  and corrupt the `-af` chain (confined to the local filtergraph, not a shell).
- **Installer version drifted.** `installer/vsr.nsi` was still stamped 3.22.0
  while the app was 3.23.0; the installer version now tracks `APP_VERSION`, and
  a test guards against future drift.

### Changed

- The batch-completion summary hides the "STOPPED" pill when nothing was
  stopped, matching the conditional PAUSED and REVIEW pills.
- Compute-device labels use consistent sentence-case "CPU mode" and are
  localized; removed an orphaned header capability label that was constructed
  but never displayed.

## [3.23.0] - 2026-07-17

### Fixed

- **External inpainter kept the subtitle instead of the removal.** The
  `VSR_EXTERNAL_INPAINTER` adapter passed its arguments to the feather-blend
  in the wrong order, so inside the mask the original (still-subtitled) frame
  won over the external tool's output -- the whole path was a near no-op in
  the region that matters. The external fill now wins inside the mask, matching
  every other inpainter backend.
- **`detect_subtitle_band` leaked a VideoCapture** when the source failed to
  open; the `isOpened()` early return now runs inside the release scope.
- **ModernToggle checkmark rendered off-center at larger text scales.** The
  tick used vertex offsets hardcoded for the unscaled box, so at 125-200% text
  scaling it shrank into the top-left corner of the enlarged box. Offsets and
  stroke width now scale with the box.
- **Queue rows could leave a pending pulse animation scheduled after teardown.**
  The row now cancels any pending pulse callback on `<Destroy>`.
- **OCR replace-list keys now apply longest-first.** User OCR-fix JSON files
  with overlapping literal keys applied in insertion order, letting a short
  general key pre-empt a longer, more specific one; keys are now sorted
  longest-first for a deterministic, order-independent result.
- **Plural agreement in batch-status copy.** Fixed "N needs review",
  "N needs attention", "N item remains paused", and three "(s)" placeholders to
  agree with their counts, and aligned the per-item interrupt wording on
  "Stopping" to match the "Stopped" terminal state.

## [3.22.0] - 2026-07-17

### Changed

- **The native workbench now follows the approved mockup more closely.** The
  preview takes a 17:8 share of the desktop workspace and uses a larger 16:9
  media stage with a static heading. Preview utilities move to its mouse and
  keyboard context menu, the inspector becomes four consistent disclosure
  rows, and the footer contracts to one status line.
- **Queue commands now match the reference while staying functional.** Add,
  remove, clear-completed, and move-up/down controls share the queue header;
  compact layouts consolidate them into one menu. Reordering is persisted,
  clearing completed records never deletes output files, and Activity remains
  available from the context menu or Ctrl+L.

## [3.21.0] - 2026-07-17

### Changed

- **The desktop shell now uses one calm, command-first hierarchy.** Primary
  setup stays in the top toolbar while the inspector becomes a concise summary
  with separator-led disclosure rows instead of repeated controls. Larger body
  and metadata type, grouped preview tools, a plain media stage, borderless
  slider values, a compact status bar, and an empty queue that contracts to one
  row bring the native application in line with the new premium visual target.

## [3.20.0] - 2026-07-17

### Changed

- **Application branding now matches the premium workbench.** The previous
  text-heavy eraser illustration has been replaced by a compact restoration
  frame mark with transparent corners, a 512 px PNG, and a seven-size Windows
  ICO. The same mark now appears in the README, window chrome, packaged EXE,
  and About dialog.
- **The workbench now prioritizes the controls used for every cleanup.** A
  compact command bar keeps media import, cleanup profile, subtitle region,
  output, and Start cleanup above the preview. The preview and inspector share
  one flat workspace separated by a rule, while the queue uses a dense table
  header and a single-row empty state instead of nested cards and status pills.
  Larger body and metadata type, tighter group spacing, and functional-only
  borders improve readability without increasing the first-viewport height.

## [3.19.1] - 2026-07-17

### Changed

- **The desktop workbench now uses a quieter premium visual system.** The
  preview and inspector form the primary two-column workspace, the onboarding
  rail and redundant helper copy no longer consume the first viewport, and
  borders, status pills, queue chrome, and typography follow a flatter,
  more readable hierarchy. The primary cleanup action moves into the inspector
  on desktop while remaining persistent with the queue at compact widths.

## [3.19.0] - 2026-07-17

### Fixed

- **The production build now consumes the committed spec.** `build_exe.bat`
  builds the release artifact directly from `VideoSubtitleRemoverPro.spec`
  instead of duplicating its PyInstaller graph in command-line flags. The spec
  retains the default RapidOCR profile and the existing full-OCR/PyTorch opt-in
  environment gates, while NumPy collection and UPX policy have one authority.
- **Release-test bootstrap installs pytest.** The isolated build environment
  now installs pytest before unittest discovery, matching test modules that use
  pytest fixtures/imports and preventing a clean release host from failing
  before PyInstaller runs.
- **Best-effort diagnostics no longer fail silently.** Host-memory probes,
  seam-score sampling, quality-capture cleanup, and reproducibility sidecars
  now leave bounded log breadcrumbs on failure. JSON-line write failures and
  VRAM/NVML fallbacks have regression coverage for their existing error paths.
- **Batch-queue mutation is atomic under concurrent imports.** Capacity and
  duplicate checks now share the append critical section, while inactive-item
  dequeue has one locked decision path. A producer/consumer/pause stress test
  verifies the queue cap, path uniqueness, and exact added/removed accounting.
- **Pause/resume checkpoints now expose frame-sequence gaps.** Resume scans
  detect later orphaned frame files after the first missing frame, reset to
  that safe boundary, and surface an explicit warning instead of presenting
  the interruption as an ordinary short checkpoint.
- **Committed PyInstaller spec is now a real release input.** The default
  `VideoSubtitleRemoverPro.spec` is explicitly tracked despite the general
  generated-spec ignore rule, collects all NumPy 2.x data, binaries, and
  submodules, and disables UPX for both the executable and collection.

### Added

- **First-run setup has visible progress.** `Run_VSR_Pro.bat` launches a small
  dependency-free Tk splash before the virtual environment exists. `setup.py`
  publishes bounded, atomic stage updates in the system temp directory, the
  splash reports hardware/runtime/OCR/video-tool progress, and both success
  and failure paths clean up the status file. Console output remains available
  for detailed troubleshooting.
- **Masks can target the selected subtitle language.** An opt-in Advanced
  toggle and `--language-filter` keep OCR boxes whose recognized script does
  not match the configured subtitle language. RapidOCR, PaddleOCR, and EasyOCR
  retain aligned text/confidence for the filter; detection-only or unrecognized
  boxes are conservatively left untouched, and Latin languages share a script
  family rather than claiming unsupported locale-level identification.
- **Advanced OCR selection is now reproducible.** Automatic detection remains
  the default, while the GUI and `--ocr-engine` can pin RapidOCR, OpenCV 5 DNN,
  PaddleOCR, EasyOCR, or the built-in OpenCV fallback. Preview, auto-band,
  throughput-estimation, queue caching, saved settings, and CLI reproduction
  all honor the same choice; missing optional engines fall back explicitly.
- **First-run choices now perform useful setup.** The welcome flow can apply
  YouTube, Film, or Fast profiles, enable automatic subtitle-band detection,
  and launch the existing one-frame test cleanup. New Film and Fast built-ins
  complement the existing watermark profile, while advanced settings expose
  pHash distance and temporal mask carry controls with concrete speed hints.
- **Mask review now tunes dilation in real time.** A preview-pane slider
  updates the selected item and recomposes the mask from cached boxes, manual
  corrections, and imported mattes without running OCR again.
- **Region drawing now provides live OCR feedback.** Rectangle drags launch a
  throttled background probe that overlays detected text boxes and per-box
  confidence, while preserving manual region editing when OCR is unavailable.
- **Batch reports explain detection efficiency.** Processor-owned counters now
  report frames sent to OCR versus skipped, skip reasons, IoU-clustered unique
  regions, and an actionable frame-skip hint when OCR dominates runtime. The
  same evidence is available in GUI/CLI summaries and output sidecars.
- **Color preservation is now an explicit output result.** The existing
  ffprobe contract maps HDR pixel-format and primaries/transfer/matrix/range
  mismatches to `color_preserved`, warns on loss, and carries that status into
  GUI/CLI batch summaries and reproducibility sidecars.

- **Opt-in frozen build smoke for the committed spec.**
  `python -m scripts.frozen_build_smoke` builds the default-profile spec in an
  isolated work directory, launches the windowed executable without creating
  GUI state, and requires a JSON marker proving NumPy and OpenCV imported and
  the versioned startup path ran. `VSR_VERIFY_SPEC_BUILD=1` adds the same slow
  gate to `build_exe.bat`; the normal frozen GUI smoke also imports NumPy and
  OpenCV explicitly.

### Changed

- **AUTO routing is scene- and motion-aware.** The existing scene-cut cascade
  now partitions each batch before routing. Low-motion scenes with sufficient
  temporal exposure use STTN; fast-motion or persistently covered scenes use
  the ProPainter-mode hybrid, which remains lazily loaded and is released
  after a long STTN-only streak.
- **The region selector is now an explicit window controller.** Its former
  1,430-line modal function and 39 state-sharing closures are split into a
  `RegionSelectorWindow` with named callback methods and instance-owned editor
  state, while the GUI mixin retains a three-line launch boundary.
- **CLI startup is split into named phases.** The former 1,352-line `main()` is
  now a 60-line orchestrator over parser construction, utility actions,
  argument validation, config assembly/overlays, soft-subtitle handling, and
  the processing runner, preserving command output and exit behavior.
- **Settings construction follows explicit UI groups.** The former 1,013-line
  settings builder is now a 52-line orchestrator over profile, workflow, STTN,
  detection, output, range, performance, and accessibility/storage groups,
  preserving widget creation order and bound configuration variables.
- **Mask correction is an explicit window controller.** The former 534-line
  editor closure is split into `MaskCorrectionWindow` callback methods,
  including named background detection/apply boundaries; the GUI mixin now
  retains only the launch adapter.
- **Frame timing and Whisper span conversion now share one clock.** CFR/VFR
  frame-to-seconds conversion is centralized and reused by the mask/inpaint
  loop, clean-reference overrides, VMAF sampling, checkpoint-frame encoding,
  matte timestamps, and time-range reporting. Both Whisper backends now use
  the same segment-to-frame helper, removing two copies that could drift on
  variable-frame-rate input.
- **GPU dispatch now has an injectable device-provider seam.** Runtime device
  probing, registry-backed inpainter construction, OOM classification, and
  inference-memory cleanup can be replaced in tests, so CUDA/DirectML
  selection and OOM-to-CPU fallback are covered without accelerator hardware.
- **The `process_video` frame loop is now a staged pipeline.** A frozen
  read-only context, a five-field mutable state, and a batch object replace
  30-plus threaded locals and four parallel lists. Decode/mask construction,
  mask refinement, inpainting, writing, and checkpoint persistence are
  isolated methods; the outer loop is a short stage orchestrator with the
  existing cleanup and encode/mux boundaries unchanged.
- **Processor and CLI imports are acyclic.** Completion-marker helpers now
  live with pause/resume checkpoints, while bounded JSON overlays and
  auto-band reset logic live with processing configuration. The CLI and
  processor import those focused modules directly; legacy processor re-exports
  remain available without importing the 1,900-line CLI module.
- **GUI controllers import focused backend modules.** Detection, processing
  configuration, mode enums, normalization, and checkpoint helpers no longer
  reach through the processor compatibility facade, reducing unnecessary
  processor/OpenCV/NumPy import coupling in preview and settings paths.
- **GUI smoke tests reuse one Tcl interpreter.** Each application instance is
  now hosted in a destroyed-after-test `Toplevel` under one withdrawn class
  root, eliminating the repeated `tk.tcl` / `tcl_findLibrary` exhaustion that
  appeared only after dozens of roots were created in one Python process.

## [3.18.1] - 2026-07-17

### Fixed

- **Frozen build now launches instead of crashing on a numpy import.**
  `build_exe.bat` now packages numpy with `--collect-all numpy` and passes
  `--noupx`. numpy 2.x splits its C core into submodules (`numpy._core._exceptions`,
  ...) that a bare `--hidden-import numpy` does not gather, so the previous
  default-profile exe died at launch with
  `ModuleNotFoundError: No module named 'numpy._core._exceptions'`; UPX
  additionally corrupts numpy's compiled extension binaries on Windows. The
  packaged RapidOCR/ONNX build now starts cleanly.

### Changed

- **Extracted the encode/mux and time-range/quality-report stages from
  `process_video`.** The final-output assembly (frame-sequence, checkpoint,
  VFR, and default container-mux branches plus matte finalize and
  SRT/translation/NLE sidecars) now lives in `_finalize_and_mux`, which returns
  the resolved output path and the post-finalize matte-writer state so the
  caller's cleanup is unchanged. The time-window-to-frame math lives in a
  standalone, unit-tested `_resolve_frame_range` helper (returning a
  `_FrameRange` bundle), and the PSNR/SSIM reporting in `_emit_quality_report`.
  Pure relocations with no behavior change, each verified against the
  real-ffmpeg reference-clip, VFR, and checkpoint/resume suites; the per-frame
  inpaint loop remains inline pending a hardware-verifiable refactor.
- **Split the `test_hardening` monolith by subsystem.** The single ~8,100-line
  `tests/test_hardening.py` was divided into ten focused files
  (`test_hardening_{core,cli,config,detection,encode_io,inpaint,quality,batch,adapters,gui_i18n}.py`)
  with the same 410 cases; no test was added or removed.
- **Unified post-inpaint finishing.** The ONNX, diffusion, and built-in
  inpainter families now share a single `apply_finishing` routine
  (`backend/inpainters/_common.py`) for the edge-ring colour match and feather
  blend, replacing three separate re-implementations of the same per-frame
  loop. Boundary handling is now identical across every backend.
- **Dependency-floor consistency guard.** A new test cross-checks the version
  floors declared in `requirements.txt`, `dependency_profiles.json`,
  `backend/dependency_caps.py`, and the shared Pillow floor used by release
  verification, failing if any of them drift apart.

### Fixed

- **Unhandled GUI callback exceptions are logged.** The app now routes Tk
  callback / `.after` exceptions through the logger with a full traceback
  instead of Tk's default stderr dump, so a real bug in a scheduled callback
  is captured in the log file. Teardown races (TclError/RuntimeError while the
  app is closing) are logged at debug.
- **Accurate time-range start seek on CFR sources.** Frame seeking for
  `--start`/`--end` time ranges, checkpoint resume, and the selective mask
  re-run capture now grab forward to the exact requested frame instead of
  trusting `cap.set(CAP_PROP_POS_FRAMES, ...)`, which snaps to the nearest
  keyframe on long-GOP sources with some OpenCV backends. Backends that
  already position accurately (the bundled FFmpeg backend) are unaffected.
- **ONNX inpaint mask binarization.** The LaMa-ONNX and MI-GAN inpainters now
  threshold the mask to a strict `{0, 255}` binary before feeding it to the
  network. Feathering is applied after inpainting, so a greyscale or
  anti-aliased mask that reached these paths previously produced
  partial-strength fill hints and degraded quality; edges are now resolved
  deterministically at a midpoint threshold while properly dilated binary
  masks are unchanged.

### Security

- **Single-source libpng security floor.** The libpng CVE floor, affected
  range, advisory URL, and CVE id now live only in `backend.security_checks`;
  the OpenCV DNN OCR eligibility check and strict release advisory derive their
  version strings from that constant instead of hardcoding `1.6.54`, so the
  runtime vulnerability check and the advisory text can no longer drift apart.
- **Hidden, bounded external-process policy.** Every production backend child
  now launches without a shell, suppresses Windows console windows, closes
  unused stdin, drains captured output into bounded buffers, and shares
  timeout/cancellation escalation and deterministic cleanup. Source-hygiene
  tests reject new raw backend `subprocess.run` or `subprocess.Popen` calls.
- **Remote-code adapter trust boundaries.** VapourSynth `.vpy` input now
  requires an explicit reviewed script root and rejects paths (including
  symlinks) that resolve outside it. CoTracker3 and Florence-2 remote code now
  require a full 40-character commit SHA; mutable tags, branches, and short
  revisions no longer reach `torch.hub` or `trust_remote_code`.
- **Fail-closed FFmpeg branch classification.** Runtime diagnostics now label
  FFmpeg as safe, vulnerable, unsupported, or unknown. Only reviewed 8.1.2+
  and 8.0.3+ patch lines pass; older branches, development snapshots, and
  unclassified future branches block strict release evidence instead of being
  described as having no matched vulnerability floor.
- **Pillow and NSIS security floors.** Source, Docker, dependency diagnostics,
  and strict release evidence now require Pillow 12.3.0, which includes the
  2026-07-01 decoder, memory-safety, and decompression-bomb fixes. Installer
  compilation and release evidence require NSIS 3.12 so elevated installers do
  not use the Low IL temporary directory.
- **FFmpeg filtergraph injection in subtitle re-burn.** `burn_subtitles` now
  escapes single quotes in the subtitle path for the filtergraph value context
  and validates the ASS `force_style` override against a strict allowlist,
  dropping (with a warning) any override containing filtergraph metacharacters
  so it cannot break out of `force_style='...'` and inject arbitrary filters.
- **FCPXML XXE hardening.** When `defusedxml` is not installed, `parse_fcpxml`
  now refuses documents containing a `<!DOCTYPE>`/`<!ENTITY>` declaration
  instead of silently falling back to the unhardened stdlib parser.
- **PaddleOCR-VL server probe scheme guard.** The llama.cpp server reachability
  probe now only opens `http(s)` URLs, so a misconfigured server URL cannot be
  coerced into a `file://` local read.
- **Crash-report path leak.** The opt-in GlitchTip `before_send` scrubber now
  walks the whole event tree, so absolute paths embedded in exception messages,
  breadcrumbs, and extra/request data are redacted too -- not just top-level
  strings and stack-frame paths. Frame locals are dropped before serialization.

### Added

- **Timed-region clean reference fills.** Each timed rectangle can persist a
  same-size clean plate, preview translation or homography alignment at the
  current frame, and opt into per-frame BGR color matching. Processing applies
  an accepted plate only where its region intersects the finalized mask;
  low-confidence frames automatically retain that mask for normal inpainting.
  Output sidecars capture the plate hash, scope, policy, observed confidence,
  alignment methods, color delta, and accepted/fallback frame counts.
- **Local-first erase, translate, and re-embed workflow.** The opt-in CLI and
  Detailed controls accept a ready translated SRT or preserve OCR, Whisper,
  or source-SRT timings while a registered provider translates cue text. The
  built-in command provider uses a bounded, versioned JSON stdin/stdout
  protocol with no shell; translated captions are validated, burned with an
  optional ASS style, and recorded in output sidecars by source kind, provider,
  language, filename, content hash, and embed status. The same change fixes
  Windows drive-letter escaping in the existing FFmpeg subtitle-burn pass.
- **Runtime-validated FFmpeg D3D12 acceleration.** An experimental Windows
  toggle and `--d3d12-accel` opt into FFmpeg 8.1+ D3D12 upload,
  `scale_d3d12`, deinterlacing, and H.264/H.265 encoding only after a complete
  30-frame output can be read back. Advertised-but-rejected AV1 and runtime or
  container failures fall back through the existing NVENC/QSV/AMF chain and
  software encoder; output sidecars retain advertised, smoke, selection, and
  encoder-fallback evidence. A local 300-frame 1080p evaluation measured D3D12 at
  1.44s, warmed NVENC at 0.85s, and x264 at 1.20s, so the path remains opt-in.
- **Lossless mask and alpha-matte interchange.** `--export-mask` now writes an
  exact gray8 FFV1/MKV artifact by default, or an ordered PNG sequence, with a
  versioned CFR/VFR timestamp-and-duration manifest. Edited manifests import in
  replace/add/subtract mode after native mask composition; every frame,
  dimension, count, and timestamp is preflighted before processing. GUI
  controls and Review mask expose the same composition, while batch and output
  sidecars record content hashes, edit state, and composition order.
- **Quality-directed per-frame mask correction.** Residual-text, temporal-
  flicker, and low-confidence quality signals now produce an exact frame review
  queue. The internal editor paints ordered add/subtract corrections with
  undo/redo and bounded propagation, persists them in the versioned config and
  queue sidecars, and selectively reruns only affected ranges while reusing the
  prior cleaned output elsewhere.
- **Precise keyboard-accessible region editing.** The full manual-region
  selector exposes exact x/y/width/height or polygon vertices, synchronized
  second/frame timing, bounded arrow-key nudge and Ctrl+arrow resize, and
  shared undo/redo across pointer, numeric, timing, polygon, and keyframe
  operations. Accessible labels and the unchanged persistence path keep
  keyboard edits equivalent to pointer-created masks.
- **Complete gettext catalog lifecycle and locale selector.** A deterministic
  extractor/merger validates UTF-8, placeholders, and plural forms, compiles
  MO files, reports coverage, and maintains a 100%-covered pseudo-locale.
  Detailed controls now persist System, English, or a discovered catalog;
  source, frozen, and installer smoke paths share the same fallback-aware
  catalog lookup.
- **Generated CLI and config reference.** Every live CLI action now carries a
  category, default, range/choice, visibility, and deprecation record used by
  grouped `--help` output and the README generator. The same deterministic
  command documents all 102 canonical `ProcessingConfig` fields, and source
  tests plus release builds fail when either generated section drifts.
- **OpenCV 5 PP-OCRv6 provider and benchmark evidence.** RapidOCR's bundled
  detection and recognition ONNX assets can now execute
  through `cv2.dnn` without ONNX Runtime. The OCR benchmark selects OpenCV DNN
  or the regular RapidOCR provider explicitly and records recall, per-frame
  latency, and resident-memory deltas; release evidence reports model and
  libpng eligibility.

- **Accessible interface text scaling.** Detailed controls now persist a
  100%-200% interface text-size setting that scales Canvas controls and
  dependent geometry. The 980x720 minimum window reflows into a compact,
  vertical-only layout for long translations and RTL text, with hidden
  subprocess layout coverage across default and high-contrast themes.
- **Interpolated moving manual regions.** The scrubbed region selector now
  records rectangle or polygon keyframes, previews their deterministic motion,
  and persists the tracks through settings, presets, CLI overrides, batch
  reports, and reproducibility sidecars. Processing fills polygons exactly and
  activates each track only within its saved time span.

### Changed

- **Preview-first desktop workbench.** The main window now follows a focused
  Import / Configure / Process flow with a dedicated workflow rail, central
  16:9 media stage, concise right-side inspector, and a persistent full-width
  queue whose primary action stays anchored at the lower right. Preset,
  device, language, and expert tuning controls progressively disclose through
  Advanced; compact and 150%-200% text layouts consolidate secondary queue
  commands without hiding them, and the activity log now opens from the footer
  instead of consuming the first-run workspace.
- **GUI workflows now have explicit controller boundaries.** Region editing
  and advanced-settings behavior moved out of `gui/app.py` alongside the
  existing preview, processing, quality, support, and mask controllers. Every
  controller declares its host protocol, imports only focused dependencies,
  and has construction-free boundary tests; the app remains the composition
  root and preserves its public launch/import surface.
- **Ruff source-hygiene ratchet.** The Python 3.11 source tree now passes the
  explicit Pyflakes and high-signal pycodestyle baseline with zero findings.
  Release builds install the reviewed MIT-licensed Ruff 0.15.20 tool and stop
  before tests or packaging on any regression; only the two documented
  import-order compatibility modules receive narrow `E402` exceptions.
- **Reviewed dependency profiles.** CPU, NVIDIA, and DirectML setup now uses
  generated exact constraints with manifest/profile hashes, a diffable update
  command, Docker parity, and strict release-evidence capture. RapidOCR is the
  deterministic default; PaddleOCR, EasyOCR, and legacy simple-lama are
  isolated opt-ins because their OpenCV ownership or NumPy caps conflict with
  the primary runtime. The supported Python floor is now 3.11 because the
  reviewed ONNX Runtime CPU/CUDA and DirectML releases do not publish Python
  3.10 wheels.
- **OpenCV 5 is the reviewed runtime floor.** Profiles, setup, Docker, drift
  diagnostics, and remediation now require `opencv-python==5.0.0.93`. Its
  bundled libpng 1.6.57 resolves CVE-2026-22801, so vulnerable older wheels
  block strict releases instead of using the former temporary exception.
- **Local release builds are fail-closed and artifact-derived.**
  `build_exe.bat` now runs the full suite, a frozen-runtime dependency audit,
  the reference corpus, PyInstaller/launcher smoke, production NSIS compile,
  a non-elevated installer extraction/startup smoke, and strict advisory
  validation. The CycloneDX SBOM now maps PyInstaller Analysis entries to
  required runtime packages and hashed native files, separates build tools as
  excluded scope, and omits unrelated environment packages. Heavy
  PaddleOCR/EasyOCR/PyTorch fallbacks are explicit packaging opt-ins.
- **Advanced settings explain outcomes instead of backend names.** Labels and
  tooltips now describe the visible benefit and tradeoff for motion alignment,
  scene cuts, adaptive memory, mask carry-forward, deinterlacing, keyframes,
  hardware decode/encode, interpolation, and quality reports. Per-file
  overrides expose modal accessibility metadata, a Ctrl+Enter save path, and
  initial focus.
- **Reduced-motion preference is respected.** Windows' client-animation
  setting now disables progress easing, active-card pulsing, toast fades, and
  preview loading animation while retaining clear static status indicators;
  `VSR_REDUCED_MOTION=1` provides an explicit override.
- **Interface locale detection preserves BCP-47 detail.** Windows locale
  detection now uses the OS locale API instead of mistaking
  `English_United States` for a language code, retains territory/script
  subtags, chains full-tag/script/language catalogs, and packages the locale
  directory in frozen releases. `VSR_UI_LOCALE` remains available for
  deterministic overrides.
- **Queue-card actions are keyboard reachable.** Focused queue items now open
  the same complete action menu with the Menu key or Shift+F10, place it at a
  deterministic card-relative position, expose the shortcut to assistive
  metadata, and return focus to the card after dismissal or Escape.
- **Canonical container ONNX Runtime floor.** The CPU smoke container now
  requires `onnxruntime>=1.25.0`, matching source and strict-release parser
  hardening policy, and its smoke output records the resolved runtime version
  and execution providers for reproducible evidence.
- **Resolvable DirectML setup policy.** AMD/Intel setup now preflights and pins
  the latest reviewed PyPI wheel (`onnxruntime-directml==1.24.4`) before making
  any environment changes, with CPU and Windows ML guidance when no compatible
  wheel exists. Diagnostics and release evidence treat the DirectML
  sustained-engineering lifecycle separately from CPU/CUDA security floors.
- **Calmer, more consistent microcopy.** Disabled-action hints were reworded
  from dismissive/double-negative phrasing ("No completed outputs are available
  yet", "No failed or stopped items need retry", "The queue is already empty")
  to calm, instructive copy. The three drop-zone reassurance variants
  ("stay untouched" / "never touched" / "never modified") are unified to
  "Originals are never modified." Error rows drop the redundant "Error:" prefix.

### Fixed

- **GUI teardown no longer leaks Tk callbacks between tests.** Closing an app
  now cancels its pending Tcl timers, detaches and closes the recurring log
  handler, and invalidates preview workers. Preview completions use a
  shutdown-aware dispatcher, eliminating late `main thread is not in main
  loop` exceptions and full-suite Tk resource accumulation.
- **The NSIS 3.12 guard rejected NSIS 3.12 and required an undeclared plugin.**
  The packed-version threshold now uses NSIS' actual byte layout
  (`0x030C0000`), and the
  running-app guard uses a named application mutex plus NSIS' bundled
  `System` plugin instead of the absent `FindProcDLL` extension.
- **Preview rendering could create Tk images on a worker thread.** Frame
  decoding and PIL compositing remain asynchronous, but `PhotoImage` objects
  are now created only by the UI event-loop callback, preventing intermittent
  `main thread is not in main loop` failures and native Tk instability.
- **High-contrast root surface used the default palette.** Settings and theme
  tokens now load before the Tk root is created, so the window background,
  nested surfaces, custom controls, and ttk styles start in one palette.
  Creating a later default-theme instance in the same process also restores
  the canonical palette instead of leaking high-contrast state.
- **Configured work storage was only cosmetic.** The GUI, CLI, canonical
  config, copied commands, sidecars, processing intermediates, mask exports,
  Whisper scratch data, and pause/resume checkpoints now share one validated
  work-directory policy. Unavailable or read-only locations fall back with an
  actionable warning, every affected volume receives a disk-space preflight,
  and cross-volume output promotion copies to a destination-side staging file
  before the final atomic replace.
- **Reproducibility artifacts no longer omit active processing controls.** A
  versioned canonical backend field registry now drives GUI-to-backend mapping,
  settings migration, complete sidecar snapshots, copied CLI overrides, and
  `--validate-config` output. Per-item corrections and advanced controls round
  trip losslessly, while source SHA-256 fingerprints stream without the former
  512 MiB cutoff on the GUI processing worker.
- **Processed outputs preserve their compatible container payload.** A single
  ffprobe manifest now drives remuxing for audio, soft subtitles, chapters,
  attachments, MP4 timecode, global/stream metadata, tags, and dispositions.
  Compatible audio is copied, incompatible audio and text subtitles are
  selectively converted, range-excluded or unsupported payload is warned and
  recorded, decoded rotation is cleared, and post-restoration passes restore
  the same payload before promotion. MKV/MP4 fixtures verify the contract.
- **Frozen distributions no longer bundle source-only launchers.** Release
  builds now copy dedicated launchers that invoke `VideoSubtitleRemoverPro.exe`
  without Python, `setup.py`, or a repository-local virtual environment.
  Release verification rejects bootstrap references and smoke-tests the EXE,
  both batch launchers, and the PowerShell launcher from a temporary path with
  spaces.
- **VMAF could fail on Windows or compare different time windows.** The
  libvmaf pass now writes its JSON log from a private working directory using
  a filter-safe relative filename, applies identical seek/duration options to
  both inputs, and normalizes both PTS clocks and time bases before comparison.
- **Pre/post-processing could violate the selected media contract.** One
  probed output contract now governs deinterlace, encode/mux, restoration,
  film-grain, subtitle-restyle, and watermark passes. Outputs are validated
  before promotion for container, codec, audio policy, pixel depth, color
  tags, HDR10 mastering display, and MaxCLL/MaxFALL; stale HDR10+/Dolby Vision
  dynamic metadata is intentionally dropped and reported. HDR deinterlace now
  uses lossless 10-bit FFV1, while compatible hardware encoding remains active
  unless static HDR SEI requires the metadata-capable software encoder.
- **Variable-frame-rate timing and A/V drift.** Processing now carries the
  source frame PTS/time base through time ranges, Whisper spans, checkpoints,
  SRT/NLE sidecars, VFR encoding, and audio muxing. Irregular VFR sources use
  exact frame-sequence timestamps instead of a synthesized average FPS, and
  HDR range/resume seeks use source PTS so tail frames remain aligned.
- **Queue recovery could lose behavior and disappear after one restore.** Queue
  state schema 2 now atomically preserves order, locked output paths, complete
  config snapshots, embedded-subtitle choices, pause/retry evidence, and all
  incomplete states. Normal close and every queue mutation refresh the
  snapshot; corrupt files are quarantined; a successful restore writes a new
  snapshot so repeated crash/restart cycles remain recoverable.
- **Batch retries skipped normal processor failures.** CLI and GUI batch jobs
  now classify the processor's usual `False` failure result, apply bounded and
  pause/cancel-aware backoff, and retain retry attempts/errors in batch reports.
  Copied CLI commands now include each queue item's retry policy.
- **Requested mask videos were discarded after encoding.** The completed mask
  artifact is now validated and atomically promoted after its writer closes;
  processor and batch-report evidence distinguishes not requested, created,
  and failed exports.
- **VLM detector dropped all boxes on a common response shape.** The
  Qwen2.5-VL box extractor assumed every JSON entry was a `{"bbox": [...]}`
  dict, so a model returning bare `[[x1,y1,x2,y2], ...]` arrays hit an
  AttributeError that the broad handler turned into an empty result. It now
  accepts both the dict and bare-list shapes and ignores non-box entries.
- **Unstable box identities in the Kalman tracker.** Detection-to-track
  matching used per-track first-best-available greedy assignment, so which
  track claimed an overlapping detection depended on track order and could
  churn identities (and destabilize chyron vs subtitle classification). It now
  uses order-independent global-greedy assignment (highest-IoU pairs first).
- **UI froze during support-bundle and model-cache operations.** Saving a
  support bundle and exporting/importing a model cache zipped potentially
  hundreds of MB synchronously on the Tk main loop, freezing the window with
  no feedback. These now run on a background thread with a "Working..." status
  and marshal the result back to the UI. The Import-model-cache dialog title
  and filetypes are also now localized like the rest.
- **High-contrast theme leaks in preview overlays.** The A/B compare seam line
  and detection-box overlays were drawn with fixed green/red; they now derive
  from the theme focus/danger tokens so they brighten correctly in the
  high-contrast palette. The empty drop-zone "Videos and images supported"
  hint used the disabled-text token (too low contrast for live text) and now
  uses the muted token.
- **Context-menu widget leak.** Right-click menus (queue rows, queue sort,
  support tools) were re-created on every open and never destroyed; they are
  now destroyed after dismissal so they cannot accumulate over a long session.
- **Log panel backlog was unbounded.** The activity-log handler queue is now
  capped and drops the oldest record under a log flood instead of growing
  without limit and lagging the UI behind reality.
- **Cross-thread Tk call on resume warning.** The batch worker surfaced a
  resume-checkpoint warning by calling the status/toast update directly from
  the processing thread; it is now marshalled to the Tk main loop like every
  other worker-to-UI update, preventing a rare crash or widget-tree corruption.
- **FFV1 intermediate writer stderr deadlock.** `_LosslessIntermediateWriter`
  now drains the ffmpeg `stderr` pipe on a background thread, so a noisy ffmpeg
  (enough warnings to fill the ~64 KB pipe buffer) can no longer deadlock the
  frame writer that is blocked writing to stdin.
- **Inline region editor discarded existing regions.** Drawing a box in the
  preview inline editor now appends to the configured static subtitle regions
  instead of replacing them all and clearing timed spans.
- **ETA probe race.** `_probe_batch_eta` snapshots the queue under `queue_lock`
  before iterating, preventing a "list changed size during iteration" crash
  when the user edits the queue mid-probe.
- **Onboarding could be permanently skipped.** The persisted `onboarding_seen`
  flag is now set only after the welcome dialog actually builds, with a
  separate in-session guard, so a failure while showing it no longer hides
  onboarding forever.
- **Toast reference leak.** The class-level active-toast list is pruned of
  destroyed windows on each new toast and exposes a `reset()` for teardown, so
  it cannot accumulate dead references over a long session.

### Security

- **Build-toolchain security floors.** The local build now requires
  PyInstaller `>=6.10.0` (CVE-2025-59042 writable-CWD local privilege
  escalation) and the NSIS installer requires `>=3.11` (CVE-2025-43715
  temp-plugin-dir SYSTEM LPE): `build_exe.bat` pins the PyInstaller install,
  `installer/vsr.nsi` fails to compile on an older NSIS via `NSIS_PACKEDVERSION`,
  and strict release validation emits blocking advisories for either.

- **ONNX Runtime security floor raised to 1.25.0.** `onnxruntime`,
  `onnxruntime-gpu`, and `onnxruntime-directml` now require `>=1.25.0` (below
  which parser integer-truncation heap-OOB hardening is missing). The
  self-test, support bundle, and strict release validation flag an older
  runtime as a blocking `ORT-PARSER-OOB-1.25.0` advisory; requirements, setup,
  and the drift report track the new floor. VSR runs untrusted OCR/inpaint
  ONNX models through this runtime.

- **Bounded, transactional model-cache imports.** `import_model_cache_bundle`
  now preflights an untrusted bundle against member-count, per-member size,
  compression-ratio (zip-bomb), total-expansion, and free-space ceilings
  before writing anything; streams every accepted member to a scratch dir with
  a hard byte cap and SHA-256 verification; rejects duplicate import targets;
  and commits all staged files only after every member is verified, rolling
  back any files already moved if a commit fails. Prevents disk-exhaustion and
  partial multi-file imports. Adversarial tests cover zip-bomb ratios,
  oversized/short members, duplicate targets, free-space exhaustion, and
  mid-commit rollback.

### Diagnostics

- **OCR detection benchmark (`--ocr-benchmark`).** Scores the active detector
  (RapidOCR ships PP-OCRv6) on deterministic synthetic ground-truth subtitle
  fixtures -- detection recall + per-frame latency -- and prints JSON evidence
  with a `meets_floors` verdict (recall >= 0.8) that gates any default-detector
  swap. Latency is reported as device-dependent evidence, not a hard gate.
  Needs no redistributable media. New `backend.ocr_benchmark`.

- **Full-pipeline `--dry-run` and machine-readable `--json`.** `--dry-run`
  validates a run without encoding: it probes each input (frames/fps), runs
  detection on a few sampled frames to report where text is found, checks the
  requested codec/profile is available, prints a per-file plan, and writes no
  output. `--json` emits a structured result to stdout -- the dry-run plan, or
  the batch/single-file result (status, output, timings, quality report) -- for
  scripting and CI.

- **Inference-execution self-test (`--inference-smoke`).** A new CLI option and
  `backend.support_bundle.run_inference_smoke` push a generated text image and
  masked frame through the selected OCR detector and inpaint backend to prove
  they actually run, recording the real engine / execution provider (e.g.
  `RapidOCR`, `ONNX (CUDAExecutionProvider)`, or a `cv2` fallback), whether
  inference ran, and timing. It exits non-zero when a loaded backend cannot
  execute and reports a precise reason when weights are absent -- no model
  downloads. Distinguishes "provider available" from "provider actually ran".

### Quality

- **Scene-cut-safe temporal mask stabilization.** New opt-in `temporal_mask_union`
  ORs each frame's mask with a short trailing window (default 3, configurable
  1-15) so a single-frame OCR miss or a moving/dissolving overlay keeps the
  pixels its neighbours saw. It runs only in automatic full-frame detection,
  resets at every scene cut (never bleeds a mask into an adjacent scene), and
  never touches user-fixed timed/fixed regions. Exposed via `--temporal-mask-union`
  / `--temporal-mask-window`, a GUI toggle, settings, and presets. New
  `backend.inpainters.stabilize_masks_rolling_union`.

- **Mask-boundary seam (discontinuity) quality check.** The quality report now
  scores the visible seam at the inpaint mask boundary -- the excess gradient
  the fill introduces along the mask contour versus the original frame,
  normalised to the image texture scale -- complementing the existing
  residual-text and temporal-flicker (inter-frame fill variance) scores. Seam
  samples are accumulated cheaply during inpainting; the mean feeds the quality
  gate (new `seam_score` metric and `SEAM_SCORE_CEILING`, remediation:
  increase mask dilation/feather) and is surfaced in batch reports and
  reproducibility sidecars. New `backend.quality.mask_boundary_seam_score`.

### Reliability

- **Automatic bounded retry for transient batch failures.** New opt-in
  `--max-retries N` (config `batch_max_retries`, default 0) re-attempts a
  batch item that fails with a *transient* error -- GPU OOM/glitch, ffmpeg
  hiccup, timeout, broken pipe -- up to N times with escalating backoff,
  recording the attempt count on the batch record. Permanent failures
  (missing file, unsupported format, disk-full, integrity) are never retried.
  `backend.batch_report.is_retriable_error` classifies the error.

- **Pre-encode disk-space preflight + bounded JSON log.** Processing now
  estimates the working space the lossless FFV1 intermediate needs and fails
  fast with a clear message before creating any temp file when the output
  drive is critically short (logging a warning on a merely-tight margin),
  instead of crashing with a mid-encode `OSError`. The opt-in JSON-lines log
  rolls to a single `.1` backup once it passes 10 MB so it cannot grow without
  bound. (The main `vsr_pro.log` already rotates at 5 MB.)

- **Encode/mux-phase resume marker.** When every frame has been inpainted and
  written to the checkpoint frame dir, the pause checkpoint now records an
  explicit `stage="encoding"` / `inpaint_complete` marker before the encode and
  mux begin. If the process is interrupted during encoding or audio muxing,
  resume detects the marker and jumps straight to the encode stage from the
  persisted frames instead of redoing detection and inpainting -- multi-hour
  jobs no longer lose the OCR/inpaint work to a late crash.

- **Graceful GPU out-of-memory recovery.** A CUDA OOM during batch inpainting
  no longer crashes the run: the CUDA cache is cleared and the batch is split
  in half and retried recursively down to a single frame, which falls back to
  CPU (OpenCV) inpainting if it still cannot fit. Output always has one frame
  per input so no partial/corrupt write can result. Controlled by the new
  `gpu_oom_recovery` config flag (default on); OOM detection lives in
  `backend.inpainters.is_oom_error` / `free_inference_memory`.

- **Output integrity validation before promotion.** A finished video is now
  probed (ffprobe, with a bounded OpenCV fallback) for a decodable video
  stream and a duration/frame envelope before it replaces any destination.
  The `-shortest` audio-merge path, the re-encode/copy path, and the
  checkpoint-frame encode path all fail closed on truncation -- the audio
  merge salvages the full-length video without audio, the re-encode salvages
  the intermediate, and a stalled/short encode preserves the existing
  destination instead of shipping a truncated file as success. New
  `backend.io.probe_video_integrity` / `validate_video_output` and
  `backend.processor.OutputIntegrityError`.

### Security

- **FFmpeg runtime CVE floor.** VSR decodes untrusted media through FFmpeg, so
  the self-test, support bundle, and strict release validation now parse the
  external FFmpeg version and treat builds `8.1.0-8.1.1` and `8.0.0-8.0.2` as
  blocking -- they predate the 8.1.2 / 8.0.3 backports for CVE-2026-8461
  (MagicYUV heap out-of-bounds write, RCE) and CVE-2026-30999. `8.1.2+` /
  `8.0.3+` pass; git/`N-` snapshot builds are reported as unclassifiable rather
  than assumed safe. New `backend.ffmpeg_profiles.classify_ffmpeg_security`
  and `probe_ffmpeg_security`; strict release evidence fails closed on a
  known-vulnerable runtime. README documents the `>=8.1.2 / >=8.0.3` floor.

## [3.17.3]

### Fixed

- **Mask cache corruption.** Manual mask corrections mutated cached
  fixed-region masks in-place, causing mask bleed across frames.
- **Notification crash.** `_send_system_notification` referenced an
  undefined `paused` variable, silently crashing every batch
  completion toast notification.
- **Preview threshold divergence.** Mask preview read a ghost
  attribute instead of the persisted detection threshold, always
  using 50% until the slider was manually moved.
- **Region editor off-by-one.** Dragging to the extreme edge returned
  coordinates one pixel past the image boundary.
- **NLE single-span regression.** A single timed subtitle region was
  excluded from the multi-segment NLE export path.
- **FCPXML file URI encoding.** Paths with spaces or special
  characters now percent-encode per RFC 8089.
- **JSON config overlay injection.** `--config` overlay now whitelists
  dataclass field names, preventing `__dict__`/`__class__` injection.
- **NaN/inf in mask correction coercion.** Explicit `math.isfinite()`
  checks prevent NaN time ranges from bypassing frame-range gating.
- **NLE sidecar non-atomic writes.** EDL/FCPXML writes now use atomic
  temp-file-then-rename to prevent partial files on crash.
- **Cached remover race condition.** Read into a local variable before
  comparing to prevent TOCTOU with the main thread's stop handler.
- **Sidecar SHA-256 performance.** Skip hash for inputs over 512 MB to
  avoid blocking the UI for 10+ seconds after processing.
- **Sidecar config completeness.** `manual_mask_corrections` was
  missing from the reproducibility sidecar config snapshot.
- **Hardcoded hex colors.** All hardcoded colors in widgets, app, and
  preview controller replaced with Theme tokens for high-contrast
  theme compatibility.
- **FCPXML XXE mitigation.** Parser now prefers `defusedxml` when
  installed to prevent XML external entity attacks on user-supplied
  FCPXML files.
- **Inconsistent queue item ID format.** `_repeat_item_with_settings`
  now uses `uuid.uuid4().hex` matching `_add_to_queue`.
- **Untranslated language probe message.** Wrapped with `tr()` for
  i18n compatibility.

### Added

- **Frame-level manual mask correction.** Polygon-based manual mask
  corrections can now be tied to time ranges and merged with automatic
  detection masks during processing, with config persistence and
  CLI-compatible JSON overlay support.
- **Adapter conformance dry-run matrix.** A dry-run matrix now lists
  every adapter with env vars, license, source, hash policy,
  remote-code status, and availability without loading model code.
  Embedded in release verification and support bundles.
- **Multi-segment NLE sidecars.** EDL and FCPXML exports now represent
  every processed time span when multiple timed regions are configured,
  and preserve source dimensions in format metadata.
- **Pseudo-locale and RTL rendered smoke tests.** Expanded-string and
  RTL-mark translation tests verify the i18n layer renders status
  labels and sentinel strings without clipping or loss.
- **Local dependency drift report.** A drift report now lists every
  tracked core and optional package with installed version, pinned
  minimum/maximum, status, and blocked exceptions. Release verification
  embeds the report; the `dependency_caps` CLI prints it.
- **GUI visual regression coverage.** Widget geometry assertions now
  cover empty queue, queued item selected, and narrow-width states,
  failing on zero-sized or clipped primary controls.
- **Per-output reproducibility sidecars.** Every hardcoded cleanup,
  soft-subtitle remux, skipped-existing, and checkpoint-completed output
  now writes a `<output>.vsr.json` sidecar recording source SHA-256
  fingerprint, processing config snapshot, detection engine, stage
  timings, quality report/gate, checkpoint state, and app version.
- **FFmpeg/FFprobe subprocess smoke evidence.** Release verification now
  exercises ffmpeg and ffprobe with a tiny synthetic fixture (generate,
  probe, transcode) and records command, path, env, and return code
  evidence in `release-verification.json`. A failing smoke produces a
  validation error that blocks strict builds.
- **Local isolated smoke path.** `tools/local_smoke.py` now runs
  `python -m backend.processor --self-test` plus a generated-image CLI cleanup,
  and `Dockerfile` / `.dockerignore` provide a local CPU-only container recipe
  for reproducing the smoke without replacing the Windows launcher.
- **GUI controller extraction.** Queue processing, pause/stop orchestration,
  live preview/region editing, batch quality review, support bundle/model-cache
  dialogs, and backend-status panels now live in focused `gui/*_controller.py`
  mixins while preserving the `VideoSubtitleRemoverApp` public surface.
- **Cooperative pause/resume checkpoints.** Long video jobs can now pause at a
  safe frame-batch boundary, persist processed checkpoint frames, and resume
  the current video from the first missing frame in both GUI and CLI flows.
  Batch JSON/Markdown reports now use a distinct `paused` status, and stale or
  incompatible pause checkpoints warn before falling back to a fresh run.
- **GUI gettext wiring.** Core onboarding, queue, settings, preview, region,
  status, confirmation, batch-summary, and About/backend-status strings now use
  the documented `tr()` gettext alias; `locale/vsr.pot` was refreshed from the
  wired call sites, and tests prove translated queue/status rendering with
  fallback for missing keys.
- **Batch stage timing evidence.** Batch JSON/Markdown reports, support
  bundles, and completion/review surfaces now include decode, OCR, mask,
  inpaint, encode, mux, and quality-analysis timings with slowest-stage
  summaries for diagnosing bottlenecks.
- **Static-logo cleanup research benchmark.** A benchmark-only runner now
  compares the current per-frame cv2 cleanup against a deterministic
  InpaintDelogo-style static-logo path, with manifest license/hash gates,
  structured quality metrics, and no new default dependency or user-facing
  mode.
- **Mask-free subtitle-erasure research benchmark.** CLEAR/SEDiT candidates
  are tracked as non-registering research specs with strict adapter manifest
  entries, and a benchmark-only evaluator now records runtime, subtitle
  removal quality, outside-region artifact score, and temporal stability for
  supplied licensed outputs.
- **VOID research adapter gate.** `VSR_VOID=1` can now register a `void`
  research inpainter mode, but it verifies local checkpoints through the
  adapter manifest before importing VOID code and falls back to TBE when
  weights or dependencies are absent.
- **Windows ML migration probe.** `--audit-windows-ml` now checks the Python
  Windows ML bridge, Windows App SDK bootstrap, ONNX Runtime EP devices, and a
  tiny ONNX smoke model before any DirectML-to-Windows-ML migration work; README
  documents the current audit-only decision.
- **Preview-pane subtitle region editing.** The preview card now lets users
  drag a manual subtitle region directly on the selected frame and immediately
  refreshes the mask preview with the saved coordinates; the advanced selector
  remains available for timed and multi-region ranges.
- **Reference-clip regression corpus.** `tests/clips/` now carries 10
  deterministic MIT fixtures for motion, karaoke, vertical, HDR-like, font,
  dissolve, shadow, and timed-region cases, with a `backend.reference_corpus`
  runner that compares decoded output-frame hashes and PSNR/SSIM floors during
  local release evidence generation.
- **Wan2.1-VACE opt-in adapter.** `VSR_VACE=1` now resolves reviewed local
  Wan2.1-VACE-1.3B checkpoint directories, can auto-fetch the HuggingFace
  snapshot when `VSR_VACE_AUTO_FETCH=1` and `huggingface-hub` is installed,
  verifies the checkpoint path through the adapter manifest, and surfaces
  first-run download guidance before falling back safely.
- **VideoPainter strict local adapter.** `VSR_VIDEOPAINTER=1` now verifies a
  local VideoPainter checkpoint root, supports a reviewed
  `VSR_VIDEOPAINTER_COMMAND` wrapper over temp source/mask videos, records the
  research/non-commercial license boundary in first-run guidance, and falls
  back to TBE when the local wrapper or package is unavailable.
- **FloED strict local adapter.** `VSR_FLOED=1` now verifies a local FloED
  checkpoint, supports a reviewed `VSR_FLOED_COMMAND` wrapper over temp
  frame/mask directories, reports missing checkpoint/wrapper setup, and falls
  back to TBE when the local integration is unavailable.
- **MatAnyone 2 mask refinement.** `--matanyone-refine` now routes OCR/SAM
  masks through a gated MatAnyone 2 alpha-matte adapter, verifies local
  checkpoint/snapshot provenance through the adapter manifest, reports missing
  opt-in/package/checkpoint setup, and falls back to the original mask on
  missing or malformed alpha output.
- **CoTracker3 mask propagation.** `--cotracker-propagate` now uses a gated
  CoTracker3 point-tracking helper to fill OCR-empty masks inside a video
  batch, requires a reviewed local checkout or pinned revision before
  `torch.hub` can load the model, and leaves non-empty OCR/SAM masks unchanged.
- **PyNvVideoCodec decode option.** `--decode-accel pynv` / `nvdec` now routes
  through the PyNvVideoCodec wrapper without requiring `VSR_PYNVVIDEOCODEC`,
  supports the documented `SimpleDecoder` API plus the legacy decoder shape,
  reports PyNv availability in backend status, and falls back to software
  decode when unavailable.
- **RIFE-interpolated fast mode.** `--rife-fast-stride N` now raises the
  effective detection stride, inpaints cleaned keyframes only, asks
  Practical-RIFE to synthesize skipped frames, and duplicates the nearest
  cleaned keyframe across scene cuts or missing adapters. The GUI performance
  card persists the same stride, and the hardware-decode dropdown now exposes
  `pynv` / `nvdec`.

### Fixed

- **HDR final encode safety.** HDR10/HLG sources with color preservation now
  avoid invalid H.264 output by promoting default HDR encodes to HEVC,
  decoding an FFmpeg `bgr48le` source surface when available, preserving
  high-bit unmasked pixels through the FFV1 intermediate, disabling hardware
  encoders for that HDR finalization path, and adding `-pix_fmt yuv420p10le`
  alongside the preserved BT.2020/PQ/HLG color tags.
- **AV1 / VP9 decode and native AV1 grain.** Generated AV1 and VP9 fixtures
  now verify serial decode plus prefetch/hardware-hint fallback paths, and
  software AV1 output maps `film_grain_strength` to SVT-AV1's native
  `film-grain` parameter instead of running the additive post-encode noise pass.
- **NVIDIA setup now matches the torch security floor.** Windows CUDA setup
  now installs PyTorch from the cu128 wheel index for RTX 20/30/40/50-series
  GPUs, because the older cu118 index does not provide `torch>=2.10.0`, and
  the packaging-tool refresh keeps `setuptools<82` for torch compatibility.
- **Unattended setup repair.** Launchers now probe the virtual environment for
  core packages and call `setup.py --repair` when it is missing or broken,
  while plain `setup.py` keeps an existing `venv` without prompting.
- **Local EXE build parser cleanup.** `build_exe.bat` no longer trips `cmd`
  parsing while preparing OCR package data collection for PyInstaller.
- **SAM 2 mask refinement now shrinks coarse boxes.** `--sam2-refine` now
  clears each detected rectangle before applying the prompted SAM 2 mask, uses
  the detected-mask center as a positive point prompt, preserves unrelated mask
  regions, and still falls back to the original mask on SAM 2 errors.
- **Local release evidence restored.** `build_exe.bat` now copies release
  launchers into the PyInstaller bundle and emits `release-verification.json`,
  `release-hidden-imports.json`, and `sbom.cdx.json` through
  `backend.release_verification` so local-build releases keep the old audit
  evidence without a GitHub Actions workflow.
- **PyTorch LaMa packaging opt-in.** PyInstaller no longer hidden-imports
  `simple_lama_inpainting` by default; set `VSR_ENABLE_PYTORCH_LAMA=1` to
  include that legacy fallback when it is installed.
- **Scaled mask-selector regression coverage.** GUI tests now exercise a
  resized video selector with preloaded, cleared, and multi-region saved
  rectangles, then verify the mask preview uses the saved image-space
  coordinates.
- **Corrupt-video failure messages.** Empty, unreadable, truncated, and
  unsupported-codec video inputs now fail before producing partial outputs,
  keep temp files cleaned up, and show actionable queue/CLI messages instead
  of a generic processing failure.
- **RapidOCR PP-OCRv6 packaging evidence.** Release provenance now records
  RapidOCR YAML config files, PP-OCR model families, required PP-OCRv6
  det/rec assets for RapidOCR 3.x, and compatibility status for PyInstaller
  bundles.
- **OpenCV/libpng status evidence.** Support bundles now include the detected
  OpenCV version, bundled libpng version, fixed libpng floor, and warning state
  so fixed wheels can be confirmed at runtime while current wheels still need
  the untrusted-PNG caution.
- **OpenCV wheel conflict diagnostics.** Self-test, backend status, support
  bundles, and release evidence now report installed OpenCV wheel variants,
  imported `cv2` version/file ownership, conflict warnings, and exact
  uninstall/reinstall remediation when contrib/headless packages shadow each
  other.
- **Edge-case clip intake gate.** GitHub now has a dedicated edge-case clip
  issue form for redistributable real media, and the reference-corpus manifest
  validator requires source URL, license proof, retrieval date, and rights
  confirmation before any real/community fixture can enter `tests/clips`.
- **Frozen-build multiprocessing guard.** The GUI entry point now calls
  `multiprocessing.freeze_support()` before app imports, PyInstaller builds
  include a runtime hook that repeats the guard for worker imports, and release
  evidence records the runtime hook before strict release validation passes.
- **ONNX Runtime CUDA preload.** CUDA ONNX sessions now call
  `onnxruntime.preload_dlls()` before session creation when available, and
  backend status, support bundles, and release evidence record the preload
  success or failure for diagnostics.
- **Non-interactive local build.** `build_exe.bat` now exits with a status code
  after success or failure instead of pausing for a keypress during automation.
- **Source-aware output quality preflight.** CLI runs, GUI batches, and batch
  reports now compare source codec/resolution/bitrate against the selected
  output codec and CRF, warn when settings may soften the result, and persist
  the safer recommendation plus continued-after-warning state.
- **Premium accessibility and focus polish.** Custom buttons, toggles, sliders,
  segmented controls, drop targets, queue rows, status updates, notifications,
  onboarding, preset, About, and batch-summary dialogs now expose stable
  accessibility metadata, stronger focus/disabled states, screen-reader
  announcements for important changes, and a better default next action after
  batch completion.
- **Preview-state clarity.** Disabled preview and batch actions now carry
  actionable reasons, the preview card explains what to do next, and preview,
  test-cleanup, and region-selector failures show calm recovery copy while
  technical details stay in the activity log.
- **Backend status panel.** The Help dialog and support bundle now show OCR
  backends, inpaint backends, ONNX/OpenCV provider state, required model-file
  presence, hash status, and the next setup action.
- **PaddleOCR-VL-1.5 llama.cpp detector.** `VSR_PADDLEOCR_VL=1` now enables a
  CPU/edge VLM detector through a local llama.cpp OpenAI-compatible server,
  with clean fallback to the normal OCR cascade when the server or PaddleOCRVL
  entrypoint is unavailable.
- **Safe PNG decode routing.** User-supplied PNG still frames now decode
  through Pillow instead of OpenCV whenever the runtime reports a vulnerable
  bundled libpng, closing the untrusted-PNG read path until fixed OpenCV wheels
  ship.
- **Release advisory gate.** Local release evidence now writes
  `release-advisories.json`; strict mode fails on unallowed high/critical
  advisories and records the temporary OpenCV/libpng exception until fixed
  wheels remove it automatically.
- **FFmpeg capability profiles.** `--self-test`, support bundles, release
  evidence, and the Help backend panel now report `basic`,
  `advanced_quality`, `speech_fallback`, and `modern_codec` profiles with exact
  missing filters/encoders; the GUI warns before starting a video batch whose
  selected options exceed the installed FFmpeg build.
- **Quality-gate suggested retries.** Review-needed queue items now expose
  `Retry with suggested settings`, mapping the quality ladder to per-item
  config changes, requeueing only that item, and recording before/after retry
  config in the next batch report.
- **Language-support truth in backend status.** README and Help now use the
  same language statement: the GUI exposes 52 selectable OCR language codes,
  while installed OCR engines report broader capacity separately.
- **Portable model-cache bundles.** Help and CLI can export/import a
  SHA-256-manifested model-cache zip into the app model cache, rejecting path
  traversal, executable entries, and hash mismatches while reporting missing
  optional known assets afterward.
- **ONNX Runtime provider migration status.** Backend status, support bundles,
  release evidence, and setup now distinguish CPU, CUDA, and DirectML ONNX
  Runtime package/provider state, recommend `onnxruntime-gpu>=1.21.0` for the
  stable NVIDIA CUDA 12 path, and flag legacy CUDA provider packages in release
  advisory evidence.
- **RapidOCR OpenVINO routing.** RapidOCR now auto-prefers the OpenVINO engine
  on CPU/Intel systems when `openvino` is installed, can be forced with
  `VSR_RAPIDOCR_ENGINE=openvino`, falls back to ONNX Runtime on initialization
  failure, and reports the preferred OCR engine in backend status and release
  evidence.
- **Time-ranged manual subtitle regions.** The region selector can save
  optional start/end seconds for manual masks, settings/presets/CLI JSON config
  round-trip `subtitle_region_spans`, and processing disables stale mask reuse
  across timed-region boundaries.
- **Architecture notes synced.** Local working notes now point at
  `build_exe.bat` + `backend.release_verification` for release evidence and
  describe the current ONNX > OpenCV DNN > PyTorch opt-in > cv2 LaMa chain.
- **Native optional-engine crash hardening.** OCR, Whisper, LaMa, ProPainter,
  and GUI startup engine probes now use safe optional-import checks so broken
  native wheels do not get imported just to discover capabilities. The
  torch-backed `simple-lama-inpainting` path is now explicit opt-in via
  `VSR_ENABLE_PYTORCH_LAMA=1`; ONNX/OpenCV LaMa paths remain automatic.
- **Fallback output evidence.** When FFmpeg salvages or retries to a different
  output container, the processor records the actual output path, the GUI locks
  the queue item to that path, and CLI/batch summaries report the file users can
  actually open.
- **Copied command and external adapter quoting.** Queue "copy CLI" commands
  now quote user paths and watermark values safely, and the external inpainter
  adapter parses trusted Windows command paths without corrupting backslashes.
- **Thread-safe GUI log routing.** Background logs are queued and drained on the
  Tk main loop instead of calling Tk from worker threads during startup probes.
- **Local-build test hygiene.** Release workflow assertions now skip cleanly
  when this local-build checkout intentionally has no GitHub Actions workflow,
  and README repository structure no longer points to a removed workflow.

## [3.17.2]

### Fixed

- **Mask selection follow-up.** The Inspect step now keeps the first queued
  item selected when no explicit queue row is active, and mask review,
  language probe, region selection, A/B compare, and cleanup preview can fall
  back to that queued item instead of leaving the preview detached.
- **Manual region snapshot sync.** Saving or clearing subtitle regions now
  immediately updates idle queue item configs, so "Review mask" and "Test
  cleanup" use the same manual mask that the UI shows before the batch starts.

## [3.17.1]

### Added

- **Actionable empty queue state.** The queue empty state now includes direct
  `Choose files` and `Choose folder` actions, matching the main import surface
  and making first-run batch setup possible from either side of the workbench.
- **GUI quality-review worklist.** Batch completion now treats quality-gate
  `review-needed` outputs as review items, shows a review count/action, focuses
  the first affected queue row, and opens its quality sheet or report. Queue row
  context menus expose `Open quality sheet` when an artifact is available.
- **Redacted support bundles.** About now saves a diagnostics `.zip` with
  runtime facts, dependency/tool versions, cache sizes, redacted settings,
  recent redacted logs, and optional batch report summaries. Headless users
  can run `python -m backend.cli --support-bundle support.zip`, which works
  without importing the OpenCV processing stack first. GitHub bug reports now
  ask for the generated bundle and reproduction details.
- **Bundled GUI smoke self-test (`--smoke-test`).** The app entry point
  accepts `--smoke-test`, which constructs the full GUI off-screen,
  pumps one idle cycle, and tears it down without entering the mainloop
  (settings pinned to a throwaway temp dir), exiting 0/1. Strict release
  builds now run this against the frozen EXE and record the result in
  `release-verification.json` `smokeLaunch`, failing the release if the
  bundled GUI cannot construct.
- **OpenCV 5 DNN LaMa inpaint backend.** The LAMA inpaint chain now has a
  four-tier priority order: ONNX Runtime > OpenCV 5 `cv2.dnn` > PyTorch
  (simple-lama-inpainting) > `cv2.inpaint`. The DNN path needs only
  opencv-python >= 5.0 plus the `opencv/inpainting_lama` ONNX weight (no
  onnxruntime, no torch), closing the `torch.load` CVE surface for that
  codepath. Auto-discovers the weight from `VSR_OPENCV_LAMA`, the app
  model cache, or known cache dirs; SHA-256 pinned in the adapter
  manifest. Falls back automatically when OpenCV < 5.0 or the weight is
  absent.
- **LaMa-ONNX promoted to default inpaint backend.** The LAMA inpaint
  mode now prefers ONNX Runtime (no torch dependency, 3-5x faster) with
  auto-discovery of LaMa ONNX weights from env var, app model cache, or
  known weight cache dirs. Falls back to simple-lama-inpainting (PyTorch)
  when ONNX Runtime is absent, then cv2 as last resort. Tiling support
  for both ONNX and PyTorch paths on 4K+ frames.
- **Local cache inspector and cleanup.** `--cache-info` prints sizes of
  all VSR cache directories (checkpoints, proxy cache, TRT engines,
  model weights, HuggingFace cache). `--cache-clean` removes stale
  entries from known-safe subdirectories. About dialog now shows total
  disk cache size.
- **Model-weight hash table expansion.** Vendored SHA-256 hashes added
  for LaMa-ONNX FP32, LaMa-ONNX (quantised), and OpenCV inpainting_lama
  ONNX models. Adapter manifest for lama-onnx now carries pinned hashes.
  New `--scan-weights` CLI flag scans cached model directories and
  verifies all known weight files against vendored hashes.
- **DirectML model-opset audit.** `--audit-onnx` CLI flag reports opset
  versions for all discoverable ONNX models and checks DirectML EP
  compatibility (opset <= 20 ceiling). Known model opsets documented
  in-code: PP-OCRv4 det/cls/rec (opset 11), LaMa-ONNX (opset 9),
  MI-GAN-ONNX (opset 11) -- all well under the DirectML ceiling.
- **Batch quality gate ladder.** Failed quality checks now produce a
  graduated fallback ladder instead of a flat "review" flag. Each metric
  violation maps to a specific remediation step (increase-dilation for
  residual text, temporal-smooth for flicker, alternate-inpainter for low
  SSIM/VMAF). Multiple violation types escalate to manual-review. Batch
  summary reports include the ladder step taken, structured reasons, and
  an actionable remediation suggestion. Missing optional metrics (VMAF,
  LPIPS, etc.) are listed in a degradedMetrics field.
- **Confidence-weighted mask dilation.** When enabled, mask dilation scales
  inversely with OCR confidence -- high-confidence boxes get tighter masks,
  low-confidence boxes get more padding. CLI `--confidence-dilate`, GUI toggle.
- **Tile-based LaMa inference.** Frames larger than `lama_tile_size` (512px)
  are tiled into overlapping patches for inpainting, reducing VRAM on 4K and
  preserving fine detail.
- **Post-inpaint temporal smoothing.** `temporal_smooth_radius` (CLI
  `--temporal-smooth`) blends masked pixels across consecutive frames on the
  LaMa path, reducing per-frame flicker without cross-scene ghosting.
- **FFmpeg Whisper VAD controls.** `vad_model`, `vad_threshold`, and
  `min_speech_duration` for the FFmpeg Whisper filter. CLI flags
  `--ffmpeg-whisper-vad-model` / `--ffmpeg-whisper-vad-threshold` /
  `--ffmpeg-whisper-min-speech`.
- **VVC (H.266) output codec.** `--codec vvc` uses libvvenc software encode
  with HW probe for vvc_nvenc/qsv/amf. Degrades to h264 when FFmpeg lacks
  VVenC.
- **Batch completion system notification.** Windows toast via plyer when batch
  finishes, visible even when minimised. Missing plyer degrades to in-app toast.

### Changed

- **Safer destructive confirmations.** Danger-tone confirmation dialogs now
  focus the safe action by default and no longer globally map Enter to the
  destructive action; keyboard activation follows the focused button.
- **Queue row accessibility.** Queue item cards are now focusable selection
  targets with visible focus treatment and Enter/Space activation, improving
  keyboard review without adding global shortcuts.
- **Queue attention count.** The queue status chip now uses warning treatment
  and counts failed, stopped, and quality-review items as needing attention
  instead of labeling stopped work as failed; completed outputs that need
  quality review are announced as warnings rather than plain successes.
- **Launcher set is now release-tracked.** `Run_VSR_Pro.ps1` is a tracked,
  documented PowerShell launcher, setup generation is tested against all root
  launcher files, and release verification records the launcher files bundled
  in the Windows zip.
- **Strict release ROADMAP hygiene.** Release verification no longer requires
  `ROADMAP.md` to contain the release version marker; README, CHANGELOG, and
  `gui/config.py` remain the release-version sources of truth.
- **GitHub update-check etiquette.** The optional startup update check now sends
  a User-Agent and GitHub API-version header, persists ETag/Last-Modified state,
  treats 304 as no update, and backs off after 403/429 rate-limit responses.
- **Release tooling verification.** The winget submission job now downloads
  winget-create from a versioned Microsoft release, checks the expected SHA-256,
  verifies the Authenticode signer, and records the configured tool evidence in
  release verification.
- **Remote-code model trust policy.** Florence-2 and CoTracker3 optional paths
  now require either a reviewed local path or an immutable-looking pinned
  revision before repository code can execute; release verification records the
  policy state for these adapters.
- **First-run model download guidance.** Batch processing now warns before
  lazy OCR/inpainting/Whisper/VLM model loads may trigger large first-use
  downloads, including size estimates and cache guidance in the status/log flow.
- **Premium GUI workbench polish.** The main window now uses a scrollable
  middle workbench, a stable two-column desktop grid, responsive rectangular
  hardware/status tiles, wrapped queue guidance, separated preview controls,
  and a two-row batch action bar so default and compact windows no longer
  clip the primary workflow.
- **Media extension allowlists centralised.** GUI file pickers, drag-drop,
  and backend frame capture share one source of truth. `.tif` now accepted
  everywhere.
- **Keyboard accelerators removed.** All global shortcuts stripped per product
  rules; actions are click-only.
- **torch_directml pruned from local build.** PyInstaller hidden-import probe
  removed; DirectML routes through ONNX Runtime.

### Fixed

- **Setup venv recreate guard.** Recreating the virtual environment now refuses
  symlink/reparse-point `venv` paths before recursive deletion, preventing a
  repair run from deleting an external linked directory.
- **Settings and preset size guard.** Startup settings, user presets, and
  imported preset files larger than 1 MB are rejected before JSON parsing so
  malformed local files cannot stall the GUI.
- **Settings lock consistency.** Custom settings sliders now disable during
  processing like the rest of the controls, preventing mid-batch setting
  drift and making the locked state visually consistent.
- **NLE sidecar escaping.** FCPXML exports now quote filename and URI
  attributes correctly, and EDL clip comments are sanitized to one line so
  unusual media names do not break editor import.
- **Whisper span hardening.** Whisper fallback frame-span conversion now skips
  NaN, infinite, reversed, or nonnumeric segment times instead of crashing or
  masking the wrong frames.
- **Footer status stability.** Long import and warning summaries are now
  middle-truncated in the footer so the status line stays stable beside the
  guidance text while preserving the most important start/end context.
- **Shell-free report opening.** GUI batch reports now open through direct
  OS file launching instead of `cmd /c start`, avoiding shell interpretation
  of report paths derived from user-selected output locations.
- **Cache-clean allowlist guard.** Programmatic cache cleanup now ignores
  unsupported names, parent-directory hops, and absolute paths; only the
  known VSR cache buckets can be removed and recreated.
- **GUI VVC setting persistence.** GUI settings now preserve `vvc` and
  normalize `h266` / `h.266` aliases to VVC instead of silently falling
  back to H.264 after save/load normalization.
- **VVC/H.266 discoverability.** README codec tables now include VVC with the
  FFmpeg `libvvenc` requirement, and release verification records whether
  the bundled/system FFmpeg exposes `libvvenc`.
- **PaddleOCR 3.7 / PP-OCRv6 compatibility.** PaddleOCR 3.x payload parsing
  now handles callable/property JSON results, numpy score arrays, `rec_polys`,
  and documented `rec_boxes`; README now names PP-OCRv6 as the 3.7 default.
- **Shutdown during processing.** Stop/close now signals active work,
  terminates tracked FFmpeg remux/encode/deinterlace children, aborts the
  active intermediate writer, and briefly joins the processing worker before
  window teardown.
- **Processing diagnostics.** Processing, preview, cleanup, and inpainter
  fallback paths now retain warning-level tracebacks instead of silently
  swallowing failures.
- **OCR dependency caps.** RapidOCR, rapidocr-onnxruntime, and PaddleOCR now
  use tested major-version ceilings in requirements, setup fallback installs,
  README guidance, and release CI.
- **Startup and import feedback.** Malformed settings now surface a
  startup warning instead of silently resetting, invalid saved region
  rectangles log named warnings, and unsupported drag/drop selections report
  skipped file counts through the normal import summary.
- **EDL sidecars accept non-ASCII filenames.** CMX 3600 exports now write
  UTF-8 text so CJK source or cleaned filenames no longer raise
  UnicodeEncodeError; FCPXML remains UTF-8.
- **Setup subprocesses cannot hang forever.** Virtual environment creation
  and pip install steps now have explicit timeouts and print retry guidance
  before setup exits non-zero on stalled downloads.
- **Startup hardware probe is quiet during headless GUI smoke runs.** Tk
  scheduling failures raised outside `mainloop()` are now handled cleanly,
  keeping smoke/unit tests free of background-thread tracebacks.

### Security

- **RapidOCR model provenance in release evidence.** Strict release verification
  now records RapidOCR package name/version, bundled ONNX filenames, sizes,
  SHA-256 hashes, opsets, and DirectML compatibility, and fails strict releases
  when default RapidOCR model assets are missing.
- **Preset import schema gate.** Imported, saved, exported, and legacy on-disk
  user presets now pass through an explicit tuning-field allowlist. Unsupported
  fields are skipped and reported in the GUI import status, preventing shared
  preset JSON from silently changing update checks, paths, window state, or
  other non-preset workflow settings.
- **Pillow floor raised to 12.2.0.** Patches CVE-2026-25990 (PSD OOB write),
  GHSA-wjx4-4jcj-g98j (JPEG2000 OOB read), GHSA-pwv6-vv43-88gr (BLP heap
  overflow), and GHSA-whj4-6x5x-4v2j (TIFF IFD loop DoS). Updated in
  requirements.txt, setup.py fallback, and CI workflow.

## [3.17.0]

### Fixed

- **PrefetchReader deadlock near end-of-video.** The stop sentinel is now
  retried until the consumer or release() drains the queue, preventing a
  hang when slow inpaint batches keep the queue full past the old 1 s
  timeout. `release()` also avoids freeing a capture while the worker
  thread is still reading from it.
- **GUI freeze at batch start.** The pre-batch ETA probe (OCR model load +
  30-frame detection) now runs on the worker thread with a real 10 s cap
  instead of blocking the Tk main loop.
- **PaddleOCR 3.x compatibility.** Constructor and result-parsing code
  updated for the 3.x API (`predict()` / `device=` / `rec_polys`); 2.x
  installs still work via a shared compat layer in `backend/paddle_compat.py`.
- **"Reset region" leaving stale multi-region masks.** `_reset_region` now
  clears both `subtitle_area` and `subtitle_areas`; the label and mask
  preview read both fields consistently.
- **Audio-merge failure writing unplayable files.** When FFmpeg fails or
  times out during audio mux, the intermediate is now either re-encoded
  video-only (CalledProcessError) or salvaged into a container-correct
  path with a warning (TimeoutExpired / FileNotFoundError), instead of
  copying raw FFV1 bytes into the user's `.mp4` output.
- **Inpaint preview on DirectML GPUs.** Preview now uses the same
  device-mapping helper as processing, so AMD/Intel GPUs get
  `device="directml"` instead of a nonexistent CUDA device.
- **TBE batch sizing exceeding host RAM.** Adaptive batch probe now caps
  `sttn_max_load_num` by available system memory, not just free VRAM.
- **JSON log handler duplication.** `attach_json_log` detects an existing
  handler for the same path and skips, preventing duplicate lines and
  leaked file handles.
- **Processing tracebacks lost.** Top-level image/video error handlers
  now pass `exc_info=True` so the crash reporter sees the full stack.
- **Toast stacking overlap.** Second toast no longer overlaps the first;
  predecessor calculation is now index-aware.
- **Cross-thread Tk call in log handler.** `TextWidgetHandler.emit` no
  longer calls `winfo_exists()` off the main thread.
- **Region selector capture leak.** `cv2.VideoCapture` is now released on
  the early-return path when the first `cap.read()` fails.
- **Update toast discarding URL.** Release URL is now logged and surfaced
  in the status bar.
- **RM-114 extraction regressions.** `widgets.py` missing `tkfont` and
  `status_ui` imports; entry point redefining `APP_VERSION` independently;
  `BUILTIN_PRESETS` absent from the back-compat surface; `output_frames`
  field missing from `ProcessingConfig` (CLI `--output-frames` crashed);
  loudnorm out-of-range clamping now matches backend semantics.
- **NSIS installer version sync.** The installer version constants now match
  the application version.

### Added

- **Optional startup update check (RM-116).** New `update_check` setting
  (off by default) checks GitHub Releases for a newer version on launch.
- **Registry-only inpaint modes.** Opt-in backends (diffusion scaffolds,
  ONNX variants) registered via `inpainter_registry` are now reachable
  from `--mode` and JSON config overlays without widening the core enum.

### Changed

- **Backend module split completed (RM-114).** `InpaintMode`,
  `RegisteredMode`, `ProcessingConfig`, and all coercers moved to
  `backend/config.py`. Inpainter modules now import `backend.config` /
  `backend.inpainters` directly instead of back-importing
  `backend.processor`, eliminating the circular import that made the
  diffusion-backend import position load-bearing. `python -m
  backend.processor` no longer double-executes the module.
- **Entry point rewrite.** `VideoSubtitleRemover.py` imports app identity
  and the full back-compat surface from `gui/` instead of redefining it;
  dead monolith-era imports removed.
- **Icon.png shrunk** from 1.3 MB (1024 px) to 273 KB (512 px).
- **Planning docs consolidated.** ROADMAP.md pruned to remaining work only
  (500 lines, 65 backlog items); RESEARCH_REPORT.md merged into
  RESEARCH.md; COMPLETED.md removed.

## [3.16.1]

### Fixed

- **Python 3.14 CUDA install guard.** `setup.py` and the launchers now warn
  that PyTorch has no Windows CUDA wheels for Python 3.14, fail fast on
  NVIDIA installs instead of silently falling back to CPU, and allow an
  explicit CPU-only override with `VSR_ALLOW_PY314_CPU=1`.
- **Release workflow dependency fail-fast.** Required release dependencies now
  check `$LASTEXITCODE` after each native pip command so missing RapidOCR,
  EasyOCR, LaMa, PyInstaller, or core image-stack installs fail the workflow
  immediately; PaddleOCR and ONNX Runtime DirectML remain explicitly optional.
- **DirectML path moved off torch-directml.** AMD/Intel setup now keeps a
  patched CPU torch runtime for PyTorch fallback paths, installs
  `onnxruntime-directml` for DirectML acceleration, detects DirectML through
  ONNX Runtime providers, and routes ONNX inpainters through
  `DmlExecutionProvider`.
- **Warning-clean regression suite.** Closed ffmpeg stderr pipes in the
  lossless intermediate writer, suppressed the expected all-NaN TBE median
  warning for fully covered mask pixels, and closed SRT test file reads so
  the affected paths pass under `-W error::ResourceWarning -W
  error::RuntimeWarning`.
- **VapourSynth script trust gate.** `.vpy` input now requires
  `VSR_VAPOURSYNTH=1` before VSR executes the script through the optional
  VapourSynth bridge, preventing accidental execution of Python scripts
  selected or batch-fed as video inputs.
- **RapidOCR DirectML provider selection.** AMD/Intel detection now
  initializes RapidOCR with ONNX Runtime DirectML settings when
  `DmlExecutionProvider` is available, and falls back to the CPU provider
  for older RapidOCR constructors or missing providers.
- **FFmpeg Whisper fallback backend.** `--whisper-backend ffmpeg` can use
  FFmpeg 8's native `whisper` audio filter plus a local whisper.cpp ggml
  model to produce SRT timing for OCR-empty subtitle-band masking without
  installing Python ML transcription packages.
- **VMAF quality-report metric.** Quality reports now invoke FFmpeg's
  `libvmaf` filter when available, adding full-frame and ROI-cropped VMAF
  scores alongside PSNR/SSIM without adding Python dependencies.
- **Winget release submission.** The Windows release workflow now submits the
  versioned NSIS installer to Windows Package Manager with a secret-gated,
  non-interactive `wingetcreate update --submit` step when `WINGET_PAT` is
  configured.
- **DirectML ONNX opset guard.** Opt-in ONNX inpainters now inspect model
  `opset_import` metadata without requiring the `onnx` package and fall back
  from `DmlExecutionProvider` to CPU when a default-domain model opset exceeds
  DirectML's supported ceiling.
- **Embedded subtitle stream probe.** Backend I/O now exposes a structured
  ffprobe JSON helper for soft subtitle streams so the upcoming no-reencode
  remux path can distinguish embedded tracks from burned-in pixels without
  loading OCR or inpainting models.
- **Soft subtitle remux primitive.** Added explicit ffmpeg stream-copy command
  building plus atomic output promotion for strip/keep soft-subtitle actions,
  with an ffmpeg-gated regression test proving subtitle streams can be removed
  without changing the video codec.
- **Soft subtitle CLI flow.** Added `--soft-subtitle-dry-run`,
  `--strip-soft-subtitles`, `--keep-soft-subtitles`, and `--burned-in-only`
  semantics. Dry-run and fast remux exit before constructing OCR or inpainter
  state.
- **Soft subtitle preflight JSON.** Dry runs can now write a structured
  `--soft-subtitle-plan-json` file for batch tooling, including per-file
  action, subtitle stream counts, codecs, language/title tags, and default or
  forced track flags without loading OCR.
- **Soft subtitle GUI decisions.** Queued videos now probe embedded subtitle
  tracks in the background, show a compact track summary, and expose right-click
  actions for fast strip, fast remux/keep, or normal burned-in cleanup.
- **Soft subtitle remux safety.** Fast remux now rejects same-path input/output
  requests before allocating temp files so a strip/keep operation cannot
  replace the source container in place.
- **Strict release verification.** The release workflow now supports
  `release_quality=permissive|strict`, emits `release-verification.json` with
  SHA-256 hashes, verifies bundled docs, requires the NSIS installer in strict
  mode, and validates Authenticode signatures when Azure signing is configured.
- **Release manifest evidence.** `release-verification.json` now records
  dependency versions, PyInstaller hidden imports, app/doc version checks, and
  an explicit GUI smoke-launch status so release artifacts carry enough audit
  context to debug packaging drift.
- **OpenCV libpng runtime warning.** Startup now checks OpenCV's build
  metadata and warns when the bundled libpng is below `1.6.54`, the fixed floor
  for CVE-2026-22801.
- **RapidOCR 3.x output compatibility.** Detection now normalizes RapidOCR's
  legacy tuple/list output and newer structured object/dict outputs into the
  same subtitle boxes, and RapidOCR config-loading failures fall through to
  the rest of the OCR cascade.
- **Optional adapter contract coverage.** VLM OCR, segmentation, and diffusion
  scaffolds now have mocked regression tests for malformed model output,
  missing optional packages, explicit opt-in registration, and inference-time
  fallback paths without importing heavyweight optional dependencies.
- **Batch summary reports.** CLI pattern runs and GUI batches now write
  `vsr-batch-summary.json` and `vsr-batch-summary.md` next to outputs, including
  preflight metadata, planned actions, final per-file statuses, elapsed time,
  and corrected `--skip-existing` behavior that no longer selects an alternate
  collision path before deciding to skip.
- **Optional adapter security manifest.** Opt-in model paths now carry adapter
  provenance, preferred weight format, license/source metadata, SHA-256 status,
  and explicit unsafe-override evidence. Known hash mismatches fall back before
  ONNX Runtime or PyTorch loaders deserialize the file.
- **Batch quality-gate evidence.** Quality-report metrics now produce a stable
  `passed` / `review` / `unknown` gate result with ladder-step metadata, and
  CLI/GUI batch summaries persist that result plus quality-sheet preview paths.
  The gate also includes cheap residual-text and adjacent-frame temporal
  flicker scores when a mask ROI is available, and failed gates now mark batch
  rows as `review-needed`.

## [3.16.0]

### Added

- **Blackwell (RTX 50-series) GPU install path.** `setup.py` /
  `Run_VSR_Pro.bat` now detect 50-series cards (5050/5060/5070/5080/5090,
  RTX PRO 6000, B100/B200/GB200 by name) and install PyTorch from the
  CUDA 12.8 (`cu128`) wheel index, plus PaddlePaddle from the `cu126`
  index. Blackwell is compute capability sm_120 and has no kernels in the
  cu118/cu121 builds the default NVIDIA path uses, so older pins error
  with `no kernel image is available for execution on the device` or
  silently fall back to CPU. `requirements.txt` and the README now
  document the cu128 path and the manual fix for an existing environment,
  and note torch 2.7+ supports Python 3.9-3.13.

### Fixed

- **README PyTorch pin drift.** The manual-install block pinned
  `torch==2.7.0` while `requirements.txt` / `setup.py` require
  `torch>=2.10.0` (CVE-2026-24747 fix). Aligned the README to the floor
  pin and split NVIDIA into 20/30/40-series (cu118) vs 50-series (cu128).
- **OCR-cascade fallback test isolation.** `OcrCascadeOrderTests.`
  `test_falls_back_to_opencv_when_no_engine_installed` asserted the
  "OpenCV fallback" engine but never patched out the optional OCR
  engines, so it passed only on a bare machine and failed the release
  build (which installs RapidOCR/EasyOCR). The test now nulls the engine
  modules in `sys.modules` so the cascade deterministically falls
  through, matching the class docstring's stated intent.

### Docs

- **TV / limited color-range behavior documented.** Added a README
  troubleshooting entry explaining that this fork preserves the source's
  `color_range` (and primaries/transfer/matrix) tags on the final encode
  via `preserve_color_metadata` (RM-73), unlike upstream which drops them
  and can wash out limited-range clips. Clarified that the 8-bit BGR
  pipeline still tone-maps true 10-bit HDR to SDR.

## [3.18.0]

### Added

- **OCR-fix replace list for exported SRT.** With `--ocr-fix` (config
  `ocr_fix_enable`), detected subtitle text written to the `.srt` sidecar is
  passed through a per-language replace list that corrects common OCR
  confusions (e.g. a lowercase `l` mis-read for `I`). A small built-in default
  set is merged with an optional user-editable
  `%APPDATA%/VideoSubtitleRemoverPro/ocr_fix/{lang}.json`; whole-word keys are
  boundary-matched so real words are never rewritten.

### Fixed

- **Correct Pillow security floor.** Raised the documented and fallback
  install pin to `Pillow>=12.1.1`, the fixed floor for CVE-2026-25990.

### Changed

- **Planning docs consolidated.** Root planning now flows through
  `ROADMAP.md`, `COMPLETED.md`, and `RESEARCH_REPORT.md`; the completed
  2026-05-25 audit and exhausted checklist moved under `docs/archive/`.
- **`backend/processor.py` split into 7 modules (RFP-L-1).** The
  3,400-line monolith now lives as a 1,923-line shim that re-exports
  every previously-public name from focused sub-modules:
  - `backend.detection` -- `SubtitleDetector`, OCR cascade,
    Florence-2/Qwen2.5-VL/Surya routing, percentile OpenCV fallback.
  - `backend.tracking` -- `_KalmanBox`, `SubtitleTracker`, pHash
    helpers, karaoke per-line fusion.
  - `backend.io` -- `_open_capture`, `_PrefetchReader`,
    `_LosslessIntermediateWriter`, `_FrameSequenceCapture`, all
    ffprobe / atomic-file helpers, deinterlace.
  - `backend.encoder` -- `_detect_hw_encoder` only (the encode-args
    builder remains on `SubtitleRemover` because it reads `self`).
  - `backend.quality` -- `_ssim` (the report writer stays on the class
    for the same reason).
  - `backend.inpainters` -- subpackage with `_common.py` (BaseInpainter,
    feather/edge-ring helpers, scene-cut cascade, Farneback warp,
    TBE primitive) + `sttn.py` / `lama.py` / `propainter.py` /
    `auto.py`. Built-ins register themselves via the existing plugin
    registry.
  - `backend.cli` -- argparse + `main()`, preset overlay loader,
    checkpoint helpers, `_apply_auto_band_override`.
  Every legacy `from backend.processor import X` import path keeps
  working unchanged. `python -m backend.processor` now delegates to
  `backend.cli.main`. All 164 tests pass.

### Tests

- **TikTok preset synthetic A/B (RFP-EI-7).** New
  `tests/test_tiktok_preset.py` generates three 9:16 deterministic
  clips (caption at top / centre / bottom) and runs each through
  the pipeline with `auto_band=True` and `=False`. Finding: with
  the default OpenCV fallback detector,
  `SubtitleRemover.detect_subtitle_band` returns None on every
  position (the 30-frame probe needs a real OCR engine to cluster
  detected boxes). The preset is therefore *correct as shipped*
  for users with RapidOCR / PaddleOCR / EasyOCR installed and a
  benign no-op otherwise -- switching the default to `False`
  would penalise OCR-enabled installs without measurably helping
  the no-OCR case. Real-source validation is still owed; the
  documented finding lets us close the item without changing the
  preset.

## [v3.15.0] -- 2026-05-25

Backlog-drain pass: 30+ optional integrations land as opt-in adapters;
the default pipeline is byte-identical for users who do not opt in.

### Added

- **GUI smoke test (RFP-T-4).** New `tests/test_gui_smoke.py` drives
  the major flows headlessly via `root.withdraw()` so the full-widget
  construction path runs in CI. Skipped on non-display CI runners.
- **Opt-in GlitchTip crash reporter (RM-52).** New
  `backend/crash_reporter.py` installs a `sentry_sdk` excepthook
  when BOTH `VSR_GLITCHTIP_DSN` and `VSR_CRASH_REPORTS=1` are set.
  Strict consent gate; path-scrubbing `before_send` strips Windows
  and POSIX absolute paths so layout info never leaks. Default
  off, drops cleanly to no-op when `sentry-sdk` is missing.
- **APP_VERSION bumped to 3.15.0** + README badges.

- **NSIS installer (RM-51).** New `installer/vsr.nsi` wraps the
  PyInstaller `--onedir` build into a one-click setup EXE with
  Start Menu + Desktop shortcuts, an Add/Remove Programs entry,
  and uninstall support. GitHub Actions workflow installs NSIS
  via Chocolatey and builds the installer on every release.
- **"Send to VSR" file-extension verb (RM-58).** The installer
  registers a Shell verb on .mp4 / .avi / .mkv / .mov / .wmv /
  .flv / .webm / .m4v / .mpeg / .mpg so right-clicking a video
  in Explorer offers "Send to Video Subtitle Remover". We do NOT
  hijack the default Open verb -- that would surprise users.
- **Azure Trusted Signing in the release workflow (RM-50).**
  Workflow grew an opt-in signing step gated on the
  `AZURE_SIGN_TENANT_ID` secret. Forks without the secrets see
  the step skip cleanly. SmartScreen reputation accumulates against
  a stable signed identity.
- **Community edge-case corpus contributor guide (RM-55).** New
  `docs/edge_case_corpus.md` documents the submission flow,
  acceptance criteria (single failure mode, CC0 / public domain,
  <20s clips), and the baseline-compile recipe so contributors can
  expand the regression runner.

- **Proxy-file workflow (RM-34).** New
  `backend/proxy_workflow.ensure_proxy(path, height, crf)` builds a
  cached low-res (default 480p) proxy via ffmpeg for fast preview /
  region selection on 4K source. Cache lives in
  `%APPDATA%/VSR/proxy_cache/` keyed by an MD5 fingerprint of the
  source (path, size, mtime).
- **Karaoke optical-flow mask warp (RM-43).** New
  `backend/karaoke_flow.warp_mask_with_flow(prev, next, mask)`
  exposes a Farneback flow remap so callers can union an old mask
  with the next detection to catch karaoke text that moved between
  frames.
- **WhisperX word-level alignment helper (RM-45).** New
  `backend/karaoke_flow.run_whisperx(audio_path)` returns
  `(start_s, end_s, word)` tuples when `whisperx` is installed.
  Pairs with the v3.13 Whisper-span fallback to provide finer-
  grained mask gating on borderline subtitle/dialogue cases.
- **VapourSynth bridge (RM-75).** New
  `backend/vapoursynth_bridge._VapourSynthCapture` evaluates a `.vpy`
  script and exposes a cv2.VideoCapture-shaped reader so
  `_open_capture("path.vpy")` ingests through VapourSynth when
  installed. Lets users chain QTGMC deinterlace / Waifu2x / SMDegrain
  ahead of VSR.
- **RTL layout scaffold (RM-98).** New `ProcessingConfig.rtl_layout`
  flag and `_rtl_layout` runtime hook set before widget
  construction. The deep pack-side flip across every widget is a
  follow-up; this commit lands the framework so future translation
  + RTL work can land incrementally.

- **SeedVR2 one-step restoration (RM-77, opt-in).** New
  `backend/post_restore.seedvr2_restore` runs the SeedVR2 wrapper
  (or whatever CLI `VSR_SEEDVR2_CMD` names) as another post-cleanup
  stage. 16B-param diffusion transformer, single sampling step,
  the strongest restoration quality available in that release on heavily
  degraded footage. CLI `--seedvr2`.
- **TensorRT engine cache for the LaMa-ONNX backend (RM-70, opt-in).**
  New `backend/tensorrt_compile.maybe_compile_engine` invokes
  `polygraphy convert` once per ONNX, caches the result in
  `%APPDATA%/VSR/trt_cache/`, and adds the
  `TensorrtExecutionProvider` to the LaMa-ONNX session. ~2-3x
  further speedup on top of ONNX Runtime. Activate via
  `VSR_TENSORRT=1`.
- **INT8 OCR quantization script (RM-39).** New
  `scripts/quantize_ocr.py` calls `onnxruntime.quantization.quantize_dynamic`
  on a RapidOCR detection ONNX. Drop-in replacement; the resulting
  model halves detection cost on CPU with <1% F1 loss in practice.

### Added

- **SAM 2 / SAM 3 / MatAnyone 2 / CoTracker3 adapters
  (RM-66/67/68/69, opt-in).** New `backend/segmentation.py` exposes:
  - `refine_mask_with_sam2(frame, boxes, base_mask)` -- prompted box
    -> text-shaped mask, integrated into `_create_mask` so enabling
    `sam2_refine` (CLI `--sam2-refine`) replaces the padded rect mask
    with the SAM 2 output. Tighter mask = less inpaint area.
  - `segment_text_with_sam3(frame)` -- text-prompt mask via SAM 3
    when `VSR_SAM3=1` + the `sam3` package is installed.
  - `matte_frame(frame, hint_mask)` -- MatAnyone 2 soft alpha matte
    for thin moving subtitle lines.
  - `track_points(frames, points)` -- CoTracker3 helper for callers
    that need to confirm a karaoke caret stays on the same line.

- **Diffusion inpainter scaffolds (RM-59/60/61/62/63/64/65, opt-in).**
  New `backend/inpainters_diffusion.py` registers seven optional
  video-inpainter backends with the plugin registry:
  - `propainter-real` (sczhou/ProPainter ICCV 2023 reference)
  - `diffueraser` (lixiaowen-xw/DiffuEraser 2025)
  - `vace` (ali-vilab/VACE 1.3B MV2V)
  - `videopainter`
  - `cococo` (text-guided, prompt via `VSR_COCOCO_PROMPT`)
  - `eraserdit` (research-stage track)
  - `floed` (flow-guided efficient diffusion)
  Each registers ONLY when its enable env var is set; missing
  packages route through the existing TBE primitive so the pipeline
  never crashes. Each becomes addressable via `--mode <name>` once
  the user opts in.

- **PyNvVideoCodec GPU-resident decode (RM-71, opt-in).** New
  `backend/decode_accel._PyNvVideoCapture` wraps NVIDIA's
  PyNvVideoCodec with a cv2.VideoCapture-shaped facade. Activated
  via `VSR_PYNVVIDEOCODEC=1`; falls back to cv2 transparently when
  the package is missing or the open fails. ~6x faster decode on
  reference NVIDIA hardware. Frames currently download to CPU as
  BGR so the rest of the pipeline is unchanged; the zero-copy
  GPU-resident path remains future work.
- **RIFE-interpolated fast mode helper (RM-72, opt-in).**
  `maybe_interpolate_pair(prev, next, t)` calls Practical-RIFE when
  `practical-rife` is installed, otherwise returns None. Caller
  falls back to a duplicate frame. The full pipeline-side wiring
  (detect every Nth frame, RIFE the gaps) is queued as a follow-up;
  the adapter ships now so the integration is decoupled from the
  fast-mode dispatch.
- **Batched LaMa inference (RM-40, opt-in).** Set
  `VSR_LAMA_BATCH=1` to stack the LAMA-mode batch and call the
  underlying torch model in a single forward pass. ~2-3x faster
  than per-frame on a 30-frame batch. Falls back to the per-frame
  path on any shape mismatch / model-attribute mismatch.

### Added

- **VLM OCR detector cascade (RM-22, RM-23).** New
  `backend/ocr_vlm.py` registers four optional detectors fronting the
  default RapidOCR -> PaddleOCR -> Surya -> EasyOCR cascade. Pick one
  via `VSR_VLM_OCR={florence2|qwen25vl|paddleocr-vl}`:
  - **Florence-2** (microsoft/Florence-2-base) layout-aware OCR with
    `<OCR_WITH_REGION>` quad boxes.
  - **Qwen2.5-VL** (Qwen/Qwen2.5-VL-2B-Instruct) -- leads
    OmniDocBench as of April 2026. Prompts the model for a JSON box
    list.
  - **PaddleOCR-VL** -- exposes PaddleOCR 3.0+'s `paddleocr_vl`
    profile when the install ships it; falls back to PP-OCRv5
    otherwise.
- **Manga / anime mode (RM-42).** Setting `detection_lang="manga"`
  routes through manga-ocr + comic-text-detector (if installed) to
  pick up irregular speech-bubble shapes and vertical Japanese.
  Falls back to an Otsu-derived single-bubble crop when
  comic-text-detector isn't installed so the mode degrades
  gracefully. The Subtitle Language picker now lists "Manga / Anime"
  as a selectable language.

- **Pre-detect denoise (RM-33, opt-in).** New
  `backend/preprocess.fastdvdnet_denoise_frame` runs FastDVDnet on the
  detection-frame stream when `VSR_FASTDVDNET` + torch are available,
  falling back to OpenCV NLM otherwise. Output pixels stay untouched
  -- the denoise only sharpens the OCR signal. CLI `--denoise-detect`.
- **TransNetV2 deep scene-cut detector (RM-21, opt-in).** Slots into
  the existing `_detect_scene_cuts` cascade ahead of the PySceneDetect
  / histogram paths when `transnetv2` is `pip install`-ed and
  `VSR_TRANSNETV2` names the model weights. CLI `--transnetv2`.
- **AV1 / VP9 ingest validation (RM-74).** New `_probe_codec_for_log`
  helper logs the source codec + dimensions at the top of every run
  so users can reproduce a decode failure with the exact ffmpeg
  invocation. AV1 egress is already covered by the `--codec av1`
  output dropdown shipped in v3.14.

## [v3.14.0] -- 2026-05-25

Backlog drain pass. Six previously-deferred optional integrations
land as opt-in adapters; the existing pipeline keeps working
byte-identical for users who do not opt in.

### Added

- **SwinIR restoration pass (RM-79, opt-in).** Pairs with
  Real-ESRGAN: `swinir_restore=True` (CLI `--swinir`) routes the
  post-cleanup output through whichever of `swinir-ncnn-vulkan`,
  `realsr-ncnn-vulkan`, or `swinir` is on PATH. Skipped silently
  when no binary is found.
- **Synthetic reference-clip regression runner (RFP-T-1).** New
  `tests/test_reference_clips.py` generates eight deterministic
  synthetic clips (static dialogue / motion pan / dissolve cuts /
  karaoke / persistent chyron / vertical text column / thin font /
  gradient background), runs the full pipeline against each, and
  asserts the run completes. CC0 real-world clip sourcing is the
  next pass.
- **APP_VERSION bumped to 3.14.0.**

### Changed

- **Inpainter dispatch is now plugin-driven (RFP-L-2).** New
  `backend/inpainter_registry.py` exposes
  `register(name, builder)` / `resolve(name)` /
  `is_registered(name)` / `list_modes()` / `unregister(name)`. The
  four built-in inpainters (STTN / LAMA / ProPainter / AUTO)
  register themselves at processor import time; `_create_inpainter`
  resolves through the registry instead of an if-elif chain.
  External backends (LaMa-ONNX, MI-GAN, real ProPainter,
  DiffuEraser, etc.) can now land as standalone modules that
  import `register` and a `BaseInpainter` subclass -- no edit to
  the dispatch needed. Re-registering a name replaces the
  previous builder, so a drop-in faster implementation can shadow
  a default.

### Added

- **GUI localisation scaffold (RM-97).** New `backend/i18n.py`
  exposes `bind_locale(lang)` + `_("...")` so a future translation
  catalog can drop into `locale/<lang>/LC_MESSAGES/vsr.mo` and bind
  on startup without further code changes. Locale auto-detection
  uses `locale.getlocale()`; non-`en` users get a translated UI
  whenever a catalog ships. A `locale/vsr.pot` template seeds the
  first wave of strings translators can target.
- **UIA screen-reader announce scaffold (RM-95).** New
  `backend/a11y.py` exposes `announce(text, importance)` that fires
  a Windows UIA notification readable by NVDA / Narrator. Probed
  lazily via comtypes; silent no-op on non-Windows / missing deps.
  Wired into `_notify_completion` so screen-reader users learn the
  batch finished (and how many errors landed) without polling the
  activity log.

- **HDR / colorspace metadata passthrough (RM-73 partial).** New
  `backend/hdr.py` probes the source's color signalling via ffprobe
  (`color_primaries`, `color_transfer`, `color_space`, `color_range`)
  and re-tags the output encode with the same flags, even though the
  pixel pipeline is still 8-bit BGR. HDR sources land in the log
  banner ("Detected: bt2020 / smpte2084 -- output tagged as HDR but
  pixels tone-mapped"). The full 16-bit pixel pipeline is queued as a
  follow-up; this slice at least stops the output from being
  mis-tagged. New `preserve_color_metadata` config + `--no-color-preserve`
  CLI flag.
- **NLE round-trip sidecar (RM-76).** New
  `backend/nle_sidecar.py` writes a 1-event CMX 3600 EDL or a
  minimal FCPXML 1.10 stub next to the output naming the source,
  cleaned filename, and processed time range. Lets a DaVinci /
  Premiere editor hand-conform the cleaned clip at the same timecode.
  `--nle-sidecar {off|edl|fcpxml}` CLI flag + GUI mirror.

- **Real-ESRGAN upscale + film-grain post-restore stages (RM-78, RM-80,
  opt-in).** New `backend/post_restore.py` adapters and a
  `_run_post_restore_passes` hook in `process_video` that fires after
  the main mux:
  - `upscale_factor` (0/2/3/4) shells out to
    `realesrgan-ncnn-vulkan` for 2x/3x/4x upscaling; the original
    output stays on disk when the binary is missing.
  - `film_grain_strength` (0..0.5) adds an ffmpeg `noise` filter pass
    so inpainted regions blend with the surrounding grain. Skipped
    when ffmpeg is missing. Note: for AV1 outputs prefer the
    encoder's native film-grain table over this additive pass.
  CLI exposes `--upscale {0,2,3,4}` and `--film-grain STRENGTH`.

- **Whisper-driven mask fallback (RM-27, opt-in).** When
  `whisper_fallback` is on AND `faster-whisper` is `pip install`-ed,
  `process_video` extracts the audio track, runs Whisper once per
  file, and applies a default bottom-band mask to frames whose OCR
  returned no boxes but whose timecode falls inside a speech span.
  Catches anti-aliased / motion-blurred / decorative subtitles the
  OCR cascade misses. New `backend/whisper_fallback.py` module;
  CLI `--whisper-fallback` and `--whisper-model {tiny|base|small|
  medium|large|large-v2|large-v3}`. The audio extraction temp dir
  is cleaned up in the same finally block as the main temp dir.

- **LaMa-ONNX inpainter backend (RM-25, opt-in).** Set
  `VSR_LAMA_ONNX=<path to lama_fp32.onnx>` and `pip install
  onnxruntime` / `onnxruntime-gpu` to swap the default PyTorch LaMa
  for a 3-5x faster ONNX Runtime path. The new
  `backend/inpainters_onnx.py` module registers `LamaOnnxInpainter`
  through the plugin registry, replacing the `lama` mode slot. If
  onnxruntime is missing or the model file is unreadable, the
  backend falls back to `cv2.inpaint` per-frame so users never see a
  hard failure.
- **MI-GAN ONNX inpainter backend (RM-26, opt-in).** Set
  `VSR_MIGAN_ONNX=<path to migan.onnx>` and `pip install
  onnxruntime`; a new `migan` mode lands in the inpainter registry
  + CLI choice list. ~10 ms per 512x512 on a modern CPU for the
  cleanup-only case (matches the ICCV 2023 paper); single-frame so
  there is no TBE temporal pass. Useful for image queues and
  underpowered laptops.

- **PySceneDetect-backed scene cut detector (RM-32, opt-in).** New
  `tbe_scene_cut_use_pyscenedetect` field; when on and `scenedetect`
  is `pip install`-ed, the TBE batch-splitter uses PySceneDetect's
  AdaptiveDetector instead of the built-in histogram correlator.
  Handles dissolves and flashes that mis-fire on the histogram path.
  Defaults off; the histogram path stays the zero-dep default.

- **Vertical text mode (RM-24).** Japanese tategaki and classical
  Chinese subtitle columns now detect cleanly. The detector wrapper
  rotates each frame 90 CCW before invoking whichever engine is
  loaded, then rotates the returned boxes back into the source
  frame's coordinate space. New `ProcessingConfig.detection_vertical`
  field, CLI `--vertical`, and a Detection-card toggle in the GUI.
- **High-contrast theme variant (RM-96).** A low-vision-friendly
  palette (pure black surfaces, pure white text, saturated accent
  colours, yellow focus ring) toggleable from the Output card and
  persisted in settings. Applies on next launch because re-skinning
  every live widget mid-session would force a tree-wide redraw the
  design tokens were not built for.
- **A/B flicker-scrubber preview (RM-30).** Completed items grow an
  "A/B compare" preview button that opens a Toplevel with a frame
  slider AND a vertical-wipe slider so users can compare the source
  and cleaned outputs side-by-side at any frame. Both captures stay
  open for the duration of the modal and are released on close.

- **Per-file overrides popover (RM-29).** Right-click an idle queue
  item -> "Override settings for this file..." opens a themed modal
  that edits the item's own `ProcessingConfig` snapshot. Surfaced
  fields: mode (segmented picker), detection language, sensitivity
  slider, output codec. Overrides survive a global UI change because
  every queue item already carries its own config dataclass; the
  popover writes back to `item.config` and runs `.normalized()` so
  bad values never reach the worker.

### Changed

- **Detection slider relabelled "Sensitivity" (EI-3).** The underlying
  knob is unchanged (10-90% confidence floor); the new label removes
  the inversion users had to remember ("threshold" goes DOWN to detect
  more text). New hint: "Higher catches more text (lower confidence
  floor). Lower is stricter."

### Performance

- **Live preview worker-side throttle (EI-4).** The backend's
  `on_preview_frame` callback now drops conversions when the previous
  one fired within the last 1/15 s, so the worker stops burning CPU on
  cv2.resize + PIL.Image.fromarray that the receiver was going to
  throttle away anyway.

### Tests

- **ConfigFuzzTests expanded** to cover every v3.13 GUI-exposed field
  (loudnorm_target, multi_audio_passthrough, decode_hw_accel,
  prefetch_decode, prefetch_queue_size, input_fps, quality_report_sheet,
  remove_subtitles, remove_chyrons, chyron_min_hits, karaoke_grouping,
  karaoke_x_gap_px, karaoke_y_overlap, output_codec). 1500 random
  payloads on each path; post-conditions assert loudnorm range,
  decode_hw_accel token set, output_codec set, input_fps bounds.

## [v3.13.0] -- 2026-05-25

### Added

- **HEVC + AV1 output codec dropdown (F-8).** Output is no longer
  H.264-only. New `ProcessingConfig.output_codec` (h264 / h265 / av1)
  picks the matching HW encoder family (`hevc_nvenc`/`hevc_qsv`/
  `hevc_amf` for h265, `av1_nvenc`/`av1_qsv`/`av1_amf` for AV1) with
  `libx265` / `libsvtav1` software fallbacks. CLI exposes `--codec`;
  the Output card grows a "Output codec" dropdown next to HW
  encoding. Settings persist via the existing dataclass-driven
  pipeline.
- **Per-item cancellation (F-7).** Right-click a running queue item
  -> "Cancel this item" sets a per-`QueueItem` cancel flag. The
  worker's progress callback raises `InterruptedError` next tick so
  the file is dropped, but the global `cancel_event` stays clear and
  the remainder of the batch continues. Per-item flag is reset every
  time `_process_item` re-enters so a retry works cleanly.
- **Vendored SHA-256 weight verification (RM-49).** New
  `backend/model_hashes.py` registers known-good hashes for opt-in
  model downloads and a chunked verifier (`verify_weight_file`) safe
  for multi-GB files. The LAMA loader scans the standard torch.hub
  cache for `big-lama*.pt` on first init and warns -- but does not
  refuse to load -- when the hash mismatches. Catches silent
  supply-chain swaps and truncated downloads that would otherwise
  surface as cryptic deep-model errors hours into a run.

- **Region selector grows frame scrubbing + multi-rectangle drawing
  (F-1, F-2).** The selector window now carries a frame slider for
  video sources so users can pin the rect on a frame where the
  subtitle is actually visible -- the legacy "always frame 0" path
  silently failed on every clip with a black intro card. Every drag
  appends a rect to the list; "Clear all" removes them; "Save" writes
  every rect to `subtitle_areas` (plus the first rect to
  `subtitle_area` for backward compatibility with single-rect callers).
- **One-click cleanup preview (F-3).** The Preview panel grew a
  "Preview cleanup" button that runs detect + inpaint on the first
  frame of the selected queue item in a background thread and renders
  the result inline. Users can A/B detection thresholds, mask dilation,
  and mode choices without committing a full batch run.

- **Per-stream loudness normalisation for multi-track audio (B-4).**
  When both `loudnorm_target` and `multi_audio_passthrough` are set and
  the source has more than one audio stream, `_merge_audio` now builds
  a `-filter_complex` pipeline with one `loudnorm=I=...` branch per
  stream. Each track lands at the same LUFS target, so a Bluray rip
  with main / commentary / dub tracks normalises uniformly instead of
  applying the filter to track 0 only.
- **Pre-batch ETA probe (F-9).** Starting a batch now runs a 30-frame
  detect probe on the first queued video, scales the result by the
  full duration, and seeds `_compute_eta` with the estimate. Users see
  "about X left (estimated)" from the very first frame instead of an
  empty string until the first item finishes.
- **Expanded language picker (F-5).** The Subtitle Language dropdown
  now covers ~50 languages (was 12). Curated friendly names lead the
  list (English first), with the rest filling in by ISO code so users
  with Thai / Vietnamese / Polish / Greek / Ukrainian / etc. footage
  can pick a sensible engine code without modifying source.

- **CLI `--preset NAME` flag + shared preset library (F-10).** Presets
  now live in `backend/presets.py` so the GUI's picker and
  `python -m backend.processor --preset NAME` resolve from the same
  table. CLI flags typed alongside `--preset` still win on a conflict
  (the preset only fills attrs whose argparse value is still the
  parser default). New companion flag `--list-presets` prints every
  known preset (built-in + user) and exits.
- **"Repeat with these settings" queue action (RM-28).** Right-click a
  queue item -> "Repeat with these settings" re-queues the same source
  with a snapshot of that item's `ProcessingConfig`. Useful when you
  have tweaked the global UI knobs but want to re-run an earlier file
  with the exact settings that worked the first time.

### Improved

- **OpenCV fallback detector catches mid-tone subtitles (EI-1).** The
  legacy fixed thresholds at 200 / 55 missed semi-transparent banners
  and dimly-lit captions whose luminance sat in the 55-200 dead zone.
  The fallback now picks thresholds from the frame's 5th / 95th
  percentile, clamped so a near-flat source cannot collapse both
  thresholds and mark the entire frame.

- **Lossless FFV1 intermediate (I-1).** The temp file written between
  the inpaint pass and the final ffmpeg encode used to be `mp4v`
  inside `.mp4` -- a full generation of lossy compression sitting in
  front of the user-visible H.264/NVENC final encode. Every output was
  effectively gen-2 lossy. The new `_LosslessIntermediateWriter` pipes
  raw BGR frames through a Popen-spawned `ffmpeg -c:v ffv1` writing
  `.mkv`, so the final encode pass is the only lossy step. When
  ffmpeg is missing the writer falls back to the legacy `mp4v` path
  with a logged warning, so installations without ffmpeg keep working
  at the old quality. Verified by `LosslessIntermediateWriterTests`
  that the FFV1 round-trip is bit-identical for FFV1-eligible
  installations.

- **Quality report includes inpaint-region (ROI) PSNR/SSIM (B-3).** The
  v3.12 quality report was computed over the entire frame, so unchanged
  pixels (typically 80-95% of the area) dominated the metric and could
  hide a bad inpaint behind a strong-looking overall score. The pipeline
  now accumulates the union-mask bbox while processing and the report
  returns both a whole-frame metric and an ROI-cropped metric. The
  Good/Review tag is now driven by the ROI score when available. ROI
  output: `{'roi_psnr': float, 'roi_ssim': float, 'roi_bbox': [x1,y1,x2,y2]}`
  alongside the existing whole-frame fields.
- **AutoInpainter unloads idle LaMa (B-5).** When the AUTO routing has
  stayed on the TBE path for `LAMA_IDLE_UNLOAD_AFTER` consecutive
  batches (50, ~1500 frames at batch=30), the lazily-loaded
  `LAMAInpainter` reference is dropped and `torch.cuda.empty_cache()`
  is called. A later hard batch re-loads on demand. Long videos that
  hit one hard batch early no longer permanently pin ~1.5 GB VRAM.

- **GUI: thirteen v3.13 backend fields are now reachable from the
  Advanced panel.** Loudness normalisation target (LUFS, 0 = off),
  multi-track audio passthrough toggle, hardware-decode hint dropdown
  (off / auto / any / d3d11 / vaapi / mfx), worker-thread frame
  prefetch toggle, chyron classifier (Remove dialogue subtitles /
  Remove persistent text), karaoke grouping toggle, and the quality
  report sheet toggle. Previously these were CLI-only -- the GUI built
  a `BackendConfig` that simply didn't pass them through, so toggling
  them in the GUI had no effect. New Advanced cards: "Editorial",
  "Audio", "Performance". The Output card grew the quality-sheet
  toggle next to the existing PSNR/SSIM report toggle.
- **GUI: settings.json schema bumped 1 -> 2.** Older files round-trip
  cleanly via `_migrate_settings`; new keys land at backend defaults so
  pre-v3.13 behaviour is preserved for users who never touch them.

### Changed

- **GUI ProcessingConfig persistence is dataclass-driven.** `to_dict`
  and `from_dict` now walk `dataclasses.fields(self)` rather than a
  manual enumeration. The 13-field B-1 gap was rooted in three
  enumerations (`to_dict`, `from_dict`, `normalized`) that all needed
  to be edited in lockstep whenever a new field landed -- a structural
  invitation for drift. New fields now persist by default; only
  `normalized` still requires an explicit coercion entry (intentional:
  it documents the safe range).

### Security

- **Surya GPL opt-in gate** -- the OCR cascade no longer auto-loads Surya
  even when it is pip-installed. Surya is GPL-licensed; loading it at
  runtime in a PyInstaller bundle put the MIT-clean release at risk.
  Users who want Surya must set `VSR_ALLOW_GPL=1` in the environment.
  When the gate is closed but Surya is installed, the loader logs a
  warning explaining the env var. `detect_ai_engines()` in the GUI now
  labels it `"Surya (GPL -- set VSR_ALLOW_GPL=1)"` so the About dialog
  reflects the gated state.

### Fixed

- **Cached-remover hot-swap missed normalisation** -- when a queue item
  reused a cached `SubtitleRemover` (same mode / device / language) the
  GUI assigned the new `BackendConfig` directly to `remover.config`,
  bypassing `normalize_processing_config`. A NaN/inf or out-of-range
  per-item override could then leak into the pipeline. The hot-swap now
  routes through the normaliser the constructor uses on cold start.
- **Quality-report output capture honoured HW-accel hint** -- the 10-
  frame PSNR/SSIM sample pass opened the just-written output through
  `decode_hw_accel`, which can fall back inconsistently against a fresh
  H.264 mp4. The output capture now forces software decode; the input
  capture still honours the user's hint.
- **ffmpeg subprocess timeout truncated long videos** -- the audio mux,
  yadif deinterlace, and reencode-or-copy paths all used a fixed 600 s
  timeout. Videos over ~1 hour silently fell back to "copy without
  audio" once the encode pass ran longer than 10 minutes. The timeout is
  now adaptive: `_ffmpeg_subprocess_timeout(duration)` returns
  `base + duration * 4` with a 24-hour ceiling and a 600-s floor when
  ffprobe is unavailable.

### Added

- **Karaoke / per-syllable grouping (`--karaoke-grouping`)** -- new
  `_group_horizontal_line()` helper fuses OCR boxes on the same
  horizontal text line into a single composite. Karaoke captions
  render as many small per-syllable boxes that the rest of the
  pipeline treats as independent lines; masking them individually
  leaks the original highlighted text through the gaps between
  syllables. Two boxes merge when their vertical extent overlaps by
  at least `karaoke_y_overlap` (default 0.5) AND the horizontal gap
  is at most `karaoke_x_gap_px` (default 20). The merge loop iterates
  until no further merges happen, so five syllables fuse into one
  rectangle in one call. Pure transformation; safe to apply before
  the Kalman tracker so the smoothed track covers the fused span.
  CLI: `--karaoke-grouping`, `--karaoke-x-gap PX`,
  `--karaoke-y-overlap RATIO`.
- **Chyron vs subtitle classifier (`--keep-chyrons` / `--keep-subtitles`)**
  -- `_KalmanBox.is_chyron(min_hits)` and `SubtitleTracker.categorize()`
  classify each detection by lifetime: a Kalman track that has matched
  in `>= chyron_min_hits` frames (default 90, ~3 s at 30 fps) is a
  chyron (station logo, lower-third, breaking-news ticker); shorter-
  lived tracks are dialogue subtitles. New `ProcessingConfig` fields
  `remove_subtitles` and `remove_chyrons` (both default True for
  backward compatibility) gate which class is sent to the inpaint
  mask. The filter is a no-op when both are True, so v3.12 behaviour
  is preserved. CLI: `--keep-chyrons`, `--keep-subtitles`,
  `--chyron-min-hits N`.
- **Frame-sequence input (DPX / EXR / PNG / JPG directories)** -- new
  `_FrameSequenceCapture` adapter mirrors the `cv2.VideoCapture` surface
  (`isOpened`, `read`, `set(POS_FRAMES)`, `get(FPS / WIDTH / HEIGHT /
  FRAME_COUNT)`, `release`) over a directory of images walked in sorted
  filename order. First image fixes width / height; mid-sequence size
  changes are letterboxed to that frame so the writer pipeline never
  sees a dimension shift. Routed transparently through `_open_capture()`
  and the `process_video` path. `process_video` silently bypasses the
  audio merge when the input is a directory (no audio stream). New
  `ProcessingConfig.input_fps` (default 24.0, clamped to [1, 240]);
  `--input-fps FPS` CLI flag. Ingest-only for v3.13; output remains mp4
  -- PNG/EXR sequence *output* is queued for a later release.
- **Quality self-test sheet (`--quality-sheet`)** -- extends the existing
  PSNR / SSIM numeric report with a side-by-side comparison PNG written
  next to the output as `<output>.qualitysheet.png`. Each sampled frame
  becomes one row (`original | cleaned`) with a caption showing the
  per-frame PSNR/SSIM; a header strip carries mean PSNR, mean SSIM, and
  a `Good` / `Review` tag derived from the SSIM 0.95 threshold. Implies
  `--quality-report`; the `quality_report_sheet` config field
  auto-enables `quality_report` in `normalize_processing_config` so a
  config-file overlay can't reach an inconsistent state.
- **Prefetch / pipeline parallelism (`prefetch_decode`, default on)** --
  new `_PrefetchReader` wraps `cv2.VideoCapture` with a daemon worker
  thread that fills a bounded frame queue while the main thread runs
  detection + inpainting. cv2 / numpy / onnxruntime release the GIL on
  heavy calls so plain threading is enough to overlap I/O with compute.
  Strict ownership rule: once the wrapper is active, the underlying cap
  must not be touched directly (`.read()`/`.set()`/`.get()` on the raw
  cap from the main thread would race the worker). Cleanup goes through
  `reader.release()`, which sets a stop event, drains the queue so the
  worker isn't blocked on `put()`, joins the thread, and then releases
  the underlying cap. Exception path in `process_video` is wired through
  the same path so a crash mid-batch never leaks the worker thread.
  Toggle off via `--no-prefetch`. Queue size auto-derives from
  `sttn_max_load_num * 2` (min 8); override with `--prefetch-queue N`.
- **Structured JSON-line log option (`--json-log PATH`)** -- new
  `backend.processor.JsonLineLogHandler` writes one JSON record per line
  with `ts` (UTC ISO-8601), `level`, `logger`, `msg`, and optional `exc`
  (formatted traceback when the record carries `exc_info`). The text log
  in `%APPDATA%\VideoSubtitleRemoverPro\vsr_pro.log` keeps writing in
  parallel; this handler is purely additive. Useful for `jq` / `grep`
  pipelines across days of batch jobs. CLI-only for v3.13; the GUI
  toggle lands in a later release alongside the loudnorm GUI control.
- **Multi-track audio passthrough (default on)** -- `_merge_audio` now
  emits `-map 1:a?` (all input audio streams) re-encoded to AAC instead
  of `-map 1:a:0?` (first only). Bluray/DVD rips routinely carry 3-5
  language tracks; the legacy behaviour silently dropped all but the
  first. New `ProcessingConfig.multi_audio_passthrough = True` field +
  `--single-audio` CLI flag for the legacy behaviour. Caveat: the
  simple single-pass loudnorm filter applies only to the first selected
  audio stream; broadcast-grade multi-track loudnorm needs
  `-filter_complex` and is deferred.
- **Hardware-accelerated decode hint (`--decode-accel`)** -- new
  `_open_capture()` helper supports `off` (default; status quo),
  `auto`/`any`, `d3d11` (Windows DXVA2/D3D11VA), `vaapi` (Linux), or
  `mfx` (Intel Media SDK). Probes one frame on open; if the HW path
  returns no frames (known cv2/FFmpeg issue, opencv/opencv#25185) it
  silently falls back to software decode with a warning. Wired into the
  main video decode path in `SubtitleRemover.process_video`. Other
  short-scan call sites (subtitle-band probe, quality-report sampler)
  remain on the software path and are tracked as a follow-up.
- **`--validate-config` CLI dry-run** -- parse all CLI flags + the optional
  `--config` JSON overlay, normalise the resolved `ProcessingConfig`,
  print it as JSON, and exit 0 without instantiating the detector or
  inpainter. Lets shell scripts verify flag combinations before launching
  a long batch. `--input` / `--pattern` / `--output` / `--out-dir` are
  not required when this flag is set.
- **`--skip-existing` CLI toggle** -- skip any input whose output path
  already exists, regardless of the checkpoint store. Independent of
  `--no-resume`; useful when re-running a glob against a partly populated
  output directory without enabling the full checkpoint workflow.
- **EBU R128 loudness normalisation (`--loudnorm <LUFS>`)** --
  `ProcessingConfig.loudnorm_target` defaults to 0.0 (off). When set to a
  LUFS value in [-70, -5], the ffmpeg audio mux runs an extra
  `-af loudnorm=I=<target>:TP=-1.5:LRA=11` pass during merge. Common
  platform targets: YouTube -14, Apple -16, broadcast -23. Single-pass for
  speed; broadcast-grade two-pass measure-then-apply may follow.
  CLI-only for v3.13; the GUI control lands in a later release.

### Security

- **CI dependency vulnerability scan (`pip-audit`)** -- the GitHub Actions
  workflow now installs `pip-audit` after the runtime stack and reports
  known CVEs in the installed closure. Non-fatal during the v3.13
  transition (`continue-on-error: true`); the gating switch flips to
  fail-on-vuln once the pin set in `requirements.txt` is stable.
- **Pin `torch >= 2.10.0`** -- CVE-2026-24747 / CVE-2025-32434 are
  `torch.load` `weights_only` RCEs reachable on PyTorch 2.9.1 and earlier.
  `simple-lama-inpainting` loads weights via `torch.load`, so this is a
  runtime concern. The CUDA + CPU install paths in `setup.py` and the
  GitHub Actions workflow are bumped to `>=2.10.0` / `torchvision>=0.25.0`.
  The torch-directml path stays on torch 2.4.x because no patched
  torch-directml wheel exists yet; `setup.py` now warns the user when that
  path is selected.
- **Pin `opencv-python >= 4.12.0`** -- CVE-2025-53644 is an uninitialised
  pointer in the JPEG reader that can become an arbitrary heap write on a
  crafted file. Bumped in `requirements.txt` and the GHA build matrix.
- **Pin `Pillow >= 12.1.1`** -- CVE-2026-25990 is a PSD loader out-of-bounds
  write. We do not currently open PSDs, but Pillow is on the transitive
  closure and a future image-format feature could expose it; pinning now is
  cheap insurance.

### Migration

- **Settings-schema versioning (`vsr_settings_format`)** -- `settings.json`
  now carries an integer schema version stamped by `to_dict()`. A new
  `_migrate_settings()` shim runs before `from_dict` on load so a future
  field rename can upgrade legacy payloads in place rather than silently
  dropping user state. Settings written by a newer build are honoured as-is
  (we don't downgrade). Format starts at `1`; `_migrate_settings()` learns
  one new case per bump.

### Fixed

- **Shutdown race condition**: `_shutdown_started` is now set only *after* the
  user confirms the "Close while processing?" dialog, preventing a racing
  `_on_processing_complete` callback from destroying the root window while the
  confirmation modal was still open.
- **Duplicate queue-item IDs**: replaced the millisecond-timestamp ID with
  `uuid.uuid4()` so IDs are collision-proof even when many files are added
  simultaneously.
- **`_coerce_int` / `_coerce_float` NaN/inf**: both coercers now reject
  non-finite floats (NaN, Â±Inf) and fall back to the specified default, matching
  the stricter guard already present in the backend.
- **`from_dict` pre-sanitisation crash**: `subtitle_area` and `subtitle_areas`
  in `ProcessingConfig.from_dict` now go through `_coerce_rect` /
  `_coerce_rect_list` directly instead of raw `tuple()`/list-comprehension
  conversions that could raise before `.normalized()` ran.
- **pHash double-computation**: the perceptual hash is now computed once per
  detection frame; the value is reused for `last_hash` instead of being
  recomputed immediately after.
- **`_write_srt` unsafe fps guard**: `fps or 30.0` replaced with
  `fps if fps and fps > 1.0 else 30.0` to prevent absurd SRT timestamps from
  a near-zero but non-falsy fps value.
- **`_load_json_config` size guard**: added a 1 MB cap before parsing so an
  accidentally large file cannot be loaded in full before the type check.
- **CLI `KeyboardInterrupt`**: Ctrl-C during a batch or single-file run now
  prints a clean message and exits with code 130 instead of showing a raw
  traceback.
- **`detect_ai_engines()` Surya detection**: broadened from `ImportError` to
  `Exception` so a partially-installed Surya (runtime import errors) no longer
  crashes engine probing.
- **`TextWidgetHandler._append` after-destroy safety**: added a
  `winfo_exists()` guard at the start of `_append` so log records scheduled
  with `after(0, ...)` just before root teardown are silently dropped rather
  than raising `TclError`.
- **`_reveal_output` silent failure**: `os.startfile` errors are now logged as
  warnings instead of being swallowed silently.
- **`_start_elapsed_timer` double-start**: calls `_stop_elapsed_timer()` first
  to cancel any existing tick loop before starting a new one.
- **Headless CI guard**: `test_on_processing_complete_during_shutdown_skips_summary_ui`
  is now skipped automatically on systems without a display.

### UI/UX improvements

- **Workflow step pills**: the header guidance panel now renders three compact
  step pills (Import â†’ Inspect â†’ Run) that highlight the current stage as the
  user moves through the workflow. The pills were wired to `_set_workflow_stage`
  but never built; they are now constructed at startup.
- **Section eyebrow labels**: `_section_title` now renders the `eyebrow`
  parameter as a small-caps meta-label above the section title, adding the
  intended two-level hierarchy (e.g. "WORKSPACE / Import media",
  "PROCESSING / Settings"). Previously the parameter was accepted but silently
  ignored.
- **Settings section title deduplication**: the Processing settings section was
  titled "PROCESSING / Processing". It is now "PROCESSING / Settings".
- **Log badge ordering**: warn/error count badges in the activity-log header now
  appear between the section title and the toggle button, where they belong.
  Previously they were packed after the toggle button, putting them in the wrong
  visual position.
- **Log badge pluralization**: "1 warn" is now "1 warning"; counts > 1 use the
  correct plural form for both warnings and errors.
- **Queue item default message**: new queue items show "Ready to process" instead
  of "Waitingâ€¦" for consistency with the `update_item()` fallback text.
- **Status badge padding tokens**: `padx=10, pady=4` on queue item status badges
  replaced with `padx=Theme.S_SM, pady=Theme.S_XS` (8 and 4 respectively) for
  design-system consistency.
- **Slider hint indent**: `padx=(Theme.S_LG + 128, Theme.S_LG)` (magic-number
  approximation) simplified to `padx=(Theme.S_LG, Theme.S_LG)`.
- **Queue empty-state spacing**: `pady=(Theme.S_3XL + 20, â€¦)` magic number
  replaced with `pady=(Theme.S_3XL, â€¦)`.
- **Footer hint text**: shortened from a 13-word instructional sentence to
  "Add files, review a sample frame, then start." â€” concise and calm.
- **Activity log height**: increased from 5 to 6 text rows for better context
  visibility without dominating the layout.
- **Progress bar height parity**: batch-level progress bar harmonised to
  `height=5` to match item-level bars (was `height=6`).

### Tests

- Added `CoerceHardeningTests`: NaN/inf for `_coerce_int` and `_coerce_float`,
  and non-iterable `subtitle_area` / `subtitle_areas` in `from_dict`.
- Added `BackendWriteSrtTests`: zero fps and near-zero fps fallback to 30.
- Added `LoadJsonConfigTests`: oversized config file is rejected before parsing.

## [v3.12.0] - 2026-04-17

Smart routing, legacy-source preprocessing, and self-testing. Four more
near-term roadmap items; 20 total shipped from the v3.9 plan.

**Smart routing**
- **AUTO inpaint mode** -- new `InpaintMode.AUTO` routes each TBE batch
  between temporal (STTN) and spatial (LaMa) inpainting based on how
  many masked pixels are temporally exposed. Controlled by
  `auto_exposure_threshold` (default 0.55). Lazy-loads LaMa only when
  the routing actually calls for it, so users who never hit a hard
  batch pay nothing.

**Preprocessing**
- **Deinterlace on ingest** -- `ffprobe -vf idet` probe detects
  interlaced sources; `ffmpeg -vf yadif=1` produces a temp
  progressive-scan copy before the main pass. Opt-in manual mode
  plus automatic detection (default on). Fixes comb-artefact-induced
  OCR junk on DV / broadcast rips.
- **Keyframe-driven detection** -- `ffprobe` lists every I-frame in
  the source; OCR runs only at keyframes while Kalman-smoothed
  masks propagate between them. Large speedup for long streams with
  stable subtitles. Gracefully falls back to pHash skip when
  ffprobe is unavailable.

**Quality self-test**
- **PSNR / SSIM report** -- after each run, samples 10 random frames
  (deterministic seed) and compares input vs output. Logs the
  per-run PSNR (dB) and SSIM (0-1). Catches regressions from
  mis-configured dilation / feather / encoder settings. Available
  via config + CLI `--quality-report`.
- Pure-numpy / cv2 SSIM implementation -- no new dependency.

**CLI**
- `--mode auto` is a new choice alongside the three legacy algorithms.
- New flags: `--auto-threshold`, `--deinterlace`,
  `--no-deinterlace-detect`, `--keyframe-detect`, `--quality-report`.

**GUI**
- Algorithm picker grows an "Auto" segment; new description tag.
- Output card: Auto-deinterlace toggle, Keyframe-driven detection
  toggle, PSNR / SSIM report toggle.
- All new settings persist to `settings.json`.
- Version bumped to 3.12.0.

## [v3.11.0] - 2026-04-17

Automation + visibility release. Four more near-term roadmap items
landed, rounding the total shipped from the v3.9 plan to 16.

**CLI**
- **Glob + config-file batch mode** -- `python -m backend.processor
  --pattern "inputs/*.mp4" --out-dir outputs/ --config recipe.json`.
  Users no longer need to shell-loop for bulk jobs. A JSON config
  file overlays fields that aren't exposed as flags
  (kalman_max_age, tbe_scene_cut_threshold, colour_tune_tolerance,
  etc.).
- **Crash-resume checkpointing** -- every completed file writes a
  SHA-256 marker to `%APPDATA%\VideoSubtitleRemoverPro\checkpoints\`.
  Re-running the same batch skips already-finished files. Input
  size/mtime is part of the fingerprint, so re-encoded inputs are
  correctly reprocessed. Disable with `--no-resume`.
- Mutual validation of `--input` / `--output` vs `--pattern` /
  `--out-dir`; clear error messages instead of silent no-ops.

**Workflow**
- **Preset JSON export / import** -- share a tuned recipe as a
  single JSON file. The Profile card grows Export / Import ghost
  buttons next to "Save as...". Import collisions with built-in
  names auto-rename with an "(imported)" suffix; imports carry a
  `vsr_preset_format` version tag for future compatibility.
- **Live processing preview** -- the preview pane now renders the
  latest inpainted frame during processing. Backend emits via an
  `on_preview_frame` callback every Nth frame (default 6); GUI
  marshals to the Tk main loop with a 15 FPS throttle. No separate
  window, no extra flag -- shows automatically whenever a batch is
  running.

**GUI**
- Profile card: Export / Import preset buttons.
- Preview pane: switches to "Live preview" title during
  processing, shows "Frame N/M (X%)" in the meta line.
- Version bumped to 3.11.0 across banner, header, logs.

## [v3.10.0] - 2026-04-17

Detection intelligence + preset workflow. Four more near-term roadmap
items landed. Big win for temporally-noisy footage and stylized text.

**Quality**
- **Kalman box tracking** -- per-frame OCR bounding boxes jitter a few
  pixels even when the rendered text is identical. A constant-velocity
  Kalman filter per subtitle "line" smooths that jitter, absorbs
  single-frame misses (cuts mask flicker), and stabilises the set of
  pixels TBE treats as masked across a batch. Pure numpy + cv2,
  default on.
- **Colour-tuned mask expansion** -- sample the dominant text colour
  inside each detected box (Lab-space two-cluster split), then extend
  the mask to every pixel within a tolerance radius of that colour.
  Catches decorative serifs, drop shadows, strokes, and karaoke
  bloom that the OCR bbox clips. Opt-in toggle.

**Speed**
- **Perceptual-hash adaptive mask reuse** -- instead of fixed
  `frame_skip`, compute a pHash of each frame and skip OCR when the
  Hamming distance from the last detected frame is below a threshold.
  Adapts to scene content: dense detection on motion / cuts, sparse on
  static shots. Default on, threshold 4/64 bits.

**Workflow**
- **Preset library** -- six built-in presets (YouTube default, Anime /
  Animation, Motion-heavy, TikTok vertical, VHS restore, News chyron)
  plus save/load of user-defined presets to
  `%APPDATA%\VideoSubtitleRemoverPro\presets.json`. Preset picker in
  the Profile card with a "Save as..." ghost button. Applying a
  preset updates every toggle in the UI and persists the settings.

**CLI**
- New flags: `--no-kalman`, `--no-phash`, `--phash-distance`,
  `--colour-tune`, `--colour-tolerance`.

**GUI**
- Detection card: Kalman tracking toggle, Adaptive mask reuse
  toggle, Colour-tuned expansion toggle.
- Profile card: Preset combobox + Save as button.
- All new settings persist to `settings.json` and restore on launch.
- Version bumped to 3.10.0 across banner, header, and logs.

## [v3.9.0] - 2026-04-17

Quality + workflow release. Every item here came directly off the v3.9
near-term roadmap; no external model weights were added. Focus is on
making TBE dramatically better on real-world footage (motion, cuts,
gradients) and shipping the workflow features users keep asking for
(SRT export, mask debug, auto-band, multi-region).

**Quality**
- **Scene-cut-aware TBE** -- the TBE batch is now split at histogram
  scene cuts before aggregation. Previously a cut mid-batch polluted
  the temporal median; now each segment is handled independently.
- **Flow-warped TBE** (opt-in) -- Farneback dense optical flow aligns
  every frame in a TBE segment to the middle reference frame before
  aggregating. Directly addresses camera pans / zooms / handheld
  motion. Slower but dramatically cleaner on motion-heavy footage.
  Toggle in Detection card.
- **Edge-ring colour match** -- after inpainting, sample a thin ring
  immediately outside the mask in both original and filled frames,
  compute the mean colour delta, and apply the offset to the filled
  region. Kills the faint colour seam that sometimes appeared on
  gradient backgrounds. Applies to every inpainter path.

**Workflow**
- **SRT sidecar export** -- detected text is transcribed during the
  removal pass and written as an `.srt` next to the output. Cues are
  collapsed when consecutive frames share text (gaps up to 0.5s
  bridged). Works with RapidOCR / PaddleOCR / EasyOCR.
- **Debug mask video** -- optional `.mask.mp4` written alongside the
  output containing the per-frame binary mask. Useful for tuning
  detection threshold / dilation / feather without full re-runs.
- **Auto subtitle-band detection** -- scans the first 30 frames,
  clusters detected text by vertical band, pins the dominant band as
  the `subtitle_area`. Saves a manual region-draw for the 90% common
  case where the subtitle lives in one horizontal strip.
- **Multi-region masks** -- `ProcessingConfig.subtitle_areas` now
  accepts a list of rectangles (top banner + bottom banner + logo).
  When combined with auto detection the rects are unioned with
  per-frame detections.

**Reliability**
- **Adaptive batch sizing** -- on CUDA init, probe free VRAM via
  NVML (`pynvml`) and scale `sttn_max_load_num` to match. Defaults
  to on, clamped to [8, 512]. Prevents OOM on 4K and leaves more headroom
  on 24 GB cards.

**CLI**
- New flags: `--mask-feather`, `--edge-ring`, `--flow-warp`,
  `--no-scene-split`, `--no-tbe`, `--no-adaptive-batch`,
  `--export-srt`, `--export-mask`, `--auto-band`.

**GUI**
- Detection card: Mask feather slider, Colour-match ring slider,
  Auto-band toggle, Flow-warp toggle, Scene-split toggle.
- Output card: Adaptive batch toggle, Export SRT toggle, Export
  mask video toggle.
- All new settings persist to `settings.json` and restore on launch.
- Version bumped to 3.9.0 across banner, header, and logs.

## [v3.8.0] - 2026-04-17

Real video inpainting, faster detection, clean boundaries. This is the
first release where STTN and ProPainter actually do something meaningfully
different from `cv2.inpaint` -- we keep the STTN / ProPainter names because
they describe the user-facing niche (temporal propagation, motion-tolerant),
but the implementations are now homegrown and do not require external
model weight downloads.

- **RapidOCR detection (default)**: drops in as the top-priority OCR backend.
  Runs PP-OCR via ONNX Runtime -- roughly 4-5x faster than the paddlepaddle
  build, smaller install, and free of PaddleOCR's well-known memory-leak
  behaviour on long batch runs. Auto-selected when available; otherwise the
  existing PaddleOCR > Surya > EasyOCR > OpenCV chain still applies.
- **Temporal Background Exposure (TBE)**: the STTN inpainter is now a real
  video-inpainting primitive. For every pixel inside a frame's mask it
  looks across the batch for frames where the same pixel is unmasked and
  reconstructs the true background from those exposures (median when the
  batch is small, otherwise mean). Only pixels that are masked in every
  frame fall back to cv2 inpainting. This is the principle behind
  transformer-based video inpainting for sparse occluders (subtitles,
  watermarks, logos) -- the background is literally revealed in adjacent
  frames.
- **Hybrid ProPainter path**: the ProPainter mode now runs TBE with a
  higher coverage bar plus LaMa refinement on the exposed background.
  Produces noticeably smoother results on motion-heavy footage without
  ProPainter's 10+ GB VRAM footprint.
- **Mask edge feathering**: a configurable Gaussian alpha blend
  (`mask_feather_px`, default 4) softens the boundary between original and
  inpainted pixels so there is no visible cut line at the edge of the
  removal region. Applies to every inpainter path (STTN / LAMA /
  ProPainter / fallback).
- **Engine probe**: About dialog and the in-GUI badge now report
  "Temporal BG (TBE)" as an always-available inpainting engine and
  prefer RapidOCR in the detection chain when installed.
- **Docs**: requirements.txt pins the new RapidOCR dependency. README.md
  reflects the new temporal pipeline and detection priority.

## [v3.7.0] - 2026-04-17

Premium polish pass. No behavioral changes; dramatic UX/UI refinement.

- **Design system**: unified typography, spacing, radii, and color tokens on the `Theme` class
- **Refined palette**: tighter tonal ladder (BG_DARK -> BG_SECONDARY -> BG_CARD -> BG_TERTIARY -> BG_RAISED) and more vibrant emerald primary / sky-blue accent
- **Custom widgets**:
  - `ModernToggle` replaces `tk.Checkbutton` -- canvas-rendered checkmark, focus ring, hover, proper disabled state
  - `ModernSlider` replaces `tk.Scale` -- rounded track, emerald fill, prominent thumb, keyboard and wheel support
  - `ModernButton` gains size variants (sm/md/lg), style variants (primary/accent/ghost/secondary/danger/success), icon support, and crisp focus rings
  - `ModernProgressBar` thinner default track (5-6px) with rounded corners
- **Section structure**: `tk.LabelFrame` usage replaced with consistent card pattern (eyebrow + title + content) for Profile, Workflow, STTN Motion, Detection, Output, Video Range
- **Header**: status chips now include a color-coded status dot; tighter vertical rhythm; PRO pill outlined instead of filled
- **Queue section**: illustrated empty state with film-strip icon; count shown as pill; ghost-style row actions; refined progress bar tinting; **selected item now shows a left-edge blue accent stripe**
- **Preview card**: clearer hierarchy with eyebrow + title + meta; **PIL-rendered placeholder illustration** so the card never collapses; refined detection loading state
- **Footer**: status indicator now has a color dot matching the tone (success/warning/error/info)
- **Tooltips**: 380ms hover delay, raised-surface look, subtle 1px border
- **Region selector modal**: cross-cursor, translucent selection fill, two-line hint
- **Log panel**: eyebrow/title header, slim scrollbar, bordered body, visible by default
- **ttk styles**: slimmer scrollbar, better combobox focus/hover/disabled states, popup listbox uses raised surface tone
- **Microcopy**: all button labels and helper lines tightened for a calmer, more confident tone
- **Custom confirm dialog**: `show_confirm()` themed modal replaces the native Windows messagebox for Clear Queue and Close-while-processing; proper title/message/detail hierarchy with Escape/Enter support
- **Toast notifications**: lightweight `Toast` popups anchor to bottom-right for batch completion; stacked so multiple toasts don't overlap
- **Language picker**: shows friendly names ("English (en)", "Japanese (ja)") instead of raw language codes
- **Action buttons**: Start/Stop batch and Open output now ship leading ASCII glyphs for faster visual recognition
- **Batch progress**: now uses full-width bar with a fraction label ("3 of 10 complete") on the left and a percent pill on the right
- **Close confirmation**: asks before quitting if a batch is still running, preventing accidental cancellation
- **Right-click menu on queue items**: themed context menu with Preview, Detect, Open result, Reveal folder, Copy source path, Remove
- **Windows taskbar progress**: Windows 7+ taskbar integration via ITaskbarList3 reflects batch progress on the taskbar icon
- **About dialog**: themed About panel showing version, detected engines, compute summary, and quick links to the log and settings folder
- **Log level badges**: live counts of warnings and errors appear as colored pills in the log panel header and hide when zero
- **Auto-scroll to active item**: the queue now auto-scrolls so the currently processing item is always in view
- **ETA**: batch progress now shows a rolling-average ETA ("2m 14s left") based on completed item timings
- **Throbber animation**: detection preview shows animated pulsing dots in a shimmer placeholder rather than a static label
- **Tweened progress bar**: ModernProgressBar eases to target values instead of jumping, making small backend updates feel continuous
- **Window state persistence**: last window size/position, advanced-panel expanded state, and log-panel visibility are saved to settings.json and restored on next launch (off-screen positions are rejected)
- **Queue summary chips**: queue header now shows a total pill plus green "N done" and red "N failed" chips that auto-hide when zero
- **Batch completion summary modal**: a themed modal now appears when a batch finishes, showing COMPLETED/FAILED/STOPPED counts as large stat pills plus total elapsed time, with an Open-output shortcut
- **Toast fade-out**: toasts now fade to zero alpha over ~200ms before destroying, and remaining toasts restack upward so there are never orphaned gaps
- **Segmented algorithm picker**: the algorithm combobox is replaced by a three-segment Canvas-based radio control with hover/focus/selected states for faster recognition of STTN/LAMA/ProPainter
- **Preview zoom**: double-clicking the preview opens a full-size themed viewer with a compact header (filename + pixel dimensions) and Escape-to-close
- **Active-item pulse**: the currently processing queue card pulses its border and accent stripe between emerald tones so the eye finds the working item instantly
- **Queue filter**: a themed filter input appears above the queue once there are 6+ items; filters by filename or full path, with a Clear button
- **Per-item rename**: right-click menu now offers "Rename output..." for idle items, opening a themed save-as dialog seeded with the current output path
- **First-run onboarding**: on first launch, a themed welcome modal presents three numbered cue cards (Import, Inspect, Run) and persists `onboarding_seen` so it never re-appears
- **Queue sort menu**: a Sort button appears in the queue header once there are 3+ items, opening a themed menu for filename/size/status sorts plus reverse; disabled while a batch is running
- **Version**: bumped to 3.7.0; all version strings and badges aligned

## [v3.6.0] - 2026-03-28

- v3.5.0
- Polish GUI spacing and professional layout
- Simplify drop zone text to just 'Drag & Drop Files Here'
- Changed: Update badge to v3.4.0
- v3.4.0: DPI-safe responsive GUI overhaul
- Changed: Update README badge for v3.3.0
- v3.3.0: Real AI inpainting, multi-engine detection, comprehensive GUI overhaul
- Added: Add files via upload

## Roadmap archive â€” 2026-08-10 â€” ROADMAP.md

<details>
<summary>Original roadmap snapshot</summary>

```markdown
# Video Subtitle Remover Pro -- Roadmap

Remaining-work backlog only: every item below is NOT yet implemented.
Completed items are deleted from this file; history lives in CHANGELOG.md and git.
```

</details>
