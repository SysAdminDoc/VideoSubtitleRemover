# Video Subtitle Remover Pro -- Roadmap

Actionable work only. Historical and completed roadmap material is archived in CHANGELOG.md; blocked work is kept in Roadmap_Blocked.md.

## Audit Findings -- 2026-08-11

Deep audit pass over v3.33.0. Baseline recorded before any finding was logged:
`py -3.12 -m pytest tests -q` = **1318 passed, 2 skipped, 351 subtests, 606s**;
`ruff check backend gui scripts VideoSubtitleRemover.py` = **All checks passed**;
`scripts/i18n_catalogs.py check|lint|coverage`, `scripts/generate_architecture_map.py --check`,
and `python -m backend.dependency_profiles check` all pass. **No pre-existing failures.**
Every item below is a new finding against that green baseline.

IDs continue the existing scheme (highest prior ID was RM-155).

### P2

- [ ] P2 -- RM-167: Roughly 120 user-visible strings can never be translated -- deferred `tr()` sinks are invisible to the extractor
  Category: ux
  Where: extractor `scripts/i18n_catalogs.py` (`extract_messages`); sinks at `gui/app.py:2266` (`display_message = tr(message)`), `gui/config.py:932,938` (`tr(_STATUS_LABELS.get(...))`), `gui/layout_build.py:1955` (`tr(label)` over a tuple), `gui/layout_helpers.py` (`_card_header`, `_create_slider`)
  Problem: The extractor records only string literals lexically inside `tr()`/`ntr()`/`N_()`. Several call sites pass a *variable* to `tr()` at render time, so their literals never enter `locale/vsr.pot` and no translator can ever see them. Affected: 77 of 78 literal `_update_status(...)` messages (the entire footer status line and its toasts), 8 of 9 queue status badge labels -- the most visible strings in the product -- plus card headers, every slider label and hint, preview placeholder bodies, and the queue table headers. The `i18n_catalogs.py lint` gate passes all of them because the value does route through `tr()` and because these helper signatures are not in `CAPTION_CALLS` -- the repo's own "gate that enumerates what it guards" pattern. Only the pseudo-locale ships today so no released translation is broken, but the `.pot` is the artifact the README's community-translation guide asks contributors to work from.
  Evidence: Direct check against the catalog -- `grep 'msgid "<s>"' locale/vsr.pot` returns nothing for "Queue cleared", "Batch processing started", "Needs Attention", "Complete", "Encoding quality", and "File name", all of which are user-visible. `scripts/i18n_catalogs.py check|lint|coverage` all pass, confirming the gates do not see the gap.
  Fix: Wrap the call-site literals in `N_()` (the extractor already understands it) so they enter the catalog while the deferred `tr()` still performs the lookup, and add these helper signatures to the lint's caption model so new ones cannot slip through. Re-run `i18n_catalogs.py update` last, once, per the CLAUDE.md note.
  Acceptance: `locale/vsr.pot` contains the queue status labels, `_update_status` messages, card headers, and slider labels; a lint rule fails when a new `_update_status` literal is added unwrapped.
  Confidence: Verified
  Effort: M

- [ ] P2 -- RM-174: The PyTorch LaMa path returns padded-size frames on non-mod-8 resolutions and crashes the job
  Category: correctness
  Where: `backend/inpainters/lama.py:636-641` (full-frame) and `:684-705` (tiled); contrast with the ONNX/DNN crops at `:415`, `:535`, and the batched crop at `:768`
  Problem: `SimpleLama.__call__` pads its inputs up to a multiple of 8 and never crops back. The ONNX and DNN paths crop with `bgr[:h, :w]`; `_inpaint_pytorch` does not, so for any frame whose height or width is not divisible by 8 the appended result is larger than the input. `apply_finishing` then fails -- `_edge_ring_color_correct` raises `IndexError` on the boolean index, or `_feather_blend` raises a broadcast `ValueError` -- and since neither is an OOM, `_inpaint_batch_resilient` re-raises and the whole job fails. In the tiled variant the accumulation `color_acc[ty1:ty2, tx1:tx2] += tile_out * win` sits *outside* the per-tile `try`, so it raises, is caught by the caller as "Tiled LaMa fell back to full-frame", and drops into the crashing full-frame path. Reached with `VSR_ENABLE_PYTORCH_LAMA=1` when neither ONNX Runtime nor an OpenCV-5 DNN weight is available; common resolutions are mod-8, but 480x270, cropped, and anamorphic sources are not.
  Evidence: `sed -n '630,645p' backend/inpainters/lama.py` shows `result_bgr` appended with no `[:h, :w]` slice, unlike the ONNX path. The subagent confirmed the padding behavior against the installed `simple_lama_inpainting` package source (`utils/util.py` `pad_img_to_modulo`, `models/model.py`).
  Fix: Crop the SimpleLama output to `[:h, :w]` in both the full-frame and tiled paths, mirroring the ONNX path, and move the tile accumulation inside the per-tile `try`.
  Acceptance: A 480x270 clip processes cleanly through the PyTorch LaMa tier; a regression test feeds a non-mod-8 frame through `_inpaint_pytorch` and asserts the output shape equals the input shape.
  Confidence: Verified
  Effort: S

- [ ] P2 -- RM-175: An exception between `is_processing = True` and the thread start wedges the entire UI until restart
  Category: reliability
  Where: `gui/processing_controller.py:91-123` (`_start_processing`)
  Problem: The method sets `is_processing = True` and locks settings, then calls `_prepare_batch_report_records()`, `_warn_output_quality_preflight()` (with an unguarded top-level import), and `_write_batch_preflight_plan()` -- which raises `OSError` if the output volume is unwritable or removed -- all with no try/except. If any raises, the exception lands in Tk's callback handler and `_processing_thread` is never created, so `_has_active_processing_thread()` is False while `is_processing` stays True. `_refresh_action_states` then permanently disables Start ("The batch is already running."), Clear, Retry, and Add. Nothing resets the flag; only restarting the app recovers.
  Evidence: `sed -n '91,123p' gui/processing_controller.py` shows the flag set at line 91 and the thread started at line 122, with three unguarded calls at 107-109 in between.
  Fix: Wrap the body after `is_processing = True` in try/except that rolls back `is_processing`, unlocks settings, restores the Start button, and surfaces the error via the standard status/toast path.
  Acceptance: Forcing `_write_batch_preflight_plan` to raise leaves the UI fully interactive with an error toast, and Start works on the next attempt.
  Confidence: Verified (consequence traced from code; the specific trigger needs a removable-volume repro)
  Effort: S

- [ ] P2 -- RM-176: Enter on a secondary dialog button activates it *and* closes the dialog
  Category: a11y
  Where: `gui/widgets.py:645-647` (`ModernButton._on_keyboard_activate` does not return `"break"`), toplevel `<Return>` bindings at `gui/onboarding.py:212` and `gui/quality_controller.py:354`
  Problem: Because the handler does not return `"break"`, a Return keypress continues down the bindtag chain to the toplevel binding. In onboarding, pressing Enter on "Enable auto-detect" or a starter-profile button runs the command and then immediately dismisses the dialog, so the "Selected: {profile}" confirmation is never seen. In the batch-summary dialog the default button may be "Open report", "Open log", or "Retry failed" -- Enter opens the report and closes the dialog in one keystroke. Space is unaffected, which confirms the divergence is unintended.
  Evidence: `sed -n '644,648p' gui/widgets.py` shows the handler calling `self.command()` with no return value. `gui/settings_controller.py:497` escapes the symptom only because its buttons destroy the dialog first, halting the remaining handlers.
  Fix: `return "break"` from `_on_keyboard_activate` (and from `ModernToggle`'s keyboard toggle for symmetry), or drop the redundant toplevel Return bindings.
  Acceptance: Enter on a non-default onboarding button runs only that button's command and leaves the dialog open; a keyboard test asserts the dialog is still mapped afterwards.
  Confidence: Verified
  Effort: S

- [ ] P2 -- RM-177: Stretched `ModernButton`s never redraw and have dead click zones
  Category: correctness
  Where: `gui/widgets.py:513-553` (`_draw` uses `self.width`), `:622-630` (`_on_release` hit-tests `0 <= event.x <= self.width`); no `<Configure>` handler on the class
  Problem: The button renders and hit-tests at its constructed width, but `gui/layout_build.py:163` and `:170` pack command buttons with `fill="x"` into grid columns with `weight=1` (`gui/app.py:1316-1329`). At wide or compact layout widths the canvas stretches beyond the drawn width: the graphic does not fill its slot, and a click in the stretched region sets `pressed` but never fires the command. `ModernProgressBar` already solves exactly this with a `<Configure>` binding, so the pattern exists in the file.
  Evidence: `grep -n "Configure" gui/widgets.py` shows bindings at lines 723, 1017, 1664, and 2044 -- none of them on `ModernButton`'s drawing path. `_Segment._draw` (`gui/widgets.py:1730`) has the same latent shape.
  Fix: Bind `<Configure>` on `ModernButton` to update `self.width`/`self.height` from `event.width`/`event.height` before redrawing, matching `ModernProgressBar`.
  Acceptance: Widening the window redraws the full-width command buttons to fill their column, and a click anywhere inside the button fires the command.
  Confidence: Verified (code path); worth a visual confirmation during implementation
  Effort: S

- [ ] P2 -- RM-178: The danger confirm button fails contrast on destructive actions
  Category: a11y
  Where: `gui/theme.py:56,59` (`INK_ON_DANGER = #ffffff`, `DANGER = #f87171`), applied at `gui/widgets.py:494-498`
  Problem: White on `#f87171` computes to **2.77:1** (hover state `#ef4444` gives 3.76:1) at 12-13px bold text -- below the WCAG AA 4.5:1 requirement and below even the 3:1 large-text threshold. This styling is used by `show_confirm(tone="danger")`, which backs the close-while-processing prompt (`gui/app.py:371`) and the clear-queue prompt (`gui/app.py:2208`) -- destructive confirmations where legibility matters most. Every other text/background pair in the palette passes AA comfortably.
  Evidence: Computed the WCAG relative-luminance ratios directly from the theme tokens: white on `DANGER` = 2.77, white on `DANGER_HOVER` = 3.76, and a dark ink such as `#2a0505` on `DANGER` = 6.76. Confirmed the rest of the palette is clean in the same pass (TEXT_MUTED >= 6.3:1, TEXT_SECONDARY >= 10.0:1, focus ring ~5.0:1).
  Fix: Use a dark ink on the red fill (`#2a0505` or similar) for filled danger buttons, or darken `DANGER` for the filled variant while keeping the light tone for danger *text* on dark surfaces.
  Acceptance: The danger confirm button's text/fill pair computes >= 4.5:1 in both normal and hover states; a token contrast test guards the pair.
  Confidence: Verified (computed)
  Effort: S

- [ ] P2 -- RM-180: The uninstaller can delete a user-chosen pre-existing directory wholesale
  Category: reliability
  Where: `installer/vsr.nsi:175` (`RMDir /r "$INSTDIR"`)
  Problem: `MUI_PAGE_DIRECTORY` lets the user install anywhere, including an existing directory that holds unrelated files (a `C:\Tools` or a shared apps folder). Uninstall then runs `RMDir /r "$INSTDIR"` unconditionally and destroys everything in that directory, not just the app's own files -- the exact scenario the NSIS documentation warns against. There is also no running-app mutex check in `un.onInit` (install checks it, uninstall does not), so uninstalling while the app runs half-deletes the tree.
  Evidence: Read `installer/vsr.nsi:175` in context; the install section checks the mutex at lines 98-101 but `Section "Uninstall"` has no equivalent guard.
  Fix: Verify a sentinel (`$INSTDIR\VideoSubtitleRemoverPro.exe`) exists before the recursive delete, or delete the known file/directory manifest explicitly and then `RMDir` the (now empty) directory; add the mutex check to `un.onInit`.
  Acceptance: Uninstalling from a directory that also contains unrelated files leaves those files intact; uninstalling while the app is running refuses with a clear message.
  Confidence: Verified (code path certain; damage conditional on the user's install-dir choice)
  Effort: M

- [ ] P2 -- RM-182: AMD/Intel GPU detection depends on `wmic`, which is absent from current Windows 11
  Category: reliability
  Where: `setup.py:329-345`
  Problem: Non-NVIDIA detection shells out to `wmic path win32_VideoController get name`. WMIC is a disabled-by-default Feature-on-Demand on clean Windows 11 24H2 and later and is slated for removal. On those machines the call fails, the broad `except` swallows it, and AMD/Intel systems silently receive the CPU dependency profile instead of DirectML -- with no message distinguishing "no GPU found" from "the probe tool is missing".
  Evidence: Read the detection block; the failure path has no diagnostic distinguishing the two cases. Not reproduced locally because this LTSC host still ships `wmic`.
  Fix: Query `Get-CimInstance Win32_VideoController` via `powershell -NoProfile` (or DXGI through ctypes) with `wmic` as a fallback, and log a distinct warning when every probe fails so the user knows detection was inconclusive rather than negative.
  Acceptance: On a machine without `wmic`, an AMD or Intel GPU is still detected and the DirectML profile is offered; a failed probe logs a specific warning.
  Confidence: Likely (Microsoft-documented deprecation; not reproduced on this host)
  Effort: S

- [ ] P2 -- RM-183: The QA pseudo-locale is offered to end users as a selectable UI language
  Category: ux
  Where: `gui/layout_build.py:1353-1358` (the picker is built from `available_catalogs()`), `VideoSubtitleRemoverPro.spec:32` (`('locale', 'locale')` bundles it into the frozen build), `locale/qps-Ploc/`
  Problem: `qps-Ploc` is a pseudo-localization catalog -- deliberately mangled accented text used to check for truncation and unwrapped strings. It is the *only* compiled catalog in the repo, and the language dropdown lists every entry `available_catalogs()` returns, so the shipped UI offers exactly three choices: System, English, and `qps-Ploc`. A user who picks the third gets a garbled interface with no indication it is a test artifact, and the `locale/` directory is bundled into the frozen build, so this reaches released users.
  Evidence: `py -3.12 -c "from backend.i18n import available_catalogs; print(available_catalogs())"` returns `('qps-Ploc',)`. Reading the catalog confirms mangled accented text. `VideoSubtitleRemoverPro.spec:32` includes `locale` in `datas`.
  Fix: Filter `qps-` prefixed catalogs out of the user-facing picker (keep them reachable through an env var or a debug flag for QA), or exclude the pseudo-locale from the frozen bundle.
  Acceptance: The language dropdown shows only System and English on a build with no real translations; `VSR_PSEUDO_LOCALE=1` (or equivalent) still exposes it for QA.
  Confidence: Verified
  Effort: S

- [ ] P2 -- RM-185: README links two documentation files that are not published (404 on GitHub)
  Category: docs
  Where: `README.md:1222` (`docs/edge_case_corpus.md` in the Project Structure tree), `README.md:1247` (`[docs/archive/](docs/archive/)`)
  Problem: Neither path is tracked. Commit `e1221da` ("chore: gitignore markdown files") removed `edge_case_corpus.md` from git, and the `.gitignore:47` `*.md` rule keeps it and everything under `docs/archive/` out. Both links 404 for anyone reading the README on GitHub. `edge_case_corpus.md` is specifically the *community contribution* handbook that asks external users to open Discussions and submit clips, so its unpublished state defeats its own purpose.
  Evidence: `git ls-files docs/` returns only `docs/architecture.md`. `git check-ignore -v docs/edge_case_corpus.md` returns `.gitignore:47:*.md docs/edge_case_corpus.md`.
  Fix: `git add -f docs/edge_case_corpus.md` (and `docs/archive/` if it should be public), or remove both README references. Given the file's contributor-facing purpose, publishing it is the better call.
  Acceptance: Every documentation link in README resolves on GitHub; a docs test asserts each relative README link points at a tracked path.
  Confidence: Verified
  Effort: S

### P3

- [ ] P3 -- RM-186: Resuming a job produces a silently incomplete SRT export, translation source, and quality-report ROI
  Category: correctness
  Where: `backend/processor.py:2141` (`self._srt_entries = []` reset per call), `:2545-2551` (the restart-from-zero guard exists only for `export_mask_video`), `:1873-1882` (SRT entries collected only for frames OCR'd this run), `backend/_finalize_mixin.py:132-147`
  Problem: A paused or crashed job resumed at frame N skips OCR for frames 0..N-1, so the exported `.srt` is missing every cue before the resume point -- and when OCR-sourced translation is enabled, that incomplete set is what gets translated and burned into the final video. `_quality_mask_bbox`/`_seam_scores` are partial for the same reason, degrading the quality-report ROI to whole-frame. Mask export received an explicit "restart from frame zero" guard for this exact completeness problem; SRT export and translation did not.
  Evidence: `sed -n '2540,2556p' backend/processor.py` shows the guard scoped to `if self.config.export_mask_video and resume_frame_count > 0`. `grep -n "_srt_entries" backend/processor.py` shows the per-call reset at 2141 and collection at 1874, with no resume-aware handling.
  Fix: Extend the restart-from-zero guard to cover `export_srt` and OCR-sourced translation, or persist `_srt_entries` in the pause checkpoint so a resume rebuilds the full cue list.
  Acceptance: Pausing and resuming a job with `export_srt` enabled produces the same cue count as an uninterrupted run.
  Confidence: Verified
  Effort: S

- [ ] P3 -- RM-187: Burned translated subtitles are offset by `time_start` on time-ranged jobs
  Category: correctness
  Where: `backend/_srt_mixin.py:145-159` (cue times computed source-absolute), `:202-217` (translation source written with `offset_frames=start_frame`), `backend/_finalize_mixin.py:601-641` (that file burned onto the trimmed output)
  Problem: With `time_start > 0` and translation enabled, the cues are written on the source clock while the trimmed output starts at zero, so every burned subtitle appears `time_start` seconds late. Whisper segments are source-absolute too. The source-absolute convention may be deliberate for the standalone `.srt` sidecar (the `offset_frames` parameter looks intentional), but the burn path is unambiguously wrong.
  Evidence: Traced the offset from `_srt_mixin.py:202-217` into the burn call in `_finalize_mixin.py:601-641`; the output being burned is the trimmed encode.
  Fix: Write the translation-burn source with `offset_frames=0` (output-relative) while leaving the sidecar's convention unchanged.
  Acceptance: A job with `--start 30` plus translation burns cues aligned to the output's own timeline.
  Confidence: Likely (burn misalignment concrete; sidecar convention arguably intentional)
  Effort: S

- [ ] P3 -- RM-188: TBE `min_coverage` is never clamped to segment length, so short scene-cut segments silently fall back entirely to cv2
  Category: correctness
  Where: `backend/inpainters/_common.py:440-453` (`_tbe_single_segment` coverage gate), callers `sttn.py:40`, `propainter.py:66`
  Problem: For a segment of n frames the maximum achievable coverage of a masked pixel is n-1. With the default `tbe_min_coverage=3` (ProPainter uses `max(2, cfg+1)` = 4), any scene-cut segment of 3 frames or fewer has `coverage < min_coverage` everywhere, so the entire segment silently goes through cv2.inpaint. Scene-cut splitting is on by default, so rapid-cut footage and batch tails routinely produce such segments. This contradicts the documented contract ("falls back to cv2 only for pixels masked in every frame of the batch") with no log line. Note `_tbe_single_segment` already special-cases n==1 but not 2..min_coverage.
  Evidence: Read the gate at `_common.py:441-443` (`has_exposure = mask_bool & (coverage >= min_coverage)`) and the default at `backend/config.py:215`. Pure arithmetic on segment length.
  Fix: Compute `effective_min = min(min_coverage, max(1, n - 1))` inside `_tbe_single_segment`, and log at debug when the clamp engages.
  Acceptance: A 3-frame scene segment uses temporal recovery where exposure exists rather than falling back wholesale to cv2; a unit test asserts the clamp.
  Confidence: Verified
  Effort: S

- [ ] P3 -- RM-189: TBE flow-warp evaluates the coverage gate in reference-frame coordinates but applies it in frame-t coordinates
  Category: correctness
  Where: `backend/inpainters/_common.py:403-448`
  Problem: With `flow_warp=True`, `coverage` is computed from masks warped into the reference frame's coordinate system, but the per-frame apply loop intersects it with frame t's *unwarped* mask while the fill values are warped back to t. The gate and the values therefore live in different coordinate spaces. Under exactly the motion this feature exists for, pixels near the moving mask boundary are misclassified -- some receive background values whose true scene point had no exposure, others fall to cv2 unnecessarily. Localized artifacts, no crash.
  Evidence: Read the coverage computation at `:406` and the gate at `:443`; the warp-back of `bg_for_t` at `:435` has no counterpart for `coverage`.
  Fix: Warp `coverage` to frame t alongside the background (nearest-neighbor is sufficient for a threshold map), or derive the per-frame gate from warped-back masks.
  Acceptance: A synthetic panning fixture shows the gate and fill agreeing on the same pixels; artifact count near the mask boundary drops.
  Confidence: Verified (code path); visual magnitude worth confirming
  Effort: M

- [ ] P3 -- RM-190: ProPainter's LaMa refinement silently no-ops on non-mod-8 resolutions while still reporting itself as active
  Category: correctness
  Where: `backend/inpainters/propainter.py:87-102`
  Problem: The same SimpleLama padding as RM-174 makes `bgr` larger than `inpainted`, so the `cv2.addWeighted` blend raises and the per-frame `except` appends the unrefined TBE frame. On a non-mod-8 video *every* frame takes that path: "TBE + LaMa refinement" degrades to plain TBE while `backend_name` -- and therefore the job's execution provenance -- still claims refinement ran. The per-frame warning is generic ("refinement failed") rather than naming the resolution cause. Secondary: the mask handed to SimpleLama is the raw, possibly soft matte, which SimpleLama binarizes at `>0`, inconsistent with the repo's `_binarize_mask` threshold of `>=128`.
  Evidence: Read the blend and its exception handler; the padding behavior is the same one confirmed for RM-174.
  Fix: Crop the LaMa output to frame size before blending (shares the RM-174 fix), pass `_binarize_mask(mask)`, and make the warning name the cause.
  Acceptance: ProPainter refinement runs on a 480x270 clip; provenance reports plain TBE when refinement genuinely could not run.
  Confidence: Verified
  Effort: S

- [ ] P3 -- RM-191: Mask reuse ignores time-gated manual corrections, so a correction that starts or ends mid-scene is applied to the wrong frames
  Category: correctness
  Where: `backend/processor.py:1711-1745` (the phash, keyframe, and frame-skip reuse branches), `:1915-1921`, `backend/mask_corrections.py:104-142`
  Problem: `state.last_mask` caches the mask *after* the corrections active at that earlier frame were composed into it. The phash-reuse path (enabled by default), the keyframe-reuse path, and the `frame_skip` path all re-attach that stale mask verbatim. Corrections are explicitly time-gated by seconds or frame index, so on a static scene -- precisely when phash skipping fires -- a correction that starts mid-scene is ignored and one that ends mid-scene keeps being applied, until the next real detection frame. The loop guards reuse against `ctx.timed_region_spans` (timed *regions*) but has no equivalent check for timed *corrections*.
  Evidence: `sed -n '1711,1745p' backend/processor.py` shows all three reuse branches gated only on `not ctx.timed_region_spans`; no correction-span check appears between the branches.
  Fix: Suppress mask reuse across any frame range where an active correction's span boundary falls, or cache the pre-correction mask and re-run `_apply_manual_mask_corrections` per frame on the reused base.
  Acceptance: A correction bounded to frames 100-200 applies to exactly those frames regardless of phash reuse; a test asserts the boundary.
  Confidence: Verified (code path); visual impact needs a repro clip
  Effort: M

- [ ] P3 -- RM-192: One backslash in a user OCR-fix value disables every fix for the job
  Category: correctness
  Where: `backend/ocr_fix.py:114-115` (`re.sub(pattern, dst, text)`), caught at `backend/_srt_mixin.py:50-55`
  Problem: For whole-word keys the replacement is passed as a `re.sub` *template*, so a user value containing a backslash (`\g`, `\1`, or any stray backslash) raises `re.error` or splices in group text. The exception is caught one level up, which returns the raw text -- so a single bad entry in `%APPDATA%\...\ocr_fix\{lang}.json` silently disables *all* fixes including the bundled ones, per cue, with only a log warning. The module docstring promises that "a malformed file must never break SRT export"; load time is guarded, apply time is not.
  Evidence: `sed -n '105,120p' backend/ocr_fix.py` shows `re.sub(rf"(?<!\w){re.escape(src)}(?!\w)", dst, text)` -- the key is escaped, the value is not.
  Fix: Pass a function replacement (`lambda m: dst`) so the value is treated literally, matching the `str.replace` behavior of the non-word branch.
  Acceptance: A user fix list containing a backslash value applies that replacement literally and does not disable other fixes; a regression test covers it.
  Confidence: Verified
  Effort: S

- [ ] P3 -- RM-193: The subtitle-burn path mis-escapes single quotes, so any subtitle path containing an apostrophe fails
  Category: correctness
  Where: `backend/post_restore.py:273-279`
  Problem: `.replace("'", "'\\\\''")` emits the 5-character sequence `'\\''` (close-quote, escaped backslash, empty quotes) where the intended close-escape-reopen sequence is `'\''`. FFmpeg's filtergraph parser consumes the backslash as an escape and drops the quote, so `O'Brien.srt` becomes `O\Brien.srt` and the burn fails with a file-not-found. It fails safe -- no injection, since the backslash is consumed -- but it is reported as a generic burn failure with no hint at the cause. The comment directly above describes the correct sequence, so this is a transcription slip.
  Evidence: `sed -n '268,282p' backend/post_restore.py` shows the four-backslash Python literal; the comment says "close-escape-reopen".
  Fix: One fewer escape level -- `.replace("'", "'\\''")` in source.
  Acceptance: Burning a subtitle file whose path contains an apostrophe succeeds; a test asserts the produced filtergraph string.
  Confidence: Verified
  Effort: S

- [ ] P3 -- RM-194: `warp_mask_with_flow` warps the mask in the wrong direction, and nothing calls it
  Category: correctness
  Where: `backend/karaoke_flow.py:32-51`
  Problem: `calcOpticalFlowFarneback(prev, next)` yields prev-to-next displacement, but `remap(mask, grid + flow)` computes `warped(p) = mask(p + flow(p))`, which reconstructs the *prev*-aligned view -- the mask moves opposite to the text. The RM-43 intent (extend the mask along the motion so the inpaint covers the moving karaoke line) is therefore inverted: the union covers the wrong side and misses where the text went. Impact is capped because grep shows no production caller anywhere in `backend/` or `gui/` -- the feature is unwired.
  Evidence: Read the flow computation and remap; the direction analysis is arithmetic. The only test (`tests/test_hardening_detection.py:216-224`) feeds two all-zero frames and asserts shape only, so direction was never covered.
  Fix: Compute the flow as `calcOpticalFlowFarneback(next_gray, prev_gray)` and remap with that, add a synthetic-translation direction test, and either wire the helper into the karaoke path or delete it.
  Acceptance: A synthetic fixture translating a block by a known offset produces a mask displaced in the same direction as the motion.
  Confidence: Verified (math)
  Effort: S

- [ ] P3 -- RM-195: Empty RapidOCR output raises a caught `TypeError` per frame, flooding the log with ERROR lines
  Category: reliability
  Where: `backend/detection.py:571-622` (`_rapid_output_to_text_boxes`), caught at `:539-545`
  Problem: For RapidOCR 2.x and newer with no detections, the structured fields are all `None`, so the parser skips the structured branch and falls through to `for entry in results:`. `RapidOCROutput` defines `__len__` but neither `__iter__` nor `__getitem__`, so the iteration raises `TypeError`. The result is functionally correct -- the caller catches it and returns `[]` -- but it logs "RapidOCR text detection error" at ERROR level for **every subtitle-free frame** whenever the language filter or SRT-with-filter is active, burying real errors and inflating the log panel's error badge. The sibling parsers `_rapid_output_to_boxes`/`_conf` have the `if not results: return []` guard this one lacks.
  Evidence: Read all three parsers; only the text variant omits the early return. `RapidOCROutput`'s shape was confirmed against the installed package.
  Fix: Add the same length-based empty guard before the iteration fallback.
  Acceptance: Processing a subtitle-free clip with the language filter enabled produces no ERROR log lines; a test feeds an empty `RapidOCROutput` and asserts a clean empty result.
  Confidence: Verified
  Effort: S

- [ ] P3 -- RM-196: A stale `VSR_LAMA_ONNX` path silently replaces the four-tier LaMa stack with a cv2-only backend
  Category: correctness
  Where: `backend/inpainters_onnx.py:335-343` (`maybe_register` shadows the `"lama"` slot on env-var presence alone), `:204-205` (the only fallback is `_cv2_fallback`)
  Problem: The registry slot is overridden purely because the environment variable is non-empty, before any path or session validation. A user whose `VSR_LAMA_ONNX` points at a moved or deleted file loses the built-in `LAMAInpainter` with its auto-discovery, OpenCV-DNN, and PyTorch tiers, and silently gets cv2 -- the weakest possible backend. The only signal is a "ONNX model not found" warning that never mentions that a better backend was displaced.
  Evidence: Read `maybe_register` and the fallback chain; the shadowing happens unconditionally on env presence.
  Fix: Validate the path (and ideally the session) before shadowing the slot, or fall back to constructing `LAMAInpainter` when session creation fails.
  Acceptance: An invalid `VSR_LAMA_ONNX` leaves the built-in LaMa tiers in place and logs that the override was ignored.
  Confidence: Verified
  Effort: S

- [ ] P3 -- RM-197: A provider-selection failure silently pins inference to CPU with no log at any level
  Category: reliability
  Where: `backend/inpainters/lama.py:224-235` (`_try_onnx_session`)
  Problem: If `_providers_for_device`, the opset audit, or the CUDA DLL preload raises, the bare `except Exception: providers = ["CPUExecutionProvider"]` silently pins a CUDA or DirectML user to CPU. The session then loads fine, and while `backend_name` honestly reports CPU, nothing tells the user *why* their GPU was skipped. Every other degradation path in this file logs.
  Evidence: Read the except block -- it has no logging statement, unlike the surrounding fallbacks.
  Fix: Log the caught exception at warning level naming the requested device and the fallback.
  Acceptance: Forcing a provider-selection failure produces a warning naming the reason; a test asserts the log record.
  Confidence: Verified
  Effort: S

- [ ] P3 -- RM-198: Execution provenance reports TBE when the cv2 path actually ran
  Category: correctness
  Where: `backend/inpainters/sttn.py:29-52` (`backend_name` keys only off `config.tbe_enable`), same pattern in `backend/inpainters/propainter.py:31-35`
  Problem: For single-frame input -- image processing, or a single-frame tail batch -- `inpaint()` takes the `len(frames) > 1` false branch and runs cv2.inpaint, but `backend_name` still reports "TBE (temporal background exposure)". RM-147 exists specifically to record which implementation actually ran rather than which mode was requested, so this defeats its purpose for exactly the cases where the difference matters most.
  Evidence: Read both `backend_name` implementations; neither consults the single-frame branch. `LAMAInpainter` demonstrates the correct pattern by tracking the last-run path on the instance.
  Fix: Record the executed path in an instance attribute during `inpaint()` and report that, mirroring `LAMAInpainter`.
  Acceptance: A single-frame job's provenance records cv2, not TBE; a test asserts the reported backend for a one-frame batch.
  Confidence: Verified
  Effort: S

- [ ] P3 -- RM-199: The external inpainter silently keeps the subtitled frame when a per-frame output is missing or malformed
  Category: reliability
  Where: `backend/inpainters/external.py:239-256`
  Problem: When the external command exits 0 but a numbered output PNG is missing, unreadable, or the wrong shape, the loop substitutes the original frame with a zeroed mask -- the subtitle stays in the output with **no log whatsoever**. Batch-level failures at lines 226-237 do log; the per-frame path does not. A partially working external tool therefore produces flickering subtitle reappearance that nothing surfaces.
  Evidence: Read the substitution branch; it has no logging and no counter.
  Fix: Count substitutions and log a warning listing the affected frame indices, and consider failing the batch past a threshold.
  Acceptance: An external tool that drops frames produces a warning naming them; a test asserts the count.
  Confidence: Verified
  Effort: S

- [ ] P3 -- RM-200: Tiled LaMa's cosine taper is exactly zero at tile edges, leaving a one-pixel uninpainted line at frame borders
  Category: correctness
  Where: `backend/inpainters/lama.py:454-476`, same pattern at `:570-597` and `:692-716`
  Problem: `taper = 0.5 - 0.5*cos(linspace(0, pi, ramp))` starts at exactly 0.0, so the first and last row/column of each tile carry zero weight. Interior tile boundaries are rescued by the overlapping neighbor, but where the mask touches the frame edge -- a full-width bottom subtitle band flush with the frame bottom after dilation clipping -- the only covering tile contributes zero, `blend_mask` is False, and the original subtitle pixel survives as a one-pixel line along the border, which feathering then blends with itself.
  Evidence: Arithmetic on the taper expression; the border case has no second tile to compensate.
  Fix: Sample the taper at half-steps so the first weight is greater than zero, or clamp the window to a small epsilon on tile faces that coincide with the frame border.
  Acceptance: A mask flush with the frame bottom is fully inpainted; a test asserts no residual masked pixels on the border row.
  Confidence: Verified (arithmetic)
  Effort: S

- [ ] P3 -- RM-201: `_verify_pytorch_weights` verifies the first globbed weight file, not the one SimpleLama actually loads
  Category: security
  Where: `backend/inpainters/lama.py:324-346`
  Problem: The manifest check globs the candidate weight directories for `big-lama*.pt` and returns after the first hit, but SimpleLama resolves its own path from the `LAMA_MODEL` environment variable or the torch-hub cache, which need not be the first glob hit. A corrupt or substituted weight in the location actually loaded can therefore pass verification because a different clean copy elsewhere was the file checked -- the repo's own "check wired to the wrong data source" class.
  Evidence: Read the glob loop; the `return` sits inside the inner loop, and nothing consults `LAMA_MODEL` or the hub cache path.
  Fix: Resolve the same path SimpleLama will use (`os.environ.get("LAMA_MODEL")`, else the hub cache path for the model URL) and verify that specific file.
  Acceptance: Corrupting the weight SimpleLama loads fails verification even when another valid copy exists on disk.
  Confidence: Verified
  Effort: S

- [ ] P3 -- RM-202: Diffusion adapters coerce float model output to black frames and silently truncate on frame-count mismatch
  Category: correctness
  Where: `backend/inpainters_diffusion.py:544-562` (`_coerce_adapter_frames`), plus `:190-192`, `:217-220`, `:936-944`, `:967-970`, `:1276-1289` (backends that bypass it)
  Problem: Two related defects. (a) `_coerce_adapter_frames` does `np.asarray(frame).astype(np.uint8)` with no dtype or range check, so a model returning float frames in [0,1] -- the most common diffusion convention -- yields all-zero/one black frames that pass feathering and ship as "inpainted" inside the mask. (b) `_PropainterRealBackend`, `_DiffuEraserBackend`, `_CocoCoBackend`, `_EraserDitBackend`, and `_VoidBackend` pass `list(out)` straight to `apply_finishing`, whose `zip` silently truncates on a count mismatch, so a model returning T-1 frames drops frames with no error. VACE, VideoPainter, and FloED do validate, which shows the intended contract. All are opt-in scaffolds, so this lands only on users who wire a real upstream.
  Evidence: Read the coercion helper and all five bypassing backends.
  Fix: Scale-and-round float outputs (or raise on unexpected dtype) in `_coerce_adapter_frames`, and route all five backends through it.
  Acceptance: A stub adapter returning float [0,1] frames produces correct pixels; one returning too few frames raises rather than truncating.
  Confidence: Verified (code); float-output likelihood high
  Effort: S-M

- [ ] P3 -- RM-203: SAM 2 refinement lets a later overlapping box erase an earlier box's accepted pixels
  Category: correctness
  Where: `backend/segmentation.py:130-161`
  Problem: For each box the code zeroes the whole rectangle in `refined` before OR-ing the current box's gated SAM mask back in. With two overlapping detections -- stacked two-line subtitles, the dense-text case SAM refinement targets -- the zeroing wipes pixels already accumulated from the previous overlapping box, and they return only if the second box's segmentation happens to cover them. Net effect is under-masking.
  Evidence: Read the per-box loop; the zeroing is unconditional over the full rectangle.
  Fix: Precompute all zeroed rectangles first and then OR every SAM mask in, or accumulate SAM results into a separate buffer that is merged once at the end.
  Acceptance: Two overlapping boxes produce a refined mask that is the union of both segmentations; a test asserts it.
  Confidence: Verified (logic); frequency needs a repro clip
  Effort: S

- [ ] P3 -- RM-204: A non-ASCII filename crashes the CLI when stdout is redirected
  Category: reliability
  Where: `backend/cli.py:1791` (`print(f"\n[batch] ({i}/{len(inputs)}) {src.name}")`) and the other bare `print` sites at 546, 1505, 1507, 1558, 1655-1660, 1961, 1976
  Problem: On Windows a redirected or piped stdout uses the locale codec (cp1252), so a CJK, Cyrillic, or emoji filename raises `UnicodeEncodeError` and aborts the batch with a traceback. Reports are still written by the `finally` block, but the remaining items are cancelled. This tool's core audience processes CJK-subtitled media, and `--json`/`--json-log` imply piped usage. The GUI's isolated path is immune only because `gui/job_supervisor.py:171` sets `PYTHONIOENCODING=utf-8` for its child; the direct CLI has no equivalent.
  Evidence: Reproduced on this machine -- `py -3.12 -c "print('\u6f22\u5b57.mp4')"` piped through `cat` reports `stdout encoding: cp1252` and raises `UnicodeEncodeError`. The batch print at `cli.py:1791` interpolates `src.name` directly.
  Fix: Call `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` (and the same for stderr) at the top of `main()`.
  Acceptance: A batch containing a CJK-named input completes with output redirected to a file.
  Confidence: Verified (reproduced empirically)
  Effort: S

- [ ] P3 -- RM-205: SIGINT is permanently rebound to "pause", so the CLI cannot be force-interrupted
  Category: reliability
  Where: `backend/cli.py:1600-1608` (`_request_pause` installed as the SIGINT handler, never restored or escalated); dead `except KeyboardInterrupt` branches at 1874, 1968, 1990
  Problem: The first Ctrl+C requests a pause. A second does nothing, because the flag is already set -- so if the pipeline wedges between `pause_check` polls (a native call, a hung ffmpeg), the process cannot be interrupted from the keyboard. It also makes all three `except KeyboardInterrupt` branches in the function unreachable dead code. Pause-on-first-Ctrl+C is a reasonable design; the missing piece is escalation.
  Evidence: `sed -n '1598,1612p' backend/cli.py` shows the handler setting a flag with no re-arm, restore, or second-press branch.
  Fix: On the second invocation, restore `signal.default_int_handler` (or raise `KeyboardInterrupt`) so a second Ctrl+C hard-cancels, and remove or make reachable the dead handlers.
  Acceptance: Two Ctrl+C presses terminate the CLI even when the pipeline is not polling.
  Confidence: Verified
  Effort: S

- [ ] P3 -- RM-207: The completion checkpoint ignores processing settings, so a rerun with different options is skipped
  Category: correctness
  Where: `backend/resume_checkpoint.py:38-53` (`_checkpoint_key`), `:50-53` (`_checkpoint_is_done`), used at `backend/cli.py:1628-1630, 1775-1779`
  Problem: The `.done` marker key fingerprints only input path, size, mtime, and output path -- not the configuration. Re-running the same input to the same output with `--mode lama` after an STTN run prints `[skip] name (checkpoint)` and leaves the stale output in place. The *pause* checkpoint carefully validates a `config_hash`, so the two checkpoint kinds disagree about whether settings are part of a job's identity.
  Evidence: `sed -n '38,53p' backend/resume_checkpoint.py` shows the fingerprint string with no config component, while `_validation_warning` (line ~294) checks `config_hash` for pause checkpoints.
  Fix: Fold `config_fingerprint(config)` into `_checkpoint_key`, or store the hash inside the marker and treat a mismatch as not-done.
  Acceptance: Re-running with a different mode reprocesses instead of skipping; `--no-resume` remains unnecessary for that case.
  Confidence: Verified
  Effort: S

- [ ] P3 -- RM-208: `--skip-existing` drops distinct inputs that share a filename stem
  Category: correctness
  Where: `backend/batch_report.py:106-120` (`choose_batch_output_path` bypasses `reserved_outputs` when skip-existing is on), `backend/cli.py:1767-1804`
  Problem: With `--skip-existing`, the canonical output path is returned deliberately so an existing file can be detected. But a recursive pattern matching `a/video.mp4` and `b/video.mp4` maps both to `out\video_no_sub.mp4`; once the first is processed, the second sees `output_exists=True`, is reported `skipped-existing` ("Output already exists"), and its content is never processed. Without `--skip-existing` the collision-proof "(2)" naming handles this correctly.
  Evidence: Read `choose_batch_output_path` -- the `if skip_existing: return base` branch precedes the reservation logic; the docstring documents the re-run intent but not this collision case.
  Fix: When skip-existing is on and the canonical path is already in this run's `reserved_outputs`, fall through to collision-proof naming instead of skipping.
  Acceptance: A recursive batch with two same-stem inputs produces two outputs; a test covers the combination.
  Confidence: Verified
  Effort: S

- [ ] P3 -- RM-209: Killing an isolated worker orphans its ffmpeg grandchildren (no job-object containment)
  Category: reliability
  Where: `gui/job_supervisor.py:358-386` (`terminate()`/`kill()` target only the direct child), `gui/processing_controller.py:744-752`; no Windows Job Object anywhere in the repo
  Problem: Terminating the worker -- including the 30-second cancel escalation -- does not reach its children. An in-flight audio mux or post-restore ffmpeg survives: the FFV1 stdin writer gets EOF and finalizes, but file-input muxes run to completion, holding the output and temp files locked. An immediate retry of the "cancelled" item can then fail with a sharing violation, and the output file can materialize *after* the item was reported cancelled. The same gap means a GUI crash leaves the worker and its children running to completion, since `job_worker.py`'s control reader deliberately treats a missing control file as "no request" and has no parent-liveness check.
  Evidence: Read both termination paths; a repo-wide search finds no Job Object usage.
  Fix: Assign the worker to a Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` at spawn, which also solves the parent-death case.
  Acceptance: Cancelling an item during the mux stage leaves no ffmpeg process running and no locked output file; an immediate retry succeeds.
  Confidence: Verified (containment absent); downstream consequence likely
  Effort: M

- [ ] P3 -- RM-210: The supervisor's scratch directory leaks permanently when cleanup races a killed child
  Category: reliability
  Where: `gui/job_supervisor.py:260-266` (`_cleanup_scratch`), `:377-386` (kill without a following wait on the deadline path)
  Problem: On the deadline path `kill()` is issued without a subsequent `wait()` -- the final wait only runs when `deadline is None` -- and `_cleanup_scratch()` runs immediately. A not-yet-reaped child still holding `request.json`, `control.json`, or preview PNGs makes `TemporaryDirectory.cleanup()` raise a Windows sharing violation, which `_cleanup_scratch` swallows *and then drops the reference* (`self._owned_scratch = None`). The `%TEMP%\vsr_job_*` directory is never retried and leaks for good.
  Evidence: Read both methods; the reference is cleared in the same block that swallows the error.
  Fix: Always wait briefly for reap after a kill before cleanup; on failure keep the reference for a later retry (or use `ignore_cleanup_errors=True` plus a startup sweep of stale `vsr_job_*` directories).
  Acceptance: Killing a worker mid-job leaves no `vsr_job_*` directory behind after the next cleanup pass.
  Confidence: Verified (logic); occurrence likely under kill
  Effort: S

- [ ] P3 -- RM-211: The retry loop leaks watchdog threads, and a stale watchdog recreates deleted scratch directories
  Category: reliability
  Where: `gui/processing_controller.py:589-657`, `:714-756` (`_watch_isolated_controls`), `backend/job_worker.py:137-147` (`write_control_file` does `mkdir(parents=True)`)
  Problem: Each retry attempt starts a new watchdog thread, but the previous one exits only when the item reaches a terminal status -- during retries the item is `LOADING`, so old watchdogs keep polling defunct supervisors at 10 Hz. If the user cancels during attempt N, every stale watchdog calls `cancel()` on its old supervisor, and `write_control_file`'s `mkdir` **recreates the already-cleaned scratch directory**, leaking one orphan directory per prior attempt.
  Evidence: Read the retry loop and the watchdog exit condition; read the `mkdir` in `write_control_file`.
  Fix: Signal the previous watchdog through an `threading.Event` before starting a new one, and drop the `mkdir` from `write_control_file` (the parent creates the directory at spawn).
  Acceptance: A job that retries three times and is then cancelled leaves exactly zero orphan scratch directories and no live watchdog threads.
  Confidence: Verified
  Effort: S

- [ ] P3 -- RM-212: The wall-clock timeout discards an already-received successful result
  Category: correctness
  Where: `gui/job_supervisor.py:400-413` (`_build_outcome` checks `_timed_out` before `_result`), `:242` (`_close_streams` closes stdout while `_pump_events` iterates it)
  Problem: In the fresh 3.33 timeout code, if the child emitted `result: complete` -- output written, checkpoint cleaned -- but its stdout had not reached EOF when the deadline expired, the job is reported as `worker_timeout` even though it succeeded. The caller then retries or fails an item whose output already exists. Related noise: closing stdout under the iterating pump thread raises `ValueError` inside the daemon thread, producing a threading excepthook traceback on stderr.
  Evidence: Read `_build_outcome` -- the `_timed_out` branch returns before `_result` is consulted.
  Fix: Prefer a received result over the timeout classification (optionally annotating that the worker was terminated late), and catch `ValueError` around the pump's iteration.
  Acceptance: A worker that publishes a result and then hangs is reported as complete, not `worker_timeout`; no threading traceback appears on stderr.
  Confidence: Verified (narrow trigger)
  Effort: S

- [ ] P3 -- RM-213: Per-item cancel is a no-op for soft-subtitle remux items
  Category: correctness
  Where: `gui/processing_controller.py:420-425` (`cancel_check=self.cancel_event.is_set`)
  Problem: The remux path polls only the global cancel event. `_cancel_queue_item` (`gui/app.py:1982-2001`) sets `item.cancel_requested` and shows "Stopping <file>", but nothing in the remux path reads that flag, so the remux runs to completion and the item finishes COMPLETE after the UI announced it was stopping. Every other processing path checks both conditions.
  Evidence: `sed -n '415,430p' gui/processing_controller.py` shows the single-condition check; `grep -n "cancel_requested"` shows the two-condition pattern at lines 554, 616, 638, 645, 1022, 1052.
  Fix: `cancel_check=lambda: self.cancel_event.is_set() or item.cancel_requested`.
  Acceptance: Cancelling a soft-subtitle item stops the remux and reports it as cancelled.
  Confidence: Verified
  Effort: S

- [ ] P3 -- RM-214: Clearing the queue filter leaves the visible rows out of model order
  Category: ux
  Where: `gui/queue_view.py:386-402` (`_apply_queue_filter` re-shows with a bare `widget.pack(fill="x")`)
  Problem: `pack` appends to the end of the packing order, so narrowing the filter to hide row 1 and then widening it re-packs row 1 *after* rows 2..n. Only `_update_queue_display` restores true order, and it is not called on filter changes, so the displayed order diverges from the processing order until the next queue mutation -- which misleads Move up/down and the user's expectation of what runs next. The filter fires on every keystroke via `trace_add`, so this is easy to hit.
  Evidence: Read the re-show loop; the ordered repack at `_update_queue_display` lines 353-357 shows the correct pattern.
  Fix: After the filter loop, repack all visible widgets in model order (or pass `before=` anchors).
  Acceptance: Typing and clearing a filter leaves the queue in model order; a test asserts widget order after a filter round-trip.
  Confidence: Likely (standard Tk pack semantics; worth a live check)
  Effort: S

- [ ] P3 -- RM-215: Inconsistent cross-thread `after` exception handling can persist a bogus ERROR state
  Category: reliability
  Where: `gui/app.py:2621-2624` (`_update_item_display` catches `RuntimeError` only), `gui/app.py:1765-1769`, `gui/processing_controller.py:347-349`, `gui/support_controller.py:207-208`
  Problem: `root.after` from a worker thread raises `RuntimeError` after the mainloop exits but `tk.TclError` once the interpreter is destroyed. Several sites catch both (for example `gui/app.py:880`, `processing_controller.py:383`); these four do not. Concretely, during close-while-processing, `_update_item_display`'s uncaught `TclError` propagates into `_process_item`'s blanket `except Exception`, which sets `item.status = ERROR` with the message `can't invoke "after" command: application has been destroyed`, and `save_queue_state` persists it -- so the next session restores a "needs attention" item for a file that was merely interrupted.
  Evidence: Compared the guarded and unguarded call sites; the propagation path into `_process_item`'s handler is direct.
  Fix: Route all worker-to-UI marshals through a shared helper that catches `(RuntimeError, tk.TclError)`, as `_dispatch_preview_ui` already does.
  Acceptance: Closing the app mid-batch leaves no item persisted in ERROR with a Tk teardown message.
  Confidence: Verified (code paths); the persisted-ERROR sequence is timing-dependent
  Effort: S

- [ ] P3 -- RM-216: The "Test cleanup" preview drops detector settings, so it does not preview what the batch will do
  Category: correctness
  Where: `gui/preview_controller.py:702-716` (hand-built `_BackendCfg`)
  Problem: The preview config passes mode, device, language, threshold, regions, dilation, feather, and TBE only. `detection_engine`, `detection_vertical`, `language_mask_filter`, `manual_mask_corrections`, and `lama_super_fast` are dropped and silently take backend defaults. A user who selected EasyOCR because RapidOCR misreads their language gets an auto-cascade preview that does not match the batch -- and the preview exists precisely to A/B settings. The mask-preview path *does* honor `detection_engine`, so the two previews can also disagree with each other.
  Evidence: `sed -n '700,716p' gui/preview_controller.py` lists the constructed fields; `:1183` shows the mask preview reading `detection_engine`.
  Fix: Build the preview config with `gui_to_backend_config(snapshot_cfg)` plus the device override rather than a hand-picked subset.
  Acceptance: Changing the detection engine changes the "Test cleanup" result; a test asserts the preview config matches the batch config for detector fields.
  Confidence: Verified
  Effort: S

- [ ] P3 -- RM-217: The review-mask preview ignores per-file language and engine overrides
  Category: correctness
  Where: `gui/preview_controller.py:1182-1184`
  Problem: `_show_preview` reads language from `self.lang_var` and engine/threshold from `self.config` (the global settings), while `mask_dilate_px` and the manual corrections at `:1207-1211` come from `item_config`. An item given a per-file language override through the overrides dialog therefore previews with the *global* language, so the reviewed mask differs from what the run will produce.
  Evidence: `sed -n '1180,1214p' gui/preview_controller.py` shows the mixed sourcing in one block.
  Fix: Read language, engine, and threshold from `item_config` like the dilation and correction fields already do.
  Acceptance: An item with a per-file language override previews using that language.
  Confidence: Verified
  Effort: S

- [ ] P3 -- RM-218: Adding a folder walks the tree on the Tk main thread and freezes the UI
  Category: perf
  Where: `gui/app.py:1891-1905` (`_add_folder_to_queue`)
  Problem: `sorted(folder.rglob("*"))` plus per-file `is_file()`/stat work runs synchronously inside the drop or dialog callback, with no cap and no progress. Dropping a large media library or a folder on a slow network share freezes the window for the entire walk. Every other slow operation in this app (hardware probe, previews, ETA) was moved off the main thread; this is the remaining synchronous walk.
  Evidence: Read the method; the walk and the queue insertion happen in the same callback.
  Fix: Enumerate on a worker thread and marshal the additions back through the existing `root.after` pattern, with a progress indication and a sane cap.
  Acceptance: Dropping a folder with thousands of files keeps the UI responsive and shows progress.
  Confidence: Verified
  Effort: M

- [ ] P3 -- RM-219: Each "Test cleanup" click builds a whole new `SubtitleRemover`, reloading every model
  Category: perf
  Where: `gui/preview_controller.py:717` (`remover = _Remover(backend_cfg)`)
  Problem: Every invocation constructs a full backend instance -- OCR plus inpainter initialization, seconds of latency and hundreds of megabytes (GPU allocations on CUDA) -- bypassing both `_cached_remover` and `_preview_detector`, and relies on garbage collection for teardown. Repeated preview clicks pay the full load each time and can transiently hold several model sets at once.
  Evidence: Read the construction site; neither cache is consulted.
  Fix: Reuse or extend the existing `_cached_remover` keyed cache, and share the preview detector.
  Acceptance: A second "Test cleanup" click with unchanged settings returns without reloading models.
  Confidence: Verified
  Effort: M

- [ ] P3 -- RM-220: ffmpeg dying mid-frame-write raises an untyped error, breaking the RM-139 typed-failure contract
  Category: correctness
  Where: `backend/io.py:1684-1701` (`_LosslessIntermediateWriter.write`), surfaced at `backend/processor.py:3020`
  Problem: The writer raises a plain `BrokenPipeError` when the FFV1 process exits before a frame is written, or when stdin breaks. Unlike every other RM-139 writer failure this is not a `MediaWriteError`, so it falls past the typed handler at `processor.py:2997` into the generic `except Exception`: `last_error_reason` becomes `"video_processing_error"` instead of a writer-specific reason, and the user sees the raw exception text. It still fails closed, but the typed contract is broken for this path, and no test exercises a mid-stream ffmpeg death during `write()` (only release-time timeout, nonzero exit, `imwrite` False, and terminate are covered).
  Evidence: Read the raise sites and the handler ordering in `process_video`; searched the encode/IO tests for a mid-write death case and found none.
  Fix: Wrap both raises in `MediaWriteError(reason="intermediate_writer_died")` and add a fake-process test whose `poll()` returns 1 mid-write.
  Acceptance: A mid-write ffmpeg death reports a writer-specific reason; the new test covers it.
  Confidence: Verified
  Effort: S

- [ ] P3 -- RM-221: A failed control-file publish is ignored, so Pause can silently do nothing
  Category: reliability
  Where: `gui/job_supervisor.py:452-475` (`_publish_control` returns False on `OSError`), callers at `gui/processing_controller.py:219`, `:743`, `:755`
  Problem: Every caller discards the return value. For cancel the 30-second watchdog escalation still rescues the situation, but for **pause** a failed publish means the child never pauses and nothing tells the user -- the UI shows paused while the job keeps running. No test covers `_publish_control` failing; the isolation tests only exercise successful round-trips.
  Evidence: Read the three call sites; none inspect the boolean.
  Fix: Surface a warning toast and log entry when the publish returns False, and add a test using an unwritable control path.
  Acceptance: Making the control file unwritable produces a visible warning instead of a silent no-op.
  Confidence: Verified
  Effort: S

- [ ] P3 -- RM-222: `ModernToggle` never removes its variable trace on destroy
  Category: maintainability
  Where: `gui/widgets.py:857-858` (`variable.trace_add("write", ...)`)
  Problem: The trace is never removed, so a toggle destroyed while its `BooleanVar` outlives it turns every later `var.set()` into a `TclError` inside the trace callback (a background Tk error). Today all toggles are app-lifetime or their vars are rebuilt with the dialog, so it is latent -- but it is the same defect class as the QueueItemWidget `<Destroy>` fix in 3.23, and any future dialog reusing an app-level var will hit it.
  Evidence: Read the constructor; there is no `<Destroy>` handler paired with the `trace_add`.
  Fix: Store the trace id and call `trace_remove` from a `<Destroy>` handler.
  Acceptance: Destroying a toggle whose variable survives and then setting that variable raises nothing.
  Confidence: Verified (latent -- no current repro path)
  Effort: S

- [ ] P3 -- RM-223: Mouse-wheel scrolling has large dead zones, and sliders steal the wheel without focus
  Category: ux
  Where: `gui/layout_build.py:76-77` (workbench binds the wheel on the canvas and content frame only), `gui/app.py:1511-1516` (queue binds only direct children), `gui/dialog_layout.py:50-75` (`bind_all` gated on canvas Enter/Leave), `gui/widgets.py:1029` (`ModernSlider` captures the wheel unconditionally)
  Problem: On Windows the wheel targets the widget under the pointer, so hovering any card, toggle, or label -- most of the surface -- does not scroll the workbench; the labels inside each queue row are dead zones; and in dialogs, moving from the canvas onto its embedded body fires `<Leave>` and unbinds, making wheel scrolling path-dependent. Meanwhile scrolling the settings column while the pointer passes over a slider silently changes that setting.
  Evidence: Read all four binding sites; the Enter/Leave gating and the per-widget bindings are visible in each.
  Fix: Install one `bind_all` wheel router that walks up from `event.widget` to the nearest scrollable ancestor, and require focus (or a modifier) before a slider consumes the wheel.
  Acceptance: The wheel scrolls the workbench from anywhere over it, and passing over a slider mid-scroll does not change its value.
  Confidence: Likely (code-verified; behavior benefits from a live check)
  Effort: M

- [ ] P3 -- RM-224: Tooltips and dialogs position themselves against the primary monitor
  Category: ux
  Where: `gui/widgets.py:360-367` (tooltip uses `winfo_screenwidth`/`screenheight`), `gui/dialog_layout.py:163-171` (`fit_dialog_to_work_area` centers on the primary screen rather than the parent window)
  Problem: Those APIs return primary-display metrics, so on a secondary monitor to the right of primary the `x + tw > sw` test is nearly always true and the tooltip is pinned to the primary screen's right edge, far from the widget it describes. The 3.32 release fixed exactly this class for window-geometry restore by moving to the `SM_*VIRTUALSCREEN` virtual desktop; tooltips and dialog centering were not covered by that change.
  Evidence: Read both sites; neither consults the virtual-desktop metrics used by the 3.32 fix.
  Fix: Reuse the existing `_desktop_bounds`-style virtual-screen helper for tooltip clamping, and center dialogs on the parent window rather than the primary screen.
  Acceptance: Tooltips appear adjacent to their widget on a secondary monitor; dialogs open centered on the parent window.
  Confidence: Verified (API semantics)
  Effort: S

- [ ] P3 -- RM-225: Microcopy and label inconsistencies across the queue and status surfaces
  Category: ux
  Where: `gui/processing_controller.py:668,1093` ("Complete!"), `gui/config.py:201` ("Needs Attention"), `gui/layout_build.py:1870` vs `gui/queue_view.py:286` (two patterns for the item count), `gui/layout_build.py:2019,2028` (queue move buttons), `gui/processing_controller.py:246-252` (elapsed tick), `gui/layout_helpers.py:127-137` and `gui/utils.py:270-287,328-333` (English-only plural suffixes)
  Problem: Several small inconsistencies in one pass: `item.message = "Complete!"` is the only exclamation mark in the product and duplicates the adjacent "Complete" badge; `"Needs Attention"` is Title Case where everything else is sentence case; the initial queue count uses `tr("{count} items")` while the live update uses the plural-aware `ntr(...)`, so the zero and one cases render inconsistently; the move buttons' text `"^"`/`"v"` doubles as their accessible label, so a screen reader would announce "^, button" once RM-160 is fixed; the 1 Hz elapsed tick overwrites the combined "42% / 1m 3s" label with bare elapsed time, so the percent flickers off and on between progress events; and `layout_helpers.py:127-137` plus `gui/utils.py` still build plurals with a hardcoded English `"s"` suffix inside translated templates, which the 3.31/3.32 `ntr()` sweep was meant to eliminate.
  Evidence: Each site read directly; the count-pattern divergence and the plural suffixes were confirmed against the `ntr()` usages elsewhere in the same files.
  Fix: Drop the exclamation mark, use sentence case for the status label, route the initial count through `ntr()`, give `ModernButton` an `accessible_label` override for glyph-only buttons, have the elapsed tick preserve the percent portion, and convert the remaining suffix patterns to `ntr()`.
  Acceptance: One consistent count pattern; no "(s)"-style or hardcoded-suffix plurals remain; the progress label keeps its percentage between ticks.
  Confidence: Verified (the elapsed-label flicker benefits from a live check)
  Effort: S

- [ ] P3 -- RM-226: Untranslated window titles and queue-info strings
  Category: ux
  Where: `gui/region_controller.py:183`, `gui/onboarding.py:36`, `gui/preview_controller.py:535,825`, `gui/utils.py:243-246,387,398`, `gui/queue_view.py:245`
  Problem: Several user-visible titles and labels are not wrapped for translation and do not appear in the catalog: the region selector's window title, the onboarding dialog title, the A/B compare and preview window titles, the queue's file-description and fallback strings, and the per-file overrides confirmation. `gui/utils.py:398` is additionally lowercase where the canonical string elsewhere is "Checking embedded subtitle tracks...". The lint's `CAPTION_CALLS` covers `set_title` but not `.title()` with a positional argument -- the same enumeration gap as RM-167.
  Evidence: Cross-checked each literal against `locale/vsr.pot`; none are present.
  Fix: Wrap them in `tr()`/`N_()`, align the casing, and extend the lint's caption model to cover `.title()`.
  Acceptance: Every listed string appears in `vsr.pot` and the lint fails if a new unwrapped `.title()` is added.
  Confidence: Verified
  Effort: S

- [ ] P3 -- RM-227: `_card_header` silently discards its eyebrow argument
  Category: maintainability
  Where: `gui/layout_helpers.py:93-98`
  Problem: Every call site passes an eyebrow caption (for example `_card_header(det_frame, "Detection", "Precision tuning")`) that is never rendered. The call sites read as though two labels appear; one is dead, which also muddies the i18n picture since those strings look like user-visible copy but are not.
  Evidence: Read the helper; the parameter is accepted and never used.
  Fix: Either render the eyebrow (the 3.7.0 card pattern intended it) or remove the parameter and update all call sites.
  Acceptance: No call site passes an argument the helper ignores.
  Confidence: Verified
  Effort: S

- [ ] P3 -- RM-229: A permanently-dead skipped test asserts on a file that can never exist
  Category: testing
  Where: `tests/test_hardening_config.py:508-514`
  Problem: `test_pillow_floor_is_12_3_0_in_build_workflow` skips whenever `.github/workflows/build.yml` is absent. That file was deleted permanently under the no-GitHub-Actions policy, so the assertion body is unreachable forever and the test will report "skipped" in every run indefinitely. The Pillow floor itself is covered elsewhere (`test_release_workflow.py:734` and `test_dependency_floor_consistency.py`), so this is dead weight rather than a coverage hole.
  Evidence: This is one of exactly 2 skips in the current run; `.github/` contains only `ISSUE_TEMPLATE/`, confirmed by listing.
  Fix: Delete the test, or repoint it at `build_exe.bat`/`requirements.txt` where the floor now lives.
  Acceptance: The suite reports 1 skip (the opt-in frozen smoke) rather than 2.
  Confidence: Verified
  Effort: S

- [ ] P3 -- RM-230: Assertion gaps in three tests that read as stronger than they are
  Category: testing
  Where: `tests/test_job_isolation.py:495-508`, `tests/test_subtitle_translation.py:336-345`, `tests/test_webvtt.py` (missing parse-side limit coverage for `backend/webvtt.py:191-197,232-235`)
  Problem: Three separate weaknesses. `test_a_crashed_outcome_marks_only_that_item_and_keeps_its_logs` builds an outcome carrying a stderr tail but asserts only the status mapping -- nothing checks the "keeps its logs" half of its own name, and the `updates` list the fixture produces is never asserted anywhere in the class. `test_provider_that_never_reads_stdin_times_out` uses `assertRaises(Exception)`, which any pre-spawn validation error would satisfy, so it does not actually pin the timeout behavior. And `parse_vtt`'s `MAX_CUES`/`MAX_BLOCKS`/`MAX_CUE_TEXT` rejection branches have no test (only the `read_vtt` byte bound and the translated-payload bound are covered), so the DoS limits could be removed without a red test.
  Evidence: Read all three tests and the uncovered limit branches.
  Fix: Assert the stderr tail lands on the item or log record; narrow the translation assertion to the timeout-specific exception type and message; add one `parse_vtt` test per limit, monkeypatching the module constants to keep it fast.
  Acceptance: Each test fails when its specific behavior is broken.
  Confidence: Verified
  Effort: S

- [ ] P3 -- RM-231: `docs/edge_case_corpus.md` promises CI enforcement that no longer exists
  Category: docs
  Where: `docs/edge_case_corpus.md:60-61,76-77`
  Problem: The guide tells contributors that clips past 20 seconds "inflate the test suite runtime past the CI budget" and that "future PRs that breach those bounds fail CI". GitHub Actions were removed in 3.17.3 and `.github/` now holds only issue templates; enforcement is the *local* reference-corpus stage in `build_exe.bat`. A contributor reading this forms a wrong model of how their submission is validated. (See also RM-185 -- this file is not even published.)
  Evidence: Read both passages; confirmed `.github/` contains only `ISSUE_TEMPLATE/`.
  Fix: Reword to reference the local release gate (`build_exe.bat` / `backend.reference_corpus`).
  Acceptance: No documentation claims CI enforcement that does not exist.
  Confidence: Verified
  Effort: S

- [ ] P3 -- RM-232: The spec and the build script disagree on env-gate truthiness, so evidence can contradict the artifact
  Category: correctness
  Where: `VideoSubtitleRemoverPro.spec:8-9` (`_enabled` accepts `1/true/yes/on`), `build_exe.bat:106-122` (compares only against `"1"`)
  Problem: With `VSR_ENABLE_FULL_OCR=true`, the spec builds paddleocr, easyocr, and torch *into* the executable while the batch script records `--exclude-module paddle --exclude-module paddleocr --exclude-module easyocr` into `release-hidden-imports.json` and the evidence's `excludedModules`. The published evidence then asserts exclusions the artifact does not have, and the TOC-derived SBOM in the same release set disagrees with the hidden-imports evidence.
  Evidence: Read both parsers; the truthy sets differ.
  Fix: Normalize both sides to the same truthy set, or better, have `release_verification` derive the gates from the environment or the spec rather than from duplicated batch-file strings.
  Acceptance: Building with `VSR_ENABLE_FULL_OCR=true` produces evidence whose `excludedModules` matches the artifact's actual contents.
  Confidence: Verified
  Effort: S

- [ ] P3 -- RM-233: Three NSIS installer defects (wrong close handle, per-user shortcuts from a per-machine install)
  Category: correctness
  Where: `installer/vsr.nsi:98-101` (mutex probe), `:127-132` and `:156-158` (shortcut creation and removal)
  Problem: Two independent issues in one file. The mutex probe stores the handle in `$R0` (`p .R0`) but closes `p r0` -- lowercase `r0` is `$0`, so it closes whatever `$0` holds and leaks the real mutex handle for the installer's lifetime (benign today, since that branch aborts the install anyway, but simply wrong). Separately, the installer requests admin and writes to HKLM, yet `$SMPROGRAMS`/`$DESKTOP` default to the elevating user's own folders without `SetShellVarContext all` -- so other users get no shortcuts, and an uninstall run by a different admin leaves the original user's shortcuts orphaned.
  Evidence: Read both sections; the register-case mismatch and the missing context call are visible in the source.
  Fix: `System::Call 'kernel32::CloseHandle(p R0)'`, and add `SetShellVarContext all` to both the install section and `Section "Uninstall"`.
  Acceptance: Shortcuts appear for all users on a per-machine install and are fully removed on uninstall.
  Confidence: Verified
  Effort: S

- [ ] P3 -- RM-234: The installer smoke silently mis-installs when the repo path contains spaces
  Category: reliability
  Where: `build_exe.bat:193-198`
  Problem: `Start-Process -ArgumentList '/S','/D=!SMOKE_INSTALL_DIR!'` lets PowerShell quote any argument containing spaces, but NSIS requires `/D=` to be the last, *unquoted* parameter. A checkout under a spaced path therefore extracts to the default directory, and the subsequent existence check fails the build with a misleading "payload is missing the frozen executable" error. Latent today because the current path has no spaces.
  Evidence: Read the invocation; the NSIS silent-install parameter rules are unambiguous about `/D=`.
  Fix: Invoke the installer through `cmd /c` (or pass the raw argument string) so `/D=` reaches NSIS unquoted, and fail with a message naming the real cause.
  Acceptance: A build from a directory containing spaces completes the installer smoke successfully.
  Confidence: Verified
  Effort: S

- [ ] P3 -- RM-235: `setup.py` is CWD-relative, so running it from elsewhere bootstraps into the wrong directory
  Category: reliability
  Where: `setup.py:27` (`Path("requirements.txt")`), `:225-230` (`os.path.abspath("venv")`), `:369`, `:723-833` (launchers written to the CWD)
  Problem: `python C:\path\to\setup.py` from another directory silently creates `venv/` and writes the launcher scripts into the caller's current directory, and skips `requirements.txt` entirely (falling back to the hardcoded core-package list). The `.bat` launchers `cd /d "%~dp0"` first, so only direct invocation is exposed -- but that is exactly what a user following a support instruction would type.
  Evidence: Read the path constructions; none are anchored to `__file__`.
  Fix: `os.chdir(Path(__file__).resolve().parent)` at entry, or anchor every path to `__file__`.
  Acceptance: Running `setup.py` from an unrelated directory creates the venv beside `setup.py` and reads the real `requirements.txt`.
  Confidence: Verified
  Effort: S

- [ ] P3 -- RM-236: Dead installer code and unused build variables
  Category: maintainability
  Where: `setup.py:486-527` (`install_paddlepaddle`, never called from `main()` at `:859-907`), `build_exe.bat:88-99` (`DATA_ARGS`/`ICON_ARG` never referenced again)
  Problem: `install_paddlepaddle` is 40 lines of maintained-looking installer logic -- including the Blackwell cu126 index selection that CLAUDE.md and `requirements.txt` comments both reference -- that no code path reaches; its step numbering (`[5/6]`) would also collide with `install_dependencies` if it were re-wired. In the build script, `DATA_ARGS` and `ICON_ARG` are computed and never used, because the build is spec-driven; the duplicated hidden-import and exclude bookkeeping exists solely to feed evidence, which is the root cause of RM-232.
  Evidence: `main()` reads end to end with no call to `install_paddlepaddle`; the batch variables have no second reference.
  Fix: Delete the dead variables; either wire `install_paddlepaddle` into an explicit opt-in lane or remove it and correct the docs that reference it.
  Acceptance: No unreferenced build variables remain, and the PaddlePaddle install path is either reachable or gone.
  Confidence: Verified
  Effort: S

- [ ] P3 -- RM-238: ffprobe receives the input path as a bare trailing positional, so a leading-dash filename is parsed as options
  Category: security
  Where: `backend/container_payload.py:83-92` (`probe_container_manifest`), same pattern at `:472`
  Problem: `str(path)` is appended as the final positional argument with no `./` prefix or `--` guard. FFmpeg tools parse leading-dash tokens as options even in trailing position, so a media file named `-loglevel.mkv` (reachable through `--pattern` globbing) is read as options rather than an input, producing a wrong or empty manifest instead of a clean failure. No shell is involved, so this is not remote code execution -- it defeats the container-preservation logic and is an argument-handling hole. Other subprocess sites are safe because they pass paths as the value of a consuming flag (`-i <path>`).
  Evidence: Read the command construction; compared against the safe `-i` pattern in `proxy_workflow`.
  Fix: Normalize relative inputs to absolute (or prefix with `./`) before handing them to ffprobe/ffmpeg.
  Acceptance: A file whose name begins with a dash probes correctly; a test covers the case.
  Confidence: Verified (exploitability limited to self-named files)
  Effort: S

- [ ] P3 -- RM-239: Sub-1.0 fps sources get wrong timing wherever the `max(fps, 1.0)` guard fires
  Category: correctness
  Where: `backend/processor.py:437-442` (`_frame_seconds`), `backend/io.py:79-89` and `:630`, `backend/_srt_mixin.py:110` (`fps > 1.0` else 30.0)
  Problem: The divide-by-zero guards clamp the denominator to 1.0 rather than to an epsilon, so a valid 0.5 fps source (a slideshow or timelapse) computes frame times at 1 fps -- wrong matte timestamps, audio seek offsets, and SRT cues, which additionally jump to a 30 fps assumption. The main fps sanitizer only rejects values `<= 0`, so such sources pass straight through.
  Evidence: Read all four guards; each uses `max(..., 1.0)`.
  Fix: Clamp with a small epsilon (`max(fps, 1e-6)`) and let the sanitizer handle genuinely invalid values.
  Acceptance: A 0.5 fps source produces frame times of 2.0 seconds per frame; a test covers it.
  Confidence: Verified (rare but valid media)
  Effort: S

- [ ] P3 -- RM-240: A refused seek on the legacy PyNv decoder surfaces as `AttributeError` instead of a typed error
  Category: reliability
  Where: `backend/processor.py:343-346` (`_seek_capture_to_frame` calls `cap.grab()`), `backend/decode_accel.py:111-266` (`_PyNvVideoCapture` defines no `grab`)
  Problem: With `--decode-accel pynv` on a legacy PyNvVideoCodec build lacking `SeekFrame`, `set()` returns False and leaves the position behind the target, so the catch-up loop calls `cap.grab()` and raises `AttributeError`, which becomes a generic `video_processing_error` with a cryptic message. RM-138 intended seek failures to be loud but *explicit*; this is an accidental crash message instead. All other capture adapters report an accurate position, so only this niche path is exposed.
  Evidence: Read the catch-up loop and the adapter's method set.
  Fix: Fall back to `read()`-and-discard when `grab` is absent, or raise a typed `MediaInputError` on seek refusal.
  Acceptance: A refused seek on that decoder produces a clear typed error naming the decoder and the requested frame.
  Confidence: Verified (niche reachability)
  Effort: S

- [ ] P3 -- RM-241: Selective mask rerun is structurally incompatible with time-ranged jobs and blames the file
  Category: ux
  Where: `backend/processor.py:2444-2470` (the guard requires `prior_frames >= end_frame`, i.e. absolute source-frame alignment), `:2553-2555`
  Problem: A prior cleaned output from a time-ranged job contains only `end_frame - start_frame` frames, but the guard and the seek both assume absolute source-frame alignment. Any correction retry on a ranged job therefore always fails with "Previous cleaned output is missing frames required for selective rerun" -- the workflow is unusable with time ranges, and the message blames the file rather than naming the limitation. The absolute indexing does at least keep the misalignment case fail-closed.
  Evidence: Read the guard and the seek; both index against `end_frame`/`start_frame` on the source clock.
  Fix: Detect `prior_frames == frames_to_process` and switch to range-relative indexing, or emit an explicit "selective rerun does not support time ranges" error.
  Acceptance: A correction retry on a time-ranged job either works or fails with a message naming the real limitation.
  Confidence: Verified
  Effort: S

- [ ] P3 -- RM-242: Confidence-weighted dilation and low-confidence review spans run on fabricated confidences for most engines
  Category: correctness
  Where: `backend/detection.py:528-534` (`_detect_axis_aligned_conf` returns a constant 1.0 for non-Rapid engines), consumed at `backend/processor.py:1802-1835`
  Problem: Only the RapidOCR path returns real confidences; PaddleOCR, EasyOCR, Surya, and the OpenCV fallback all get a hardcoded 1.0, even though Paddle and EasyOCR expose genuine scores (already used by `detect_with_text`). As a result `confidence_weighted_dilation` silently degrades to uniform dilation, and the quality report's low-confidence review spans can never trigger on those engines -- with nothing in the report noting that the confidences were synthetic.
  Evidence: Read the conf-returning branch and its consumers.
  Fix: Route the Paddle and EasyOCR scores through the same path, or record in the report that confidences were synthetic for the engine used.
  Acceptance: With PaddleOCR selected, confidence-weighted dilation varies per box and low-confidence spans can trigger.
  Confidence: Verified
  Effort: M

- [ ] P3 -- RM-243: Assorted small robustness gaps (tracker overflow, correction bounds, VL box union, dead flags, private API)
  Category: correctness
  Where: `backend/tracking.py:89-107`; `backend/mask_corrections.py:109-114` and `:196-233`; `backend/ocr_vlm.py:370-395,467-472`; `backend/output_quality_preflight.py:108-109`; `backend/inpainters/_common.py:235`; `backend/_encode_mixin.py:39-45`
  Problem: Six small independent defects, grouped because each is a few lines. (a) `_box_from_state` checks `np.isfinite` on the centers but not on width/height, so an infinite `w` reaches `int(round(...))` and raises `OverflowError` out of `SubtitleTracker.update`, failing the job. (b) A correction carrying `end_frame` but no `start_frame` -- permitted by `normalize_mask_correction` -- falls into the seconds branch when a frame index is supplied, silently ignoring its frame bound; and `merge_review_spans` only merges against the immediately previous span, so an interleaved other-kind span leaves overlapping same-kind spans unmerged and duplicates review work. (c) `_collect_vl_boxes` collapses a bare list of N `[x1,y1,x2,y2]` detections into a single union box, turning two far-apart captions into one frame-spanning mask. (d) `output_quality_preflight` sets `overrideRequired` and `overridden` from the same expression, so anything requiring an override is already overridden -- currently only misleading JSON (no consumer), but structurally unable to gate. (e) `_common.py:235` drives PySceneDetect's private `_process_frame`, whose signature has shifted across releases, so the `prefer_pyscenedetect` toggle can silently stop doing anything on upgrade. (f) `_encode_mixin`'s local `_frame_seconds` shadow checks `timing.frame_pts`, but the dataclass exposes `timestamps`, so the branch can never fire -- harmless today (its only caller passes no timing) but a silent CFR trap for any future caller.
  Evidence: Each site read individually; (d) confirmed by finding no consumer of either key anywhere in `backend/` or `gui/`.
  Fix: Extend the finiteness guard to w/h; honor a frame-only correction bound and merge spans against all prior same-kind spans; recurse per row for (N,4) shapes with N>1; drive `overridden` from a real acknowledgment or drop both fields; move to PySceneDetect's public API; delete the `_frame_seconds` shadow in favor of the real one.
  Acceptance: Each sub-item has a unit test; no dead branch or always-satisfied flag remains.
  Confidence: Verified (individually); (a) requires filter-state blow-up to trigger
  Effort: M (six small changes)

- [ ] P3 -- RM-244: README install commands drop their version pins to shell redirection
  Category: docs
  Where: `README.md:122-129`, `README.md:1084`
  Problem: `pip install torch>=2.10.0 ...` is written unquoted, and `>` is a redirection operator in cmd, bash, and PowerShell alike -- so the pin is silently dropped and a file named `=2.10.0` is created instead. At line 122-129 the accompanying profile constraint still rescues the resolved version; at line 1084 there is no `--constraint`, so following the documented command genuinely installs an unpinned latest torch.
  Evidence: Read both command blocks; neither quotes the specifier.
  Fix: Quote the specifiers (`pip install "torch>=2.10.0"`) in both places and add the missing constraint at line 1084.
  Acceptance: Copying any README pip command into a shell installs the pinned version and creates no stray file.
  Confidence: Verified
  Effort: S

- [ ] P3 -- RM-245: `import_se_ocrfix.py --lang` is mandatory but affects nothing
  Category: maintainability
  Where: `scripts/import_se_ocrfix.py:91-92,101,129`
  Problem: The normalized language key is only interpolated into the success message; the output filename and content both come from `--out`. A required argument that changes no behavior invites the assumption that it validates key/filename agreement, which it does not.
  Evidence: Read every use of the parsed value.
  Fix: Either validate the `--out` stem against the normalized language, or make `--lang` optional and derive it from `--out`.
  Acceptance: `--lang` either affects output or is no longer required.
  Confidence: Verified
  Effort: S

- [ ] P3 -- RM-246: Repository hygiene: stray `%temp%` directory and `.dockerignore` gap
  Category: maintainability
  Where: repo root `%temp%/comtypes-1.4.16-py3-none-any.whl`, `.dockerignore`, repo root `out/`
  Problem: A literal `%temp%` directory sits in the repo root holding a wheel -- litter from a past unexpanded `%TEMP%` in a shell command (mtime 2026-08-10, predating this audit). It is untracked and harmless, but it confuses globbing, backup, and archive tooling. Separately, `.dockerignore` excludes `output` but not the also-present `out/`, so `COPY . .` ships it into the smoke image.
  Evidence: `stat` confirms the directory and its single wheel; `git status --ignored` shows `out/` and `output/` as untracked/ignored; `.dockerignore` lists `output` only.
  Fix: Delete the `%temp%` directory, and add `out/` to `.dockerignore`.
  Acceptance: The repo root holds no `%temp%` directory and the Docker build context excludes both scratch directories.
  Confidence: Verified
  Effort: S

- [ ] P3 -- RM-247: v3.32.0 and v3.33.0 are shipped in CHANGELOG but neither tagged nor released
  Category: docs
  Where: `README.md:9` (version badge 3.33.0) and `:82-87` (Prebuilt Download points at `releases/latest`); `CLAUDE.md` "Current Status"
  Problem: The badge and CHANGELOG document 3.33.0, but the newest tag and GitHub Release is `v3.31.0` (2026-07-30), so the README's download instruction serves a build two versions behind the documented one. The repo's own precedent is that intermediate versions ship inside the next release, so this is a process gap rather than an error in either file -- but the download link and the badge currently disagree about what a user gets. Related: `CLAUDE.md`'s "Current Status" still names v3.31.0 as the working version and cites 1297 tests, now stale by two releases and 21 tests.
  Evidence: `git tag --list | sort -V` ends at `v3.31.0`; `gh release list` shows v3.31.0 as Latest; `gui/config.py:137` is `APP_VERSION = "3.33.0"`; the measured suite is 1318 tests.
  Fix: Tag and publish v3.33.0 (unsigned, per policy) after the P0/P1 items land, or note in the README that badges track source rather than the latest release. Refresh the CLAUDE.md status block in the same pass.
  Acceptance: The newest tag matches `APP_VERSION`, or the README states the distinction explicitly.
  Confidence: Verified
  Effort: S

- [ ] P3 -- RM-249: `test_cli_support_bundle_entrypoint_is_dependency_light` is flaky under full-suite load
  Category: testing
  Where: `tests/test_support_bundle.py:153-173`
  Problem: The test spawns `python -m backend.cli --support-bundle` as a
  subprocess with a fixed `timeout=30`. Running it inside the full suite on a
  loaded machine intermittently exceeds that budget and fails the run; the
  same test passes in isolation and on a re-run. A flaky gate trains everyone
  to re-run rather than read failures.
  Evidence: Observed 2026-08-11 -- one full-suite run reported this single
  failure, an immediate isolated run passed in 10.9 s, and the next full run
  passed with 1349 tests. Nothing in the code path changed between runs.
  Fix: Scale the timeout (60-120 s) or mark the subprocess-spawning tests so
  they are not competing with the rest of the suite for CPU. Prefer raising
  the timeout: the assertion is about dependency weight, not speed.
  Acceptance: Ten consecutive full-suite runs with no spurious failure from
  this test.
  Confidence: Verified
  Effort: S

### Unaudited -- needs a pass

- [ ] P3 -- RM-248: Areas this audit did not cover
  Category: docs
  Where: n/a (tracking item)
  Problem: For honesty about coverage, the following were not audited in this pass and should get their own: (1) **runtime GPU/CUDA/NVENC/DirectML behavior** -- no GPU-capable host here, so every provider-selection, OOM-recovery, and hardware-encode path was read but never executed; (2) **real-media output quality** -- no reference clips were processed, so inpainting quality, TBE behavior on real footage, and the quality-gate thresholds are unverified empirically; (3) **the frozen executable** -- no PyInstaller build was run, so spec correctness, hidden-import completeness, and frozen-only code paths are static-only (and RM-156 means a clean release build cannot currently complete); (4) **the NSIS installer end-to-end** -- install, upgrade, and uninstall were not executed; (5) **live GUI interaction** -- the app was not launched, so all visual, layout, focus-order, and reflow findings are code-derived (contrast was computed numerically from tokens, which is objective, but rendering was not observed); (6) **`backend/inpainters_diffusion.py` opt-in adapters** beyond the coercion path -- none of the external model integrations were exercised; (7) **`backend/ocr_vlm.py` VLM backends** past the load-failure path; (8) **long-video and 4K endurance** -- RM-163's threshold is arithmetic, not measured.
  Evidence: n/a
  Fix: Schedule a GPU-host session for (1), (2), (6), (7); a release-build session for (3), (4) after RM-156 lands; and a driven-GUI session for (5).
  Acceptance: Each area either audited or explicitly accepted as out of reach.
  Confidence: Verified
  Effort: L

## Research-Driven Additions

External research pass 2026-08-11 (~120 sources; conclusions in RESEARCH.md). IDs
continue the existing scheme from RM-249. These are ecosystem gaps and
opportunities, not defects found by the audit above -- no item here duplicates
RM-167..RM-249. Every premise was checked against the code first; the ones that
turned out to be already implemented (VMAF, flicker metrics, "Copy CLI command",
chyron dwell heuristics, moving-watermark tracking) are recorded as rejections in
RESEARCH.md instead of appearing here.

### P1

### P2

- [ ] P2 -- RM-258: Compositing corrects colour but not gradient, so seams survive on skin tones and gradients
  Why: `_edge_ring_color_correct` fixes a flat colour offset and `_feather_blend` softens the boundary, but neither matches the gradient field across the seam, which is what reads as a soft halo on smooth content.
  Evidence: `grep -rn "seamlessClone\|[Pp]oisson" backend/` returns nothing; finishing is edge-ring then feather for every inpainter via the shared `apply_finishing` introduced in 3.18.0. `cv2.seamlessClone` is in every OpenCV build the repo already requires -- https://docs.opencv.org/4.x/df/da0/group__photo__clone.html. Implement the modified-Poisson variant, not vanilla: naive Poisson blending is a documented source of temporal bleeding, which matters for video -- https://link.springer.com/article/10.1007/s41095-015-0027-z.
  Touches: `backend/inpainters/_common.py` (`apply_finishing`, `_edge_ring_color_correct`), `backend/config.py`, `backend/cli.py`, `tests/`
  Acceptance: An opt-in gradient-domain finishing mode runs a modified-Poisson seam correction inside the dilated mask boundary before feathering, shared by every inpainter through `apply_finishing`; it is skipped for masks touching the frame edge and for degenerate mask areas; a gradient-background fixture shows lower seam residual than the edge-ring path, and `temporal_profile.masked_flicker` does not regress across a multi-frame fixture.
  Complexity: M

- [ ] P2 -- RM-259: The fill is smoother than the source it replaces, and nothing puts the grain back
  Why: A temporal median is a low-pass filter by construction, so on any grainy or noisy source the filled region reads as a plastic patch no matter how well the seam is blended.
  Evidence: Every `film_grain` reference is in `backend/_encode_mixin.py:51-65` and only emits `-svtav1-params film-grain=N` for AV1 final encodes -- an encoder-side setting that does nothing for the fill region and does not apply to H.264/HEVC outputs at all. Parametric autoregressive grain fitted from a clean neighbouring patch is a well-specified classical technique -- https://norkin.org/pdf/DCC_2018_AV1_film_grain.pdf, https://www.lirmm.fr/~nfaraj/publications/film_grain_ipol/2017_Newson_film_grain.pdf.
  Touches: `backend/post_restore.py`, `backend/inpainters/_common.py`, `backend/config.py`, `backend/cli.py`, `tests/`
  Acceptance: Grain statistics are estimated from unmasked pixels adjacent to the mask, synthetic grain matching those statistics is added inside the feathered region with per-frame temporal decorrelation, and a small blue-noise dither is applied before the return to 8-bit so the float blend does not band; a grainy fixture shows noise variance inside the fill within tolerance of the surrounding region, and a clean synthetic fixture is left untouched.
  Complexity: M

- [ ] P2 -- RM-260: PaddleOCR 3.6.0 silently changed the PP-OCRv5 default from mobile to server models
  Why: If detection relies on library defaults rather than explicit model names, latency and memory changed under the project without a code change or a note.
  Evidence: PaddleOCR 3.6.0 release notes state the default model for both PP-OCRv5 detection and recognition changed from the mobile to the server variants -- https://sourceforge.net/projects/paddleocr.mirror/files/v3.6.0/. `dependency_profiles.json` pins `paddleocr==3.6.0`. The 2.x -> 3.x API removals (`det`/`rec` kwargs on `.ocr()`, `show_log`, `use_onnx`, `PPStructure`) are documented upstream and worth re-checking against `backend/paddle_compat.py` in the same pass -- http://www.paddleocr.ai/main/en/update/upgrade_notes.html. That the project takes whatever the library defaults to is Verified: `backend/paddle_compat.py:47-60` constructs `PaddleOCR(...)` with no `text_detection_model_name`, `text_recognition_model_name`, or `ocr_version`, on both the 3.x and the 2.x fallback path. That 3.6.0 specifically flipped the default is Reported -- confirm against the installed package before changing behaviour. `backend/detection.py:175-193` likewise builds RapidOCR with `params={}`.
  Touches: `backend/detection.py` (`_load_model`), `backend/paddle_compat.py`, `README.md`, `tests/`
  Acceptance: PaddleOCR detection and recognition model variants are named explicitly rather than defaulted, the chosen variant is recorded in execution provenance alongside the engine, and README documents the mobile/server tradeoff so a user on constrained hardware can pick.
  Complexity: S

- [ ] P2 -- RM-261: No unattended watch-folder mode, which is the automation shape with the clearest external demand
  Why: Archivists and bulk users want a directory that drains itself; every primitive already exists (glob batch, checkpoints, skip-existing, output contracts, job isolation) and only the loop is missing.
  Evidence: `grep -rn "watch_folder\|--watch"` over `backend/` and `README.md` returns nothing. Demand is concrete: the upstream project has an open request for portable CLI and headless server support (YaoFANGUK #246), and `axllent/vidfxr` is a Docker image whose entire purpose is polling a directory and stripping subtitles from newly-written files -- https://hub.docker.com/r/axllent/vidfxr, https://github.com/YaoFANGUK/video-subtitle-remover/issues. Every commercial competitor surveyed gates batch behind a paid tier.
  Touches: `backend/cli.py`, `backend/resume_checkpoint.py`, `backend/output_contract.py`, `README.md`, `tests/`
  Acceptance: `--watch <dir>` polls for new inputs on a configurable interval, waits for each file to stop growing before claiming it, processes it through the existing single-file path with `--skip-existing` semantics, writes per-item outcomes to the existing batch report, survives a failed item without exiting, and shuts down cleanly on the existing cancellation signal; a fixture drops files into a temp directory mid-run and asserts each is processed exactly once.
  Complexity: M

- [ ] P2 -- RM-262: The Docker image validates the pipeline but cannot run it
  Why: A container that only self-tests forces every headless or Linux user to build their own, which is exactly why third-party wrappers exist for the upstream project.
  Evidence: `Dockerfile` installs only numpy, opencv-headless, Pillow and onnxruntime from the CPU constraints, and its CMD is `["python", "tools/local_smoke.py"]`; `dependency_profiles.json` `intentionalExceptions` states plainly that it "is a local pipeline smoke image, not the frozen Windows distribution". `leeyeel/video-subtitle-remover-docker` exists specifically because upstream ships no first-class container -- https://github.com/leeyeel/video-subtitle-remover-docker.
  Touches: `Dockerfile`, `.dockerignore`, `dependency_profiles/cpu.txt`, `README.md`, `tests/`
  Acceptance: The image installs the full CPU profile including RapidOCR and the ONNX LaMa tier, its entrypoint is the CLI (`python -m backend.cli`) with the smoke test kept reachable as an explicit command, input and output are mounted volumes, and README documents a working one-line `docker run` that processes a file end to end; the smoke stage still runs at build time so a broken image fails to build. Pairs with RM-261 -- the watch-folder mode is what makes the container useful unattended. Note: RM-246 already covers the separate `.dockerignore` hygiene gap.
  Complexity: M

### P3

- [ ] P3 -- RM-263: Semi-transparent subtitle boxes are treated as binary masks, though the pipeline already computes both endpoints needed to solve them properly
  Why: A translucent caption bar is a linear mix, not an occlusion; removing it as an occlusion discards recoverable background, and it is a named open problem in the current literature.
  Evidence: No `unmix`, `alpha` or `translucent` handling exists in `backend/`; the mask path is binary plus morphological dilation plus Gaussian feather. Both endpoints of `I = a*FG + (1-a)*BG` are already produced -- the foreground colour by the Lab two-cluster split that `--colour-tune` runs (`backend/inpainters/_common.py:145`) and the background by the TBE aggregate. Both 2026 subtitle-erasure papers name semi-transparent and gradient captions as the unsolved case -- https://arxiv.org/html/2603.21901, https://arxiv.org/abs/2605.14894 -- and both are diffusion models with no released code, so the classical solve is the only one available.
  Touches: `backend/inpainters/_common.py`, `backend/segmentation.py`, `backend/config.py`, `backend/cli.py`, `tests/`
  Acceptance: A translucency detector decides per region whether the masked pixels fit a two-endpoint linear mix (low residual against the estimated FG colour and the TBE background) and, when they do, solves per-pixel alpha in closed form and composites with it instead of the binary mask; regions that do not fit fall back to the current path with the decision logged; a synthetic translucent-bar fixture recovers the known background within tolerance, and an opaque-caption fixture is byte-identical to today's output.
  Complexity: L

- [ ] P3 -- RM-264: Mask dilation is a fixed slider, so outlined and drop-shadowed glyphs are systematically under- or over-dilated
  Why: The correct dilation depends on the glyph's outline and shadow thickness, which varies per source and is measurable from the detected box.
  Evidence: `--mask-dilate` is a fixed 0-20px setting and `--confidence-dilate` scales by detector confidence -- which RM-242 already flags as fabricated for most engines, so that path cannot be the adaptive mechanism. Nothing measures the actual stroke or shadow extent, and PP-OCR-family DB detection heads localize the glyph body, not its outline.
  Touches: `backend/segmentation.py`, `backend/detection.py`, `backend/config.py`, `backend/cli.py`, `tests/`
  Acceptance: An auto-dilate mode measures the intensity-gradient falloff distance outward from the binarized glyph inside each detected box, derives a per-box dilation radius and clamps it to the existing 0-20px range; a soft continuous mask built with `cv2.distanceTransform` replaces the discrete dilate-then-feather pair so the two steps cannot disagree at the boundary; outlined-glyph and plain-glyph fixtures each get an appropriate radius, and the manual slider still overrides. Depends on RM-242 if the confidence path is to be reused.
  Complexity: M

- [ ] P3 -- RM-265: The NVIDIA TensorRT-RTX execution provider is not declared as a provider lane
  Why: The provider-lane table is the project's honest map of what inference paths exist and how tested each is; a consumer-RTX-targeted EP that ONNX Runtime now ships is missing from it.
  Evidence: `backend/dependency_caps.PROVIDER_LANES` declares CPU, CUDA-12, CUDA-13 and DirectML, with CUDA-13 already carried as an untested manual lane -- the exact pattern this needs. ONNX Runtime documents the NV TensorRT-RTX EP as targeting Ampere-and-newer consumer RTX and as more straightforward than the legacy TensorRT EP and more performant than the CUDA EP -- https://onnxruntime.ai/docs/execution-providers/TensorRTRTX-ExecutionProvider.html. The repo's existing TensorRT support is a polygraphy-based ahead-of-time engine cache behind `VSR_TENSORRT=1` (`backend/tensorrt_compile.py`), a different mechanism.
  Touches: `backend/dependency_caps.py`, `backend/device_provider.py`, `backend/onnxruntime_cuda.py`, `README.md`, `tests/`
  Acceptance: TensorRT-RTX appears in PROVIDER_LANES as an untested manual lane with its own security and tested state, `device_provider` probes for and reports it the way `windowsml_status()` reports Windows ML, and README's provider table lists it. Live benchmarking stays out -- it needs NVIDIA hardware and belongs in Roadmap_Blocked.md.
  Complexity: S

- [ ] P3 -- RM-266: README advertises a winget install path that the no-signing policy makes unreliable
  Why: winget's non-interactive upgrade flow has no way past a SmartScreen block, so the documented command can work on first install and then fail on upgrade.
  Evidence: `README.md:65` promises "winget-ready installer metadata" and `:103-107` gives `winget install SysAdminDoc.VideoSubtitleRemoverPro` for "the unsigned release". A winget-pkgs thread treats unresolved SmartScreen and Mark-of-the-Web blocking of an unsigned installer as a release blocker precisely because upgrades must run non-interactively -- https://github.com/microsoft/winget-pkgs/issues/385483. Code signing is prohibited by repository policy, so the constraint is permanent rather than a to-do. Confidence: Reported.
  Touches: `README.md`
  Acceptance: The winget section states the SmartScreen / Mark-of-the-Web constraint for unsigned packages and names direct download plus "More info -> Run anyway" as the supported path, keeping the manifest-metadata claim only if the manifest is actually published. Distinct from RM-185, which covers broken documentation links.
  Complexity: S

- [ ] P3 -- RM-267: The TTML/IMSC rejection cites reasoning that predates IMSC 1.3 becoming a W3C Recommendation
  Why: A correct decision resting on a stale citation invites the next reviewer to reopen it from scratch.
  Evidence: `backend/webvtt.py:21` and `backend/subtitle_translation.py:293-301` state that TTML and IMSC are deliberately unsupported. IMSC Text Profile 1.3 became a W3C Recommendation on 2026-05-21 -- https://www.w3.org/TR/ttml-imsc1.3/. The decision to stay out is still correct (no demand was found this cycle, and the XML plus region/image-profile surface is materially larger than the WebVTT work that shipped in 3.31.0), but the notes should say so against the current spec.
  Touches: `backend/webvtt.py`, `backend/subtitle_translation.py`
  Acceptance: Both module docstrings cite IMSC 1.3 with its Recommendation date and state the demand-and-surface reason for staying out, so the rejection is re-readable rather than re-litigable.
  Complexity: S

- [ ] P3 -- RM-268: RGVI is an unevaluated flow-guided inpainting candidate that may fit where the diffusion models cannot
  Why: Every other 2025-2026 candidate is a multi-billion-parameter diffusion model; RGVI is flow-guided, architecturally much closer to what TBE already does and plausibly viable at consumer VRAM.
  Evidence: RGVI (AAAI 2025) elevates flow-guided video inpainting with a learned reference-frame generator -- https://github.com/suhwan-cho/rgvi. It appears in none of this repo's prior evaluations and is absent from both RESEARCH.md's rejection list and Roadmap_Blocked.md. License and weight availability were not confirmed during the 2026-08-11 research pass.
  Touches: `RESEARCH.md`, `Roadmap_Blocked.md`
  Acceptance: The repository's LICENSE file, weight availability and stated VRAM requirement are read and recorded; the result is either a permissive-and-light candidate promoted to a GPU-validation entry in Roadmap_Blocked.md beside ROSE, or a dated rejection line in RESEARCH.md naming the blocking reason. No integration work in this item.
  Complexity: S

- [ ] P3 -- RM-269: The offline guarantee -- the project's strongest differentiator -- is buried
  Why: A commercial competitor charges for exactly this guarantee, and the forums where the target users ask this question still answer that it cannot be done at all.
  Evidence: EchoSubs AI sells "100% offline-first... your source files never leave your computer" and air-gap compatibility at $5.99/mo or $49 lifetime, and explicitly positions against "Open Source CLI tools (GitHub)" as being for people who can compile Python scripts -- https://www.echosubs.com/hardcoded-subtitle-remover-offline. Meanwhile the standing answer on VideoHelp and Doom9 is still that hardcoded subtitles cannot be removed without cropping or overlaying -- https://forum.videohelp.com/threads/418726-Is-there-a-way-to-remove-hardcoded-subtitles-without-cropping. `README.md:22` opens on capability, not on the local-only property; the network posture is inferable from the feature list but never stated as a guarantee.
  Touches: `README.md`
  Acceptance: The Overview states plainly that all processing is local, that no account or upload is required, that the only outbound requests are the opt-in update check and opt-in crash reporting, and names the flags that disable them -- verified against `backend/update_check.py` and `backend/crash_reporter.py` rather than asserted.
  Complexity: S
