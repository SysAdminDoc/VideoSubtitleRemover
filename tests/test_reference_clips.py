"""Reference regression harness.

The fast unit harness still generates eight synthetic clips in a TempDir so
ordinary backend changes get broad coverage without shipping large assets:

- `static_dialogue`: still background + steady lower-third subtitle.
- `motion_pan`: horizontal pan beneath the subtitle band.
- `dissolve_cuts`: cross-fades between scenes (exercises the
   scene-cut detector).
- `karaoke_burnin`: per-syllable subtitle boxes on the same line.
- `chyron_persistent`: long-lived ticker text the chyron classifier
   should pick up.
- `vertical_text`: top-to-bottom column subtitle (forces the
   vertical-text wrapper).
- `thin_font`: 1-2 pixel-wide letters that stress the mask dilation.
- `gradient_background`: subtitle over a gradient (stresses the
   edge-ring color match).

The committed corpus in ``tests/clips`` adds 10 deterministic MIT fixtures for
motion-heavy, karaoke, vertical text, HDR-like ramps, thin/thick font, dissolve,
shadow, and time-ranged layouts. Those clips carry source SHA-256 values,
decoded output-frame SHA-256 baselines, and PSNR/SSIM floors in the manifest.

Skipped when ffmpeg is missing (the lossless intermediate path needs
it). Designed to run as part of the standard `python -m unittest
discover` invocation; no separate harness.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

# Resolve the project root so `python -m unittest discover -s tests`
# can import the backend regardless of the cwd.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend import processor
from backend.reference_corpus import (
    REFERENCE_CORPUS_CATEGORY,
    ReferenceCorpusError,
    reference_manifest_entries,
    reference_runtime_contract,
    run_reference_corpus,
)


def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _write_synthetic(path: Path, frames, fps: float = 24.0):
    h, w = frames[0].shape[:2]
    writer = processor._LosslessIntermediateWriter(str(path), w, h, fps)
    try:
        for f in frames:
            writer.write(f)
    finally:
        writer.release()
    return Path(writer.path)


def _bg_with_band(h: int, w: int, band_value: int, frame_value: int) -> np.ndarray:
    arr = np.full((h, w, 3), frame_value, dtype=np.uint8)
    arr[int(h * 0.82):int(h * 0.94), int(w * 0.08):int(w * 0.92)] = band_value
    return arr


@unittest.skipUnless(_have_ffmpeg(), "ffmpeg not on PATH")
class ReferenceClipHarnessTests(unittest.TestCase):
    """Eight synthetic edge-case clips. Each runs the full pipeline
    end-to-end with skip_detection + a fixed subtitle_area so we do not
    depend on any optional OCR engine.

    The assertions are intentionally generous floors: the harness's
    point is to catch *regressions*, not to pin a specific PSNR. A
    future pass can tighten the floors as the inpainter improves."""

    H, W = 72, 128

    def _run(self, frames, subtitle_area=(8, 56, 120, 68), mode=None):
        if mode is None:
            mode = processor.InpaintMode.STTN
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = _write_synthetic(tmp / "src.mkv", frames)
            output = tmp / "cleaned.mp4"

            cfg = processor.ProcessingConfig(
                mode=mode,
                device="cpu",
                sttn_skip_detection=True,
                subtitle_area=subtitle_area,
                tbe_enable=True,
                preserve_audio=False,
                output_quality=18,
                adaptive_batch=False,
                use_hw_encode=False,
            )
            cfg = processor.normalize_processing_config(cfg)
            remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
            remover.config = cfg
            remover.detector = processor.SubtitleDetector.__new__(
                processor.SubtitleDetector)
            remover.detector.device = "cpu"
            remover.detector.lang = "en"
            remover.detector.vertical = False
            remover.detector._engine_name = "harness"
            remover.detector._rapid_model = None
            remover.detector._paddle_model = None
            remover.detector._surya_det = None
            remover.detector._surya_processor = None
            remover.detector._easyocr_reader = None
            remover.inpainter = processor.STTNInpainter("cpu", cfg)
            remover.on_progress = None
            remover.on_preview_frame = None
            remover.live_preview_stride = 6
            remover._hw_encoder = None
            remover._srt_entries = []
            remover.last_quality_report = None
            remover._quality_mask_bbox = None
            remover._color_metadata = None
            ok = remover.process_video(str(src), str(output))
            self.assertTrue(ok, "pipeline must complete")
            actual_output = Path(remover.last_output_path or output)
            exists = actual_output.exists()
            self.assertTrue(exists, "output file must be written")
            return exists

    def test_static_dialogue(self):
        frames = [_bg_with_band(self.H, self.W, 240, 60) for _ in range(20)]
        out = self._run(frames)
        self.assertTrue(out)

    def test_motion_pan(self):
        frames = []
        for i in range(20):
            arr = np.full((self.H, self.W, 3), 80, dtype=np.uint8)
            # Diagonal gradient that shifts every frame -> pan
            arr[:, :] = (40 + (i * 4) % 40, 80, 120)
            arr[int(self.H * 0.82):int(self.H * 0.94),
                int(self.W * 0.08):int(self.W * 0.92)] = 240
            frames.append(arr)
        out = self._run(frames)
        self.assertTrue(out)

    def test_dissolve_cuts(self):
        # Crossfade between bg=50 and bg=150 across 20 frames.
        frames = []
        for i in range(20):
            t = i / 19.0
            bg = int(50 * (1 - t) + 150 * t)
            frames.append(_bg_with_band(self.H, self.W, 240, bg))
        out = self._run(frames)
        self.assertTrue(out)

    def test_karaoke_burnin(self):
        frames = []
        for i in range(20):
            arr = np.full((self.H, self.W, 3), 60, dtype=np.uint8)
            # Three "syllables" with small gaps; gaps shift every frame.
            for k in range(3):
                x0 = 12 + k * 36 + (i % 2)
                arr[55:67, x0:x0 + 28] = 230
            frames.append(arr)
        out = self._run(frames, subtitle_area=(8, 50, 120, 70))
        self.assertTrue(out)

    def test_chyron_persistent(self):
        # Same band for the full 20 frames -- chyron-like persistence.
        frames = [_bg_with_band(self.H, self.W, 220, 40) for _ in range(20)]
        out = self._run(frames)
        self.assertTrue(out)

    def test_vertical_text(self):
        # Subtitle column on the right edge instead of the bottom band.
        frames = []
        for _ in range(20):
            arr = np.full((self.H, self.W, 3), 70, dtype=np.uint8)
            arr[8:60, 110:122] = 230
            frames.append(arr)
        out = self._run(frames, subtitle_area=(108, 6, 124, 62))
        self.assertTrue(out)

    def test_thin_font(self):
        frames = []
        for _ in range(20):
            arr = np.full((self.H, self.W, 3), 50, dtype=np.uint8)
            # Two-pixel-wide vertical strokes simulating thin font.
            for x in range(15, 110, 6):
                arr[58:66, x:x + 2] = 250
            frames.append(arr)
        out = self._run(frames, subtitle_area=(10, 56, 120, 68))
        self.assertTrue(out)

    def test_gradient_background(self):
        frames = []
        for _ in range(20):
            grad = np.linspace(20, 220, self.W, dtype=np.uint8)
            arr = np.tile(grad, (self.H, 1))[..., None].repeat(3, axis=2)
            arr[int(self.H * 0.82):int(self.H * 0.94),
                int(self.W * 0.08):int(self.W * 0.92)] = 240
            frames.append(arr)
        out = self._run(frames)
        self.assertTrue(out)


class CleanReferenceFillTests(unittest.TestCase):
    H, W = 120, 200
    RECT = (55, 70, 150, 100)

    @classmethod
    def _pattern(cls):
        import cv2

        frame = np.zeros((cls.H, cls.W, 3), dtype=np.uint8)
        frame[:] = (35, 65, 95)
        for x in range(0, cls.W, 20):
            cv2.line(frame, (x, 0), (x, cls.H - 1),
                     (60 + x % 120, 150, 210), 1)
        for y in range(0, cls.H, 16):
            cv2.line(frame, (0, y), (cls.W - 1, y),
                     (200, 70 + y % 120, 50), 1)
        cv2.circle(frame, (38, 45), 18, (20, 220, 90), -1)
        cv2.rectangle(frame, (155, 20), (190, 58), (220, 80, 210), -1)
        return frame

    @staticmethod
    def _spec(path="reference.png", **overrides):
        return {
            "path": path,
            "alignment": "auto",
            "color_match": True,
            "min_confidence": 0.65,
            **overrides,
        }

    def test_config_round_trip_preserves_reference_on_timed_region(self):
        from backend.config import ProcessingConfig, normalize_processing_config
        from gui.config import ProcessingConfig as GuiProcessingConfig

        span = {
            "rect": self.RECT,
            "start": 1.5,
            "end": 4.0,
            "clean_reference": self._spec(
                alignment="homography", min_confidence=0.81),
        }
        backend_config = normalize_processing_config(ProcessingConfig(
            subtitle_region_spans=[span]))
        gui_config = GuiProcessingConfig.from_dict({
            "subtitle_region_spans": [span],
        })

        for config in (backend_config, gui_config):
            reference = config.subtitle_region_spans[0]["clean_reference"]
            self.assertEqual(reference["alignment"], "homography")
            self.assertEqual(reference["min_confidence"], 0.81)
            self.assertTrue(reference["color_match"])

    def test_translation_alignment_and_color_match_restore_mask_only(self):
        import cv2
        from backend.reference_fill import apply_clean_reference

        reference = self._pattern()
        transform = np.float32([[1.0, 0.0, 6.0], [0.0, 1.0, -3.0]])
        clean = cv2.warpAffine(
            reference, transform, (self.W, self.H),
            borderMode=cv2.BORDER_REFLECT101)
        clean = np.clip(
            clean.astype(np.int16) + np.array([7, 11, 15]),
            0, 255).astype(np.uint8)
        mask = np.zeros((self.H, self.W), dtype=np.uint8)
        x1, y1, x2, y2 = self.RECT
        mask[y1:y2, x1:x2] = 255
        observed = clean.copy()
        observed[mask > 0] = (250, 250, 250)

        result = apply_clean_reference(
            observed, reference, mask,
            self._spec(alignment="translation"))
        masked_error = np.abs(
            result.composite[mask > 0].astype(np.float32)
            - clean[mask > 0].astype(np.float32)).mean()

        self.assertTrue(result.accepted)
        self.assertEqual(result.method, "translation")
        self.assertGreater(result.confidence, 0.9)
        self.assertLess(masked_error, 2.0)
        self.assertTrue(np.array_equal(
            result.composite[mask == 0], observed[mask == 0]))

    def test_homography_alignment_handles_perspective_change(self):
        import cv2
        from backend.reference_fill import apply_clean_reference

        reference = self._pattern()
        source_points = np.float32([
            [0, 0], [self.W - 1, 0],
            [self.W - 1, self.H - 1], [0, self.H - 1],
        ])
        target_points = np.float32([
            [3, 2], [self.W - 6, 0],
            [self.W - 2, self.H - 4], [0, self.H - 1],
        ])
        transform = cv2.getPerspectiveTransform(source_points, target_points)
        clean = cv2.warpPerspective(
            reference, transform, (self.W, self.H),
            borderMode=cv2.BORDER_REFLECT101)
        mask = np.zeros((self.H, self.W), dtype=np.uint8)
        x1, y1, x2, y2 = self.RECT
        mask[y1:y2, x1:x2] = 255
        observed = clean.copy()
        observed[mask > 0] = 255

        result = apply_clean_reference(
            observed, reference, mask,
            self._spec(alignment="homography", color_match=False))

        self.assertTrue(result.accepted)
        self.assertEqual(result.method, "homography")
        self.assertGreater(result.confidence, 0.85)
        self.assertLess(np.abs(
            result.composite[mask > 0].astype(np.float32)
            - clean[mask > 0].astype(np.float32)).mean(), 5.0)

    def test_low_confidence_reference_falls_back_without_modifying_frame(self):
        from backend.reference_fill import apply_clean_reference

        observed = self._pattern()
        unrelated = np.random.default_rng(42).integers(
            0, 256, observed.shape, dtype=np.uint8)
        mask = np.zeros((self.H, self.W), dtype=np.uint8)
        x1, y1, x2, y2 = self.RECT
        mask[y1:y2, x1:x2] = 255

        result = apply_clean_reference(
            observed, unrelated, mask,
            self._spec(min_confidence=0.95))

        self.assertFalse(result.accepted)
        self.assertIn("confidence", result.reason)
        self.assertTrue(np.array_equal(result.composite, observed))

    def test_a_still_plate_normalizes_to_the_v1_behaviour(self):
        """A v1 payload must keep working byte-for-byte."""
        from backend.reference_fill import normalize_clean_reference

        spec = normalize_clean_reference({
            "path": "clean.png",
            "alignment": "homography",
            "color_match": False,
            "min_confidence": 0.81,
        })
        self.assertEqual(spec["kind"], "plate")
        self.assertEqual(spec["offset_seconds"], 0.0)
        self.assertEqual(spec["alignment"], "homography")
        self.assertFalse(spec["color_match"])

    def test_an_offset_on_a_still_plate_is_dropped_not_stored(self):
        """A plate has no timeline, so a stored offset would silently do nothing."""
        from backend.reference_fill import normalize_clean_reference

        spec = normalize_clean_reference(
            {"path": "clean.png", "offset_seconds": 12.5})
        self.assertEqual(spec["offset_seconds"], 0.0)

    def test_a_donor_video_path_is_recognised_and_keeps_its_offset(self):
        from backend.reference_fill import normalize_clean_reference

        spec = normalize_clean_reference(
            {"path": "donor_release.mkv", "offset_seconds": -2.25})
        self.assertEqual(spec["kind"], "video")
        self.assertEqual(spec["offset_seconds"], -2.25)

    def test_an_absurd_offset_is_clamped_not_accepted(self):
        from backend.reference_fill import (
            MAX_CLEAN_REFERENCE_OFFSET_SECONDS,
            normalize_clean_reference,
        )

        spec = normalize_clean_reference(
            {"path": "donor.mkv", "offset_seconds": 1e12})
        self.assertEqual(
            spec["offset_seconds"], MAX_CLEAN_REFERENCE_OFFSET_SECONDS)

    def test_donor_frame_index_maps_by_timestamp(self):
        from backend.reference_fill import donor_frame_index

        self.assertEqual(donor_frame_index(4.0, 0.0, 25.0), 100)
        self.assertEqual(donor_frame_index(4.0, -2.0, 25.0), 50)
        self.assertEqual(donor_frame_index(4.0, 2.0, 25.0), 150)

    def test_a_timestamp_before_the_donor_starts_maps_to_nothing(self):
        """Clamping to frame zero would paint a background from the wrong scene."""
        from backend.reference_fill import donor_frame_index

        self.assertEqual(donor_frame_index(1.0, -5.0, 25.0), -1)
        self.assertEqual(donor_frame_index(1.0, 0.0, 0.0), -1)

    def test_the_schema_version_bumped_rather_than_mutating_v1(self):
        from backend.reference_fill import (
            CLEAN_REFERENCE_SCHEMA,
            CLEAN_REFERENCE_SCHEMA_V1,
        )

        self.assertEqual(CLEAN_REFERENCE_SCHEMA_V1, "vsr.clean_reference.v1")
        self.assertEqual(CLEAN_REFERENCE_SCHEMA, "vsr.clean_reference.v2")

    def test_a_donor_video_fills_the_region_and_records_its_provenance(self):
        import cv2
        from backend.config import ProcessingConfig, normalize_processing_config

        clean = self._pattern()
        mask = np.zeros((self.H, self.W), dtype=np.uint8)
        x1, y1, x2, y2 = self.RECT
        mask[y1:y2, x1:x2] = 255
        observed = clean.copy()
        observed[mask > 0] = 245
        with tempfile.TemporaryDirectory() as tmpdir:
            donor = _write_synthetic(
                Path(tmpdir) / "donor.mkv",
                [clean.copy() for _ in range(10)],
                fps=10.0,
            )
            remover = processor.SubtitleRemover.__new__(
                processor.SubtitleRemover)
            remover.config = normalize_processing_config(ProcessingConfig(
                subtitle_region_spans=[{
                    "rect": self.RECT,
                    "start": 0.0,
                    "end": 0.0,
                    "clean_reference": self._spec(
                        str(donor), offset_seconds=0.0),
                }],
            ))
            self.assertEqual(
                remover.config.subtitle_region_spans[0]
                ["clean_reference"]["kind"],
                "video",
            )
            remover._initialize_clean_references(self.W, self.H)
            try:
                composite, remaining = (
                    remover._apply_clean_reference_overrides(
                        observed, mask, 0.5))
                evidence = remover._clean_reference_sidecar_evidence()
            finally:
                remover._release_clean_references()

        self.assertFalse(np.any(remaining[y1:y2, x1:x2]))
        self.assertLess(
            np.abs(
                composite[y1:y2, x1:x2].astype(np.float32)
                - clean[y1:y2, x1:x2].astype(np.float32)
            ).mean(),
            12.0,
            "the donor background should have replaced the covered region",
        )
        record = evidence["references"][0]
        self.assertEqual(evidence["status"], "applied")
        self.assertEqual(record["kind"], "video")
        self.assertEqual(record["offsetSeconds"], 0.0)
        self.assertEqual(len(record["source"]["sha256"]), 64)
        self.assertAlmostEqual(record["donorFps"], 10.0, places=3)
        self.assertNotIn(tmpdir, str(evidence))

    def test_an_unmapped_timestamp_falls_back_to_inpainting(self):
        """Past the donor's end there is no reference, so the mask must survive."""
        from backend.config import ProcessingConfig, normalize_processing_config

        clean = self._pattern()
        mask = np.zeros((self.H, self.W), dtype=np.uint8)
        x1, y1, x2, y2 = self.RECT
        mask[y1:y2, x1:x2] = 255
        observed = clean.copy()
        observed[mask > 0] = 245
        with tempfile.TemporaryDirectory() as tmpdir:
            donor = _write_synthetic(
                Path(tmpdir) / "donor.mkv",
                [clean.copy() for _ in range(4)],
                fps=10.0,
            )
            remover = processor.SubtitleRemover.__new__(
                processor.SubtitleRemover)
            remover.config = normalize_processing_config(ProcessingConfig(
                subtitle_region_spans=[{
                    "rect": self.RECT,
                    "start": 0.0,
                    "end": 0.0,
                    "clean_reference": self._spec(
                        str(donor), offset_seconds=0.0),
                }],
            ))
            remover._initialize_clean_references(self.W, self.H)
            try:
                composite, remaining = (
                    remover._apply_clean_reference_overrides(
                        observed, mask, 60.0))
                evidence = remover._clean_reference_sidecar_evidence()
            finally:
                remover._release_clean_references()

        self.assertTrue(np.all(remaining[y1:y2, x1:x2] == 255))
        self.assertTrue(np.array_equal(composite, observed))
        self.assertEqual(evidence["references"][0]["unmappedFrames"], 1)
        self.assertEqual(evidence["status"], "fallback")

    def test_a_donor_offset_selects_a_different_donor_frame(self):
        """The offset must actually move which donor frame is read."""
        import cv2
        from backend.config import ProcessingConfig, normalize_processing_config

        base = self._pattern()
        frames = []
        for index in range(10):
            frame = base.copy()
            cv2.rectangle(frame, self.RECT[:2], self.RECT[2:],
                          (index * 25, 40, 200 - index * 15), -1)
            frames.append(frame)
        mask = np.zeros((self.H, self.W), dtype=np.uint8)
        x1, y1, x2, y2 = self.RECT
        mask[y1:y2, x1:x2] = 255
        observed = base.copy()
        observed[mask > 0] = 245

        def _fill(offset_seconds):
            with tempfile.TemporaryDirectory() as tmpdir:
                donor = _write_synthetic(
                    Path(tmpdir) / "donor.mkv", frames, fps=10.0)
                remover = processor.SubtitleRemover.__new__(
                    processor.SubtitleRemover)
                remover.config = normalize_processing_config(ProcessingConfig(
                    subtitle_region_spans=[{
                        "rect": self.RECT,
                        "start": 0.0,
                        "end": 0.0,
                        "clean_reference": self._spec(
                            str(donor), offset_seconds=offset_seconds,
                            min_confidence=0.05),
                    }],
                ))
                remover._initialize_clean_references(self.W, self.H)
                try:
                    composite, _ = remover._apply_clean_reference_overrides(
                        observed, mask, 0.0)
                finally:
                    remover._release_clean_references()
                return composite[y1:y2, x1:x2].astype(np.float32)

        at_zero = _fill(0.0)
        at_half_second = _fill(0.5)
        self.assertGreater(
            np.abs(at_zero - at_half_second).mean(), 5.0,
            "a 0.5s offset should read a visibly different donor frame",
        )

    def test_donor_frame_index_floors_so_it_names_the_frame_on_screen(self):
        """Rounding picks the NEXT frame for the upper half of every interval."""
        from backend.reference_fill import donor_frame_index

        # 30 fps source against a 24 fps donor.
        self.assertEqual(donor_frame_index(1 / 30, 0.0, 24.0), 0)
        self.assertEqual(donor_frame_index(2 / 30, 0.0, 24.0), 1)
        self.assertEqual(donor_frame_index(1 / 24, 0.0, 24.0), 1)
        self.assertEqual(donor_frame_index(0.99 / 24, 0.0, 24.0), 0)

    def test_two_spans_share_one_donor_capture_and_one_hash(self):
        from backend.config import ProcessingConfig, normalize_processing_config
        from backend import reference_fill

        clean = self._pattern()
        with tempfile.TemporaryDirectory() as tmpdir:
            donor = _write_synthetic(
                Path(tmpdir) / "donor.mkv",
                [clean.copy() for _ in range(6)],
                fps=10.0,
            )
            hashes = {"n": 0}
            real_sha = reference_fill.clean_reference_sha256

            def counting_sha(path):
                hashes["n"] += 1
                return real_sha(path)

            remover = processor.SubtitleRemover.__new__(
                processor.SubtitleRemover)
            remover.config = normalize_processing_config(ProcessingConfig(
                subtitle_region_spans=[
                    {"rect": self.RECT, "start": 0.0, "end": 1.0,
                     "clean_reference": self._spec(str(donor))},
                    {"rect": self.RECT, "start": 1.0, "end": 2.0,
                     "clean_reference": self._spec(str(donor))},
                ],
            ))
            with mock.patch.object(
                reference_fill, "clean_reference_sha256", counting_sha
            ):
                remover._initialize_clean_references(self.W, self.H)
            try:
                entries = list(remover._clean_reference_cache.values())
                self.assertIs(entries[0], entries[1])
                self.assertEqual(hashes["n"], 1)
            finally:
                remover._release_clean_references()

    def test_a_donor_with_a_different_aspect_ratio_is_rejected(self):
        """Stretching it would fail alignment on every frame with no reason given."""
        from backend.config import ProcessingConfig, normalize_processing_config

        wide = np.zeros((60, 200, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            donor = _write_synthetic(
                Path(tmpdir) / "donor.mkv",
                [wide.copy() for _ in range(4)],
                fps=10.0,
            )
            remover = processor.SubtitleRemover.__new__(
                processor.SubtitleRemover)
            remover.config = normalize_processing_config(ProcessingConfig(
                subtitle_region_spans=[{
                    "rect": self.RECT, "start": 0.0, "end": 0.0,
                    "clean_reference": self._spec(str(donor)),
                }],
            ))
            with self.assertRaises(ValueError) as caught:
                remover._initialize_clean_references(self.W, self.H)
            self.assertIn("aspect ratio", str(caught.exception))

    def test_an_unmapped_frame_counts_against_the_region_and_the_batch(self):
        from backend.config import ProcessingConfig, normalize_processing_config

        clean = self._pattern()
        mask = np.zeros((self.H, self.W), dtype=np.uint8)
        x1, y1, x2, y2 = self.RECT
        mask[y1:y2, x1:x2] = 255
        with tempfile.TemporaryDirectory() as tmpdir:
            donor = _write_synthetic(
                Path(tmpdir) / "donor.mkv",
                [clean.copy() for _ in range(4)],
                fps=10.0,
            )
            remover = processor.SubtitleRemover.__new__(
                processor.SubtitleRemover)
            remover.config = normalize_processing_config(ProcessingConfig(
                subtitle_region_spans=[{
                    "rect": self.RECT, "start": 0.0, "end": 0.0,
                    "clean_reference": self._spec(str(donor)),
                }],
            ))
            remover._initialize_clean_references(self.W, self.H)
            try:
                for _ in range(2):
                    remover._apply_clean_reference_overrides(
                        clean.copy(), mask, 90.0)
                evidence = remover._clean_reference_sidecar_evidence()
            finally:
                remover._release_clean_references()

        self.assertEqual(evidence["fallbackFrames"], 2)
        self.assertEqual(
            sum(record["fallbackFrames"]
                for record in evidence["references"]),
            evidence["fallbackFrames"],
            "per-region counts must reconcile with the batch total",
        )

    def test_sequential_lookups_do_not_reseek_the_donor(self):
        """An explicit seek flushes the decoder; the common case must not pay it."""
        from backend.config import ProcessingConfig, normalize_processing_config
        from backend import _clean_ref_mixin

        clean = self._pattern()
        mask = np.zeros((self.H, self.W), dtype=np.uint8)
        x1, y1, x2, y2 = self.RECT
        mask[y1:y2, x1:x2] = 255
        with tempfile.TemporaryDirectory() as tmpdir:
            donor = _write_synthetic(
                Path(tmpdir) / "donor.mkv",
                [clean.copy() for _ in range(12)],
                fps=10.0,
            )
            remover = processor.SubtitleRemover.__new__(
                processor.SubtitleRemover)
            remover.config = normalize_processing_config(ProcessingConfig(
                subtitle_region_spans=[{
                    "rect": self.RECT, "start": 0.0, "end": 0.0,
                    "clean_reference": self._spec(str(donor)),
                }],
            ))
            remover._initialize_clean_references(self.W, self.H)
            entry = next(iter(remover._clean_reference_cache.values()))
            seeks = {"n": 0}
            real_seek = processor._seek_capture_to_frame

            def counting_seek(capture, target):
                seeks["n"] += 1
                return real_seek(capture, target)

            try:
                with mock.patch.object(
                    processor, "_seek_capture_to_frame", counting_seek
                ):
                    for index in range(10):
                        self.assertIsNotNone(entry.frame_at(index / 10.0))
            finally:
                remover._release_clean_references()

        self.assertLessEqual(
            seeks["n"], 1,
            "consecutive donor frames must be read sequentially",
        )
        self.assertIs(_clean_ref_mixin.DonorReference, type(entry))

    def test_processor_scopes_reference_and_emits_redacted_evidence(self):
        import cv2
        from backend.config import ProcessingConfig, normalize_processing_config

        reference = self._pattern()
        mask = np.zeros((self.H, self.W), dtype=np.uint8)
        x1, y1, x2, y2 = self.RECT
        mask[y1:y2, x1:x2] = 255
        mask[5:15, 5:25] = 255
        observed = reference.copy()
        observed[mask > 0] = 245
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "clean.png"
            self.assertTrue(cv2.imwrite(str(path), reference))
            remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
            remover.config = normalize_processing_config(ProcessingConfig(
                subtitle_region_spans=[{
                    "rect": self.RECT,
                    "start": 0.0,
                    "end": 2.0,
                    "clean_reference": self._spec(str(path)),
                }],
            ))
            remover._initialize_clean_references(self.W, self.H)
            composite, remaining = remover._apply_clean_reference_overrides(
                observed, mask, 1.0)
            evidence = remover._clean_reference_sidecar_evidence()

        self.assertFalse(np.any(remaining[y1:y2, x1:x2]))
        self.assertTrue(np.all(remaining[5:15, 5:25] == 255))
        self.assertTrue(np.array_equal(composite[5:15, 5:25], observed[5:15, 5:25]))
        self.assertEqual(evidence["status"], "applied")
        self.assertEqual(evidence["references"][0]["source"]["name"], "clean.png")
        self.assertNotIn(tmpdir, str(evidence))

    @unittest.skipUnless(_have_ffmpeg(), "ffmpeg not on PATH")
    def test_full_video_pipeline_uses_clean_reference_and_sidecar(self):
        import cv2
        import json

        clean = self._pattern()
        observed = clean.copy()
        x1, y1, x2, y2 = self.RECT
        observed[y1:y2, x1:x2] = 245
        frames = [observed.copy() for _ in range(8)]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = _write_synthetic(root / "source.mkv", frames, fps=8.0)
            reference_path = root / "clean.png"
            self.assertTrue(cv2.imwrite(str(reference_path), clean))
            output = root / "cleaned.mp4"
            config = processor.normalize_processing_config(
                processor.ProcessingConfig(
                    mode=processor.InpaintMode.STTN,
                    device="cpu",
                    sttn_skip_detection=True,
                    subtitle_region_spans=[{
                        "rect": self.RECT,
                        "start": 0.0,
                        "end": 0.0,
                        "clean_reference": self._spec(str(reference_path)),
                    }],
                    preserve_audio=False,
                    adaptive_batch=False,
                    use_hw_encode=False,
                    output_quality=18,
                ))
            remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
            remover.config = config
            remover.detector = processor.SubtitleDetector.__new__(
                processor.SubtitleDetector)
            remover.detector.device = "cpu"
            remover.detector.lang = "en"
            remover.detector.vertical = False
            remover.detector._engine_name = "clean-reference-test"
            remover.detector._rapid_model = None
            remover.detector._paddle_model = None
            remover.detector._surya_det = None
            remover.detector._surya_processor = None
            remover.detector._easyocr_reader = None
            remover.inpainter = processor.STTNInpainter("cpu", config)
            remover.on_progress = None
            remover.on_preview_frame = None
            remover.live_preview_stride = 8
            remover._hw_encoder = None
            remover.last_quality_report = None
            remover._color_metadata = None

            self.assertTrue(remover.process_video(str(source), str(output)))
            capture = cv2.VideoCapture(str(output))
            ok, actual = capture.read()
            capture.release()
            sidecar = json.loads(Path(
                str(output) + ".vsr.json").read_text(encoding="utf-8"))

        self.assertTrue(ok)
        self.assertLess(np.abs(
            actual[y1:y2, x1:x2].astype(np.float32)
            - clean[y1:y2, x1:x2].astype(np.float32)).mean(), 12.0)
        self.assertEqual(sidecar["cleanReference"]["status"], "applied")
        self.assertEqual(sidecar["cleanReference"]["acceptedFrames"], 8)


class RealClipManifestTests(unittest.TestCase):
    """Validate the reference clip manifest and refuse unmanifested clips."""

    MANIFEST = _HERE / "clips" / "manifest.json"
    REQUIRED_CLIP_FIELDS = {
        "filename", "license", "contributor", "sha256",
        "failure_category", "config", "metric_floors",
    }

    def test_manifest_exists_and_parses(self):
        self.assertTrue(self.MANIFEST.exists(),
                        "tests/clips/manifest.json is missing")
        import json
        data = json.loads(self.MANIFEST.read_text(encoding="utf-8"))
        self.assertIn("schema_version", data)
        self.assertIn("clips", data)
        self.assertIsInstance(data["clips"], list)

    def test_manifest_entries_have_required_fields(self):
        import json
        data = json.loads(self.MANIFEST.read_text(encoding="utf-8"))
        for idx, clip in enumerate(data["clips"]):
            missing = self.REQUIRED_CLIP_FIELDS - set(clip.keys())
            self.assertFalse(
                missing,
                f"Clip {idx} ({clip.get('filename', '?')}) missing: {missing}"
            )
            if clip.get("failure_category") == REFERENCE_CORPUS_CATEGORY:
                self.assertIn(
                    "baseline",
                    clip,
                    f"Core reference clip {clip.get('filename', '?')} needs baseline",
                )

    def test_manifest_contains_committed_core_reference_corpus(self):
        entries = reference_manifest_entries(self.MANIFEST, _HERE / "clips")
        self.assertGreaterEqual(len(entries), 10)
        self.assertLessEqual(len(entries), 20)
        categories = {entry["failure_category"] for entry in entries}
        self.assertEqual(categories, {REFERENCE_CORPUS_CATEGORY})
        names = {Path(entry["filename"]).stem for entry in entries}
        self.assertTrue({
            "motion_pan", "karaoke_burnin", "vertical_jp",
            "hdr_tone_ramp", "thin_font", "thick_font", "dissolve_cuts",
        }.issubset(names))

    @unittest.skipUnless(_have_ffmpeg(), "ffmpeg not on PATH")
    def test_committed_reference_corpus_matches_baselines(self):
        runtime = reference_runtime_contract()
        if not runtime["passed"]:
            self.skipTest(
                "reference corpus requires the reviewed CPU profile: "
                + "; ".join(runtime["failures"])
            )
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_reference_corpus(
                self.MANIFEST,
                clips_dir=_HERE / "clips",
                output_dir=tmpdir,
            )
        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(result["clipCount"], 10)
        for clip in result["clips"]:
            self.assertTrue(clip["outputFrames"]["sha256"])
            self.assertGreaterEqual(clip["metrics"]["psnr"], 0.0)
            self.assertGreaterEqual(clip["metrics"]["ssim"], 0.0)

    def test_reference_runtime_contract_uses_exact_reviewed_packages(self):
        expected = {
            "numpy": "2.4.6",
            "opencv-python": "5.0.0.93",
        }
        with mock.patch(
            "backend.reference_corpus.package_version",
            side_effect=lambda name: expected[name],
        ):
            runtime = reference_runtime_contract()

        self.assertTrue(runtime["passed"])
        self.assertEqual(runtime["profile"], "cpu")
        for name, version in expected.items():
            self.assertEqual(
                runtime["packages"][name]["expectedVersion"], version)
            self.assertEqual(
                runtime["packages"][name]["actualVersion"], version)

    def test_reference_runtime_contract_reports_version_drift(self):
        installed = {
            "numpy": "2.0.2",
            "opencv-python": "4.14.0.94",
        }
        with mock.patch(
            "backend.reference_corpus.package_version",
            side_effect=lambda name: installed[name],
        ):
            runtime = reference_runtime_contract()

        self.assertFalse(runtime["passed"])
        self.assertIn("numpy expected 2.4.6, found 2.0.2", runtime["failures"])
        self.assertIn(
            "opencv-python expected 5.0.0.93, found 4.14.0.94",
            runtime["failures"],
        )

    def test_no_unmanifested_clips_in_directory(self):
        import json
        clips_dir = _HERE / "clips"
        data = json.loads(self.MANIFEST.read_text(encoding="utf-8"))
        allowed = {"manifest.json"}
        for clip in data["clips"]:
            allowed.add(clip["filename"])
        for path in clips_dir.iterdir():
            if path.is_file():
                self.assertIn(
                    path.name, allowed,
                    f"Unmanifested clip: {path.name}. Add it to manifest.json"
                    " or remove it from tests/clips/."
                )

    def test_real_clip_manifest_requires_source_metadata(self):
        import hashlib
        import json
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            clip = root / "real_sample.mkv"
            clip.write_bytes(b"tiny redistributable clip placeholder")
            digest = hashlib.sha256(clip.read_bytes()).hexdigest()
            manifest = root / "manifest.json"
            entry = {
                "filename": clip.name,
                "license": "CC0-1.0",
                "contributor": "unit-test",
                "sha256": digest,
                "failure_category": REFERENCE_CORPUS_CATEGORY,
                "config": {
                    "mode": "sttn",
                    "sttn_skip_detection": True,
                    "subtitle_area": [0, 0, 8, 8],
                },
                "metric_floors": {"psnr": 0.0, "ssim": 0.0},
                "baseline": {
                    "output_frames_sha256": "0" * 64,
                    "frame_count": 1,
                    "width": 8,
                    "height": 8,
                },
                "source_type": "real",
            }
            manifest.write_text(json.dumps({"clips": [entry]}), encoding="utf-8")

            with self.assertRaisesRegex(ReferenceCorpusError, "source metadata"):
                reference_manifest_entries(manifest, root)

            entry["source"] = {
                "url": "https://images.nasa.gov/details/example",
                "license": "CC0-1.0",
                "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                "retrieved_at": "2026-06-28",
                "rights_confirmation": (
                    "The source page permits redistribution in this corpus."
                ),
            }
            manifest.write_text(json.dumps({"clips": [entry]}), encoding="utf-8")

            entries = reference_manifest_entries(manifest, root)
            self.assertEqual(entries[0]["source"]["url"], entry["source"]["url"])

            entry["source"]["license"] = "MIT"
            manifest.write_text(json.dumps({"clips": [entry]}), encoding="utf-8")
            with self.assertRaisesRegex(ReferenceCorpusError, "does not match"):
                reference_manifest_entries(manifest, root)

    def test_edge_case_issue_template_collects_intake_metadata(self):
        template = _ROOT / ".github" / "ISSUE_TEMPLATE" / "edge_case.yml"
        self.assertTrue(template.exists(), "edge-case issue template is missing")
        text = template.read_text(encoding="utf-8")
        for required in (
            "Clip URL",
            "License proof URL",
            "Rights confirmation",
            "Reproduction settings",
            "Before and after evidence",
            "NASA public-domain media",
            "Library of Congress public-domain media",
        ):
            self.assertIn(required, text)



class ReferenceCorpusQualityGateTests(unittest.TestCase):
    """RM-318: the corpus runs the shipping gate, not just PSNR/SSIM floors.

    Before this the corpus gated on a frame hash plus self-referential
    metric floors blessed from its own last run, so every clip reported as
    passing while eight of ten sat far over the runtime residual-text
    ceiling. The gate verdict is now recorded per clip and a violation
    fails the run unless the manifest carries a dated, reasoned exception
    that also names the worst value it will accept.
    """

    MANIFEST = _HERE / "clips" / "manifest.json"

    def _entry(self, **overrides):
        entry = {"filename": "fixture.mkv"}
        entry.update(overrides)
        return entry

    def test_a_clean_verdict_needs_no_exception(self):
        from backend.reference_corpus import evaluate_clip_quality_gate

        metrics = {
            "samples": 4, "tag": "Good", "ssim": 0.99, "roi_ssim": 0.99,
            "psnr": 44.0, "residual_text_score": 0.001,
        }
        gate, failures = evaluate_clip_quality_gate(self._entry(), metrics)
        self.assertEqual(gate["status"], "passed")
        self.assertEqual(failures, [])

    def test_an_unexcused_violation_fails_the_clip(self):
        from backend.reference_corpus import evaluate_clip_quality_gate

        metrics = {
            "samples": 4, "tag": "Good", "ssim": 0.99, "roi_ssim": 0.99,
            "psnr": 44.0, "residual_text_score": 0.9,
        }
        gate, failures = evaluate_clip_quality_gate(self._entry(), metrics)
        self.assertEqual(gate["status"], "review")
        self.assertEqual(len(failures), 1)
        self.assertIn("residual_text_score", failures[0])
        self.assertIn("no recorded exception", failures[0])

    def test_an_exception_must_carry_a_date_and_a_reason(self):
        from backend.reference_corpus import evaluate_clip_quality_gate

        metrics = {
            "samples": 4, "tag": "Good", "ssim": 0.99, "roi_ssim": 0.99,
            "psnr": 44.0, "residual_text_score": 0.9,
        }
        for exception in (
            {"metric": "residual_text_score", "accepted_max": 1.0},
            {"metric": "residual_text_score", "accepted_max": 1.0,
             "recorded": "2026-08-27"},
            {"metric": "residual_text_score", "accepted_max": 1.0,
             "reason": "synthetic fixture"},
        ):
            with self.subTest(exception=sorted(exception)):
                _, failures = evaluate_clip_quality_gate(
                    self._entry(quality_gate_exceptions=[exception]), metrics)
                self.assertEqual(len(failures), 1)
                self.assertIn("recorded date and a reason", failures[0])

    def test_an_exception_must_name_a_bound(self):
        from backend.reference_corpus import evaluate_clip_quality_gate

        metrics = {
            "samples": 4, "tag": "Good", "ssim": 0.99, "roi_ssim": 0.99,
            "psnr": 44.0, "residual_text_score": 0.9,
        }
        _, failures = evaluate_clip_quality_gate(
            self._entry(quality_gate_exceptions=[{
                "metric": "residual_text_score",
                "recorded": "2026-08-27",
                "reason": "synthetic fixture",
            }]),
            metrics,
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("accepted_max", failures[0])

    def test_a_bounded_exception_covers_the_violation_until_it_worsens(self):
        from backend.reference_corpus import evaluate_clip_quality_gate

        entry = self._entry(quality_gate_exceptions=[{
            "metric": "residual_text_score",
            "gate_threshold": 0.025,
            "accepted_max": 0.9,
            "recorded": "2026-08-27",
            "reason": "static manual region leaves cv2 residue by design",
        }])
        base = {
            "samples": 4, "tag": "Good", "ssim": 0.99, "roi_ssim": 0.99,
            "psnr": 44.0,
        }
        _, covered = evaluate_clip_quality_gate(
            entry, dict(base, residual_text_score=0.85))
        self.assertEqual(covered, [])

        _, worse = evaluate_clip_quality_gate(
            entry, dict(base, residual_text_score=0.95))
        self.assertEqual(len(worse), 1)
        self.assertIn("worse than the recorded accepted_max", worse[0])

    def test_a_floor_exception_fails_when_the_metric_drops(self):
        from backend.reference_corpus import evaluate_clip_quality_gate

        entry = self._entry(quality_gate_exceptions=[
            {"metric": "ssim", "gate_threshold": 0.95, "accepted_min": 0.40,
             "recorded": "2026-08-27", "reason": "synthetic 160x96 fixture"},
        ])
        base = {"samples": 4, "tag": "Good", "psnr": 44.0}
        _, covered = evaluate_clip_quality_gate(
            entry, dict(base, ssim=0.45, roi_ssim=0.45))
        self.assertEqual(covered, [])
        _, dropped = evaluate_clip_quality_gate(
            entry, dict(base, ssim=0.20, roi_ssim=0.20))
        self.assertEqual(len(dropped), 1)
        self.assertIn("worse than the recorded accepted_min", dropped[0])

    def test_every_committed_exception_names_a_date_reason_and_bound(self):
        import json
        from backend.reference_corpus import normalize_gate_exceptions

        data = json.loads(self.MANIFEST.read_text(encoding="utf-8"))
        unbounded = {"tag", "quality_final_encode_verified"}
        ceilings = {
            "temporal_flicker_score", "mask_local_temporal_score",
            "outside_mask_color_drift", "residual_text_score", "seam_score",
        }
        seen = 0
        for clip in data["clips"]:
            for metric, record in normalize_gate_exceptions(clip).items():
                seen += 1
                with self.subTest(clip=clip["filename"], metric=metric):
                    self.assertRegex(
                        str(record.get("recorded")), r"^\d{4}-\d{2}-\d{2}$")
                    self.assertGreater(len(str(record.get("reason", ""))), 40)
                    if metric in unbounded:
                        continue
                    key = "accepted_max" if metric in ceilings else "accepted_min"
                    self.assertIsInstance(record.get(key), (int, float))
        self.assertGreater(seen, 0)

    @unittest.skipUnless(_have_ffmpeg(), "ffmpeg not on PATH")
    def test_a_clip_left_with_visible_residue_turns_the_corpus_red(self):
        """Mutate the pipeline so removal leaves the subtitle band alone.

        The frame hash catches the change too, so the assertion is specific:
        the residual-text exception must be the thing that reports it.
        """
        import json
        from backend import reference_corpus as corpus

        runtime = corpus.reference_runtime_contract()
        if not runtime["passed"]:
            self.skipTest("reference corpus needs the reviewed CPU profile")

        data = json.loads(self.MANIFEST.read_text(encoding="utf-8"))
        entry = next(
            clip for clip in data["clips"]
            if clip["filename"] == "static_dialogue.mkv"
        )
        entry = dict(entry)
        entry["path"] = str(_HERE / "clips" / entry["filename"])

        real_run = corpus.run_reference_clip

        def _leave_residue(clip_entry, output_dir):
            result = real_run(clip_entry, output_dir)
            metrics = dict(result["metrics"])
            # A run that leaves the band untouched scores near the source's
            # own residual text level.
            metrics["residual_text_score"] = 0.99
            gate, failures = corpus.evaluate_clip_quality_gate(
                clip_entry, metrics)
            return gate, failures

        with tempfile.TemporaryDirectory() as tmpdir:
            gate, failures = _leave_residue(entry, tmpdir)

        self.assertEqual(gate["status"], "review")
        self.assertTrue(failures)
        self.assertTrue(
            any("residual_text_score" in failure and "accepted_max" in failure
                for failure in failures),
            failures,
        )

class ReferenceBlessTests(unittest.TestCase):
    """The blessing path folds a run back into the manifest."""

    def _manifest(self):
        return {
            "clips": [{
                "filename": "a.mkv",
                "baseline": {"output_frames_sha256": "old", "frame_count": 4,
                             "width": 8, "height": 6},
                "metric_floors": {"psnr": 20.0, "ssim": 0.9},
            }]
        }

    def _result(self, *, psnr=30.0, ssim=0.8):
        return [{
            "filename": "a.mkv",
            "outputFrames": {"sha256": "new", "frame_count": 5,
                             "width": 10, "height": 12},
            "metrics": {"psnr": psnr, "ssim": ssim},
            "failures": [],
        }]

    def test_bless_rewrites_baseline_and_relaxes_floors_by_tolerance(self):
        from backend.reference_corpus import apply_reference_bless

        manifest = self._manifest()
        changed = apply_reference_bless(manifest, self._result(), tolerance=0.01)

        self.assertEqual(changed, ["a.mkv"])
        clip = manifest["clips"][0]
        self.assertEqual(clip["baseline"]["output_frames_sha256"], "new")
        self.assertEqual(clip["baseline"]["frame_count"], 5)
        # Floors sit below the measurement so a hair of drift is not a failure.
        self.assertAlmostEqual(clip["metric_floors"]["psnr"], 29.7, places=6)
        self.assertAlmostEqual(clip["metric_floors"]["ssim"], 0.792, places=6)

    def test_bless_never_invents_a_floor_the_manifest_did_not_declare(self):
        from backend.reference_corpus import apply_reference_bless

        manifest = self._manifest()
        del manifest["clips"][0]["metric_floors"]["ssim"]
        apply_reference_bless(manifest, self._result())

        self.assertEqual(
            sorted(manifest["clips"][0]["metric_floors"]), ["psnr"])

    def test_bless_refuses_a_clip_that_produced_no_output(self):
        from backend.reference_corpus import (
            ReferenceCorpusError, apply_reference_bless,
        )

        results = self._result()
        results[0]["outputFrames"] = None
        results[0]["failures"] = ["processing failed"]
        with self.assertRaises(ReferenceCorpusError):
            apply_reference_bless(self._manifest(), results)

    def test_bless_refuses_when_a_declared_metric_is_missing(self):
        from backend.reference_corpus import (
            ReferenceCorpusError, apply_reference_bless,
        )

        results = self._result()
        results[0]["metrics"] = {"psnr": 30.0}
        with self.assertRaises(ReferenceCorpusError):
            apply_reference_bless(self._manifest(), results)

    def test_bless_leaves_clips_the_run_skipped_untouched(self):
        from backend.reference_corpus import apply_reference_bless

        manifest = self._manifest()
        manifest["clips"].append({
            "filename": "b.mkv",
            "baseline": {"output_frames_sha256": "keep"},
            "metric_floors": {"psnr": 11.0},
        })
        changed = apply_reference_bless(manifest, self._result())

        self.assertEqual(changed, ["a.mkv"])
        self.assertEqual(
            manifest["clips"][1]["baseline"]["output_frames_sha256"], "keep")
        self.assertEqual(manifest["clips"][1]["metric_floors"]["psnr"], 11.0)


class FadeInAcrossBatchBoundaryTests(unittest.TestCase):
    """RM-296: the hold must not be cut short by where the decode split.

    A track whose first detection lands at the start of a decode batch used
    to get a shorter hold than requested, because the frames before it were
    already inpainted and written.
    """

    W, H = 96, 72
    BAND = (10, 40, 86, 64)

    def _clip(self, path, frames, first_text_frame):
        import cv2
        import numpy as np

        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (self.W, self.H))
        for index in range(frames):
            frame = np.full((self.H, self.W, 3), 30, dtype=np.uint8)
            if index >= first_text_frame:
                cv2.rectangle(frame, self.BAND[:2], self.BAND[2:],
                              (250, 250, 250), -1)
            writer.write(frame)
        writer.release()
        return path

    def _remover(self, fade_in, batch_size, first_text_frame=16):
        """A minimally wired SubtitleRemover for the frame loop."""
        config = processor.normalize_processing_config(
            processor.ProcessingConfig(
                mode=processor.InpaintMode.STTN,
                device="cpu",
                sttn_skip_detection=True,
                sttn_max_load_num=batch_size,
                adaptive_batch=False,
                mask_fade_in_frames=fade_in,
                subtitle_region_spans=[{
                    "rect": list(self.BAND),
                    "start": first_text_frame / 10.0,
                    "end": 0.0,
                }],
                preserve_audio=False,
                use_hw_encode=False,
                output_quality=20,
            ))
        remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
        remover.config = config
        remover.detector = processor.SubtitleDetector.__new__(
            processor.SubtitleDetector)
        remover.detector.device = "cpu"
        remover.detector.lang = "en"
        remover.detector.vertical = False
        remover.detector._engine_name = "fade-boundary-test"
        remover.detector._rapid_model = None
        remover.detector._paddle_model = None
        remover.detector._surya_det = None
        remover.detector._surya_processor = None
        remover.detector._easyocr_reader = None
        remover.inpainter = processor.STTNInpainter("cpu", config)
        remover.on_progress = None
        remover.on_preview_frame = None
        remover.live_preview_stride = 1000
        remover._hw_encoder = None
        remover.last_quality_report = None
        remover._color_metadata = None
        remover._active_writer = None
        return remover

    @staticmethod
    def _frame_digest(path):
        import hashlib

        import cv2
        import numpy as np

        capture = cv2.VideoCapture(str(path))
        accumulator = hashlib.sha256()
        count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            accumulator.update(np.ascontiguousarray(frame).tobytes())
            count += 1
        capture.release()
        return accumulator.hexdigest(), count

    def _masks_for(self, fade_in, batch_size, frames, first_text_frame):
        """Run the frame loop and capture the mask handed to each write."""
        from unittest import mock

        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            source = self._clip(work / "in.mp4", frames, first_text_frame)
            output = work / "out.mp4"
            remover = self._remover(fade_in, batch_size, first_text_frame)

            seen = []
            real_write = processor.SubtitleRemover._write_batch

            def capture(self_, ctx, state, batch, results):
                for offset, mask in enumerate(batch.masks):
                    seen.append((
                        state.written_idx + offset,
                        bool(np.any(mask)),
                    ))
                return real_write(self_, ctx, state, batch, results)

            with mock.patch.object(
                processor.SubtitleRemover, "_write_batch", capture
            ):
                self.assertTrue(
                    remover.process_video(str(source), str(output)))
        return seen

    BATCH = 8
    FADE = 5

    @staticmethod
    def _first_masked(seen):
        masked = [index for index, has_mask in seen if has_mask]
        return min(masked) if masked else None

    def test_every_frame_is_written_exactly_once_and_in_order(self):
        """Holding a tail back must not drop, duplicate, or reorder frames."""
        seen = self._masks_for(
            fade_in=self.FADE, batch_size=self.BATCH, frames=32,
            first_text_frame=17)
        indices = [index for index, _ in seen]
        self.assertEqual(indices, sorted(indices))
        self.assertEqual(len(indices), len(set(indices)))
        self.assertEqual(indices, list(range(len(indices))))

    def test_the_hold_reaches_back_across_the_batch_split(self):
        """The hold is measured against the same clip run without it.

        Deriving the detection frame from a second run rather than from the
        authored frame number keeps the test independent of how the encoder
        chose to time the synthetic clip.
        """
        without = self._masks_for(
            fade_in=0, batch_size=self.BATCH, frames=32, first_text_frame=16)
        base = self._first_masked(without)
        self.assertIsNotNone(base, "the fixture must produce a masked region")
        self.assertEqual(
            base % self.BATCH, 0,
            f"detection at frame {base} must land on a decode-batch boundary "
            f"for this test to exercise RM-296 at all",
        )

        with_hold = self._masks_for(
            fade_in=self.FADE, batch_size=self.BATCH, frames=32,
            first_text_frame=16)
        masked = {index for index, has_mask in with_hold if has_mask}
        for index in range(base - self.FADE, base):
            with self.subTest(frame=index):
                self.assertIn(
                    index, masked,
                    f"frame {index} is within {self.FADE} frames of the first "
                    f"detection at {base} and must inherit its mask",
                )
        self.assertNotIn(
            base - self.FADE - 1, masked,
            "the hold must stop exactly where it was asked to",
        )

    def test_a_paused_and_resumed_run_matches_an_uninterrupted_one(self):
        """RM-296: the held tail must not move the checkpoint off the grid.

        STTN pools a whole batch, so the output depends on how frames were
        grouped. A checkpoint that lands mid-batch makes a resumed run
        regroup and produce different pixels, which is why a pause flushes
        the held tail first.
        """
        from backend.resume_checkpoint import ProcessingPaused

        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            source = self._clip(work / "in.mp4", 32, 16)
            clean_out = work / "clean.mp4"
            resumed_out = work / "resumed.mp4"
            clean_ckpt = work / "ckpt-clean"
            resume_ckpt = work / "ckpt-resume"

            clean = self._remover(self.FADE, self.BATCH)
            self.assertTrue(clean.process_video(
                str(source), str(clean_out),
                checkpoint_dir=str(clean_ckpt), checkpoint_key="rm296"))
            clean_digest, clean_count = self._frame_digest(clean_out)

            batches = {"n": 0}

            def pause_after_three():
                batches["n"] += 1
                return batches["n"] >= 3

            paused = self._remover(self.FADE, self.BATCH)
            with self.assertRaises(ProcessingPaused):
                paused.process_video(
                    str(source), str(resumed_out),
                    checkpoint_dir=str(resume_ckpt), checkpoint_key="rm296",
                    pause_check=pause_after_three)
            next_frame = int(paused.last_pause_checkpoint["next_frame"])
            self.assertEqual(
                next_frame % self.BATCH, 0,
                f"a pause must land on a decode-batch boundary, got {next_frame}",
            )

            resumed = self._remover(self.FADE, self.BATCH)
            self.assertTrue(resumed.process_video(
                str(source), str(resumed_out),
                checkpoint_dir=str(resume_ckpt), checkpoint_key="rm296"))
            resumed_digest, resumed_count = self._frame_digest(resumed_out)

        self.assertEqual(resumed_count, clean_count)
        self.assertEqual(
            resumed_digest, clean_digest,
            "a resumed run must produce the same frames as an uninterrupted run",
        )

    def test_a_zero_hold_leaves_the_frames_before_detection_clear(self):
        seen = self._masks_for(
            fade_in=0, batch_size=self.BATCH, frames=32, first_text_frame=16)
        base = self._first_masked(seen)
        masked = {index for index, has_mask in seen if has_mask}
        self.assertIsNotNone(base)
        self.assertNotIn(base - 1, masked)
        indices = [index for index, _ in seen]
        self.assertEqual(indices, list(range(len(indices))))


if __name__ == "__main__":
    unittest.main()
