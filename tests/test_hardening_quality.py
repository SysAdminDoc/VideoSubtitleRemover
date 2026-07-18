import json
import os
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import VideoSubtitleRemover as gui
from backend import processor


def _has_display() -> bool:
    """Return True if a GUI display is available."""
    if sys.platform == "win32":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class QualitySheetTests(unittest.TestCase):
    """_write_quality_sheet must produce a single PNG with one row per
    sampled pair and a header carrying the mean metrics + Good/Review tag."""

    def test_sheet_written_with_expected_dimensions(self):
        import numpy as _np
        import cv2 as _cv2
        with tempfile.TemporaryDirectory() as tmp:
            out_path = str(Path(tmp) / "result.mp4")
            # Three synthetic pairs.
            pairs = []
            for i in range(3):
                a = _np.full((120, 160, 3), 100 + i, dtype=_np.uint8)
                b = _np.full((120, 160, 3), 110 + i, dtype=_np.uint8)
                pairs.append((i * 5, a, b, 35.0 + i, 0.96 - 0.01 * i))
            remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
            remover.config = processor.ProcessingConfig()
            sheet_path = remover._write_quality_sheet(
                out_path, pairs, mean_psnr=36.0, mean_ssim=0.95, tag="Good",
            )
            self.assertTrue(Path(sheet_path).exists())
            sheet = _cv2.imread(sheet_path)
            self.assertIsNotNone(sheet)
            # Width should match a single pair-row (two scaled frames + gap).
            # Height must include the header + N rows + N caption strips.
            self.assertGreater(sheet.shape[0], 200)
            self.assertGreater(sheet.shape[1], 200)
            # Filename convention.
            self.assertTrue(sheet_path.endswith(".qualitysheet.png"))


class QualityReportMaskedRoiTests(unittest.TestCase):
    """B-3: union-mask bbox accumulator + ROI-cropped PSNR/SSIM metric so
    a bad inpaint is no longer masked by 80-95% of unchanged pixels."""

    def _bare_remover(self):
        remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
        remover.config = processor.ProcessingConfig()
        remover._quality_mask_bbox = None
        return remover

    def test_accumulator_ignores_empty_mask(self):
        import numpy as _np
        r = self._bare_remover()
        r._accumulate_quality_bbox(_np.zeros((10, 10), dtype=_np.uint8))
        self.assertIsNone(r._quality_mask_bbox)

    def test_accumulator_tracks_single_box(self):
        import numpy as _np
        r = self._bare_remover()
        mask = _np.zeros((100, 200), dtype=_np.uint8)
        mask[20:40, 50:120] = 255
        r._accumulate_quality_bbox(mask)
        self.assertEqual(r._quality_mask_bbox, (50, 20, 120, 40))

    def test_accumulator_unions_across_frames(self):
        import numpy as _np
        r = self._bare_remover()
        m1 = _np.zeros((100, 200), dtype=_np.uint8)
        m1[20:40, 50:120] = 255
        m2 = _np.zeros((100, 200), dtype=_np.uint8)
        m2[60:90, 30:80] = 255
        r._accumulate_quality_bbox(m1)
        r._accumulate_quality_bbox(m2)
        self.assertEqual(r._quality_mask_bbox, (30, 20, 120, 90))

    def test_libvmaf_available_detects_filter(self):
        from unittest import mock
        from backend import quality as _q
        completed = SimpleNamespace(
            returncode=0,
            stdout=" .. libvmaf           VV->V      Calculate the VMAF between two video streams.\n",
            stderr="",
        )
        with mock.patch.object(_q.shutil, "which", return_value="ffmpeg"):
            with mock.patch.object(_q, "run_process", return_value=completed):
                self.assertTrue(_q.ffmpeg_libvmaf_available())

    def test_compute_vmaf_parses_mocked_log(self):
        from unittest import mock
        from backend import quality as _q

        seen = {}

        def fake_run(cmd, check, capture_output, timeout, cwd=None):
            filt = cmd[cmd.index("-lavfi") + 1]
            seen["command"] = list(cmd)
            seen["cwd"] = cwd
            log_name = _q._unescape_filter_value(
                filt.split("log_path=", 1)[1])
            log_path = Path(cwd) / log_name
            log_path.write_text(
                json.dumps({"pooled_metrics": {"vmaf": {"mean": 91.25}}}),
                encoding="utf-8",
            )
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

        with tempfile.TemporaryDirectory() as tmpdir:
            ref = Path(tmpdir) / "ref.mp4"
            dist = Path(tmpdir) / "dist.mp4"
            ref.write_bytes(b"ref")
            dist.write_bytes(b"dist")
            with mock.patch.object(_q, "ffmpeg_libvmaf_available", return_value=True):
                with mock.patch.object(_q, "run_process", side_effect=fake_run):
                    self.assertEqual(
                        _q.compute_vmaf(
                            str(ref), str(dist), start_seconds=2.5,
                            duration_seconds=1.0,
                            roi=(2, 3, 12, 23),
                        ),
                        91.25,
                    )
        command = seen["command"]
        self.assertIsNotNone(seen["cwd"])
        inputs = [
            index for index, value in enumerate(command) if value == "-i"
        ]
        self.assertEqual(len(inputs), 2)
        for index in inputs:
            self.assertEqual(command[index - 4:index], [
                "-ss", "2.500", "-t", "1.000",
            ])
        filt = command[command.index("-lavfi") + 1]
        self.assertIn("[0:v]settb=AVTB,setpts=PTS-STARTPTS", filt)
        self.assertIn("[1:v]settb=AVTB,setpts=PTS-STARTPTS", filt)
        self.assertIn("log_path=vmaf.json", filt)

    def test_compute_vmaf_real_windows_motion_windows(self):
        from backend import quality as _q

        if not _q.ffmpeg_libvmaf_available():
            self.skipTest("ffmpeg libvmaf filter unavailable")
        with tempfile.TemporaryDirectory(prefix="vsr vmaf windows ") as tmpdir:
            root = Path(tmpdir)
            reference = root / "reference motion.mkv"
            distorted = root / "distorted motion.mkv"
            subprocess.run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i",
                "testsrc2=size=96x64:rate=12:duration=4",
                "-c:v", "ffv1", str(reference),
            ], check=True, timeout=30)
            subprocess.run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(reference),
                "-vf",
                "drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:"
                "enable='gte(t,2)'",
                "-c:v", "ffv1", str(distorted),
            ], check=True, timeout=30)

            early = _q.compute_vmaf(
                str(reference), str(distorted),
                start_seconds=0.25, duration_seconds=1.0,
            )
            late = _q.compute_vmaf(
                str(reference), str(distorted),
                start_seconds=2.5, duration_seconds=1.0,
            )

        self.assertIsNotNone(early)
        self.assertIsNotNone(late)
        self.assertGreater(early, 95.0)
        self.assertGreater(early - late, 20.0)

    def test_quality_report_includes_vmaf_when_available(self):
        from unittest import mock
        import numpy as _np
        r = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
        r.config = processor.ProcessingConfig(quality_report=True)
        r._quality_mask_bbox = (10, 10, 50, 50)
        frame_in = _np.full((80, 96, 3), 128, dtype=_np.uint8)
        frame_out = frame_in.copy()

        class FakeCapture:
            def __init__(self, frame):
                self.frame = frame
                self._pos = 0
            def isOpened(self):
                return True
            def get(self, prop):
                if prop == processor.cv2.CAP_PROP_FRAME_COUNT:
                    return 4
                if prop == processor.cv2.CAP_PROP_POS_FRAMES:
                    return float(self._pos)
                return 0
            def set(self, prop, value):
                if prop == processor.cv2.CAP_PROP_POS_FRAMES:
                    self._pos = int(value)
                return True
            def grab(self):
                self._pos += 1
                return True
            def read(self):
                return True, self.frame.copy()
            def release(self):
                return None

        with mock.patch(
            "backend._quality_mixin._open_capture",
            side_effect=[FakeCapture(frame_in), FakeCapture(frame_out)],
        ):
            with mock.patch(
                "backend._quality_mixin.compute_vmaf",
                side_effect=[95.0, 93.0],
            ):
                metrics = r._compute_quality_report(
                    "input.mp4", "output.mp4", 0, 4, 24.0, n_samples=2
                )

        self.assertEqual(metrics["vmaf"], 95.0)
        self.assertEqual(metrics["roi_vmaf"], 93.0)
        self.assertEqual(metrics["temporal_flicker_score"], 0.0)
        self.assertEqual(metrics["residual_text_score"], 0.0)
        self.assertEqual(metrics["quality_gate"]["status"], "passed")


class QualityGateTests(unittest.TestCase):
    """#108: quality metrics produce graduated ladder steps with
    actionable remediations, not just binary pass/review."""

    def test_passes_when_roi_metrics_clear_thresholds(self):
        from backend.quality_gate import evaluate_quality_gate
        gate = evaluate_quality_gate({
            "samples": 4,
            "tag": "Good",
            "ssim": 0.98,
            "roi_ssim": 0.97,
            "vmaf": 95.0,
            "roi_vmaf": 92.0,
        })
        self.assertEqual(gate["status"], "passed")
        self.assertEqual(gate["ladderStep"], "none")
        self.assertEqual(gate["reasons"], [])
        self.assertEqual(gate["remediation"], "")

    def test_temporal_flicker_score_uses_adjacent_samples_only(self):
        import numpy as _np
        from backend.quality import temporal_flicker_score
        black = _np.zeros((16, 16, 3), dtype=_np.uint8)
        white = _np.full((16, 16, 3), 255, dtype=_np.uint8)
        self.assertIsNone(temporal_flicker_score([(0, black), (4, white)]))
        self.assertEqual(temporal_flicker_score([(0, black), (1, white)]), 1.0)

    def test_residual_text_score_flags_text_like_roi(self):
        import cv2 as _cv2
        import numpy as _np
        from backend.quality import residual_text_score
        flat = _np.full((80, 220, 3), 128, dtype=_np.uint8)
        text = flat.copy()
        _cv2.putText(
            text,
            "SUBTITLE",
            (12, 50),
            _cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            _cv2.LINE_AA,
        )
        self.assertEqual(residual_text_score(flat), 0.0)
        self.assertGreater(residual_text_score(text), 0.025)

    def test_ladder_temporal_smooth_on_flicker(self):
        from backend.quality_gate import evaluate_quality_gate
        gate = evaluate_quality_gate({
            "samples": 4,
            "tag": "Good",
            "ssim": 0.99,
            "roi_ssim": 0.98,
            "temporal_flicker_score": 0.2,
        })
        self.assertEqual(gate["status"], "review")
        self.assertEqual(gate["ladderStep"], "temporal-smooth")
        self.assertIn("temporal flicker", gate["reason"])
        self.assertIn("temporal smooth", gate["remediation"].lower())
        self.assertEqual(len(gate["reasons"]), 1)
        self.assertEqual(gate["reasons"][0]["metric"], "temporal_flicker_score")

    def test_ladder_increase_dilation_on_residual_text(self):
        from backend.quality_gate import evaluate_quality_gate
        gate = evaluate_quality_gate({
            "samples": 4,
            "tag": "Good",
            "ssim": 0.99,
            "roi_ssim": 0.98,
            "residual_text_score": 0.1,
        })
        self.assertEqual(gate["status"], "review")
        self.assertEqual(gate["ladderStep"], "increase-dilation")
        self.assertIn("residual text score", gate["reason"])
        self.assertIn("dilation", gate["remediation"].lower())

    def test_ladder_alternate_inpainter_on_low_ssim(self):
        from backend.quality_gate import evaluate_quality_gate
        gate = evaluate_quality_gate({
            "samples": 4,
            "tag": "Good",
            "ssim": 0.99,
            "roi_ssim": 0.90,
        })
        self.assertEqual(gate["status"], "review")
        self.assertEqual(gate["ladderStep"], "alternate-inpainter")
        self.assertIn("ROI SSIM", gate["reason"])
        self.assertIn("inpaint mode", gate["remediation"].lower())

    def test_ladder_escalates_to_manual_on_multiple_violation_types(self):
        from backend.quality_gate import evaluate_quality_gate
        gate = evaluate_quality_gate({
            "samples": 4,
            "tag": "Good",
            "ssim": 0.99,
            "roi_ssim": 0.90,
            "residual_text_score": 0.1,
        })
        self.assertEqual(gate["status"], "review")
        self.assertEqual(gate["ladderStep"], "manual-review")
        self.assertGreaterEqual(len(gate["reasons"]), 2)

    def test_review_when_roi_metric_fails_and_sheet_is_preview(self):
        from backend.quality_gate import evaluate_quality_gate
        gate = evaluate_quality_gate({
            "samples": 4,
            "tag": "Review",
            "ssim": 0.99,
            "roi_ssim": 0.90,
            "sheet": "clip.qualitysheet.png",
        })
        self.assertEqual(gate["status"], "review")
        self.assertEqual(gate["ladderStep"], "manual-review")
        self.assertIn("ROI SSIM", gate["reason"])
        self.assertEqual(gate["previewFramePaths"], ["clip.qualitysheet.png"])

    def test_unknown_without_metrics(self):
        from backend.quality_gate import evaluate_quality_gate
        gate = evaluate_quality_gate(None)
        self.assertEqual(gate["status"], "unknown")
        self.assertEqual(gate["ladderStep"], "not-run")
        self.assertIn("remediation", gate)
        self.assertIsInstance(gate["reasons"], list)

    def test_degraded_metrics_reported_when_optional_absent(self):
        from backend.quality_gate import evaluate_quality_gate
        gate = evaluate_quality_gate({
            "samples": 4,
            "tag": "Good",
            "ssim": 0.98,
            "roi_ssim": 0.97,
        })
        self.assertEqual(gate["status"], "passed")
        self.assertIn("vmaf", gate["degradedMetrics"])
        self.assertIn("roi_vmaf", gate["degradedMetrics"])
        self.assertIn("lpips", gate["degradedMetrics"])
        self.assertIn("dists", gate["degradedMetrics"])

    def test_degraded_metrics_empty_when_all_present(self):
        from backend.quality_gate import evaluate_quality_gate
        gate = evaluate_quality_gate({
            "samples": 4,
            "tag": "Good",
            "ssim": 0.98,
            "roi_ssim": 0.97,
            "vmaf": 95.0,
            "roi_vmaf": 92.0,
            "lpips": 0.01,
            "dists": 0.02,
            "temporal_consistency": 0.99,
        })
        self.assertEqual(gate["degradedMetrics"], [])

    def test_structured_reasons_carry_metric_and_value(self):
        from backend.quality_gate import evaluate_quality_gate
        gate = evaluate_quality_gate({
            "samples": 4,
            "tag": "Good",
            "ssim": 0.99,
            "roi_ssim": 0.98,
            "residual_text_score": 0.1,
        })
        self.assertEqual(len(gate["reasons"]), 1)
        reason = gate["reasons"][0]
        self.assertEqual(reason["metric"], "residual_text_score")
        self.assertAlmostEqual(reason["value"], 0.1)
        self.assertAlmostEqual(reason["threshold"], 0.025)
        self.assertEqual(reason["ladder"], "increase-dilation")

    def test_retry_config_patches_follow_ladder_steps(self):
        from backend.quality_gate import retry_config_patch_for_gate

        self.assertEqual(
            retry_config_patch_for_gate(
                {"status": "review", "ladderStep": "increase-dilation"},
                {"mask_dilate_px": 8},
            )["mask_dilate_px"],
            12,
        )
        self.assertEqual(
            retry_config_patch_for_gate(
                {"status": "review", "ladderStep": "temporal-smooth"},
                {"temporal_smooth_radius": 0},
            )["temporal_smooth_radius"],
            2,
        )
        self.assertEqual(
            retry_config_patch_for_gate(
                {"status": "review", "ladderStep": "alternate-inpainter"},
                {"mode": "sttn"},
            )["mode"],
            "lama",
        )

    def test_manual_review_retry_patch_combines_structured_reasons(self):
        from backend.quality_gate import retry_config_patch_for_gate

        patch = retry_config_patch_for_gate(
            {
                "status": "review",
                "ladderStep": "manual-review",
                "reasons": [
                    {"metric": "residual_text_score", "ladder": "increase-dilation"},
                    {"metric": "temporal_flicker_score", "ladder": "temporal-smooth"},
                    {"metric": "ssim", "ladder": "alternate-inpainter"},
                ],
            },
            {
                "mode": "sttn",
                "mask_dilate_px": 8,
                "temporal_smooth_radius": 0,
            },
        )

        self.assertEqual(patch["mask_dilate_px"], 12)
        self.assertEqual(patch["temporal_smooth_radius"], 2)
        self.assertEqual(patch["mode"], "lama")
        self.assertTrue(patch["quality_report"])


class ExtendedMetricsTests(unittest.TestCase):
    """RM-102: temporal quality metric expansion."""

    def test_compute_extended_metrics_returns_empty_without_pyiqa(self):
        import numpy as _np
        from unittest import mock
        from backend.quality import compute_extended_metrics
        with mock.patch.dict("sys.modules", {"pyiqa": None}):
            result = compute_extended_metrics([
                (_np.zeros((32, 32, 3), _np.uint8),
                 _np.ones((32, 32, 3), _np.uint8))])
            self.assertEqual(result, {})

    def test_temporal_consistency_score_perfect(self):
        import numpy as _np
        from backend.quality import temporal_consistency_score
        frame = _np.full((32, 32, 3), 128, dtype=_np.uint8)
        score = temporal_consistency_score([frame, frame, frame])
        self.assertIsNotNone(score)
        self.assertGreater(score, 0.99)

    def test_temporal_consistency_score_insufficient_frames(self):
        import numpy as _np
        from backend.quality import temporal_consistency_score
        self.assertIsNone(temporal_consistency_score([]))
        self.assertIsNone(temporal_consistency_score(
            [_np.zeros((8, 8, 3), _np.uint8)]))

    def test_quality_report_schema_includes_extended_fields(self):
        report = {
            'psnr': 35.0, 'ssim': 0.97,
            'roi_psnr': 30.0, 'roi_ssim': 0.92,
            'vmaf': None, 'roi_vmaf': None,
            'roi_bbox': [10, 20, 100, 50],
            'temporal_flicker_score': 0.01,
            'temporal_consistency': 0.98,
            'residual_text_score': 0.02,
            'lpips': None, 'dists': None,
            'samples': 10, 'tag': 'Good', 'sheet': None,
        }
        for key in ('temporal_consistency', 'lpips', 'dists'):
            self.assertIn(key, report)


class OtsuFallbackDetectionTests(unittest.TestCase):
    """EI-1: the OpenCV fallback detector must catch mid-tone subtitle
    luminance the fixed 200 / 55 thresholds missed."""

    def test_fallback_finds_grey_text_on_grey(self):
        import numpy as _np
        # Mid-tone grey frame with slightly darker grey text-shaped strip
        # in the bottom band -- both within the [55, 200] dead zone of
        # the old fixed thresholds.
        frame = _np.full((180, 320, 3), 130, dtype=_np.uint8)
        frame[150:170, 40:280] = 100  # darker grey "subtitle"
        detector = processor.SubtitleDetector.__new__(processor.SubtitleDetector)
        detector._engine_name = "OpenCV fallback"
        detector._rapid_model = None
        detector._paddle_model = None
        detector._surya_det = None
        detector._easyocr_reader = None
        boxes = detector._fallback_detection(frame)
        self.assertTrue(boxes, "Otsu fallback must detect the mid-tone band")


class TemporalSmoothTests(unittest.TestCase):
    """Post-inpaint temporal consistency filter for LaMa path."""

    def test_temporal_smooth_reduces_variance(self):
        import numpy as _np
        from backend.inpainters._common import _temporal_smooth_inpainted
        mask = _np.zeros((100, 100), dtype=_np.uint8)
        mask[40:60, 40:60] = 255
        masks = [mask] * 5
        frames = []
        for i in range(5):
            f = _np.full((100, 100, 3), 128, dtype=_np.uint8)
            f[40:60, 40:60] = _np.random.randint(50, 200, (20, 20, 3), dtype=_np.uint8)
            frames.append(f)
        smoothed = _temporal_smooth_inpainted(frames, masks, radius=2)
        self.assertEqual(len(smoothed), 5)
        orig_var = _np.var([f[50, 50, 0] for f in frames])
        smooth_var = _np.var([f[50, 50, 0] for f in smoothed])
        self.assertLessEqual(smooth_var, orig_var)

    def test_temporal_smooth_skips_unmasked(self):
        import numpy as _np
        from backend.inpainters._common import _temporal_smooth_inpainted
        mask = _np.zeros((50, 50), dtype=_np.uint8)
        frame = _np.full((50, 50, 3), 100, dtype=_np.uint8)
        result = _temporal_smooth_inpainted([frame, frame], [mask, mask], radius=2)
        _np.testing.assert_array_equal(result[0], frame)

    def test_config_defaults_off(self):
        cfg = gui.ProcessingConfig()
        self.assertEqual(cfg.temporal_smooth_radius, 0)


class ConfidenceWeightedDilationTests(unittest.TestCase):
    """Confidence-weighted mask dilation scales padding by OCR confidence."""

    def test_config_defaults_off(self):
        cfg = gui.ProcessingConfig()
        self.assertFalse(cfg.confidence_weighted_dilation)
        self.assertAlmostEqual(cfg.confidence_dilation_scale, 1.5)

    def test_config_round_trip(self):
        cfg = gui.ProcessingConfig()
        cfg.confidence_weighted_dilation = True
        cfg.confidence_dilation_scale = 2.0
        d = cfg.to_dict()
        self.assertTrue(d["confidence_weighted_dilation"])
        cfg2 = gui.ProcessingConfig.from_dict(d)
        self.assertTrue(cfg2.confidence_weighted_dilation)
        self.assertAlmostEqual(cfg2.confidence_dilation_scale, 2.0)

    def test_high_confidence_gets_less_dilation(self):
        """With confidence weighting on, a high-confidence box should produce
        a smaller dilated area than a low-confidence box."""
        cfg_hi = gui.ProcessingConfig()
        cfg_hi.confidence_weighted_dilation = True
        cfg_hi.mask_dilate_px = 8
        cfg_hi.confidence_dilation_scale = 1.5
        cfg_lo = gui.ProcessingConfig()
        cfg_lo.confidence_weighted_dilation = True
        cfg_lo.mask_dilate_px = 8
        cfg_lo.confidence_dilation_scale = 1.5
        shape = (200, 400, 3)
        boxes = [(100, 80, 300, 120)]
        remover_hi = SimpleNamespace(config=cfg_hi)
        remover_lo = SimpleNamespace(config=cfg_lo)
        mask_hi = processor.SubtitleRemover._create_mask(
            remover_hi, shape, boxes, confidences=[0.95])
        mask_lo = processor.SubtitleRemover._create_mask(
            remover_lo, shape, boxes, confidences=[0.3])
        self.assertGreater(mask_lo.sum(), mask_hi.sum())


class TemporalMaskStabilizationTests(unittest.TestCase):
    """P1: scene-cut-safe rolling mask union."""

    def _mask(self, on):
        m = np.zeros((16, 32), np.uint8)
        if on:
            m[4:12, 8:24] = 255
        return m

    def test_single_frame_miss_is_recovered_within_scene(self):
        from backend.inpainters import stabilize_masks_rolling_union
        masks = [self._mask(True), self._mask(False), self._mask(True)]
        out = stabilize_masks_rolling_union(masks, scene_starts=[0], window=3)
        # frame 1 missed detection but neighbours saw text -> recovered
        self.assertGreater(int(out[1].max()), 0)

    def test_union_never_crosses_scene_cut(self):
        from backend.inpainters import stabilize_masks_rolling_union
        # scene A has text; scene B (starts at idx 2) is clean
        masks = [self._mask(True), self._mask(True),
                 self._mask(False), self._mask(False)]
        out = stabilize_masks_rolling_union(masks, scene_starts=[0, 2], window=4)
        self.assertEqual(int(out[2].max()), 0)
        self.assertEqual(int(out[3].max()), 0)

    def test_window_one_is_noop(self):
        from backend.inpainters import stabilize_masks_rolling_union
        masks = [self._mask(True), self._mask(False)]
        out = stabilize_masks_rolling_union(masks, [0], window=1)
        self.assertEqual(int(out[1].max()), 0)

    def test_config_defaults_off_and_round_trip(self):
        from backend.config import ProcessingConfig, normalize_processing_config
        cfg = ProcessingConfig()
        self.assertFalse(cfg.temporal_mask_union)
        cfg.temporal_mask_union = True
        cfg.temporal_mask_window = 99
        normalize_processing_config(cfg)
        self.assertTrue(cfg.temporal_mask_union)
        self.assertLessEqual(cfg.temporal_mask_window, 15)


class SeamQualityTests(unittest.TestCase):
    """P1: mask-boundary seam (discontinuity) quality check."""

    def _smooth_bg(self):
        import cv2
        yy, xx = np.mgrid[0:80, 0:80]
        bg = np.stack([(xx * 2) % 256, (yy * 2) % 256, (xx + yy) % 256], -1)
        return cv2.GaussianBlur(bg.astype(np.uint8), (9, 9), 0)

    def _mask(self):
        m = np.zeros((80, 80), np.uint8)
        m[24:56, 24:56] = 255
        return m

    def test_seamless_fill_scores_low(self):
        from backend.quality import mask_boundary_seam_score
        bg = self._smooth_bg()
        score = mask_boundary_seam_score(bg, bg.copy(), self._mask())
        self.assertIsNotNone(score)
        self.assertLess(score, 0.1)

    def test_hard_box_fill_scores_high(self):
        from backend.quality import mask_boundary_seam_score
        bg = self._smooth_bg()
        bad = bg.copy()
        bad[24:56, 24:56] = 128  # flat box with a hard boundary step
        score = mask_boundary_seam_score(bg, bad, self._mask())
        self.assertIsNotNone(score)
        self.assertGreater(score, 0.35)

    def test_seam_metric_handles_empty_mask(self):
        from backend.quality import mask_boundary_seam_score
        bg = self._smooth_bg()
        self.assertIsNone(
            mask_boundary_seam_score(bg, bg.copy(), np.zeros((80, 80), np.uint8))
        )

    def test_quality_gate_flags_high_seam(self):
        from backend.quality_gate import evaluate_quality_gate, SEAM_SCORE_CEILING
        gate = evaluate_quality_gate({
            "samples": 5, "ssim": 0.99,
            "seam_score": SEAM_SCORE_CEILING + 0.2,
        })
        self.assertEqual(gate["status"], "review")
        metrics = {r["metric"] for r in gate["reasons"]}
        self.assertIn("seam_score", metrics)

    def test_quality_gate_passes_low_seam(self):
        from backend.quality_gate import evaluate_quality_gate
        gate = evaluate_quality_gate({
            "samples": 5, "ssim": 0.99, "seam_score": 0.05,
        })
        self.assertEqual(gate["status"], "passed")

    def test_accumulator_records_bounded_samples(self):
        from backend.processor import SubtitleRemover
        from backend.config import ProcessingConfig
        remover = SubtitleRemover.__new__(SubtitleRemover)
        remover.config = ProcessingConfig()
        remover._seam_scores = []
        bg = self._smooth_bg()
        mask = self._mask()
        frames = [bg.copy() for _ in range(6)]
        results = [bg.copy() for _ in range(6)]
        masks = [mask for _ in range(6)]
        for _ in range(20):
            remover._accumulate_seam_scores(frames, results, masks)
        self.assertLessEqual(len(remover._seam_scores), 32)
        self.assertTrue(remover._seam_scores)

    def test_accumulator_logs_seam_failure_once(self):
        from backend import processor
        from backend.config import ProcessingConfig
        remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
        remover.config = ProcessingConfig()
        remover._seam_scores = []
        frame = self._smooth_bg()
        mask = self._mask()

        import backend._quality_mixin as _qm
        with unittest.mock.patch.object(
            _qm, "mask_boundary_seam_score",
            side_effect=RuntimeError("bad mask"),
        ):
            with self.assertLogs("backend._quality_mixin", level="WARNING") as logs:
                remover._accumulate_seam_scores([frame], [frame], [mask])
                remover._accumulate_seam_scores([frame], [frame], [mask])

        warnings = [line for line in logs.output if "Seam-score" in line]
        self.assertEqual(len(warnings), 1)
        self.assertIn("omit boundary-seam evidence", warnings[0])



if __name__ == "__main__":
    unittest.main()
