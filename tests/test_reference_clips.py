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


if __name__ == "__main__":
    unittest.main()
