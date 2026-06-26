from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.inpainters.external import deterministic_static_logo_cleanup
from backend.presets import (
    BUILTIN_PRESETS,
    benchmark_preset_fields,
    list_preset_names,
)
from backend.static_logo_benchmark import (
    STATIC_LOGO_BENCHMARK_SCHEMA,
    StaticLogoBenchmarkError,
    iter_static_logo_manifest_entries,
    run_static_logo_cleanup_benchmark,
)


def _synthetic_logo_frames(count: int = 6):
    frames = []
    references = []
    masks = []
    height, width = 72, 128
    for idx in range(count):
        ref = np.full((height, width, 3), (60 + idx * 2, 90, 120), dtype=np.uint8)
        frame = ref.copy()
        frame[18:34, 78:112] = (230 - idx * 9, 235, 245)
        frame[20:32, 84:106] = (15 + idx * 12, 15, 15)
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[16:36, 76:114] = 255
        frames.append(frame)
        references.append(ref)
        masks.append(mask)
    return frames, masks, references


class StaticLogoBenchmarkTests(unittest.TestCase):
    def test_deterministic_cleanup_preserves_shape_and_count(self):
        frames, masks, _references = _synthetic_logo_frames()
        cleaned = deterministic_static_logo_cleanup(frames, masks)
        self.assertEqual(len(cleaned), len(frames))
        for original, output in zip(frames, cleaned):
            self.assertEqual(output.shape, original.shape)
            self.assertEqual(output.dtype, original.dtype)
        self.assertFalse(np.array_equal(cleaned[0], frames[0]))

    def test_static_logo_benchmark_reports_methods_and_metrics(self):
        frames, masks, references = _synthetic_logo_frames()
        result = run_static_logo_cleanup_benchmark(
            frames,
            masks,
            reference_frames=references,
        )
        self.assertEqual(result["schema"], STATIC_LOGO_BENCHMARK_SCHEMA)
        self.assertEqual(
            result["methods"], ["current_cv2", "deterministic_static"])
        self.assertGreater(result["metrics"]["maskCoverage"], 0.0)
        methods = result["metrics"]["methods"]
        for name in ("current_cv2", "deterministic_static"):
            self.assertIn(name, methods)
            self.assertEqual(methods[name]["roiFrameCount"], len(frames))
            self.assertIn("residualTextScoreMean", methods[name])
            self.assertIn("temporalFlickerScore", methods[name])
            self.assertIn("temporalConsistency", methods[name])
            self.assertIn("ssimVsReferenceMean", methods[name])
        self.assertIn("lowestFlicker", result["metrics"]["winners"])

    def test_benchmark_rejects_mismatched_inputs(self):
        frames, masks, _references = _synthetic_logo_frames()
        with self.assertRaises(StaticLogoBenchmarkError):
            run_static_logo_cleanup_benchmark(frames, masks[:-1])

    def test_static_logo_manifest_entries_require_license_and_hash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            clip = root / "logo.mp4"
            clip.write_bytes(b"licensed static logo clip placeholder")
            sha = hashlib.sha256(clip.read_bytes()).hexdigest()
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "schema_version": 1,
                "clips": [{
                    "filename": "logo.mp4",
                    "license": "CC0-1.0",
                    "contributor": "Unit Test",
                    "sha256": sha,
                    "failure_category": "static_logo",
                    "config": {"profile": "static_logo_cleanup"},
                    "metric_floors": {"ssimVsReferenceMean": 0.9},
                }],
            }), encoding="utf-8")
            entries = iter_static_logo_manifest_entries(manifest, root)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["path"], str(clip))

            bad = json.loads(manifest.read_text(encoding="utf-8"))
            bad["clips"][0]["license"] = "unknown-reuse"
            manifest.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(StaticLogoBenchmarkError):
                iter_static_logo_manifest_entries(manifest, root)

    def test_static_logo_benchmark_preset_is_not_user_facing(self):
        fields = benchmark_preset_fields("static_logo_cleanup")
        self.assertIsNotNone(fields)
        self.assertEqual(fields["mode"], "LAMA")
        self.assertTrue(fields["quality_report"])
        self.assertNotIn("static_logo_cleanup", BUILTIN_PRESETS)
        self.assertNotIn("static_logo_cleanup", list_preset_names({}))


if __name__ == "__main__":
    unittest.main()
