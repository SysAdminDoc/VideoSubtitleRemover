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

from backend.mask_free_benchmark import (
    MASK_FREE_BENCHMARK_SCHEMA,
    MaskFreeBenchmarkError,
    iter_mask_free_manifest_entries,
    masks_from_subtitle_regions,
    run_mask_free_subtitle_benchmark,
)


def _subtitle_frames(count: int = 5):
    originals = []
    references = []
    height, width = 72, 128
    for idx in range(count):
        ref = np.full((height, width, 3), (70, 90 + idx, 120), dtype=np.uint8)
        frame = ref.copy()
        frame[54:66, 22:106] = 245
        frame[57:63, 34:94] = 10
        originals.append(frame)
        references.append(ref)
    return originals, references


class MaskFreeBenchmarkTests(unittest.TestCase):
    def test_masks_from_regions_supports_static_regions(self):
        masks = masks_from_subtitle_regions(
            (72, 128, 3),
            3,
            [[22, 54, 106, 66]],
        )
        self.assertEqual(len(masks), 3)
        self.assertEqual(masks[0].shape, (72, 128))
        self.assertGreater(np.count_nonzero(masks[0]), 0)

    def test_benchmark_records_runtime_artifacts_and_subtitle_quality(self):
        originals, references = _subtitle_frames()
        noisy = [frame.copy() for frame in references]
        for frame in noisy:
            frame[:8, :8] = 255

        result = run_mask_free_subtitle_benchmark(
            originals,
            {
                "clean": references,
                "noop": originals,
                "artifact": noisy,
            },
            subtitle_regions=[[22, 54, 106, 66]],
            reference_frames=references,
            runtime_seconds={
                "clean": 2.5,
                "noop": 0.1,
                "artifact": 4.0,
            },
        )

        self.assertEqual(result["schema"], MASK_FREE_BENCHMARK_SCHEMA)
        self.assertEqual(result["frameCount"], len(originals))
        methods = result["metrics"]["methods"]
        self.assertEqual(methods["clean"]["runtimeSeconds"], 2.5)
        self.assertAlmostEqual(
            methods["clean"]["secondsPerFrame"], 2.5 / len(originals))
        self.assertGreater(
            methods["clean"]["subtitleRemovalQuality"],
            methods["noop"]["subtitleRemovalQuality"],
        )
        self.assertLess(
            methods["clean"]["artifactScore"],
            methods["artifact"]["artifactScore"],
        )
        self.assertEqual(
            result["metrics"]["winners"]["highestSubtitleRemovalQuality"],
            "clean",
        )

    def test_benchmark_rejects_missing_regions(self):
        originals, _references = _subtitle_frames()
        with self.assertRaises(MaskFreeBenchmarkError):
            run_mask_free_subtitle_benchmark(
                originals,
                {"noop": originals},
            )

    def test_manifest_entries_require_evaluation_regions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            clip = root / "subtitle.mp4"
            clip.write_bytes(b"licensed mask-free subtitle clip placeholder")
            sha = hashlib.sha256(clip.read_bytes()).hexdigest()
            manifest = root / "manifest.json"
            payload = {
                "schema_version": 1,
                "clips": [{
                    "filename": "subtitle.mp4",
                    "license": "CC-BY-4.0",
                    "contributor": "Unit Test",
                    "sha256": sha,
                    "failure_category": "mask_free_subtitle",
                    "config": {"profile": "mask_free_subtitle_erasure"},
                    "metric_floors": {"subtitleRemovalQuality": 0.9},
                    "evaluation": {"subtitle_regions": [[22, 54, 106, 66]]},
                }],
            }
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            entries = iter_mask_free_manifest_entries(manifest, root)
            self.assertEqual(entries[0]["path"], str(clip))

            payload["clips"][0]["evaluation"] = {}
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(MaskFreeBenchmarkError):
                iter_mask_free_manifest_entries(manifest, root)

    def test_mask_free_research_specs_are_not_registered_modes(self):
        from backend import inpainter_registry
        from backend.inpainters_diffusion import (
            _OPT_INS,
            mask_free_research_adapter_specs,
        )

        specs = mask_free_research_adapter_specs()
        names = {spec["adapter_manifest"] for spec in specs}
        self.assertEqual(names, {"clear-maskfree", "sedit-maskfree"})
        opt_in_envs = {env_name for env_name, _mode, _klass in _OPT_INS}
        self.assertNotIn("VSR_CLEAR_WEIGHTS", opt_in_envs)
        self.assertNotIn("VSR_SEDIT_WEIGHTS", opt_in_envs)
        self.assertFalse(inpainter_registry.is_registered("clear"))
        self.assertFalse(inpainter_registry.is_registered("sedit"))

    def test_mask_free_adapter_manifest_entries_fail_closed(self):
        from backend import adapter_manifest as manifest

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "clear.pt"
            p.write_bytes(b"unverified weights")
            result = manifest.verify_adapter_path(
                "clear-maskfree",
                str(p),
                env={},
            )
            override = manifest.verify_adapter_path(
                "clear-maskfree",
                str(p),
                env={manifest.UNSAFE_OVERRIDE_ENV: "1"},
            )

        self.assertFalse(result.allowed)
        self.assertEqual(result.hash_status, "unknown")
        self.assertTrue(result.strict_unknown)
        self.assertTrue(override.allowed)
        self.assertEqual(override.hash_status, "unsafe_override")


if __name__ == "__main__":
    unittest.main()
