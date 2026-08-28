"""RM-321: a static manual region must not be reported as a clean success.

STTN and ProPainter recover masked pixels from other frames. A manual region
with automatic detection switched off is identical in every frame, so every
masked pixel has zero temporal coverage and the whole band falls back to
`cv2.inpaint`. The run succeeded, the engine the user chose did not run, and
the interface reported the state in success green.
"""

from __future__ import annotations

import hashlib
import unittest

import numpy as np

from backend.config import (
    InpaintMode,
    ProcessingConfig,
    static_region_degrades_to_cv2,
)
from backend.inpainters.sttn import (
    TEMPORAL_EXPOSURE_DEGRADED_FRACTION,
    STTNInpainter,
)


def _frames(count: int = 6) -> list:
    rng = np.random.default_rng(11)
    base = rng.integers(0, 255, (32, 48, 3), dtype=np.uint8)
    out = []
    for index in range(count):
        frame = np.roll(base, index * 3, axis=1).copy()
        frame[10:18, 12:30] = 240
        out.append(frame)
    return out


def _static_mask() -> np.ndarray:
    mask = np.zeros((32, 48), np.uint8)
    mask[10:18, 12:30] = 255
    return mask


def _moving_mask(index: int) -> np.ndarray:
    mask = np.zeros((32, 48), np.uint8)
    mask[10:18, 4 + index * 3:16 + index * 3] = 255
    return mask


def _digest(frames) -> str:
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(np.ascontiguousarray(frame).tobytes())
    return digest.hexdigest()


class StaticRegionPredicateTests(unittest.TestCase):
    def _config(self, **overrides) -> ProcessingConfig:
        config = ProcessingConfig()
        config.mode = InpaintMode.STTN
        config.sttn_skip_detection = True
        config.subtitle_area = (12, 10, 30, 18)
        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    def test_a_static_region_with_a_temporal_engine_is_flagged(self):
        self.assertTrue(static_region_degrades_to_cv2(self._config()))
        self.assertTrue(static_region_degrades_to_cv2(
            self._config(mode=InpaintMode.PROPAINTER)))

    def test_lama_is_not_flagged_because_it_needs_no_exposure(self):
        self.assertFalse(static_region_degrades_to_cv2(
            self._config(mode=InpaintMode.LAMA)))

    def test_automatic_detection_is_not_flagged(self):
        self.assertFalse(static_region_degrades_to_cv2(
            self._config(sttn_skip_detection=False)))

    def test_a_timed_or_keyframed_region_is_not_flagged(self):
        spans = [{"rect": (12, 10, 30, 18), "start": 0.0, "end": 1.0}]
        self.assertFalse(static_region_degrades_to_cv2(
            self._config(subtitle_region_spans=spans)))

    def test_no_region_at_all_is_not_flagged(self):
        self.assertFalse(static_region_degrades_to_cv2(
            self._config(subtitle_area=None, subtitle_areas=None)))


class ExposureProvenanceTests(unittest.TestCase):
    def _config(self) -> ProcessingConfig:
        config = ProcessingConfig()
        config.tbe_enable = True
        return config

    def test_a_static_region_job_reports_a_cv2_fallback_in_provenance(self):
        inpainter = STTNInpainter("cpu", self._config())
        inpainter.inpaint(_frames(), [_static_mask() for _ in range(6)])

        stats = inpainter.last_exposure_stats
        self.assertTrue(stats["degradedToCv2"])
        self.assertGreater(stats["maskedPixels"], 0)
        self.assertLess(
            stats["exposedFraction"], TEMPORAL_EXPOSURE_DEGRADED_FRACTION)
        self.assertIn("cv2", inpainter.backend_name)

        identity = inpainter.execution_identity()
        self.assertEqual(len(identity["fallbackChain"]), 1)
        step = identity["fallbackChain"][0]
        self.assertEqual(step["outcome"], "degraded")
        self.assertIn("cv2", step["provider"])
        self.assertIn("exposed", step["reason"])
        self.assertIn("LaMa", step["recoveryHint"])

    def test_a_moving_region_job_reports_no_fallback(self):
        inpainter = STTNInpainter("cpu", self._config())
        inpainter.inpaint(_frames(), [_moving_mask(i) for i in range(6)])

        stats = inpainter.last_exposure_stats
        self.assertFalse(stats["degradedToCv2"])
        self.assertGreater(
            stats["exposedFraction"], TEMPORAL_EXPOSURE_DEGRADED_FRACTION)
        self.assertNotIn("cv2", inpainter.backend_name)
        self.assertEqual(inpainter.execution_identity()["fallbackChain"], [])

    def test_cv2_really_repaired_the_band_not_the_temporal_path(self):
        """The measurement, not the label: who filled the pixels.

        This is the provider-independent half of the claim and runs
        everywhere. It says the region the user asked STTN to repair was
        repaired by cv2.inpaint almost in its entirety.
        """
        inpainter = STTNInpainter("cpu", self._config())
        inpainter.inpaint(_frames(), [_static_mask() for _ in range(6)])
        stats = inpainter.last_exposure_stats

        self.assertGreater(
            stats["cv2Pixels"] / stats["maskedPixels"], 0.95)
        self.assertEqual(
            stats["cv2Pixels"] + stats["exposedPixels"],
            stats["maskedPixels"],
        )

    def test_the_lama_route_produces_a_different_output(self):
        """The fallback is not cosmetic: the pixels really are different.

        Skips where no LaMa provider is installed, because the comparison
        needs a route that does not depend on temporal exposure and LaMa is
        the one the interface offers.
        """
        from backend.execution_provenance import RequestedStageError
        from backend.inpainters.lama import LAMAInpainter

        frames = _frames()
        masks = [_static_mask() for _ in range(6)]

        sttn = STTNInpainter("cpu", self._config())
        sttn_out = sttn.inpaint([f.copy() for f in frames],
                                [m.copy() for m in masks])
        self.assertTrue(sttn.last_exposure_stats["degradedToCv2"])

        lama_config = self._config()
        lama_config.mode = InpaintMode.LAMA
        try:
            lama = LAMAInpainter("cpu", lama_config)
        except RequestedStageError as exc:
            self.skipTest(f"no LaMa provider on this host: {exc}")
        lama_out = lama.inpaint([f.copy() for f in frames],
                                [m.copy() for m in masks])

        self.assertEqual(len(sttn_out), len(lama_out))
        self.assertNotEqual(_digest(sttn_out), _digest(lama_out))


class MessageTests(unittest.TestCase):
    def test_the_shared_message_names_the_engine_and_the_way_out(self):
        from backend.config import STATIC_REGION_DEGRADES_MESSAGE

        text = STATIC_REGION_DEGRADES_MESSAGE.format(mode="STTN")
        self.assertIn("STTN", text)
        self.assertIn("cv2", text)
        self.assertIn("LaMa", text)


if __name__ == "__main__":
    unittest.main()
