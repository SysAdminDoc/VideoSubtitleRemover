"""STTN-style temporal background exposure inpainter."""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from backend.inpainters._common import (
    BaseInpainter,
    _cv2_inpaint,
    apply_finishing,
    _temporal_background_expose,
)


# RM-321: below this share of masked pixels recovered from other frames, the
# visible result is a cv2.inpaint result and calling the run temporal is
# untrue. It is not zero because global-motion alignment warps the mask a
# little, so even a perfectly static region picks up a few exposed pixels at
# the edges. The measured fraction is always reported alongside the verdict.
TEMPORAL_EXPOSURE_DEGRADED_FRACTION = 0.05


class STTNInpainter(BaseInpainter):
    """Temporal-propagation video inpainting via Temporal Background
    Exposure. Falls back to cv2.inpaint only for pixels masked in every
    frame of the batch.
    """

    def __init__(self, device: str = "cuda:0", config=None):
        self.device = device
        from backend.config import ProcessingConfig
        self.config = config or ProcessingConfig()
        # Seed the provenance from the configuration so a report taken before
        # the first batch names the backend this instance will actually use;
        # inpaint() then records the path it really took.
        self._last_backend_name = (
            "TBE (temporal background exposure)"
            if self.config.tbe_enable else "cv2"
        )
        # RM-321: accumulated across every batch of the run so the report
        # describes the job rather than whichever batch happened to be last.
        self.last_exposure_stats: dict = {}
        self._exposure_totals: dict = {}

    @property
    def backend_name(self) -> str:
        return self._last_backend_name

    def _record_exposure(self, stats: dict) -> None:
        """Name the path that actually repaired the pixels.

        RM-321: a fully static region gives every masked pixel zero temporal
        coverage, so temporal background exposure contributes nothing and the
        whole band is repaired by cv2.inpaint. Reporting the temporal engine
        for that run is a truthful-execution failure, not a naming quibble:
        the user picked an engine that did not run.
        """
        # Accumulate across every batch of the run. A per-call verdict
        # would let one moving batch at the end of an otherwise static video
        # report the whole job as a clean temporal run, and a trailing
        # single-frame batch would leave the previous batch's numbers
        # standing as if they described this one.
        totals = getattr(self, "_exposure_totals", None) or {
            "maskedPixels": 0, "exposedPixels": 0, "cv2Pixels": 0,
            "batches": 0,
        }
        totals["maskedPixels"] += int(stats.get("maskedPixels", 0) or 0)
        totals["exposedPixels"] += int(stats.get("exposedPixels", 0) or 0)
        totals["cv2Pixels"] += int(stats.get("cv2Pixels", 0) or 0)
        totals["batches"] += 1
        self._exposure_totals = totals

        masked = int(totals["maskedPixels"])
        exposed = int(totals["exposedPixels"])
        fraction = (exposed / masked) if masked else None
        self.last_exposure_stats = {
            "maskedPixels": masked,
            "exposedPixels": exposed,
            "cv2Pixels": int(totals["cv2Pixels"]),
            "batches": int(totals["batches"]),
            "exposedFraction": fraction,
            "degradedThreshold": TEMPORAL_EXPOSURE_DEGRADED_FRACTION,
            "degradedToCv2": bool(
                masked and fraction is not None
                and fraction < TEMPORAL_EXPOSURE_DEGRADED_FRACTION
            ),
        }
        if self.last_exposure_stats["degradedToCv2"]:
            self._last_backend_name = "cv2 (no temporal exposure)"

    def execution_identity(self) -> dict:
        identity = super().execution_identity()
        stats = getattr(self, "last_exposure_stats", None)
        if not stats:
            return identity
        identity["exposure"] = dict(stats)
        if stats.get("degradedToCv2"):
            identity["fallbackChain"] = list(identity.get("fallbackChain", [])) + [{
                "implementation": "sttn",
                "outcome": "degraded",
                "provider": "cv2 (no temporal exposure)",
                "effectiveDevice": str(getattr(self, "device", "") or "unknown"),
                "reason": (
                    "only "
                    f"{(stats.get('exposedFraction') or 0.0) * 100:.1f}% of "
                    "masked pixels were ever exposed in another frame, so "
                    "temporal background exposure contributed almost nothing "
                    "and cv2.inpaint repaired the region"
                ),
                "recoveryHint": (
                    "Let automatic detection run per frame, or switch the "
                    "job to LaMa, which does not depend on temporal exposure."
                ),
            }]
        return identity

    def inpaint(self, frames: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
        if self.config.tbe_enable and len(frames) > 1:
            self._last_backend_name = "TBE (temporal background exposure)"
            stats: dict = {}
            result = _temporal_background_expose(
                frames, masks,
                min_coverage=max(1, self.config.tbe_min_coverage),
                use_median=self.config.tbe_use_median,
                feather_px=self.config.mask_feather_px,
                edge_ring_px=self.config.edge_ring_px,
                flow_warp=self.config.tbe_flow_warp,
                flow_estimator=getattr(
                    self.config, "tbe_flow_estimator", "dis"),
                global_motion_align=getattr(
                    self.config, "tbe_global_motion_align", True),
                grain_strength=getattr(
                    self.config, "film_grain_strength", 0.0),
                scene_cut_split=self.config.tbe_scene_cut_split,
                scene_cut_threshold=self.config.tbe_scene_cut_threshold,
                scene_cut_use_pyscenedetect=self.config.tbe_scene_cut_use_pyscenedetect,
                scene_cut_use_transnetv2=self.config.tbe_scene_cut_use_transnetv2,
                translucency_enable=getattr(
                    self.config, "translucency_enable", True),
                exposure_stats=stats,
            )
            self._record_exposure(stats)
            return result
        # A batch that takes the cv2 route contributed no temporal
        # exposure at all, and leaving the previous batch's numbers in place
        # would report its verdict against this one.
        self._record_exposure({})
        self._last_backend_name = "cv2"
        filled = [_cv2_inpaint(f, m, 3, cv2.INPAINT_TELEA)
                  for f, m in zip(frames, masks, strict=True)]
        return apply_finishing(frames, filled, masks, self.config)
