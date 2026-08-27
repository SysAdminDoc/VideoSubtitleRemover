"""AUTO inpainter: per-scene STTN/ProPainter routing with lazy loading."""

from __future__ import annotations

import logging
import sys
from typing import List, Optional

import numpy as np

import cv2

from backend.execution_provenance import (
    FAILURE_DEPENDENCY_MISSING,
    FAILURE_INITIALIZATION,
    FAILURE_OUTPUT_INVALID,
    FAILURE_OUTPUT_MISSING,
    FAILURE_POLICY_BLOCKED,
    FAILURE_RUNTIME,
    RequestedStageError,
)
from backend.inpainters._common import BaseInpainter, _detect_scene_cuts
from backend.inpainters.sttn import STTNInpainter
from backend.inpainters.propainter import ProPainterInpainter

logger = logging.getLogger(__name__)


class AutoInpainter(BaseInpainter):
    """Route each scene to STTN or motion-robust ProPainter mode.

    STTN handles low-motion scenes whose masked pixels become exposed over
    time. ProPainter mode handles fast motion or persistently covered pixels.
    Its optional LaMa refinement remains lazy and is unloaded after a long
    streak of STTN scenes so easy videos do not pin model memory.
    """

    PROPAINTER_IDLE_UNLOAD_AFTER = 50
    MOTION_THRESHOLD = 0.04

    def __init__(self, device: str = "cuda:0", config=None):
        self.device = device
        from backend.config import ProcessingConfig
        self.config = config or ProcessingConfig()
        self._sttn = STTNInpainter(device, self.config)
        self._propainter: Optional[ProPainterInpainter] = None
        self._sttn_streak: int = 0
        self._execution_counts: dict[tuple[str, str], int] = {}
        self._route_chain: list[dict] = []

    @property
    def backend_name(self) -> str:
        if self._propainter is not None:
            return f"AUTO ({self._propainter.backend_name})"
        return f"AUTO ({self._sttn.backend_name})"

    def _ensure_propainter(self) -> ProPainterInpainter:
        if self._propainter is None:
            self._propainter = ProPainterInpainter(self.device, self.config)
        return self._propainter

    @staticmethod
    def _provider_name(value: object, default: str) -> str:
        try:
            return str(getattr(value, "backend_name", "") or default)
        except Exception:
            return default

    def _effective_device_for_provider(
        self, provider: str, fallback: str = ""
    ) -> str:
        lowered = str(provider or "").lower()
        if any(token in lowered for token in ("cv2", "opencv", "tbe")):
            return "cpu"
        if "cuda" in lowered or "tensorrt" in lowered:
            return self.device if str(self.device).lower().startswith("cuda") else "cuda"
        if "directml" in lowered or "dml" in lowered:
            return "directml"
        if "cpu" in lowered or "openvino" in lowered:
            return "cpu"
        return str(fallback or self.device)

    def _maybe_unload_propainter(self) -> None:
        if self._propainter is None:
            return
        if self._sttn_streak < self.PROPAINTER_IDLE_UNLOAD_AFTER:
            return
        logger.info(
            "AUTO: unloading idle ProPainter mode after %d STTN scenes",
            self._sttn_streak,
        )
        self._propainter = None
        try:
            import gc as _gc
            _gc.collect()
        except Exception:
            logger.warning(
                "AUTO ProPainter idle GC cleanup failed", exc_info=True)
        torch_mod = sys.modules.get("torch")
        if torch_mod is None:
            return
        try:
            if hasattr(torch_mod, "cuda") and torch_mod.cuda.is_available():
                torch_mod.cuda.empty_cache()
        except Exception:
            logger.warning(
                "AUTO ProPainter idle CUDA cleanup failed", exc_info=True)

    @staticmethod
    def _exposure_score(masks: List[np.ndarray]) -> float:
        if len(masks) < 2:
            return 0.0
        stack = np.stack(masks, axis=0)
        unmasked = (stack == 0)
        any_union = unmasked.any(axis=0)
        ever_masked = (stack > 0).any(axis=0)
        total = int(ever_masked.sum())
        if total == 0:
            return 1.0
        exposed = int((ever_masked & any_union).sum())
        return exposed / float(total)

    @staticmethod
    def _motion_score(frames: List[np.ndarray]) -> float:
        """Return mean normalized luminance change between adjacent frames."""
        if len(frames) < 2:
            return 0.0
        scores = []
        previous = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        for frame in frames[1:]:
            current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            scores.append(float(cv2.absdiff(previous, current).mean()) / 255.0)
            previous = current
        return sum(scores) / len(scores)

    def _scene_starts(self, frames: List[np.ndarray]) -> List[int]:
        if not self.config.tbe_scene_cut_split or len(frames) < 2:
            return [0]
        return _detect_scene_cuts(
            frames,
            self.config.tbe_scene_cut_threshold,
            prefer_pyscenedetect=self.config.tbe_scene_cut_use_pyscenedetect,
            prefer_transnetv2=self.config.tbe_scene_cut_use_transnetv2,
        )

    def _inpaint_scene(
        self, frames: List[np.ndarray], masks: List[np.ndarray], scene_index: int
    ) -> List[np.ndarray]:
        threshold = self.config.auto_exposure_threshold
        exposure = self._exposure_score(masks)
        motion = self._motion_score(frames)
        if exposure >= threshold and motion < self.MOTION_THRESHOLD:
            logger.debug(
                "AUTO scene %d: STTN path (exposure=%.2f, motion=%.3f)",
                scene_index,
                exposure,
                motion,
            )
            self._sttn_streak += 1
            self._maybe_unload_propainter()
            provider = self._provider_name(self._sttn, "STTNInpainter")
            try:
                result = self._sttn.inpaint(frames, masks)
            except Exception as exc:
                raise self._route_failure(
                    "sttn", provider, exc
                ) from exc
            self._record_route("sttn", provider, len(frames))
            return result
        logger.debug(
            "AUTO scene %d: ProPainter path (exposure=%.2f, motion=%.3f)",
            scene_index,
            exposure,
            motion,
        )
        self._sttn_streak = 0
        try:
            propainter = self._ensure_propainter()
            result = propainter.inpaint(frames, masks)
        except Exception as exc:
            provider = (
                str(exc.provider)
                if isinstance(exc, RequestedStageError) and exc.provider
                else self._provider_name(
                    getattr(self, "_propainter", None),
                    "ProPainterInpainter",
                )
            )
            raise self._route_failure(
                "propainter", provider, exc
            ) from exc
        provider = self._provider_name(
            propainter, type(propainter).__name__
        )
        self._record_route(
            "propainter",
            provider,
            len(frames),
        )
        return result

    def _route_failure(
        self,
        implementation: str,
        provider: str,
        exc: BaseException,
    ) -> RequestedStageError:
        if isinstance(exc, RequestedStageError):
            failure_class = exc.failure_class
            detail = exc.detail or str(exc)
            recovery_hint = exc.recovery_hint
            retriable = exc.retriable
        else:
            failure_class = FAILURE_RUNTIME
            detail = str(exc)
            recovery_hint = (
                "Verify the selected Auto route and provider, then retry."
            )
            retriable = False
        outcome = {
            FAILURE_DEPENDENCY_MISSING: "load_failed",
            FAILURE_POLICY_BLOCKED: "load_failed",
            FAILURE_INITIALIZATION: "load_failed",
            FAILURE_RUNTIME: "runtime_failed",
            FAILURE_OUTPUT_MISSING: "output_failed",
            FAILURE_OUTPUT_INVALID: "output_failed",
        }[failure_class]
        nested_chain = (
            list(exc.fallback_chain)
            if isinstance(exc, RequestedStageError)
            else []
        )
        if nested_chain:
            failure_steps = []
            for raw in nested_chain:
                step = dict(raw)
                step["implementation"] = str(
                    step.get("implementation") or implementation
                )
                step["provider"] = str(step.get("provider") or provider)
                step["effectiveDevice"] = self._effective_device_for_provider(
                    step["provider"], str(step.get("effectiveDevice") or "")
                )
                failure_steps.append(step)
        else:
            failure_steps = [{
                "implementation": implementation,
                "outcome": outcome,
                "provider": provider,
                "effectiveDevice": self._effective_device_for_provider(provider),
                "failureClass": failure_class,
                "reason": detail,
                "recoveryHint": recovery_hint,
            }]
        self._route_chain.extend(failure_steps)
        return RequestedStageError(
            stage="inpaint",
            requested_implementation="auto",
            actual_implementation=(
                exc.actual_implementation
                if isinstance(exc, RequestedStageError)
                and exc.actual_implementation
                else implementation
            ),
            provider=(
                exc.provider
                if isinstance(exc, RequestedStageError) and exc.provider
                else provider
            ),
            failure_class=failure_class,
            detail=detail,
            recovery_hint=recovery_hint,
            fallback_chain=list(self._route_chain),
            selection_policy="auto",
            cause=exc,
            retriable=retriable,
        )

    def _record_route(
        self, implementation: str, provider: str, frame_count: int
    ) -> None:
        key = (implementation, provider)
        self._execution_counts[key] = (
            self._execution_counts.get(key, 0) + max(0, int(frame_count))
        )
        self._route_chain.append({
            "implementation": implementation,
            "outcome": "executed",
            "provider": provider,
            "effectiveDevice": self._effective_device_for_provider(provider),
            "failureClass": "",
            "reason": "scene routing",
            "recoveryHint": "",
        })

    def execution_identity(self) -> dict:
        executions = [
            {
                "implementation": implementation,
                "provider": provider,
                "effectiveDevice": self._effective_device_for_provider(provider),
                "executionCount": count,
            }
            for (implementation, provider), count
            in sorted(self._execution_counts.items())
        ]
        implementations = {
            item["implementation"] for item in executions
        }
        actual = (
            "mixed" if len(implementations) > 1
            else next(iter(implementations), "")
        )
        return {
            "implementation": actual,
            "provider": self.backend_name,
            "effectiveDevice": self.device,
            "actualExecutions": executions,
            # Scene routing is an execution trace, not a failure fallback.
            "fallbackChain": list(self._route_chain),
        }

    def inpaint(
        self, frames: List[np.ndarray], masks: List[np.ndarray]
    ) -> List[np.ndarray]:
        if not frames:
            return []
        starts = self._scene_starts(frames)
        results: List[np.ndarray] = []
        for scene_index, start in enumerate(starts):
            end = starts[scene_index + 1] if scene_index + 1 < len(starts) else len(frames)
            results.extend(self._inpaint_scene(
                frames[start:end], masks[start:end], scene_index))
        return results
