"""RM-349: running an inpainter, and proving it did something.

A named inpainting request that quietly returns the frame it was given is
worse than one that fails, because the report records success. These methods
are the ones that execute an inpainter and then refuse to believe it: they
check the result is not the input, classify the failure when it is, retry a
GPU allocation failure on the CPU, and carry the RIFE fast path.

Mixed into `SubtitleRemover`, so they keep full `self` access.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

import numpy as np

from backend.execution_provenance import (
    FAILURE_OUTPUT_INVALID,
    FAILURE_RUNTIME,
    RequestedStageError,
)
from backend.inpainters import (
    BaseInpainter,
    _detect_scene_cuts,
    free_inference_memory,
    is_oom_error,
)

logger = logging.getLogger(__name__)


class _InpaintMixin:
    """Inpainter execution, result validation, and OOM recovery."""

    def _validate_inpaint_results(
        self,
        frames: List[np.ndarray],
        results: Any,
        masks: Optional[List[np.ndarray]] = None,
    ) -> List[np.ndarray]:
        requested, implementation, provider, chain, execution_contract = (
            self._inpaint_failure_identity()
        )

        def invalid_output(
            detail: str,
            recovery_hint: str,
        ) -> RequestedStageError:
            return RequestedStageError(
                stage="inpaint",
                requested_implementation=requested,
                actual_implementation=implementation,
                provider=provider,
                failure_class=FAILURE_OUTPUT_INVALID,
                detail=detail,
                recovery_hint=recovery_hint,
                fallback_chain=chain,
                selection_policy=("auto" if requested == "auto" else "explicit"),
            )

        if not isinstance(results, (list, tuple)):
            raise invalid_output(
                "the inpainter returned a non-sequence result",
                "Verify the selected inpainter output contract.",
            )
        if len(results) != len(frames):
            raise invalid_output(
                (
                    f"the inpainter returned {len(results)} frame(s) for "
                    f"{len(frames)} input frame(s)"
                ),
                "Verify the selected inpainter output contract.",
            )
        validated: List[np.ndarray] = []
        for index, (source, candidate) in enumerate(zip(frames, results, strict=True)):
            if not isinstance(candidate, np.ndarray) or candidate.shape != source.shape:
                raise invalid_output(
                    f"inpainter frame {index} has an invalid shape or type",
                    "Verify the selected inpainter output contract.",
                )
            if candidate.dtype != np.uint8:
                raise invalid_output(
                    f"inpainter frame {index} is not uint8",
                    "Return uint8 BGR frames from the selected inpainter.",
                )
            validated.append(np.ascontiguousarray(candidate))
        if masks is not None:
            active_masks = 0
            changed_masked_pixels = False
            for source, candidate, mask in zip(frames, validated, masks, strict=True):
                mask_array = np.asarray(mask)
                if mask_array.ndim == 3:
                    active = np.any(mask_array != 0, axis=2)
                elif mask_array.ndim == 2:
                    active = mask_array != 0
                else:
                    continue
                if active.shape != source.shape[:2] or not np.any(active):
                    continue
                active_masks += 1
                if np.any(candidate[active] != source[active]):
                    changed_masked_pixels = True
                    break
            if (
                active_masks
                and not changed_masked_pixels
                and execution_contract != "vsr-inpaint-v1"
            ):
                raise invalid_output(
                    "the inpainter left every active masked pixel unchanged",
                    (
                        "Verify the selected inpainter model and checkpoint, "
                        "then retry."
                    ),
                )
        return validated

    def _inpaint_failure_identity(
        self,
    ) -> tuple[str, str, str, list[dict], str]:
        """Return the last observed route without mistaking Auto for a model."""
        inpainter = self.inpainter
        collect = getattr(inpainter, "execution_identity", None)
        try:
            identity = collect() if callable(collect) else {}
        except Exception:
            identity = {}
        if not isinstance(identity, dict):
            identity = {}

        requested = self.config.mode.value
        try:
            registered = getattr(
                inpainter, "_vsr_registered_implementation", ""
            )
        except Exception:
            registered = ""
        implementation = str(
            identity.get("implementation") or registered or requested
        )
        provider = str(
            identity.get("provider")
            or self._inpainter_provider_name(inpainter)
        )
        executions = identity.get("actualExecutions")
        if isinstance(executions, list):
            executed = []
            for item in executions:
                if not isinstance(item, dict):
                    continue
                try:
                    count = int(item.get("executionCount") or 0)
                except (TypeError, ValueError):
                    count = 0
                if count > 0:
                    executed.append(item)
            implementations = {
                str(item.get("implementation") or "")
                for item in executed
                if str(item.get("implementation") or "")
            }
            providers = {
                str(item.get("provider") or "")
                for item in executed
                if str(item.get("provider") or "")
            }
            if implementations:
                implementation = (
                    next(iter(implementations))
                    if len(implementations) == 1 else "mixed"
                )
            if providers:
                provider = next(iter(providers)) if len(providers) == 1 else "mixed"
        chain = identity.get("fallbackChain")
        return (
            requested,
            implementation,
            provider,
            list(chain) if isinstance(chain, list) else [],
            str(identity.get("executionContract") or ""),
        )

    def _execute_inpainter(
        self,
        frames: List[np.ndarray],
        masks: List[np.ndarray],
    ) -> Any:
        """Run the selected inpainter and classify every runtime failure."""
        try:
            result = self.inpainter.inpaint(frames, masks)
            result = self._validate_inpaint_results(frames, result, masks)
            self._sync_inpaint_provenance(len(frames))
            return result
        except RequestedStageError:
            raise
        except Exception as exc:
            requested, implementation, provider, chain, _execution_contract = (
                self._inpaint_failure_identity()
            )
            raise RequestedStageError(
                stage="inpaint",
                requested_implementation=requested,
                actual_implementation=implementation,
                provider=provider,
                failure_class=FAILURE_RUNTIME,
                detail=str(exc),
                recovery_hint=str(
                    getattr(self.inpainter, "_vsr_recovery_hint", "")
                    or "Verify the selected inpainter and provider, then retry."
                ),
                fallback_chain=chain,
                selection_policy=("auto" if requested == "auto" else "explicit"),
                cause=exc,
            ) from exc

    def _create_inpainter(self) -> BaseInpainter:
        """Construct the configured backend through the device strategy."""
        return self.device_provider.create_inpainter(
            self.config.mode.value,
            self.config.device,
            self.config,
        )

    def _is_inference_oom(self, exc: BaseException) -> bool:
        provider = getattr(self, "device_provider", None)
        check = getattr(provider, "is_oom_error", None)
        return bool(check(exc)) if callable(check) else is_oom_error(exc)

    def _free_inference_memory(self) -> None:
        provider = getattr(self, "device_provider", None)
        release = getattr(provider, "free_inference_memory", None)
        if callable(release):
            release()
        else:
            free_inference_memory()

    def _rife_segment_has_scene_cut(self, frames: List[np.ndarray],
                                    start: int, end: int) -> bool:
        if end <= start + 1:
            return False
        segment = frames[start:end + 1]
        try:
            cuts = _detect_scene_cuts(
                segment,
                threshold=getattr(self.config, "tbe_scene_cut_threshold", 0.35),
                prefer_pyscenedetect=getattr(
                    self.config, "tbe_scene_cut_use_pyscenedetect", False),
                prefer_transnetv2=getattr(
                    self.config, "tbe_scene_cut_use_transnetv2", False),
            )
        except Exception as exc:
            logger.debug(f"RIFE scene-cut probe failed: {exc}")
            return False
        return any(cut > 0 for cut in cuts)

    def _inpaint_with_optional_rife_fast(self,
                                         frames: List[np.ndarray],
                                         masks: List[np.ndarray]) -> List[np.ndarray]:
        stride = self._rife_fast_stride()
        if stride <= 1 or len(frames) < 3:
            return self._execute_inpainter(frames, masks)

        key_indices = list(range(0, len(frames), stride))
        if key_indices[-1] != len(frames) - 1:
            key_indices.append(len(frames) - 1)
        if len(key_indices) >= len(frames):
            return self._execute_inpainter(frames, masks)

        key_frames = [frames[i] for i in key_indices]
        key_masks = [masks[i] for i in key_indices]
        key_results = self._execute_inpainter(key_frames, key_masks)
        if len(key_results) != len(key_indices):
            logger.warning(
                "RIFE fast mode disabled for batch: inpainter returned "
                f"{len(key_results)} keyframes for {len(key_indices)} inputs"
            )
            return self._execute_inpainter(frames, masks)

        results: List[Optional[np.ndarray]] = [None] * len(frames)
        for key_idx, cleaned in zip(key_indices, key_results, strict=True):
            results[key_idx] = self._valid_output_frame(cleaned, frames[key_idx])

        try:
            from backend.decode_accel import maybe_interpolate_pair
        except Exception as exc:
            logger.debug(f"Could not import RIFE adapter: {exc}")
            maybe_interpolate_pair = None

        interpolation_missing_logged = False
        for left_pos, start_idx in enumerate(key_indices[:-1]):
            end_idx = key_indices[left_pos + 1]
            prev_clean = results[start_idx]
            next_clean = results[end_idx]
            if prev_clean is None or next_clean is None:
                continue

            scene_cut = self._rife_segment_has_scene_cut(
                frames, start_idx, end_idx)
            for out_idx in range(start_idx + 1, end_idx):
                t = (out_idx - start_idx) / max(1, end_idx - start_idx)
                fallback = prev_clean if t < 0.5 else next_clean
                if scene_cut or maybe_interpolate_pair is None:
                    results[out_idx] = fallback.copy()
                    continue
                interpolated = maybe_interpolate_pair(prev_clean, next_clean, t)
                if interpolated is None:
                    if not interpolation_missing_logged:
                        logger.info(
                            "RIFE fast mode is using nearest-keyframe fallback; "
                            "install practical-rife to synthesize intermediates."
                        )
                        interpolation_missing_logged = True
                    results[out_idx] = fallback.copy()
                    continue
                results[out_idx] = self._valid_output_frame(
                    interpolated, fallback)

        return [
            result if result is not None else frames[idx].copy()
            for idx, result in enumerate(results)
        ]

    def _inpaint_batch_resilient(self, frames: List[np.ndarray],
                                 masks: List[np.ndarray]) -> List[np.ndarray]:
        """Inpaint a batch, recovering from GPU OOM without changing models.

        On an out-of-memory failure the CUDA cache is cleared and the batch is
        split in half and retried recursively down to a single frame. A frame
        that still cannot run on the GPU retries the same registered
        implementation on CPU.
        The output list always has one frame per input, so a partial/corrupt
        write can never result from a recovered batch.
        """
        if not getattr(self.config, "gpu_oom_recovery", True):
            return self._inpaint_with_optional_rife_fast(frames, masks)
        try:
            return self._inpaint_with_optional_rife_fast(frames, masks)
        except Exception as exc:  # noqa: BLE001 - re-raised unless it is OOM
            if not self._is_inference_oom(exc):
                raise
            self._free_inference_memory()
            if len(frames) <= 1:
                logger.warning(
                    "GPU out of memory on a single frame; retrying the same "
                    "inpainting implementation on CPU."
                )
                return self._retry_inpaint_on_cpu(frames, masks, exc)
            half = max(1, len(frames) // 2)
            logger.warning(
                "GPU out of memory on a batch of %d frames; clearing cache and "
                "retrying as %d + %d.", len(frames), half, len(frames) - half,
            )
            left = self._inpaint_batch_resilient(frames[:half], masks[:half])
            right = self._inpaint_batch_resilient(frames[half:], masks[half:])
            return left + right

    def _retry_inpaint_on_cpu(
        self,
        frames: List[np.ndarray],
        masks: List[np.ndarray],
        original_error: BaseException,
    ) -> List[np.ndarray]:
        requested = self.config.mode.value
        previous = self.inpainter
        previous_identity = str(
            getattr(previous, "_vsr_registered_implementation", "")
            or requested
        )
        try:
            cpu_inpainter = self.device_provider.create_inpainter(
                requested, "cpu", self.config
            )
            cpu_identity = str(
                getattr(cpu_inpainter, "_vsr_registered_implementation", "")
                or requested
            )
            if cpu_identity != previous_identity:
                raise RequestedStageError(
                    stage="inpaint",
                    requested_implementation=requested,
                    actual_implementation=cpu_identity,
                    failure_class=FAILURE_RUNTIME,
                    detail=(
                        "CPU recovery resolved to a different implementation "
                        f"({previous_identity} to {cpu_identity})"
                    ),
                    recovery_hint=(
                        "Reduce the batch size or choose Auto before retrying."
                    ),
                )
            self.inpainter = cpu_inpainter
            self.config.device = "cpu"
            self._device_fallback_reason = (
                f"{previous_identity} exhausted GPU memory and retried on CPU"
            )
            self.execution_provenance.effective_device = "cpu"
            self.execution_provenance.device_fallback_reason = (
                self._device_fallback_reason
            )
            stage = self.execution_provenance.stage("inpaint")
            if stage is not None:
                stage.fallback_chain.extend([
                    {
                        "implementation": previous_identity,
                        "outcome": "runtime_failed",
                        "provider": str(
                            self._inpainter_provider_name(previous)
                        ),
                        "effectiveDevice": stage.effective_device,
                        "failureClass": FAILURE_RUNTIME,
                        "reason": str(original_error),
                    },
                    {
                        "implementation": cpu_identity,
                        "outcome": "selected",
                        "provider": str(
                            self._inpainter_provider_name(cpu_inpainter)
                        ),
                        "effectiveDevice": "cpu",
                        "reason": "same implementation CPU retry",
                    },
                ])
            return self._inpaint_with_optional_rife_fast(frames, masks)
        except RequestedStageError:
            raise
        except Exception as exc:
            raise RequestedStageError(
                stage="inpaint",
                requested_implementation=requested,
                actual_implementation=previous_identity,
                provider=self._inpainter_provider_name(previous),
                failure_class=FAILURE_RUNTIME,
                detail=f"GPU and same-implementation CPU retries failed: {exc}",
                recovery_hint=(
                    "Reduce the batch size, repair the selected provider, or "
                    "choose Auto before retrying."
                ),
                cause=exc,
            ) from exc
