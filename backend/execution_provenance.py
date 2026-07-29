"""Requested vs. effective execution provenance for a processed job.

RM-147: a CUDA request can silently execute RapidOCR on CPU and LaMa on cv2 --
both legitimate fallbacks, but the queue, the JSON report, the sidecar, and the
support bundle only ever showed an ambiguous engine/backend label, so a user
comparing a "GPU" run against a "CPU" run had no way to see they ran the same
way. This module is the single vocabulary for what was asked for, what
actually ran, why it differed, and how fast it went.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


EXECUTION_PROVENANCE_SCHEMA = "vsr.execution_provenance.v1"

# Effective compute classes. Anything that is not the requested class is a
# fallback and must carry a reason.
DEVICE_CPU = "cpu"
DEVICE_CUDA = "cuda"
DEVICE_DIRECTML = "directml"
DEVICE_UNKNOWN = "unknown"


def normalize_device(value: Any) -> str:
    """Map a device string ("cuda:0", "directml", "cpu") to a compute class."""
    text = str(value or "").strip().lower()
    if not text:
        return DEVICE_UNKNOWN
    if text.startswith("cuda"):
        return DEVICE_CUDA
    if "directml" in text or text.startswith("dml"):
        return DEVICE_DIRECTML
    if text.startswith("cpu"):
        return DEVICE_CPU
    return DEVICE_UNKNOWN


def device_from_provider(provider: Any) -> str:
    """Map an ONNX Runtime / OpenVINO provider name to a compute class."""
    text = str(provider or "").strip().lower()
    if not text:
        return DEVICE_UNKNOWN
    if "cuda" in text or "tensorrt" in text:
        return DEVICE_CUDA
    if "dml" in text or "directml" in text:
        return DEVICE_DIRECTML
    if "cpu" in text or "openvino" in text or text in {"cv2", "opencv"}:
        return DEVICE_CPU
    return DEVICE_UNKNOWN


@dataclass
class StageProvenance:
    """One inference stage (OCR detection or inpainting)."""

    stage: str
    requested_device: str = ""
    effective_device: str = ""
    engine: str = ""
    backend: str = ""
    provider: str = ""
    fallback_reason: str = ""

    @property
    def fell_back(self) -> bool:
        requested = normalize_device(self.requested_device)
        effective = normalize_device(self.effective_device)
        if requested in (DEVICE_UNKNOWN, "") or effective in (DEVICE_UNKNOWN, ""):
            return False
        return requested != effective

    def label(self) -> str:
        """Short human label, e.g. ``RapidOCR on CPU (CUDA requested)``."""
        name = self.engine or self.backend or self.stage
        effective = normalize_device(self.effective_device)
        shown = {
            DEVICE_CPU: "CPU",
            DEVICE_CUDA: "CUDA",
            DEVICE_DIRECTML: "DirectML",
        }.get(effective, effective or "unknown")
        text = f"{name} on {shown}"
        if self.fell_back:
            requested = normalize_device(self.requested_device)
            asked = {
                DEVICE_CPU: "CPU",
                DEVICE_CUDA: "CUDA",
                DEVICE_DIRECTML: "DirectML",
            }.get(requested, requested)
            text += f" ({asked} requested)"
        return text

    def to_dict(self) -> dict:
        payload = {
            "stage": self.stage,
            "requestedDevice": normalize_device(self.requested_device),
            "requestedDeviceRaw": str(self.requested_device or ""),
            "effectiveDevice": normalize_device(self.effective_device),
            "effectiveDeviceRaw": str(self.effective_device or ""),
            "engine": self.engine,
            "backend": self.backend,
            "provider": self.provider,
            "fellBack": self.fell_back,
            "fallbackReason": self.fallback_reason,
            "label": self.label(),
        }
        return payload


@dataclass
class ExecutionProvenance:
    """Everything a job records about how it actually executed."""

    requested_device: str = ""
    effective_device: str = ""
    device_fallback_reason: str = ""
    inpaint_mode: str = ""
    stages: dict = field(default_factory=dict)
    frames_processed: int = 0
    processing_seconds: float = 0.0

    def set_stage(self, stage: StageProvenance) -> None:
        self.stages[stage.stage] = stage

    def stage(self, name: str) -> Optional[StageProvenance]:
        return self.stages.get(name)

    @property
    def frames_per_second(self) -> Optional[float]:
        if self.frames_processed <= 0 or self.processing_seconds <= 0:
            return None
        return round(self.frames_processed / self.processing_seconds, 3)

    @property
    def any_fallback(self) -> bool:
        if normalize_device(self.requested_device) != normalize_device(
                self.effective_device):
            return True
        return any(item.fell_back for item in self.stages.values())

    def summary(self) -> str:
        """One line for the queue row and the batch summary."""
        parts = []
        for name in ("ocr", "inpaint"):
            item = self.stages.get(name)
            if item is not None:
                parts.append(item.label())
        if not parts:
            return ""
        speed = self.frames_per_second
        text = " | ".join(parts)
        if speed:
            text += f" | {speed:g} fps"
        return text

    def to_dict(self) -> dict:
        return {
            "schema": EXECUTION_PROVENANCE_SCHEMA,
            "requestedDevice": normalize_device(self.requested_device),
            "requestedDeviceRaw": str(self.requested_device or ""),
            "effectiveDevice": normalize_device(self.effective_device),
            "effectiveDeviceRaw": str(self.effective_device or ""),
            "deviceFallbackReason": self.device_fallback_reason,
            "inpaintMode": self.inpaint_mode,
            "anyFallback": self.any_fallback,
            "stages": {
                name: item.to_dict()
                for name, item in sorted(self.stages.items())
            },
            "framesProcessed": int(self.frames_processed),
            "processingSeconds": round(float(self.processing_seconds), 3),
            "framesPerSecond": self.frames_per_second,
            "summary": self.summary(),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "ExecutionProvenance":
        if not isinstance(payload, dict):
            return cls()
        item = cls(
            requested_device=str(payload.get("requestedDeviceRaw")
                                 or payload.get("requestedDevice") or ""),
            effective_device=str(payload.get("effectiveDeviceRaw")
                                 or payload.get("effectiveDevice") or ""),
            device_fallback_reason=str(payload.get("deviceFallbackReason") or ""),
            inpaint_mode=str(payload.get("inpaintMode") or ""),
            frames_processed=int(payload.get("framesProcessed") or 0),
            processing_seconds=float(payload.get("processingSeconds") or 0.0),
        )
        stages = payload.get("stages")
        if isinstance(stages, dict):
            for name, raw in stages.items():
                if not isinstance(raw, dict):
                    continue
                item.set_stage(StageProvenance(
                    stage=str(name),
                    requested_device=str(raw.get("requestedDeviceRaw")
                                         or raw.get("requestedDevice") or ""),
                    effective_device=str(raw.get("effectiveDeviceRaw")
                                         or raw.get("effectiveDevice") or ""),
                    engine=str(raw.get("engine") or ""),
                    backend=str(raw.get("backend") or ""),
                    provider=str(raw.get("provider") or ""),
                    fallback_reason=str(raw.get("fallbackReason") or ""),
                ))
        return item
