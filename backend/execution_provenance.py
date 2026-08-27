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


EXECUTION_PROVENANCE_SCHEMA = "vsr.execution_provenance.v2"

FAILURE_DEPENDENCY_MISSING = "dependency_missing"
FAILURE_POLICY_BLOCKED = "policy_blocked"
FAILURE_INITIALIZATION = "initialization_failed"
FAILURE_RUNTIME = "runtime_failed"
FAILURE_OUTPUT_MISSING = "output_missing"
FAILURE_OUTPUT_INVALID = "output_invalid"

STAGE_FAILURE_CLASSES = frozenset({
    FAILURE_DEPENDENCY_MISSING,
    FAILURE_POLICY_BLOCKED,
    FAILURE_INITIALIZATION,
    FAILURE_RUNTIME,
    FAILURE_OUTPUT_MISSING,
    FAILURE_OUTPUT_INVALID,
})

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


def normalize_implementation(value: Any) -> str:
    """Normalize a stage implementation name for truthful comparisons."""
    return str(value or "").strip().casefold().replace("_", "-")


def _fallback_step(value: Any) -> dict:
    """Return one bounded, JSON-safe fallback-chain entry."""
    if isinstance(value, str):
        return {
            "implementation": value,
            "outcome": "attempted",
            "provider": "",
            "effectiveDevice": "unknown",
            "failureClass": "",
            "reason": "",
            "recoveryHint": "",
        }
    if not isinstance(value, dict):
        return {
            "implementation": str(value or ""),
            "outcome": "attempted",
            "provider": "",
            "effectiveDevice": "unknown",
            "failureClass": "",
            "reason": "",
            "recoveryHint": "",
        }
    failure_class = str(
        value.get("failureClass") or value.get("failure_class") or ""
    )
    if failure_class not in STAGE_FAILURE_CLASSES:
        failure_class = ""
    return {
        "implementation": str(value.get("implementation") or ""),
        "outcome": str(value.get("outcome") or "attempted"),
        "provider": str(value.get("provider") or ""),
        "effectiveDevice": normalize_device(
            value.get("effectiveDevice") or value.get("effective_device")
        ),
        "failureClass": failure_class,
        "reason": str(value.get("reason") or ""),
        "recoveryHint": str(
            value.get("recoveryHint") or value.get("recovery_hint") or ""
        ),
    }


def _actual_execution(value: Any) -> dict:
    if not isinstance(value, dict):
        value = {"implementation": str(value or "")}
    try:
        count = max(0, int(value.get("executionCount") or 0))
    except (TypeError, ValueError):
        count = 0
    return {
        "implementation": str(value.get("implementation") or ""),
        "provider": str(value.get("provider") or ""),
        "effectiveDevice": normalize_device(
            value.get("effectiveDevice") or value.get("effective_device")
        ),
        "executionCount": count,
    }


def _bounded_list(value: Any) -> list:
    """Return list-like schema input without iterating strings or mappings."""
    return list(value) if isinstance(value, (list, tuple)) else []


@dataclass
class StageProvenance:
    """One requested stage and the implementation that actually ran."""

    stage: str
    requested_device: str = ""
    effective_device: str = ""
    engine: str = ""
    backend: str = ""
    provider: str = ""
    fallback_reason: str = ""
    requested_implementation: str = ""
    actual_implementation: str = ""
    fallback_chain: list[dict] = field(default_factory=list)
    actual_executions: list[dict] = field(default_factory=list)
    selection_policy: str = ""
    outcome: str = "executed"
    failure_class: str = ""
    recovery_hint: str = ""
    model_provenance: dict = field(default_factory=dict)

    @property
    def implementation_fell_back(self) -> bool:
        requested = normalize_implementation(self.requested_implementation)
        actual = normalize_implementation(self.actual_implementation)
        if not requested or requested == "auto" or not actual:
            return False
        return requested != actual

    @property
    def chain_fell_back(self) -> bool:
        failed_seen = False
        for raw in self.fallback_chain:
            step = _fallback_step(raw)
            if step["failureClass"] or step["outcome"].endswith("failed"):
                failed_seen = True
            elif failed_seen and step["outcome"] in {"selected", "executed"}:
                return True
        return False

    @property
    def device_fell_back(self) -> bool:
        requested = normalize_device(self.requested_device)
        effective = normalize_device(self.effective_device)
        if requested in (DEVICE_UNKNOWN, "") or effective in (DEVICE_UNKNOWN, ""):
            return False
        return requested != effective

    @property
    def failed(self) -> bool:
        return self.failure_class in STAGE_FAILURE_CLASSES

    @property
    def fell_back(self) -> bool:
        return (
            self.device_fell_back
            or self.implementation_fell_back
            or self.chain_fell_back
        )

    @property
    def resolved_actual_implementation(self) -> str:
        implementations = {
            normalize_implementation(item.get("implementation"))
            for item in self.actual_executions
            if normalize_implementation(item.get("implementation"))
        }
        if len(implementations) > 1:
            return "mixed"
        if len(implementations) == 1:
            return next(iter(implementations))
        return self.actual_implementation

    def record_execution(
        self,
        implementation: str,
        *,
        provider: str = "",
        effective_device: str = "",
        count: int = 1,
    ) -> None:
        """Add an observed execution without discarding earlier Auto routes."""
        normalized = normalize_implementation(implementation)
        device = normalize_device(effective_device)
        for item in self.actual_executions:
            if (
                normalize_implementation(item.get("implementation")) == normalized
                and str(item.get("provider") or "") == str(provider or "")
                and normalize_device(item.get("effectiveDevice")) == device
            ):
                item["executionCount"] = max(
                    0, int(item.get("executionCount") or 0)
                ) + max(0, int(count))
                self.actual_implementation = self.resolved_actual_implementation
                return
        self.actual_executions.append(_actual_execution({
            "implementation": implementation,
            "provider": provider,
            "effectiveDevice": effective_device,
            "executionCount": count,
        }))
        self.actual_implementation = self.resolved_actual_implementation

    def label(self) -> str:
        """Short human label, e.g. ``RapidOCR on CPU (CUDA requested)``."""
        name = (
            self.engine
            or self.resolved_actual_implementation
            or self.requested_implementation
            or self.backend
            or self.stage
        )
        if self.failed:
            return f"{name} failed"
        if self.outcome != "executed":
            return (
                f"{name} initialized, not run"
                if self.outcome == "initialized"
                else f"{name} not run"
            )
        effective = normalize_device(self.effective_device)
        shown = {
            DEVICE_CPU: "CPU",
            DEVICE_CUDA: "CUDA",
            DEVICE_DIRECTML: "DirectML",
        }.get(effective, effective or "unknown")
        text = f"{name} on {shown}"
        if self.fell_back:
            if self.implementation_fell_back:
                text += f" ({self.requested_implementation} requested)"
            elif self.device_fell_back:
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
            "selectionPolicy": (
                self.selection_policy
                if self.selection_policy in {"explicit", "auto"}
                else (
                    "auto"
                    if normalize_implementation(self.requested_implementation)
                    == "auto" else "explicit"
                )
            ),
            "requestedImplementation": self.requested_implementation,
            "actualImplementation": self.resolved_actual_implementation,
            "actualExecutions": [
                _actual_execution(item) for item in self.actual_executions
            ],
            "fellBack": self.fell_back,
            "implementationFellBack": self.implementation_fell_back,
            "deviceFellBack": self.device_fell_back,
            "chainFellBack": self.chain_fell_back,
            "fallbackReason": self.fallback_reason,
            "fallbackChain": [
                _fallback_step(step) for step in self.fallback_chain
            ],
            "failureClass": (
                self.failure_class
                if self.failure_class in STAGE_FAILURE_CLASSES else ""
            ),
            "recoveryHint": self.recovery_hint,
            "outcome": "failed" if self.failed else self.outcome,
            "status": (
                "failed"
                if self.failed else (
                    "succeeded" if self.outcome == "executed" else "not_run"
                )
            ),
            "label": self.label(),
        }
        if self.model_provenance:
            payload["modelProvenance"] = dict(self.model_provenance)
        return payload


class RequestedStageError(RuntimeError):
    """A named stage could not run without changing implementation."""

    def __init__(
        self,
        *,
        stage: str,
        requested_implementation: str,
        failure_class: str,
        recovery_hint: str,
        detail: str = "",
        actual_implementation: str = "",
        provider: str = "",
        fallback_chain: Optional[list[dict]] = None,
        selection_policy: str = "",
        cause: Optional[BaseException] = None,
        retriable: bool = False,
    ) -> None:
        if failure_class not in STAGE_FAILURE_CLASSES:
            raise ValueError(f"unknown stage failure class: {failure_class}")
        self.stage = str(stage or "stage")
        self.requested_implementation = str(requested_implementation or "")
        self.actual_implementation = str(actual_implementation or "")
        self.provider = str(provider or "")
        self.failure_class = failure_class
        self.recovery_hint = str(recovery_hint or "")
        self.detail = str(detail or "")
        self.fallback_chain = [
            _fallback_step(step) for step in _bounded_list(fallback_chain)
        ]
        self.selection_policy = (
            selection_policy
            if selection_policy in {"explicit", "auto"}
            else (
                "auto"
                if normalize_implementation(self.requested_implementation)
                == "auto" else "explicit"
            )
        )
        self.cause = cause
        self.retriable = bool(retriable)
        message = (
            f"Requested {self.stage} implementation "
            f"'{self.requested_implementation}' failed ({self.failure_class})"
        )
        if self.detail:
            message += f": {self.detail}"
        if self.recovery_hint:
            message += f". Recovery: {self.recovery_hint}"
        super().__init__(message)

    def stage_provenance(
        self,
        *,
        requested_device: str = "",
        effective_device: str = "",
    ) -> StageProvenance:
        return StageProvenance(
            stage=self.stage,
            requested_device=requested_device,
            effective_device=effective_device,
            engine=self.requested_implementation,
            backend=self.actual_implementation,
            provider=self.provider,
            requested_implementation=self.requested_implementation,
            actual_implementation=self.actual_implementation,
            fallback_chain=list(self.fallback_chain),
            selection_policy=self.selection_policy,
            outcome="failed",
            failure_class=self.failure_class,
            recovery_hint=self.recovery_hint,
        )


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

    def begin_stage(
        self,
        name: str,
        *,
        requested_implementation: str,
        requested_device: str = "",
        selection_policy: str = "explicit",
    ) -> StageProvenance:
        stage = StageProvenance(
            stage=name,
            requested_device=requested_device or self.requested_device,
            requested_implementation=requested_implementation,
            selection_policy=selection_policy,
            outcome="notApplicable",
        )
        self.set_stage(stage)
        return stage

    def record_success(
        self,
        name: str,
        *,
        implementation: str,
        provider: str = "",
        effective_device: str = "",
        count: int = 1,
    ) -> StageProvenance:
        stage = self.stages.get(name)
        if stage is None:
            stage = self.begin_stage(
                name,
                requested_implementation=implementation,
                requested_device=self.requested_device,
            )
        stage.actual_implementation = implementation
        stage.provider = provider
        stage.effective_device = effective_device or self.effective_device
        stage.outcome = "executed"
        stage.failure_class = ""
        stage.recovery_hint = ""
        stage.record_execution(
            implementation,
            provider=provider,
            effective_device=stage.effective_device,
            count=count,
        )
        return stage

    def record_failure(
        self,
        error: RequestedStageError,
        *,
        requested_device: str = "",
        effective_device: str = "",
    ) -> StageProvenance:
        previous = self.stages.get(error.stage)
        stage = error.stage_provenance(
            requested_device=requested_device or self.requested_device,
            effective_device=effective_device or self.effective_device,
        )
        if previous is not None:
            stage.actual_executions = [
                _actual_execution(item) for item in previous.actual_executions
            ]
            previous_chain = [
                _fallback_step(step) for step in previous.fallback_chain
            ]
            failure_chain = [
                _fallback_step(step) for step in stage.fallback_chain
            ]
            overlap = 0
            for count in range(
                min(len(previous_chain), len(failure_chain)), 0, -1
            ):
                if previous_chain[-count:] == failure_chain[:count]:
                    overlap = count
                    break
            stage.fallback_chain = previous_chain + failure_chain[overlap:]
            if not stage.actual_implementation:
                stage.actual_implementation = previous.actual_implementation
            if not stage.provider:
                stage.provider = previous.provider
        self.set_stage(stage)
        return stage

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
                    requested_implementation=str(
                        raw.get("requestedImplementation") or ""),
                    actual_implementation=str(
                        raw.get("actualImplementation") or ""),
                    fallback_chain=[
                        _fallback_step(step)
                        for step in _bounded_list(raw.get("fallbackChain"))
                    ],
                    actual_executions=[
                        _actual_execution(execution)
                        for execution in _bounded_list(
                            raw.get("actualExecutions")
                        )
                    ],
                    selection_policy=str(raw.get("selectionPolicy") or ""),
                    outcome=str(raw.get("outcome") or "executed"),
                    failure_class=str(raw.get("failureClass") or ""),
                    recovery_hint=str(raw.get("recoveryHint") or ""),
                    model_provenance=(
                        dict(raw.get("modelProvenance"))
                        if isinstance(raw.get("modelProvenance"), dict)
                        else {}
                    ),
                ))
        return item
