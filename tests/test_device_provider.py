from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from backend.config import InpaintMode, ProcessingConfig
from backend.device_provider import InpainterUnavailableError, RuntimeDeviceProvider
from backend.processor import SubtitleRemover


def test_runtime_provider_selects_available_cuda_and_directml():
    cuda = RuntimeDeviceProvider(
        "cuda:2",
        cuda_probe=lambda index: index == 2,
    )
    directml = RuntimeDeviceProvider(
        "directml",
        directml_probe=lambda: True,
    )
    assert cuda.probe_available() == "cuda:2"
    assert directml.probe_available() == "directml"


def test_tensorrtrtx_status_reports_the_manual_provider_lane():
    from backend.device_provider import tensorrtrtx_status

    fake_ort = SimpleNamespace(
        get_available_providers=lambda: [
            "NvTensorRTRTXExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    def fake_version(package):
        if package == "onnxruntime-gpu":
            return "1.26.0"
        raise ModuleNotFoundError(package)

    with mock.patch.dict("sys.modules", {"onnxruntime": fake_ort}), \
            mock.patch("importlib.metadata.version", side_effect=fake_version):
        status = tensorrtrtx_status()

    assert status["available"] is True
    assert status["package"] == "onnxruntime-gpu"
    assert status["provider"] == "NvTensorRTRTXExecutionProvider"
    assert "CPUExecutionProvider" in status["providers"]


def test_runtime_provider_falls_back_when_accelerator_is_unavailable():
    cuda = RuntimeDeviceProvider("cuda:0", cuda_probe=lambda index: False)
    directml = RuntimeDeviceProvider(
        "directml",
        directml_probe=lambda: False,
    )
    assert cuda.probe_available() == "cpu"
    assert directml.probe_available() == "cpu"


def test_runtime_provider_constructs_registry_backend():
    calls = []

    def resolver(name):
        if name == "missing":
            raise KeyError(name)

        def build(device, config):
            calls.append((name, device, config))
            return SimpleNamespace(name=name, device=device)

        return build

    provider = RuntimeDeviceProvider("cpu", resolver=resolver)
    config = object()
    direct = provider.create_inpainter("lama", "cpu", config)
    assert direct.name == "lama"
    assert calls == [("lama", "cpu", config)]


def test_runtime_provider_raises_actionable_error_for_unregistered_mode():
    """Issue #7: an unregistered model must fail loudly, not silently
    become STTN (which made every model selection produce identical
    output)."""

    def resolver(name):
        raise KeyError(name)

    provider = RuntimeDeviceProvider("cpu", resolver=resolver)
    with pytest.raises(InpainterUnavailableError) as excinfo:
        provider.create_inpainter("diffueraser", "cpu", object())

    message = str(excinfo.value)
    assert "diffueraser" in message
    assert "Registered backends" in message
    assert excinfo.value.requested == "diffueraser"


def test_unavailable_inpainter_keeps_typed_failure_classification():
    from backend.failure_reason import (
        REASON_MODEL_MISSING,
        classify_failure_reason,
    )

    error = InpainterUnavailableError("does-not-exist", ["sttn"])

    assert classify_failure_reason(exc=error) == REASON_MODEL_MISSING


def test_unknown_inpaint_mode_string_raises_value_error():
    """Issue #7 companion: config-level coercion must also reject unknown
    model names instead of silently substituting STTN."""
    from backend.config import _coerce_backend_mode

    with pytest.raises(ValueError) as excinfo:
        _coerce_backend_mode("diffueraser-typo")

    message = str(excinfo.value)
    assert "diffueraser-typo" in message
    assert "Known modes" in message


def test_subtitle_remover_uses_injected_provider_for_selection_and_factory():
    inpainter = SimpleNamespace(inpaint=lambda frames, masks: frames)

    class Provider:
        def __init__(self):
            self.created = []

        def probe_available(self):
            return "cpu"

        def create_inpainter(self, name, device, config):
            self.created.append((name, device, config.device))
            return inpainter

    provider = Provider()
    detector = SimpleNamespace(_engine_name="test")
    config = ProcessingConfig(device="cuda:0", adaptive_batch=False)
    with mock.patch.object(
        SubtitleRemover, "_resolve_work_directory"
    ), mock.patch.object(
        SubtitleRemover, "_select_hw_encoder"
    ), mock.patch(
        "backend.processor.SubtitleDetector", return_value=detector
    ) as detector_factory:
        remover = SubtitleRemover(config, device_provider=provider)

    assert remover.inpainter is inpainter
    assert remover.config.device == "cpu"
    assert provider.created == [("sttn", "cpu", "cpu")]
    detector_factory.assert_called_once_with(
        "cpu", lang="en", vertical=False, engine="auto",
        rapidocr_variant="v6", paddleocr_variant="mobile"
    )


def test_oom_recovery_retries_same_implementation_on_cpu():
    class AlwaysOom:
        backend_name = "same-model-cuda"
        _vsr_registered_implementation = "sttn"

        def inpaint(self, frames, masks):
            raise RuntimeError("synthetic provider OOM")

    class CpuInpainter:
        backend_name = "same-model-cpu"
        _vsr_registered_implementation = "sttn"

        def inpaint(self, frames, masks):
            return [frame.copy() for frame in frames]

    class Provider:
        def __init__(self):
            self.freed = 0
            self.created = []

        def is_oom_error(self, exc):
            return "provider OOM" in str(exc)

        def free_inference_memory(self):
            self.freed += 1

        def create_inpainter(self, name, device, config):
            self.created.append((name, device))
            return CpuInpainter()

    remover = SubtitleRemover.__new__(SubtitleRemover)
    remover.config = ProcessingConfig(device="cpu")
    remover.inpainter = AlwaysOom()
    remover.device_provider = Provider()
    frames = [np.full((8, 8, 3), 120, np.uint8)]
    masks = [np.zeros((8, 8), np.uint8)]

    output = remover._inpaint_batch_resilient(frames, masks)

    assert len(output) == 1
    assert remover.device_provider.freed == 1
    assert remover.device_provider.created == [("sttn", "cpu")]
    assert remover.inpainter.backend_name == "same-model-cpu"


def test_each_successful_inpaint_call_records_its_actual_provider():
    class Inpainter:
        _vsr_registered_implementation = "sttn"

        def __init__(self, provider):
            self.backend_name = provider

        def inpaint(self, frames, masks):
            return [frame.copy() for frame in frames]

    remover = SubtitleRemover.__new__(SubtitleRemover)
    remover.config = ProcessingConfig(device="cuda:0")
    remover.inpainter = Inpainter("CUDAExecutionProvider")
    frame = np.full((8, 8, 3), 120, np.uint8)
    mask = np.zeros((8, 8), np.uint8)

    remover._execute_inpainter([frame, frame], [mask, mask])
    remover.config.device = "cpu"
    remover.inpainter = Inpainter("CPUExecutionProvider")
    remover._execute_inpainter([frame], [mask])

    stage = remover.execution_provenance.to_dict()["stages"]["inpaint"]
    executions = {
        (item["provider"], item["effectiveDevice"]): item["executionCount"]
        for item in stage["actualExecutions"]
    }
    assert executions == {
        ("CUDAExecutionProvider", "cuda"): 2,
        ("CPUExecutionProvider", "cpu"): 1,
    }


def test_cumulative_auto_identity_merges_deltas_across_provider_instances():
    class AutoRoute:
        _vsr_registered_implementation = "auto"

        def __init__(self, implementation, provider, device):
            self.implementation = implementation
            self.backend_name = provider
            self.device = device
            self.count = 0

        def inpaint(self, frames, masks):
            self.count += len(frames)
            return [frame.copy() for frame in frames]

        def execution_identity(self):
            return {
                "implementation": self.implementation,
                "provider": self.backend_name,
                "effectiveDevice": self.device,
                "actualExecutions": [{
                    "implementation": self.implementation,
                    "provider": self.backend_name,
                    "effectiveDevice": self.device,
                    "executionCount": self.count,
                }],
                "fallbackChain": [],
            }

    remover = SubtitleRemover.__new__(SubtitleRemover)
    remover.config = ProcessingConfig(mode=InpaintMode.AUTO, device="cuda:0")
    frame = np.full((8, 8, 3), 120, np.uint8)
    mask = np.zeros((8, 8), np.uint8)
    remover.inpainter = AutoRoute(
        "sttn", "CUDAExecutionProvider", "cuda:0"
    )

    remover._execute_inpainter([frame, frame], [mask, mask])
    remover._sync_inpaint_provenance(2)
    remover.config.device = "cpu"
    remover.inpainter = AutoRoute(
        "propainter", "CPUExecutionProvider", "cpu"
    )
    remover._execute_inpainter([frame], [mask])

    stage = remover.execution_provenance.to_dict()["stages"]["inpaint"]
    executions = {
        item["implementation"]: item["executionCount"]
        for item in stage["actualExecutions"]
    }
    assert executions == {"sttn": 2, "propainter": 1}
    assert stage["actualImplementation"] == "mixed"


def test_provenance_sync_supports_slotted_inpainter():
    class SlottedInpainter:
        __slots__ = ()
        backend_name = "CPUExecutionProvider"
        _vsr_registered_implementation = "sttn"

        def inpaint(self, frames, masks):
            return [frame.copy() for frame in frames]

    remover = SubtitleRemover.__new__(SubtitleRemover)
    remover.config = ProcessingConfig(device="cpu")
    remover.inpainter = SlottedInpainter()
    frame = np.full((8, 8, 3), 120, np.uint8)
    mask = np.zeros((8, 8), np.uint8)

    output = remover._execute_inpainter([frame], [mask])

    assert len(output) == 1
    stage = remover.execution_provenance.to_dict()["stages"]["inpaint"]
    assert stage["actualImplementation"] == "sttn"
    assert stage["actualExecutions"][0]["executionCount"] == 1


def test_auto_tbe_route_reports_cpu_under_a_cuda_request():
    class AutoTbeRoute:
        backend_name = "AUTO (TBE)"
        _vsr_registered_implementation = "auto"

        def inpaint(self, frames, masks):
            return [frame.copy() for frame in frames]

        def execution_identity(self):
            return {
                "implementation": "sttn",
                "provider": "TBE",
                "effectiveDevice": "cuda:0",
                "actualExecutions": [{
                    "implementation": "sttn",
                    "provider": "TBE",
                    "effectiveDevice": "cuda:0",
                    "executionCount": 1,
                }],
                "fallbackChain": [],
            }

    remover = SubtitleRemover.__new__(SubtitleRemover)
    remover.config = ProcessingConfig(
        mode=InpaintMode.AUTO,
        device="cuda:0",
    )
    remover._requested_device = "cuda:0"
    remover.inpainter = AutoTbeRoute()
    remover._refresh_execution_provenance()
    frame = np.full((8, 8, 3), 120, np.uint8)
    mask = np.zeros((8, 8), np.uint8)

    remover._execute_inpainter([frame], [mask])

    stage = remover.execution_provenance.to_dict()["stages"]["inpaint"]
    assert stage["actualExecutions"][0]["effectiveDevice"] == "cpu"
    assert stage["deviceFellBack"] is True


def test_adaptive_vram_probe_and_shutdown_failures_leave_warnings(caplog):
    inpainter = SimpleNamespace(inpaint=lambda frames, masks: frames)

    class Provider:
        def probe_available(self):
            return "cuda:0"

        def create_inpainter(self, _name, _device, _config):
            return inpainter

    fake_nvml = SimpleNamespace(
        nvmlInit=mock.Mock(),
        nvmlDeviceGetHandleByIndex=mock.Mock(return_value=object()),
        nvmlDeviceGetMemoryInfo=mock.Mock(
            side_effect=RuntimeError("probe failed")),
        nvmlShutdown=mock.Mock(side_effect=RuntimeError("shutdown failed")),
    )
    detector = SimpleNamespace(_engine_name="test")
    config = ProcessingConfig(device="cuda:0", adaptive_batch=True)

    with mock.patch.object(
        SubtitleRemover, "_resolve_work_directory"
    ), mock.patch.object(
        SubtitleRemover, "_select_hw_encoder"
    ), mock.patch(
        "backend.processor.SubtitleDetector", return_value=detector
    ), mock.patch.dict("sys.modules", {"pynvml": fake_nvml}):
        with caplog.at_level("WARNING", logger="backend.processor"):
            SubtitleRemover(config, device_provider=Provider())

    messages = [record.getMessage() for record in caplog.records]
    assert "Adaptive batch VRAM probe failed" in messages
    assert "NVML shutdown failed" in messages
