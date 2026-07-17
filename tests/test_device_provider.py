from types import SimpleNamespace
from unittest import mock

import numpy as np

from backend.config import ProcessingConfig
from backend.device_provider import RuntimeDeviceProvider
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


def test_runtime_provider_falls_back_when_accelerator_is_unavailable():
    cuda = RuntimeDeviceProvider("cuda:0", cuda_probe=lambda index: False)
    directml = RuntimeDeviceProvider(
        "directml",
        directml_probe=lambda: False,
    )
    assert cuda.probe_available() == "cpu"
    assert directml.probe_available() == "cpu"


def test_runtime_provider_constructs_registry_backend_and_sttn_fallback():
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
    fallback = provider.create_inpainter("missing", "cpu", config)
    assert direct.name == "lama"
    assert fallback.name == "sttn"
    assert calls == [("lama", "cpu", config), ("sttn", "cpu", config)]


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
        "cpu", lang="en", vertical=False, engine="auto"
    )


def test_oom_recovery_uses_injected_memory_hooks_without_gpu():
    class AlwaysOom:
        def inpaint(self, frames, masks):
            raise RuntimeError("synthetic provider OOM")

    class Provider:
        def __init__(self):
            self.freed = 0

        def is_oom_error(self, exc):
            return "provider OOM" in str(exc)

        def free_inference_memory(self):
            self.freed += 1

    remover = SubtitleRemover.__new__(SubtitleRemover)
    remover.config = ProcessingConfig(device="cpu")
    remover.inpainter = AlwaysOom()
    remover.device_provider = Provider()
    frames = [np.full((8, 8, 3), 120, np.uint8)]
    masks = [np.zeros((8, 8), np.uint8)]

    output = remover._inpaint_batch_resilient(frames, masks)

    assert len(output) == 1
    assert remover.device_provider.freed == 1


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
