from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.onnx_model_info import (
    WINDOWS_ML_BOOTSTRAP_MODULE,
    WINDOWS_ML_BRIDGE_MODULE,
    _tiny_identity_onnx_bytes,
    collect_windows_ml_probe,
    read_onnx_opset_imports,
)


class WindowsMlProbeTests(unittest.TestCase):
    def test_tiny_identity_model_declares_supported_opset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "identity.onnx"
            model.write_bytes(_tiny_identity_onnx_bytes())
            opsets = read_onnx_opset_imports(model)
        self.assertEqual([(item.domain, item.version) for item in opsets], [("", 13)])

    def test_probe_is_not_applicable_off_windows(self):
        status = collect_windows_ml_probe(platform_name="Linux")
        self.assertEqual(status["decision"], "not_applicable")
        self.assertFalse(status["smoke"]["attempted"])

    def test_probe_blocks_when_python_bridge_is_missing(self):
        status = collect_windows_ml_probe(
            platform_name="Windows",
            importer=lambda name: (_ for _ in ()).throw(ImportError(name)),
        )
        self.assertEqual(status["decision"], "blocked")
        self.assertIn("bridge", status["reason"].lower())
        self.assertFalse(status["pythonBridgeInstalled"])

    def test_probe_runs_smoke_with_fake_windows_ml_runtime(self):
        class Bootstrap:
            class InitializeOptions:
                ON_NO_MATCH_SHOW_UI = object()

            @staticmethod
            def initialize(options=None):
                class Context:
                    def __enter__(self):
                        return self

                    def __exit__(self, exc_type, exc, tb):
                        return False

                return Context()

        class Catalog:
            registered = False

            @staticmethod
            def GetDefault():
                return Catalog()

            def RegisterCertifiedAsync(self):
                Catalog.registered = True
                return None

        ml_module = types.SimpleNamespace(ExecutionProviderCatalog=Catalog)

        class FakeSession:
            def __init__(self, path, providers):
                self.path = path
                self.providers = list(providers)

            def get_providers(self):
                return self.providers

            def run(self, _names, inputs):
                return [np.array(inputs["x"], copy=True)]

        fake_ort = types.SimpleNamespace(
            get_available_providers=lambda: ["CPUExecutionProvider"],
            get_ep_devices=lambda: [
                types.SimpleNamespace(
                    ep_name="CPUExecutionProvider",
                    hardware_device=types.SimpleNamespace(type="CPU"),
                )
            ],
            InferenceSession=FakeSession,
        )

        def importer(name):
            if name == WINDOWS_ML_BOOTSTRAP_MODULE:
                return Bootstrap
            if name == WINDOWS_ML_BRIDGE_MODULE:
                return ml_module
            raise ImportError(name)

        status = collect_windows_ml_probe(
            platform_name="Windows",
            importer=importer,
            ort_module=fake_ort,
        )

        self.assertEqual(status["decision"], "candidate")
        self.assertTrue(status["pythonBridgeInstalled"])
        self.assertTrue(status["bootstrapInstalled"])
        self.assertTrue(status["registeredCertifiedProviders"])
        self.assertTrue(Catalog.registered)
        self.assertTrue(status["smoke"]["passed"])
        self.assertEqual(
            status["smoke"]["activeProviders"], ["CPUExecutionProvider"])


if __name__ == "__main__":
    unittest.main()
