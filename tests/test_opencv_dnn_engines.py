"""RM-149: selectable OpenCV 5 DNN inference engines are contract-tested."""

from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend import opencv_ocr


class _FakeNet:
    def __init__(self, behaviour):
        self._behaviour = behaviour
        self.engine = None
        self._blob = None

    def setPreferableEngine(self, value):
        self.engine = value

    def setInput(self, blob):
        self._blob = blob

    def forward(self):
        return self._behaviour(self.engine, self._blob)


def _fake_cv(behaviour, *, engines=("auto", "classic", "new"),
             supports_selection=True):
    dnn = types.SimpleNamespace(readNetFromONNX=lambda path: _FakeNet(behaviour))
    for index, name in enumerate(engines):
        setattr(dnn, opencv_ocr._DNN_ENGINE_ATTRIBUTES[name], index)
    if not supports_selection:
        class _NoSelect(_FakeNet):
            setPreferableEngine = None

        dnn.readNetFromONNX = lambda path: _NoSelect(behaviour)
    return types.SimpleNamespace(__version__="5.0.0.93", dnn=dnn)


def _agreeing(_engine, blob):
    return np.asarray(blob, dtype=np.float32) * 2.0


class EngineDiscoveryTests(unittest.TestCase):
    def test_documented_engines_are_reported_from_the_installed_build(self):
        cv = _fake_cv(_agreeing)
        self.assertEqual(
            opencv_ocr.available_dnn_engines(cv), ["auto", "classic", "new"])

    def test_partial_builds_only_report_what_they_expose(self):
        cv = _fake_cv(_agreeing, engines=("auto",))
        self.assertEqual(opencv_ocr.available_dnn_engines(cv), ["auto"])

    def test_opencv_4_exposes_no_engine_selection(self):
        cv = types.SimpleNamespace(
            __version__="4.11.0", dnn=types.SimpleNamespace(readNetFromONNX=None))
        self.assertEqual(opencv_ocr.available_dnn_engines(cv), [])

    def test_unknown_engine_name_has_no_constant(self):
        self.assertIsNone(
            opencv_ocr.dnn_engine_value("banana", _fake_cv(_agreeing)))


class EngineContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        model = Path(self.tmp.name) / "PP-OCRv6_det_small.onnx"
        model.write_bytes(b"onnx")
        self.model = model
        self._status = mock.patch.object(
            opencv_ocr, "collect_opencv_dnn_ocr_status",
            lambda **kwargs: {
                "models": {"det": {"present": True, "path": str(model)}},
            },
        )
        self._status.start()
        self.addCleanup(self._status.stop)

    def _run(self, cv):
        return opencv_ocr.run_opencv_dnn_engine_contract(
            cv_module=cv, include_lama=False)

    def test_agreeing_engines_pass_and_record_the_engine(self):
        result = self._run(_fake_cv(_agreeing))
        self.assertTrue(result["ran"])
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["availableEngines"], ["auto", "classic", "new"])
        model = result["models"][0]
        self.assertEqual(
            [item["engine"] for item in model["engines"]],
            ["auto", "classic", "new"],
        )
        for item in model["engines"]:
            self.assertTrue(item["loaded"])
            self.assertTrue(item["ran"])
            self.assertEqual(item["outputShape"], [1, 3, 64, 64])

    def test_a_load_failure_on_one_engine_fails_the_contract(self):
        def behaviour(engine, blob):
            if engine == 2:  # the "new" engine
                raise RuntimeError("unsupported layer")
            return _agreeing(engine, blob)

        result = self._run(_fake_cv(behaviour))
        self.assertFalse(result["passed"])
        self.assertTrue(any("new engine failed" in e for e in result["errors"]))

    def test_a_shape_regression_fails_the_contract(self):
        def behaviour(engine, blob):
            if engine == 1:
                return np.zeros((1, 3, 32, 32), dtype=np.float32)
            return _agreeing(engine, blob)

        result = self._run(_fake_cv(behaviour))
        self.assertFalse(result["passed"])
        self.assertTrue(
            any("output shape" in e for e in result["errors"]), result["errors"])

    def test_materially_divergent_output_fails_the_contract(self):
        def behaviour(engine, blob):
            base = _agreeing(engine, blob)
            return base + 5.0 if engine == 2 else base

        result = self._run(_fake_cv(behaviour))
        self.assertFalse(result["passed"])
        self.assertTrue(any("diverges" in e for e in result["errors"]))

    def test_negligible_numeric_noise_still_passes(self):
        def behaviour(engine, blob):
            base = _agreeing(engine, blob)
            return base + (1e-7 if engine else 0.0)

        result = self._run(_fake_cv(behaviour))
        self.assertTrue(result["passed"], result["errors"])

    def test_no_engine_selection_is_reported_not_assumed_to_pass(self):
        cv = types.SimpleNamespace(
            __version__="4.11.0", dnn=types.SimpleNamespace(readNetFromONNX=None))
        result = opencv_ocr.run_opencv_dnn_engine_contract(cv_module=cv)
        self.assertFalse(result["ran"])
        self.assertIsNone(result["passed"])
        self.assertTrue(result["errors"])

    def test_missing_bundled_model_is_reported(self):
        with mock.patch.object(
            opencv_ocr, "collect_opencv_dnn_ocr_status",
            lambda **kwargs: {"models": {"det": {"present": False, "path": ""}}},
        ):
            result = self._run(_fake_cv(_agreeing))
        self.assertFalse(result["ran"])
        self.assertTrue(
            any("PP-OCRv6" in item for item in result["errors"]))

    def test_contract_does_not_require_an_ort_linked_dnn_build(self):
        # The fake cv module has no onnxruntime attribute at all.
        cv = _fake_cv(_agreeing)
        self.assertFalse(hasattr(cv, "onnxruntime"))
        self.assertTrue(self._run(cv)["passed"])


class SessionEngineSelectionTests(unittest.TestCase):
    def test_session_applies_and_records_the_requested_engine(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "det.onnx"
            model.write_bytes(b"onnx")
            cv = _fake_cv(_agreeing)
            with mock.patch.object(
                opencv_ocr, "read_onnx_metadata_props", lambda _p: {}
            ):
                session = opencv_ocr.OpenCVDnnSession(
                    {"model_path": str(model)}, cv_module=cv, engine="new")
            self.assertEqual(session.engine, "new")
            self.assertEqual(session._net.engine, 2)

    def test_session_without_engine_selection_stays_neutral(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "det.onnx"
            model.write_bytes(b"onnx")
            cv = _fake_cv(_agreeing, supports_selection=False)
            with mock.patch.object(
                opencv_ocr, "read_onnx_metadata_props", lambda _p: {}
            ):
                session = opencv_ocr.OpenCVDnnSession(
                    {"model_path": str(model)}, cv_module=cv, engine="new")
            self.assertEqual(session.engine, "")


class ReleaseEvidenceTests(unittest.TestCase):
    def test_strict_verification_fails_on_a_failed_engine_contract(self):
        from backend.release_verification import _validation_errors

        evidence = {
            "releaseTools": {
                "opencvDnnEngines": {
                    "ran": True,
                    "passed": False,
                    "errors": ["pp-ocrv6-det: new engine failed: boom"],
                },
            },
        }
        messages = list(_validation_errors(evidence))
        self.assertTrue(
            any("OpenCV DNN engine contract failed" in item for item in messages),
            messages,
        )

    def test_a_skipped_contract_is_not_a_strict_failure(self):
        from backend.release_verification import _validation_errors

        evidence = {
            "releaseTools": {
                "opencvDnnEngines": {"ran": False, "passed": None, "errors": ["x"]},
            },
        }
        self.assertFalse(any(
            "OpenCV DNN engine contract" in item
            for item in _validation_errors(evidence)
        ))


if __name__ == "__main__":
    unittest.main()
