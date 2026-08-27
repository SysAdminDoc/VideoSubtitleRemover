"""RM-147: requested vs. effective execution provenance is persisted."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.execution_provenance import (  # noqa: E402
    ExecutionProvenance,
    RequestedStageError,
    StageProvenance,
    device_from_provider,
    normalize_device,
)


class DeviceNormalizationTests(unittest.TestCase):
    def test_device_strings_map_to_compute_classes(self):
        self.assertEqual(normalize_device("cuda:0"), "cuda")
        self.assertEqual(normalize_device("CUDA:1"), "cuda")
        self.assertEqual(normalize_device("directml"), "directml")
        self.assertEqual(normalize_device("cpu"), "cpu")
        self.assertEqual(normalize_device(""), "unknown")

    def test_providers_map_to_compute_classes(self):
        self.assertEqual(
            device_from_provider("CUDAExecutionProvider"), "cuda")
        self.assertEqual(
            device_from_provider("DmlExecutionProvider"), "directml")
        self.assertEqual(
            device_from_provider("CPUExecutionProvider"), "cpu")
        self.assertEqual(device_from_provider("OpenVINO"), "cpu")
        self.assertEqual(device_from_provider("cv2"), "cpu")


class StageProvenanceTests(unittest.TestCase):
    def test_cpu_execution_under_a_cuda_request_is_labelled_a_fallback(self):
        stage = StageProvenance(
            stage="ocr",
            requested_device="cuda:0",
            effective_device="cpu",
            engine="RapidOCR",
            provider="CPUExecutionProvider",
            fallback_reason="RapidOCR runs on cpu in this build",
        )
        self.assertTrue(stage.fell_back)
        self.assertEqual(stage.label(), "RapidOCR on CPU (CUDA requested)")
        payload = stage.to_dict()
        self.assertEqual(payload["requestedDevice"], "cuda")
        self.assertEqual(payload["effectiveDevice"], "cpu")
        self.assertTrue(payload["fellBack"])
        self.assertIn("cpu", payload["fallbackReason"])

    def test_matching_devices_are_not_a_fallback(self):
        stage = StageProvenance(
            stage="inpaint",
            requested_device="cuda:0",
            effective_device="cuda:0",
            engine="LAMA",
        )
        self.assertFalse(stage.fell_back)
        self.assertEqual(stage.label(), "LAMA on CUDA")

    def test_initialized_stage_is_not_reported_as_succeeded(self):
        stage = StageProvenance(
            stage="ocr",
            requested_device="cpu",
            effective_device="cpu",
            engine="RapidOCR",
            requested_implementation="rapidocr",
            actual_implementation="rapidocr",
            outcome="initialized",
        )
        payload = stage.to_dict()
        self.assertEqual(payload["status"], "not_run")
        self.assertEqual(payload["outcome"], "initialized")
        self.assertEqual(stage.label(), "RapidOCR initialized, not run")

    def test_auto_scene_routes_record_mixed_execution_without_false_fallback(self):
        stage = StageProvenance(
            stage="inpaint",
            requested_device="cpu",
            effective_device="cpu",
            engine="Auto",
            requested_implementation="auto",
            selection_policy="auto",
        )
        stage.record_execution("sttn", provider="TBE", effective_device="cpu", count=4)
        stage.record_execution(
            "propainter", provider="TBE plus LaMa", effective_device="cpu", count=3
        )
        stage.fallback_chain = [
            {"implementation": "sttn", "outcome": "executed", "reason": "scene routing"},
            {
                "implementation": "propainter",
                "outcome": "executed",
                "reason": "scene routing",
            },
        ]

        payload = stage.to_dict()
        self.assertEqual(payload["actualImplementation"], "mixed")
        self.assertEqual(
            [item["executionCount"] for item in payload["actualExecutions"]],
            [4, 3],
        )
        self.assertFalse(payload["chainFellBack"])
        self.assertFalse(payload["fellBack"])


class ExecutionProvenanceTests(unittest.TestCase):
    def _cuda_request_that_ran_on_cpu(self):
        provenance = ExecutionProvenance(
            requested_device="cuda:0",
            effective_device="cuda:0",
            inpaint_mode="LAMA",
        )
        provenance.set_stage(StageProvenance(
            stage="ocr", requested_device="cuda:0", effective_device="cpu",
            engine="RapidOCR", provider="CPUExecutionProvider",
            fallback_reason="RapidOCR runs on cpu in this build"))
        provenance.set_stage(StageProvenance(
            stage="inpaint", requested_device="cuda:0",
            effective_device="cpu", engine="LAMA", backend="cv2",
            provider="cv2",
            fallback_reason="cv2 has no GPU implementation in this build"))
        provenance.frames_processed = 240
        provenance.processing_seconds = 60.0
        return provenance

    def test_a_cuda_request_running_cpu_is_visibly_labelled(self):
        provenance = self._cuda_request_that_ran_on_cpu()
        self.assertTrue(provenance.any_fallback)
        summary = provenance.summary()
        self.assertIn("RapidOCR on CPU (CUDA requested)", summary)
        self.assertIn("LAMA on CPU (CUDA requested)", summary)
        self.assertIn("4 fps", summary)
        self.assertEqual(provenance.frames_per_second, 4.0)

    def test_payload_records_every_required_field(self):
        payload = self._cuda_request_that_ran_on_cpu().to_dict()
        self.assertEqual(payload["schema"], "vsr.execution_provenance.v2")
        self.assertEqual(payload["requestedDevice"], "cuda")
        self.assertEqual(payload["inpaintMode"], "LAMA")
        self.assertTrue(payload["anyFallback"])
        self.assertEqual(payload["framesProcessed"], 240)
        self.assertEqual(payload["framesPerSecond"], 4.0)
        ocr = payload["stages"]["ocr"]
        self.assertEqual(ocr["engine"], "RapidOCR")
        self.assertEqual(ocr["provider"], "CPUExecutionProvider")
        self.assertTrue(ocr["fellBack"])
        inpaint = payload["stages"]["inpaint"]
        self.assertEqual(inpaint["backend"], "cv2")
        self.assertTrue(inpaint["fellBack"])
        # Must survive a JSON round trip for the report and sidecar.
        restored = ExecutionProvenance.from_dict(
            json.loads(json.dumps(payload)))
        self.assertEqual(restored.to_dict(), payload)

    def test_no_fallback_when_everything_ran_where_requested(self):
        provenance = ExecutionProvenance(
            requested_device="cpu", effective_device="cpu")
        provenance.set_stage(StageProvenance(
            stage="ocr", requested_device="cpu", effective_device="cpu",
            engine="RapidOCR"))
        self.assertFalse(provenance.any_fallback)
        self.assertFalse(provenance.to_dict()["anyFallback"])

    def test_throughput_is_none_without_timing(self):
        self.assertIsNone(ExecutionProvenance().frames_per_second)

    def test_failure_preserves_execution_and_deduplicates_route_prefix(self):
        provenance = ExecutionProvenance(
            requested_device="cpu", effective_device="cpu"
        )
        stage = provenance.begin_stage(
            "ocr",
            requested_implementation="auto",
            selection_policy="auto",
        )
        selected = {
            "implementation": "rapidocr",
            "outcome": "selected",
            "provider": "CPUExecutionProvider",
        }
        stage.fallback_chain = [selected]
        provenance.record_success(
            "ocr",
            implementation="rapidocr",
            provider="CPUExecutionProvider",
            effective_device="cpu",
            count=2,
        )
        error = RequestedStageError(
            stage="ocr",
            requested_implementation="auto",
            actual_implementation="rapidocr",
            provider="CPUExecutionProvider",
            failure_class="runtime_failed",
            detail="synthetic inference failure",
            recovery_hint="Repair RapidOCR or select another OCR engine.",
            fallback_chain=[
                selected,
                {
                    "implementation": "rapidocr",
                    "outcome": "runtime_failed",
                    "provider": "CPUExecutionProvider",
                    "failureClass": "runtime_failed",
                    "reason": "synthetic inference failure",
                },
            ],
        )

        failed = provenance.record_failure(error)
        payload = provenance.to_dict()
        self.assertEqual(failed.selection_policy, "auto")
        self.assertEqual(payload["stages"]["ocr"]["status"], "failed")
        self.assertEqual(payload["stages"]["ocr"]["failureClass"], "runtime_failed")
        self.assertEqual(len(payload["stages"]["ocr"]["fallbackChain"]), 2)
        self.assertEqual(
            payload["stages"]["ocr"]["actualExecutions"][0]["executionCount"],
            2,
        )
        restored = ExecutionProvenance.from_dict(json.loads(json.dumps(payload)))
        self.assertEqual(restored.to_dict(), payload)

    def test_processor_records_a_requested_stage_failure_without_false_success(self):
        from backend import processor

        remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
        remover.config = processor.ProcessingConfig(device="cpu")
        remover.execution_provenance = ExecutionProvenance(
            requested_device="cpu", effective_device="cpu"
        )
        error = RequestedStageError(
            stage="inpaint",
            requested_implementation="lama",
            actual_implementation="lama",
            provider="OpenCV DNN",
            failure_class="runtime_failed",
            detail="synthetic model failure",
            recovery_hint="Verify the LaMa model and retry.",
        )

        remover._record_requested_stage_failure(error)

        stage = remover.execution_provenance.to_dict()["stages"]["inpaint"]
        self.assertEqual(stage["status"], "failed")
        self.assertEqual(stage["provider"], "OpenCV DNN")
        self.assertEqual(stage["recoveryHint"], error.recovery_hint)
        self.assertEqual(remover.last_error_reason, "requested_stage_failed")

    def test_refresh_does_not_overwrite_a_failed_ocr_stage(self):
        from types import SimpleNamespace
        from backend import processor

        remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
        remover.config = processor.ProcessingConfig(device="cpu")
        remover.detector = SimpleNamespace(
            execution_provenance=lambda: StageProvenance(
                stage="ocr",
                requested_implementation="rapidocr",
                actual_implementation="rapidocr",
                outcome="initialized",
            )
        )
        remover.execution_provenance = ExecutionProvenance(
            requested_device="cpu", effective_device="cpu"
        )
        remover.execution_provenance.record_failure(RequestedStageError(
            stage="ocr",
            requested_implementation="rapidocr",
            actual_implementation="rapidocr",
            provider="CPUExecutionProvider",
            failure_class="runtime_failed",
            detail="synthetic OCR failure",
            recovery_hint="Repair RapidOCR.",
        ))

        remover._refresh_execution_provenance()

        stage = remover.execution_provenance.to_dict()["stages"]["ocr"]
        self.assertEqual(stage["status"], "failed")
        self.assertEqual(stage["failureClass"], "runtime_failed")


class DetectorProvenanceTests(unittest.TestCase):
    def test_detector_reports_a_concrete_stage_record(self):
        from backend.detection import SubtitleDetector

        detector = SubtitleDetector.__new__(SubtitleDetector)
        detector.device = "cuda:0"
        detector._engine_name = "OpenCV fallback"
        detector._provider_name = "cv2"
        detector._effective_device = "cpu"
        detector._provenance_reason = "OpenCV fallback runs on cpu"
        stage = detector.execution_provenance()
        self.assertEqual(stage.stage, "ocr")
        self.assertTrue(stage.fell_back)
        self.assertEqual(stage.to_dict()["effectiveDevice"], "cpu")

    def test_finalize_infers_cpu_for_opencv_fallback(self):
        from backend.detection import SubtitleDetector

        detector = SubtitleDetector.__new__(SubtitleDetector)
        detector.device = "cuda:0"
        detector._engine_name = "OpenCV fallback"
        detector._provider_name = "cv2"
        detector._effective_device = ""
        detector._provenance_reason = ""
        detector._finalize_provenance()
        self.assertEqual(detector._effective_device, "cpu")
        self.assertIn("cpu", detector._provenance_reason)


class InpainterBackendNameTests(unittest.TestCase):
    def test_every_builtin_inpainter_reports_a_backend(self):
        from backend.config import ProcessingConfig
        from backend.inpainters.sttn import STTNInpainter

        config = ProcessingConfig()
        sttn = STTNInpainter("cpu", config)
        self.assertIn("TBE", sttn.backend_name)
        config.tbe_enable = False
        self.assertEqual(STTNInpainter("cpu", config).backend_name, "cv2")

    def test_base_inpainter_default_is_the_class_name(self):
        import numpy as np

        from backend.inpainters._common import BaseInpainter

        class _Toy(BaseInpainter):
            def inpaint(self, frames, masks):
                return [np.zeros((2, 2, 3), dtype=np.uint8)]

        self.assertEqual(_Toy().backend_name, "_Toy")


class SidecarAndReportTests(unittest.TestCase):
    def _provenance(self):
        provenance = ExecutionProvenance(
            requested_device="cuda:0", effective_device="cuda:0",
            inpaint_mode="LAMA")
        provenance.set_stage(StageProvenance(
            stage="ocr", requested_device="cuda:0", effective_device="cpu",
            engine="RapidOCR", provider="CPUExecutionProvider"))
        return provenance.to_dict()

    def test_sidecar_carries_execution_provenance(self):
        from backend.batch_report import build_output_sidecar
        from backend.config import ProcessingConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "in.mp4"
            source.write_bytes(b"video")
            payload = build_output_sidecar(
                input_path=str(source),
                output_path=str(Path(tmpdir) / "out.mp4"),
                config=ProcessingConfig(),
                status="processed",
                execution_provenance=self._provenance(),
            )
        self.assertIn("executionProvenance", payload)
        self.assertTrue(payload["executionProvenance"]["anyFallback"])
        self.assertEqual(payload["engine"], "RapidOCR")

    def test_sidecar_engine_is_unrecorded_without_provenance(self):
        from backend.batch_report import build_output_sidecar
        from backend.config import ProcessingConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "in.mp4"
            source.write_bytes(b"video")
            payload = build_output_sidecar(
                input_path=str(source),
                output_path=str(Path(tmpdir) / "out.mp4"),
                config=ProcessingConfig(),
                status="processed",
            )
        self.assertEqual(payload["engine"], "unrecorded")
        self.assertNotIn("executionProvenance", payload)

    def test_batch_record_carries_execution_provenance(self):
        from backend.batch_report import finish_batch_item

        record = {"status": "", "stage_timings": {}, "detection_stats": {}}
        finish_batch_item(
            record, "completed", execution_provenance=self._provenance())
        self.assertTrue(record["execution_provenance"]["anyFallback"])
        self.assertEqual(
            record["execution_provenance"]["stages"]["ocr"]["engine"],
            "RapidOCR",
        )


class QueueSurfaceTests(unittest.TestCase):
    def test_queue_row_labels_a_fallback_run(self):
        from gui.config import (
            InpaintMode, ProcessingConfig, ProcessingStatus, QueueItem,
        )
        from gui.utils import _queue_item_execution_text

        provenance = ExecutionProvenance(
            requested_device="cuda:0", effective_device="cuda:0",
            inpaint_mode="LAMA")
        provenance.set_stage(StageProvenance(
            stage="ocr", requested_device="cuda:0", effective_device="cpu",
            engine="RapidOCR", provider="CPUExecutionProvider"))
        provenance.set_stage(StageProvenance(
            stage="inpaint", requested_device="cuda:0",
            effective_device="cpu", engine="LAMA", backend="cv2"))
        item = QueueItem(
            id="1", file_path="a.mp4", output_path="b.mp4",
            config=ProcessingConfig(mode=InpaintMode.LAMA),
            status=ProcessingStatus.COMPLETE,
            execution_provenance=provenance.to_dict(),
        )
        text = _queue_item_execution_text(item)
        self.assertIn("RapidOCR on CPU (CUDA requested)", text)
        self.assertIn("LAMA on CPU (CUDA requested)", text)
        self.assertIn("fallback from CUDA", text)

    def test_queue_row_is_quiet_without_provenance(self):
        from gui.config import ProcessingConfig, QueueItem
        from gui.utils import _queue_item_execution_text

        item = QueueItem(
            id="1", file_path="a.mp4", output_path="b.mp4",
            config=ProcessingConfig())
        self.assertEqual(_queue_item_execution_text(item), "")

    def test_queue_state_round_trips_provenance(self):
        from unittest import mock

        from gui import config as gcfg

        provenance = ExecutionProvenance(
            requested_device="cuda:0", effective_device="cpu").to_dict()
        item = gcfg.QueueItem(
            id="1", file_path="a.mp4", output_path="b.mp4",
            config=gcfg.ProcessingConfig(),
            execution_provenance=provenance,
        )
        captured = {}
        with mock.patch.object(
            gcfg, "_write_json_atomic",
            side_effect=lambda path, payload: captured.update(payload),
        ):
            gcfg.save_queue_state([item])
        self.assertEqual(
            captured["items"][0]["execution_provenance"], provenance)


if __name__ == "__main__":
    unittest.main()
