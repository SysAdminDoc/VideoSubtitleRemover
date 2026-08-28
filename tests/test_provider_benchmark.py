"""RM-315: the provider benchmark must measure, not estimate.

Users could not predict throughput, memory, or fidelity before choosing a
build, and the package's other benchmarks score removal quality rather than
what it costs to get. These tests cover the measurement itself and the
committed evidence, so a number in the README always traces to a run.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend.provider_benchmark import (
    PROVIDER_BENCHMARK_SCHEMA,
    GpuMemoryWatcher,
    config_digest,
    gpu_host_facts,
    manifest_config_for,
    peak_working_set_bytes,
    run_provider_benchmark,
)

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = ROOT / "docs" / "benchmarks"
CLIP = ROOT / "tests" / "clips" / "static_dialogue.mkv"


def _have_ffmpeg() -> bool:
    return bool(shutil.which("ffmpeg"))


class MeasurementTests(unittest.TestCase):
    def test_peak_memory_is_a_real_number_on_this_host(self):
        peak = peak_working_set_bytes()
        self.assertIsNotNone(peak)
        # A Python process that has imported numpy and OpenCV is not 1 MiB.
        self.assertGreater(peak, 8 * 1024 * 1024)

    def test_the_gpu_watcher_says_so_when_it_cannot_measure(self):
        with mock.patch("backend.provider_benchmark._nvidia_smi",
                        return_value=[]):
            with GpuMemoryWatcher() as watcher:
                pass
            evidence = watcher.evidence()
        self.assertFalse(evidence["measured"])
        self.assertEqual(evidence["scope"], "unavailable")
        self.assertNotIn("peakMiB", evidence)

    def test_the_gpu_watcher_labels_its_reading_device_wide(self):
        readings = ["1000", "1000", "1400", "1200"]
        index = {"n": 0}

        def _fake(query):
            if "memory.used" not in query:
                return ["NVIDIA Test, 999.99, 12282"]
            value = readings[min(index["n"], len(readings) - 1)]
            index["n"] += 1
            return [value]

        with mock.patch("backend.provider_benchmark._nvidia_smi",
                        side_effect=_fake):
            watcher = GpuMemoryWatcher(interval=0.01)
            with watcher:
                import time

                time.sleep(0.08)
            evidence = watcher.evidence()
        self.assertTrue(evidence["measured"])
        # Per-process VRAM is [N/A] on consumer WDDM, so claiming it would
        # be a number the host cannot actually produce.
        self.assertEqual(evidence["scope"], "device-wide")
        # The baseline is a reading taken inside the window, so the peak can
        # never be below it.
        self.assertGreaterEqual(evidence["peakMiB"], evidence["baselineMiB"])
        self.assertGreaterEqual(evidence["deltaMiB"], 0)

    def test_gpu_facts_report_absence_rather_than_guessing(self):
        with mock.patch("backend.provider_benchmark._nvidia_smi",
                        return_value=[]):
            facts = gpu_host_facts()
        self.assertFalse(facts["present"])
        self.assertEqual(facts["name"], "")
        self.assertIsNone(facts["memoryTotalMiB"])

    def test_the_config_hash_moves_with_the_config(self):
        from backend.config import ProcessingConfig

        first = ProcessingConfig()
        second = ProcessingConfig()
        self.assertEqual(config_digest(first), config_digest(second))
        second.mask_dilate_px = int(first.mask_dilate_px) + 3
        self.assertNotEqual(config_digest(first), config_digest(second))

    def test_the_manifest_config_is_used_rather_than_an_invented_one(self):
        config = manifest_config_for(CLIP)
        self.assertTrue(config)
        # Benchmarking under a config the clip does not match measures the
        # error path: static_dialogue is a fixed-region STTN clip.
        self.assertEqual(config["mode"], "sttn")
        self.assertIn("subtitle_area", config)
        self.assertEqual(manifest_config_for("not-in-the-manifest.mkv"), {})


class BenchmarkRunTests(unittest.TestCase):
    @unittest.skipUnless(_have_ffmpeg(), "ffmpeg not on PATH")
    def test_a_cpu_run_records_every_field_the_evidence_promises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            evidence = run_provider_benchmark(
                CLIP, device="cpu", profile="cpu",
                output_dir=tmpdir, warm_runs=1,
            )

        self.assertEqual(evidence["schema"], PROVIDER_BENCHMARK_SCHEMA)
        self.assertTrue(evidence["passed"], evidence["errors"])

        self.assertEqual(len(evidence["input"]["sha256"]), 64)
        self.assertEqual(len(evidence["config"]["sha256"]), 64)
        self.assertNotIn(":", evidence["input"]["path"],
                         "committed evidence must not carry absolute paths")

        labels = [run["label"] for run in evidence["runs"]]
        self.assertEqual(labels, ["cold", "warm1"])
        for run in evidence["runs"]:
            self.assertTrue(run["ok"])
            self.assertGreater(run["seconds"], 0.0)
            self.assertGreater(run["frames"], 0)
            self.assertGreater(run["fps"], 0.0)

        self.assertIsNotNone(evidence["memory"]["peakWorkingSetBytes"])
        self.assertIn("scope", evidence["memory"]["gpu"])

        # Exact timing: the runs have to have done the same work.
        self.assertTrue(evidence["timing"]["identicalOutputs"])
        self.assertEqual(len(evidence["timing"]["outputDigests"]), 1)

        self.assertIn("status", evidence["quality"]["gate"])
        self.assertEqual(
            evidence["runtime"]["activeProviders"][:1],
            ["CPUExecutionProvider"],
        )
        self.assertTrue(evidence["runtime"]["providerVerified"])

    @unittest.skipUnless(_have_ffmpeg(), "ffmpeg not on PATH")
    def test_diverging_runs_are_reported_rather_than_averaged(self):
        """Runs that produced different output are not timing the same work.

        Averaging them would hand out an FPS number for a job nobody ran, so
        the evidence has to fail instead.
        """
        digests = iter([
            {"sha256": "a" * 64, "frame_count": 16, "width": 160, "height": 96},
            {"sha256": "b" * 64, "frame_count": 16, "width": 160, "height": 96},
        ])

        with mock.patch(
            "backend.reference_corpus.decoded_frame_digest",
            side_effect=lambda *_a, **_k: next(digests),
        ):
            with tempfile.TemporaryDirectory() as tmpdir:
                evidence = run_provider_benchmark(
                    CLIP, device="cpu", profile="cpu",
                    output_dir=tmpdir, warm_runs=1,
                )

        self.assertFalse(evidence["timing"]["identicalOutputs"])
        self.assertEqual(len(evidence["timing"]["outputDigests"]), 2)
        self.assertFalse(evidence["passed"])
        self.assertTrue(
            any("not comparable" in error for error in evidence["errors"]),
            evidence["errors"],
        )

    def test_a_failed_run_marks_the_evidence_failed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "absent.mkv"
            missing.write_bytes(b"not a video")
            evidence = run_provider_benchmark(
                missing, device="cpu", profile="cpu",
                output_dir=tmpdir, warm_runs=0,
            )
        self.assertFalse(evidence["passed"])
        self.assertTrue(evidence["errors"])


class CommittedEvidenceTests(unittest.TestCase):
    """The evidence in the repo has to be real, current, and comparable."""

    def _load(self, name):
        path = EVIDENCE_DIR / name
        self.assertTrue(path.is_file(), path)
        return json.loads(path.read_text(encoding="utf-8"))

    def test_both_lanes_have_committed_evidence(self):
        for name in ("provider-benchmark-cpu.json",
                     "provider-benchmark-nvidia.json"):
            with self.subTest(name=name):
                payload = self._load(name)
                self.assertEqual(
                    payload["schema"], PROVIDER_BENCHMARK_SCHEMA)
                self.assertTrue(payload["passed"], payload["errors"])
                self.assertTrue(payload["timing"]["identicalOutputs"])
                self.assertIsNotNone(
                    payload["memory"]["peakWorkingSetBytes"])
                self.assertIn("status", payload["quality"]["gate"])

    def test_the_two_lanes_measured_the_same_input_and_config(self):
        cpu = self._load("provider-benchmark-cpu.json")
        gpu = self._load("provider-benchmark-nvidia.json")
        self.assertEqual(cpu["input"]["sha256"], gpu["input"]["sha256"])
        self.assertEqual(cpu["input"]["path"], gpu["input"]["path"])
        # The device differs on purpose; nothing else about the job may.
        self.assertEqual(cpu["config"]["device"], "cpu")
        self.assertEqual(gpu["config"]["device"], "cuda:0")
        cpu_overrides = dict(cpu["config"]["overrides"])
        gpu_overrides = dict(gpu["config"]["overrides"])
        cpu_overrides.pop("device", None)
        gpu_overrides.pop("device", None)
        self.assertEqual(cpu_overrides, gpu_overrides)

    def test_the_nvidia_evidence_really_ran_on_cuda(self):
        gpu = self._load("provider-benchmark-nvidia.json")
        self.assertEqual(
            gpu["runtime"]["activeProviders"][:1], ["CUDAExecutionProvider"])
        self.assertTrue(gpu["runtime"]["providerVerified"])
        self.assertTrue(gpu["host"]["gpu"]["present"])
        self.assertTrue(gpu["host"]["gpu"]["name"])
        self.assertTrue(gpu["host"]["gpu"]["driver"])

    def test_the_cpu_evidence_claims_no_gpu_work(self):
        cpu = self._load("provider-benchmark-cpu.json")
        self.assertEqual(
            cpu["runtime"]["activeProviders"][:1], ["CPUExecutionProvider"])

    def test_the_evidence_carries_no_absolute_paths(self):
        for name in ("provider-benchmark-cpu.json",
                     "provider-benchmark-nvidia.json"):
            with self.subTest(name=name):
                payload = self._load(name)
                self.assertNotIn(":", payload["input"]["path"])
                self.assertFalse(payload["input"]["path"].startswith("/"))


class BenchmarkCliTests(unittest.TestCase):
    @unittest.skipUnless(_have_ffmpeg(), "ffmpeg not on PATH")
    def test_the_command_writes_machine_readable_json(self):
        import sys

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "evidence.json"
            result = subprocess.run(
                [sys.executable, "-m", "backend.provider_benchmark",
                 str(CLIP), "--device", "cpu", "--profile", "cpu",
                 "--warm-runs", "0", "--output", str(out)],
                cwd=ROOT, capture_output=True, text=True, timeout=900,
            )
            self.assertEqual(result.returncode, 0, result.stderr[-800:])
            payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema"], PROVIDER_BENCHMARK_SCHEMA)
        self.assertEqual(len(payload["runs"]), 1)


if __name__ == "__main__":
    unittest.main()
