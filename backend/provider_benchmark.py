"""RM-315: measured, machine-readable evidence for one provider lane.

Users cannot predict throughput, memory, or output fidelity before they pick
a build, and the project had no measurement to give them: the benchmarks in
this package score removal quality, not the cost of getting it. This module
runs one reference clip end to end on a named device and records what it
cost, so a CPU bundle and a CUDA bundle can be compared on numbers taken the
same way rather than on adjectives.

Everything here is measured on the host that runs it. Nothing is inferred
from hardware names, and the fields that cannot be measured on a given host
say so rather than reporting zero.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import platform
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

PROVIDER_BENCHMARK_SCHEMA = "vsr.provider_benchmark.v1"

# Sampling interval for the device-wide VRAM watcher. Short enough to catch a
# transient allocation in a few-second run, long enough not to distort it.
_GPU_SAMPLE_SECONDS = 0.25


class _ProcessMemoryCounters(ctypes.Structure):
    """PROCESS_MEMORY_COUNTERS. Declared once at module scope on purpose.

    Building a ctypes Structure inside a repeatedly called function leaks a
    new type per call and is measurably slow.
    """

    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def peak_working_set_bytes() -> Optional[int]:
    """This process's peak resident memory, or None where unavailable.

    Windows keeps the peak for us, so no sampling thread is needed and the
    number cannot miss a spike between samples.
    """
    if platform.system() == "Windows":
        try:
            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
            kernel32 = ctypes.windll.kernel32
            # The pseudo-handle is 0xFFFFFFFFFFFFFFFF, which ctypes truncates
            # to a 32-bit int without an explicit restype, and the call then
            # fails silently and reports no memory at all.
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            psapi = ctypes.windll.psapi
            psapi.GetProcessMemoryInfo.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(_ProcessMemoryCounters),
                ctypes.c_ulong,
            ]
            psapi.GetProcessMemoryInfo.restype = ctypes.c_int
            ok = psapi.GetProcessMemoryInfo(
                kernel32.GetCurrentProcess(), ctypes.byref(counters),
                counters.cb)
            return int(counters.PeakWorkingSetSize) if ok else None
        except Exception:
            return None
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kibibytes, macOS reports bytes.
        return int(peak) * (1024 if platform.system() == "Linux" else 1)
    except Exception:
        return None


def _nvidia_smi(query: str, *, timeout: float = 20.0) -> list:
    if not shutil.which("nvidia-smi"):
        return []
    from backend.subprocess_policy import run_process

    try:
        result = run_process(
            ["nvidia-smi", f"--query-gpu={query}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def gpu_host_facts() -> dict:
    """Name the GPU and driver, or say plainly that there is none."""
    rows = _nvidia_smi("name,driver_version,memory.total")
    if not rows:
        return {"present": False, "name": "", "driver": "", "memoryTotalMiB": None}
    parts = [item.strip() for item in rows[0].split(",")]
    while len(parts) < 3:
        parts.append("")
    try:
        total = int(float(parts[2]))
    except ValueError:
        total = None
    return {
        "present": True,
        "name": parts[0],
        "driver": parts[1],
        "memoryTotalMiB": total,
    }


class GpuMemoryWatcher:
    """Sample device-wide VRAM while a run is in flight.

    Consumer GeForce cards under WDDM report `[N/A]` for per-process VRAM, so
    this is deliberately device-wide and labelled as such. It is only
    meaningful on an otherwise idle card, which the evidence records.
    """

    def __init__(self, interval: float = _GPU_SAMPLE_SECONDS):
        self.interval = interval
        self.samples: list[int] = []
        self.baseline: Optional[int] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.available = bool(_nvidia_smi("memory.used"))

    def _read(self) -> Optional[int]:
        # A short timeout on the sampling path: the watcher must not be able
        # to outlive the block it is sampling, and a hung probe is a missing
        # sample rather than a reason to hold the run open.
        rows = _nvidia_smi("memory.used", timeout=5.0)
        if not rows:
            return None
        try:
            return int(float(rows[0]))
        except ValueError:
            return None

    def __enter__(self) -> "GpuMemoryWatcher":
        if not self.available:
            return self
        self.baseline = self._read()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            value = self._read()
            if value is not None:
                self.samples.append(value)

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            # Longer than the sampling timeout above, so the thread is
            # always finished before evidence() reads its samples.
            self._thread.join(timeout=10)
            if self._thread.is_alive():  # pragma: no cover - hung driver
                logger.warning(
                    "The GPU memory watcher did not stop; its last sample "
                    "may be missing from the evidence")
        self._thread = None

    def evidence(self) -> dict:
        if not self.available:
            return {
                "measured": False,
                "scope": "unavailable",
                "reason": "nvidia-smi did not report device memory",
            }
        readings = list(self.samples)
        if self.baseline is not None:
            readings.append(self.baseline)
        peak = max(readings) if readings else None
        delta = None
        if peak is not None and self.baseline is not None:
            delta = max(0, peak - self.baseline)
        return {
            "measured": True,
            # Named honestly: this is the whole card, not this process.
            # Per-process VRAM is `[N/A]` on consumer WDDM drivers.
            "scope": "device-wide",
            "baselineMiB": self.baseline,
            "peakMiB": peak,
            "deltaMiB": delta,
            "sampleCount": len(self.samples),
            "sampleIntervalSeconds": self.interval,
        }


_ABSOLUTE_PATH_MARKERS = ("\\", "/")


def _looks_like_a_path(value: str) -> bool:
    """Whether a recorded string names a place on this machine."""
    text = str(value)
    if len(text) < 4:
        return False
    if text[1:3] == ":\\" or text[1:3] == ":/":
        return True
    return text.startswith("/") and any(
        marker in text for marker in _ABSOLUTE_PATH_MARKERS)


def scrub_local_paths(value):
    """Replace absolute paths in recorded evidence with a marker.

    The quality report carries overlay and sheet paths under the run's
    temporary directory. Committed evidence must not name one machine, and
    those paths point at a directory that no longer exists by the time
    anyone reads it, so the fact that a file was produced is kept and the
    location is not.
    """
    if isinstance(value, dict):
        return {key: scrub_local_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [scrub_local_paths(item) for item in value]
    if isinstance(value, str) and _looks_like_a_path(value):
        return "<local path removed>"
    return value


def _repo_relative(path: Path) -> str:
    """Return `path` relative to the repository when it lives inside it."""
    root = Path(__file__).resolve().parent.parent
    try:
        return Path(path).resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_digest(config: Any) -> str:
    """A stable hash of the exact configuration a run used."""
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
    elif isinstance(config, Mapping):
        payload = dict(config)
    else:
        payload = {
            key: value
            for key, value in vars(config or object()).items()
            if not key.startswith("_")
        }
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def host_facts() -> dict:
    import os

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cpuCount": os.cpu_count(),
        "gpu": gpu_host_facts(),
    }


def _runtime_provider_facts(device: str) -> dict:
    facts = {
        "requestedDevice": device,
        "onnxruntime": "",
        "availableProviders": [],
        "activeProviders": [],
        "providerVerified": None,
        "error": "",
    }
    try:
        import onnxruntime as ort  # type: ignore
    except Exception as exc:
        facts["error"] = f"onnxruntime is not importable: {exc}"
        return facts
    facts["onnxruntime"] = getattr(ort, "__version__", "")
    try:
        facts["availableProviders"] = list(ort.get_available_providers())
    except Exception as exc:
        facts["error"] = str(exc)
        return facts

    from backend.inpainters_onnx import _providers_for_device
    from backend.onnx_model_info import _tiny_identity_onnx_bytes
    from backend.onnxruntime_cuda import preload_onnxruntime_cuda_dlls_if_needed

    providers = _providers_for_device(device)
    preload_onnxruntime_cuda_dlls_if_needed(ort, providers)
    try:
        session = ort.InferenceSession(
            _tiny_identity_onnx_bytes(), providers=providers)
        facts["activeProviders"] = list(session.get_providers())
    except Exception as exc:
        facts["error"] = str(exc)
        return facts
    from backend.device_provider import (
        ProviderFellBackError,
        named_accelerator_provider,
        verify_active_provider,
    )

    if not named_accelerator_provider(device):
        facts["providerVerified"] = True
        return facts
    try:
        verify_active_provider(device, facts["activeProviders"])
        facts["providerVerified"] = True
    except ProviderFellBackError as exc:
        facts["providerVerified"] = False
        facts["error"] = str(exc)
    return facts


def manifest_config_for(clip_path: Path | str,
                        manifest_path: Path | str | None = None) -> dict:
    """Return the reviewed config the corpus uses for this clip, if listed.

    Benchmarking a clip under a config that does not match it measures the
    error path rather than the pipeline, so prefer the manifest's own entry
    over an invented one.
    """
    from backend.reference_corpus import DEFAULT_MANIFEST

    path = Path(manifest_path or DEFAULT_MANIFEST)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    name = Path(clip_path).name
    for entry in manifest.get("clips", []):
        if isinstance(entry, Mapping) and entry.get("filename") == name:
            config = entry.get("config")
            return dict(config) if isinstance(config, Mapping) else {}
    return {}


def run_provider_benchmark(
    clip_path: Path | str,
    *,
    device: str = "cpu",
    profile: str = "cpu",
    config_overrides: Optional[Mapping[str, object]] = None,
    output_dir: Path | str,
    warm_runs: int = 1,
    manifest_path: Path | str | None = None,
) -> dict:
    """Process one clip and record what the run cost and what it produced."""
    from backend import processor
    from backend.quality_gate import evaluate_quality_gate
    from backend.reference_corpus import (
        _apply_config_overrides,
        decoded_frame_digest,
    )

    source = Path(clip_path)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_overrides = manifest_config_for(source, manifest_path)
    overrides = dict(manifest_overrides)
    overrides.update(dict(config_overrides or {}))
    overrides["device"] = device
    # The timed runs measure removal. The quality report samples frames,
    # re-encodes overlays, and re-runs the detector over the repaired
    # region, which on a sixteen-frame fixture costs several times the
    # removal itself; folding that into the FPS would report a number for
    # work the product does not do on a normal run. One extra untimed run
    # produces the gate verdict.
    timed_overrides = dict(overrides)
    timed_overrides["quality_report"] = False
    timed_overrides["quality_report_sheet"] = False
    quality_overrides = dict(overrides)
    quality_overrides["quality_report"] = True
    config = _apply_config_overrides(timed_overrides)
    quality_config = _apply_config_overrides(quality_overrides)

    evidence: dict = {
        "schema": PROVIDER_BENCHMARK_SCHEMA,
        "profile": profile,
        "host": host_facts(),
        "runtime": _runtime_provider_facts(device),
        "input": {
            # Repo-relative where possible: committed evidence must not
            # carry one machine's absolute paths.
            "path": _repo_relative(source),
            "sha256": _sha256_file(source),
            "bytes": source.stat().st_size,
        },
        "config": {
            "device": device,
            "sha256": config_digest(config),
            "overrides": {
                key: timed_overrides[key] for key in sorted(timed_overrides)},
            "qualityOverrides": {
                key: quality_overrides[key]
                for key in sorted(quality_overrides)
            },
            # Without this, "the manifest had no entry for this clip so a
            # bare default config was benchmarked" is indistinguishable from
            # "the reviewed config was used", which is the one thing
            # manifest_config_for exists to guarantee.
            "fromManifest": bool(manifest_overrides),
        },
        "runs": [],
        "memory": {},
        "timing": {},
        "quality": {},
        "errors": [],
    }

    digests: list[str] = []
    with GpuMemoryWatcher() as watcher:
        for index in range(1 + max(0, int(warm_runs))):
            label = "cold" if index == 0 else f"warm{index}"
            output = out_root / f"{source.stem}_{profile}_{label}.mp4"
            remover = processor.SubtitleRemover(config)
            started = time.perf_counter()
            ok = remover.process_video(str(source), str(output))
            elapsed = time.perf_counter() - started
            actual = Path(remover.last_output_path or output)
            digest = decoded_frame_digest(actual) if ok else None
            if digest:
                digests.append(digest["sha256"])
            frames = int((digest or {}).get("frame_count") or 0)
            evidence["runs"].append({
                "label": label,
                "ok": bool(ok),
                "seconds": round(elapsed, 6),
                "frames": frames,
                "fps": round(frames / elapsed, 4) if elapsed > 0 else None,
                "outputFrames": digest,
                "error": str(remover.last_error_message or ""),
            })
            if not ok:
                evidence["errors"].append(
                    f"{label} run failed: {remover.last_error_message}")
                break

    if not evidence["errors"]:
        # Untimed, and after the watcher has stopped: this run exists to
        # produce the quality-gate verdict, not to contribute a number to
        # the timings above.
        quality_output = out_root / f"{source.stem}_{profile}_quality.mp4"
        quality_remover = processor.SubtitleRemover(quality_config)
        quality_ok = quality_remover.process_video(
            str(source), str(quality_output))
        metrics = dict(quality_remover.last_quality_report or {})
        gate = evaluate_quality_gate(dict(metrics))
        previews = gate.pop("previewFramePaths", []) or []
        gate["previewFrameCount"] = len(previews)
        evidence["quality"] = {
            "ran": bool(quality_ok),
            "timed": False,
            "metrics": scrub_local_paths(metrics),
            "gate": scrub_local_paths(gate),
        }
        if not quality_ok:
            evidence["errors"].append(
                "the quality run failed: "
                f"{quality_remover.last_error_message}")

    evidence["memory"] = {
        "peakWorkingSetBytes": peak_working_set_bytes(),
        "gpu": watcher.evidence(),
    }
    evidence["timingScope"] = (
        "removal only; the quality report and its re-detection pass run "
        "separately and are not in the timings"
    )
    # Exact timing: every run has to produce the same frames, or the numbers
    # above are timing different amounts of work.
    evidence["timing"] = {
        "runCount": len(evidence["runs"]),
        "identicalOutputs": bool(digests) and len(set(digests)) == 1,
        "outputDigests": sorted(set(digests)),
    }
    if evidence["runs"] and not evidence["timing"]["identicalOutputs"]:
        evidence["errors"].append(
            "runs produced different output, so the timings are not "
            "comparable"
        )
    evidence["passed"] = not evidence["errors"]
    return evidence


def main(argv: Optional[list] = None) -> int:
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(
        description="Measure one provider lane on one clip")
    parser.add_argument("clip", help="path to the input clip")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--profile", default="cpu")
    parser.add_argument("--warm-runs", type=int, default=1)
    parser.add_argument(
        "--manifest", default="",
        help="clip manifest to take the reviewed config from")
    parser.add_argument("--output", default="", help="write the JSON here")
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="vsr-provider-bench-") as tmpdir:
        evidence = run_provider_benchmark(
            args.clip,
            device=args.device,
            profile=args.profile,
            output_dir=tmpdir,
            warm_runs=args.warm_runs,
            manifest_path=args.manifest or None,
        )
    payload = json.dumps(evidence, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    import sys

    sys.exit(main())
