"""RM-153: freeze an approved matte as a reusable queue input."""

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend import processor
from backend.frozen_matte import (
    FROZEN_MATTE_SCHEMA,
    SOURCE_DIGEST_SCHEME,
    FrozenMatteError,
    default_manifest_for_output,
    freeze_matte,
    frozen_matte_summary,
    normalize_frozen_matte,
    source_fingerprint,
    timing_digest,
    validate_frozen_matte,
)
from backend.matte_interchange import MASK_INTERCHANGE_SCHEMA


def _write_manifest(root: Path, *, frames=3, width=64, height=48,
                    start=0, end=3, timestamps=None, durations=None,
                    is_vfr=False, time_base=0.1) -> tuple[Path, Path]:
    """Write a PNG matte sequence plus a matching v1 manifest."""
    artifact = root / "clip.mask"
    artifact.mkdir(parents=True, exist_ok=True)
    for index in range(frames):
        cv2.imwrite(
            str(artifact / f"frame_{index:08d}.png"),
            np.full((height, width), 10 * (index + 1), dtype=np.uint8),
        )
    stamps = timestamps if timestamps is not None else [
        round(index * time_base, 9) for index in range(frames)]
    spans = durations if durations is not None else [time_base] * frames
    from backend.matte_interchange import _sha256_sequence

    payload = {
        "schema": MASK_INTERCHANGE_SCHEMA,
        "format": "png",
        "artifact": artifact.name,
        "artifact_sha256": _sha256_sequence(artifact, frames),
        "pixel_format": "gray8",
        "width": width,
        "height": height,
        "frame_count": frames,
        "source_start_frame": start,
        "source_end_frame": end,
        "source_is_vfr": is_vfr,
        "source_time_base_seconds": time_base,
        "timestamps_seconds": stamps,
        "durations_seconds": spans,
    }
    manifest = root / "clip.mask.json"
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest, artifact


def _write_source(path: Path, *, size: int = 4096, fill: bytes = b"\x11") -> Path:
    path.write_bytes(fill * size)
    return path


class SourceFingerprintTests(unittest.TestCase):
    def test_a_fingerprint_is_stable_for_unchanged_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = _write_source(Path(tmp) / "a.bin")
            first = source_fingerprint(source)
            second = source_fingerprint(source)
            self.assertEqual(first["digest"], second["digest"])
            self.assertEqual(first["digest_scheme"], SOURCE_DIGEST_SCHEME)
            self.assertEqual(first["size_bytes"], 4096)

    def test_changed_content_at_the_same_length_changes_the_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "a.bin"
            _write_source(source, fill=b"\x11")
            before = source_fingerprint(source)["digest"]
            _write_source(source, fill=b"\x22")
            self.assertNotEqual(source_fingerprint(source)["digest"], before)

    def test_a_tail_edit_of_a_large_file_is_detected(self):
        # The head/tail sampling exists so multi-gigabyte sources stay
        # cheap; a truncation or re-mux that only touches the tail must
        # still be caught.
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "big.bin"
            span = 8 * 1024 * 1024
            source.write_bytes(b"\x00" * (span * 2 + 1024))
            before = source_fingerprint(source)["digest"]
            data = bytearray(source.read_bytes())
            data[-1] = 0xFF
            source.write_bytes(bytes(data))
            self.assertNotEqual(source_fingerprint(source)["digest"], before)

    def test_a_missing_source_fails_closed(self):
        with self.assertRaises(FrozenMatteError) as ctx:
            source_fingerprint(Path("does") / "not" / "exist.mp4")
        self.assertEqual(ctx.exception.reason, "source_missing")

    def test_timing_digest_notices_a_single_shifted_frame(self):
        base = timing_digest([0.0, 0.1, 0.2], [0.1, 0.1, 0.1])
        self.assertEqual(base, timing_digest([0.0, 0.1, 0.2], [0.1, 0.1, 0.1]))
        self.assertNotEqual(
            base, timing_digest([0.0, 0.1, 0.3], [0.1, 0.1, 0.1]))
        self.assertNotEqual(
            base, timing_digest([0.0, 0.1, 0.2], [0.1, 0.1, 0.2]))
        # Timestamps and durations must not be interchangeable.
        self.assertNotEqual(
            timing_digest([0.1], [0.2]), timing_digest([0.2], [0.1]))


class FreezeTests(unittest.TestCase):
    def test_a_freeze_captures_hashes_geometry_timing_and_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, _artifact = _write_manifest(root)
            source = _write_source(root / "clip.mp4")
            record = freeze_matte(manifest, source)

            self.assertEqual(record["schema"], FROZEN_MATTE_SCHEMA)
            self.assertEqual(record["format"], "png")
            self.assertEqual((record["width"], record["height"]), (64, 48))
            self.assertEqual(record["frame_count"], 3)
            self.assertEqual(record["source_start_frame"], 0)
            self.assertEqual(record["source_end_frame"], 3)
            self.assertTrue(record["artifact_sha256"])
            self.assertTrue(record["manifest_sha256"])
            self.assertTrue(record["timing_sha256"])
            self.assertEqual(
                record["source"]["digest_scheme"], SOURCE_DIGEST_SCHEME)
            self.assertIn("64x48", frozen_matte_summary(record))

    def test_freezing_a_matte_edited_after_export_is_refused(self):
        # The manifest's own hash would already contradict the pixels on
        # disk, so freezing the pair would bake in a lie.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, artifact = _write_manifest(root)
            source = _write_source(root / "clip.mp4")
            cv2.imwrite(
                str(artifact / "frame_00000000.png"),
                np.full((48, 64), 200, dtype=np.uint8),
            )
            with self.assertRaises(FrozenMatteError) as ctx:
                freeze_matte(manifest, source)
            self.assertEqual(
                ctx.exception.reason, "artifact_edited_since_export")

    def test_freezing_without_the_artifact_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, artifact = _write_manifest(root)
            source = _write_source(root / "clip.mp4")
            shutil.rmtree(artifact)
            with self.assertRaises(FrozenMatteError) as ctx:
                freeze_matte(manifest, source)
            self.assertEqual(ctx.exception.reason, "artifact_missing")

    def test_freezing_an_unparsable_manifest_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "broken.mask.json"
            manifest.write_text("{not json", encoding="utf-8")
            with self.assertRaises(FrozenMatteError) as ctx:
                freeze_matte(manifest, _write_source(root / "clip.mp4"))
            self.assertEqual(ctx.exception.reason, "manifest_invalid")

    def test_default_manifest_lookup_only_returns_existing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertIsNone(default_manifest_for_output(root / "out.mp4"))
            (root / "out.mask.json").write_text("{}", encoding="utf-8")
            found = default_manifest_for_output(root / "out.mp4")
            self.assertIsNotNone(found)
            self.assertEqual(found.name, "out.mask.json")


class NormalizationTests(unittest.TestCase):
    def test_a_valid_record_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, _artifact = _write_manifest(root)
            record = freeze_matte(manifest, _write_source(root / "clip.mp4"))
            self.assertEqual(normalize_frozen_matte(record), record)
            # A JSON round-trip is how it actually reaches the next
            # session, via the queue-state file.
            restored = normalize_frozen_matte(json.loads(json.dumps(record)))
            self.assertEqual(restored["artifact_sha256"],
                             record["artifact_sha256"])

    def test_anything_unrecognised_normalizes_to_nothing(self):
        # A half-parsed freeze must not be treated as a freeze; falling
        # back to detection is always safe, trusting a fragment is not.
        for value in (None, {}, [], "frozen", 3,
                      {"schema": "vsr.frozen_matte.v0"},
                      {"schema": FROZEN_MATTE_SCHEMA}):
            with self.subTest(value=value):
                self.assertEqual(normalize_frozen_matte(value), {})

    def test_a_record_missing_any_required_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, _artifact = _write_manifest(root)
            record = freeze_matte(manifest, _write_source(root / "clip.mp4"))
            for key in ("manifest", "manifest_sha256", "artifact",
                        "artifact_sha256", "timing_sha256", "source"):
                with self.subTest(missing=key):
                    broken = dict(record)
                    broken.pop(key)
                    self.assertEqual(normalize_frozen_matte(broken), {})

    def test_a_malformed_numeric_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, _artifact = _write_manifest(root)
            record = freeze_matte(manifest, _write_source(root / "clip.mp4"))
            for key in ("width", "frame_count", "source_start_frame",
                        "source_time_base_seconds"):
                with self.subTest(field=key):
                    broken = dict(record)
                    broken[key] = "not-a-number"
                    self.assertEqual(normalize_frozen_matte(broken), {})

    def test_summary_of_an_invalid_record_is_empty(self):
        self.assertEqual(frozen_matte_summary({"schema": "nope"}), "")


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.manifest, self.artifact = _write_manifest(self.root)
        self.source = _write_source(self.root / "clip.mp4")
        self.record = freeze_matte(self.manifest, self.source)

    def tearDown(self):
        self._tmp.cleanup()

    def _validate(self, **overrides):
        kwargs = {
            "source_path": self.source,
            "width": 64,
            "height": 48,
            "start_frame": 0,
            "end_frame": 3,
            "timestamps": [0.0, 0.1, 0.2],
            "durations": [0.1, 0.1, 0.1],
            "is_vfr": False,
            "source_time_base": 0.1,
        }
        kwargs.update(overrides)
        return validate_frozen_matte(self.record, **kwargs)

    def _expect(self, reason, **overrides):
        with self.assertRaises(FrozenMatteError) as ctx:
            self._validate(**overrides)
        self.assertEqual(ctx.exception.reason, reason)
        self.assertTrue(ctx.exception.needs_revalidation)
        # The message has to tell a person what to do next.
        self.assertTrue(ctx.exception.user_message.strip())
        return ctx.exception

    def test_a_matching_job_validates_and_reports_the_bypass(self):
        evidence = self._validate()
        self.assertEqual(evidence["status"], "validated")
        self.assertEqual(evidence["frame_count"], 3)
        self.assertEqual(
            evidence["bypassed_stages"], ["ocr", "tracking", "mask_refiners"])
        self.assertTrue(evidence["artifact_verified"])
        self.assertEqual(
            evidence["artifact_sha256"], self.record["artifact_sha256"])

    def test_geometry_range_and_timing_mismatches_each_fail_closed(self):
        self._expect("geometry_changed", width=32)
        self._expect("geometry_changed", height=24)
        self._expect("range_changed", start_frame=1)
        self._expect("range_changed", end_frame=4)
        self._expect("timing_mode_changed", is_vfr=True)
        self._expect("timing_changed", timestamps=[0.0, 0.1, 0.9])
        self._expect("timing_changed", durations=[0.1, 0.1, 0.5])
        self._expect("time_base_changed", source_time_base=0.5)

    def test_a_substituted_source_fails_closed(self):
        other = _write_source(self.root / "other.mp4", fill=b"\x99")
        self._expect("source_changed", source_path=other)

    def test_a_resized_source_fails_closed(self):
        bigger = _write_source(self.root / "bigger.mp4", size=8192)
        self._expect("source_size_changed", source_path=bigger)

    def test_an_edited_matte_fails_closed(self):
        cv2.imwrite(
            str(self.artifact / "frame_00000001.png"),
            np.full((48, 64), 3, dtype=np.uint8),
        )
        self._expect("artifact_changed")

    def test_a_moved_artifact_fails_closed(self):
        shutil.rmtree(self.artifact)
        self._expect("artifact_missing")

    def test_a_rewritten_manifest_fails_closed(self):
        payload = json.loads(self.manifest.read_text(encoding="utf-8"))
        payload["width"] = 64  # same value, but reserialized differently
        self.manifest.write_text(json.dumps(payload), encoding="utf-8")
        self._expect("manifest_changed")

    def test_a_deleted_manifest_fails_closed(self):
        self.manifest.unlink()
        self._expect("manifest_missing")

    def test_an_unreadable_record_fails_closed(self):
        self.record = {"schema": "nope"}
        self._expect("record_invalid")

    def test_an_older_fingerprint_scheme_fails_closed(self):
        self.record = dict(self.record)
        self.record["source"] = dict(self.record["source"])
        self.record["source"]["digest_scheme"] = "vsr.sampled-sha256.v0"
        self._expect("source_scheme_changed")

    def test_skipping_artifact_verification_is_recorded_not_assumed(self):
        # A caller may trade the rehash for speed, but the evidence must
        # say so rather than implying the pixels were checked.
        cv2.imwrite(
            str(self.artifact / "frame_00000001.png"),
            np.full((48, 64), 7, dtype=np.uint8),
        )
        evidence = self._validate(verify_artifact=False)
        self.assertFalse(evidence["artifact_verified"])


class ConfigPersistenceTests(unittest.TestCase):
    def test_a_frozen_record_survives_a_queue_state_round_trip(self):
        from gui.config import ProcessingConfig

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, _artifact = _write_manifest(root)
            record = freeze_matte(manifest, _write_source(root / "clip.mp4"))
            config = ProcessingConfig(frozen_matte=record).normalized()
            restored = ProcessingConfig.from_dict(config.to_dict()).normalized()
            self.assertEqual(
                restored.frozen_matte["artifact_sha256"],
                record["artifact_sha256"],
            )
            self.assertEqual(
                restored.frozen_matte["timing_sha256"],
                record["timing_sha256"],
            )

    def test_a_corrupt_persisted_record_degrades_to_detection(self):
        from gui.config import ProcessingConfig

        config = ProcessingConfig(
            frozen_matte={"schema": FROZEN_MATTE_SCHEMA, "manifest": "x"},
        ).normalized()
        self.assertEqual(config.frozen_matte, {})

    def test_the_default_is_no_frozen_matte(self):
        from gui.config import ProcessingConfig

        self.assertEqual(ProcessingConfig().normalized().frozen_matte, {})


class ProcessorBypassTests(unittest.TestCase):
    """The point of the whole item: a rerun that skips detection."""

    @staticmethod
    def _write_clip(path: Path):
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
        if not writer.isOpened():
            raise unittest.SkipTest("OpenCV MJPG writer unavailable")
        try:
            for value in (30, 60, 90, 120):
                writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
        finally:
            writer.release()

    @staticmethod
    def _stub_remover(config, inpainter):
        remover = processor.SubtitleRemover.__new__(processor.SubtitleRemover)
        remover.config = processor.normalize_processing_config(config)
        remover.detector = processor.SubtitleDetector.__new__(
            processor.SubtitleDetector)
        remover.detector.device = "cpu"
        remover.detector.lang = "en"
        remover.detector.vertical = False
        remover.detector._engine_name = "skip"
        remover.detector._rapid_model = None
        remover.detector._paddle_model = None
        remover.detector._surya_det = None
        remover.detector._easyocr_reader = None
        remover.inpainter = inpainter
        remover.on_progress = None
        remover.on_preview_frame = None
        remover.live_preview_stride = 6
        remover._hw_encoder = None
        remover._active_writer = None
        remover._active_subprocess = None
        remover._teardown_requested = False
        remover.last_quality_report = None
        remover.last_error_message = None
        remover.last_error_reason = None
        remover.last_resume_warning = None
        remover.last_pause_checkpoint = None
        remover.last_pause_checkpoint_path = None
        return remover

    def _base_config(self, **overrides):
        kwargs = dict(
            mode=processor.InpaintMode.STTN,
            device="cpu",
            sttn_skip_detection=True,
            preserve_audio=False,
            adaptive_batch=False,
            use_hw_encode=False,
            prefetch_decode=False,
            sttn_max_load_num=4,
            subtitle_area=(8, 32, 56, 44),
        )
        kwargs.update(overrides)
        return processor.ProcessingConfig(**kwargs)

    def _export_matte(self, root: Path, source: Path):
        class PassthroughInpainter:
            def inpaint(self, frames, _masks):
                return frames

        exporter = self._stub_remover(
            self._base_config(export_mask_video=True, mask_export_format="png"),
            PassthroughInpainter(),
        )
        self.assertTrue(exporter.process_video(
            str(source), str(root / "exported.mp4")))
        return (
            Path(exporter.last_mask_export["manifest"]),
            Path(exporter.last_mask_export["path"]),
        )

    def setUp(self):
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not on PATH")

    def test_a_frozen_matte_is_painted_verbatim_and_skips_detection(self):
        class RecordingInpainter:
            def __init__(self):
                self.mask_sums = []

            def inpaint(self, frames, masks):
                self.mask_sums.extend(int(mask.sum()) for mask in masks)
                return frames

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.avi"
            self._write_clip(source)
            manifest, artifact = self._export_matte(root, source)

            # Stand in for a human-reviewed matte: one distinctive frame.
            for index in range(4):
                cv2.imwrite(
                    str(artifact / f"frame_{index:08d}.png"),
                    np.full((48, 64), 117 if index == 2 else 0, dtype=np.uint8),
                )
            # Re-export the manifest hash so the pair agrees, the way a
            # real re-export after review would.
            from backend.matte_interchange import _sha256_sequence
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["artifact_sha256"] = _sha256_sequence(artifact, 4)
            manifest.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")

            record = freeze_matte(manifest, source)
            recorder = RecordingInpainter()
            rerun = self._stub_remover(
                self._base_config(frozen_matte=record), recorder)
            self.assertTrue(rerun.process_video(
                str(source), str(root / "frozen.mp4")))

            # The approved pixels reached the inpainter untouched: no
            # refiner, dilation, or stabilization pass altered them.
            self.assertEqual(recorder.mask_sums, [0, 0, 117 * 64 * 48, 0])
            # Every frame skipped detection, and none ran OCR.
            stats = rerun.last_detection_stats
            self.assertEqual(stats["frames_total"], 4)
            self.assertEqual(stats["frames_skipped"], 4)
            self.assertEqual(stats["frames_ocr"], 0)
            self.assertEqual(stats["skip_reasons"], {"frozen_matte": 4})
            self.assertEqual(
                rerun.last_frozen_matte["status"], "validated")
            self.assertEqual(
                rerun.last_frozen_matte["bypassed_stages"],
                ["ocr", "tracking", "mask_refiners"],
            )

            sidecar = json.loads(
                (root / "frozen.mp4.vsr.json").read_text(encoding="utf-8"))
            self.assertEqual(
                sidecar["frozenMatte"]["artifact_sha256"],
                record["artifact_sha256"],
            )
            self.assertEqual(
                sidecar["frozenMatte"]["source_digest_scheme"],
                SOURCE_DIGEST_SCHEME,
            )

    def test_a_stale_frozen_matte_stops_the_run_before_any_frame(self):
        class ExplodingInpainter:
            def inpaint(self, frames, masks):
                raise AssertionError("the run should not have started")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.avi"
            self._write_clip(source)
            manifest, artifact = self._export_matte(root, source)
            record = freeze_matte(manifest, source)

            # Edit the matte after freezing: the exact failure the freeze
            # promise exists to catch.
            cv2.imwrite(
                str(artifact / "frame_00000000.png"),
                np.full((48, 64), 250, dtype=np.uint8),
            )
            rerun = self._stub_remover(
                self._base_config(frozen_matte=record), ExplodingInpainter())
            # The processor's contract is a False return plus a typed
            # reason, the same shape every other input rejection uses.
            self.assertFalse(rerun.process_video(
                str(source), str(root / "stale.mp4")))
            self.assertEqual(
                rerun.last_error_reason, "frozen_matte_artifact_changed")
            self.assertIn("Freeze", rerun.last_error_message)
            self.assertEqual(rerun.last_frozen_matte["status"], "invalid")
            self.assertEqual(
                rerun.last_frozen_matte["reason"], "artifact_changed")
            self.assertTrue(
                rerun.last_frozen_matte["needs_revalidation"])
            # Nothing was decoded, so no output was produced.
            self.assertFalse((root / "stale.mp4").exists())

    def test_a_frozen_matte_and_a_manual_import_cannot_both_drive_a_job(self):
        class PassthroughInpainter:
            def inpaint(self, frames, _masks):
                return frames

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.avi"
            self._write_clip(source)
            manifest, _artifact = self._export_matte(root, source)
            record = freeze_matte(manifest, source)
            rerun = self._stub_remover(
                self._base_config(
                    frozen_matte=record,
                    mask_import_path=str(manifest),
                ),
                PassthroughInpainter(),
            )
            self.assertFalse(rerun.process_video(
                str(source), str(root / "both.mp4")))
            self.assertIn("cannot", rerun.last_error_message)
            self.assertFalse((root / "both.mp4").exists())


class CommandLineTests(unittest.TestCase):
    """`--frozen-matte` parity so the freeze is not GUI-only."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.manifest, self.artifact = _write_manifest(self.root)
        self.source = _write_source(self.root / "clip.mp4")

    def tearDown(self):
        self._tmp.cleanup()

    @staticmethod
    def _run_cli(*args, expect=None):
        proc = __import__("subprocess").run(
            [sys.executable, "-m", "backend.cli", *args],
            cwd=_ROOT, capture_output=True, text=True, timeout=180,
        )
        if expect is not None:
            assert proc.returncode == expect, (
                proc.returncode, proc.stdout[-1500:], proc.stderr[-1500:])
        return proc

    def test_the_flag_is_documented_in_help(self):
        proc = self._run_cli("--help", expect=0)
        self.assertIn("--frozen-matte", proc.stdout)
        # The help has to name the fail-closed behaviour, because that is
        # the surprising half of the feature.
        self.assertIn("Fails", proc.stdout)

    def test_a_frozen_matte_and_an_imported_matte_are_refused_together(self):
        proc = self._run_cli(
            "--input", str(self.source), "--output", str(self.root / "out.mp4"),
            "--frozen-matte", str(self.manifest),
            "--import-mask", str(self.manifest),
            expect=2,
        )
        self.assertIn("mutually exclusive", proc.stderr)

    def test_a_frozen_matte_cannot_describe_a_glob_of_inputs(self):
        proc = self._run_cli(
            "--pattern", str(self.root / "*.mp4"),
            "--out-dir", str(self.root / "out"),
            "--frozen-matte", str(self.manifest),
            expect=2,
        )
        self.assertIn("single --input", proc.stderr)

    def test_the_builder_turns_the_flag_into_a_validated_record(self):
        from argparse import Namespace
        from backend.cli import _frozen_matte_from_args

        args = Namespace(frozen_matte=str(self.manifest), input=str(self.source))
        record = normalize_frozen_matte(_frozen_matte_from_args(args))
        self.assertTrue(record)
        self.assertEqual(record["frame_count"], 3)

    def test_no_flag_means_no_record(self):
        from argparse import Namespace
        from backend.cli import _frozen_matte_from_args

        for value in ("", "   ", None):
            with self.subTest(value=value):
                args = Namespace(frozen_matte=value, input=str(self.source))
                self.assertEqual(_frozen_matte_from_args(args), {})

    def test_the_builder_fails_closed_on_a_mismatched_source(self):
        from argparse import Namespace
        from backend.cli import _frozen_matte_from_args

        args = Namespace(
            frozen_matte=str(self.manifest),
            input=str(self.root / "missing.mp4"),
        )
        with self.assertRaises(FrozenMatteError) as ctx:
            _frozen_matte_from_args(args)
        self.assertEqual(ctx.exception.reason, "source_missing")


class QueueActionTests(unittest.TestCase):
    """The GUI seam: freeze, release, and refuse with a reason."""

    def setUp(self):
        from gui.config import ProcessingConfig, ProcessingStatus, QueueItem

        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.source = _write_source(self.root / "clip.mp4")
        self.output = self.root / "clip.out.mp4"
        self.output.write_bytes(b"output")
        # The exporter writes `<output stem>.mask.json` beside the output.
        self.manifest, self.artifact = _write_manifest(self.root)
        self.manifest.rename(self.root / "clip.out.mask.json")
        (self.root / "clip.mask").rename(self.root / "clip.out.mask")
        self.manifest = self.root / "clip.out.mask.json"
        payload = json.loads(self.manifest.read_text(encoding="utf-8"))
        payload["artifact"] = "clip.out.mask"
        self.manifest.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        self.artifact = self.root / "clip.out.mask"

        self.item = QueueItem(
            id="item-1",
            file_path=str(self.source),
            output_path=str(self.output),
            config=ProcessingConfig(),
            status=ProcessingStatus.COMPLETE,
        )
        self.statuses = []
        self.saved = 0

        controller = self

        class Host:
            queue = [self.item]

            def _update_status(self, message, tone="neutral", toast=False):
                controller.statuses.append((message, tone))

            def _update_queue_display(self):
                controller.saved += 1

        from gui.quality_controller import QualityReviewControllerMixin

        class App(QualityReviewControllerMixin, Host):
            pass

        self.app = App()

    def tearDown(self):
        self._tmp.cleanup()

    def _freeze(self, action="freeze"):
        from unittest import mock

        with mock.patch("gui.quality_controller.save_queue_state") as saver:
            self.app._set_frozen_matte_for_item(self.item.id, action)
            return saver.call_count

    def test_freezing_records_the_matte_on_the_queue_item(self):
        self.assertEqual(self._freeze(), 1)
        record = normalize_frozen_matte(self.item.config.frozen_matte)
        self.assertTrue(record)
        self.assertEqual(record["frame_count"], 3)
        self.assertEqual(self.statuses[-1][1], "success")
        self.assertIn("64x48", self.statuses[-1][0])

    def test_releasing_clears_the_record_and_says_so(self):
        self._freeze()
        self.assertEqual(self._freeze("clear"), 1)
        self.assertEqual(self.item.config.frozen_matte, {})
        self.assertEqual(self.statuses[-1][1], "info")

    def test_freezing_without_an_exported_matte_explains_why_not(self):
        self.manifest.unlink()
        self.assertEqual(self._freeze(), 0)
        self.assertEqual(self.item.config.frozen_matte, {})
        self.assertEqual(self.statuses[-1][1], "warning")
        self.assertIn("matte export", self.statuses[-1][0])

    def test_freezing_an_edited_matte_surfaces_the_specific_reason(self):
        cv2.imwrite(
            str(self.artifact / "frame_00000000.png"),
            np.full((48, 64), 240, dtype=np.uint8),
        )
        self.assertEqual(self._freeze(), 0)
        self.assertEqual(self.item.config.frozen_matte, {})
        message, tone = self.statuses[-1]
        self.assertEqual(tone, "error")
        self.assertIn("edited", message)

    def test_an_unknown_item_id_is_a_no_op(self):
        self.assertEqual(
            self.app._set_frozen_matte_for_item("missing", "freeze"), None)
        self.assertEqual(self.statuses, [])


if __name__ == "__main__":
    unittest.main()
