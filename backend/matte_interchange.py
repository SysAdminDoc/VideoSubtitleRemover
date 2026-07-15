"""Lossless mask/alpha-matte export, import, and composition."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, Optional

import cv2
import numpy as np

from backend.io import (
    _LosslessIntermediateWriter,
    _allocate_temp_output_path,
    _promote_temp_output,
    _write_text_atomic,
)
from backend.safe_image import safe_imread


MASK_INTERCHANGE_SCHEMA = "vsr.mask_interchange.v1"
MASK_EXPORT_FORMATS = {"ffv1", "png"}
MASK_IMPORT_MODES = {"replace", "add", "subtract"}
MAX_MANIFEST_BYTES = 64 * 1024 * 1024


def normalize_mask_export_format(value: object) -> str:
    normalized = str(value or "ffv1").strip().lower()
    return normalized if normalized in MASK_EXPORT_FORMATS else "ffv1"


def normalize_mask_import_mode(value: object) -> str:
    normalized = str(value or "replace").strip().lower()
    return normalized if normalized in MASK_IMPORT_MODES else "replace"


def mask_interchange_paths(output_path: str | Path, export_format: str) -> tuple[Path, Path]:
    base = Path(output_path).with_suffix("")
    manifest = base.with_name(base.name + ".mask.json")
    if normalize_mask_export_format(export_format) == "png":
        return base.with_name(base.name + ".mask"), manifest
    return base.with_name(base.name + ".mask.mkv"), manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_sequence(directory: Path, frame_count: int) -> str:
    digest = hashlib.sha256()
    for index in range(frame_count):
        name = f"frame_{index:08d}.png"
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(_sha256_file(directory / name).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _coerce_gray_u8(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    if array.shape != (height, width):
        raise ValueError(
            f"Matte dimensions {array.shape[::-1]} do not match {width}x{height}"
        )
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def compose_imported_matte(
    base_mask: np.ndarray,
    imported_matte: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Compose an imported 8-bit matte after native mask generation."""
    imported = _coerce_gray_u8(
        imported_matte, base_mask.shape[1], base_mask.shape[0])
    normalized_mode = normalize_mask_import_mode(mode)
    if normalized_mode == "replace":
        return imported.copy()
    if normalized_mode == "add":
        return np.maximum(base_mask, imported)
    return cv2.subtract(base_mask, imported)


class MaskInterchangeWriter:
    """Transactionally write a lossless matte plus timing manifest."""

    def __init__(
        self,
        output_path: str | Path,
        export_format: str,
        *,
        width: int,
        height: int,
        fps: float,
        start_frame: int,
        end_frame: int,
        timestamps: Iterable[float],
        durations: Iterable[float],
        is_vfr: bool,
        source_time_base: float,
    ):
        self.export_format = normalize_mask_export_format(export_format)
        self.width = int(width)
        self.height = int(height)
        self.fps = max(float(fps), 1e-9)
        self.start_frame = max(0, int(start_frame))
        self.end_frame = max(self.start_frame, int(end_frame))
        self.timestamps = [round(float(value), 9) for value in timestamps]
        self.durations = [round(float(value), 9) for value in durations]
        self.is_vfr = bool(is_vfr)
        self.source_time_base = max(0.0, float(source_time_base or 0.0))
        self.artifact_path, self.manifest_path = mask_interchange_paths(
            output_path, self.export_format)
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        self._frame_count = 0
        self._finished = False
        self._video_writer: Optional[_LosslessIntermediateWriter] = None
        self._temp_artifact: Optional[Path] = None
        self._temp_dir: Optional[Path] = None

        if len(self.timestamps) != self.end_frame - self.start_frame:
            raise ValueError("Matte timestamp count does not match the processed range")
        if len(self.durations) != len(self.timestamps):
            raise ValueError("Matte duration count does not match the processed range")
        if self.export_format == "ffv1":
            self._temp_artifact = _allocate_temp_output_path(self.artifact_path)
            self._video_writer = _LosslessIntermediateWriter(
                str(self._temp_artifact), self.width, self.height, self.fps,
                pixel_format="gray", allow_lossy_fallback=False,
            )
            if not self._video_writer.isOpened() or not self._video_writer.lossless:
                self.abort()
                raise RuntimeError(
                    "FFV1 matte export requires an available FFmpeg executable"
                )
        else:
            self._temp_dir = Path(tempfile.mkdtemp(
                prefix=f".{self.artifact_path.name}.",
                suffix=".tmp",
                dir=str(self.artifact_path.parent),
            ))

    def write(self, mask: np.ndarray) -> None:
        if self._finished:
            raise RuntimeError("Matte writer is already finalized")
        matte = _coerce_gray_u8(mask, self.width, self.height)
        if self.export_format == "ffv1":
            assert self._video_writer is not None
            self._video_writer.write(matte)
        else:
            assert self._temp_dir is not None
            frame_path = self._temp_dir / f"frame_{self._frame_count:08d}.png"
            if not cv2.imwrite(
                str(frame_path), matte, [cv2.IMWRITE_PNG_COMPRESSION, 9]
            ):
                raise OSError(f"Could not write matte frame {frame_path.name}")
        self._frame_count += 1

    def finalize(self) -> dict:
        expected = self.end_frame - self.start_frame
        if self._frame_count != expected:
            raise ValueError(
                f"Matte writer received {self._frame_count} frames; expected {expected}"
            )
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
        if self.export_format == "ffv1":
            assert self._temp_artifact is not None
            if not self._temp_artifact.is_file() or not self._temp_artifact.stat().st_size:
                raise OSError("FFV1 matte writer produced no artifact")
            content_sha256 = _sha256_file(self._temp_artifact)
            _promote_temp_output(self._temp_artifact, self.artifact_path)
            self._temp_artifact = None
        else:
            assert self._temp_dir is not None
            content_sha256 = _sha256_sequence(self._temp_dir, expected)
            if self.artifact_path.exists():
                if self.artifact_path.is_dir():
                    shutil.rmtree(self.artifact_path)
                else:
                    self.artifact_path.unlink()
            os.replace(self._temp_dir, self.artifact_path)
            self._temp_dir = None

        payload = {
            "schema": MASK_INTERCHANGE_SCHEMA,
            "format": self.export_format,
            "artifact": self.artifact_path.name,
            "artifact_sha256": content_sha256,
            "pixel_format": "gray8",
            "width": self.width,
            "height": self.height,
            "frame_count": expected,
            "source_start_frame": self.start_frame,
            "source_end_frame": self.end_frame,
            "source_is_vfr": self.is_vfr,
            "source_time_base_seconds": round(self.source_time_base, 12),
            "timestamps_seconds": self.timestamps,
            "durations_seconds": self.durations,
        }
        _write_text_atomic(
            self.manifest_path,
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
        )
        manifest_sha256 = _sha256_file(self.manifest_path)
        self._finished = True
        return {
            "schema": MASK_INTERCHANGE_SCHEMA,
            "status": "created",
            "format": self.export_format,
            "path": str(self.artifact_path),
            "manifest": str(self.manifest_path),
            "artifact_sha256": content_sha256,
            "manifest_sha256": manifest_sha256,
            "frame_count": expected,
            "pixel_format": "gray8",
        }

    def abort(self) -> None:
        if self._video_writer is not None:
            self._video_writer.terminate()
            self._video_writer = None
        if self._temp_artifact is not None:
            try:
                self._temp_artifact.unlink()
            except OSError:
                pass
            self._temp_artifact = None
        if self._temp_dir is not None:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None


def _load_manifest(path: str | Path) -> tuple[Path, dict]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ValueError("Imported matte manifest does not exist")
    if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("Imported matte manifest exceeds the 64 MiB limit")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Imported matte manifest is invalid: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != MASK_INTERCHANGE_SCHEMA:
        raise ValueError("Imported matte manifest has an unsupported schema")
    if str(payload.get("format") or "").strip().lower() not in MASK_EXPORT_FORMATS:
        raise ValueError("Imported matte manifest has an unsupported format")
    return manifest_path, payload


def _artifact_from_manifest(manifest_path: Path, payload: dict) -> Path:
    artifact_name = str(payload.get("artifact") or "")
    if not artifact_name or Path(artifact_name).name != artifact_name:
        raise ValueError("Imported matte artifact path must be a sibling name")
    return manifest_path.parent / artifact_name


def inspect_matte_manifest(path: str | Path) -> dict:
    """Validate the manifest envelope and return compact UI-safe metadata."""
    manifest_path, payload = _load_manifest(path)
    artifact = _artifact_from_manifest(manifest_path, payload)
    return {
        "schema": MASK_INTERCHANGE_SCHEMA,
        "manifest": str(manifest_path),
        "artifact": str(artifact),
        "format": normalize_mask_export_format(payload.get("format")),
        "width": int(payload.get("width", 0) or 0),
        "height": int(payload.get("height", 0) or 0),
        "frame_count": int(payload.get("frame_count", 0) or 0),
    }


class MaskInterchangeReader:
    """Preflight and read an edited matte using exact source timing."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        width: int,
        height: int,
        start_frame: int,
        end_frame: int,
        timestamps: Iterable[float],
        durations: Iterable[float],
        is_vfr: bool,
        source_time_base: float,
        mode: str,
    ):
        self.manifest_path, self.manifest = _load_manifest(manifest_path)
        self.artifact_path = _artifact_from_manifest(
            self.manifest_path, self.manifest)
        self.export_format = normalize_mask_export_format(
            self.manifest.get("format"))
        self.mode = normalize_mask_import_mode(mode)
        self.width = int(width)
        self.height = int(height)
        self.frame_count = max(0, int(end_frame) - int(start_frame))
        self._capture: Optional[cv2.VideoCapture] = None
        self._capture_position = 0
        self._validate_metadata(
            start_frame, end_frame, timestamps, durations,
            is_vfr, source_time_base)
        self._validate_artifact()
        current_hash = (
            _sha256_sequence(self.artifact_path, self.frame_count)
            if self.export_format == "png"
            else _sha256_file(self.artifact_path)
        )
        exported_hash = str(self.manifest.get("artifact_sha256") or "")
        self.evidence = {
            "schema": MASK_INTERCHANGE_SCHEMA,
            "manifest": str(self.manifest_path),
            "manifest_sha256": _sha256_file(self.manifest_path),
            "artifact": str(self.artifact_path),
            "artifact_sha256": current_hash,
            "exported_artifact_sha256": exported_hash,
            "edited_since_export": bool(exported_hash and exported_hash != current_hash),
            "format": self.export_format,
            "mode": self.mode,
            "frame_count": self.frame_count,
            "composition_order": [
                "ocr_and_manual_regions",
                "manual_add_subtract_corrections",
                "mask_refiners_and_stabilization",
                f"imported_matte_{self.mode}",
            ],
        }

    def _validate_metadata(
        self,
        start_frame: int,
        end_frame: int,
        timestamps: Iterable[float],
        durations: Iterable[float],
        is_vfr: bool,
        source_time_base: float,
    ) -> None:
        expected_timestamps = [float(value) for value in timestamps]
        expected_durations = [float(value) for value in durations]
        if (
            len(expected_timestamps) != self.frame_count
            or len(expected_durations) != self.frame_count
        ):
            raise ValueError("Source timing count does not match the selected range")
        checks = {
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
            "source_start_frame": int(start_frame),
            "source_end_frame": int(end_frame),
        }
        for key, expected in checks.items():
            try:
                actual = int(self.manifest.get(key, -1))
            except (TypeError, ValueError):
                actual = -1
            if actual != expected:
                raise ValueError(
                    f"Imported matte {key} is {actual}; expected {expected}"
                )
        if str(self.manifest.get("pixel_format") or "") != "gray8":
            raise ValueError("Imported matte must use the gray8 pixel format")
        if bool(self.manifest.get("source_is_vfr", False)) != bool(is_vfr):
            raise ValueError("Imported matte CFR/VFR timing mode does not match the source")
        try:
            manifest_time_base = float(
                self.manifest.get("source_time_base_seconds", 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError("Imported matte time base is malformed") from exc
        raw_timestamps = self.manifest.get("timestamps_seconds")
        if not isinstance(raw_timestamps, list) or len(raw_timestamps) != self.frame_count:
            raise ValueError("Imported matte timestamp count does not match the source")
        try:
            actual_timestamps = [float(value) for value in raw_timestamps]
        except (TypeError, ValueError) as exc:
            raise ValueError("Imported matte timestamps are malformed") from exc
        tolerance = max(1e-6, abs(float(source_time_base or 0.0)) * 0.51)
        if abs(manifest_time_base - float(source_time_base or 0.0)) > tolerance:
            raise ValueError("Imported matte time base does not match the source")
        for index, (actual, expected) in enumerate(
            zip(actual_timestamps, expected_timestamps)
        ):
            if not np.isfinite(actual) or abs(actual - expected) > tolerance:
                raise ValueError(
                    "Imported matte timestamp mismatch at frame "
                    f"{index}: {actual} vs {expected}"
                )
        raw_durations = self.manifest.get("durations_seconds")
        if not isinstance(raw_durations, list) or len(raw_durations) != self.frame_count:
            raise ValueError("Imported matte duration count does not match the source")
        try:
            actual_durations = [float(value) for value in raw_durations]
        except (TypeError, ValueError) as exc:
            raise ValueError("Imported matte durations are malformed") from exc
        for index, (actual, expected) in enumerate(
            zip(actual_durations, expected_durations)
        ):
            if not np.isfinite(actual) or abs(actual - expected) > tolerance:
                raise ValueError(
                    "Imported matte duration mismatch at frame "
                    f"{index}: {actual} vs {expected}"
                )

    def _validate_artifact(self) -> None:
        if self.export_format == "png":
            if not self.artifact_path.is_dir():
                raise ValueError("Imported PNG matte sequence directory is missing")
            frames = sorted(self.artifact_path.glob("frame_*.png"))
            if len(frames) != self.frame_count:
                raise ValueError(
                    f"Imported matte has {len(frames)} PNG frames; "
                    f"expected {self.frame_count}"
                )
            for index in range(self.frame_count):
                expected = self.artifact_path / f"frame_{index:08d}.png"
                frame = safe_imread(expected, cv2.IMREAD_UNCHANGED)
                if frame is None:
                    raise ValueError(f"Imported matte frame {index} is unreadable")
                _coerce_gray_u8(frame, self.width, self.height)
            return
        if not self.artifact_path.is_file():
            raise ValueError("Imported FFV1 matte video is missing")
        capture = cv2.VideoCapture(str(self.artifact_path))
        try:
            if not capture.isOpened():
                raise ValueError("Imported FFV1 matte video could not be opened")
            count = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                _coerce_gray_u8(frame, self.width, self.height)
                count += 1
            if count != self.frame_count:
                raise ValueError(
                    f"Imported matte has {count} video frames; "
                    f"expected {self.frame_count}"
                )
        finally:
            capture.release()

    def read(self, index: int) -> np.ndarray:
        index = int(index)
        if not 0 <= index < self.frame_count:
            raise IndexError(f"Matte frame index {index} is out of range")
        if self.export_format == "png":
            frame = safe_imread(
                self.artifact_path / f"frame_{index:08d}.png",
                cv2.IMREAD_UNCHANGED,
            )
            if frame is None:
                raise ValueError(f"Imported matte frame {index} became unreadable")
            return _coerce_gray_u8(frame, self.width, self.height)
        if self._capture is None:
            self._capture = cv2.VideoCapture(str(self.artifact_path))
            self._capture_position = 0
        if index != self._capture_position:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            self._capture_position = index
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise ValueError(f"Imported matte frame {index} became unreadable")
        self._capture_position = index + 1
        return _coerce_gray_u8(frame, self.width, self.height)

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


def load_matte_preview_frame(
    manifest_path: str | Path,
    *,
    frame_index: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, dict]:
    """Read one manifest-backed matte frame for GUI composition preview."""
    path, payload = _load_manifest(manifest_path)
    artifact = _artifact_from_manifest(path, payload)
    export_format = normalize_mask_export_format(payload.get("format"))
    count = int(payload.get("frame_count", 0) or 0)
    index = max(0, min(max(0, count - 1), int(frame_index)))
    if int(payload.get("width", -1)) != width or int(payload.get("height", -1)) != height:
        raise ValueError("Imported matte dimensions do not match the preview frame")
    if export_format == "png":
        frame = safe_imread(
            artifact / f"frame_{index:08d}.png", cv2.IMREAD_UNCHANGED)
    else:
        capture = cv2.VideoCapture(str(artifact))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                frame = None
        finally:
            capture.release()
    if frame is None:
        raise ValueError("Imported matte preview frame is unreadable")
    return _coerce_gray_u8(frame, width, height), {
        "format": export_format,
        "frame_index": index,
        "artifact": str(artifact),
    }
