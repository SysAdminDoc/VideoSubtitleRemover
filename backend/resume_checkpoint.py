"""Durable pause/resume checkpoints for long video runs."""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from backend.io import _write_text_atomic


SCHEMA = "vsr.pause_checkpoint.v1"


class ProcessingPaused(InterruptedError):
    """Raised when processing stopped at a resumable frame boundary."""

    def __init__(self, message: str, checkpoint_path: Optional[Path] = None):
        super().__init__(message)
        self.checkpoint_path = checkpoint_path


@dataclass
class CheckpointState:
    path: Path
    frame_dir: Path
    next_frame: int
    payload: dict
    warning: str = ""


def pause_checkpoint_path(checkpoint_dir: Path, key: str) -> Path:
    return Path(checkpoint_dir) / f"{key}.pause.json"


def pause_frame_dir(checkpoint_dir: Path, key: str) -> Path:
    return Path(checkpoint_dir) / f"{key}.frames"


def file_fingerprint(path: str) -> dict:
    p = Path(path)
    try:
        stat = p.stat()
    except OSError:
        return {"path": str(p), "exists": False}
    return {
        "path": str(p),
        "exists": True,
        "size": int(stat.st_size),
        "mtime": int(stat.st_mtime),
    }


def config_fingerprint(config: Any) -> str:
    payload = _jsonable_config(config)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


def load_pause_checkpoint(
    checkpoint_dir: Path,
    key: str,
    *,
    input_path: str,
    output_path: str,
    config_hash: str,
    total_frames: int,
    width: int,
    height: int,
    fps: float,
) -> CheckpointState:
    ckpt_path = pause_checkpoint_path(checkpoint_dir, key)
    default_frames = pause_frame_dir(checkpoint_dir, key)
    if not ckpt_path.exists():
        return CheckpointState(ckpt_path, default_frames, 0, {})
    try:
        payload = json.loads(ckpt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return CheckpointState(
            ckpt_path,
            default_frames,
            0,
            {},
            warning=f"Ignoring unreadable pause checkpoint {ckpt_path}: {exc}",
        )
    warning = _validation_warning(
        payload,
        input_path=input_path,
        output_path=output_path,
        config_hash=config_hash,
        total_frames=total_frames,
        width=width,
        height=height,
        fps=fps,
    )
    if warning:
        return CheckpointState(ckpt_path, default_frames, 0, {}, warning=warning)
    frame_dir = Path(payload.get("frame_dir") or default_frames)
    expected = _safe_int(payload.get("next_frame"), 0)
    contiguous = count_contiguous_frames(frame_dir)
    next_frame = max(0, min(total_frames, min(expected, contiguous)))
    if next_frame <= 0:
        return CheckpointState(
            ckpt_path,
            frame_dir,
            0,
            payload,
            warning=(
                f"Ignoring empty pause checkpoint {ckpt_path}; "
                "no contiguous processed frames were found."
            ),
        )
    if next_frame != expected:
        warning = (
            f"Pause checkpoint {ckpt_path} expected frame {expected}, "
            f"but only {next_frame} contiguous frame files were present."
        )
    return CheckpointState(ckpt_path, frame_dir, next_frame, payload, warning)


def write_pause_checkpoint(
    checkpoint_dir: Path,
    key: str,
    *,
    input_path: str,
    output_path: str,
    config_hash: str,
    frame_dir: Path,
    next_frame: int,
    total_frames: int,
    width: int,
    height: int,
    fps: float,
    status: str,
) -> dict:
    now = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    payload = {
        "schema": SCHEMA,
        "status": status,
        "input": str(input_path),
        "output": str(output_path),
        "input_fingerprint": file_fingerprint(input_path),
        "config_hash": config_hash,
        "frame_dir": str(frame_dir),
        "next_frame": max(0, int(next_frame)),
        "total_frames": max(0, int(total_frames)),
        "width": max(0, int(width)),
        "height": max(0, int(height)),
        "fps": round(float(fps), 6),
        "updated_at": now,
    }
    path = pause_checkpoint_path(checkpoint_dir, key)
    _write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def cleanup_pause_checkpoint(checkpoint_dir: Path, key: str, *,
                             remove_frames: bool = True) -> None:
    path = pause_checkpoint_path(checkpoint_dir, key)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    if remove_frames:
        shutil.rmtree(pause_frame_dir(checkpoint_dir, key), ignore_errors=True)


def count_contiguous_frames(frame_dir: Path, *, prefix: str = "frame",
                            ext: str = ".png") -> int:
    frame_dir = Path(frame_dir)
    idx = 0
    while (frame_dir / f"{prefix}_{idx:06d}{ext}").is_file():
        idx += 1
    return idx


def _validation_warning(payload: dict, *, input_path: str, output_path: str,
                        config_hash: str, total_frames: int, width: int,
                        height: int, fps: float) -> str:
    if not isinstance(payload, dict):
        return "Ignoring pause checkpoint because it is not a JSON object."
    if payload.get("schema") != SCHEMA:
        return "Ignoring pause checkpoint with an unsupported schema."
    expected_input = file_fingerprint(input_path)
    if payload.get("input_fingerprint") != expected_input:
        return "Ignoring pause checkpoint because the input file changed."
    if str(payload.get("output") or "") != str(output_path):
        return "Ignoring pause checkpoint because the output path changed."
    if str(payload.get("config_hash") or "") != str(config_hash):
        return "Ignoring pause checkpoint because processing settings changed."
    checks = {
        "total_frames": int(total_frames),
        "width": int(width),
        "height": int(height),
    }
    for key, expected in checks.items():
        if _safe_int(payload.get(key), -1) != expected:
            return f"Ignoring pause checkpoint because {key} changed."
    old_fps = _safe_float(payload.get("fps"), -1.0)
    if abs(old_fps - float(fps)) > 0.001:
        return "Ignoring pause checkpoint because the source frame rate changed."
    return ""


def _jsonable_config(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _jsonable_config(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (str, int, float, bool)):
        return enum_value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable_config(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
