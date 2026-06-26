"""External inpainter command adapter.

Process-isolated bridge that lets advanced users run any research
inpainter in a subprocess without bundling licenses, weights, or
dependency stacks. The external command receives input frames, masks,
and an output directory, then writes inpainted frames as PNGs.

Gated behind VSR_EXTERNAL_INPAINTER env var which must point to an
executable or script. The adapter enforces a timeout, validates paths,
and checks exit codes.

Contract:
    <command> --input-dir <frames> --mask-dir <masks> --output-dir <out>
              --config <json-string>

    Input/mask frames are numbered PNGs (000000.png, 000001.png, ...).
    The command must write identically-named PNGs into --output-dir.
    Exit 0 on success; non-zero fails the batch with the stderr excerpt.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

from backend.inpainters._common import BaseInpainter, _cv2_inpaint, _feather_blend
from backend.safe_image import safe_imread

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 600


def _normalized_static_mask(masks: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    prepared: List[np.ndarray] = []
    shape = None
    for mask in masks:
        if mask is None:
            continue
        arr = np.asarray(mask)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        if arr.ndim != 2:
            continue
        if shape is None:
            shape = arr.shape
        if arr.shape != shape:
            raise ValueError("all static-logo masks must have the same shape")
        prepared.append((arr > 0).astype(np.uint8) * 255)
    if not prepared:
        return None
    static = np.maximum.reduce(prepared)
    return static if int(static.max()) > 0 else None


def deterministic_static_logo_cleanup(
    frames: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    *,
    radius: int = 7,
    feather_px: int = 4,
    temporal_blend: bool = True,
) -> List[np.ndarray]:
    """Deterministic static-logo removal candidate for benchmark runs.

    The method never calls external commands or model weights. It unions the
    fixed logo mask across the clip, performs per-frame OpenCV inpainting, then
    optionally stabilizes the masked pixels with a temporal median. This gives
    the research harness an InpaintDelogo-style no-dependency baseline to
    compare against the current per-frame cv2 cleanup path.
    """
    frames = list(frames)
    masks = list(masks)
    if len(frames) != len(masks):
        raise ValueError("frames and masks must have the same length")
    if not frames:
        return []
    first_shape = frames[0].shape
    for idx, frame in enumerate(frames):
        if frame.shape != first_shape:
            raise ValueError(f"frame {idx} shape does not match frame 0")
    static_mask = _normalized_static_mask(masks)
    if static_mask is None:
        return [frame.copy() for frame in frames]
    if static_mask.shape != first_shape[:2]:
        raise ValueError("static-logo mask shape does not match frames")

    filled_frames: List[np.ndarray] = []
    for frame in frames:
        filled_frames.append(_cv2_inpaint(frame, static_mask, radius, cv2.INPAINT_TELEA))

    if temporal_blend and len(filled_frames) > 1:
        stable = np.median(np.stack(filled_frames, axis=0), axis=0).astype(np.uint8)
    else:
        stable = None

    outputs: List[np.ndarray] = []
    for original, mask, filled in zip(frames, masks, filled_frames):
        frame_mask = np.asarray(mask)
        if frame_mask.ndim == 3:
            frame_mask = cv2.cvtColor(frame_mask, cv2.COLOR_BGR2GRAY)
        if int(frame_mask.max()) == 0:
            outputs.append(original.copy())
            continue
        candidate = filled.copy()
        if stable is not None:
            candidate[frame_mask > 0] = stable[frame_mask > 0]
        outputs.append(_feather_blend(original, candidate, frame_mask, feather_px))
    return outputs


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _split_external_command(command: str) -> List[str]:
    """Split the trusted env command without corrupting Windows paths."""
    try:
        parts = shlex.split(command, posix=(os.name != "nt"))
    except ValueError:
        return []
    return [_strip_wrapping_quotes(part) for part in parts if part]


def _external_command() -> Optional[List[str]]:
    cmd = os.environ.get("VSR_EXTERNAL_INPAINTER", "").strip()
    if not cmd:
        return None
    parts = _split_external_command(cmd)
    if not parts:
        logger.warning("VSR_EXTERNAL_INPAINTER command could not be parsed")
        return None
    executable = parts[0]
    if shutil.which(executable) is None and not Path(executable).is_file():
        logger.warning(
            f"VSR_EXTERNAL_INPAINTER command not found: {executable}")
        return None
    return parts


def is_available() -> bool:
    return _external_command() is not None


class ExternalInpainter(BaseInpainter):
    """Adapter that delegates inpainting to an external subprocess."""

    def __init__(self, device: str = "cpu", config=None):
        self._cmd = _external_command()
        self._timeout = int(os.environ.get(
            "VSR_EXTERNAL_TIMEOUT", str(DEFAULT_TIMEOUT)))
        self._config = config
        if self._cmd is None:
            raise RuntimeError(
                "VSR_EXTERNAL_INPAINTER is not set or the command is "
                "not found on PATH"
            )

    def inpaint(self, frames: List[np.ndarray],
                masks: List[np.ndarray]) -> List[np.ndarray]:
        if not frames:
            return []
        work = tempfile.mkdtemp(prefix="vsr_ext_")
        try:
            return self._run(frames, masks, work)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _run(self, frames: List[np.ndarray], masks: List[np.ndarray],
             work: str) -> List[np.ndarray]:
        in_dir = os.path.join(work, "input")
        mask_dir = os.path.join(work, "masks")
        out_dir = os.path.join(work, "output")
        os.makedirs(in_dir)
        os.makedirs(mask_dir)
        os.makedirs(out_dir)

        for i, (frame, mask) in enumerate(zip(frames, masks)):
            name = f"{i:06d}.png"
            cv2.imwrite(os.path.join(in_dir, name), frame)
            cv2.imwrite(os.path.join(mask_dir, name), mask)

        cfg_str = "{}"
        if self._config is not None:
            try:
                cfg_str = json.dumps(self._config.to_dict()
                                     if hasattr(self._config, "to_dict")
                                     else {})
            except Exception:
                cfg_str = "{}"

        cmd_parts = list(self._cmd)
        cmd_parts.extend([
            "--input-dir", in_dir,
            "--mask-dir", mask_dir,
            "--output-dir", out_dir,
            "--config", cfg_str,
        ])

        try:
            result = subprocess.run(
                cmd_parts,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
        except subprocess.TimeoutExpired:
            logger.error(
                f"External inpainter timed out after {self._timeout}s")
            return list(frames)
        except FileNotFoundError:
            logger.error(f"External inpainter not found: {cmd_parts[0]}")
            return list(frames)

        if result.returncode != 0:
            stderr = (result.stderr or "")[:400]
            logger.error(
                f"External inpainter exit {result.returncode}: {stderr}")
            return list(frames)

        results = []
        for i, (frame, mask) in enumerate(zip(frames, masks)):
            out_path = os.path.join(out_dir, f"{i:06d}.png")
            if os.path.isfile(out_path):
                out_frame = safe_imread(out_path)
                if out_frame is not None and out_frame.shape == frame.shape:
                    blended = _feather_blend(out_frame, frame, mask)
                    results.append(blended)
                else:
                    results.append(frame.copy())
            else:
                results.append(frame.copy())
        return results
