"""Safe image decoding helpers for user-supplied still frames."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from backend.security_checks import opencv_libpng_status

logger = logging.getLogger(__name__)


PathLike = Union[str, Path]


def _is_png(path: PathLike) -> bool:
    return Path(path).suffix.lower() == ".png"


def _pillow_read_png(path: PathLike, flags: Optional[int]) -> Optional[np.ndarray]:
    try:
        import cv2
        from PIL import Image
    except Exception as exc:
        logger.warning(
            "Cannot safely decode PNG without Pillow while OpenCV libpng is "
            "vulnerable: %s",
            exc,
        )
        return None
    try:
        with Image.open(path) as image:
            if flags == getattr(cv2, "IMREAD_GRAYSCALE", 0):
                return np.array(image.convert("L"))
            if flags == getattr(cv2, "IMREAD_UNCHANGED", -1):
                if image.mode in {"RGBA", "LA"} or (
                        image.mode == "P" and "transparency" in image.info):
                    rgba = np.array(image.convert("RGBA"))
                    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
            rgb = np.array(image.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception as exc:
        logger.warning("Safe PNG decode failed for %s: %s", path, exc)
        return None


def _read_all_bytes(path: PathLike) -> Optional[bytes]:
    """Read a file whole, mirroring `cv2.imread`'s silent-None on failure."""
    try:
        with open(path, "rb") as handle:
            return handle.read()
    except OSError as exc:
        logger.debug("Image read failed for %s: %s", path, exc)
        return None


def libpng_vulnerable() -> bool:
    """Whether OpenCV's bundled libpng is the CVE-affected version."""
    return opencv_libpng_status().get("vulnerable") is True


def safe_imread(path: PathLike, flags: Optional[int] = None, *,
                png_vulnerable: Optional[bool] = None) -> Optional[np.ndarray]:
    """Read an image as OpenCV would, diverting vulnerable PNG paths to Pillow.

    User-controlled PNG input must not touch OpenCV's bundled libpng when
    `opencv_libpng_status().vulnerable` is true. Non-PNG files and fixed
    OpenCV builds keep the normal `cv2.imread` behavior.

    `png_vulnerable` lets a caller reading many frames (e.g. a PNG frame
    sequence) resolve the process-static libpng status once and pass it in,
    avoiding a `cv2.getBuildInformation()` parse on every frame. When omitted
    the status is resolved per call, preserving the original behavior.
    """
    import cv2

    if _is_png(path):
        vulnerable = libpng_vulnerable() if png_vulnerable is None else png_vulnerable
        if vulnerable:
            return _pillow_read_png(path, flags)
    payload = _read_all_bytes(path)
    if payload is None:
        return None
    buffer = np.frombuffer(payload, dtype=np.uint8)
    if buffer.size == 0:
        return None
    decode_flags = cv2.IMREAD_COLOR if flags is None else flags
    try:
        frame = cv2.imdecode(buffer, decode_flags)
    except cv2.error as exc:
        logger.warning("Image decode failed for %s: %s", path, exc)
        return None
    return frame


def safe_imwrite(path: PathLike, image: np.ndarray,
                 params: Optional[Sequence[int]] = None) -> bool:
    """Write an image as OpenCV would, without OpenCV touching the path.

    RM-317: `cv2.imwrite` passes the filename to the C++ layer as a narrow
    byte string, so any path holding CJK, Cyrillic, or accented Latin
    characters fails silently by returning False. Encoding in memory and
    writing the bytes through Python keeps full Unicode path support while
    preserving the `cv2.imwrite` contract: True on success, False on an
    unsupported extension, an unencodable array, or a failed write.
    """
    import cv2

    target = Path(path)
    suffix = target.suffix
    if not suffix:
        return False
    try:
        ok, buffer = cv2.imencode(suffix, image,
                                  list(params) if params is not None else [])
    except cv2.error as exc:
        logger.warning("Image encode failed for %s: %s", target, exc)
        return False
    if not ok or buffer is None:
        return False
    try:
        with open(target, "wb") as handle:
            handle.write(buffer.tobytes())
    except OSError as exc:
        logger.warning("Image write failed for %s: %s", target, exc)
        return False
    return True
