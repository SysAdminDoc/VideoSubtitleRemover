"""Safe image decoding helpers for user-supplied still frames."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

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


def safe_imread(path: PathLike, flags: Optional[int] = None) -> Optional[np.ndarray]:
    """Read an image as OpenCV would, diverting vulnerable PNG paths to Pillow.

    User-controlled PNG input must not touch OpenCV's bundled libpng when
    `opencv_libpng_status().vulnerable` is true. Non-PNG files and fixed
    OpenCV builds keep the normal `cv2.imread` behavior.
    """
    import cv2

    if _is_png(path) and opencv_libpng_status().get("vulnerable") is True:
        return _pillow_read_png(path, flags)
    if flags is None:
        return cv2.imread(str(path))
    return cv2.imread(str(path), flags)
