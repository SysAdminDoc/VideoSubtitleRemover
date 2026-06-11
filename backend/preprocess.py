"""Optional pre-detect / pre-inpaint preprocessing adapters.

RM-33 FastDVDnet denoise -- run a denoise pass on detection frames so
the OCR cascade sees a cleaner signal. Output frames are NOT modified;
the inpainter still operates on the original BGR data. Useful for VHS
rips, low-light phone clips, etc.

RM-21 TransNetV2 deep scene-cut detector -- alternative to the
histogram heuristic and to PySceneDetect. Tracks short-form video
benchmarks (F1 ~0.80 on SHOT vs ~0.6 for histogram).

Both adapters import lazily and return None / no-op when the optional
dependency or weight file is missing.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


_FASTDVDNET_STATE = {"probed": False, "model": None}


def fastdvdnet_denoise_frame(frame: np.ndarray,
                              sigma_noise: float = 25.0) -> np.ndarray:
    """RM-33: denoise a single BGR frame for OCR-side use only.

    Pre-trained FastDVDnet weights live at the path named by
    `VSR_FASTDVDNET` (a `.pth` checkpoint shipped with the upstream
    repo). When unavailable we fall back to OpenCV's
    `fastNlMeansDenoisingColored` which is conservative but ships with
    every cv2 build.
    """
    weight_path = os.environ.get("VSR_FASTDVDNET", "")
    if weight_path:
        model = _load_fastdvdnet(weight_path)
        if model is not None:
            try:
                return _run_fastdvdnet(model, frame)
            except Exception as exc:
                logger.debug(f"FastDVDnet inference failed; falling back: {exc}")
    # cv2 fallback -- a few ms on a 1080p frame; cheap enough to run
    # per-detection-frame when the user opts in.
    return cv2.fastNlMeansDenoisingColored(frame, None, 7, 7, 7, 21)


def _load_fastdvdnet(weight_path: str):
    if _FASTDVDNET_STATE["probed"]:
        return _FASTDVDNET_STATE["model"]
    _FASTDVDNET_STATE["probed"] = True
    try:
        import torch  # type: ignore
        if not Path(weight_path).is_file():
            logger.warning(
                f"FastDVDnet weight file not found at {weight_path!r}; "
                f"using cv2 NLM fallback."
            )
            return None
        try:
            from backend.adapter_manifest import (
                log_adapter_verification,
                verify_adapter_path,
            )
            verification = verify_adapter_path("fastdvdnet", weight_path)
            log_adapter_verification(verification)
            if not verification.allowed:
                return None
        except Exception as exc:
            logger.warning(f"FastDVDnet weight verification failed: {exc}")
            return None
        # The upstream FastDVDnet ships its own model definition; we
        # import the module if the user has it on PYTHONPATH. The repo
        # itself is intentionally NOT vendored to avoid the ~5 MB
        # weight + ~200 LOC model code bloat.
        try:
            from fastdvdnet.models import FastDVDnet  # type: ignore
        except ImportError:
            logger.info(
                "fastdvdnet package not importable; install from "
                "https://github.com/m-tassano/fastdvdnet or "
                "VSR_FASTDVDNET pointing at a packaged variant."
            )
            return None
        net = FastDVDnet(num_input_frames=5)
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        net.load_state_dict(state)
        net.eval()
        if torch.cuda.is_available():
            net = net.cuda()
        _FASTDVDNET_STATE["model"] = net
        return net
    except Exception as exc:
        logger.debug(f"FastDVDnet load failed: {exc}")
        return None


def _run_fastdvdnet(model, frame: np.ndarray) -> np.ndarray:
    """Single-frame proxy: FastDVDnet expects 5 frames; we repeat the
    input to get a clean reference baseline. Real OCR usage should
    feed the temporal window from the detection loop -- this single-
    frame variant is for the per-frame fallback path."""
    import torch  # type: ignore
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    seq = np.stack([rgb] * 5, axis=0)  # (5, H, W, 3)
    t = torch.from_numpy(seq).permute(0, 3, 1, 2).unsqueeze(0)
    if next(model.parameters()).is_cuda:
        t = t.cuda()
    with torch.no_grad():
        out = model(t).clamp_(0.0, 1.0).cpu().numpy()
    rgb_out = (out[0].transpose(1, 2, 0) * 255.0).astype(np.uint8)
    return cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)


_TRANSNETV2_STATE = {"probed": False, "model": None}


def transnetv2_scene_cuts(frames: List[np.ndarray]) -> Optional[List[int]]:
    """RM-21: deep scene-cut indices via TransNetV2.

    Returns the list of cut start indices (always starts with 0) or
    None when the optional dependency / weights are missing. The
    caller falls back to the histogram or PySceneDetect path.
    """
    if not frames or len(frames) < 2:
        return [0] if frames else None
    weight_path = os.environ.get("VSR_TRANSNETV2", "")
    model = _load_transnetv2(weight_path) if weight_path else None
    if model is None:
        return None
    try:
        # TransNetV2 takes uint8 frames at 48x27 in (T, 27, 48, 3).
        resized = np.stack([
            cv2.resize(f, (48, 27), interpolation=cv2.INTER_AREA)
            for f in frames
        ], axis=0)
        predictions, _ = model(resized)  # caller signature varies
        # `predictions` is per-frame [0, 1]; high value = cut start.
        cuts = [0]
        for i, p in enumerate(predictions):
            if i > 0 and float(p) > 0.5:
                cuts.append(i)
        return sorted(set(cuts))
    except Exception as exc:
        logger.debug(f"TransNetV2 inference failed: {exc}")
        return None


def _load_transnetv2(weight_path: str):
    if _TRANSNETV2_STATE["probed"]:
        return _TRANSNETV2_STATE["model"]
    _TRANSNETV2_STATE["probed"] = True
    try:
        if weight_path and not Path(weight_path).is_file():
            logger.warning(
                f"TransNetV2 weight not found at {weight_path!r}"
            )
            return None
        if weight_path:
            try:
                from backend.adapter_manifest import (
                    log_adapter_verification,
                    verify_adapter_path,
                )
                verification = verify_adapter_path("transnetv2", weight_path)
                log_adapter_verification(verification)
                if not verification.allowed:
                    return None
            except Exception as exc:
                logger.warning(f"TransNetV2 weight verification failed: {exc}")
                return None
        try:
            from transnetv2 import TransNetV2  # type: ignore
        except ImportError:
            logger.info(
                "transnetv2 package not importable; install with "
                "`pip install transnetv2` to enable the deep "
                "scene-cut detector."
            )
            return None
        model = TransNetV2(model_path=weight_path) if weight_path else TransNetV2()
        _TRANSNETV2_STATE["model"] = model
        return model
    except Exception as exc:
        logger.debug(f"TransNetV2 load failed: {exc}")
        return None
