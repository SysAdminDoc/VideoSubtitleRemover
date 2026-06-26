"""Opt-in diffusion video-inpainter backends.

Each backend below registers itself with the inpainter registry when
the user has set its enable env var AND the optional dependency
imports successfully. Missing deps mean the backend is invisible to
the dispatch -- the user keeps the default TBE / LaMa / ProPainter
(hybrid) path.

Backends shipped here (all are opt-in scaffolds calling the upstream
reference repo when present; full integration with each model's quirky
input formats is the upstream project's concern):

- RM-59 ProPainterReal (the ICCV 2023 sczhou/ProPainter reference).
- RM-60 DiffuEraserBackend (lixiaowen-xw/DiffuEraser).
- RM-61 VaceBackend (ali-vilab/VACE 1.3B MV2V).
- RM-62 VideoPainterBackend.
- RM-63 CoCoCoBackend (text-guided AAAI 2025).
- RM-64 EraserDiTBackend (track only; experimental).
- RM-65 FloedBackend (NevSNev/FloED, flow-guided efficient diffusion).
- VOID (Netflix research, opt-in and strict-local weights only).
- CLEAR / SEDiT mask-free subtitle erasure are tracked as research
  benchmark candidates only; they are deliberately not registered as
  inpainter modes here.

The registry mode names live alongside the four core inpainters from
inpainter_registry. Each scaffold falls back to TBE + cv2 inpainting
when the heavy model fails so the user always gets a result.
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
from typing import List, Optional

import cv2
import numpy as np

from backend.inpainter_registry import register
from backend.config import ProcessingConfig
from backend.inpainters import (
    BaseInpainter,
    _cv2_inpaint,
    _edge_ring_color_correct,
    _feather_blend,
    _temporal_background_expose,
)
from backend.safe_image import safe_imread

logger = logging.getLogger(__name__)

VACE_DEFAULT_REPO_ID = "Wan-AI/Wan2.1-VACE-1.3B"


MASK_FREE_RESEARCH_ADAPTERS = {
    "clear": {
        "display_name": "CLEAR",
        "env_var": "VSR_CLEAR_WEIGHTS",
        "adapter_manifest": "clear-maskfree",
        "benchmark_category": "mask_free_subtitle",
        "source_url": "https://huggingface.co/joeyz0z/CLEAR",
        "default_registered": False,
    },
    "sedit": {
        "display_name": "SEDiT",
        "env_var": "VSR_SEDIT_WEIGHTS",
        "adapter_manifest": "sedit-maskfree",
        "benchmark_category": "mask_free_subtitle",
        "source_url": "https://arxiv.org/abs/2509.18774",
        "default_registered": False,
    },
}


def mask_free_research_adapter_specs() -> List[dict]:
    """Return mask-free subtitle-erasure research candidates.

    These specs are for benchmark/reporting code only. They are not included in
    ``_OPT_INS`` because no mask-free adapter is production-ready in VSR yet.
    """
    return [dict(value) for _key, value in sorted(MASK_FREE_RESEARCH_ADAPTERS.items())]


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _fallback_to_tbe(config: ProcessingConfig,
                     frames: List[np.ndarray],
                     masks: List[np.ndarray]) -> List[np.ndarray]:
    """Common fallback when a diffusion backend's heavy model is
    unavailable. We delegate to the TBE primitive so the user still
    gets a clean output."""
    if config.tbe_enable and len(frames) > 1:
        return _temporal_background_expose(
            frames, masks,
            min_coverage=max(2, config.tbe_min_coverage + 1),
            use_median=True,
            feather_px=config.mask_feather_px,
            edge_ring_px=config.edge_ring_px,
            flow_warp=config.tbe_flow_warp,
            scene_cut_split=config.tbe_scene_cut_split,
            scene_cut_threshold=config.tbe_scene_cut_threshold,
            scene_cut_use_pyscenedetect=config.tbe_scene_cut_use_pyscenedetect,
            scene_cut_use_transnetv2=config.tbe_scene_cut_use_transnetv2,
        )
    out: List[np.ndarray] = []
    for f, m in zip(frames, masks):
        filled = _cv2_inpaint(f, m, 5, cv2.INPAINT_TELEA)
        if config.edge_ring_px > 0:
            filled = _edge_ring_color_correct(f, filled, m, config.edge_ring_px)
        out.append(_feather_blend(f, filled, m, config.mask_feather_px))
    return out


class _DiffusionBackendBase(BaseInpainter):
    """Common scaffolding -- subclasses set ``MODE_NAME``, ``REPO_HINT``,
    and implement ``_run_model``. When the optional dep / weights are
    missing we log once and route to the TBE primitive."""

    MODE_NAME = "diffusion"
    REPO_HINT = ""

    def __init__(self, device: str, config: ProcessingConfig):
        self.device = device
        self.config = config
        self._loaded = False
        self._model = None
        self._warned = False

    def _load(self):  # subclasses override
        return None

    def _run_model(self, frames, masks):  # subclasses override
        raise NotImplementedError

    def inpaint(self, frames: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
        if not self._loaded:
            self._model = self._load()
            self._loaded = True
            if self._model is None and not self._warned:
                logger.info(
                    f"{self.MODE_NAME} weights/deps unavailable; falling "
                    f"back to TBE. {self.REPO_HINT}"
                )
                self._warned = True
        if self._model is None:
            return _fallback_to_tbe(self.config, frames, masks)
        try:
            return self._run_model(frames, masks)
        except Exception as exc:
            logger.warning(
                f"{self.MODE_NAME} inference failed ({exc}); falling back to TBE"
            )
            return _fallback_to_tbe(self.config, frames, masks)


# ---------------------------------------------------------------------------
# RM-59: Real ProPainter (ICCV 2023 reference)
# ---------------------------------------------------------------------------


class _PropainterRealBackend(_DiffusionBackendBase):
    MODE_NAME = "propainter-real"
    REPO_HINT = (
        "Install via `pip install propainter` or clone "
        "github.com/sczhou/ProPainter and set VSR_PROPAINTER_REAL=1."
    )

    def _load(self):
        try:
            import propainter  # type: ignore
            return propainter
        except ImportError:
            return None

    def _run_model(self, frames, masks):
        # ProPainter expects (T, H, W, 3) uint8 frames and (T, H, W)
        # bool masks at consistent dimensions. We adapt the call when
        # the package exposes a `propainter.inpaint(frames, masks)`
        # entry; if not, fall back to TBE.
        if not hasattr(self._model, "inpaint"):
            raise RuntimeError("propainter package missing top-level `inpaint`")
        frames_arr = np.stack(frames, axis=0)
        masks_arr = np.stack([(m > 0).astype(np.uint8) for m in masks], axis=0)
        out = self._model.inpaint(frames_arr, masks_arr)
        results = []
        for i, (f, m) in enumerate(zip(frames, masks)):
            results.append(
                _feather_blend(f, out[i], m, self.config.mask_feather_px)
            )
        return results


# ---------------------------------------------------------------------------
# RM-60: DiffuEraser
# ---------------------------------------------------------------------------


class _DiffuEraserBackend(_DiffusionBackendBase):
    MODE_NAME = "diffueraser"
    REPO_HINT = (
        "Install via `pip install diffueraser` or clone "
        "github.com/lixiaowen-xw/DiffuEraser and set VSR_DIFFUERASER=1."
    )

    def _load(self):
        try:
            from diffueraser import DiffuEraser  # type: ignore
        except ImportError:
            return None
        try:
            return DiffuEraser(device=self.device)
        except Exception:
            return None

    def _run_model(self, frames, masks):
        result = self._model.run(frames, masks)
        out = []
        for f, r, m in zip(frames, result, masks):
            out.append(_feather_blend(f, r, m, self.config.mask_feather_px))
        return out


# ---------------------------------------------------------------------------
# RM-61: Wan2.1-VACE
# ---------------------------------------------------------------------------


class _VaceBackend(_DiffusionBackendBase):
    MODE_NAME = "vace"
    REPO_HINT = (
        "Set VSR_VACE_CKPT_DIR to a reviewed Wan2.1-VACE-1.3B snapshot, "
        "or set VSR_VACE_AUTO_FETCH=1 with huggingface_hub installed."
    )

    def _load(self):
        ckpt_dir = _resolve_vace_checkpoint_dir()
        if ckpt_dir is None:
            return None
        try:
            from vace import VACE  # type: ignore
        except Exception:
            VACE = None
        if VACE is not None:
            for kwargs in (
                {"ckpt_dir": str(ckpt_dir), "device": self.device},
                {"model_dir": str(ckpt_dir), "device": self.device},
                {"checkpoint_dir": str(ckpt_dir), "device": self.device},
                {"device": self.device},
                {},
            ):
                try:
                    model = VACE(**kwargs)
                    try:
                        setattr(model, "vsr_ckpt_dir", str(ckpt_dir))
                    except Exception:
                        pass
                    return model
                except TypeError:
                    continue
                except Exception:
                    logger.debug("VACE constructor failed", exc_info=True)
                    return None
        try:
            from vace.vace_wan_inference import main as wan_main  # type: ignore
            return _VaceWanScriptAdapter(wan_main, ckpt_dir)
        except Exception:
            logger.debug("VACE Wan inference entrypoint unavailable",
                         exc_info=True)
        return None

    def _run_model(self, frames, masks):
        prompt = os.environ.get(
            "VSR_VACE_PROMPT",
            "remove subtitles and reconstruct the original background",
        )
        out = _call_vace_model(self._model, frames, masks, prompt)
        out = _coerce_adapter_frames(out, len(frames), "VACE")
        return [
            _feather_blend(f, r, m, self.config.mask_feather_px)
            for f, r, m in zip(frames, out, masks)
        ]


def _vace_cache_dir(env=None) -> Path:
    source = os.environ if env is None else env
    appdata = str(source.get("APPDATA", "") or "").strip()
    if appdata:
        return (Path(appdata) / "VideoSubtitleRemoverPro" / "models"
                / "vace" / "Wan2.1-VACE-1.3B")
    home = str(source.get("USERPROFILE") or source.get("HOME") or "").strip()
    base = Path(home) if home else Path.home()
    return (base / ".cache" / "VideoSubtitleRemoverPro" / "models"
            / "vace" / "Wan2.1-VACE-1.3B")


def _configured_vace_path(env=None) -> Optional[Path]:
    source = os.environ if env is None else env
    for key in ("VSR_VACE_CKPT_DIR", "VSR_VACE_MODEL_DIR", "VSR_VACE_WEIGHTS"):
        value = str(source.get(key, "") or "").strip()
        if value:
            return Path(value)
    return None


def _verify_vace_checkpoint_path(path: Path) -> bool:
    try:
        from backend.adapter_manifest import (
            log_adapter_verification,
            verify_adapter_path,
        )
        result = verify_adapter_path("vace-wan13b", str(path))
        log_adapter_verification(result)
        return bool(result.allowed)
    except Exception as exc:
        logger.warning(f"VACE checkpoint verification failed: {exc}")
        return False


def _resolve_vace_checkpoint_dir(env=None, *, auto_fetch: bool = True) -> Optional[Path]:
    source = os.environ if env is None else env
    configured = _configured_vace_path(source)
    if configured is not None and configured.exists():
        candidate = configured.parent if configured.is_file() else configured
        return candidate if _verify_vace_checkpoint_path(candidate) else None

    auto_fetch_enabled = (
        str(source.get("VSR_VACE_AUTO_FETCH", "") or "").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    if not auto_fetch or not auto_fetch_enabled:
        logger.info(
            "VACE enabled but no checkpoint directory is configured. Set "
            "VSR_VACE_CKPT_DIR or VSR_VACE_AUTO_FETCH=1."
        )
        return None

    repo_id = str(source.get("VSR_VACE_REPO_ID") or VACE_DEFAULT_REPO_ID)
    revision = str(source.get("VSR_VACE_REVISION") or "").strip() or None
    local_dir = configured if configured is not None else _vace_cache_dir(source)
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception:
        logger.warning(
            "VSR_VACE_AUTO_FETCH=1 requires huggingface_hub. Install "
            "`huggingface-hub` or set VSR_VACE_CKPT_DIR to a local snapshot."
        )
        return None
    try:
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        candidate = Path(snapshot)
        return candidate if _verify_vace_checkpoint_path(candidate) else None
    except Exception as exc:
        logger.warning(f"VACE checkpoint auto-fetch failed: {exc}")
        return None


class _VaceWanScriptAdapter:
    """Bridge VSR frame batches into the upstream VACE Wan MV2V entrypoint."""

    def __init__(self, main_fn, ckpt_dir: Path):
        self._main_fn = main_fn
        self._ckpt_dir = ckpt_dir

    def mv2v(self, frames=None, masks=None, prompt=None):
        frame_list = [] if frames is None else list(frames)
        mask_list = [] if masks is None else list(masks)
        return _run_vace_wan_script(
            self._main_fn,
            self._ckpt_dir,
            frame_list,
            mask_list,
            prompt or "remove subtitles and reconstruct the original background",
        )


def _vace_bool_env(name: str):
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return None
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _vace_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _vace_float_env(name: str, default):
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _vace_fps() -> float:
    fps = _vace_float_env("VSR_VACE_FPS", 16.0)
    if fps is None or not np.isfinite(float(fps)) or float(fps) <= 0:
        return 16.0
    return float(fps)


def _vace_padded_count(count: int) -> int:
    if count <= 0:
        return 0
    remainder = count % 4
    return count if remainder == 1 else count + ((1 - remainder) % 4)


def _pad_vace_inputs(frames: List[np.ndarray],
                     masks: List[np.ndarray]) -> tuple[List[np.ndarray], List[np.ndarray]]:
    target = _vace_padded_count(len(frames))
    if target <= len(frames):
        return list(frames), list(masks)
    padded_frames = list(frames)
    padded_masks = list(masks)
    while len(padded_frames) < target:
        padded_frames.append(np.asarray(frames[-1]).copy())
        padded_masks.append(np.asarray(masks[-1]).copy())
    return padded_frames, padded_masks


def _write_adapter_video(path: Path,
                         frames: List[np.ndarray],
                         *,
                         fps: float,
    mask: bool = False) -> None:
    if not frames:
        raise RuntimeError("adapter input frame list is empty")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create adapter input video: {path}")
    try:
        for item in frames:
            if mask:
                frame = np.asarray(item)
                if frame.ndim == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = np.where(frame > 0, 255, 0).astype(np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                frame = np.asarray(item).astype(np.uint8)
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height),
                                   interpolation=cv2.INTER_NEAREST)
            writer.write(frame)
    finally:
        writer.release()


def _read_adapter_output_video(path: Path,
                               expected_count: int,
                               target_shape) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"adapter output video could not be opened: {path}")
    frames: List[np.ndarray] = []
    try:
        while len(frames) < expected_count:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[:2] != target_shape[:2]:
                frame = cv2.resize(frame, (target_shape[1], target_shape[0]),
                                   interpolation=cv2.INTER_LINEAR)
            frames.append(frame.astype(np.uint8))
    finally:
        cap.release()
    if len(frames) != expected_count:
        raise RuntimeError("adapter output frame count does not match input")
    return frames


def _run_vace_wan_script(main_fn,
                         ckpt_dir: Path,
                         frames: List[np.ndarray],
                         masks: List[np.ndarray],
                         prompt: str) -> List[np.ndarray]:
    if not frames or not masks:
        return []
    frame_count = min(len(frames), len(masks))
    frames = frames[:frame_count]
    masks = masks[:frame_count]
    padded_frames, padded_masks = _pad_vace_inputs(frames, masks)
    with tempfile.TemporaryDirectory(prefix="vsr_vace_") as tmpdir:
        work = Path(tmpdir)
        src_video = work / "src_video.mp4"
        src_mask = work / "src_mask.mp4"
        out_video = work / "out_video.mp4"
        save_dir = work / "results"
        fps = _vace_fps()
        _write_adapter_video(src_video, padded_frames, fps=fps)
        _write_adapter_video(src_mask, padded_masks, fps=fps, mask=True)
        args = {
            "model_name": os.environ.get("VSR_VACE_MODEL_NAME", "vace-1.3B"),
            "size": os.environ.get("VSR_VACE_SIZE", "480p"),
            "frame_num": len(padded_frames),
            "ckpt_dir": str(ckpt_dir),
            "offload_model": _vace_bool_env("VSR_VACE_OFFLOAD_MODEL"),
            "ulysses_size": _vace_int_env("VSR_VACE_ULYSSES_SIZE", 1),
            "ring_size": _vace_int_env("VSR_VACE_RING_SIZE", 1),
            "t5_fsdp": bool(_vace_bool_env("VSR_VACE_T5_FSDP") or False),
            "t5_cpu": bool(_vace_bool_env("VSR_VACE_T5_CPU") or False),
            "dit_fsdp": bool(_vace_bool_env("VSR_VACE_DIT_FSDP") or False),
            "save_dir": str(save_dir),
            "save_file": str(out_video),
            "src_video": str(src_video),
            "src_mask": str(src_mask),
            "src_ref_images": None,
            "prompt": prompt,
            "use_prompt_extend": os.environ.get("VSR_VACE_PROMPT_EXTEND", "plain"),
            "base_seed": _vace_int_env("VSR_VACE_SEED", 2025),
            "sample_solver": os.environ.get("VSR_VACE_SAMPLE_SOLVER", "unipc"),
            "sample_steps": _vace_int_env("VSR_VACE_SAMPLE_STEPS", 50),
            "sample_shift": _vace_float_env("VSR_VACE_SAMPLE_SHIFT", 16.0),
            "sample_guide_scale": _vace_float_env(
                "VSR_VACE_GUIDE_SCALE", 5.0),
        }
        result = main_fn(args)
        if isinstance(result, dict) and result.get("out_video"):
            out_video = Path(str(result["out_video"]))
        return _read_adapter_output_video(out_video, frame_count, frames[0].shape)


def _coerce_adapter_frames(value, expected_count: int, label: str) -> List[np.ndarray]:
    if isinstance(value, dict):
        for key in ("frames", "video", "output", "outputs"):
            if key in value:
                value = value[key]
                break
    if isinstance(value, np.ndarray):
        if value.ndim == 4:
            frames = [np.asarray(frame).astype(np.uint8) for frame in value]
            if len(frames) != expected_count:
                raise RuntimeError(f"{label} output frame count does not match input")
            return frames
        raise RuntimeError(f"{label} output ndarray must be T,H,W,C")
    if not isinstance(value, (list, tuple)):
        raise RuntimeError(f"{label} output must be a frame list or ndarray")
    frames = [np.asarray(frame).astype(np.uint8) for frame in value]
    if len(frames) != expected_count:
        raise RuntimeError(f"{label} output frame count does not match input")
    return frames


def _call_vace_model(model, frames, masks, prompt: str):
    for name in ("mv2v", "inpaint", "run"):
        fn = getattr(model, name, None)
        if fn is None:
            continue
        for args, kwargs in (
            ((), {"frames": frames, "masks": masks, "prompt": prompt}),
            ((frames, masks), {"prompt": prompt}),
            ((frames, masks), {}),
        ):
            try:
                return fn(*args, **kwargs)
            except TypeError:
                continue
    raise RuntimeError("VACE package missing mv2v/inpaint/run frame API")


# ---------------------------------------------------------------------------
# RM-62: VideoPainter
# ---------------------------------------------------------------------------


VIDEOPAINTER_DEFAULT_REPO_ID = "TencentARC/VideoPainter"
VIDEOPAINTER_BASE_REPO_ID = "THUDM/CogVideoX-5b-I2V"


def _strip_wrapping_quotes(value: str) -> str:
    text = str(value)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1]
    return text


def _split_adapter_command(command: str) -> List[str]:
    try:
        parts = shlex.split(command, posix=(os.name != "nt"))
    except ValueError:
        return []
    return [_strip_wrapping_quotes(part) for part in parts if part]


def _adapter_command_from_env(env_name: str) -> Optional[List[str]]:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return None
    parts = _split_adapter_command(raw)
    if not parts:
        logger.warning("%s command could not be parsed", env_name)
        return None
    executable = parts[0]
    if shutil.which(executable) is None and not Path(executable).is_file():
        logger.warning("%s command not found: %s", env_name, executable)
        return None
    return parts


def _adapter_config_dict(config: ProcessingConfig) -> dict:
    try:
        return config.to_dict() if hasattr(config, "to_dict") else dict(config)
    except Exception:
        return {}


def _adapter_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _adapter_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(value) or value <= 0:
        return default
    return value


def _configured_videopainter_path(env=None) -> Optional[Path]:
    source = os.environ if env is None else env
    for key in (
        "VSR_VIDEOPAINTER_CKPT_DIR",
        "VSR_VIDEOPAINTER_MODEL_DIR",
        "VSR_VIDEOPAINTER_WEIGHTS",
        "VSR_VIDEOPAINTER_BRANCH_DIR",
    ):
        value = str(source.get(key, "") or "").strip()
        if value:
            return Path(value)
    return None


def _videopainter_cache_dir(env=None) -> Path:
    source = os.environ if env is None else env
    appdata = str(source.get("APPDATA", "") or "").strip()
    if appdata:
        return (Path(appdata) / "VideoSubtitleRemoverPro" / "models"
                / "videopainter")
    home = str(source.get("USERPROFILE") or source.get("HOME") or "").strip()
    base = Path(home) if home else Path.home()
    return base / ".cache" / "VideoSubtitleRemoverPro" / "models" / "videopainter"


def _verify_videopainter_checkpoint_path(path: Path) -> bool:
    try:
        from backend.adapter_manifest import (
            log_adapter_verification,
            verify_adapter_path,
        )
        result = verify_adapter_path("videopainter", str(path))
        log_adapter_verification(result)
        return bool(result.allowed)
    except Exception as exc:
        logger.warning(f"VideoPainter checkpoint verification failed: {exc}")
        return False


def _resolve_videopainter_checkpoint_dir(env=None) -> Optional[Path]:
    source = os.environ if env is None else env
    configured = _configured_videopainter_path(source)
    candidates: List[Path] = []
    if configured is not None:
        candidates.append(configured.parent if configured.is_file() else configured)
    candidates.append(_videopainter_cache_dir(source))
    for candidate in candidates:
        if candidate.exists() and _verify_videopainter_checkpoint_path(candidate):
            return candidate
    logger.info(
        "VideoPainter enabled but no verified local checkpoint directory is "
        "available. Set VSR_VIDEOPAINTER_CKPT_DIR after reviewing upstream "
        "licenses and use VSR_ALLOW_UNVERIFIED_MODELS=1 for unpinned research "
        "weights."
    )
    return None


def _first_existing_path(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _videopainter_paths(root: Path, env=None) -> dict:
    source = os.environ if env is None else env
    branch = str(source.get("VSR_VIDEOPAINTER_BRANCH_DIR", "") or "").strip()
    base = str(source.get("VSR_COGVIDEOX_MODEL_DIR", "") or "").strip()
    ident = str(source.get("VSR_VIDEOPAINTER_ID_DIR", "") or "").strip()
    branch_path = Path(branch) if branch else (
        _first_existing_path([
            root / "VideoPainter" / "checkpoints" / "branch",
            root / "checkpoints" / "branch",
            root / "branch",
        ]) or root / "VideoPainter" / "checkpoints" / "branch"
    )
    base_path = Path(base) if base else (
        _first_existing_path([
            root / "CogVideoX-5b-I2V",
            root / "THUDM" / "CogVideoX-5b-I2V",
        ]) or root / "CogVideoX-5b-I2V"
    )
    id_path = Path(ident) if ident else (
        _first_existing_path([
            root / "VideoPainterID" / "checkpoints",
            root / "VideoPainterID",
        ])
    )
    return {
        "model_dir": root,
        "branch_dir": branch_path,
        "base_model_dir": base_path,
        "id_adapter_dir": id_path,
    }


class _VideoPainterCommandAdapter:
    """Run a reviewed local VideoPainter wrapper command over temp videos."""

    def __init__(self, command: List[str], paths: dict, config: ProcessingConfig):
        self._command = command
        self._paths = paths
        self._config = config
        self._timeout = _adapter_int_env("VSR_VIDEOPAINTER_TIMEOUT", 7200)

    def inpaint(self, frames=None, masks=None, prompt=None):
        frame_list = [] if frames is None else list(frames)
        mask_list = [] if masks is None else list(masks)
        return _run_videopainter_command(
            self._command,
            self._paths,
            self._config,
            frame_list,
            mask_list,
            prompt or "remove subtitles and reconstruct the original background",
            self._timeout,
        )


def _run_videopainter_command(command: List[str],
                              paths: dict,
                              config: ProcessingConfig,
                              frames: List[np.ndarray],
                              masks: List[np.ndarray],
                              prompt: str,
                              timeout: int) -> List[np.ndarray]:
    if not frames or not masks:
        return []
    frame_count = min(len(frames), len(masks))
    frames = frames[:frame_count]
    masks = masks[:frame_count]
    with tempfile.TemporaryDirectory(prefix="vsr_videopainter_") as tmpdir:
        work = Path(tmpdir)
        src_video = work / "src_video.mp4"
        src_mask = work / "src_mask.mp4"
        out_video = work / "out_video.mp4"
        config_path = work / "config.json"
        fps = _adapter_float_env("VSR_VIDEOPAINTER_FPS", 8.0)
        _write_adapter_video(src_video, frames, fps=fps)
        _write_adapter_video(src_mask, masks, fps=fps, mask=True)
        payload = {
            "adapter": "videopainter",
            "prompt": prompt,
            "fps": fps,
            "frameCount": frame_count,
            "modelDir": str(paths["model_dir"]),
            "branchDir": str(paths["branch_dir"]),
            "baseModelDir": str(paths["base_model_dir"]),
            "idAdapterDir": (
                str(paths["id_adapter_dir"])
                if paths.get("id_adapter_dir") is not None else None
            ),
            "processingConfig": _adapter_config_dict(config),
        }
        config_path.write_text(json.dumps(payload, ensure_ascii=True),
                               encoding="utf-8")
        cmd = list(command) + [
            "--input-video", str(src_video),
            "--mask-video", str(src_mask),
            "--output-video", str(out_video),
            "--model-dir", str(paths["model_dir"]),
            "--branch-dir", str(paths["branch_dir"]),
            "--base-model-dir", str(paths["base_model_dir"]),
            "--prompt", prompt,
            "--config", str(config_path),
        ]
        if paths.get("id_adapter_dir") is not None:
            cmd.extend(["--id-adapter-dir", str(paths["id_adapter_dir"])])
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=max(1, int(timeout)),
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"VideoPainter command timed out after {timeout}s"
            ) from exc
        except OSError as exc:
            raise RuntimeError(f"VideoPainter command failed to start: {exc}") from exc
        if result.returncode != 0:
            stderr = (result.stderr or "")[-600:]
            raise RuntimeError(
                f"VideoPainter command exited {result.returncode}: {stderr}"
            )
        output_path = out_video
        stdout = (result.stdout or "").strip()
        if stdout.startswith("{"):
            try:
                data = json.loads(stdout)
                if data.get("outputVideo"):
                    output_path = Path(str(data["outputVideo"]))
            except Exception:
                pass
        return _read_adapter_output_video(output_path, frame_count, frames[0].shape)


def _call_videopainter_model(model, frames, masks, prompt: str):
    for name in ("inpaint", "run", "mv2v"):
        fn = getattr(model, name, None)
        if fn is None:
            continue
        for args, kwargs in (
            ((), {"frames": frames, "masks": masks, "prompt": prompt}),
            ((frames, masks), {"prompt": prompt}),
            ((frames, masks), {}),
        ):
            try:
                return fn(*args, **kwargs)
            except TypeError:
                continue
    raise RuntimeError("VideoPainter package missing inpaint/run/mv2v frame API")


class _VideoPainterBackend(_DiffusionBackendBase):
    MODE_NAME = "videopainter"
    REPO_HINT = (
        "Set VSR_VIDEOPAINTER_CKPT_DIR plus VSR_VIDEOPAINTER_COMMAND to a "
        "reviewed local VideoPainter wrapper, or install a compatible local "
        "videopainter package. The upstream sample is research/non-commercial."
    )

    def _load(self):
        root = _resolve_videopainter_checkpoint_dir()
        if root is None:
            return None
        paths = _videopainter_paths(root)
        command = _adapter_command_from_env("VSR_VIDEOPAINTER_COMMAND")
        if command is not None:
            return _VideoPainterCommandAdapter(command, paths, self.config)
        try:
            from videopainter import VideoPainter  # type: ignore
        except Exception:
            return None
        for kwargs in (
            {
                "model_dir": str(paths["model_dir"]),
                "branch_dir": str(paths["branch_dir"]),
                "base_model_dir": str(paths["base_model_dir"]),
                "device": self.device,
            },
            {"ckpt_dir": str(root), "device": self.device},
            {"device": self.device},
            {},
        ):
            try:
                return VideoPainter(**kwargs)
            except TypeError:
                continue
            except Exception:
                logger.debug("VideoPainter constructor failed", exc_info=True)
                return None
        return None

    def _run_model(self, frames, masks):
        prompt = os.environ.get(
            "VSR_VIDEOPAINTER_PROMPT",
            "remove subtitles and reconstruct the original background",
        )
        out = _call_videopainter_model(self._model, frames, masks, prompt)
        out = _coerce_adapter_frames(out, len(frames), "VideoPainter")
        return [
            _feather_blend(f, r, m, self.config.mask_feather_px)
            for f, r, m in zip(frames, out, masks)
        ]


# ---------------------------------------------------------------------------
# RM-63: CoCoCo (text-guided)
# ---------------------------------------------------------------------------


class _CocoCoBackend(_DiffusionBackendBase):
    MODE_NAME = "cococo"
    REPO_HINT = (
        "Install via the upstream COCOCO project and set VSR_COCOCO=1."
    )

    def _load(self):
        try:
            from cococo import CoCoCo  # type: ignore
            return CoCoCo(device=self.device)
        except Exception:
            return None

    def _run_model(self, frames, masks):
        # Text prompt defaults to "background" so the model fills the
        # subtitle region with the surrounding scene rather than
        # generating arbitrary content. Users who need a custom prompt
        # set VSR_COCOCO_PROMPT.
        prompt = os.environ.get("VSR_COCOCO_PROMPT", "background")
        out = self._model.inpaint(frames, masks, prompt=prompt)
        return [
            _feather_blend(f, r, m, self.config.mask_feather_px)
            for f, r, m in zip(frames, out, masks)
        ]


# ---------------------------------------------------------------------------
# RM-64: EraserDiT (track only)
# ---------------------------------------------------------------------------


class _EraserDitBackend(_DiffusionBackendBase):
    MODE_NAME = "eraserdit"
    REPO_HINT = (
        "EraserDiT is research-stage; integration is opt-in via "
        "VSR_ERASERDIT=1 once the upstream releases a pip-installable "
        "package."
    )

    def _load(self):
        try:
            from eraserdit import EraserDiT  # type: ignore
            return EraserDiT(device=self.device)
        except Exception:
            return None

    def _run_model(self, frames, masks):
        out = self._model.inpaint(frames, masks)
        return [
            _feather_blend(f, r, m, self.config.mask_feather_px)
            for f, r, m in zip(frames, out, masks)
        ]


# ---------------------------------------------------------------------------
# RM-65: FloED
# ---------------------------------------------------------------------------


def _configured_floed_path(env=None) -> Optional[Path]:
    source = os.environ if env is None else env
    for key in ("VSR_FLOED_WEIGHTS", "VSR_FLOED_CKPT", "VSR_FLOED_CKPT_DIR"):
        value = str(source.get(key, "") or "").strip()
        if value:
            return Path(value)
    return None


def _floed_cache_dir(env=None) -> Path:
    source = os.environ if env is None else env
    appdata = str(source.get("APPDATA", "") or "").strip()
    if appdata:
        return (Path(appdata) / "VideoSubtitleRemoverPro" / "models"
                / "floed")
    home = str(source.get("USERPROFILE") or source.get("HOME") or "").strip()
    base = Path(home) if home else Path.home()
    return base / ".cache" / "VideoSubtitleRemoverPro" / "models" / "floed"


def _verify_floed_checkpoint_path(path: Path) -> bool:
    try:
        from backend.adapter_manifest import (
            log_adapter_verification,
            verify_adapter_path,
        )
        result = verify_adapter_path("floed", str(path))
        log_adapter_verification(result)
        return bool(result.allowed)
    except Exception as exc:
        logger.warning(f"FloED checkpoint verification failed: {exc}")
        return False


def _resolve_floed_checkpoint_path(env=None) -> Optional[Path]:
    source = os.environ if env is None else env
    configured = _configured_floed_path(source)
    candidates: List[Path] = []
    if configured is not None:
        candidates.append(configured)
    candidates.append(_floed_cache_dir(source))
    for candidate in candidates:
        if candidate.exists() and _verify_floed_checkpoint_path(candidate):
            return candidate
    logger.info(
        "FloED enabled but no verified local checkpoint is available. Set "
        "VSR_FLOED_WEIGHTS or VSR_FLOED_CKPT_DIR after reviewing the upstream "
        "Apache-2.0 repo and use VSR_ALLOW_UNVERIFIED_MODELS=1 for unpinned "
        "research weights."
    )
    return None


def _write_adapter_frame_dirs(frames: List[np.ndarray],
                              masks: List[np.ndarray],
                              work: Path) -> tuple[Path, Path, List[str]]:
    in_dir = work / "input"
    mask_dir = work / "masks"
    in_dir.mkdir()
    mask_dir.mkdir()
    names: List[str] = []
    for idx, (frame, mask) in enumerate(zip(frames, masks)):
        name = f"{idx:06d}.png"
        names.append(name)
        frame_arr = np.asarray(frame).astype(np.uint8)
        mask_arr = np.asarray(mask)
        if mask_arr.ndim == 3:
            mask_arr = cv2.cvtColor(mask_arr, cv2.COLOR_BGR2GRAY)
        mask_arr = np.where(mask_arr > 0, 255, 0).astype(np.uint8)
        if not cv2.imwrite(str(in_dir / name), frame_arr):
            raise RuntimeError(f"could not write adapter input frame: {name}")
        if not cv2.imwrite(str(mask_dir / name), mask_arr):
            raise RuntimeError(f"could not write adapter mask frame: {name}")
    return in_dir, mask_dir, names


def _read_adapter_frame_dir(out_dir: Path,
                            names: List[str],
                            target_frames: List[np.ndarray]) -> List[np.ndarray]:
    output: List[np.ndarray] = []
    for name, original in zip(names, target_frames):
        path = out_dir / name
        frame = safe_imread(path)
        if frame is None:
            raise RuntimeError(f"adapter output frame is missing: {path}")
        if frame.shape[:2] != original.shape[:2]:
            frame = cv2.resize(frame, (original.shape[1], original.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
        output.append(frame.astype(np.uint8))
    return output


class _FloedCommandAdapter:
    """Run a reviewed local FloED wrapper command over frame directories."""

    def __init__(self, command: List[str], weights: Path, config: ProcessingConfig):
        self._command = command
        self._weights = weights
        self._config = config
        self._timeout = _adapter_int_env("VSR_FLOED_TIMEOUT", 7200)

    def inpaint(self, frames=None, masks=None, prompt=None):
        frame_list = [] if frames is None else list(frames)
        mask_list = [] if masks is None else list(masks)
        return _run_floed_command(
            self._command,
            self._weights,
            self._config,
            frame_list,
            mask_list,
            prompt or "remove subtitles and reconstruct the original background",
            self._timeout,
        )


def _run_floed_command(command: List[str],
                       weights: Path,
                       config: ProcessingConfig,
                       frames: List[np.ndarray],
                       masks: List[np.ndarray],
                       prompt: str,
                       timeout: int) -> List[np.ndarray]:
    if not frames or not masks:
        return []
    frame_count = min(len(frames), len(masks))
    frames = frames[:frame_count]
    masks = masks[:frame_count]
    with tempfile.TemporaryDirectory(prefix="vsr_floed_") as tmpdir:
        work = Path(tmpdir)
        out_dir = work / "output"
        out_dir.mkdir()
        input_dir, mask_dir, names = _write_adapter_frame_dirs(frames, masks, work)
        config_path = work / "config.json"
        payload = {
            "adapter": "floed",
            "prompt": prompt,
            "frameCount": frame_count,
            "weights": str(weights),
            "processingConfig": _adapter_config_dict(config),
        }
        config_path.write_text(json.dumps(payload, ensure_ascii=True),
                               encoding="utf-8")
        cmd = list(command) + [
            "--input-dir", str(input_dir),
            "--mask-dir", str(mask_dir),
            "--output-dir", str(out_dir),
            "--weights", str(weights),
            "--prompt", prompt,
            "--config", str(config_path),
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=max(1, int(timeout)),
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"FloED command timed out after {timeout}s") from exc
        except OSError as exc:
            raise RuntimeError(f"FloED command failed to start: {exc}") from exc
        if result.returncode != 0:
            stderr = (result.stderr or "")[-600:]
            raise RuntimeError(f"FloED command exited {result.returncode}: {stderr}")
        stdout = (result.stdout or "").strip()
        if stdout.startswith("{"):
            try:
                data = json.loads(stdout)
                if data.get("outputDir"):
                    out_dir = Path(str(data["outputDir"]))
            except Exception:
                pass
        return _read_adapter_frame_dir(out_dir, names, frames)


def _call_floed_model(model, frames, masks, prompt: str):
    for name in ("inpaint", "run", "mv2v"):
        fn = getattr(model, name, None)
        if fn is None:
            continue
        for args, kwargs in (
            ((), {"frames": frames, "masks": masks, "prompt": prompt}),
            ((frames, masks), {"prompt": prompt}),
            ((frames, masks), {}),
        ):
            try:
                return fn(*args, **kwargs)
            except TypeError:
                continue
    raise RuntimeError("FloED package missing inpaint/run/mv2v frame API")


class _FloedBackend(_DiffusionBackendBase):
    MODE_NAME = "floed"
    REPO_HINT = (
        "Set VSR_FLOED_WEIGHTS plus VSR_FLOED_COMMAND to a reviewed local "
        "FloED wrapper, or install a compatible local floed package."
    )

    def _load(self):
        weights = _resolve_floed_checkpoint_path()
        if weights is None:
            return None
        command = _adapter_command_from_env("VSR_FLOED_COMMAND")
        if command is not None:
            return _FloedCommandAdapter(command, weights, self.config)
        try:
            from floed import FloED  # type: ignore
        except Exception:
            return None
        for kwargs in (
            {"weights": str(weights), "device": self.device},
            {"ckpt_path": str(weights), "device": self.device},
            {"checkpoint": str(weights), "device": self.device},
            {"device": self.device},
            {},
        ):
            try:
                return FloED(**kwargs)
            except TypeError:
                continue
            except Exception:
                logger.debug("FloED constructor failed", exc_info=True)
                return None
        return None

    def _run_model(self, frames, masks):
        prompt = os.environ.get(
            "VSR_FLOED_PROMPT",
            "remove subtitles and reconstruct the original background",
        )
        out = _call_floed_model(self._model, frames, masks, prompt)
        out = _coerce_adapter_frames(out, len(frames), "FloED")
        return [
            _feather_blend(f, r, m, self.config.mask_feather_px)
            for f, r, m in zip(frames, out, masks)
        ]


def _void_weight_paths() -> Optional[List[str]]:
    primary = (
        os.environ.get("VSR_VOID_WEIGHTS", "").strip()
        or os.environ.get("VSR_VOID_PASS1", "").strip()
    )
    if not primary:
        return None
    paths = [primary]
    pass2 = os.environ.get("VSR_VOID_PASS2", "").strip()
    if pass2:
        paths.append(pass2)
    return paths


class _VoidBackend(_DiffusionBackendBase):
    MODE_NAME = "void"
    REPO_HINT = (
        "VOID is research-only. Set VSR_VOID=1 plus VSR_VOID_WEIGHTS "
        "(or VSR_VOID_PASS1/VSR_VOID_PASS2) to reviewed local checkpoints."
    )

    def _load(self):
        paths = _void_weight_paths()
        if not paths:
            logger.info(
                "VOID enabled but no local weights were configured; falling "
                "back to TBE. Set VSR_VOID_WEIGHTS or VSR_VOID_PASS1."
            )
            return None
        try:
            from backend.adapter_manifest import (
                log_adapter_verification,
                verify_adapter_path,
            )
            for path in paths:
                result = verify_adapter_path(
                    "void",
                    path,
                    strict_unknown=True,
                )
                log_adapter_verification(result)
                if not result.allowed:
                    return None
        except Exception as exc:
            logger.warning(f"VOID adapter manifest verification failed: {exc}")
            return None

        try:
            import void_model  # type: ignore
            return {"module": void_model, "weights": paths}
        except Exception:
            try:
                import void  # type: ignore
                return {"module": void, "weights": paths}
            except Exception:
                logger.info(
                    "VOID package is unavailable; install the reviewed "
                    "upstream checkout before enabling inference."
                )
                return None

    def _run_model(self, frames, masks):
        module = self._model["module"]
        kwargs = {
            "weights": self._model["weights"],
            "device": self.device,
        }
        if hasattr(module, "inpaint"):
            out = module.inpaint(frames, masks, **kwargs)
        elif hasattr(module, "run"):
            out = module.run(frames, masks, **kwargs)
        else:
            raise RuntimeError("VOID package missing `inpaint` or `run` entrypoint")
        return [
            _feather_blend(f, r, m, self.config.mask_feather_px)
            for f, r, m in zip(frames, out, masks)
        ]


# Map an env var to (mode-name, builder).
_OPT_INS = [
    ("VSR_PROPAINTER_REAL", "propainter-real", _PropainterRealBackend),
    ("VSR_DIFFUERASER",     "diffueraser",     _DiffuEraserBackend),
    ("VSR_VACE",            "vace",            _VaceBackend),
    ("VSR_VIDEOPAINTER",    "videopainter",    _VideoPainterBackend),
    ("VSR_COCOCO",          "cococo",          _CocoCoBackend),
    ("VSR_ERASERDIT",       "eraserdit",       _EraserDitBackend),
    ("VSR_FLOED",           "floed",           _FloedBackend),
    ("VSR_VOID",            "void",            _VoidBackend),
]


def maybe_register() -> List[str]:
    """Register every diffusion backend the user has opted into. Returns
    the list of mode names that were added so callers can log."""
    registered: List[str] = []
    for env_name, mode_name, klass in _OPT_INS:
        if _env_enabled(env_name):
            register(mode_name, lambda device, config, K=klass: K(device, config))
            registered.append(mode_name)
            logger.info(f"Diffusion backend '{mode_name}' enabled via {env_name}")
    return registered


maybe_register()
