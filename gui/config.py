"""Configuration, settings I/O, and preset library."""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from gui.theme import Theme

logger = logging.getLogger(__name__)

# -- App identity -----------------------------------------------------------

APP_NAME = "Video Subtitle Remover Pro"
APP_VERSION = "3.16.1"
APP_AUTHOR = "SysAdminDoc"

LOG_DIR = Path(os.environ.get("APPDATA", Path.home())) / "VideoSubtitleRemoverPro"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "vsr_pro.log"
SETTINGS_FILE = LOG_DIR / "settings.json"

VSR_SETTINGS_FORMAT = 2

# -- Enums ------------------------------------------------------------------


class InpaintMode(Enum):
    AUTO = "Auto"
    STTN = "STTN"
    LAMA = "LAMA"
    PROPAINTER = "ProPainter"


class ProcessingStatus(Enum):
    IDLE = "idle"
    LOADING = "loading"
    DETECTING = "detecting"
    PROCESSING = "processing"
    MERGING = "merging"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"


STATUS_UI = {
    ProcessingStatus.IDLE: {
        "label": "Ready",
        "color": Theme.TEXT_SECONDARY,
        "bg": Theme.BG_TERTIARY,
    },
    ProcessingStatus.LOADING: {
        "label": "Loading",
        "color": Theme.INFO,
        "bg": Theme.INFO_BG,
    },
    ProcessingStatus.DETECTING: {
        "label": "Scanning",
        "color": Theme.INFO,
        "bg": Theme.INFO_BG,
    },
    ProcessingStatus.PROCESSING: {
        "label": "Removing",
        "color": Theme.SUCCESS,
        "bg": Theme.SUCCESS_BG,
    },
    ProcessingStatus.MERGING: {
        "label": "Finishing",
        "color": Theme.WARNING,
        "bg": Theme.WARNING_BG,
    },
    ProcessingStatus.COMPLETE: {
        "label": "Complete",
        "color": Theme.SUCCESS,
        "bg": Theme.SUCCESS_BG,
    },
    ProcessingStatus.ERROR: {
        "label": "Needs Attention",
        "color": Theme.ERROR,
        "bg": Theme.ERROR_BG,
    },
    ProcessingStatus.CANCELLED: {
        "label": "Stopped",
        "color": Theme.TEXT_MUTED,
        "bg": Theme.BG_TERTIARY,
    },
}

# -- Coercion helpers -------------------------------------------------------


def _coerce_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off", ""}:
            return False
    return default


def _coerce_int(value, default: int, min_value: Optional[int] = None,
                max_value: Optional[int] = None) -> int:
    try:
        fv = float(value)
        if not math.isfinite(fv):
            raise ValueError("non-finite float")
        coerced = int(fv)
    except (TypeError, ValueError):
        coerced = default
    if min_value is not None:
        coerced = max(min_value, coerced)
    if max_value is not None:
        coerced = min(max_value, coerced)
    return coerced


def _coerce_float(value, default: float, min_value: Optional[float] = None,
                  max_value: Optional[float] = None) -> float:
    try:
        coerced = float(value)
        if not math.isfinite(coerced):
            raise ValueError("non-finite float")
    except (TypeError, ValueError):
        coerced = default
    if min_value is not None:
        coerced = max(min_value, coerced)
    if max_value is not None:
        coerced = min(max_value, coerced)
    return coerced


def _coerce_text(value, default: str, max_length: int = 256) -> str:
    if isinstance(value, str):
        text = value.strip()
        if len(text) > max_length:
            text = text[:max_length]
        return text
    return default


def _coerce_rect(value) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(float(v)) for v in value]
    except (TypeError, ValueError):
        return None
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = max(0, x2), max(0, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _coerce_rect_list(value) -> Optional[List[Tuple[int, int, int, int]]]:
    if not isinstance(value, (list, tuple)):
        return None
    rects = []
    for item in value:
        rect = _coerce_rect(item)
        if rect:
            rects.append(rect)
    return rects or None


def _coerce_gui_mode(value) -> InpaintMode:
    if isinstance(value, InpaintMode):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        mode_map = {
            "auto": InpaintMode.AUTO,
            "sttn": InpaintMode.STTN,
            "lama": InpaintMode.LAMA,
            "propainter": InpaintMode.PROPAINTER,
            "pro painter": InpaintMode.PROPAINTER,
        }
        if normalized in mode_map:
            return mode_map[normalized]
    return InpaintMode.STTN


# -- ProcessingConfig -------------------------------------------------------


@dataclass
class ProcessingConfig:
    """Configuration for subtitle removal processing."""
    mode: InpaintMode = InpaintMode.STTN
    use_gpu: bool = True
    gpu_id: int = 0

    sttn_skip_detection: bool = False
    sttn_neighbor_stride: int = 10
    sttn_reference_length: int = 10
    sttn_max_load_num: int = 30

    lama_super_fast: bool = False

    subtitle_area: Optional[Tuple[int, int, int, int]] = None
    detection_lang: str = "en"
    detection_threshold: float = 0.5
    detection_vertical: bool = False
    whisper_fallback: bool = False
    whisper_backend: str = "faster-whisper"
    whisper_model_size: str = "tiny"
    whisper_model_path: str = ""
    whisper_queue_seconds: float = 3.0
    upscale_factor: int = 0
    film_grain_strength: float = 0.0
    swinir_restore: bool = False
    seedvr2_restore: bool = False
    preserve_color_metadata: bool = True
    nle_sidecar: str = "off"

    time_start: float = 0.0
    time_end: float = 0.0

    detection_frame_skip: int = 0
    mask_dilate_px: int = 8
    mask_feather_px: int = 4

    tbe_enable: bool = True
    tbe_min_coverage: int = 3
    tbe_use_median: bool = True

    tbe_flow_warp: bool = False
    tbe_scene_cut_split: bool = True
    tbe_scene_cut_threshold: float = 0.35
    tbe_scene_cut_use_pyscenedetect: bool = False
    tbe_scene_cut_use_transnetv2: bool = False
    detection_denoise: bool = False
    sam2_refine: bool = False
    edge_ring_px: int = 2

    subtitle_areas: Optional[List[Tuple[int, int, int, int]]] = None
    auto_band: bool = False
    export_srt: bool = False
    export_mask_video: bool = False
    adaptive_batch: bool = True

    auto_exposure_threshold: float = 0.55
    deinterlace: bool = False
    deinterlace_auto: bool = True
    keyframe_detection: bool = False
    quality_report: bool = False

    kalman_tracking: bool = True
    kalman_iou_threshold: float = 0.3
    kalman_max_age: int = 2
    phash_skip_enable: bool = True
    phash_skip_distance: int = 4
    colour_tune_enable: bool = False
    colour_tune_tolerance: int = 25

    output_format: str = "mp4"
    preserve_audio: bool = True
    output_quality: int = 23
    use_hw_encode: bool = True
    output_codec: str = "h264"

    loudnorm_target: float = 0.0
    multi_audio_passthrough: bool = True
    decode_hw_accel: str = "off"
    prefetch_decode: bool = True
    prefetch_queue_size: int = 0
    input_fps: float = 24.0
    quality_report_sheet: bool = False
    remove_subtitles: bool = True
    remove_chyrons: bool = True
    chyron_min_hits: int = 90
    karaoke_grouping: bool = False
    karaoke_x_gap_px: int = 20
    karaoke_y_overlap: float = 0.5

    # UI state (persisted across sessions)
    window_geometry: str = ""
    adv_panel_open: bool = False
    log_panel_open: bool = True
    onboarding_seen: bool = False
    high_contrast: bool = False
    rtl_layout: bool = False
    update_check: bool = False

    def to_dict(self) -> dict:
        from dataclasses import fields as _dc_fields
        payload: dict = {}
        for field_def in _dc_fields(self):
            value = getattr(self, field_def.name)
            if isinstance(value, InpaintMode):
                payload[field_def.name] = value.value
            elif field_def.name == "subtitle_area":
                payload[field_def.name] = list(value) if value else None
            elif field_def.name == "subtitle_areas":
                payload[field_def.name] = (
                    [list(r) for r in value] if value else None
                )
            else:
                payload[field_def.name] = value
        payload["vsr_settings_format"] = VSR_SETTINGS_FORMAT
        return payload

    def normalized(self) -> "ProcessingConfig":
        self.mode = _coerce_gui_mode(self.mode)
        self.use_gpu = _coerce_bool(self.use_gpu, True)
        self.gpu_id = max(0, _coerce_int(self.gpu_id, 0))
        self.sttn_skip_detection = _coerce_bool(self.sttn_skip_detection, False)
        self.sttn_neighbor_stride = _coerce_int(self.sttn_neighbor_stride, 10, 5, 30)
        self.sttn_reference_length = _coerce_int(self.sttn_reference_length, 10, 5, 30)
        self.sttn_max_load_num = _coerce_int(self.sttn_max_load_num, 30, 10, 100)
        self.lama_super_fast = _coerce_bool(self.lama_super_fast, False)
        self.subtitle_area = _coerce_rect(self.subtitle_area)
        self.subtitle_areas = _coerce_rect_list(self.subtitle_areas)
        self.detection_lang = _coerce_text(self.detection_lang, "en", 24).lower()
        self.detection_threshold = _coerce_float(self.detection_threshold, 0.5, 0.1, 0.9)
        self.detection_vertical = _coerce_bool(self.detection_vertical, False)
        self.whisper_fallback = _coerce_bool(self.whisper_fallback, False)
        wb = _coerce_text(self.whisper_backend, "faster-whisper", 32).lower()
        if wb in {"faster", "faster_whisper"}:
            wb = "faster-whisper"
        if wb not in {"faster-whisper", "ffmpeg"}:
            wb = "faster-whisper"
        self.whisper_backend = wb
        wm = _coerce_text(self.whisper_model_size, "tiny", 16).lower()
        if wm not in {"tiny", "base", "small", "medium", "large",
                       "large-v2", "large-v3"}:
            wm = "tiny"
        self.whisper_model_size = wm
        self.whisper_model_path = _coerce_text(self.whisper_model_path, "", 512)
        self.whisper_queue_seconds = _coerce_float(
            self.whisper_queue_seconds, 3.0, 0.02, 3600.0)
        upscale = _coerce_int(self.upscale_factor, 0, 0, 4)
        if upscale not in (0, 2, 3, 4):
            upscale = 0
        self.upscale_factor = upscale
        self.film_grain_strength = _coerce_float(
            self.film_grain_strength, 0.0, 0.0, 0.5)
        self.swinir_restore = _coerce_bool(self.swinir_restore, False)
        self.seedvr2_restore = _coerce_bool(self.seedvr2_restore, False)
        self.preserve_color_metadata = _coerce_bool(
            self.preserve_color_metadata, True)
        sidecar = _coerce_text(self.nle_sidecar, "off", 16).lower()
        if sidecar not in {"off", "edl", "fcpxml"}:
            sidecar = "off"
        self.nle_sidecar = sidecar
        self.time_start = max(0.0, _coerce_float(self.time_start, 0.0))
        self.time_end = max(0.0, _coerce_float(self.time_end, 0.0))
        if self.time_end and self.time_end < self.time_start:
            self.time_end = 0.0
        self.detection_frame_skip = _coerce_int(
            self.detection_frame_skip, 0, 0, 10)
        self.mask_dilate_px = _coerce_int(self.mask_dilate_px, 8, 0, 20)
        self.mask_feather_px = _coerce_int(self.mask_feather_px, 4, 0, 15)
        self.tbe_enable = _coerce_bool(self.tbe_enable, True)
        self.tbe_min_coverage = _coerce_int(self.tbe_min_coverage, 3, 1, 10)
        self.tbe_use_median = _coerce_bool(self.tbe_use_median, True)
        self.tbe_flow_warp = _coerce_bool(self.tbe_flow_warp, False)
        self.tbe_scene_cut_split = _coerce_bool(self.tbe_scene_cut_split, True)
        self.tbe_scene_cut_threshold = _coerce_float(
            self.tbe_scene_cut_threshold, 0.35, 0.0, 1.0)
        self.tbe_scene_cut_use_pyscenedetect = _coerce_bool(
            self.tbe_scene_cut_use_pyscenedetect, False)
        self.tbe_scene_cut_use_transnetv2 = _coerce_bool(
            self.tbe_scene_cut_use_transnetv2, False)
        self.detection_denoise = _coerce_bool(self.detection_denoise, False)
        self.sam2_refine = _coerce_bool(self.sam2_refine, False)
        self.edge_ring_px = _coerce_int(self.edge_ring_px, 2, 0, 20)
        self.auto_band = _coerce_bool(self.auto_band, False)
        self.export_srt = _coerce_bool(self.export_srt, False)
        self.export_mask_video = _coerce_bool(self.export_mask_video, False)
        self.adaptive_batch = _coerce_bool(self.adaptive_batch, True)
        self.auto_exposure_threshold = _coerce_float(
            self.auto_exposure_threshold, 0.55, 0.0, 1.0)
        self.deinterlace = _coerce_bool(self.deinterlace, False)
        self.deinterlace_auto = _coerce_bool(self.deinterlace_auto, True)
        self.keyframe_detection = _coerce_bool(self.keyframe_detection, False)
        self.quality_report = _coerce_bool(self.quality_report, False)
        self.kalman_tracking = _coerce_bool(self.kalman_tracking, True)
        self.kalman_iou_threshold = _coerce_float(
            self.kalman_iou_threshold, 0.3, 0.01, 0.9)
        self.kalman_max_age = _coerce_int(self.kalman_max_age, 2, 1, 30)
        self.phash_skip_enable = _coerce_bool(self.phash_skip_enable, True)
        self.phash_skip_distance = _coerce_int(
            self.phash_skip_distance, 4, 0, 64)
        self.colour_tune_enable = _coerce_bool(self.colour_tune_enable, False)
        self.colour_tune_tolerance = _coerce_int(
            self.colour_tune_tolerance, 25, 1, 100)
        self.output_format = _coerce_text(
            self.output_format, "mp4", 8).lower()
        self.preserve_audio = _coerce_bool(self.preserve_audio, True)
        self.output_quality = _coerce_int(self.output_quality, 23, 15, 35)
        self.use_hw_encode = _coerce_bool(self.use_hw_encode, True)
        codec = _coerce_text(self.output_codec, "h264", 8).lower()
        if codec not in {"h264", "h265", "av1"}:
            codec = "h264"
        self.output_codec = codec
        self.loudnorm_target = _coerce_float(
            self.loudnorm_target, 0.0, -70.0, 0.0)
        if self.loudnorm_target != 0.0 and self.loudnorm_target > -5.0:
            self.loudnorm_target = -5.0
        self.multi_audio_passthrough = _coerce_bool(
            self.multi_audio_passthrough, True)
        accel = _coerce_text(self.decode_hw_accel, "off", 16).lower()
        if accel not in {"off", "auto", "any", "d3d11", "vaapi", "mfx"}:
            accel = "off"
        self.decode_hw_accel = accel
        self.prefetch_decode = _coerce_bool(self.prefetch_decode, True)
        self.prefetch_queue_size = _coerce_int(
            self.prefetch_queue_size, 0, 0, 256)
        self.input_fps = _coerce_float(self.input_fps, 24.0, 1.0, 120.0)
        self.quality_report_sheet = _coerce_bool(
            self.quality_report_sheet, False)
        if self.quality_report_sheet:
            self.quality_report = True
        self.remove_subtitles = _coerce_bool(self.remove_subtitles, True)
        self.remove_chyrons = _coerce_bool(self.remove_chyrons, True)
        self.chyron_min_hits = _coerce_int(self.chyron_min_hits, 90, 1, 9000)
        self.karaoke_grouping = _coerce_bool(self.karaoke_grouping, False)
        self.karaoke_x_gap_px = _coerce_int(
            self.karaoke_x_gap_px, 20, 0, 200)
        self.karaoke_y_overlap = _coerce_float(
            self.karaoke_y_overlap, 0.5, 0.0, 1.0)
        self.window_geometry = _coerce_text(self.window_geometry, "", 64)
        self.adv_panel_open = _coerce_bool(self.adv_panel_open, False)
        self.log_panel_open = _coerce_bool(self.log_panel_open, True)
        self.onboarding_seen = _coerce_bool(self.onboarding_seen, False)
        self.high_contrast = _coerce_bool(self.high_contrast, False)
        self.rtl_layout = _coerce_bool(self.rtl_layout, False)
        self.update_check = _coerce_bool(self.update_check, False)
        return self

    @classmethod
    def from_dict(cls, data: dict) -> "ProcessingConfig":
        if not isinstance(data, dict):
            data = {}
        from dataclasses import fields as _dc_fields
        kwargs: dict = {}
        for field_def in _dc_fields(cls):
            name = field_def.name
            if name not in data:
                continue
            raw = data.get(name)
            if name == "mode":
                kwargs[name] = raw
            elif name == "subtitle_area":
                kwargs[name] = _coerce_rect(raw)
            elif name == "subtitle_areas":
                kwargs[name] = _coerce_rect_list(raw)
            else:
                kwargs[name] = raw
        return cls(**kwargs).normalized()


# -- QueueItem --------------------------------------------------------------


@dataclass
class QueueItem:
    """Represents an item in the processing queue."""
    id: str
    file_path: str
    output_path: str
    config: ProcessingConfig
    output_path_locked: bool = False
    status: ProcessingStatus = ProcessingStatus.IDLE
    progress: float = 0.0
    message: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    quality_report: Optional[dict] = None
    soft_subtitle_streams: List[dict] = field(default_factory=list)
    soft_subtitle_probe_done: bool = False
    soft_subtitle_action: str = "burned_in"
    cancel_requested: bool = False


# -- Helpers ----------------------------------------------------------------


def status_ui(status: ProcessingStatus) -> dict:
    return STATUS_UI.get(
        status,
        {"label": status.value.title(),
         "color": Theme.TEXT_MUTED,
         "bg": Theme.BG_TERTIARY},
    )


# -- Settings I/O -----------------------------------------------------------


def _read_json_object(path: Path, label: str) -> Optional[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Could not read {label} from {path}: {exc}")
        return None
    if not isinstance(payload, dict):
        logger.warning(
            f"Ignoring {label} at {path}: expected a JSON object")
        return None
    return payload


def _write_json_atomic(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
        )
        temp_path = Path(temp_name)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(temp_path, path)
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def _migrate_settings(data: dict) -> dict:
    if not isinstance(data, dict):
        return {}
    version = data.get("vsr_settings_format")
    try:
        version = int(version) if version is not None else 0
    except (TypeError, ValueError):
        version = 0

    if version > VSR_SETTINGS_FORMAT:
        logger.info(
            f"settings.json reports vsr_settings_format={version} "
            f"(this build understands up to {VSR_SETTINGS_FORMAT}); "
            f"unknown keys will be ignored."
        )
        return data

    if version < 1:
        data = dict(data)
        data["vsr_settings_format"] = 1
        version = 1

    if version < 2:
        data = dict(data)
        data["vsr_settings_format"] = 2
        version = 2

    return data


def load_settings() -> ProcessingConfig:
    try:
        if SETTINGS_FILE.exists():
            data = _read_json_object(SETTINGS_FILE, "settings")
            if not data:
                return ProcessingConfig()
            data = _migrate_settings(data)
            logger.info(f"Settings loaded from {SETTINGS_FILE}")
            return ProcessingConfig.from_dict(data)
    except Exception as e:
        logger.warning(f"Could not load settings: {e}")
    return ProcessingConfig()


def save_settings(config: ProcessingConfig):
    try:
        _write_json_atomic(SETTINGS_FILE, config.normalized().to_dict())
        logger.info(f"Settings saved to {SETTINGS_FILE}")
    except Exception as e:
        logger.warning(f"Could not save settings: {e}")


# -- Preset library ---------------------------------------------------------

PRESETS_FILE = LOG_DIR / "presets.json"

from backend.presets import BUILTIN_PRESETS  # noqa: E402


def _load_user_presets() -> dict:
    if PRESETS_FILE.exists():
        payload = _read_json_object(PRESETS_FILE, "user presets")
        if payload is not None:
            return payload
    return {}


def _save_user_presets(presets: dict):
    try:
        _write_json_atomic(PRESETS_FILE, presets)
    except Exception as exc:
        logger.warning(f"Could not save user presets: {exc}")


def list_presets() -> List[Tuple[str, str]]:
    items = [
        (n, p.get("description", ""))
        for n, p in BUILTIN_PRESETS.items()
    ]
    for name, payload in _load_user_presets().items():
        if isinstance(payload, dict):
            items.append((
                name,
                _coerce_text(
                    payload.get("description", "User preset"),
                    "User preset", 120),
            ))
    return items


def apply_preset(config: ProcessingConfig, name: str) -> bool:
    preset = BUILTIN_PRESETS.get(name)
    if preset is None:
        preset = _load_user_presets().get(name)
    if not isinstance(preset, dict):
        return False
    fields = preset.get("fields", {})
    if not isinstance(fields, dict):
        return False
    for k, v in fields.items():
        if k == "mode":
            config.mode = _coerce_gui_mode(v)
            continue
        if hasattr(config, k):
            setattr(config, k, v)
    config.normalized()
    return True


def save_user_preset(name: str, description: str,
                     config: ProcessingConfig,
                     fields: Optional[List[str]] = None) -> bool:
    name = _coerce_text(name, "", 80)
    description = (
        _coerce_text(description, "User preset", 160) or "User preset"
    )
    if not name:
        return False
    if name in BUILTIN_PRESETS:
        return False
    default_fields = [
        "mode", "detection_threshold", "mask_dilate_px",
        "mask_feather_px", "edge_ring_px", "tbe_flow_warp",
        "tbe_scene_cut_split", "colour_tune_enable",
        "colour_tune_tolerance", "kalman_tracking",
        "phash_skip_enable", "phash_skip_distance", "auto_band",
    ]
    fields = fields or default_fields
    config = config.normalized()
    snap = {}
    for k in fields:
        v = getattr(config, k, None)
        if k == "mode" and hasattr(v, "value"):
            v = v.value
        if v is not None:
            snap[k] = v
    user = _load_user_presets()
    user[name] = {"description": description, "fields": snap}
    _save_user_presets(user)
    return True


def delete_user_preset(name: str) -> bool:
    if name in BUILTIN_PRESETS:
        return False
    user = _load_user_presets()
    if name not in user:
        return False
    del user[name]
    _save_user_presets(user)
    return True


def export_preset(name: str, path: str) -> bool:
    preset = BUILTIN_PRESETS.get(name) or _load_user_presets().get(name)
    if not preset:
        return False
    payload = {
        "name": name,
        "description": preset.get("description", ""),
        "fields": preset.get("fields", {}),
        "vsr_preset_format": 1,
    }
    try:
        _write_json_atomic(Path(path), payload)
        return True
    except Exception as exc:
        logger.warning(
            f"Could not export preset '{name}' to {path}: {exc}")
        return False


def import_preset(path: str) -> Optional[str]:
    payload = _read_json_object(Path(path), "preset import")
    if payload is None:
        return None
    if payload.get("vsr_preset_format") != 1:
        logger.warning(f"Not a v1 VSR preset: {path}")
        return None
    name = _coerce_text(payload.get("name", ""), "", 80)
    fields = payload.get("fields", {})
    description = _coerce_text(
        payload.get("description", "Imported preset"),
        "Imported preset", 160,
    )
    if not name or not isinstance(fields, dict):
        return None
    if name in BUILTIN_PRESETS:
        name = f"{name} (imported)"
    user = _load_user_presets()
    user[name] = {"description": description, "fields": fields}
    _save_user_presets(user)
    return name
