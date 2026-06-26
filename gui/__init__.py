"""GUI subpackage -- re-exports for backward compatibility.

After the RM-114 extraction, existing ``import VideoSubtitleRemover``
and ``from VideoSubtitleRemover import X`` statements keep working
because the monolith re-imports these names.  New code should import
from the focused submodules directly.
"""

from gui.theme import (  # noqa: F401
    Theme,
    apply_default_theme,
    apply_high_contrast_theme,
    f,
    mono,
)

from gui.config import (  # noqa: F401
    APP_AUTHOR,
    APP_NAME,
    APP_VERSION,
    BUILTIN_PRESETS,
    InpaintMode,
    LOG_DIR,
    LOG_FILE,
    PRESETS_FILE,
    ProcessingConfig,
    ProcessingStatus,
    QueueItem,
    SETTINGS_FILE,
    STATUS_UI,
    VSR_SETTINGS_FORMAT,
    SAFE_PRESET_FIELDS,
    _coerce_bool,
    _coerce_float,
    _coerce_gui_mode,
    _coerce_int,
    _coerce_rect,
    _coerce_rect_list,
    _coerce_text,
    _migrate_settings,
    _read_json_object,
    _write_json_atomic,
    apply_preset,
    consume_preset_import_notice,
    consume_settings_load_notice,
    delete_user_preset,
    export_preset,
    import_preset,
    list_presets,
    load_settings,
    save_settings,
    save_user_preset,
    status_ui,
)

from gui.utils import (  # noqa: F401
    _build_language_list,
    _CURATED_LANG_NAMES,
    _engine_supported_languages,
    _format_soft_subtitle_summary,
    _queue_item_info_text,
    _soft_subtitle_stream_record,
    detect_ai_engines,
    detect_ffmpeg,
    detect_gpu,
    format_quality_report,
    format_size,
    format_time,
    get_app_dir,
    get_file_info,
    is_image_file,
    is_video_file,
    language_support_status,
    summarize_quality_reports,
    truncate_middle,
)

from gui.app import VideoSubtitleRemoverApp  # noqa: F401
