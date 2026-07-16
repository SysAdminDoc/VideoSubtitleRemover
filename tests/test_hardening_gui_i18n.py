import io
import json
import logging
import os
import sys
import tempfile
import threading
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import VideoSubtitleRemover as gui
from backend import processor


def _has_display() -> bool:
    """Return True if a GUI display is available."""
    if sys.platform == "win32":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class GuiFfmpegProfilePreflightTests(unittest.TestCase):
    def test_batch_start_warns_when_selected_codec_profile_is_missing(self):
        app = gui.VideoSubtitleRemoverApp.__new__(gui.VideoSubtitleRemoverApp)
        app.root = object()
        app.queue_lock = threading.Lock()
        app._update_status = unittest.mock.Mock()
        app.ffmpeg_profiles = {}
        app._current_ffmpeg_profiles = unittest.mock.Mock(return_value={
            "schema": "vsr.ffmpeg_profiles.v1",
            "profiles": [
                {
                    "name": "basic",
                    "available": True,
                    "missing": {"tools": [], "filters": [], "encoders": []},
                    "reason": "ready",
                },
                {
                    "name": "advanced_quality",
                    "available": True,
                    "missing": {"tools": [], "filters": [], "encoders": []},
                    "reason": "ready",
                },
                {
                    "name": "speech_fallback",
                    "available": True,
                    "missing": {"tools": [], "filters": [], "encoders": []},
                    "reason": "ready",
                },
                {
                    "name": "modern_codec",
                    "available": False,
                    "missing": {
                        "encoder_groups": [{
                            "name": "vvc",
                            "any_of": ["libvvenc", "vvc_nvenc"],
                        }],
                    },
                    "reason": "missing encoder groups: vvc (libvvenc/vvc_nvenc)",
                },
            ],
        })
        app.queue = [
            gui.QueueItem(
                id="vvc",
                file_path="movie.mp4",
                output_path="movie_no_sub.mp4",
                config=gui.ProcessingConfig(
                    output_codec="vvc",
                    preserve_audio=False,
                ),
            )
        ]

        with unittest.mock.patch("gui.app.show_confirm", return_value=False) as confirm:
            self.assertFalse(app._confirm_ffmpeg_profile_coverage())

        detail = confirm.call_args.kwargs["detail"]
        self.assertIn("movie.mp4", detail)
        self.assertIn("libvvenc", detail)
        app._update_status.assert_called_once()


class JsonLineLogHandlerTests(unittest.TestCase):
    """JsonLineLogHandler must write exactly one JSON record per emit,
    include the level / logger / msg / ts fields, and capture exception
    text when the record carries exc_info."""

    def test_emit_writes_one_json_record_per_line(self):
        sink = io.StringIO()
        handler = processor.JsonLineLogHandler(sink)
        record = logging.LogRecord(
            "vsr_test", logging.WARNING, __file__, 42,
            "hello %s", ("world",), None,
        )
        handler.emit(record)
        lines = sink.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        payload = json.loads(lines[0])
        self.assertEqual(payload["level"], "WARNING")
        self.assertEqual(payload["msg"], "hello world")
        self.assertEqual(payload["logger"], "vsr_test")
        self.assertIn("ts", payload)
        self.assertNotIn("exc", payload)

    def test_emit_includes_exception_text_when_present(self):
        sink = io.StringIO()
        handler = processor.JsonLineLogHandler(sink)
        try:
            raise RuntimeError("kaboom")
        except RuntimeError:
            record = logging.LogRecord(
                "vsr_test", logging.ERROR, __file__, 42,
                "fell over", (), sys.exc_info(),
            )
        handler.emit(record)
        payload = json.loads(sink.getvalue())
        self.assertIn("RuntimeError", payload["exc"])
        self.assertIn("kaboom", payload["exc"])


class LanguagePickerTests(unittest.TestCase):
    """F-5: lang picker must expose more than the legacy 12 languages
    while keeping the curated English-first ordering."""

    def test_language_list_starts_with_english(self):
        langs = gui._build_language_list()
        self.assertEqual(langs[0][0], "en")
        self.assertEqual(langs[0][1], "English")

    def test_language_list_includes_extra_codes(self):
        codes = {code for code, _ in gui._build_language_list()}
        # Languages outside the legacy 12-language set.
        for new_code in ("th", "vi", "pl", "tr", "uk", "el"):
            self.assertIn(new_code, codes, f"expected {new_code} in expanded list")

    def test_language_list_deduplicates(self):
        codes = [code for code, _ in gui._build_language_list()]
        self.assertEqual(len(codes), len(set(codes)),
                         "language picker must not contain duplicate codes")


class I18nScaffoldTests(unittest.TestCase):
    """RM-97: gettext scaffold must pass strings through unchanged when
    no catalog is bound, and bind cleanly when one is."""

    class _FakeTranslations:
        def __init__(self, mapping):
            self.mapping = dict(mapping)

        def gettext(self, text):
            return self.mapping.get(text, text)

    def test_passthrough_without_catalog(self):
        from backend import i18n
        i18n.bind_locale(None)
        self.assertEqual(i18n._("Start batch"), "Start batch")
        self.assertFalse(i18n.is_translation_active())

    def test_unknown_locale_falls_back(self):
        from backend import i18n
        # Bind a locale that almost certainly has no catalog shipped --
        # the helper should swallow the FileNotFoundError and keep the
        # NullTranslations in place.
        i18n.bind_locale("zz")
        self.assertEqual(i18n._("Start batch"), "Start batch")
        self.assertFalse(i18n.is_translation_active())

    def test_locale_tags_keep_script_and_territory_fallbacks(self):
        from backend import i18n

        self.assertEqual(i18n.normalise_locale_tag("pt_BR.UTF-8"), "pt-BR")
        self.assertEqual(
            i18n.locale_fallback_chain("zh_Hant_TW"),
            ("zh-Hant-TW", "zh-Hant", "zh"),
        )
        with unittest.mock.patch.dict(
            os.environ, {"VSR_UI_LOCALE": "sr_Latn_RS"}, clear=False,
        ):
            self.assertEqual(i18n.system_locale_tag(), "sr-Latn-RS")

    def test_catalog_chain_uses_territory_then_language(self):
        from backend import i18n

        class FakeCatalog:
            def __init__(self, mapping):
                self.mapping = mapping
                self.fallback = None

            def add_fallback(self, fallback):
                self.fallback = fallback

            def gettext(self, text):
                if text in self.mapping:
                    return self.mapping[text]
                return self.fallback.gettext(text) if self.fallback else text

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for tag in ("pt-BR", "pt"):
                catalog = root / tag / "LC_MESSAGES" / "vsr.mo"
                catalog.parent.mkdir(parents=True)
                catalog.write_bytes(b"catalog fixture")
            catalogs = [
                FakeCatalog({"Start batch": "Iniciar lote"}),
                FakeCatalog({"Settings": "Definicoes"}),
            ]
            with unittest.mock.patch(
                "backend.i18n._candidate_locale_dirs", return_value=[root],
            ), unittest.mock.patch(
                "backend.i18n.gettext.GNUTranslations", side_effect=catalogs,
            ):
                try:
                    self.assertEqual(i18n.bind_locale("pt_BR"), "pt-BR")
                    self.assertEqual(i18n.tr("Start batch"), "Iniciar lote")
                    self.assertEqual(i18n.tr("Settings"), "Definicoes")
                finally:
                    i18n.bind_locale(None)

    def test_status_ui_uses_bound_catalog_and_missing_keys_fall_back(self):
        from backend import i18n
        from gui.config import ProcessingStatus, status_ui

        previous = i18n._active_translation
        try:
            i18n._active_translation = self._FakeTranslations({
                "Ready": "Listo",
                "Ready to process": "Listo para procesar",
                "Remove": "Quitar",
                "Start batch": "Iniciar lote",
            })

            self.assertEqual(i18n.tr("Start batch"), "Iniciar lote")
            self.assertEqual(i18n.tr("Missing key"), "Missing key")
            self.assertEqual(status_ui(ProcessingStatus.IDLE)["label"], "Listo")
        finally:
            i18n._active_translation = previous

    def test_locale_template_contains_representative_gui_strings(self):
        pot = Path("locale/vsr.pot").read_text(encoding="utf-8")
        for expected in (
            'msgid "Start batch"',
            'msgid "Settings"',
            'msgid "Backend status and app version."',
            'msgid "Per-file overrides"',
            'msgid "Preview tools are locked while a batch is running."',
        ):
            self.assertIn(expected, pot)


class PseudoLocaleRtlSmokeTests(unittest.TestCase):
    """RM-RD: pseudo-locale and RTL smoke tests verify translated or
    expanded strings fit the main GUI without clipping."""

    def _make_pseudo_translations(self):
        """Return a fake translation that wraps every string in brackets
        and expands it by ~30% to simulate a verbose locale."""
        class _PseudoTranslations:
            def gettext(self, text):
                pad = "x" * max(1, len(text) // 3)
                return f"[{text} {pad}]"
        return _PseudoTranslations()

    def test_pseudo_locale_strings_are_expanded(self):
        from backend import i18n
        previous = i18n._active_translation
        try:
            i18n._active_translation = self._make_pseudo_translations()
            self.assertTrue(i18n.is_translation_active())
            result = i18n.tr("Start batch")
            self.assertTrue(result.startswith("["))
            self.assertTrue(result.endswith("]"))
            self.assertGreater(len(result), len("Start batch"))
        finally:
            i18n._active_translation = previous

    def test_pseudo_locale_status_labels_render_with_fallback(self):
        from backend import i18n
        from gui.config import ProcessingStatus, status_ui

        previous = i18n._active_translation
        try:
            i18n._active_translation = self._make_pseudo_translations()
            ui = status_ui(ProcessingStatus.IDLE)
            self.assertTrue(ui["label"].startswith("["))
            self.assertIn("Ready", ui["label"])
        finally:
            i18n._active_translation = previous

    def test_rtl_layout_sentinel_strings_preserve_content(self):
        from backend import i18n
        rtl_strings = [
            "Start batch",
            "Settings",
            "Preview unavailable",
        ]
        rtl_mark = chr(0x200f)
        previous = i18n._active_translation
        try:
            class _RtlTranslations:
                def gettext(self, text):
                    return rtl_mark + text + rtl_mark
            i18n._active_translation = _RtlTranslations()
            for source in rtl_strings:
                result = i18n.tr(source)
                self.assertIn(source, result)
                self.assertTrue(result.startswith(rtl_mark))
        finally:
            i18n._active_translation = previous


class A11yScaffoldTests(unittest.TestCase):
    """RM-95: announce() must be safe to call when UIA / pywin32 is
    unavailable. Returns silently rather than raising."""

    def test_announce_noop_when_provider_missing(self):
        from backend import a11y
        # Force the probed cache to "no provider" so announce() takes
        # the silent path without trying to import comtypes.
        a11y._PROBED = True
        a11y._PROVIDER = None
        try:
            a11y.announce("Hello, world", importance="normal")
            a11y.announce("Critical!", importance="high")
        except Exception as exc:
            self.fail(f"announce raised: {exc}")

    def test_accessible_metadata_formats_concise_text(self):
        from backend import a11y

        widget = SimpleNamespace()
        metadata = a11y.set_accessible_metadata(
            widget,
            role="button",
            label="Start batch",
            state="enabled, focused",
            value="",
            description="Cleans all queued files",
        )

        self.assertEqual(a11y.accessible_metadata(widget), metadata)
        self.assertEqual(
            a11y.accessible_text(metadata),
            "Start batch. button. enabled, focused. Cleans all queued files",
        )


class HighContrastThemeTests(unittest.TestCase):
    """RM-96: apply_high_contrast_theme must swap the palette in place
    and apply_default_theme must restore it byte-identical."""

    def test_swap_and_restore(self):
        original_bg = gui.Theme.BG_DARK
        original_text = gui.Theme.TEXT_PRIMARY
        try:
            gui.apply_high_contrast_theme()
            self.assertEqual(gui.Theme.BG_DARK, "#000000")
            self.assertEqual(gui.Theme.TEXT_PRIMARY, "#ffffff")
        finally:
            gui.apply_default_theme()
        self.assertEqual(gui.Theme.BG_DARK, original_bg)
        self.assertEqual(gui.Theme.TEXT_PRIMARY, original_text)

    def test_reduced_motion_environment_override_is_deterministic(self):
        from gui.theme import prefers_reduced_motion

        with unittest.mock.patch.dict(
            os.environ, {"VSR_REDUCED_MOTION": "1"}, clear=False,
        ):
            self.assertTrue(prefers_reduced_motion())
        with unittest.mock.patch.dict(
            os.environ, {"VSR_REDUCED_MOTION": "0"}, clear=False,
        ):
            self.assertFalse(prefers_reduced_motion())


class JsonLogGuiTests(unittest.TestCase):
    """RM-53: structured JSON log GUI toggle."""

    def test_config_field_defaults_off(self):
        cfg = gui.ProcessingConfig()
        self.assertFalse(cfg.json_log_enabled)

    def test_config_round_trip(self):
        cfg = gui.ProcessingConfig()
        cfg.json_log_enabled = True
        d = cfg.to_dict()
        self.assertTrue(d["json_log_enabled"])
        cfg2 = gui.ProcessingConfig.from_dict(d)
        self.assertTrue(cfg2.json_log_enabled)


class GpuOomRecoveryTests(unittest.TestCase):
    """P1: recover gracefully from a GPU OOM mid-batch."""

    def _remover(self, inpainter):
        from backend.processor import SubtitleRemover
        from backend.config import ProcessingConfig
        remover = SubtitleRemover.__new__(SubtitleRemover)
        remover.config = ProcessingConfig()
        remover.inpainter = inpainter
        return remover

    def test_oom_detection_markers(self):
        from backend.inpainters import is_oom_error
        self.assertTrue(is_oom_error(RuntimeError("CUDA out of memory")))
        self.assertTrue(is_oom_error(MemoryError()))
        self.assertTrue(is_oom_error(RuntimeError("CUBLAS_STATUS_ALLOC_FAILED")))
        self.assertFalse(is_oom_error(ValueError("bad shape")))

    def test_batch_halves_until_it_fits(self):
        # Inpainter OOMs on any batch larger than 2 frames; the resilient
        # wrapper must split down until every sub-batch fits.
        calls = []

        class FlakyInpainter:
            def inpaint(self, frames, masks):
                calls.append(len(frames))
                if len(frames) > 2:
                    raise RuntimeError("CUDA out of memory. Tried to allocate")
                return [f.copy() for f in frames]

        remover = self._remover(FlakyInpainter())
        frames = [np.zeros((8, 8, 3), np.uint8) for _ in range(8)]
        masks = [np.zeros((8, 8), np.uint8) for _ in range(8)]
        with unittest.mock.patch(
            "backend.processor.free_inference_memory"
        ) as freed:
            out = remover._inpaint_batch_resilient(frames, masks)
        self.assertEqual(len(out), 8)
        self.assertTrue(all(c <= 2 for c in calls if c <= 2))
        self.assertGreaterEqual(freed.call_count, 1)

    def test_single_frame_oom_falls_back_to_cpu(self):
        class AlwaysOom:
            def inpaint(self, frames, masks):
                raise RuntimeError("CUDA out of memory")

        remover = self._remover(AlwaysOom())
        frames = [np.full((8, 8, 3), 200, np.uint8) for _ in range(2)]
        masks = [np.zeros((8, 8), np.uint8) for _ in range(2)]
        masks[0][2:6, 2:6] = 255
        with unittest.mock.patch("backend.processor.free_inference_memory"):
            out = remover._inpaint_batch_resilient(frames, masks)
        self.assertEqual(len(out), 2)
        for frame in out:
            self.assertEqual(frame.shape, (8, 8, 3))

    def test_non_oom_error_propagates(self):
        class Broken:
            def inpaint(self, frames, masks):
                raise ValueError("model shape mismatch")

        remover = self._remover(Broken())
        frames = [np.zeros((8, 8, 3), np.uint8) for _ in range(4)]
        masks = [np.zeros((8, 8), np.uint8) for _ in range(4)]
        with self.assertRaises(ValueError):
            remover._inpaint_batch_resilient(frames, masks)

    def test_recovery_can_be_disabled(self):
        class AlwaysOom:
            def inpaint(self, frames, masks):
                raise RuntimeError("CUDA out of memory")

        remover = self._remover(AlwaysOom())
        remover.config.gpu_oom_recovery = False
        frames = [np.zeros((8, 8, 3), np.uint8) for _ in range(2)]
        masks = [np.zeros((8, 8), np.uint8) for _ in range(2)]
        with self.assertRaises(RuntimeError):
            remover._inpaint_batch_resilient(frames, masks)


class PreviewThreadSafetyTests(unittest.TestCase):
    """Keep Tk image construction on the event-loop thread."""

    def test_normal_preview_defers_photoimage_creation_to_ui_callback(self):
        from gui.config import ProcessingStatus
        from gui.preview_controller import PreviewControllerMixin

        callbacks = []
        photo_threads = []
        main_thread = threading.get_ident()

        class FakeRoot:
            @staticmethod
            def after(_delay, callback, *args):
                callbacks.append((callback, args))

        class FakeImage:
            @staticmethod
            def thumbnail(_size, _resample):
                return None

        controller = PreviewControllerMixin()
        controller.root = FakeRoot()
        controller._preview_request_id = 7
        controller._selected_queue_item_id = "queue-item"
        controller._throbber_id = None
        controller._preview_photo = None
        controller.preview_title_label = unittest.mock.Mock()
        controller.preview_meta_label = unittest.mock.Mock()
        controller._preview_label = unittest.mock.Mock()

        def make_photo(_image):
            photo_threads.append(threading.get_ident())
            return object()

        with unittest.mock.patch(
            "gui.preview_controller.ImageTk.PhotoImage", side_effect=make_photo,
        ):
            worker = threading.Thread(
                target=controller._preview_bg_normal,
                args=(
                    object(), "clip.mp4", "queue-item", ProcessingStatus.IDLE,
                    "missing.mp4", None, [], 7, 390, 260,
                    SimpleNamespace(), lambda _frame: FakeImage(),
                    lambda _path: None,
                ),
            )
            worker.start()
            worker.join(timeout=2)
            self.assertFalse(worker.is_alive())
            self.assertEqual(photo_threads, [])
            self.assertEqual(len(callbacks), 1)

            callback, args = callbacks.pop()
            callback(*args)

        self.assertEqual(photo_threads, [main_thread])
        controller._preview_label.config.assert_called_once()



if __name__ == "__main__":
    unittest.main()
