"""Hidden subprocess probe for text scaling and translation-safe reflow."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _translation(locale: str):
    rtl_mark = chr(0x200F)
    rtl_word = "".join(chr(value) for value in (0x05DE, 0x05DE, 0x05E9, 0x05E7))

    def translate(text: str) -> str:
        if locale == "pseudo":
            padding = " expanded" * max(1, len(text) // 18)
            return "[[ " + text + padding + " ]]"
        if locale == "rtl":
            return rtl_mark + rtl_word + " " + text + rtl_mark
        return text

    return translate


def _walk(widget):
    yield widget
    for child in widget.winfo_children():
        yield from _walk(child)


def run_probe(scale: int, high_contrast: bool, locale: str) -> dict:
    os.environ["VSR_UI_BACKGROUND"] = "1"
    with tempfile.TemporaryDirectory(prefix="vsr_ui_scale_") as tmpdir:
        os.environ["APPDATA"] = tmpdir
        import gui.config as gui_config
        from gui.app import VideoSubtitleRemoverApp
        from gui.theme import Theme
        from gui.widgets import ModernButton, ModernToggle
        import tkinter as tk
        import tkinter.font as tkfont

        gui_config.SETTINGS_FILE = Path(tmpdir) / "settings.json"
        gui_config.QUEUE_STATE_FILE = Path(tmpdir) / "queue.json"
        gui_config.save_settings(gui_config.ProcessingConfig(
            text_scale_percent=scale,
            high_contrast=high_contrast,
            rtl_layout=(locale == "rtl"),
            onboarding_seen=True,
            log_panel_open=False,
        ))

        translate = _translation(locale)
        for module_name, module in list(sys.modules.items()):
            if module_name == "gui" or module_name.startswith("gui."):
                if hasattr(module, "tr"):
                    setattr(module, "tr", translate)

        app = None
        try:
            with mock.patch.object(
                VideoSubtitleRemoverApp, "_start_startup_hardware_probe"
            ), mock.patch.object(
                VideoSubtitleRemoverApp, "_maybe_restore_queue"
            ):
                app = VideoSubtitleRemoverApp()
            app.root.update_idletasks()
            app._apply_responsive_layout(980)
            app.root.update_idletasks()
            app._on_content_canvas_configure(type(
                "ConfigureEvent",
                (),
                {"width": app._content_canvas.winfo_width()},
            )())
            app.root.update_idletasks()

            widgets = list(_walk(app.root))
            buttons = [widget for widget in widgets if isinstance(widget, ModernButton)]
            toggles = [widget for widget in widgets if isinstance(widget, ModernToggle)]
            labels = [widget for widget in widgets if isinstance(widget, tk.Label)]
            major_buttons = [
                app.start_btn,
                app.open_output_btn,
                app.preview_region_btn,
                app.preview_mask_btn,
                app.preview_inpaint_btn,
                app.adv_toggle,
                app._header_help_btn,
            ]
            failures = []
            if app.root.state() != "withdrawn":
                failures.append("root is not withdrawn")
            if (app.root.winfo_width(), app.root.winfo_height()) != (980, 720):
                failures.append("root is not at the 980x720 minimum viewport")
            if app._content_canvas.xview() != (0.0, 1.0):
                failures.append("content requires horizontal scrolling")
            if app._layout_mode != "stacked":
                failures.append("minimum viewport did not use stacked layout")
            if app._footer.winfo_height() < app._footer.winfo_reqheight():
                failures.append("footer is clipped")
            if app._content_canvas.winfo_height() < 100:
                failures.append("scrollable workbench is too short")

            for button in major_buttons:
                if button.enabled and int(button.cget("takefocus")) != 1:
                    failures.append("major action is not keyboard focusable")
                if button.winfo_reqwidth() <= 1 or button.winfo_reqheight() <= 1:
                    failures.append("major action has zero geometry")
                parent_width = button.master.winfo_width()
                if parent_width > 1 and button.winfo_width() > parent_width:
                    failures.append("major action exceeds its row width")
                bbox = button.bbox("all")
                if bbox and (
                    bbox[0] < -3 or bbox[1] < -3
                    or bbox[2] > button.winfo_reqwidth() + 3
                    or bbox[3] > button.winfo_reqheight() + 3
                ):
                    failures.append("major action Canvas content is clipped")

            expected_height = round(36 * scale / 100)
            if max(
                app.start_btn.winfo_height(),
                app.start_btn.winfo_reqheight(),
            ) < expected_height:
                failures.append("button height did not scale with its text")
            header_font = tkfont.Font(font=app._header_left.winfo_children()[0].cget("font"))
            if abs(int(header_font.cget("size"))) < round(22 * scale / 100):
                failures.append("display font did not reach the requested scale")

            verbose_labels = [
                label for label in labels
                if len(str(label.cget("text") or "")) >= 40
            ]
            if not verbose_labels:
                failures.append("translation fixture produced no verbose labels")
            for label in verbose_labels:
                if int(float(str(label.cget("wraplength") or 0))) <= 0:
                    failures.append("verbose label has no wrap length")
                    break

            if locale == "pseudo" and not any(
                str(label.cget("text")).startswith("[[") for label in labels
            ):
                failures.append("pseudo-localized strings were not rendered")
            if locale == "rtl":
                if not Theme.RTL_LAYOUT:
                    failures.append("RTL theme direction was not enabled")
                if not any(str(label.cget("justify")) == "right" for label in labels):
                    failures.append("RTL labels were not right-justified")
                if toggles:
                    text_items = [
                        item for item in toggles[0].find_all()
                        if toggles[0].type(item) == "text"
                    ]
                    if not text_items or toggles[0].itemcget(
                        text_items[-1], "anchor"
                    ) != "e":
                        failures.append("RTL toggle geometry was not mirrored")
            if high_contrast and Theme.BG_DARK != "#000000":
                failures.append("high-contrast palette was not applied")

            original_toggle_text = app.adv_toggle.text
            app.adv_toggle.set_text(translate("Detailed controls"))
            app.root.update_idletasks()
            text_items = [
                item for item in app.adv_toggle.find_all()
                if app.adv_toggle.type(item) == "text"
            ]
            if not text_items or translate("Detailed controls") not in str(
                app.adv_toggle.itemcget(text_items[-1], "text")
            ):
                failures.append("dynamic Canvas button text was not reflowed")
            app.adv_toggle.set_text(original_toggle_text)

            return {
                "ok": not failures,
                "failures": failures,
                "scale": scale,
                "theme": "high-contrast" if high_contrast else "default",
                "locale": locale,
                "buttons": len(buttons),
                "labels": len(labels),
                "contentHeight": app._content_canvas.winfo_height(),
                "contentScrollHeight": (app._content_canvas.bbox("all") or (0, 0, 0, 0))[3],
            }
        finally:
            if app is not None:
                app.root.destroy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, choices=(100, 125, 150, 175, 200), required=True)
    parser.add_argument("--theme", choices=("default", "high-contrast"), required=True)
    parser.add_argument("--locale", choices=("en", "pseudo", "rtl"), required=True)
    args = parser.parse_args()
    result = run_probe(args.scale, args.theme == "high-contrast", args.locale)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
