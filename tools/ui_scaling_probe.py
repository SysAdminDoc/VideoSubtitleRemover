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


def _probe_dialog_fit(app, work_area, tk) -> list[str]:
    """RM-148: every major dialog must fit the work area or scroll inside it.

    The dialogs are built for real (onboarding modal, region editor, and
    mask-correction editor), measured against a simulated work area, then torn
    down. A dialog taller than the work area is only acceptable when it exposes
    an internal scroll path and a keyboard-focusable scroll surface.
    """
    from gui.dialog_layout import fit_dialog_to_work_area

    failures: list[str] = []
    app.root._vsr_work_area_override = work_area
    area_w, area_h = work_area

    def _check(name, dialog):
        try:
            dialog.update_idletasks()
            width = max(dialog.winfo_width(), 1)
            height = max(dialog.winfo_height(), 1)
            if width > area_w + 2 or height > area_h + 2:
                failures.append(f"{name} exceeds the work area")
            canvas = getattr(dialog, "_vsr_scroll_canvas", None)
            if canvas is None:
                failures.append(f"{name} has no internal scroll path")
                return
            if int(canvas.cget("takefocus")) != 1:
                failures.append(f"{name} scroll surface is not focusable")
            body = getattr(dialog, "_vsr_scroll_body", None)
            if body is None or body.winfo_reqheight() <= 1:
                failures.append(f"{name} scroll body has no content")
                return
            bbox = canvas.bbox("all") or (0, 0, 0, 0)
            if bbox[3] - bbox[1] > canvas.winfo_height() + 1:
                # Content is taller than the viewport, so the scrollbar must be
                # mapped and the view must actually be able to move.
                canvas.yview_moveto(1.0)
                canvas.update_idletasks()
                if canvas.yview()[0] <= 0.0:
                    failures.append(f"{name} content cannot be scrolled")
                canvas.yview_moveto(0.0)
        except tk.TclError as exc:
            failures.append(f"{name} probe failed: {exc}")

    # Onboarding modal.
    dialog = None
    try:
        app.config.onboarding_seen = False
        app._show_onboarding()
        dialog = next(
            (child for child in app.root.winfo_children()
             if isinstance(child, tk.Toplevel)), None)
        if dialog is None:
            failures.append("onboarding dialog was not created")
        else:
            fit_dialog_to_work_area(dialog, app.root)
            _check("onboarding", dialog)
    except Exception as exc:  # noqa: BLE001 - the probe reports, never raises
        failures.append(f"onboarding probe raised: {exc}")
    finally:
        if dialog is not None:
            try:
                dialog.grab_release()
            except tk.TclError:
                pass
            dialog.destroy()

    # A synthetic dialog that is deliberately taller than the work area proves
    # the shared helper clamps and scrolls rather than overflowing.
    tall = None
    try:
        from gui.dialog_layout import scrollable_dialog_body

        tall = tk.Toplevel(app.root)
        tall.withdraw()
        body = scrollable_dialog_body(tall)
        for index in range(80):
            tk.Label(body, text=f"row {index}" * 6).pack(anchor="w")
        fit_dialog_to_work_area(tall, app.root)
        _check("oversized-dialog", tall)
    except Exception as exc:  # noqa: BLE001
        failures.append(f"oversized dialog probe raised: {exc}")
    finally:
        if tall is not None:
            tall.destroy()

    try:
        del app.root._vsr_work_area_override
    except AttributeError:
        pass
    return failures


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
                app.command_start_btn,
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
                app.command_start_btn.winfo_height(),
                app.command_start_btn.winfo_reqheight(),
            ) < expected_height:
                failures.append("button height did not scale with its text")
            header_font = tkfont.Font(font=app._header_title_label.cget("font"))
            if abs(int(header_font.cget("size"))) < round(
                Theme.F_DISPLAY * scale / 100
            ):
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

            # RM-148: dialogs at both the minimum and a wide work area.
            for area in ((980, 720), (2752, 1152)):
                failures.extend(
                    f"{item} @ {area[0]}x{area[1]}"
                    for item in _probe_dialog_fit(app, area, tk)
                )

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
