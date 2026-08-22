"""Packaged GUI probe for scaling, translation, and disclosure safety."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest import mock

from backend.i18n import tr


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


def _tab_cycle(root, start, *, limit: int = 512) -> set[str]:
    """Return one native Tk forward-focus cycle from ``start``."""
    if not str(root.tk.call("info", "commands", "tk_focusNext")):
        if not root.tk.call("auto_load", "tk_focusNext"):
            raise RuntimeError("Tk focus traversal command is unavailable")
    paths: set[str] = set()
    current = str(start)
    for _index in range(limit):
        next_path = str(root.tk.call("tk_focusNext", current) or "")
        if not next_path or next_path in paths:
            break
        paths.add(next_path)
        current = next_path
    return paths


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
            vbar = getattr(dialog, "_vsr_scroll_vbar", None)
            hbar = getattr(dialog, "_vsr_scroll_hbar", None)
            effective_width = canvas.winfo_width() + (
                vbar.winfo_reqwidth()
                if vbar is not None and dialog._vsr_scroll_vbar_visible
                else 0
            )
            effective_height = canvas.winfo_height() + (
                hbar.winfo_reqheight()
                if hbar is not None and dialog._vsr_scroll_hbar_visible
                else 0
            )
            needs_vertical = (
                bbox[3] - bbox[1] > effective_height + 1)
            needs_horizontal = (
                bbox[2] - bbox[0] > effective_width + 1)
            if vbar is None or bool(
                dialog._vsr_scroll_vbar_visible
            ) != needs_vertical:
                failures.append(f"{name} vertical scrollbar state is stale")
            if hbar is None or bool(
                dialog._vsr_scroll_hbar_visible
            ) != needs_horizontal:
                failures.append(f"{name} horizontal scrollbar state is stale")
            if needs_vertical:
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

    # RM-152: the About dialog is a real, media-free workflow surface.
    about = None
    try:
        app._show_about()
        about = next(
            (child for child in app.root.winfo_children()
             if isinstance(child, tk.Toplevel)), None)
        if about is None:
            failures.append("about dialog was not created")
        else:
            fit_dialog_to_work_area(about, app.root)
            _check("about", about)
    except Exception as exc:  # noqa: BLE001
        failures.append(f"about probe raised: {exc}")
    finally:
        if about is not None:
            try:
                about.grab_release()
            except tk.TclError:
                pass
            about.destroy()

    try:
        del app.root._vsr_work_area_override
    except AttributeError:
        pass
    return failures


def _probe_direction_mirror(app, tk) -> list[str]:
    """RM-152: prove the logical-to-physical mirror reached the live tree.

    Every assertion here is written against widgets the app built itself
    from logical (`left`, `w`, `left`-justified) values, so a regression
    that bypasses the mirror -- a hand-rolled physical value, or a widget
    created before `install_direction_mirror()` -- shows up immediately.
    """
    from gui.direction import direction_mirror_installed, mirror_sticky

    failures: list[str] = []
    if not direction_mirror_installed():
        failures.append("direction mirror was not installed")
        return failures

    # A freshly packed row proves the mirror is live on this tree. The
    # aggregate LTR-vs-RTL comparison lives in the test, which can run
    # both locales and check the histograms are mirror images; the app
    # packs in both directions, so a surviving "left" here is expected.
    try:
        holder = tk.Frame(app.root)
        probe = tk.Label(holder, text="probe", anchor="w", justify="left")
        probe.pack(side="left", anchor="nw")
        info = probe.pack_info()
        if str(info.get("side")) != "right":
            failures.append("a freshly packed row was not mirrored")
        if str(info.get("anchor")) != "ne":
            failures.append("a pack anchor was not mirrored")
        if str(probe.cget("anchor")) != "e":
            failures.append("a widget anchor was not mirrored")
        if str(probe.cget("justify")) != "right":
            failures.append("a widget justification was not mirrored")
        holder.destroy()
    except tk.TclError as exc:
        failures.append(f"pack mirror probe failed: {exc}")

    if not any(
        str(widget.cget("anchor")) == "e"
        for widget in _walk(app.root)
        if isinstance(widget, tk.Label)
    ):
        failures.append("no label was anchored east under RTL")

    # Directional arrow affordances inside menu captions.
    try:
        menu = tk.Menu(app.root, tearoff=0)
        menu.add_command(label=tr("Filename (A -> Z)"))
        if "<-" not in str(menu.entrycget(0, "label")):
            failures.append("menu arrow affordance was not mirrored")
        menu.destroy()
    except tk.TclError as exc:
        failures.append(f"menu mirror probe failed: {exc}")

    # Grid sticky masks mirror their west/east components.
    if mirror_sticky("nw") != "ne" or mirror_sticky("ew") != "we":
        failures.append("grid sticky mask was not mirrored")

    # Canvas items are coordinate-coupled and must be left alone: the
    # Canvas-drawn widgets mirror their own geometry.
    try:
        canvas = tk.Canvas(app.root, width=40, height=20)
        item = canvas.create_text(5, 5, text="x", anchor="w")
        if str(canvas.itemcget(item, "anchor")) != "w":
            failures.append("canvas item anchor was wrongly mirrored")
        canvas.destroy()
    except tk.TclError as exc:
        failures.append(f"canvas mirror probe failed: {exc}")

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
                ("start cleanup", app.command_start_btn),
                ("open output", app.open_output_btn),
                ("set region", app.preview_region_btn),
                ("review mask", app.preview_mask_btn),
                ("test cleanup", app.preview_inpaint_btn),
                ("advanced settings", app.adv_toggle),
                ("help", app._header_help_btn),
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

            for name, button in major_buttons:
                if button.enabled and int(button.cget("takefocus")) != 1:
                    failures.append(f"{name} action is not keyboard focusable")
                if button.winfo_reqwidth() <= 1 or button.winfo_reqheight() <= 1:
                    failures.append(f"{name} action has zero geometry")
                parent_width = button.master.winfo_width()
                if parent_width > 1 and button.winfo_width() > parent_width:
                    failures.append(
                        f"{name} action exceeds its row width "
                        f"({button.winfo_width()} > {parent_width})"
                    )
                bbox = button.bbox("all")
                has_layout_geometry = (
                    button.winfo_width() > 1 and button.winfo_height() > 1
                )
                available_width = (
                    button.winfo_width() if has_layout_geometry
                    else button.winfo_reqwidth()
                )
                available_height = (
                    button.winfo_height() if has_layout_geometry
                    else button.winfo_reqheight()
                )
                if bbox and (
                    bbox[0] < -3 or bbox[1] < -3
                    or bbox[2] > available_width + 3
                    or bbox[3] > available_height + 3
                ):
                    item_boxes = [
                        (button.type(item), button.bbox(item))
                        for item in button.find_all()
                    ]
                    failures.append(
                        f"{name} action Canvas content is clipped "
                        f"(bbox={bbox}, request="
                        f"{button.winfo_reqwidth()}x{button.winfo_reqheight()}, "
                        f"actual={button.winfo_width()}x{button.winfo_height()}, "
                        f"items={item_boxes})"
                    )

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
                failures.extend(_probe_direction_mirror(app, tk))
            if high_contrast and Theme.BG_DARK != "#000000":
                failures.append("high-contrast palette was not applied")

            detail_roots = [
                *app._inspector_primary_detail_roots,
                *app._inspector_advanced_cards,
            ]
            detail_descendants = {
                widget
                for root in detail_roots
                for widget in _walk(root)
            }
            detail_descendants.add(app.adv_panel)
            focusable_descendants = {
                widget
                for widget in detail_descendants
                if str(getattr(widget, "_vsr_a11y_saved_takefocus", "0"))
                not in {"", "0", "false"}
            }
            hidden_paths = {str(widget) for widget in focusable_descendants}
            collapsed_tabs: set[str] = set()
            expanded_tabs: set[str] = set()
            collapsed_again: set[str] = set()
            if not focusable_descendants:
                failures.append("advanced panels expose no focusable descendants")
            if any(
                getattr(widget, "_vsr_a11y_control_view", True) is not False
                for widget in detail_descendants
            ):
                failures.append("collapsed advanced controls remain in control view")
            if any(
                str(widget.cget("takefocus")) not in {"", "0", "false"}
                for widget in focusable_descendants
            ):
                failures.append("collapsed advanced controls remain in tab order")
            original_alpha = 1.0
            try:
                try:
                    original_alpha = float(app.root.attributes("-alpha"))
                    app.root.attributes("-alpha", 0.0)
                except (tk.TclError, TypeError, ValueError):
                    original_alpha = 1.0
                app.root.deiconify()
                app.root.update()
                collapsed_tabs = _tab_cycle(
                    app.root, app._inspector_advanced_button)
                if not collapsed_tabs.isdisjoint(hidden_paths):
                    failures.append("collapsed advanced controls receive Tab focus")
                for section in (
                    "detection", "inpainting", "encoding", "advanced",
                ):
                    app._open_inspector_details(section)
                    app.root.update_idletasks()
                    active = {app.adv_panel}
                    for panel, _pack_options in (
                        app._inspector_section_primary_panels[section]
                    ):
                        active.update(_walk(panel))
                    for panel in app._inspector_section_advanced_cards[section]:
                        active.update(_walk(panel))
                    if any(
                        getattr(widget, "_vsr_a11y_control_view", False)
                        is not True
                        for widget in active
                    ):
                        failures.append(
                            f"expanded {section} controls are absent from control view")
                    if any(
                        getattr(widget, "_vsr_a11y_control_view", True)
                        is not False
                        for widget in detail_descendants - active
                    ):
                        failures.append(
                            f"collapsed controls leak while {section} is expanded")
                    states = {
                        key: getattr(button, "_vsr_a11y", {}).get("state")
                        for key, button in app._inspector_summary_buttons.items()
                    }
                    if states.get(section) != "expanded" or any(
                        state != "collapsed"
                        for key, state in states.items()
                        if key != section
                    ):
                        failures.append(
                            f"{section} disclosure state is inconsistent")
                    section_tabs = _tab_cycle(
                        app.root, app._inspector_summary_buttons[section])
                    active_paths = {
                        str(widget)
                        for widget in active.intersection(focusable_descendants)
                    }
                    if not section_tabs.intersection(active_paths):
                        failures.append(
                            f"expanded {section} controls are absent from tab order")
                    if not section_tabs.isdisjoint(hidden_paths - active_paths):
                        failures.append(
                            f"hidden controls receive Tab focus while {section} is open")
                    if section == "advanced":
                        expanded_tabs = section_tabs
                    app._open_inspector_details(section)
                    app.root.update_idletasks()
                app.root.update_idletasks()
                collapsed_again = _tab_cycle(
                    app.root, app._inspector_advanced_button)
                if not collapsed_again.isdisjoint(hidden_paths):
                    failures.append("re-collapsed advanced controls receive Tab focus")
            except (RuntimeError, tk.TclError) as exc:
                failures.append(f"advanced focus traversal probe failed: {exc}")
            finally:
                if getattr(app, "adv_visible", False):
                    app._set_inspector_section(None)
                    app.root.update_idletasks()
                app.root.withdraw()
                try:
                    app.root.attributes("-alpha", original_alpha)
                except tk.TclError:
                    pass

            original_toggle_text = app.adv_toggle.text
            dynamic_label = tr("Detailed controls")
            app.adv_toggle.set_text(dynamic_label)
            app.root.update_idletasks()
            text_items = [
                item for item in app.adv_toggle.find_all()
                if app.adv_toggle.type(item) == "text"
            ]
            if not text_items or dynamic_label not in str(
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

            # RM-152: the direction census. A test that runs the same
            # scale and theme under `en` and `rtl` can assert these two
            # histograms are exact mirror images, which is the only way
            # to prove *every* row flipped rather than merely some.
            pack_sides = {"left": 0, "right": 0}
            label_anchors = {"w": 0, "e": 0}
            for widget in widgets:
                try:
                    side = str(widget.pack_info().get("side") or "")
                except (tk.TclError, AttributeError, TypeError):
                    side = ""
                if side in pack_sides:
                    pack_sides[side] += 1
                if isinstance(widget, tk.Label):
                    anchor = str(widget.cget("anchor"))
                    if anchor in label_anchors:
                        label_anchors[anchor] += 1

            layout_surfaces = []
            for child in app._content_canvas.master.master.winfo_children():
                pack_info = child.pack_info() if child.winfo_manager() == "pack" else {}
                layout_surfaces.append({
                    "type": type(child).__name__,
                    "side": str(pack_info.get("side", "")),
                    "height": child.winfo_height(),
                    "requestedHeight": child.winfo_reqheight(),
                })
            command_blocks = []
            for block in app._command_blocks:
                command_blocks.append({
                    "height": block.winfo_height(),
                    "requestedHeight": block.winfo_reqheight(),
                    "width": block.winfo_width(),
                    "requestedWidth": block.winfo_reqwidth(),
                    "children": [
                        {
                            "type": type(child).__name__,
                            "height": child.winfo_height(),
                            "requestedHeight": child.winfo_reqheight(),
                            "width": child.winfo_width(),
                            "requestedWidth": child.winfo_reqwidth(),
                        }
                        for child in block.winfo_children()
                    ],
                })

            return {
                "ok": not failures,
                "failures": failures,
                "packSides": pack_sides,
                "labelAnchors": label_anchors,
                "scale": scale,
                "theme": "high-contrast" if high_contrast else "default",
                "locale": locale,
                "buttons": len(buttons),
                "labels": len(labels),
                "contentHeight": app._content_canvas.winfo_height(),
                "contentScrollHeight": (app._content_canvas.bbox("all") or (0, 0, 0, 0))[3],
                "tkScaling": float(app.root.tk.call("tk", "scaling")),
                "layoutSurfaces": layout_surfaces,
                "commandBlocks": command_blocks,
                "advancedTabStops": {
                    "collapsed": len(collapsed_tabs),
                    "expanded": len(expanded_tabs),
                    "collapsedAgain": len(collapsed_again),
                },
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
