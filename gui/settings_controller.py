from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Protocol, Tuple

try:
    import tkinter as tk
    from tkinter import filedialog
except ImportError:  # pragma: no cover - tkinter is optional for headless imports
    pass

from backend.i18n import tr
from gui.config import (
    BUILTIN_PRESETS,
    InpaintMode,
    ProcessingConfig,
    _load_user_presets,
    apply_preset,
    consume_preset_import_notice,
    export_preset,
    import_preset,
    list_presets,
    save_settings,
    save_user_preset,
)
from gui.theme import Theme, f
from gui.widgets import (
    ModernButton,
    ModernSlider,
)

logger = logging.getLogger(__name__)


class SettingsControllerHost(Protocol):
    """Host surface required by AdvancedSettingsControllerMixin."""

    root: Any
    config: ProcessingConfig

    def _update_status(self, message: str, tone: str = "neutral", toast: bool = False): ...


class AdvancedSettingsControllerMixin:
    """Focused behavior mixed into the composed GUI host."""

    def _create_slider(self, parent, label, min_val, max_val, default, attr_name,
                       hint: str = ""):
        """Create a labeled slider row with a quiet inline value readout."""
        parent_bg = parent.cget("bg") if hasattr(parent, "cget") else Theme.BG_CARD
        row = tk.Frame(parent, bg=parent_bg)
        row.pack(fill="x", padx=Theme.S_LG, pady=(Theme.S_XS, 2))

        tk.Label(row, text=tr(label), font=f(Theme.F_BODY_SM),
                 bg=parent_bg, fg=Theme.TEXT_SECONDARY,
                 width=16, anchor="w").pack(side="left")

        value_label = tk.Label(
            row,
            text=str(default),
            font=f(Theme.F_BODY_SM, "bold"),
            bg=parent_bg,
            fg=Theme.TEXT_PRIMARY,
            padx=4,
            width=4,
            anchor="e",
        )
        value_label.pack(side="right", padx=(Theme.S_MD, 0))

        slider = ModernSlider(row, from_=min_val, to=max_val, value=default,
                              bg=parent_bg, accessible_label=tr(label))
        slider.pack(side="left", fill="x", expand=True, padx=(Theme.S_SM, 0))
        self._settings_sliders.append(slider)
        self._settings_slider_by_attr[attr_name] = (slider, value_label)

        def on_change(val):
            value_label.config(text=str(int(val)))
            setattr(self.config, attr_name, int(val))

        slider.command = on_change

        if hint:
            tk.Label(parent, text=tr(hint), font=f(Theme.F_META),
                     bg=parent_bg, fg=Theme.TEXT_MUTED,
                     anchor="w", justify="left").pack(
                         fill="x", padx=(Theme.S_LG, Theme.S_LG),
                         pady=(0, Theme.S_XS))

    def _toggle_advanced(self, event=None):
        """Toggle advanced settings visibility."""
        self.adv_visible = not self.adv_visible
        if self.adv_visible:
            self.adv_toggle.icon = "-"
            self.adv_toggle.set_text(tr("Hide advanced settings"))
            for panel, pack_options in getattr(
                self, "_inspector_detail_panels", ()
            ):
                panel.pack(**pack_options)
            self.adv_panel.pack(fill="x")
        else:
            self.adv_toggle.icon = "+"
            self.adv_toggle.set_text(tr("Advanced settings"))
            for panel, _pack_options in getattr(
                self, "_inspector_detail_panels", ()
            ):
                panel.pack_forget()
            self.adv_panel.pack_forget()
        if hasattr(self, "_sync_inspector_disclosure_state"):
            self._sync_inspector_disclosure_state()

    def _get_algo_description(self) -> str:
        """Get description for current algorithm."""
        descriptions = {
            "Auto": tr("Automatically chooses the best cleanup method for each file."),
            "STTN": tr("Fast temporal recovery for live action and changing backgrounds."),
            "LAMA": tr("Detailed single-frame fill for stills, animation, and clean scenes."),
            "ProPainter": tr("High-quality temporal fill for motion-heavy footage and thick text."),
        }
        return descriptions.get(self.mode_var.get(), "")

    def _on_mode_changed(self, event=None):
        """Handle algorithm mode change."""
        self.config.mode = InpaintMode(self.mode_var.get())
        self.algo_desc.config(text=self._get_algo_description())
        self._update_mode_options()
        self._update_status(
            tr("Switched to the {profile} profile").format(
                profile=self.mode_var.get()))

    def _on_mode_picker_changed(self, value: str):
        """Segmented picker callback -- keep `mode_var` and the combobox path
        compatible."""
        self.mode_var.set(value)
        self._on_mode_changed()

    def _on_preset_applied(self, event=None):
        """Apply the chosen preset to the live config and refresh the UI."""
        name = self.preset_var.get()
        if name == "(custom)":
            return
        if not apply_preset(self.config, name):
            self._update_status(f"Preset '{name}' not found", "warning")
            return
        # Reflect preset changes in the mode picker + toggle vars that back
        # the detection / quality / output cards. The dataclass carries the
        # authoritative state; just push it out to every widget we track.
        self.mode_var.set(self.config.mode.value)
        try:
            self.mode_picker.set(self.config.mode.value)
        except Exception:
            pass
        for attr, field in (
            ("auto_band_var", "auto_band"),
            ("flow_warp_var", "tbe_flow_warp"),
            ("scene_split_var", "tbe_scene_cut_split"),
            ("kalman_var", "kalman_tracking"),
            ("phash_var", "phash_skip_enable"),
            ("colour_tune_var", "colour_tune_enable"),
            ("adaptive_batch_var", "adaptive_batch"),
            ("temporal_mask_union_var", "temporal_mask_union"),
            ("export_srt_var", "export_srt"),
            ("export_mask_var", "export_mask_video"),
            ("mask_export_format_var", "mask_export_format"),
            ("mask_import_mode_var", "mask_import_mode"),
            ("language_filter_var", "language_mask_filter"),
        ):
            if hasattr(self, attr):
                getattr(self, attr).set(getattr(self.config, field))
        for field, (slider, value_label) in getattr(
            self, "_settings_slider_by_attr", {}
        ).items():
            value = int(getattr(self.config, field))
            slider.set_value(value)
            value_label.config(text=str(value))
        if hasattr(self, "mask_import_label_var"):
            self.mask_import_label_var.set(
                Path(self.config.mask_import_path).name
                if self.config.mask_import_path else tr("No imported matte"))
        self._on_mode_changed()
        save_settings(self.config)
        self._update_status(f"Applied preset '{name}'", "success")

    def _choose_mask_import_manifest(self):
        path = filedialog.askopenfilename(
            parent=self.root,
            title=tr("Import edited mask / alpha matte"),
            filetypes=[
                (tr("VSR matte manifest"), "*.mask.json"),
                (tr("JSON manifest"), "*.json"),
                (tr("All files"), "*.*"),
            ],
        )
        if not path:
            return
        try:
            from backend.matte_interchange import inspect_matte_manifest

            info = inspect_matte_manifest(path)
        except Exception as exc:
            self._update_status(
                tr("Matte manifest could not be imported: {error}").format(
                    error=exc),
                "warning",
            )
            return
        self.config.mask_import_path = str(path)
        if hasattr(self, "mask_import_label_var"):
            self.mask_import_label_var.set(Path(path).name)
        self._apply_current_settings_to_idle_items()
        save_settings(self.config)
        self._update_status(
            tr("Imported {format} matte manifest ({frames} frames)").format(
                format=str(info["format"]).upper(),
                frames=info["frame_count"],
            ),
            "success",
        )

    def _choose_translated_srt(self):
        path = filedialog.askopenfilename(
            parent=self.root,
            title=tr("Choose translated subtitle file"),
            filetypes=[
                (tr("SubRip subtitles"), "*.srt"),
                (tr("All files"), "*.*"),
            ],
        )
        if not path:
            return
        try:
            from backend.subtitle_translation import provided_translation_evidence
            provided_translation_evidence(path)
        except Exception as exc:
            self._update_status(
                tr("Translated SRT could not be used: {error}").format(error=exc),
                "warning",
            )
            return
        self.translation_srt_var.set(path)
        self.translation_srt_label_var.set(Path(path).name)
        self.translation_enabled_var.set(True)
        self._update_status(tr("Translated SRT ready to re-embed."), "success")

    def _choose_translation_command(self):
        path = filedialog.askopenfilename(
            parent=self.root,
            title=tr("Choose local translation command"),
            filetypes=[
                (tr("Executables and Python scripts"), "*.exe *.py"),
                (tr("All files"), "*.*"),
            ],
        )
        if path:
            self.translation_command_var.set(path)

    def _clear_translated_srt(self):
        self.translation_srt_var.set("")
        self.translation_srt_label_var.set(tr("No translated SRT selected"))

    def _clear_mask_import_manifest(self):
        self.config.mask_import_path = ""
        if hasattr(self, "mask_import_label_var"):
            self.mask_import_label_var.set(tr("No imported matte"))
        self._apply_current_settings_to_idle_items()
        save_settings(self.config)
        self._update_status(tr("Cleared imported matte"), "info")

    def _export_preset_dialog(self):
        """Export the currently-selected preset to a shareable JSON file."""
        try:
            from tkinter import filedialog
            name = self.preset_var.get()
            if name == "(custom)":
                self._update_status("Pick a preset first, then export", "warning")
                return
            path = filedialog.asksaveasfilename(
                parent=self.root,
                title=f"Export preset '{name}'",
                defaultextension=".json",
                filetypes=[("VSR preset", "*.json"), ("All files", "*.*")],
                initialfile=f"{name.replace('/', '-')}.vsr-preset.json",
            )
            if not path:
                return
            if export_preset(name, path):
                self._update_status(f"Exported '{name}' to {Path(path).name}", "success")
            else:
                self._update_status("Export failed", "error")
        except Exception as exc:
            self._update_status(f"Export failed: {exc}", "error")

    def _import_preset_dialog(self):
        """Import a preset JSON into the user library and select it."""
        try:
            from tkinter import filedialog
            path = filedialog.askopenfilename(
                parent=self.root,
                title="Import preset",
                filetypes=[("VSR preset", "*.json"), ("All files", "*.*")],
            )
            if not path:
                return
            new_name = import_preset(path)
            if new_name is None:
                self._update_status("Not a valid VSR preset file", "error")
                return
            self.preset_combo['values'] = ["(custom)"] + [n for n, _ in list_presets()]
            self.preset_var.set(new_name)
            self._on_preset_applied()
            notice = consume_preset_import_notice()
            if notice:
                self._update_status(
                    f"Imported preset '{new_name}'. {notice}",
                    "warning",
                )
            else:
                self._update_status(f"Imported preset '{new_name}'", "success")
        except Exception as exc:
            self._update_status(f"Import failed: {exc}", "error")

    def _prompt_preset_details(self) -> Optional[Tuple[str, str]]:
        """Open a themed modal for naming and describing a user preset."""
        result = {"value": None}

        dialog = tk.Toplevel(self.root)
        dialog.withdraw()
        dialog.title(tr("Save preset"))
        dialog.configure(bg=Theme.BG_OVERLAY)
        dialog.resizable(False, False)
        dialog.transient(self.root)
        try:
            from backend.a11y import set_accessible_metadata
            set_accessible_metadata(
                dialog,
                role="dialog",
                label=tr("Save preset"),
                state="modal",
                description=(
                    tr("Name the current cleanup settings and save them "
                       "to the user preset library.")
                ),
            )
        except Exception:
            pass

        outer = tk.Frame(dialog, bg=Theme.BORDER, padx=1, pady=1)
        outer.pack()
        body = tk.Frame(outer, bg=Theme.BG_SECONDARY)
        body.pack()

        content = tk.Frame(body, bg=Theme.BG_SECONDARY)
        content.pack(padx=28, pady=(24, 14))

        tk.Label(content, text=tr("Save the current setup as a preset"),
                 font=f(Theme.F_HEADING, "bold"),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY).pack(anchor="w")
        tk.Label(content,
                 text=tr("Use a short name you will recognize later. Saving to an existing user preset name will update it."),
                 font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED,
                 justify="left", wraplength=420).pack(anchor="w", pady=(6, Theme.S_LG))

        form = tk.Frame(content, bg=Theme.BG_SECONDARY)
        form.pack(fill="x")

        def entry_row(label_text: str, initial: str = ""):
            row = tk.Frame(form, bg=Theme.BG_SECONDARY)
            row.pack(fill="x", pady=(0, Theme.S_MD))
            tk.Label(row, text=tr(label_text), font=f(Theme.F_BODY_SM),
                     bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY).pack(anchor="w")
            entry = tk.Entry(
                row, bg=Theme.BG_TERTIARY, fg=Theme.TEXT_PRIMARY,
                insertbackground=Theme.TEXT_PRIMARY,
                font=f(Theme.F_BODY_SM), relief="flat", bd=6,
                highlightthickness=1, highlightbackground=Theme.BORDER,
                highlightcolor=Theme.BORDER_FOCUS,
            )
            entry.pack(fill="x", pady=(Theme.S_XS, 0))
            entry.insert(0, initial)
            return entry

        name_entry = entry_row("Preset name")
        desc_entry = entry_row("Description", tr("User preset"))

        helper = tk.Label(content, text=tr("Built-in preset names are reserved."),
                          font=f(Theme.F_META), bg=Theme.BG_SECONDARY,
                          fg=Theme.TEXT_MUTED)
        helper.pack(anchor="w")

        error_label = tk.Label(content, text="", font=f(Theme.F_META, "bold"),
                               bg=Theme.BG_SECONDARY, fg=Theme.ERROR)
        error_label.pack(anchor="w", pady=(Theme.S_SM, 0))

        actions = tk.Frame(body, bg=Theme.BG_CARD)
        actions.pack(fill="x")
        actions_inner = tk.Frame(actions, bg=Theme.BG_CARD)
        actions_inner.pack(side="right", padx=16, pady=14)

        def _cancel():
            dialog.grab_release()
            dialog.destroy()

        def _submit():
            name = name_entry.get().strip()
            description = desc_entry.get().strip() or tr("User preset")
            if not name:
                error_label.config(text=tr("Give this preset a short name."))
                name_entry.focus_set()
                try:
                    from backend.a11y import announce
                    announce(tr("Give this preset a short name."), importance="high")
                except Exception:
                    pass
                return
            if name in BUILTIN_PRESETS:
                error_label.config(text=tr("Built-in preset names are reserved."))
                name_entry.focus_set()
                try:
                    from backend.a11y import announce
                    announce(tr("Built-in preset names are reserved."), importance="high")
                except Exception:
                    pass
                return
            result["value"] = (name, description)
            dialog.grab_release()
            dialog.destroy()

        ModernButton(actions_inner, text=tr("Cancel"), width=96,
                     command=_cancel, style="ghost", size="md").pack(side="left")
        ModernButton(actions_inner, text=tr("Save preset"), width=120,
                     command=_submit, style="primary", size="md").pack(
                         side="left", padx=(Theme.S_SM, 0))

        dialog.bind("<Escape>", lambda e: _cancel())
        dialog.bind("<Return>", lambda e: _submit())
        dialog.protocol("WM_DELETE_WINDOW", _cancel)

        dialog.update_idletasks()
        try:
            px = self.root.winfo_rootx()
            py = self.root.winfo_rooty()
            pw = self.root.winfo_width()
            ph = self.root.winfo_height()
            dw = dialog.winfo_reqwidth()
            dh = dialog.winfo_reqheight()
            dialog.geometry(f"+{px + (pw - dw) // 2}+{py + (ph - dh) // 3}")
        except Exception:
            pass

        dialog.deiconify()
        dialog.grab_set()
        name_entry.focus_set()
        dialog.wait_window()
        return result["value"]

    def _save_preset_dialog(self):
        """Prompt for a name + description and save a user preset."""
        try:
            details = self._prompt_preset_details()
            if not details:
                return
            name, description = details
            existing_user = name in _load_user_presets()
            self._sync_config_from_ui()
            if save_user_preset(name, description, self.config):
                # Refresh combo
                self.preset_combo['values'] = ["(custom)"] + [n for n, _ in list_presets()]
                self.preset_var.set(name)
                verb = "Updated" if existing_user else "Saved"
                self._update_status(f"{verb} preset '{name}'", "success")
            else:
                self._update_status(f"Could not save preset '{name}'", "error")
        except Exception as exc:
            self._update_status(f"Save preset failed: {exc}", "error")

    def _update_mode_options(self):
        """Enable/disable mode-specific toggles based on selected algorithm."""
        mode = self.mode_var.get()

        # Skip detection only for STTN
        if mode == "STTN":
            self.skip_check.set_enabled(True)
        else:
            self.skip_detection_var.set(False)
            self.skip_check.set_enabled(False)

        # LAMA fast only for LAMA
        if mode == "LAMA":
            self.lama_check.set_enabled(True)
        else:
            self.lama_fast_var.set(False)
            self.lama_check.set_enabled(False)

    def _set_lang_display(self, code: str):
        """Sync the friendly-name label to the underlying lang code."""
        for label, (c, _) in zip(self._lang_labels, self._lang_display):
            if c == code:
                self._lang_display_var.set(label)
                return
        # Unknown -- default to English
        self._lang_display_var.set(self._lang_labels[0])
        self.lang_var.set(self._lang_display[0][0])

    def _on_lang_changed(self, event=None):
        """Map selected friendly label back to the lang code."""
        label = self._lang_display_var.get()
        code = self._lang_by_label.get(label)
        if code:
            self.lang_var.set(code)
            self.config.detection_lang = code

    def _on_ocr_engine_changed(self, event=None):
        """Persist the selected detector and invalidate preview model caches."""
        del event
        self.config.detection_engine = self._ocr_engine_by_label.get(
            self.ocr_engine_var.get(), "auto")
        self._preview_detector = None
        self._preview_detector_lang = None
        self._preview_detector_engine = None

    def _on_gpu_changed(self, event=None):
        """Handle GPU device selection change."""
        selection = self.gpu_var.get()
        for i, gpu in enumerate(self.gpus):
            label = f"{gpu['name']} ({gpu['memory']})"
            if label == selection:
                self.config.gpu_id = gpu['index']
                self.config.use_gpu = True
                self._update_status(f"Compute device set to {gpu['name']}", "info")
                logger.info(f"GPU set to: {gpu['name']} (index {gpu['index']})")
                break

    def _choose_output_dir(self):
        """Let user pick a custom output directory."""
        d = filedialog.askdirectory(title=tr("Select Output Directory"))
        if d:
            self._output_dir = Path(d)
            self._update_output_label()
            refreshed = self._refresh_idle_output_paths()
            if refreshed:
                self._update_queue_display()
            message = "Custom output location selected"
            if refreshed:
                message += f". Updated {refreshed} idle output path{'s' if refreshed != 1 else ''}"
            self._update_status(message, "success")
            logger.info(f"Output directory: {self._output_dir}")

    def _choose_work_directory(self):
        """Choose and immediately persist a processing-work root."""
        initial = self.work_dir_var.get().strip() if hasattr(
            self, "work_dir_var") else ""
        selected = filedialog.askdirectory(
            title=tr("Select Processing Work Directory"),
            initialdir=initial or None,
        )
        if not selected:
            return
        self.work_dir_var.set(selected)
        self.config.work_directory = selected
        self._apply_current_settings_to_idle_items()
        save_settings(self.config)
        self._update_status(
            "Work directory selected for temporary and resume files",
            "success",
            toast=True,
        )

    def _reset_work_directory(self):
        """Return processing scratch storage to the system temp policy."""
        self.work_dir_var.set("")
        self.config.work_directory = ""
        self._apply_current_settings_to_idle_items()
        save_settings(self.config)
        self._update_status(
            "Work directory reset to the system temporary location",
            "info",
            toast=True,
        )

    def _reset_output_dir(self):
        """Reset output directory to default (input_dir/output/)."""
        self._output_dir = None
        self._update_output_label()
        refreshed = self._refresh_idle_output_paths()
        if refreshed:
            self._update_queue_display()
        message = "Output location reset to the default per-folder workflow"
        if refreshed:
            message += f". Updated {refreshed} idle output path{'s' if refreshed != 1 else ''}"
        self._update_status(message)
