from __future__ import annotations

import ctypes
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except ImportError:  # pragma: no cover - tkinter is optional in headless imports
    pass

try:
    from PIL import Image, ImageTk, ImageDraw, ImageFilter
    PIL_AVAILABLE = True
except ImportError:  # pragma: no cover - preview features degrade without Pillow
    PIL_AVAILABLE = False

from gui.theme import Theme, f, mono
from gui.config import (
    APP_NAME, APP_VERSION, LOG_DIR, LOG_FILE, SETTINGS_FILE,
    InpaintMode, ProcessingConfig, ProcessingStatus, QueueItem,
    clear_queue_state, save_queue_state, status_ui,
)
from gui.utils import (
    _format_soft_subtitle_summary, format_quality_report, format_size,
    format_time, get_app_dir, get_file_info, is_image_file, is_video_file,
    summarize_quality_reports, truncate_middle,
)
from gui.widgets import (
    ModernButton, ModernProgressBar, TaskbarProgress, Tooltip,
    make_themed_menu, show_confirm,
)
from backend.ffmpeg_profiles import ffmpeg_profile_entries
from backend.i18n import tr
from backend.model_downloads import installed_backend_status
from backend.resume_checkpoint import ProcessingPaused
from backend.safe_image import safe_imread

logger = logging.getLogger(__name__)


class SupportControllerMixin:
    """Focused controller methods mixed into VideoSubtitleRemoverApp."""

    def _build_log_panel(self, parent):
        """Embedded, collapsible activity log."""
        log_section = self._create_surface(parent)
        log_section.pack(fill="x", pady=(Theme.S_MD, 0))

        log_header = tk.Frame(log_section, bg=Theme.BG_SECONDARY)
        log_header.pack(fill="x", padx=Theme.S_XL, pady=(Theme.S_MD, 0))

        # Title cluster (left)
        title_cluster = tk.Frame(log_header, bg=Theme.BG_SECONDARY)
        title_cluster.pack(side="left")
        tk.Label(title_cluster, text=tr("ACTIVITY"), font=f(Theme.F_EYEBROW, "bold"),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED).pack(anchor="w")
        tk.Label(title_cluster, text=tr("Runtime log"),
                 font=f(Theme.F_BODY, "bold"),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY).pack(anchor="w", pady=(2, 0))

        # Level badges: warn / error counts, packed in a row between title and toggle
        self._badge_row = tk.Frame(log_header, bg=Theme.BG_SECONDARY)
        self._badge_row.pack(side="left", padx=(Theme.S_MD, 0))
        self._log_warn_badge = tk.Label(
            self._badge_row, text="", font=f(Theme.F_META, "bold"),
            bg=Theme.WARNING_BG, fg=Theme.WARNING, padx=8, pady=3)
        self._log_error_badge = tk.Label(
            self._badge_row, text="", font=f(Theme.F_META, "bold"),
            bg=Theme.ERROR_BG, fg=Theme.ERROR, padx=8, pady=3)

        self._log_visible = True
        self._log_toggle_btn = ModernButton(log_header, text=tr("Hide activity"), width=120,
                                            command=self._toggle_log_panel,
                                            style="ghost", size="sm")
        self._log_toggle_btn.pack(side="left", padx=(Theme.S_MD, 0))

        open_log_btn = ModernButton(
            log_header, text=tr("Open log file"), width=118,
            command=self._open_log_file,
            style="ghost", size="sm")
        open_log_btn.pack(side="right")

        clear_log_btn = ModernButton(log_header, text=tr("Clear"), width=72,
                                     command=self._clear_log,
                                     style="ghost", size="sm")
        clear_log_btn.pack(side="right", padx=(0, Theme.S_SM))

        self._log_body = tk.Frame(log_section, bg=Theme.BG_LOG,
                                  highlightthickness=1,
                                  highlightbackground=Theme.BORDER_SUBTLE)
        self._log_body.pack(fill="x", padx=Theme.S_XL, pady=(Theme.S_SM, Theme.S_LG))

        self.log_text = tk.Text(self._log_body, height=6, bg=Theme.BG_LOG,
                                fg=Theme.TEXT_SECONDARY, font=mono(Theme.F_BODY_SM),
                                relief="flat", bd=8, state="disabled",
                                wrap="word", insertbackground=Theme.TEXT_PRIMARY,
                                selectbackground=Theme.BLUE_MUTED)
        log_scroll = ttk.Scrollbar(self._log_body, orient="vertical",
                                   command=self.log_text.yview,
                                   style="Dark.Vertical.TScrollbar")
        self.log_text.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side="right", fill="y")
        self.log_text.pack(side="left", fill="both", expand=True)

        # Tag colors
        self.log_text.tag_configure("info", foreground=Theme.TEXT_MUTED)
        self.log_text.tag_configure("warning", foreground=Theme.WARNING)
        self.log_text.tag_configure("error", foreground=Theme.ERROR)

        # Initialize closed-state toggle (no flip on first run)
        # We start visible, so text stays "Hide activity"

    def _toggle_log_panel(self):
        """Toggle log panel visibility."""
        self._log_visible = not self._log_visible
        if self._log_visible:
            self._log_body.pack(fill="x", padx=Theme.S_XL, pady=(Theme.S_SM, Theme.S_LG))
            self._log_toggle_btn.set_text(tr("Hide activity"))
        else:
            self._log_body.pack_forget()
            self._log_toggle_btn.set_text(tr("Show activity"))

    def _update_log_badges(self, warn_count: int, error_count: int):
        """Show/hide warn/error count pills in the log header (always before toggle)."""
        try:
            if warn_count > 0:
                self._log_warn_badge.config(
                    text=tr("{count} warning{suffix}").format(
                        count=warn_count,
                        suffix="s" if warn_count != 1 else ""))
                self._log_warn_badge.pack(side="left", padx=(0, Theme.S_XS))
            else:
                self._log_warn_badge.pack_forget()
            if error_count > 0:
                self._log_error_badge.config(
                    text=tr("{count} error{suffix}").format(
                        count=error_count,
                        suffix="s" if error_count != 1 else ""))
                self._log_error_badge.pack(side="left", padx=(0, Theme.S_XS))
            else:
                self._log_error_badge.pack_forget()
        except Exception:
            pass

    def _clear_log(self):
        """Clear the log panel."""
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")
        if hasattr(self, "_log_handler"):
            self._log_handler.reset_counts()
        self._update_status(tr("Activity log cleared"))

    def _open_log_file(self):
        """Reveal the current log file in the system shell."""
        if not LOG_FILE.exists():
            self._update_status(tr("The log file is not available yet"), "warning")
            return
        try:
            os.startfile(str(LOG_FILE))
            self._update_status(tr("Opened the log file"), "info")
        except Exception:
            self._update_status(tr("The log file could not be opened"), "warning")

    def _open_settings_folder(self):
        try:
            os.startfile(str(LOG_DIR))
            self._update_status(tr("Opened the settings folder"), "info")
        except Exception:
            self._update_status(tr("The settings folder could not be opened"), "warning")

    def _save_support_bundle(self):
        """Save a redacted diagnostics zip for bug reports."""
        try:
            initial = (
                "vsr-support-"
                + datetime.now().strftime("%Y%m%d-%H%M%S")
                + ".zip"
            )
            path = filedialog.asksaveasfilename(
                parent=self.root,
                title=tr("Save support bundle"),
                defaultextension=".zip",
                filetypes=[(tr("Support bundle"), "*.zip"), (tr("All files"), "*.*")],
                initialfile=initial,
            )
            if not path:
                return
            from backend.support_bundle import create_support_bundle
            bundle = create_support_bundle(
                path,
                settings_path=SETTINGS_FILE,
                log_path=LOG_FILE,
                batch_report_paths=getattr(self, "_last_batch_report_paths", []),
                app_version=APP_VERSION,
                extra_facts={
                    "ffmpeg_ready": self.ffmpeg_ready,
                    "detection_engines": self.ai_engines.get("detection", []),
                    "inpainting_engines": self.ai_engines.get("inpainting", []),
                    "gpu_count": len(self.gpus),
                    "gpus": self.gpus,
                    "queue_count": len(self.queue),
                },
            )
            self._update_status(
                tr("Saved redacted support bundle to {name}").format(
                    name=Path(bundle).name),
                "success",
            )
        except Exception as exc:
            logger.warning("Support bundle save failed: %s", exc, exc_info=True)
            self._update_status(tr("Support bundle could not be saved"), "error")

    @staticmethod
    def _model_cache_missing_summary(status: dict) -> str:
        missing = list((status or {}).get("missing_known_filenames", []) or [])
        if not missing:
            return ""
        shown = ", ".join(missing[:3])
        if len(missing) > 3:
            shown += f", +{len(missing) - 3} more"
        return f"; missing optional assets: {shown}"

    def _export_model_cache_bundle(self):
        """Export verified model-cache files to a portable zip."""
        try:
            initial = (
                "vsr-model-cache-"
                + datetime.now().strftime("%Y%m%d-%H%M%S")
                + ".zip"
            )
            path = filedialog.asksaveasfilename(
                parent=self.root,
                title=tr("Export model cache"),
                defaultextension=".zip",
                filetypes=[(tr("Model cache bundle"), "*.zip"), (tr("All files"), "*.*")],
                initialfile=initial,
            )
            if not path:
                return
            from backend.cache_inventory import export_model_cache_bundle
            result = export_model_cache_bundle(path)
            missing = self._model_cache_missing_summary(
                result.get("status_after_export", {})
            )
            skipped = len(result.get("skipped", []) or [])
            suffix = f"; skipped {skipped} unsafe or invalid file(s)" if skipped else ""
            self._update_status(
                f"Exported {len(result.get('files', []))} model-cache file(s) "
                f"to {Path(result['output']).name}{suffix}{missing}",
                "warning" if skipped else "success",
            )
        except Exception as exc:
            logger.warning("Model cache export failed: %s", exc, exc_info=True)
            self._update_status("Model cache could not be exported", "error")

    def _import_model_cache_bundle(self):
        """Import a portable model-cache zip into the app model cache."""
        try:
            path = filedialog.askopenfilename(
                parent=self.root,
                title="Import model cache",
                filetypes=[("Model cache bundle", "*.zip"), ("All files", "*.*")],
            )
            if not path:
                return
            from backend.cache_inventory import import_model_cache_bundle
            result = import_model_cache_bundle(path)
            try:
                self.backend_status = installed_backend_status(self.config)
            except Exception:
                logger.warning("Backend status refresh after cache import failed",
                               exc_info=True)
            missing = self._model_cache_missing_summary(
                result.get("status_after_import", {})
            )
            rejected = len(result.get("rejected", []) or [])
            suffix = (
                f"; rejected {rejected} unsafe or invalid file(s)"
                if rejected else ""
            )
            self._update_status(
                f"Imported {len(result.get('imported', []))} model-cache file(s)"
                f"{suffix}{missing}",
                "warning" if rejected else "success",
            )
        except Exception as exc:
            logger.warning("Model cache import failed: %s", exc, exc_info=True)
            self._update_status("Model cache could not be imported", "error")

    def _open_model_cache_menu(self, anchor):
        """Open model-cache actions from the About dialog."""
        menu = make_themed_menu(self.root)
        menu.add_command(
            label=tr("Export model cache..."),
            command=self._export_model_cache_bundle,
        )
        menu.add_command(
            label=tr("Import model cache..."),
            command=self._import_model_cache_bundle,
        )
        try:
            menu.tk_popup(
                anchor.winfo_rootx(),
                anchor.winfo_rooty() + anchor.winfo_height() + 2,
            )
        finally:
            menu.grab_release()

    @staticmethod
    def _backend_status_tone_color(tone: str) -> str:
        return {
            "success": Theme.SUCCESS,
            "warning": Theme.WARNING,
            "error": Theme.ERROR,
            "info": Theme.INFO,
            "neutral": Theme.TEXT_SECONDARY,
        }.get(str(tone or "").lower(), Theme.TEXT_SECONDARY)

    def _build_backend_status_panel(self, parent):
        """Render installed backend/model status in the About dialog."""
        status = getattr(self, "backend_status", {}) or {}
        summary = status.get("summary", {}) if isinstance(status, dict) else {}
        rows = [
            (tr("Detection"), summary.get("detection") or tr("Unknown")),
            (tr("Inpainting"), summary.get("inpainting") or tr("Unknown")),
            (tr("Providers"), summary.get("providers") or tr("Unknown")),
            (tr("Languages"), summary.get("language_support") or tr("Unknown")),
            (tr("Model files"), summary.get("model_files") or tr("Unknown")),
            (tr("Hash status"), summary.get("hash_status") or tr("Unknown")),
            (tr("Next action"), summary.get("next_action") or tr("No action needed.")),
        ]
        profile_rows = [
            (entry["name"], entry["available"], entry["reason"])
            for entry in ffmpeg_profile_entries(
                getattr(self, "ffmpeg_profiles", None)
            )
        ]
        profile_labels = {
            "basic": tr("FFmpeg basic"),
            "advanced_quality": tr("FFmpeg quality"),
            "speech_fallback": tr("FFmpeg speech"),
            "modern_codec": tr("FFmpeg codecs"),
        }
        for name, available, reason in profile_rows:
            rows.append((
                profile_labels.get(name, tr("FFmpeg {name}").format(name=name)),
                (tr("ready") if available else reason),
            ))
        card = tk.Frame(parent, bg=Theme.BG_CARD, highlightthickness=1,
                        highlightbackground=Theme.BORDER_SUBTLE)
        card.pack(fill="x", pady=(Theme.S_MD, 0))

        header = tk.Frame(card, bg=Theme.BG_CARD)
        header.pack(fill="x", padx=14, pady=(10, 4))
        tk.Label(
            header,
            text=tr("BACKEND STATUS"),
            font=f(Theme.F_EYEBROW, "bold"),
            bg=Theme.BG_CARD,
            fg=Theme.TEXT_MUTED,
        ).pack(side="left")
        tone = str(summary.get("tone") or "neutral")
        tk.Label(
            header,
            text=tone.upper(),
            font=f(Theme.F_META, "bold"),
            bg=Theme.BG_TERTIARY,
            fg=self._backend_status_tone_color(tone),
            padx=8,
            pady=2,
        ).pack(side="right")

        grid = tk.Frame(card, bg=Theme.BG_CARD)
        grid.pack(fill="x", padx=14, pady=(0, 10))
        grid.columnconfigure(1, weight=1)
        for row_idx, (label, value) in enumerate(rows):
            tk.Label(
                grid,
                text=label,
                font=f(Theme.F_BODY_SM),
                bg=Theme.BG_CARD,
                fg=Theme.TEXT_MUTED,
                anchor="w",
                width=12,
            ).grid(row=row_idx, column=0, sticky="nw", pady=3)
            row_tone = (
                self._backend_status_tone_color(tone)
                if label == tr("Next action") else Theme.TEXT_PRIMARY
            )
            tk.Label(
                grid,
                text=str(value),
                font=f(Theme.F_BODY_SM, "bold" if row_idx < 2 else "normal"),
                bg=Theme.BG_CARD,
                fg=row_tone,
                anchor="w",
                justify="left",
                wraplength=430,
            ).grid(row=row_idx, column=1, sticky="ew",
                   pady=3, padx=(Theme.S_SM, 0))

    def _show_about(self):
        """Open a themed About dialog with version, credits, and quick links."""
        dialog = tk.Toplevel(self.root)
        dialog.withdraw()
        dialog.title(tr("About {app_name}").format(app_name=APP_NAME))
        dialog.configure(bg=Theme.BG_OVERLAY)
        dialog.resizable(False, False)
        dialog.transient(self.root)
        try:
            from backend.a11y import set_accessible_metadata
            status = (self.backend_status or {}).get("summary", {})
            set_accessible_metadata(
                dialog,
                role="dialog",
                label=tr("About {app_name}").format(app_name=APP_NAME),
                state="modal",
                description=str(status.get("next_action") or tr("Backend status and app version.")),
            )
        except Exception:
            pass

        def _close_about():
            try:
                dialog.grab_release()
            except tk.TclError:
                pass
            dialog.destroy()

        outer = tk.Frame(dialog, bg=Theme.BORDER, padx=1, pady=1)
        outer.pack()
        body = tk.Frame(outer, bg=Theme.BG_SECONDARY)
        body.pack()

        content = tk.Frame(body, bg=Theme.BG_SECONDARY)
        content.pack(padx=32, pady=(28, 14))

        # Brand row
        brand_row = tk.Frame(content, bg=Theme.BG_SECONDARY)
        brand_row.pack(anchor="w")
        if self._brand_photo:
            tk.Label(brand_row, image=self._brand_photo,
                     bg=Theme.BG_SECONDARY).pack(side="left", padx=(0, Theme.S_MD))
        title_stack = tk.Frame(brand_row, bg=Theme.BG_SECONDARY)
        title_stack.pack(side="left")
        tk.Label(title_stack, text=APP_NAME, font=f(Theme.F_HEADING, "bold"),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY).pack(anchor="w")
        tk.Label(title_stack, text=tr("Version {version}").format(version=APP_VERSION),
                 font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED).pack(anchor="w", pady=(2, 0))

        # Fact rows
        fact_card = tk.Frame(content, bg=Theme.BG_CARD, highlightthickness=1,
                             highlightbackground=Theme.BORDER_SUBTLE)
        fact_card.pack(fill="x", pady=(Theme.S_LG, 0))

        def fact(label, value, tone=Theme.TEXT_PRIMARY):
            row = tk.Frame(fact_card, bg=Theme.BG_CARD)
            row.pack(fill="x", padx=14, pady=6)
            tk.Label(row, text=tr(label), font=f(Theme.F_BODY_SM),
                     bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED).pack(side="left")
            display = truncate_middle(str(value), 58)
            value_label = tk.Label(row, text=display,
                                   font=f(Theme.F_BODY_SM, "bold"),
                                   bg=Theme.BG_CARD, fg=tone)
            value_label.pack(side="right")
            if display != str(value):
                Tooltip(value_label, str(value))

        det_label = ", ".join(self.ai_engines["detection"]) or tr("None")
        inp_label = ", ".join(self.ai_engines["inpainting"]) or tr("None")
        gpu_count = len(self.gpus)
        gpu_label = (
            tr("{count} GPU{suffix}").format(
                count=gpu_count, suffix="s" if gpu_count != 1 else "")
            if self.gpus else tr("CPU only")
        )

        fact(tr("Detection engines"), det_label, Theme.INFO)
        fact(tr("Inpainting engines"), inp_label, Theme.SUCCESS)
        fact(tr("Compute"), gpu_label,
             Theme.SUCCESS if self.gpus else Theme.WARNING)
        fact(tr("FFmpeg"), tr("Ready") if self.ffmpeg_ready else tr("Missing"),
             Theme.SUCCESS if self.ffmpeg_ready else Theme.WARNING)
        fact(tr("Input"), tr("Click Import or drag files onto the queue"))
        fact(tr("Settings"), str(SETTINGS_FILE))
        fact(tr("Log file"), str(LOG_FILE))
        try:
            from backend.cache_inventory import discover_caches, _format_bytes
            total = sum(e.total_bytes for e in discover_caches())
            fact(tr("Disk cache"), _format_bytes(total))
        except Exception:
            fact(tr("Disk cache"), tr("unavailable"))

        self._build_backend_status_panel(content)

        # Action row
        actions = tk.Frame(body, bg=Theme.BG_CARD)
        actions.pack(fill="x")
        actions_inner = tk.Frame(actions, bg=Theme.BG_CARD)
        actions_inner.pack(side="right", padx=16, pady=14)

        ModernButton(actions_inner, text=tr("Open log"), width=96,
                     command=self._open_log_file, style="ghost", size="md").pack(side="left")
        model_cache_btn = ModernButton(actions_inner, text=tr("Model cache"), width=116,
                                       command=None, style="ghost", size="md")
        model_cache_btn.command = (
            lambda btn=model_cache_btn: self._open_model_cache_menu(btn)
        )
        model_cache_btn.pack(side="left", padx=(Theme.S_SM, 0))
        ModernButton(actions_inner, text=tr("Support bundle"), width=128,
                     command=self._save_support_bundle, style="ghost",
                     size="md").pack(side="left", padx=(Theme.S_SM, 0))
        ModernButton(actions_inner, text=tr("Settings folder"), width=132,
                     command=self._open_settings_folder, style="ghost",
                     size="md").pack(side="left", padx=(Theme.S_SM, 0))
        ModernButton(actions_inner, text=tr("Close"), width=84,
                     command=_close_about,
                     style="primary", size="md").pack(side="left", padx=(Theme.S_SM, 0))

        dialog.bind("<Escape>", lambda e: _close_about())
        dialog.protocol("WM_DELETE_WINDOW", _close_about)

        dialog.update_idletasks()
        try:
            px, py = self.root.winfo_rootx(), self.root.winfo_rooty()
            pw, ph = self.root.winfo_width(), self.root.winfo_height()
            dw, dh = dialog.winfo_reqwidth(), dialog.winfo_reqheight()
            dialog.geometry(f"+{px + (pw - dw) // 2}+{py + (ph - dh) // 3}")
        except Exception:
            pass
        dialog.deiconify()
        dialog.grab_set()

