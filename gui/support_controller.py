from __future__ import annotations

import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

try:
    import tkinter as tk
    from tkinter import ttk, filedialog
except ImportError:  # pragma: no cover - tkinter is optional in headless imports
    pass

from gui.theme import Theme, f, mono
from gui.config import (
    APP_NAME, APP_VERSION, LOG_DIR, LOG_FILE, SETTINGS_FILE,
)
from gui.utils import (
    dispatch_to_ui,
    truncate_middle,
)
from gui.widgets import (
    ModernButton, Tooltip,
    make_themed_menu,
)
from gui.dialog_layout import (
    fit_dialog_to_work_area,
    scrollable_dialog_body,
)
from backend.ffmpeg_profiles import ffmpeg_profile_entries
from backend.i18n import ntr, tr
from backend.model_downloads import installed_backend_status

logger = logging.getLogger(__name__)


class SupportControllerHost(Protocol):
    """Root and surface factories required by the support controller."""

    root: Any
    config: Any

    def _create_surface(self, parent, bg: str = Theme.BG_SECONDARY):
        ...


class SupportControllerMixin:
    """Focused controller methods mixed into VideoSubtitleRemoverApp."""

    def _build_log_panel(self, parent):
        """Embedded, collapsible activity log."""
        log_section = self._create_surface(parent)
        log_section.pack(side="bottom", fill="x", pady=(Theme.S_MD, 0))
        self._log_section = log_section

        log_header = tk.Frame(log_section, bg=Theme.BG_SECONDARY)
        log_header.pack(fill="x", padx=Theme.S_XL, pady=(Theme.S_MD, 0))

        # Title cluster (left)
        title_cluster = tk.Frame(log_header, bg=Theme.BG_SECONDARY)
        title_cluster.pack(side="left")
        self._log_title_cluster = title_cluster
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
        self._log_open_btn = open_log_btn

        clear_log_btn = ModernButton(log_header, text=tr("Clear"), width=72,
                                     command=self._clear_log,
                                     style="ghost", size="sm")
        clear_log_btn.pack(side="right", padx=(0, Theme.S_SM))
        self._log_clear_btn = clear_log_btn

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
            if not self._log_section.winfo_manager():
                self._log_section.pack(
                    side="bottom", fill="x", pady=(Theme.S_MD, 0))
            self._log_body.pack(fill="x", padx=Theme.S_XL, pady=(Theme.S_SM, Theme.S_LG))
            self._log_toggle_btn.set_text(tr("Hide activity"))
            if hasattr(self, "_footer_activity_btn"):
                self._footer_activity_btn.set_text(tr("Hide activity"))
        else:
            self._log_body.pack_forget()
            self._log_section.pack_forget()
            self._log_toggle_btn.set_text(tr("Show activity"))
            if hasattr(self, "_footer_activity_btn"):
                self._footer_activity_btn.set_text(tr("Activity"))

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

    def _run_support_task(self, busy_message: str, error_message: str,
                          work, describe_result) -> None:
        """Run heavy diagnostics I/O off the Tk main thread with feedback.

        ``work()`` runs on a worker thread and returns a result;
        ``describe_result(result) -> (text, tone)`` and all status updates run
        back on the main loop. Bundling/zipping model caches or logs can take
        seconds and hundreds of MB, which must not freeze the event loop.
        """
        self._update_status(tr(busy_message), "info")

        def _worker():
            try:
                result = work()
            except Exception as exc:  # noqa: BLE001 - surfaced to the user
                logger.warning("%s: %s", error_message, exc, exc_info=True)
                self.root.after(
                    0, lambda: self._update_status(tr(error_message), "error"))
                return

            def _done():
                try:
                    text, tone = describe_result(result)
                    self._update_status(text, tone)
                except Exception:  # noqa: BLE001
                    self._update_status(tr(error_message), "error")

            dispatch_to_ui(self.root, _done)

        threading.Thread(
            target=_worker, name="vsr-support-task", daemon=True
        ).start()

    def _save_support_bundle(self):
        """Save a redacted diagnostics zip for bug reports."""
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
        extra_facts = {
            "ffmpeg_ready": self.ffmpeg_ready,
            "detection_engines": self.ai_engines.get("detection", []),
            "inpainting_engines": self.ai_engines.get("inpainting", []),
            "gpu_count": len(self.gpus),
            "gpus": self.gpus,
            "queue_count": len(self.queue),
        }
        report_paths = list(getattr(self, "_last_batch_report_paths", []))

        def _work():
            from backend.support_bundle import create_support_bundle
            return create_support_bundle(
                path,
                settings_path=SETTINGS_FILE,
                log_path=LOG_FILE,
                batch_report_paths=report_paths,
                app_version=APP_VERSION,
                extra_facts=extra_facts,
            )

        def _describe(bundle):
            return (
                tr("Saved redacted support bundle to {name}").format(
                    name=Path(bundle).name),
                "success",
            )

        self._run_support_task(
            "Building support bundle...",
            "Support bundle could not be saved",
            _work, _describe,
        )

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

        def _work():
            from backend.cache_inventory import export_model_cache_bundle
            return export_model_cache_bundle(path)

        def _describe(result):
            missing = self._model_cache_missing_summary(
                result.get("status_after_export", {})
            )
            skipped = len(result.get("skipped", []) or [])
            suffix = ""
            if skipped:
                suffix = "; " + ntr(
                    "skipped {n} unsafe or invalid file",
                    "skipped {n} unsafe or invalid files",
                    skipped,
                ).format(n=skipped)
            exported = len(result.get("files", []))
            summary = ntr(
                "Exported {n} model-cache file",
                "Exported {n} model-cache files",
                exported,
            ).format(n=exported)
            return (
                f"{summary} to {Path(result['output']).name}{suffix}{missing}",
                "warning" if skipped else "success",
            )

        self._run_support_task(
            "Exporting model cache...",
            "Model cache could not be exported",
            _work, _describe,
        )

    def _import_model_cache_bundle(self):
        """Import a portable model-cache zip into the app model cache."""
        path = filedialog.askopenfilename(
            parent=self.root,
            title=tr("Import model cache"),
            filetypes=[(tr("Model cache bundle"), "*.zip"), (tr("All files"), "*.*")],
        )
        if not path:
            return

        def _work():
            from backend.cache_inventory import import_model_cache_bundle
            result = import_model_cache_bundle(path)
            try:
                self.backend_status = installed_backend_status(self.config)
            except Exception:
                logger.warning("Backend status refresh after cache import failed",
                               exc_info=True)
            return result

        def _describe(result):
            missing = self._model_cache_missing_summary(
                result.get("status_after_import", {})
            )
            rejected = len(result.get("rejected", []) or [])
            suffix = (
                f"; rejected {rejected} unsafe or invalid file(s)"
                if rejected else ""
            )
            return (
                f"Imported {len(result.get('imported', []))} model-cache file(s)"
                f"{suffix}{missing}",
                "warning" if rejected else "success",
            )

        self._run_support_task(
            "Importing model cache...",
            "Model cache could not be imported",
            _work, _describe,
        )

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
            menu.destroy()

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

    def _last_execution_summary(self):
        """Most recent recorded execution provenance across the queue."""
        best = None
        for item in reversed(list(getattr(self, "queue", []) or [])):
            payload = getattr(item, "execution_provenance", None)
            if isinstance(payload, dict) and payload.get("summary"):
                best = payload
                break
        if best is None:
            return None
        return {
            "summary": str(best.get("summary") or ""),
            "fell_back": bool(best.get("anyFallback")),
        }

    def _show_about(self):
        """Open the compact help and diagnostics surface."""
        dialog = tk.Toplevel(self.root)
        dialog.withdraw()
        dialog.title(tr("Help & diagnostics"))
        dialog.configure(bg=Theme.BG_OVERLAY)
        dialog.resizable(True, True)
        dialog.transient(self.root)
        try:
            from backend.a11y import set_accessible_metadata
            status = (self.backend_status or {}).get("summary", {})
            set_accessible_metadata(
                dialog,
                role="dialog",
                label=tr("Help & diagnostics"),
                state="modal",
                description=str(
                    status.get("next_action") or tr("System and runtime status.")),
            )
        except Exception:
            pass

        def _close_about():
            try:
                dialog.grab_release()
            except tk.TclError:
                pass
            dialog.destroy()

        scroll_body = scrollable_dialog_body(dialog, bg=Theme.BG_SECONDARY)
        body = tk.Frame(scroll_body, bg=Theme.BG_SECONDARY)
        body.pack(fill="both", expand=True)

        content = tk.Frame(body, bg=Theme.BG_SECONDARY)
        content.pack(fill="both", expand=True, padx=32, pady=(28, 24))

        brand_row = tk.Frame(content, bg=Theme.BG_SECONDARY)
        brand_row.pack(fill="x")
        if self._brand_photo:
            tk.Label(brand_row, image=self._brand_photo,
                     bg=Theme.BG_SECONDARY).pack(side="left", padx=(0, Theme.S_MD))
        title_stack = tk.Frame(brand_row, bg=Theme.BG_SECONDARY)
        title_stack.pack(side="left", fill="x", expand=True)
        tk.Label(title_stack, text=tr("Help & diagnostics"),
                 font=f(Theme.F_DISPLAY, "bold"),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY).pack(anchor="w")
        tk.Label(
            title_stack,
            text=f"{APP_NAME}  v{APP_VERSION}",
            font=f(Theme.F_BODY_SM),
            bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_MUTED,
        ).pack(anchor="w", pady=(2, 0))

        det_label = ", ".join(self.ai_engines["detection"]) or tr("None")
        inp_label = ", ".join(self.ai_engines["inpainting"]) or tr("None")
        gpu_count = len(self.gpus)
        gpu_label = (
            ntr("{count} GPU", "{count} GPUs", gpu_count).format(
                count=gpu_count)
            if self.gpus else tr("CPU only")
        )
        cache_label = tr("Unavailable")
        try:
            from backend.cache_inventory import discover_caches, _format_bytes
            total = sum(e.total_bytes for e in discover_caches())
            cache_label = _format_bytes(total)
        except Exception:
            pass

        columns = tk.Frame(content, bg=Theme.BG_SECONDARY)
        columns.pack(fill="both", expand=True, pady=(Theme.S_XL, 0))
        columns.columnconfigure(0, weight=1, uniform="help")
        columns.columnconfigure(1, weight=1, uniform="help")

        system = tk.Frame(columns, bg=Theme.BG_SECONDARY)
        system.grid(row=0, column=0, sticky="nsew", padx=(0, Theme.S_XL))
        runtime = tk.Frame(columns, bg=Theme.BG_SECONDARY)
        runtime.grid(row=0, column=1, sticky="nsew", padx=(Theme.S_XL, 0))
        tk.Frame(
            columns, bg=Theme.BORDER_SUBTLE, width=1,
        ).grid(row=0, column=0, sticky="nse", padx=(0, 0))

        def section_title(parent, text):
            tk.Label(
                parent, text=text, font=f(Theme.F_TITLE, "bold"),
                bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY,
            ).pack(anchor="w", pady=(0, Theme.S_SM))

        def fact(parent, label, value, tone=Theme.TEXT_PRIMARY):
            row = tk.Frame(parent, bg=Theme.BG_SECONDARY)
            row.pack(fill="x", pady=Theme.S_XS)
            tk.Label(
                row, text=label, font=f(Theme.F_BODY_SM),
                bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED,
            ).pack(side="left")
            full_value = str(value)
            display = truncate_middle(full_value, 34)
            value_label = tk.Label(
                row, text=display, font=f(Theme.F_BODY_SM, "bold"),
                bg=Theme.BG_SECONDARY, fg=tone,
            )
            value_label.pack(side="right", padx=(Theme.S_SM, 0))
            if display != full_value:
                Tooltip(value_label, full_value)

        section_title(system, tr("System"))
        fact(system, tr("Compute"), gpu_label)
        fact(system, tr("Detection"), det_label)
        fact(system, tr("Inpainting"), inp_label)
        fact(
            system,
            tr("FFmpeg"),
            tr("Ready") if self.ffmpeg_ready else tr("Missing"),
            Theme.SUCCESS if self.ffmpeg_ready else Theme.WARNING,
        )
        fact(system, tr("Model cache"), cache_label)
        fact(
            system,
            tr("Shortcuts"),
            tr("Ctrl+O open, Ctrl+L log, Ctrl+F filter, F1 help"),
        )

        summary = (self.backend_status or {}).get("summary", {})
        tone = str(summary.get("tone") or "neutral")
        ready = tone == "success"
        section_title(runtime, tr("Runtime status"))
        status_row = tk.Frame(runtime, bg=Theme.BG_SECONDARY)
        status_row.pack(fill="x", pady=(Theme.S_XS, Theme.S_MD))
        tk.Label(
            status_row, text="*", font=f(Theme.F_BODY, "bold"),
            bg=Theme.BG_SECONDARY,
            fg=Theme.SUCCESS if ready else Theme.WARNING,
        ).pack(side="left")
        tk.Label(
            status_row,
            text=tr("Ready") if ready else tr("Needs attention"),
            font=f(Theme.F_BODY, "bold"),
            bg=Theme.BG_SECONDARY,
            fg=Theme.SUCCESS if ready else Theme.WARNING,
        ).pack(side="left", padx=(Theme.S_SM, 0))
        fact(runtime, tr("Detection"), summary.get("detection") or tr("Unknown"))
        fact(runtime, tr("Inpainting"), summary.get("inpainting") or tr("Unknown"))
        fact(runtime, tr("Models"), summary.get("model_files") or tr("Unknown"))
        last_execution = self._last_execution_summary()
        if last_execution:
            fact(
                runtime,
                tr("Last run"),
                last_execution["summary"],
                Theme.WARNING if last_execution["fell_back"] else Theme.SUCCESS,
            )
        next_action = str(summary.get("next_action") or tr("No action needed."))
        tk.Label(
            runtime, text=next_action, font=f(Theme.F_BODY_SM),
            bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_SECONDARY,
            wraplength=310,
            justify="left",
        ).pack(anchor="w", pady=(Theme.S_MD, 0))

        actions = tk.Frame(body, bg=Theme.BG_SECONDARY)
        actions.pack(fill="x")
        tk.Frame(actions, bg=Theme.BORDER_SUBTLE, height=1).pack(fill="x")
        actions_inner = tk.Frame(actions, bg=Theme.BG_SECONDARY)
        actions_inner.pack(side="right", padx=24, pady=16)

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
        close_btn = ModernButton(actions_inner, text=tr("Close"), width=84,
                     command=_close_about,
                     style="primary", size="md")
        close_btn.pack(side="left", padx=(Theme.S_SM, 0))

        dialog.bind("<Escape>", lambda e: _close_about())
        dialog.protocol("WM_DELETE_WINDOW", _close_about)

        fit_dialog_to_work_area(
            dialog, self.root, min_width=820, min_height=560)
        dialog.deiconify()
        dialog.grab_set()
        try:
            close_btn.focus_set()
        except Exception:
            pass

