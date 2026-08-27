"""RM-275: review the detected text tracks before anything is destroyed.

The strongest pattern in this niche is review-before-destroy: show the user
every temporal text track the detector found, let them keep or remove each
one, and only then run. The backend work lives in ``backend.track_plan``;
this mixin owns the scan thread and the review dialog.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Protocol

try:
    import tkinter as tk
    from tkinter import filedialog
except ImportError:  # pragma: no cover - headless imports have no Tk
    pass

from gui.theme import Theme, f
from gui.dialog_layout import fit_dialog_to_work_area, scrollable_dialog_body
from gui.utils import dispatch_to_ui
from gui.widgets import ModernButton, ModernToggle
from backend.a11y import set_accessible_metadata
from backend.i18n import N_, ntr, tr

logger = logging.getLogger(__name__)


class TrackPlanHost(Protocol):
    """Surface the track-plan controller needs from the app."""

    root: Any
    is_processing: bool

    def _update_status(
        self, message: str, tone: str = "neutral", toast: bool = False
    ) -> None:
        ...


class TrackPlanControllerMixin:
    """Scan-and-review workflow for pre-run track plans."""

    def _open_track_plan_review(self):
        """Scan the selected item for text tracks and open the review."""
        item = self._get_selected_queue_item(fallback_to_first=True)
        if not item:
            self._update_status(N_("Add a file to the queue first"), "warning")
            return
        if self.is_processing:
            self._update_status(
                N_("Pause the batch before planning tracks"), "warning")
            return
        if getattr(self, "_track_plan_scan_active", False):
            self._update_status(
                N_("A track scan is already running"), "warning")
            return
        self._track_plan_scan_active = True
        self._update_status(N_("Scanning for text tracks..."), "info")
        threading.Thread(
            target=self._track_plan_worker,
            args=(item.id, item.file_path),
            name="vsr-track-plan",
            daemon=True,
        ).start()

    def _track_plan_worker(self, item_id: str, file_path: str):
        try:
            from backend.track_plan import scan_track_plan

            item = next(
                (it for it in self.queue if it.id == item_id), None)
            config = item.config if item is not None else self.config

            def _progress(frame_idx, total):
                if total > 0 and frame_idx % 100 == 0:
                    dispatch_to_ui(
                        self.root, self._update_status,
                        tr("Scanning for text tracks... {pct}%").format(
                            pct=min(99, int(100 * frame_idx / total))),
                        "info")

            plan = scan_track_plan(
                file_path, config=config, on_progress=_progress)
        except Exception as exc:
            logger.warning("Track plan scan failed", exc_info=True)
            dispatch_to_ui(
                self.root, self._track_plan_scan_failed, str(exc))
            return
        dispatch_to_ui(self.root, self._show_track_plan_dialog, item_id, plan)

    def _track_plan_scan_failed(self, message: str):
        self._track_plan_scan_active = False
        self._update_status(
            tr("Track scan failed: {error}").format(error=message), "error")

    def _show_track_plan_dialog(self, item_id: str, plan: dict):
        self._track_plan_scan_active = False
        tracks = plan.get("tracks", [])
        if not tracks:
            self._update_status(
                N_("No text tracks were detected in this file"), "info")
            return

        dialog = tk.Toplevel(self.root)
        dialog.withdraw()
        dialog.title(tr("Review text tracks"))
        dialog.configure(bg=Theme.BG_OVERLAY)
        dialog.transient(self.root)
        try:
            set_accessible_metadata(
                dialog,
                role="dialog",
                label=tr("Review text tracks"),
                state="modal",
                description=tr("Choose which detected text tracks to keep or remove"),
            )
        except Exception:
            pass
        body = scrollable_dialog_body(dialog, bg=Theme.BG_SECONDARY)
        content = tk.Frame(body, bg=Theme.BG_SECONDARY)
        content.pack(fill="both", expand=True, padx=24, pady=(20, 12))

        tk.Label(
            content, text=tr("Review text tracks"),
            font=f(Theme.F_HEADING, "bold"),
            bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY,
        ).pack(anchor="w")
        tk.Label(
            content,
            text=ntr(
                "{n} track found. Select it to remove the text, or clear it "
                "to keep the text visible.",
                "{n} tracks found. Select the tracks to remove, or clear a "
                "track to keep its text visible.",
                len(tracks),
            ).format(n=len(tracks)),
            font=f(Theme.F_BODY_SM), bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_SECONDARY, wraplength=640, justify="left",
        ).pack(anchor="w", pady=(2, Theme.S_LG))

        fps = float(plan.get("fps") or 0.0) or 30.0
        remove_vars = []
        thumbnails = []
        track_rows = []
        selection_var = tk.StringVar()

        def _refresh_selection():
            selected = sum(1 for _track, var in remove_vars if var.get())
            selection_var.set(
                ntr(
                    "{n} track selected for removal",
                    "{n} tracks selected for removal",
                    selected,
                ).format(n=selected)
            )
            for row, var in track_rows:
                row.configure(
                    highlightbackground=(
                        Theme.BLUE_PRIMARY if var.get()
                        else Theme.BORDER_SUBTLE
                    )
                )

        def _set_all(value: bool):
            for _track, var in remove_vars:
                var.set(value)
            _refresh_selection()

        selection_bar = tk.Frame(content, bg=Theme.BG_SECONDARY)
        selection_bar.pack(fill="x", pady=(0, Theme.S_SM))
        tk.Label(
            selection_bar,
            textvariable=selection_var,
            font=f(Theme.F_META, "bold"),
            bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_SECONDARY,
        ).pack(side="left")
        bulk_actions = tk.Frame(selection_bar, bg=Theme.BG_SECONDARY)
        bulk_actions.pack(side="right")
        ModernButton(
            bulk_actions, text=tr("Remove all"), width=102,
            command=lambda: _set_all(True), style="ghost", size="sm",
        ).pack(side="left")
        ModernButton(
            bulk_actions, text=tr("Keep all"), width=88,
            command=lambda: _set_all(False), style="ghost", size="sm",
        ).pack(side="left", padx=(Theme.S_SM, 0))

        for track in tracks:
            row = tk.Frame(
                content,
                bg=Theme.BG_CARD,
                highlightthickness=1,
                highlightbackground=Theme.BORDER_SUBTLE,
            )
            row.pack(fill="x", pady=(0, Theme.S_SM))
            var = tk.BooleanVar(value=not track.get("keep"))
            remove_vars.append((track, var))
            track_rows.append((row, var))
            ModernToggle(
                row, text=tr("Remove from video"), variable=var,
                command=_refresh_selection,
                bg=Theme.BG_CARD,
            ).pack(side="left", padx=Theme.S_MD, pady=Theme.S_SM)
            encoded = track.get("thumbnail_png_base64") or ""
            if encoded:
                try:
                    photo = tk.PhotoImage(data=encoded)
                    thumbnails.append(photo)
                    tk.Label(row, image=photo, bg=Theme.BG_CARD).pack(
                        side="left", padx=(0, Theme.S_MD))
                except tk.TclError:
                    pass
            text_col = tk.Frame(row, bg=Theme.BG_CARD)
            text_col.pack(side="left", fill="x", expand=True,
                          pady=Theme.S_SM)
            sample = track.get("sample_text") or tr("(no readable text)")
            tk.Label(
                text_col, text=sample[:70], font=f(Theme.F_BODY_SM, "bold"),
                bg=Theme.BG_CARD, fg=Theme.TEXT_PRIMARY, anchor="w",
            ).pack(anchor="w")
            start, end = track["start_frame"], track["end_frame"]
            tk.Label(
                text_col,
                text=tr("Frames {start}-{end}, {begin}s to {finish}s").format(
                    start=start, end=end,
                    begin=round(start / fps, 1), finish=round(end / fps, 1)),
                font=f(Theme.F_META), bg=Theme.BG_CARD,
                fg=Theme.TEXT_MUTED, anchor="w",
            ).pack(anchor="w")
        _refresh_selection()
        # PhotoImages are garbage collected unless referenced; pin them to
        # the dialog for its lifetime.
        dialog._vsr_track_thumbnails = thumbnails

        def _close():
            try:
                dialog.grab_release()
            except tk.TclError:
                pass
            dialog.destroy()

        def _apply():
            for track, var in remove_vars:
                track["keep"] = not bool(var.get())
            self._apply_track_plan(item_id, plan)
            _close()

        def _save():
            for track, var in remove_vars:
                track["keep"] = not bool(var.get())
            target = filedialog.asksaveasfilename(
                parent=dialog,
                title=tr("Save track plan"),
                defaultextension=".json",
                filetypes=[(tr("Track plan"), "*.json")],
            )
            if not target:
                return
            from backend.track_plan import save_track_plan
            save_track_plan(plan, target)
            self._update_status(
                tr("Track plan saved to {name}").format(
                    name=Path(target).name), "success")

        actions = tk.Frame(content, bg=Theme.BG_SECONDARY)
        actions.pack(fill="x", pady=(Theme.S_LG, 0))
        ModernButton(
            actions, text=tr("Save plan..."), width=120,
            command=_save, style="ghost", size="md",
        ).pack(side="left")
        apply_btn = ModernButton(
            actions, text=tr("Apply selection"), width=142,
            command=_apply, style="primary", size="md",
        )
        apply_btn.pack(side="right")
        ModernButton(
            actions, text=tr("Cancel"), width=96,
            command=_close, style="ghost", size="md",
        ).pack(side="right", padx=(0, Theme.S_SM))

        dialog.bind("<Escape>", lambda _event: _close())
        dialog.bind(
            "<Return>",
            lambda _event: apply_btn.command() if apply_btn.command else None,
        )
        dialog.protocol("WM_DELETE_WINDOW", _close)
        fit_dialog_to_work_area(dialog, self.root, min_width=700,
                                min_height=440)
        dialog.deiconify()
        dialog.grab_set()
        try:
            apply_btn.focus_set()
        except Exception:
            pass

    def _apply_track_plan(self, item_id: str, plan: dict):
        """Merge the plan's kept tracks into the item's mask corrections."""
        from backend.track_plan import plan_to_mask_corrections

        exclusions = plan_to_mask_corrections(plan)
        item = next((it for it in self.queue if it.id == item_id), None)
        target = item.config if item is not None else self.config
        existing = list(
            getattr(target, "manual_mask_corrections", None) or [])
        target.manual_mask_corrections = existing + exclusions
        kept = sum(1 for track in plan.get("tracks", []) if track.get("keep"))
        if kept:
            self._update_status(
                ntr("{n} track kept: its frames are excluded from cleanup",
                    "{n} tracks kept: their frames are excluded from cleanup",
                    kept).format(n=kept),
                "success", toast=True)
        else:
            self._update_status(
                N_("All tracks marked for removal; nothing was excluded"),
                "info")
