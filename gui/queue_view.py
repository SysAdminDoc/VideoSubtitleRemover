"""Queue list rendering, sorting, filtering and per-file overrides extracted from app.py."""

from __future__ import annotations

import logging
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    pass

from gui.theme import (
    Theme, f,
)
from gui.config import (
    InpaintMode, ProcessingStatus, save_queue_state,
)
from gui.utils import (
    truncate_middle,
)
from gui.widgets import (
    ModernButton, make_themed_menu,
    SegmentedPicker, QueueItemWidget,
)
from backend.i18n import tr

logger = logging.getLogger(__name__)


class QueueViewMixin:
    """Queue list rendering, sorting, filtering and per-file overrides extracted from app.py."""

    def _open_sort_menu(self):
        """Pop up a themed sort menu anchored to the sort button."""
        if self.is_processing:
            self._update_status(
                "Sorting is disabled while a batch is running", "warning")
            return
        menu = make_themed_menu(self.root)
        menu.add_command(label=tr("Filename (A -> Z)"),
                         command=lambda: self._sort_queue("name_asc"))
        menu.add_command(label=tr("Filename (Z -> A)"),
                         command=lambda: self._sort_queue("name_desc"))
        menu.add_separator()
        menu.add_command(label=tr("File size (largest first)"),
                         command=lambda: self._sort_queue("size_desc"))
        menu.add_command(label=tr("File size (smallest first)"),
                         command=lambda: self._sort_queue("size_asc"))
        menu.add_separator()
        menu.add_command(label=tr("Status (pending first)"),
                         command=lambda: self._sort_queue("status"))
        menu.add_command(label=tr("Reverse current order"),
                         command=lambda: self._sort_queue("reverse"))
        try:
            bx = self._sort_btn.winfo_rootx()
            by = self._sort_btn.winfo_rooty() + self._sort_btn.winfo_height() + 2
            menu.tk_popup(bx, by)
        finally:
            menu.grab_release()
            menu.destroy()

    def _sort_queue(self, strategy: str):
        """Reorder queue items by the chosen strategy and re-render."""
        if self.is_processing:
            return
        key_map = {
            "name_asc": lambda it: Path(it.file_path).name.lower(),
            "name_desc": lambda it: Path(it.file_path).name.lower(),
            "size_asc": lambda it: self._safe_size(it.file_path),
            "size_desc": lambda it: self._safe_size(it.file_path),
            "status": lambda it: {
                ProcessingStatus.IDLE: 0,
                ProcessingStatus.LOADING: 1,
                ProcessingStatus.DETECTING: 2,
                ProcessingStatus.PROCESSING: 3,
                ProcessingStatus.MERGING: 4,
                ProcessingStatus.COMPLETE: 5,
                ProcessingStatus.PAUSED: 6,
                ProcessingStatus.CANCELLED: 7,
                ProcessingStatus.ERROR: 8,
            }.get(it.status, 99),
        }
        with self.queue_lock:
            if strategy == "reverse":
                self.queue.reverse()
            elif strategy in key_map:
                reverse = strategy.endswith("_desc")
                self.queue.sort(key=key_map[strategy], reverse=reverse)
        # Destroy all widgets so they get rebuilt in new order
        for wid, w in list(self.queue_widgets.items()):
            try:
                w.destroy()
            except Exception:
                pass
        self.queue_widgets.clear()
        self._update_queue_display()
        save_queue_state(self.queue)
        self._update_status("Queue sorted")

    def _open_per_file_overrides(self, item_id: str):
        """RM-29: themed popover that edits a single queue item's
        ProcessingConfig without touching the global UI state.

        Only the most-asked fields are surfaced (mode, language,
        sensitivity, output codec). The rest of the config carries over
        from the snapshot taken when the item was queued.
        """
        item = next((it for it in self.queue if it.id == item_id), None)
        if item is None or item.status != ProcessingStatus.IDLE:
            return

        dialog = tk.Toplevel(self.root)
        dialog.title(
            tr("Override settings: {name}").format(
                name=Path(item.file_path).name))
        dialog.configure(bg=Theme.BG_OVERLAY)
        dialog.resizable(False, False)
        dialog.transient(self.root)
        try:
            from backend.a11y import set_accessible_metadata
            set_accessible_metadata(
                dialog,
                role="dialog",
                label=tr("Per-file overrides"),
                state="modal",
                description=tr(
                    "Change cleanup settings for this queue item only. "
                    "Press Control+Enter to save or Escape to cancel."
                ),
            )
        except Exception:
            pass

        outer = tk.Frame(dialog, bg=Theme.BORDER, padx=1, pady=1)
        outer.pack()
        body = tk.Frame(outer, bg=Theme.BG_SECONDARY)
        body.pack()

        content = tk.Frame(body, bg=Theme.BG_SECONDARY)
        content.pack(padx=24, pady=(20, 12))

        tk.Label(content, text=tr("Per-file overrides"),
                 font=f(Theme.F_HEADING, "bold"),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY).pack(anchor="w")
        tk.Label(content,
                 text=tr("These apply to this queued item only and survive a global settings change."),
                 font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY,
                 wraplength=380, justify="left").pack(anchor="w", pady=(2, Theme.S_LG))

        # Mode picker.
        mode_var = tk.StringVar(value=item.config.mode.value)
        tk.Label(content, text=tr("Mode"), font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY).pack(anchor="w")
        mode_picker = SegmentedPicker(
            content,
            options=[(m.value, m.value) for m in InpaintMode],
            value=mode_var.get(),
            command=lambda v: mode_var.set(v),
            bg=Theme.BG_SECONDARY,
        )
        mode_picker.pack(fill="x", pady=(2, Theme.S_MD))

        # Detection language.
        lang_row = tk.Frame(content, bg=Theme.BG_SECONDARY)
        lang_row.pack(fill="x", pady=(0, Theme.S_SM))
        tk.Label(lang_row, text=tr("Subtitle language"), font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY).pack(side="left")
        lang_var = tk.StringVar(value=item.config.detection_lang)
        lang_combo = ttk.Combobox(
            lang_row, textvariable=lang_var, width=18,
            values=self._lang_labels,
            state="readonly", style="Dark.TCombobox", font=f(Theme.F_BODY_SM),
        )
        # Match the friendly label for the current code.
        for label, (code, _) in zip(self._lang_labels, self._lang_display):
            if code == lang_var.get():
                lang_combo.set(label)
                break
        lang_combo.pack(side="right")

        # Sensitivity slider (1-9 maps to 0.1-0.9).
        sens_row = tk.Frame(content, bg=Theme.BG_SECONDARY)
        sens_row.pack(fill="x", pady=(Theme.S_SM, Theme.S_SM))
        sens_var = tk.IntVar(value=int(round(item.config.detection_threshold * 100)))
        tk.Label(sens_row, text=tr("Sensitivity"), font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY).pack(side="left")
        sens_label = tk.Label(sens_row, text=f"{sens_var.get()}%",
                              font=f(Theme.F_BODY_SM, "bold"),
                              bg=Theme.BG_SECONDARY, fg=Theme.BLUE_PRIMARY)
        sens_label.pack(side="right")

        def _on_sens(value):
            try:
                sens_var.set(int(value))
                sens_label.config(text=f"{int(value)}%")
            except (TypeError, ValueError):
                pass

        sens_slider = tk.Scale(
            content, from_=10, to=90, orient="horizontal",
            command=_on_sens, showvalue=False, length=380,
            bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY,
            troughcolor=Theme.BG_TERTIARY,
            activebackground=Theme.BLUE_PRIMARY,
            highlightthickness=0,
        )
        sens_slider.set(sens_var.get())
        sens_slider.pack(fill="x", pady=(0, Theme.S_MD))

        # Output codec.
        codec_row = tk.Frame(content, bg=Theme.BG_SECONDARY)
        codec_row.pack(fill="x", pady=(0, Theme.S_SM))
        tk.Label(codec_row, text=tr("Output codec"), font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY).pack(side="left")
        codec_var = tk.StringVar(value=getattr(item.config, "output_codec", "h264"))
        ttk.Combobox(
            codec_row, textvariable=codec_var, width=8,
            values=["h264", "h265", "av1", "vvc"],
            state="readonly", style="Dark.TCombobox", font=f(Theme.F_BODY_SM),
        ).pack(side="right")

        # Action buttons.
        actions = tk.Frame(body, bg=Theme.BG_CARD)
        actions.pack(fill="x")
        actions_inner = tk.Frame(actions, bg=Theme.BG_CARD)
        actions_inner.pack(side="right", padx=16, pady=14)

        def _save():
            try:
                item.config.mode = InpaintMode(mode_var.get())
            except ValueError:
                pass
            label = lang_combo.get()
            new_code = self._lang_by_label.get(label, item.config.detection_lang)
            item.config.detection_lang = new_code
            item.config.detection_threshold = sens_var.get() / 100.0
            item.config.output_codec = codec_var.get()
            item.config.normalized()
            if item.id in self.queue_widgets:
                self.queue_widgets[item.id].update_item(item)
            save_queue_state(self.queue)
            self._update_status(
                f"Overrides saved for {Path(item.file_path).name}",
                "success",
            )
            dialog.destroy()

        ModernButton(actions_inner, text=tr("Cancel"), command=dialog.destroy,
                     style="ghost", size="md", width=96).pack(side="left")
        ModernButton(actions_inner, text=tr("Save"), command=_save,
                     style="primary", size="md", width=96).pack(
                         side="left", padx=(Theme.S_SM, 0))

        dialog.bind("<Escape>", lambda e: dialog.destroy())
        dialog.bind("<Control-Return>", lambda e: _save())
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.update_idletasks()
        try:
            px, py = self.root.winfo_rootx(), self.root.winfo_rooty()
            pw, ph = self.root.winfo_width(), self.root.winfo_height()
            dw, dh = dialog.winfo_reqwidth(), dialog.winfo_reqheight()
            dialog.geometry(f"+{px + (pw - dw) // 2}+{py + (ph - dh) // 3}")
        except Exception:
            pass
        mode_picker.focus_set()

    def _update_queue_display(self):
        """Update the queue display. Only rebuilds widgets that changed."""
        with self.queue_lock:
            current_ids = {item.id for item in self.queue}

        # Remove widgets for items no longer in queue
        stale_ids = [wid for wid in self.queue_widgets if wid not in current_ids]
        for wid in stale_ids:
            self.queue_widgets[wid].destroy()
            del self.queue_widgets[wid]

        # Update the inline queue summary.
        total = len(self.queue)
        self.queue_count.config(text=f"{total} item{'s' if total != 1 else ''}")
        done = sum(1 for i in self.queue if i.status == ProcessingStatus.COMPLETE)
        attention = self._queue_attention_count(self.queue)
        if done > 0:
            self.queue_done_lbl.config(text=f" / {done} done")
            self.queue_done_pill.pack(side="left", padx=(Theme.S_XS, 0))
        else:
            self.queue_done_pill.pack_forget()
        if attention > 0:
            _need = "needs" if attention == 1 else "need"
            self.queue_err_lbl.config(text=f" / {attention} {_need} attention")
            self.queue_err_pill.pack(side="left", padx=(Theme.S_XS, 0))
        else:
            self.queue_err_pill.pack_forget()
        # Sort button visibility
        try:
            if total >= 3:
                self._sort_btn.pack(side="left", padx=(Theme.S_SM, 0))
            else:
                self._sort_btn.pack_forget()
        except Exception:
            logger.debug("sort button visibility update failed", exc_info=True)

        if not self.queue:
            # Clear any remaining children and show empty state
            for widget in self.queue_frame.winfo_children():
                widget.destroy()
            self.queue_widgets.clear()
            self._hide_filter_empty_state()
            self._build_queue_empty_state()
            self._set_preview_placeholder(
                "Preview a sample frame",
                "Add files to preview them. Use Set region to mark the subtitle band before processing.",
            )
        else:
            # Remove empty label if present
            for child in self.queue_frame.winfo_children():
                if child not in self.queue_widgets.values():
                    child.destroy()

            # Add widgets for new items only
            for item in self.queue:
                if item.id not in self.queue_widgets:
                    widget = QueueItemWidget(self.queue_frame, item, self._remove_from_queue,
                                             on_select=self._show_preview,
                                             on_rename=self._rename_output_for,
                                             on_repeat=self._repeat_item_with_settings,
                                             on_cancel_item=self._cancel_queue_item,
                                             on_override=self._open_per_file_overrides,
                                             on_soft_action=self._set_soft_subtitle_action,
                                             on_retry_suggested=self._retry_review_item_with_suggested_settings)
                    widget.pack(fill="x")
                    self.queue_widgets[item.id] = widget
                    # Forward mousewheel to queue canvas
                    widget.bind("<MouseWheel>", self._on_mousewheel)
                    for child in widget.winfo_children():
                        child.bind("<MouseWheel>", self._on_mousewheel)
                        for subchild in child.winfo_children():
                            subchild.bind("<MouseWheel>", self._on_mousewheel)
                else:
                    self.queue_widgets[item.id].update_item(item)

            # Existing Tk children retain their original pack order. Repack
            # them after a queue move so the visible order follows the model.
            for item in self.queue:
                widget = self.queue_widgets.get(item.id)
                if widget is not None:
                    widget.pack_forget()
                    widget.pack(fill="x")

        if self._selected_queue_item_id and self._selected_queue_item_id in self.queue_widgets:
            self._set_selected_queue_item(self._selected_queue_item_id)
        elif self.queue:
            self._set_selected_queue_item(self.queue[0].id)
        else:
            self._set_selected_queue_item(None)
        self._refresh_action_states()
        self._layout_queue_actions(
            compact=self._layout_mode == "stacked",
            dense=self._text_scale_percent >= 150,
        )
        # Show filter only when the queue is long enough to justify it
        try:
            if len(self.queue) >= 6:
                self._queue_filter_frame.pack(
                    fill="x", padx=Theme.S_MD, pady=(0, Theme.S_SM),
                    before=self._queue_container)
            else:
                self._queue_filter_frame.pack_forget()
                if self._queue_filter_var.get():
                    self._queue_filter_var.set("")
        except Exception:
            logger.debug("filter visibility update failed", exc_info=True)
        # Re-apply any active filter so newly added items get filtered too
        if self._queue_filter_var.get():
            self._apply_queue_filter()

    def _apply_queue_filter(self):
        """Hide/show queue widgets whose filename doesn't match the filter."""
        query = (self._queue_filter_var.get() or "").strip().lower()
        visible = 0
        total = len(self.queue)
        for item in self.queue:
            widget = self.queue_widgets.get(item.id)
            if not widget:
                continue
            fname = Path(item.file_path).name.lower()
            match = (query in fname) or (query in item.file_path.lower())
            if not query or match:
                if not widget.winfo_ismapped():
                    widget.pack(fill="x")
                visible += 1
            else:
                widget.pack_forget()
        if query:
            self.queue_count.config(text=f"{visible} of {total} shown")
        else:
            self.queue_count.config(text=f"{total} item{'s' if total != 1 else ''}")

        if query and total and visible == 0:
            self._ensure_filter_empty_state()
            self._filter_empty_title.config(
                text=f'No items match "{truncate_middle(query, 28)}"')
            self._filter_empty_body.config(
                text="Try a shorter filename search, or clear the filter to see the full batch again.")
            if not self._filter_empty_container.winfo_ismapped():
                self._filter_empty_container.pack(
                    pady=(Theme.S_3XL, Theme.S_LG), fill="x")
        else:
            self._hide_filter_empty_state()
