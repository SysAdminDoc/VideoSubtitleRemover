"""Responsive layout and content-scroll handlers extracted from app.py."""

from __future__ import annotations

import logging

try:
    import tkinter as tk
except ImportError:
    pass

from gui.theme import (
    Theme,
)
from backend.i18n import tr

logger = logging.getLogger(__name__)


class ResponsiveLayoutMixin:
    """Responsive layout and content-scroll handlers extracted from app.py."""

    def _on_content_configure(self, event):
        """Keep the middle workbench scrollable when settings exceed the viewport."""
        if hasattr(self, "_content_canvas"):
            self._content_canvas.configure(
                scrollregion=self._content_canvas.bbox("all"))
            self._sync_content_scrollbar()

    def _on_content_canvas_configure(self, event):
        """Lock the scrollable content frame to the canvas width."""
        if hasattr(self, "_content_window"):
            scrollbar = getattr(self, "_content_scrollbar", None)
            gutter = scrollbar.winfo_reqwidth() if scrollbar is not None else 0
            self._content_canvas.itemconfig(
                self._content_window,
                width=max(1, event.width - gutter),
            )
            self._fit_content_window_height()
            self._content_canvas.configure(
                scrollregion=self._content_canvas.bbox("all"))
            self._sync_content_scrollbar()

    def _fit_content_window_height(self) -> None:
        """Fill spare workbench height while preserving overflow scrolling."""
        canvas = getattr(self, "_content_canvas", None)
        content = getattr(self, "_content", None)
        window = getattr(self, "_content_window", None)
        if canvas is None or content is None or window is None:
            return
        try:
            target_height = max(
                canvas.winfo_height(), content.winfo_reqheight())
            canvas.itemconfig(window, height=max(1, target_height))
        except tk.TclError:
            pass

    def _sync_content_scrollbar(self):
        """Show workbench scrolling only when the content is taller."""
        canvas = getattr(self, "_content_canvas", None)
        scrollbar = getattr(self, "_content_scrollbar", None)
        if canvas is None or scrollbar is None:
            return
        bbox = canvas.bbox("all") or (0, 0, 0, 0)
        needs_scroll = bbox[3] - bbox[1] > canvas.winfo_height() + 1
        if needs_scroll and not scrollbar.winfo_manager():
            scrollbar.place(relx=1.0, rely=0.0, relheight=1.0, anchor="ne")
            scrollbar.lift()
        elif not needs_scroll and scrollbar.winfo_manager():
            scrollbar.place_forget()
            canvas.yview_moveto(0.0)

    def _on_root_configure(self, event):
        """Keep layout responsive as the window width changes."""
        if event.widget is not self.root:
            return
        self._apply_responsive_layout(event.width)

    def _apply_responsive_layout(self, width: int):
        """Reflow the workbench without changing its workflow order."""
        if not hasattr(self, "_content"):
            return
        if width < 1260 and getattr(self, "_log_visible", False):
            self._toggle_log_panel()
        compact = width < 1260 or self._text_scale_percent >= 150
        self._layout_command_strip(compact=compact)
        self._layout_preview_actions(compact=compact)
        if hasattr(self, "_header_title_label"):
            title_wrap = (
                1200 if self._text_scale_percent >= 150
                else (520 if compact else 760)
            )
            self._header_title_label.configure(wraplength=title_wrap)
            self._header_version_label.pack_forget()
            self._header_intro_label.pack_forget()
            if not compact:
                self._header_version_label.pack(
                    side="left", padx=(Theme.S_SM, 0), anchor="center")
                self._header_intro_label.pack(
                    side="left", padx=(Theme.S_MD, 0), anchor="center")
        if hasattr(self, "_log_title_cluster"):
            self._log_title_cluster.pack_forget()
            self._badge_row.pack_forget()
            self._log_open_btn.pack_forget()
            self._log_clear_btn.pack_forget()
            self._log_toggle_btn.pack_forget()
            if compact:
                self._log_toggle_btn.pack(side="left")
            else:
                self._log_title_cluster.pack(side="left")
                self._badge_row.pack(side="left", padx=(Theme.S_MD, 0))
                self._log_toggle_btn.pack(side="left", padx=(Theme.S_MD, 0))
                self._log_open_btn.pack(side="right")
                self._log_clear_btn.pack(side="right", padx=(0, Theme.S_SM))

        mode = "stacked" if compact else "wide"
        spacer = getattr(self, "_queue_inspector_spacer", None)
        if spacer is not None:
            spacer.pack_forget()
        if mode == self._layout_mode:
            self._layout_header(width=width, compact=compact)
            self._layout_queue_actions(
                compact=compact,
                dense=self._text_scale_percent >= 150,
            )
            if hasattr(self, "preview_meta_label"):
                self.preview_meta_label.config(
                    wraplength=720 if mode == "stacked" else 520)
            if hasattr(self, "preview_action_hint"):
                self.preview_action_hint.config(
                    wraplength=720 if mode == "stacked" else 520)
            if hasattr(self, "status_hint"):
                self.status_hint.config(
                    wraplength=520 if mode == "stacked" else 360)
            return

        self._layout_mode = mode
        stacked = (mode == "stacked")
        self._layout_queue_actions(
            compact=stacked,
            dense=self._text_scale_percent >= 150,
        )

        self._workflow_col.grid_forget()
        self._preview_col.grid_forget()
        self._settings_col.grid_forget()

        self._header_left.pack_forget()
        self._header_right.pack_forget()
        self._header_chips.pack_forget()

        if stacked:
            self._content.columnconfigure(0, weight=1, minsize=0, uniform="")
            self._content.columnconfigure(1, weight=0, minsize=0, uniform="")
            self._content.columnconfigure(2, weight=0, minsize=0, uniform="")
            for row in range(2):
                self._content.rowconfigure(row, weight=0)
            self._content.rowconfigure(0, weight=1)
            self._preview_col.grid(row=0, column=0, sticky="nsew",
                                   pady=(0, Theme.S_MD))
            self._settings_col.grid(row=1, column=0, sticky="nsew",
                                    pady=(0, Theme.S_MD))

            self._footer_left.pack_forget()
            self._footer_left.pack(
                anchor="w", padx=Theme.S_LG, pady=Theme.S_XS)
            self.status_hint.pack_forget()
        else:
            self._content.columnconfigure(
                0, weight=1, minsize=620, uniform="")
            self._content.columnconfigure(
                1, weight=0, minsize=380, uniform="")
            self._content.columnconfigure(2, weight=0, minsize=0, uniform="")
            self._content.rowconfigure(0, weight=1)
            self._content.rowconfigure(1, weight=0)
            self._content.rowconfigure(2, weight=0)
            preview_sticky = (
                "new"
                if getattr(self, "_inspector_open_section", None)
                else "nsew"
            )
            self._preview_col.grid(row=0, column=0, sticky=preview_sticky,
                                   padx=(0, 0))
            self._settings_col.grid(row=0, column=1, sticky="nsew")

            self._footer_left.pack_forget()
            self._footer_left.pack(
                side="left", padx=Theme.S_LG, pady=Theme.S_XS)
            self.status_hint.pack_forget()

        self.preview_meta_label.config(wraplength=720 if stacked else 520)
        if hasattr(self, "preview_action_hint"):
            self.preview_action_hint.config(wraplength=720 if stacked else 520)
        self.status_hint.config(wraplength=520 if stacked else 360)
        self._layout_header(width=width, compact=compact)
        self._render_header_chips()

    def _layout_header(self, *, width: int, compact: bool) -> None:
        """Keep header actions readable without stealing workbench height."""
        self._header_left.pack_forget()
        self._header_right.pack_forget()
        self._header_chips.pack_forget()
        compact_actions = self._text_scale_percent >= 150
        compact_title = self._text_scale_percent >= 200
        if compact_title != getattr(self, "_header_title_compact", None):
            self._header_title_label.configure(
                text=(
                    tr("VSR Pro")
                    if compact_title
                    else self._header_title_label._vsr_full_text
                )
            )
            self._header_title_compact = compact_title
        if compact_actions != getattr(
            self, "_header_actions_compact", None
        ):
            for button, icon in (
                (self._header_settings_btn, "\u2699"),
                (self._header_help_btn, "?"),
            ):
                if compact_actions:
                    button._minimum_width = 44
                    button.icon = icon
                    button.set_text("")
                else:
                    button._minimum_width = button._vsr_header_full_minimum_width
                    button.icon = None
                    button.set_text(button._vsr_header_full_text)
            self._header_actions_compact = compact_actions
        self._header_right.pack(side="right", anchor="n")
        self._header_left.pack(side="left", fill="y", expand=True)
        if compact:
            self._header_chips.pack(
                side="right", padx=(Theme.S_SM, Theme.S_LG))
        elif width >= 1260:
            self._header_chips.pack(
                side="right", padx=(Theme.S_XL, Theme.S_LG))

    def _layout_workflow_rail(self, *, compact: bool):
        """Switch the workflow rail between horizontal and vertical forms."""
        if not hasattr(self, "_workflow_step_blocks"):
            return
        for block in self._workflow_step_blocks:
            block.pack_forget()
        for connector in self._workflow_connectors:
            connector.pack_forget()

        if compact:
            for index, block in enumerate(self._workflow_step_blocks):
                block.pack(
                    side="left", fill="x", expand=True,
                    padx=(0 if index == 0 else Theme.S_SM, 0),
                )
                description = self._workflow_pills[index].get("description")
                if description is not None:
                    description.pack_forget()
            self._header_guidance_panel.pack_forget()
        else:
            for index, block in enumerate(self._workflow_step_blocks):
                block.pack(fill="x")
                description = self._workflow_pills[index].get("description")
                if description is not None and not description.winfo_manager():
                    description.pack(anchor="w", pady=(2, 0))
                if index < len(self._workflow_connectors):
                    self._workflow_connectors[index].pack(
                        anchor="w", padx=14, pady=3)
            if not self._header_guidance_panel.winfo_manager():
                self._header_guidance_panel.pack(
                    side="bottom", fill="x", padx=Theme.S_MD,
                    pady=(Theme.S_XL, Theme.S_MD))

    def _layout_preview_actions(self, *, compact: bool) -> None:
        """Keep the common preview workflow visible at every supported width."""
        row = getattr(self, "_preview_primary_actions", None)
        if row is None:
            return
        buttons = (
            self.preview_region_btn,
            self.preview_mask_btn,
            self.preview_inpaint_btn,
            self.preview_track_plan_btn,
            self.preview_zoom_btn,
            self.preview_correction_btn,
        )
        for button in buttons:
            button.pack_forget()
        self.preview_region_btn.pack(side="left")
        self.preview_mask_btn.pack(side="left", padx=(Theme.S_SM, 0))
        self.preview_inpaint_btn.pack(side="left", padx=(Theme.S_SM, 0))
        if not compact:
            self.preview_track_plan_btn.pack(
                side="left", padx=(Theme.S_SM, 0))
            self.preview_zoom_btn.pack(side="left", padx=(Theme.S_SM, 0))

    def _layout_queue_actions(self, *, compact: bool, dense: bool):
        """Keep primary queue controls visible at narrow or scaled layouts."""
        if not hasattr(self, "_queue_action_frame"):
            return
        if hasattr(self, "_queue_row"):
            if dense and not self.queue:
                self._queue_row.pack_forget()
            elif not self._queue_row.winfo_manager():
                self._queue_row.pack(
                    side="bottom", fill="x")
        for button in (
            self.start_btn, self.queue_add_btn, self.open_output_btn, self.retry_btn,
            self.repeat_btn, self.clear_btn, self._queue_more_btn,
            self.queue_remove_btn, self.queue_clear_completed_btn,
            self.queue_move_up_btn, self.queue_move_down_btn,
        ):
            button.pack_forget()
        for separator in getattr(self, "_queue_action_separators", ()):
            separator.pack_forget()

        if hasattr(self, "inspector_start_btn"):
            self.inspector_start_btn.pack_forget()
        self.queue_add_btn.pack(side="left")
        if compact or dense:
            self._queue_more_btn.pack(side="left", padx=(Theme.S_SM, 0))
        else:
            separators = getattr(self, "_queue_action_separators", ())
            actions = (
                self.queue_remove_btn,
                self.queue_clear_completed_btn,
                self.queue_move_up_btn,
                self.queue_move_down_btn,
            )
            for index, button in enumerate(actions):
                if index < len(separators) and index < 3:
                    separators[index].pack(
                        side="left", fill="y", padx=Theme.S_SM, pady=2)
                button.pack(side="left")

        queue_empty = not self.queue and not self.is_processing
        single_idle = len(self.queue) == 1 and not self.is_processing
        queue_height = 40 if queue_empty else (60 if single_idle else (64 if dense else 88))
        self.queue_canvas.configure(height=queue_height)
        self._queue_dense_mode = dense
        self._queue_subtitle_label.pack_forget()
        if hasattr(self, "_queue_count_cluster"):
            if queue_empty:
                self._queue_count_cluster.pack_forget()
            elif not self._queue_count_cluster.winfo_manager():
                self._queue_count_cluster.pack(side="right", padx=(0, Theme.S_MD))
        if dense or queue_empty or single_idle:
            self._queue_batch_frame.pack_forget()
            self._queue_batch_bar_frame.pack_forget()
            if hasattr(self, "_queue_table_header"):
                self._queue_table_header.pack_forget()
        elif (
            hasattr(self, "_queue_table_header")
            and not self._queue_table_header.winfo_manager()
        ):
            self._queue_table_header.pack(
                fill="x", padx=Theme.S_MD, pady=(0, 1),
                before=self._queue_container,
            )
        show_batch_progress = bool(
            (self.queue or self.is_processing) and not single_idle)
        if not dense and show_batch_progress:
            if not self._queue_batch_frame.winfo_manager():
                self._queue_batch_frame.pack(
                    fill="x", padx=Theme.S_MD, pady=(Theme.S_SM, 0),
                    before=self._queue_container,
                )
            if not self._queue_batch_bar_frame.winfo_manager():
                self._queue_batch_bar_frame.pack(
                    fill="x", padx=Theme.S_MD,
                    pady=(4, Theme.S_SM), before=self._queue_container,
                )
        elif not show_batch_progress:
            self._queue_batch_frame.pack_forget()
            self._queue_batch_bar_frame.pack_forget()
        dense_expanded = getattr(self, "_queue_dense_expanded", False)
        if dense and not dense_expanded:
            self._queue_container.pack_forget()
        elif not self._queue_container.winfo_manager():
            self._queue_container.pack(
                fill="both", expand=True, padx=Theme.S_MD,
                pady=(0, Theme.S_SM), before=self._queue_action_frame,
            )

    def _open_queue_actions_menu(self):
        """Expose less frequent queue commands in compact layouts."""
        menu = tk.Menu(
            self.root, tearoff=False,
            bg=Theme.BG_RAISED, fg=Theme.TEXT_PRIMARY,
            activebackground=Theme.BLUE_MUTED,
            activeforeground=Theme.TEXT_PRIMARY,
            bd=1, relief="solid",
        )
        for label, command, enabled in (
            (tr("Remove"), self._remove_selected_queue_item,
             self.queue_remove_btn.enabled),
            (tr("Clear completed"), self._clear_completed_queue_items,
             self.queue_clear_completed_btn.enabled),
            (tr("Move up"), lambda: self._move_selected_queue_item(-1),
             self.queue_move_up_btn.enabled),
            (tr("Move down"), lambda: self._move_selected_queue_item(1),
             self.queue_move_down_btn.enabled),
            (tr("Open output"), self._open_output_folder,
             self.open_output_btn.enabled),
            (tr("Retry failed"), self._retry_failed, self.retry_btn.enabled),
            (tr("Repeat last"), self._repeat_last_job, self.repeat_btn.enabled),
            (tr("Clear queue"), self._clear_queue, self.clear_btn.enabled),
        ):
            menu.add_command(
                label=label, command=command,
                state="normal" if enabled else "disabled",
            )
        if getattr(self, "_queue_dense_mode", False):
            menu.add_separator()
            menu.add_command(
                label=(
                    tr("Hide queued files")
                    if getattr(self, "_queue_dense_expanded", False)
                    else tr("Show queued files")
                ),
                command=self._toggle_dense_queue_list,
            )
        menu.add_separator()
        menu.add_command(label=tr("Activity"), command=self._toggle_log_panel)
        try:
            menu.tk_popup(
                self._queue_more_btn.winfo_rootx(),
                self._queue_more_btn.winfo_rooty()
                + self._queue_more_btn.winfo_height(),
            )
        finally:
            menu.grab_release()

    def _toggle_dense_queue_list(self):
        """Expand or collapse the queue list at very large text scales."""
        self._queue_dense_expanded = not getattr(
            self, "_queue_dense_expanded", False)
        self._layout_queue_actions(compact=True, dense=True)
