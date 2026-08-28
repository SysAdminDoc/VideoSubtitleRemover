"""First-run onboarding modal extracted from app.py."""

from __future__ import annotations

import logging

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    pass

from gui.theme import (
    Theme, f,
)
from gui.config import (
    APP_NAME, APP_VERSION, save_settings,
)
from gui.widgets import (
    ModernButton,
)
from backend.i18n import tr
from gui.dialog_layout import (
    fit_dialog_to_work_area,
    scrollable_dialog_body,
)

logger = logging.getLogger(__name__)


class OnboardingMixin:
    """First-run onboarding modal extracted from app.py."""

    def _show_onboarding(self):
        dialog = tk.Toplevel(self.root)
        dialog.withdraw()
        dialog.title(tr("Welcome to {app}").format(app=APP_NAME))
        dialog.configure(bg=Theme.BG_OVERLAY)
        dialog.resizable(True, True)
        dialog.transient(self.root)
        try:
            from backend.a11y import set_accessible_metadata
            set_accessible_metadata(
                dialog,
                role="dialog",
                label=tr("Welcome to {app_name}").format(app_name=APP_NAME),
                state="modal",
                description=tr("Choose a cleanup profile and start locally."),
            )
        except Exception:
            pass

        scroll_body = scrollable_dialog_body(dialog, bg=Theme.BG_OVERLAY)
        body = tk.Frame(scroll_body, bg=Theme.BG_SECONDARY)
        body.pack(fill="both", expand=True)

        content = tk.Frame(body, bg=Theme.BG_SECONDARY)
        content.pack(fill="both", expand=True, padx=32, pady=(28, 24))
        content.columnconfigure(0, weight=3, minsize=400)
        content.columnconfigure(1, weight=2, minsize=280)

        left = tk.Frame(content, bg=Theme.BG_SECONDARY)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 36))
        brand = tk.Frame(left, bg=Theme.BG_SECONDARY)
        brand.pack(anchor="w")
        tk.Label(
            brand, text=APP_NAME, font=f(Theme.F_TITLE, "bold"),
            bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY,
        ).pack(side="left")
        tk.Label(
            brand, text=f"v{APP_VERSION}", font=f(Theme.F_META),
            bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED,
        ).pack(side="left", padx=(Theme.S_SM, 0), pady=(3, 0))

        tk.Label(
            left, text=tr("Remove subtitles. Keep the frame."),
            font=f(Theme.F_DISPLAY, "bold"), bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_PRIMARY,
        ).pack(anchor="w", pady=(Theme.S_LG, Theme.S_XS))
        tk.Label(
            left, text=tr("A short workflow for clean, local processing."),
            font=f(Theme.F_BODY), bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_SECONDARY,
        ).pack(anchor="w", pady=(0, Theme.S_XL))

        def step(number: str, heading: str, body_text: str):
            row = tk.Frame(left, bg=Theme.BG_SECONDARY)
            row.pack(fill="x", pady=(0, Theme.S_LG))
            tk.Label(
                row, text=number, font=f(Theme.F_BODY, "bold"),
                bg=Theme.BG_SECONDARY, fg=Theme.BLUE_HOVER, width=2,
                anchor="w",
            ).pack(side="left", anchor="n")
            copy = tk.Frame(row, bg=Theme.BG_SECONDARY)
            copy.pack(side="left", fill="x", expand=True)
            tk.Label(
                copy, text=tr(heading), font=f(Theme.F_BODY, "bold"),
                bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY,
            ).pack(anchor="w")
            tk.Label(
                copy, text=tr(body_text), font=f(Theme.F_BODY_SM),
                bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED,
                wraplength=350, justify="left",
            ).pack(anchor="w", pady=(2, 0))

        step("1", "Add media", "Choose videos, images, or a whole folder.")
        step("2", "Set the region", "Use automatic detection or draw the subtitle area.")
        step("3", "Start cleanup", "Review the queue, then process the batch.")

        starter = tk.Frame(content, bg=Theme.BG_SECONDARY)
        starter.grid(row=0, column=1, sticky="nsew")
        tk.Label(
            starter,
            text=tr("Cleanup profile"),
            font=f(Theme.F_TITLE, "bold"),
            bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_PRIMARY,
        ).pack(anchor="w")
        tk.Label(
            starter,
            text=tr("Choose a starting point. Every setting remains editable."),
            font=f(Theme.F_BODY_SM),
            bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_MUTED,
            wraplength=280, justify="left",
        ).pack(anchor="w", pady=(Theme.S_XS, Theme.S_MD))
        onboarding_choice_var = tk.StringVar(value=tr("Balanced"))
        profile_combo = ttk.Combobox(
            starter,
            textvariable=onboarding_choice_var,
            values=(tr("Balanced"), tr("Film"), tr("Fast")),
            state="readonly",
            style="Dark.TCombobox",
            font=f(Theme.F_BODY),
            width=24,
        )
        profile_combo.pack(fill="x")

        def _choose_preset(_event=None):
            profiles = {
                tr("Balanced"): "YouTube (default)",
                tr("Film"): "Film / Live action",
                tr("Fast"): "Fast",
            }
            self._apply_onboarding_preset(
                profiles.get(onboarding_choice_var.get(), "YouTube (default)"))

        profile_combo.bind("<<ComboboxSelected>>", _choose_preset)

        tk.Frame(
            starter, bg=Theme.BORDER_SUBTLE, height=1,
        ).pack(fill="x", pady=Theme.S_XL)
        tk.Label(
            starter,
            text=tr("Local processing"),
            font=f(Theme.F_BODY, "bold"),
            bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_PRIMARY,
        ).pack(anchor="w")
        tk.Label(
            starter,
            text=tr("Media stays on this computer. The app checks required models before the first run."),
            font=f(Theme.F_BODY_SM),
            bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_MUTED,
            wraplength=280,
            justify="left",
        ).pack(anchor="w", pady=(Theme.S_XS, 0))

        actions = tk.Frame(body, bg=Theme.BG_SECONDARY)
        actions.pack(fill="x")
        tk.Frame(actions, bg=Theme.BORDER_SUBTLE, height=1).pack(fill="x")
        actions_inner = tk.Frame(actions, bg=Theme.BG_SECONDARY)
        actions_inner.pack(side="right", padx=24, pady=16)

        def _close():
            self.config.onboarding_seen = True
            save_settings(self.config)
            dialog.grab_release()
            dialog.destroy()

        def _continue():
            _choose_preset()
            _close()

        ModernButton(
            actions_inner, text=tr("Skip"), width=88,
            command=_close, style="ghost", size="md",
        ).pack(side="left")
        continue_btn = ModernButton(
            actions_inner, text=tr("Continue"), width=118,
            command=_continue, style="primary", size="md",
        )
        continue_btn.pack(side="left", padx=(Theme.S_SM, 0))

        dialog.bind("<Escape>", lambda e: _close())
        dialog.bind("<Return>", lambda e: _continue())
        dialog.protocol("WM_DELETE_WINDOW", _close)

        try:
            fit_dialog_to_work_area(
                dialog, self.root, min_width=760, min_height=500)
        except Exception:
            logger.warning("Onboarding dialog fit failed", exc_info=True)
        dialog.deiconify()
        dialog.grab_set()
        try:
            continue_btn.focus_set()
        except Exception:
            pass
        # The dialog is now on screen; mark it seen in memory. The close path
        # persists the flag so a background-scheduled dialog cannot write
        # unrelated in-progress settings before the user dismisses it.
        self.config.onboarding_seen = True

    def replay_onboarding(self):
        """RM-341: show the welcome flow again on request.

        `onboarding_seen` was set as the dialog was built and nothing ever
        cleared it, so a user who dismissed the walkthrough could never see
        it again. Help offers this.
        """
        self._onboarding_scheduled = False
        self._show_onboarding()
