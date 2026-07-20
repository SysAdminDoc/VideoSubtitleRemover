"""First-run onboarding modal extracted from app.py."""

from __future__ import annotations

import logging

try:
    import tkinter as tk
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

logger = logging.getLogger(__name__)


class OnboardingMixin:
    """First-run onboarding modal extracted from app.py."""

    def _show_onboarding(self):
        dialog = tk.Toplevel(self.root)
        dialog.withdraw()
        dialog.title(f"Welcome to {APP_NAME}")
        dialog.configure(bg=Theme.BG_OVERLAY)
        dialog.resizable(False, False)
        dialog.transient(self.root)
        try:
            from backend.a11y import set_accessible_metadata
            set_accessible_metadata(
                dialog,
                role="dialog",
                label=tr("Welcome to {app_name}").format(app_name=APP_NAME),
                state="modal",
                description=(
                    tr("Three first-run cues: import media, inspect the "
                       "region, and run the batch.")
                ),
            )
        except Exception:
            pass

        outer = tk.Frame(dialog, bg=Theme.BORDER, padx=1, pady=1)
        outer.pack()
        body = tk.Frame(outer, bg=Theme.BG_SECONDARY)
        body.pack()

        content = tk.Frame(body, bg=Theme.BG_SECONDARY)
        content.pack(padx=36, pady=(28, 16))

        # Headline
        hero = tk.Frame(content, bg=Theme.BG_SECONDARY)
        hero.pack(anchor="w")
        tk.Label(hero, text=tr("Welcome"), font=f(Theme.F_DISPLAY, "bold"),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY).pack(
                     side="left")
        tk.Label(hero, text=f"v{APP_VERSION}", font=f(Theme.F_BODY_SM),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED).pack(
                     side="left", padx=(Theme.S_SM, 0), pady=(14, 0))

        tk.Label(content,
                 text=tr("Three things that make batch cleanup painless."),
                 font=f(Theme.F_BODY),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY).pack(
                     anchor="w", pady=(4, Theme.S_LG))

        # Cue cards
        cards = tk.Frame(content, bg=Theme.BG_SECONDARY)
        cards.pack(anchor="w")

        def card(num: str, heading: str, body_text: str, tone: str):
            c = tk.Frame(cards, bg=Theme.BG_CARD, highlightthickness=1,
                         highlightbackground=Theme.BORDER)
            inner = tk.Frame(c, bg=Theme.BG_CARD)
            inner.pack(fill="both", expand=True, padx=16, pady=14)
            top = tk.Frame(inner, bg=Theme.BG_CARD)
            top.pack(anchor="w")
            # Numbered step badge
            badge_bg = {"info": Theme.INFO_BG, "success": Theme.SUCCESS_BG,
                        "warning": Theme.WARNING_BG}.get(tone, Theme.BG_TERTIARY)
            badge_fg = {"info": Theme.INFO, "success": Theme.SUCCESS,
                        "warning": Theme.WARNING}.get(tone, Theme.TEXT_SECONDARY)
            tk.Label(top, text=num, font=f(Theme.F_BODY_SM, "bold"),
                     bg=badge_bg, fg=badge_fg, padx=8, pady=2).pack(side="left")
            tk.Label(top, text=tr(heading), font=f(Theme.F_BODY, "bold"),
                     bg=Theme.BG_CARD, fg=Theme.TEXT_PRIMARY).pack(
                         side="left", padx=(Theme.S_SM, 0))
            tk.Label(inner, text=tr(body_text), font=f(Theme.F_BODY_SM),
                     bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY,
                     wraplength=220, justify="left", anchor="w").pack(
                         anchor="w", pady=(Theme.S_SM, 0))
            return c

        card("1", "Import media",
             "Drop videos or images on the left, or pick an entire folder. "
             "Originals are never modified.",
             "info").pack(side="left", fill="both", expand=True,
                          padx=(0, Theme.S_SM))
        card("2", "Inspect the region",
             "Click a queue item, then use Set region to draw the subtitle band "
             "or Review mask to see what the detector finds.",
             "warning").pack(side="left", fill="both", expand=True,
                             padx=(0, Theme.S_SM))
        card("3", "Run the batch",
             "Hit Start batch when the framing looks right. Progress, ETA, "
             "and completion summary are all live.",
             "success").pack(side="left", fill="both", expand=True)

        # First-run profile chooser. These use the normal preset application
        # path so every dependent toggle and slider refreshes immediately.
        starter = tk.Frame(content, bg=Theme.BG_SECONDARY)
        starter.pack(fill="x", pady=(Theme.S_LG, 0))
        tk.Label(
            starter,
            text=tr("Choose a starting profile"),
            font=f(Theme.F_BODY, "bold"),
            bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_PRIMARY,
        ).pack(anchor="w")
        tk.Label(
            starter,
            text=tr("You can change every setting later."),
            font=f(Theme.F_META),
            bg=Theme.BG_SECONDARY,
            fg=Theme.TEXT_MUTED,
        ).pack(anchor="w", pady=(2, Theme.S_SM))
        starter_buttons = tk.Frame(starter, bg=Theme.BG_SECONDARY)
        starter_buttons.pack(anchor="w")
        onboarding_choice_var = tk.StringVar(value="")

        def _choose_preset(name: str):
            self._apply_onboarding_preset(name)
            onboarding_choice_var.set(
                tr("Selected: {profile}").format(profile=name)
            )

        for index, (label, preset_name) in enumerate((
            ("YouTube", "YouTube (default)"),
            ("Film", "Film / Live action"),
            ("Fast", "Fast"),
        )):
            ModernButton(
                starter_buttons,
                text=tr(label),
                width=104,
                command=lambda name=preset_name: _choose_preset(name),
                style="ghost",
                size="sm",
            ).pack(side="left", padx=(0 if index == 0 else Theme.S_SM, 0))
        tk.Label(
            starter_buttons,
            textvariable=onboarding_choice_var,
            font=f(Theme.F_META),
            bg=Theme.BG_SECONDARY,
            fg=Theme.SUCCESS,
        ).pack(side="left", padx=(Theme.S_MD, 0))

        # Action row
        actions = tk.Frame(body, bg=Theme.BG_CARD)
        actions.pack(fill="x")
        quick_actions = tk.Frame(actions, bg=Theme.BG_CARD)
        quick_actions.pack(side="left", padx=16, pady=14)
        actions_inner = tk.Frame(actions, bg=Theme.BG_CARD)
        actions_inner.pack(side="right", padx=16, pady=14)

        def _close():
            self.config.onboarding_seen = True
            save_settings(self.config)
            dialog.grab_release()
            dialog.destroy()

        def _try_cleanup():
            _close()
            self._schedule_onboarding_test_cleanup()

        ModernButton(
            quick_actions,
            text=tr("Enable auto-detect"),
            width=156,
            command=self._enable_onboarding_auto_band,
            style="ghost",
            size="sm",
        ).pack(side="left")
        ModernButton(
            quick_actions,
            text=tr("Try test cleanup"),
            width=142,
            command=_try_cleanup,
            style="ghost",
            size="sm",
        ).pack(side="left", padx=(Theme.S_SM, 0))
        ModernButton(actions_inner, text=tr("Got it"), width=118,
                     command=_close, style="primary", size="md").pack(
                         side="left")

        dialog.bind("<Escape>", lambda e: _close())
        dialog.bind("<Return>", lambda e: _close())
        dialog.protocol("WM_DELETE_WINDOW", _close)

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
        # The dialog is now on screen; mark it seen in memory. The close path
        # persists the flag so a background-scheduled dialog cannot write
        # unrelated in-progress settings before the user dismisses it.
        self.config.onboarding_seen = True
