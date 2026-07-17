"""Small first-run setup progress window for the source launcher."""

from __future__ import annotations

import argparse
import tkinter as tk
from pathlib import Path
from tkinter import ttk


def parse_progress(payload: str) -> tuple[str, str, int]:
    """Parse ``STATE|message|percent`` without trusting malformed values."""
    state, separator, remainder = str(payload or "").strip().partition("|")
    message, second_separator, raw_percent = remainder.rpartition("|")
    if not separator or not second_separator:
        return "RUNNING", "Preparing the local runtime...", 2
    state = state.strip().upper()
    if state not in {"RUNNING", "DONE", "ERROR"}:
        state = "RUNNING"
    try:
        percent = max(0, min(100, int(raw_percent.strip())))
    except ValueError:
        percent = 2
    return state, message.strip() or "Preparing the local runtime...", percent


class SetupSplash:
    """Poll setup progress without importing any project dependencies."""

    def __init__(self, progress_file: Path):
        self.progress_file = progress_file
        self.root = tk.Tk()
        self.root.title("Video Subtitle Remover Pro setup")
        self.root.configure(bg="#10131a")
        self.root.resizable(False, False)
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.bind("<Escape>", self._dismiss)

        frame = tk.Frame(
            self.root,
            bg="#171b24",
            highlightbackground="#303848",
            highlightthickness=1,
            padx=28,
            pady=24,
        )
        frame.pack(fill="both", expand=True)
        tk.Label(
            frame,
            text="VIDEO SUBTITLE REMOVER PRO",
            bg="#171b24",
            fg="#f4f7fb",
            font=("Segoe UI", 15, "bold"),
        ).pack(anchor="w")
        tk.Label(
            frame,
            text="First-run setup",
            bg="#171b24",
            fg="#72a7ff",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", pady=(4, 18))
        self.message = tk.StringVar(value="Preparing the local runtime...")
        tk.Label(
            frame,
            textvariable=self.message,
            bg="#171b24",
            fg="#c9d1dc",
            font=("Segoe UI", 10),
            justify="left",
            wraplength=430,
        ).pack(anchor="w")
        self.progress = ttk.Progressbar(
            frame, length=430, maximum=100, mode="determinate")
        self.progress.pack(fill="x", pady=(18, 12))
        tk.Label(
            frame,
            text="Setup can take several minutes. Press Esc to hide this window.",
            bg="#171b24",
            fg="#7f8a9a",
            font=("Segoe UI", 8),
        ).pack(anchor="w")
        self._center(488, 188)
        self.root.after(100, self._poll)

    def _center(self, width: int, height: int) -> None:
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = max(0, (screen_width - width) // 2)
        y = max(0, (screen_height - height) // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _dismiss(self, _event=None) -> None:
        self.root.destroy()

    def _poll(self) -> None:
        try:
            payload = self.progress_file.read_text(
                encoding="utf-8", errors="replace")
        except OSError:
            payload = "RUNNING|Preparing the local runtime...|2"
        state, message, percent = parse_progress(payload)
        self.message.set(message)
        self.progress["value"] = percent
        if state in {"DONE", "ERROR"}:
            try:
                self.progress_file.unlink(missing_ok=True)
            except OSError:
                pass
            delay = 900 if state == "DONE" else 3000
            self.root.after(delay, self.root.destroy)
            return
        self.root.after(250, self._poll)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress-file", type=Path, required=True)
    args = parser.parse_args()
    SetupSplash(args.progress_file).run()


if __name__ == "__main__":
    main()
