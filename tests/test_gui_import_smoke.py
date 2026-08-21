"""Import-only GUI smoke: the archived Tk tests no longer catch a broken import."""

from __future__ import annotations

import importlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_gui_modules_import_without_constructing_tk():
    names = [
        "gui.app",
        "gui.widgets",
        "gui.dialog_layout",
        "gui.layout_build",
        "gui.failure_copy",
        "gui.onboarding",
        "gui.queue_view",
        "gui.utils",
        "gui.theme",
        "gui.config",
    ]
    for path in sorted((ROOT / "gui").glob("*_controller.py")):
        names.append(f"gui.{path.stem}")
    for name in names:
        importlib.import_module(name)
