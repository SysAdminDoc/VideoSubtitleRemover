from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_importing_processor_does_not_import_cli_module():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import backend.processor; "
                "raise SystemExit(1 if 'backend.cli' in sys.modules else 0)"
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_legacy_processor_helpers_reexport_focused_owners():
    from backend import config, processor, resume_checkpoint

    assert processor._load_json_config is config._load_json_config
    assert (
        processor._apply_auto_band_override
        is config._apply_auto_band_override
    )
    assert processor._checkpoint_key is resume_checkpoint._checkpoint_key
    assert (
        processor._default_checkpoint_dir
        is resume_checkpoint._default_checkpoint_dir
    )


def test_gui_uses_focused_config_detection_and_checkpoint_modules():
    gui_root = ROOT / "gui"
    forbidden = (
        "from backend.processor import InpaintMode",
        "from backend.processor import ProcessingConfig",
        "from backend.processor import SubtitleDetector",
        "from backend.processor import normalize_processing_config",
        "from backend.processor import _checkpoint_key",
        "from backend.processor import _default_checkpoint_dir",
    )
    for path in gui_root.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for statement in forbidden:
            assert statement not in source, f"{path.name}: {statement}"
