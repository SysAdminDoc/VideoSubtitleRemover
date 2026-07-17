# -*- mode: python ; coding: utf-8 -*-
import importlib.util
import os

from PyInstaller.utils.hooks import collect_data_files, collect_all


def _enabled(name):
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _available(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


datas = [('backend', 'backend'), ('assets', 'assets'), ('locale', 'locale'), ('icon.png', '.'), ('icon.ico', '.')]
hiddenimports = [
    'PIL._tkinter_finder', 'cv2', 'numpy', 'backend.opencv_ocr',
    'tkinter', 'tkinter.ttk', 'tkinter.filedialog', 'tkinter.messagebox',
]
for package in ('rapidocr', 'rapidocr_onnxruntime'):
    if _available(package):
        hiddenimports.append(package)
        datas += collect_data_files(package)

full_ocr = _enabled('VSR_ENABLE_FULL_OCR')
pytorch_lama = _enabled('VSR_ENABLE_PYTORCH_LAMA')
if full_ocr:
    hiddenimports += [name for name in ('paddleocr', 'easyocr') if _available(name)]
if pytorch_lama and _available('simple_lama_inpainting'):
    hiddenimports.append('simple_lama_inpainting')

excludes = []
if not full_ocr:
    excludes += ['paddle', 'paddleocr', 'easyocr']
if not pytorch_lama:
    excludes.append('simple_lama_inpainting')
if not full_ocr and not pytorch_lama:
    excludes += ['torch', 'torchvision']

# numpy 2.x splits its C core into submodules (numpy._core._exceptions, ...)
# that a bare 'numpy' hiddenimport does not pull in; a partial collection makes
# the frozen exe die at launch with ModuleNotFoundError. collect_all() gathers
# the full package (data, binaries, submodules). UPX is disabled below because
# it corrupts numpy's compiled extension binaries on Windows.
np_datas, np_binaries, np_hiddenimports = collect_all('numpy')
datas += np_datas


a = Analysis(
    ['VideoSubtitleRemover.py'],
    pathex=[],
    binaries=np_binaries,
    datas=datas,
    hiddenimports=hiddenimports + np_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['assets\\runtime_hook_mp.py'],
    # Default release profile is the maintainable RapidOCR/ONNX path. The
    # batch build exposes explicit opt-ins for the multi-gigabyte
    # PaddleOCR/EasyOCR/PyTorch fallbacks.
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VideoSubtitleRemoverPro',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['icon.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='VideoSubtitleRemoverPro',
)
