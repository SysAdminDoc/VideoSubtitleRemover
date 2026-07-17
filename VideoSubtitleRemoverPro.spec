# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_all

datas = [('backend', 'backend'), ('assets', 'assets'), ('locale', 'locale'), ('icon.png', '.'), ('icon.ico', '.')]
datas += collect_data_files('rapidocr')

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
    hiddenimports=['PIL._tkinter_finder', 'cv2', 'numpy', 'tkinter', 'tkinter.ttk', 'tkinter.filedialog', 'tkinter.messagebox', 'rapidocr'] + np_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['assets\\runtime_hook_mp.py'],
    # Default release profile is the maintainable RapidOCR/ONNX path. The
    # batch build exposes explicit opt-ins for the multi-gigabyte
    # PaddleOCR/EasyOCR/PyTorch fallbacks.
    excludes=['paddle', 'paddleocr', 'easyocr', 'torch', 'torchvision', 'simple_lama_inpainting'],
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
