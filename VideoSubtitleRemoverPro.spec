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


def _build_profile():
    """RM-350: the dependency profile this artifact is being built from.

    Stamped into the bundle so the frozen executable, and the release
    verification that reads it back, can say which lane this payload is
    rather than inferring it from a filename.
    """
    import sys

    sys.path.insert(0, os.path.abspath(SPECPATH))
    from backend.build_profile import normalize_profile, write_build_profile
    from backend.dependency_profiles import PROFILE_ENV
    from gui.config import APP_VERSION

    name = normalize_profile(os.environ.get(PROFILE_ENV, "")) or "cpu"
    stamp_dir = os.path.join(os.path.abspath(SPECPATH), "build", "profile-stamp")
    path = write_build_profile(stamp_dir, name, app_version=APP_VERSION)
    return name, str(path)


def _package_payload(entry, package):
    """Return whether a TOC entry physically belongs to an excluded package."""
    destination, source, _kind = entry
    package = package.lower()
    destination = destination.replace('/', '\\').lower()
    source = source.replace('/', '\\').lower()
    return (
        destination == package
        or destination.startswith(package + '\\')
        or ('\\site-packages\\' + package + '\\') in source
    )


# RM-350: the frozen build reads its own dependency profile manifest to
# report which provider it activates, and backend.dependency_profiles
# resolves it relative to the bundle root.
datas = [('backend', 'backend'), ('locale', 'locale'), ('icon.png', '.'), ('icon.ico', '.'),
         ('dependency_profiles.json', '.'), ('dependency_profiles', 'dependency_profiles')]
hiddenimports = [
    'PIL._tkinter_finder', 'cv2', 'numpy', 'backend.opencv_ocr',
    'tkinter', 'tkinter.ttk', 'tkinter.filedialog', 'tkinter.messagebox',
]
for package in ('rapidocr', 'rapidocr_onnxruntime'):
    if _available(package):
        hiddenimports.append(package)
        datas += collect_data_files(package)

build_profile, build_profile_path = _build_profile()
datas.append((build_profile_path, '.'))

full_ocr = _enabled('VSR_ENABLE_FULL_OCR')
pytorch_lama = _enabled('VSR_ENABLE_PYTORCH_LAMA')
if full_ocr:
    hiddenimports += [name for name in ('paddleocr', 'easyocr') if _available(name)]
if pytorch_lama and _available('simple_lama_inpainting'):
    hiddenimports.append('simple_lama_inpainting')

# RM-319/RM-350: the CUDA lane loads its cuBLAS and cuDNN runtime out of
# the torch cu130 wheel, so onnxruntime_providers_cuda.dll cannot load
# without it. Stripping torch from an NVIDIA artifact would produce exactly
# the CPU-only bundle under a CUDA name that RM-350 exists to stop.
needs_torch = build_profile == 'nvidia'

excludes = []
if not full_ocr:
    excludes += ['paddle', 'paddleocr', 'easyocr']
if not pytorch_lama:
    excludes.append('simple_lama_inpainting')
if not full_ocr and not pytorch_lama and not needs_torch:
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
if not full_ocr and not pytorch_lama and not needs_torch:
    # Hooks for other GPU providers can discover CUDA DLLs inside torch even
    # when the torch module is excluded. Keep the default RapidOCR artifact
    # physically free of PyTorch so its SBOM and dependency audit match the
    # selected release profile.
    a.binaries = [entry for entry in a.binaries if not _package_payload(entry, 'torch')]
    a.datas = [entry for entry in a.datas if not _package_payload(entry, 'torch')]
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
