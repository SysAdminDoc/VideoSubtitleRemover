"""Supply-chain manifest for optional model adapters.

The entries here describe opt-in model paths that are controlled by
environment variables or optional packages. Existing legacy adapters may
continue when no vendored hash is available, but new strict adapters can fail
closed on unknown weights unless the operator sets the explicit unsafe
override.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

from backend.model_hashes import hash_file

logger = logging.getLogger(__name__)

UNSAFE_OVERRIDE_ENV = "VSR_ALLOW_UNVERIFIED_MODELS"
_TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AdapterManifestEntry:
    name: str
    env_vars: Tuple[str, ...]
    expected_filenames: Tuple[str, ...]
    sha256: Mapping[str, str] = field(default_factory=dict)
    license: str = "unknown"
    source_url: str = ""
    preferred_format: str = "unknown"
    remote_code_required: bool = False
    allow_directories: bool = False


@dataclass(frozen=True)
class AdapterVerification:
    adapter: AdapterManifestEntry
    path: Optional[str]
    configured: bool
    exists: bool
    allowed: bool
    hash_status: str
    reason: str
    expected_sha256: Optional[str] = None
    actual_sha256: Optional[str] = None
    unsafe_override: bool = False
    strict_unknown: bool = False

    def as_dict(self, include_path: bool = False) -> Dict[str, object]:
        filename = Path(self.path).name if self.path else None
        payload: Dict[str, object] = {
            "name": self.adapter.name,
            "envVars": list(self.adapter.env_vars),
            "configured": self.configured,
            "filename": filename,
            "expectedFilenames": list(self.adapter.expected_filenames),
            "exists": self.exists,
            "allowed": self.allowed,
            "hashStatus": self.hash_status,
            "reason": self.reason,
            "expectedSha256": self.expected_sha256,
            "actualSha256": self.actual_sha256,
            "unsafeOverrideEnv": UNSAFE_OVERRIDE_ENV,
            "unsafeOverride": self.unsafe_override,
            "strictUnknown": self.strict_unknown,
            "preferredFormat": self.adapter.preferred_format,
            "license": self.adapter.license,
            "sourceUrl": self.adapter.source_url,
            "remoteCodeRequired": self.adapter.remote_code_required,
            "allowDirectories": self.adapter.allow_directories,
        }
        if include_path:
            payload["path"] = self.path
        return payload


ADAPTER_MANIFEST: Dict[str, AdapterManifestEntry] = {
    "lama-onnx": AdapterManifestEntry(
        name="lama-onnx",
        env_vars=("VSR_LAMA_ONNX",),
        expected_filenames=("lama_fp32.onnx", "lama.onnx"),
        sha256={
            "lama_fp32.onnx": (
                "1faef5301d78db7dda502fe59966957ec4b79dd64e16f0"
                "3ed96913c7a4eb68d6"
            ),
            "lama.onnx": (
                "351e481e287f345b7fbfd026068cfb9ec0c7f24b440e65"
                "01458ebe54a833d1a1"
            ),
        },
        license="Apache-2.0 or upstream model-card terms",
        source_url="https://huggingface.co/Carve/LaMa-ONNX",
        preferred_format="ONNX",
        remote_code_required=False,
    ),
    "opencv-lama": AdapterManifestEntry(
        name="opencv-lama",
        env_vars=("VSR_OPENCV_LAMA",),
        expected_filenames=("inpainting_lama_2025jan.onnx",),
        sha256={
            "inpainting_lama_2025jan.onnx": (
                "7df918ac3921d3daf0aae1d219776cf0dc4e4935f035af"
                "81841b40adcf74fdf2"
            ),
        },
        license="Apache-2.0",
        source_url="https://huggingface.co/opencv/inpainting_lama",
        preferred_format="ONNX",
        remote_code_required=False,
    ),
    "migan-onnx": AdapterManifestEntry(
        name="migan-onnx",
        env_vars=("VSR_MIGAN_ONNX",),
        expected_filenames=("migan_pipeline_v2.onnx", "migan.onnx"),
        sha256={},
        license="upstream model-card terms",
        source_url="https://github.com/Picsart-AI-Research/MI-GAN",
        preferred_format="ONNX",
        remote_code_required=False,
    ),
    "fastdvdnet": AdapterManifestEntry(
        name="fastdvdnet",
        env_vars=("VSR_FASTDVDNET",),
        expected_filenames=("model.pth", "net_gray.pth", "net_rgb.pth"),
        sha256={},
        license="upstream repository terms",
        source_url="https://github.com/m-tassano/fastdvdnet",
        preferred_format="trusted PyTorch state_dict",
        remote_code_required=True,
    ),
    "clear-maskfree": AdapterManifestEntry(
        name="clear-maskfree",
        env_vars=("VSR_CLEAR_WEIGHTS",),
        expected_filenames=("clear.safetensors", "clear.pt", "pytorch_model.bin"),
        sha256={},
        license="Apache-2.0 model card plus upstream Wan2.1 terms",
        source_url="https://huggingface.co/joeyz0z/CLEAR",
        preferred_format="strict local weights only",
        remote_code_required=True,
    ),
    "sedit-maskfree": AdapterManifestEntry(
        name="sedit-maskfree",
        env_vars=("VSR_SEDIT_WEIGHTS",),
        expected_filenames=("sedit.safetensors", "sedit.pt", "pytorch_model.bin"),
        sha256={},
        license="research paper; redistribution rights not verified",
        source_url="https://arxiv.org/abs/2509.18774",
        preferred_format="strict local weights only",
        remote_code_required=True,
    ),
    "void": AdapterManifestEntry(
        name="void",
        env_vars=("VSR_VOID_WEIGHTS", "VSR_VOID_PASS1", "VSR_VOID_PASS2"),
        expected_filenames=(
            "void_pass1.safetensors",
            "void_pass2.safetensors",
            "pytorch_model.bin",
        ),
        sha256={},
        license="Apache-2.0",
        source_url="https://github.com/netflix/void-model",
        preferred_format="strict local VOID checkpoints",
        remote_code_required=True,
    ),
    "vace-wan13b": AdapterManifestEntry(
        name="vace-wan13b",
        env_vars=(
            "VSR_VACE_CKPT_DIR",
            "VSR_VACE_MODEL_DIR",
            "VSR_VACE_WEIGHTS",
            "VSR_VACE_MANIFEST",
        ),
        expected_filenames=(
            "diffusion_pytorch_model.safetensors",
            "models_t5_umt5-xxl-enc-bf16.pth",
            "Wan2.1_VAE.pth",
            "model_index.json",
        ),
        sha256={},
        license="Apache-2.0",
        source_url="https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B",
        preferred_format="HuggingFace snapshot directory",
        remote_code_required=False,
        allow_directories=True,
    ),
    "videopainter": AdapterManifestEntry(
        name="videopainter",
        env_vars=(
            "VSR_VIDEOPAINTER_CKPT_DIR",
            "VSR_VIDEOPAINTER_MODEL_DIR",
            "VSR_VIDEOPAINTER_WEIGHTS",
            "VSR_VIDEOPAINTER_BRANCH_DIR",
            "VSR_COGVIDEOX_MODEL_DIR",
        ),
        expected_filenames=(
            "VideoPainter/checkpoints/branch/config.json",
            "VideoPainter/checkpoints/branch/diffusion_pytorch_model.safetensors",
            "VideoPainterID/checkpoints/pytorch_lora_weights.safetensors",
            "CogVideoX-5b-I2V/model_index.json",
        ),
        sha256={},
        license="VideoPainter research/non-commercial plus CogVideoX terms",
        source_url="https://huggingface.co/TencentARC/VideoPainter",
        preferred_format="reviewed local VideoPainter checkpoint directory",
        remote_code_required=True,
        allow_directories=True,
    ),
    "floed": AdapterManifestEntry(
        name="floed",
        env_vars=(
            "VSR_FLOED_WEIGHTS",
            "VSR_FLOED_CKPT",
            "VSR_FLOED_CKPT_DIR",
            "VSR_FLOED_COMMAND",
        ),
        expected_filenames=(
            "floed.ckpt",
            "floed.safetensors",
            "motion_module.ckpt",
            "animatediff.ckpt",
        ),
        sha256={},
        license="Apache-2.0",
        source_url="https://github.com/NevSNev/FloED-main",
        preferred_format="reviewed local FloED checkpoint",
        remote_code_required=True,
        allow_directories=True,
    ),
    "matanyone2": AdapterManifestEntry(
        name="matanyone2",
        env_vars=("VSR_MATANYONE_PATH",),
        expected_filenames=(
            "matanyone2.pth",
            "pytorch_model.bin",
            "model.safetensors",
        ),
        sha256={},
        license="NTU S-Lab License 1.0",
        source_url="https://github.com/pq-yang/MatAnyone2",
        preferred_format="reviewed local MatAnyone 2 checkpoint or snapshot",
        remote_code_required=True,
        allow_directories=True,
    ),
    "transnetv2": AdapterManifestEntry(
        name="transnetv2",
        env_vars=("VSR_TRANSNETV2",),
        expected_filenames=("transnetv2-weights", "transnetv2-weights.pth"),
        sha256={},
        license="MIT",
        source_url="https://github.com/soCzech/TransNetV2",
        preferred_format="trusted PyTorch state_dict",
        remote_code_required=False,
    ),
    "simple-lama": AdapterManifestEntry(
        name="simple-lama",
        env_vars=(),
        expected_filenames=("big-lama.pt",),
        sha256={
            "big-lama.pt": (
                "344c77bbcb158f17dd143070d1e789f38a66c04202311ae"
                "3a258ef66667a9ea9"
            ),
        },
        license="Apache-2.0",
        source_url="https://github.com/enesmsahin/simple-lama-inpainting",
        preferred_format="trusted PyTorch state_dict",
        remote_code_required=False,
    ),
}


def unsafe_override_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    source = os.environ if env is None else env
    return str(source.get(UNSAFE_OVERRIDE_ENV, "")).strip().lower() in _TRUE_VALUES


def get_manifest_entry(adapter_name: str) -> AdapterManifestEntry:
    try:
        return ADAPTER_MANIFEST[adapter_name]
    except KeyError as exc:
        raise KeyError(f"Unknown adapter manifest entry: {adapter_name}") from exc


def _expected_hash(entry: AdapterManifestEntry, path: Path) -> Optional[str]:
    direct = entry.sha256.get(path.name)
    if direct:
        return direct.lower()
    if len(entry.sha256) == 1 and path.name in entry.expected_filenames:
        return next(iter(entry.sha256.values())).lower()
    return None


def verify_adapter_path(
    adapter_name: str,
    model_path: str,
    *,
    strict_unknown: bool = False,
    env: Optional[Mapping[str, str]] = None,
) -> AdapterVerification:
    """Verify an adapter model path against the local manifest.

    ``strict_unknown`` is for new mask-free adapters: if a model file has no
    pinned SHA-256, it is rejected unless the operator explicitly enables the
    unsafe override. Current legacy adapters call this in non-strict mode so
    existing documented env-var paths keep working until pinned hashes are
    available.
    """

    entry = get_manifest_entry(adapter_name)
    override = unsafe_override_enabled(env)
    if not model_path:
        return AdapterVerification(
            adapter=entry,
            path=None,
            configured=False,
            exists=False,
            allowed=False,
            hash_status="not_configured",
            reason="adapter model path is not configured",
            unsafe_override=override,
            strict_unknown=strict_unknown,
        )
    path = Path(model_path)
    if not path.is_file() and not (entry.allow_directories and path.is_dir()):
        return AdapterVerification(
            adapter=entry,
            path=str(path),
            configured=True,
            exists=False,
            allowed=False,
            hash_status="missing",
            reason=(
                f"adapter model directory does not exist: {path}"
                if entry.allow_directories else
                f"adapter model file does not exist: {path}"
            ),
            unsafe_override=override,
            strict_unknown=strict_unknown,
        )

    expected = _expected_hash(entry, path)
    if expected:
        actual = hash_file(path).lower()
        if actual == expected:
            return AdapterVerification(
                adapter=entry,
                path=str(path),
                configured=True,
                exists=True,
                allowed=True,
                hash_status="verified",
                reason="adapter model SHA-256 matches manifest",
                expected_sha256=expected,
                actual_sha256=actual,
                unsafe_override=override,
                strict_unknown=strict_unknown,
            )
        if override:
            return AdapterVerification(
                adapter=entry,
                path=str(path),
                configured=True,
                exists=True,
                allowed=True,
                hash_status="unsafe_override",
                reason="unsafe override allowed a mismatched adapter hash",
                expected_sha256=expected,
                actual_sha256=actual,
                unsafe_override=True,
                strict_unknown=strict_unknown,
            )
        return AdapterVerification(
            adapter=entry,
            path=str(path),
            configured=True,
            exists=True,
            allowed=False,
            hash_status="mismatch",
            reason="adapter model SHA-256 does not match manifest",
            expected_sha256=expected,
            actual_sha256=actual,
            unsafe_override=False,
            strict_unknown=strict_unknown,
        )

    if override:
        return AdapterVerification(
            adapter=entry,
            path=str(path),
            configured=True,
            exists=True,
            allowed=True,
            hash_status="unsafe_override",
            reason="unsafe override allowed an adapter with no pinned hash",
            unsafe_override=True,
            strict_unknown=strict_unknown,
        )
    if strict_unknown or entry.remote_code_required:
        return AdapterVerification(
            adapter=entry,
            path=str(path),
            configured=True,
            exists=True,
            allowed=False,
            hash_status="unknown",
            reason=(
                "adapter requires remote code execution and has no "
                "pinned SHA-256 in the manifest"
                if entry.remote_code_required else
                "adapter model has no pinned SHA-256 in the manifest"
            ),
            unsafe_override=False,
            strict_unknown=strict_unknown or entry.remote_code_required,
        )
    return AdapterVerification(
        adapter=entry,
        path=str(path),
        configured=True,
        exists=True,
        allowed=True,
        hash_status="unknown",
        reason="adapter model has no pinned SHA-256 in the manifest",
        unsafe_override=False,
        strict_unknown=False,
    )


CONFORMANCE_MATRIX_SCHEMA = "vsr.adapter_conformance.v1"


def collect_adapter_conformance_matrix(
    env: Optional[Mapping[str, str]] = None,
) -> dict:
    """Build an operator-readable dry-run matrix for every adapter.

    Lists env vars, license, source, expected weight paths, hash policy,
    import-before-trust status, and availability without loading any
    untrusted model code.
    """
    source = os.environ if env is None else env
    override = unsafe_override_enabled(source)
    adapters = []
    for name in sorted(ADAPTER_MANIFEST):
        entry = ADAPTER_MANIFEST[name]
        configured_var = None
        configured_path = None
        for env_var in entry.env_vars:
            value = str(source.get(env_var, "")).strip()
            if value:
                configured_var = env_var
                configured_path = value
                break
        path_exists = False
        if configured_path:
            p = Path(configured_path)
            path_exists = p.is_file() or (entry.allow_directories and p.is_dir())
        has_pinned_hash = bool(entry.sha256)
        adapters.append({
            "name": name,
            "envVars": list(entry.env_vars),
            "configuredVar": configured_var,
            "configured": configured_path is not None,
            "pathExists": path_exists,
            "license": entry.license,
            "sourceUrl": entry.source_url,
            "preferredFormat": entry.preferred_format,
            "expectedFiles": list(entry.expected_filenames),
            "hasPinnedHash": has_pinned_hash,
            "pinnedFileCount": len(entry.sha256),
            "remoteCodeRequired": entry.remote_code_required,
            "allowDirectories": entry.allow_directories,
            "availability": (
                "ready" if configured_path and path_exists
                else "configured-missing" if configured_path
                else "not-configured"
            ),
        })
    return {
        "schema": CONFORMANCE_MATRIX_SCHEMA,
        "unsafeOverride": override,
        "adapterCount": len(adapters),
        "adapters": adapters,
        "summary": {
            "ready": sum(1 for a in adapters if a["availability"] == "ready"),
            "configuredMissing": sum(
                1 for a in adapters if a["availability"] == "configured-missing"
            ),
            "notConfigured": sum(
                1 for a in adapters if a["availability"] == "not-configured"
            ),
        },
    }


def format_adapter_conformance_matrix(matrix: dict) -> str:
    """Format the conformance matrix as a human-readable table."""
    lines = ["Adapter Conformance Matrix", "=" * 70]
    lines.append(
        f"{'Adapter':<20} {'Available':<18} {'Hash':<8} "
        f"{'License':<30} {'Remote Code':<12}"
    )
    lines.append("-" * 88)
    for a in matrix.get("adapters", []):
        lines.append(
            f"{a['name']:<20} {a['availability']:<18} "
            f"{'yes' if a['hasPinnedHash'] else 'no':<8} "
            f"{a['license'][:29]:<30} "
            f"{'yes' if a['remoteCodeRequired'] else 'no':<12}"
        )
    summary = matrix.get("summary", {})
    lines.append("-" * 88)
    lines.append(
        f"Total: {matrix.get('adapterCount', 0)}  "
        f"Ready: {summary.get('ready', 0)}  "
        f"Configured-missing: {summary.get('configuredMissing', 0)}  "
        f"Not configured: {summary.get('notConfigured', 0)}"
    )
    if matrix.get("unsafeOverride"):
        lines.append(f"WARNING: {UNSAFE_OVERRIDE_ENV} is set")
    return "\n".join(lines) + "\n"


def log_adapter_verification(result: AdapterVerification) -> None:
    if result.allowed and result.hash_status == "verified":
        logger.info("Adapter %s verified by SHA-256", result.adapter.name)
    elif result.allowed and result.hash_status == "unsafe_override":
        logger.warning("Adapter %s allowed with unsafe override: %s",
                       result.adapter.name, result.reason)
    elif result.allowed and result.hash_status == "unknown":
        logger.info("Adapter %s has no pinned hash; continuing in legacy mode",
                    result.adapter.name)
    else:
        logger.warning("Adapter %s unavailable: %s", result.adapter.name,
                       result.reason)


def release_manifest_status(
    env: Optional[Mapping[str, str]] = None,
) -> Tuple[Dict[str, object], ...]:
    source = os.environ if env is None else env
    items = []
    for name in sorted(ADAPTER_MANIFEST):
        entry = ADAPTER_MANIFEST[name]
        configured_var = None
        configured_path = None
        for env_var in entry.env_vars:
            value = str(source.get(env_var, "")).strip()
            if value:
                configured_var = env_var
                configured_path = value
                break
        if configured_path is None:
            result = AdapterVerification(
                adapter=entry,
                path=None,
                configured=False,
                exists=False,
                allowed=False,
                hash_status="not_configured",
                reason="adapter model path is not configured",
                unsafe_override=unsafe_override_enabled(source),
                strict_unknown=False,
            )
            payload = result.as_dict()
        else:
            result = verify_adapter_path(name, configured_path, env=source)
            payload = result.as_dict()
        payload["configuredEnvVar"] = configured_var
        items.append(payload)
    return tuple(items)
