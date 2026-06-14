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
    if not path.is_file():
        return AdapterVerification(
            adapter=entry,
            path=str(path),
            configured=True,
            exists=False,
            allowed=False,
            hash_status="missing",
            reason=f"adapter model file does not exist: {path}",
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
    if strict_unknown:
        return AdapterVerification(
            adapter=entry,
            path=str(path),
            configured=True,
            exists=True,
            allowed=False,
            hash_status="unknown",
            reason="adapter model has no pinned SHA-256 in the manifest",
            unsafe_override=False,
            strict_unknown=True,
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
