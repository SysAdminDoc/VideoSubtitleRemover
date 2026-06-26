"""Trust policy for optional adapters that execute model repository code.

These adapters are default-off, but once a user opts in they can execute
Python from a model repository (Hugging Face ``trust_remote_code`` or
``torch.hub``). Keep that boundary explicit: a loader may run only when
the user points to a reviewed local checkout or names an immutable-looking
revision/tag.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Tuple


_PINNED_REV_RE = re.compile(r"^(?:[0-9a-fA-F]{7,40}|v?\d+(?:\.\d+){1,3}(?:[-._][A-Za-z0-9]+)*)$")
_UNPINNED_NAMES = {"", "main", "master", "latest", "dev", "develop", "trunk"}


@dataclass(frozen=True)
class RemoteModelPolicy:
    name: str
    repo: str
    path_env: str
    revision_env: str
    executes_code: bool = True


@dataclass(frozen=True)
class RemoteModelSource:
    policy: RemoteModelPolicy
    allowed: bool
    reason: str
    source: Optional[str] = None
    source_type: str = "missing"
    revision: Optional[str] = None
    sha256: Optional[str] = None
    configured_env_var: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "name": self.policy.name,
            "repo": self.policy.repo,
            "executesCode": self.policy.executes_code,
            "allowed": self.allowed,
            "reason": self.reason,
            "source": self.source,
            "sourceType": self.source_type,
            "revision": self.revision,
            "sha256": self.sha256,
            "configuredEnvVar": self.configured_env_var,
            "pathEnv": self.policy.path_env,
            "revisionEnv": self.policy.revision_env,
        }


REMOTE_MODEL_POLICIES = {
    "florence2": RemoteModelPolicy(
        name="florence2",
        repo="microsoft/Florence-2-base",
        path_env="VSR_FLORENCE2_PATH",
        revision_env="VSR_FLORENCE2_REVISION",
    ),
    "qwen25vl": RemoteModelPolicy(
        name="qwen25vl",
        repo="Qwen/Qwen2.5-VL-2B-Instruct",
        path_env="VSR_QWEN25VL_PATH",
        revision_env="VSR_QWEN25VL_REVISION",
        executes_code=False,
    ),
    "sam2": RemoteModelPolicy(
        name="sam2",
        repo="facebookresearch/sam2",
        path_env="VSR_SAM2_CHECKPOINT",
        revision_env="VSR_SAM2_REVISION",
        executes_code=False,
    ),
    "matanyone": RemoteModelPolicy(
        name="matanyone",
        repo="pq-yang/MatAnyone2",
        path_env="VSR_MATANYONE_PATH",
        revision_env="VSR_MATANYONE_REVISION",
        executes_code=False,
    ),
    "cotracker3": RemoteModelPolicy(
        name="cotracker3",
        repo="facebookresearch/co-tracker",
        path_env="VSR_COTRACKER_REPO",
        revision_env="VSR_COTRACKER_REF",
    ),
}


def _env(env: Optional[Mapping[str, str]]) -> Mapping[str, str]:
    return os.environ if env is None else env


def _looks_pinned_revision(value: str) -> bool:
    value = value.strip()
    if value.lower() in _UNPINNED_NAMES:
        return False
    return bool(_PINNED_REV_RE.match(value))


def _hash_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_remote_model_source(
    name: str,
    env: Optional[Mapping[str, str]] = None,
) -> RemoteModelSource:
    source_env = _env(env)
    policy = REMOTE_MODEL_POLICIES[name]

    local_path = str(source_env.get(policy.path_env, "") or "").strip()
    if local_path:
        p = Path(local_path)
        if not p.exists():
            return RemoteModelSource(
                policy=policy,
                allowed=False,
                reason=f"configured local path does not exist: {p}",
                source=local_path,
                source_type="local",
                configured_env_var=policy.path_env,
            )
        return RemoteModelSource(
            policy=policy,
            allowed=True,
            reason="approved local model path",
            source=str(p),
            source_type="local",
            sha256=_hash_file(p),
            configured_env_var=policy.path_env,
        )

    revision = str(source_env.get(policy.revision_env, "") or "").strip()
    if revision:
        if not _looks_pinned_revision(revision):
            return RemoteModelSource(
                policy=policy,
                allowed=False,
                reason=(
                    f"{policy.revision_env} must be a pinned commit SHA or "
                    "version tag, not a moving branch"
                ),
                source=policy.repo,
                source_type="remote",
                revision=revision,
                configured_env_var=policy.revision_env,
            )
        return RemoteModelSource(
            policy=policy,
            allowed=True,
            reason="approved pinned remote revision",
            source=policy.repo,
            source_type="remote",
            revision=revision,
            configured_env_var=policy.revision_env,
        )

    return RemoteModelSource(
        policy=policy,
        allowed=False,
        reason=(
            f"{policy.name} executes repository code; set {policy.path_env} "
            f"to a reviewed local checkout or {policy.revision_env} to a "
            "pinned commit/tag"
        ),
    )


def release_remote_model_status(
    env: Optional[Mapping[str, str]] = None,
) -> Tuple[dict, ...]:
    return tuple(
        resolve_remote_model_source(name, env).as_dict()
        for name in sorted(REMOTE_MODEL_POLICIES)
    )
