"""Consented, hash-verified download of the optional LaMa inpainting weights.

RM-354: every LaMa backend needs a weight file the installer does not ship.
The ONNX and OpenCV DNN paths look for ``VSR_LAMA_ONNX`` / ``VSR_OPENCV_LAMA``
or a file in the app model cache, and the PyTorch rung is excluded from the
frozen build entirely, so a user who picked LaMa in a release build could not
succeed under any configuration.  ``backend/adapter_manifest.py`` already
carried the source repository and a pinned SHA-256 for every filename; what
was missing was the fetch itself.

Design notes:

* This is plain HTTPS against an immutable ``/resolve/<commit>/<file>`` URL.
  ``huggingface_hub`` is an optional dependency that is *not* installed in the
  shipped profiles, so the VACE ``snapshot_download`` path cannot be reused.
  The filename is taken from the manifest rather than from the response, which
  is why the RM-336 client floor does not apply here: nothing remote chooses
  where bytes land.
* Nothing calls this implicitly.  A fetch happens because a person asked for
  one, from the GUI or from ``--fetch-model``.
* The download streams to ``<name>.part`` and is only renamed into place after
  the digest matches the manifest pin.  A cancelled, failed or mismatched
  download leaves no file behind, because a truncated ONNX model fails hours
  into a render rather than at load time.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path
import shutil
import ssl
from typing import Callable, Mapping, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import (
    HTTPRedirectHandler,
    HTTPSHandler,
    Request,
    build_opener,
)

from backend.adapter_manifest import (
    AdapterManifestEntry,
    get_manifest_entry,
    model_provenance_from_verification,
    verify_adapter_path,
    write_adapter_provenance,
)

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://huggingface.co"
ENDPOINT_ENV_VAR = "VSR_MODEL_ENDPOINT"
CONNECT_TIMEOUT_SECONDS = 30
CHUNK_BYTES = 1024 * 1024
USER_AGENT = (
    "VideoSubtitleRemover "
    "(+https://github.com/SysAdminDoc/VideoSubtitleRemover)"
)

# Ordered smallest first: a user who just wants LaMa to work should not wait
# for 208 MB when 92 MB covers the OpenCV DNN rung.  Both are Apache-2.0.
FETCHABLE_WEIGHTS: Tuple[Tuple[str, str], ...] = (
    ("opencv-lama", "inpainting_lama_2025jan.onnx"),
    ("lama-onnx", "lama_fp32.onnx"),
    ("lama-onnx", "lama.onnx"),
)

_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


@dataclass(frozen=True)
class FetchResult:
    """Outcome of one weight download."""

    adapter: str
    filename: str
    ok: bool
    reason: str
    detail: str = ""
    path: Optional[str] = None
    url: str = ""
    bytes_read: int = 0
    bytes_total: Optional[int] = None
    sha256: str = ""
    expected_sha256: str = ""

    def as_dict(self) -> dict:
        return {
            "adapter": self.adapter,
            "filename": self.filename,
            "ok": self.ok,
            "reason": self.reason,
            "detail": self.detail,
            "path": self.path,
            "url": self.url,
            "bytesRead": self.bytes_read,
            "bytesTotal": self.bytes_total,
            "sha256": self.sha256,
            "expectedSha256": self.expected_sha256,
        }


class FetchCancelled(Exception):
    """Raised internally when the caller's cancel callback returns True."""


def default_endpoint(env: Optional[Mapping[str, str]] = None) -> str:
    source = os.environ if env is None else env
    value = str(source.get(ENDPOINT_ENV_VAR, "") or "").strip()
    return value.rstrip("/") if value else DEFAULT_ENDPOINT


def model_cache_dir(env: Optional[Mapping[str, str]] = None) -> Path:
    """Where a fetched weight lands so the inpainter's search finds it.

    Mirrors ``backend/inpainters/lama.py``'s first search directory.  A file
    written anywhere else is downloaded and then not used.
    """
    source = os.environ if env is None else env
    appdata = str(source.get("APPDATA", "") or "").strip()
    if appdata:
        return Path(appdata) / "VideoSubtitleRemoverPro" / "models"
    home = str(source.get("USERPROFILE") or source.get("HOME") or "").strip()
    root = Path(home) if home else Path.home()
    return root / ".config" / "VideoSubtitleRemoverPro" / "models"


def fetchable_weights() -> Tuple[dict, ...]:
    """Describe every weight this module knows how to download."""
    described = []
    for adapter, filename in FETCHABLE_WEIGHTS:
        try:
            entry = get_manifest_entry(adapter)
        except KeyError:
            continue
        described.append({
            "adapter": adapter,
            "filename": filename,
            "repository": entry.repository,
            "revision": entry.revision,
            "license": entry.license,
            "sourceUrl": entry.source_url,
            "sha256": entry.sha256.get(filename, ""),
        })
    return tuple(described)


def resolve_url(
    entry: AdapterManifestEntry,
    filename: str,
    endpoint: str,
) -> str:
    return (
        f"{endpoint.rstrip('/')}/{entry.repository}"
        f"/resolve/{entry.revision}/{filename}"
    )


def _endpoint_problem(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    if parsed.scheme == "https":
        return ""
    host = (parsed.hostname or "").lower()
    if parsed.scheme == "http" and host in _LOCAL_HOSTS:
        return ""
    return (
        f"{ENDPOINT_ENV_VAR} must be an https endpoint "
        f"(got {endpoint!r})"
    )


class _CheckedRedirectHandler(HTTPRedirectHandler):
    """Re-apply the transport rule to every hop, not just the first.

    Hugging Face answers a resolve URL with a redirect to its CDN or Xet
    storage, so redirects have to be followed. But urlopen follows them into
    any scheme, so validating only the configured endpoint left the rule
    checkable at the front door and unenforced after it. A downgrade to plain
    http on a non-loopback host is refused here instead.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        problem = _endpoint_problem(newurl)
        if problem:
            raise URLError(f"refusing redirect to {newurl!r}: {problem}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_url(url: str, timeout: int):
    opener = build_opener(HTTPSHandler(), _CheckedRedirectHandler())
    return opener.open(Request(url, headers={"User-Agent": USER_AGENT}),
                       timeout=timeout)


def _pin_problem(entry: AdapterManifestEntry, filename: str) -> str:
    if filename not in entry.expected_filenames:
        return (
            f"{filename} is not a file the {entry.name} manifest expects"
        )
    if not entry.repository or len(entry.revision) != 40:
        return (
            f"{entry.name} has no pinned repository and immutable commit; "
            "refusing to fetch from a moving reference"
        )
    if not entry.sha256.get(filename):
        return f"no SHA-256 is pinned for {filename}"
    return ""


def fetch_weight(
    adapter: str,
    filename: str = "",
    *,
    env: Optional[Mapping[str, str]] = None,
    endpoint: str = "",
    dest_dir: Optional[Path] = None,
    progress: Optional[Callable[[int, Optional[int]], None]] = None,
    cancel: Optional[Callable[[], bool]] = None,
) -> FetchResult:
    """Download one pinned weight and verify it before it is usable.

    *progress* receives ``(bytes_read, bytes_total_or_None)`` for each chunk so
    a caller can report real bytes rather than a spinner.  *cancel* is polled
    per chunk; returning True aborts and removes the partial file.
    """
    source = os.environ if env is None else env
    try:
        entry = get_manifest_entry(adapter)
    except KeyError:
        return FetchResult(adapter, filename, False, "unknown_adapter",
                           f"no manifest entry named {adapter!r}")

    if not filename:
        filename = next(
            (name for known, name in FETCHABLE_WEIGHTS if known == adapter),
            "",
        )
    problem = _pin_problem(entry, filename)
    if problem:
        return FetchResult(adapter, filename, False, "not_pinned", problem)

    endpoint = (endpoint or default_endpoint(source)).rstrip("/")
    problem = _endpoint_problem(endpoint)
    if problem:
        return FetchResult(adapter, filename, False, "bad_endpoint", problem)

    expected = entry.sha256[filename]
    url = resolve_url(entry, filename, endpoint)
    target_dir = Path(dest_dir) if dest_dir is not None else model_cache_dir(source)
    final = target_dir / filename

    if final.is_file():
        verification = verify_adapter_path(adapter, str(final), env=source)
        if verification.allowed:
            return FetchResult(
                adapter, filename, True, "already_present",
                f"{final} already verifies against the manifest",
                path=str(final), url=url,
                bytes_read=final.stat().st_size,
                bytes_total=final.stat().st_size,
                sha256=verification.actual_sha256 or "",
                expected_sha256=expected,
            )
        logger.warning(
            "Replacing %s: it does not match the manifest (%s)",
            final, verification.hash_status,
        )

    partial = target_dir / f"{filename}.part"
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return FetchResult(adapter, filename, False, "cache_unwritable",
                           str(exc), url=url)

    digest = hashlib.sha256()
    read = 0
    total: Optional[int] = None
    try:
        with _open_url(url, CONNECT_TIMEOUT_SECONDS) as response:
            length = response.headers.get("Content-Length")
            if length and length.isdigit():
                total = int(length)
            if progress is not None:
                progress(0, total)
            with open(partial, "wb") as handle:
                while True:
                    if cancel is not None and cancel():
                        raise FetchCancelled()
                    chunk = response.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    handle.write(chunk)
                    digest.update(chunk)
                    read += len(chunk)
                    if progress is not None:
                        progress(read, total)
    except FetchCancelled:
        _discard(partial)
        return FetchResult(adapter, filename, False, "cancelled",
                           "cancelled before the download finished",
                           url=url, bytes_read=read, bytes_total=total,
                           expected_sha256=expected)
    except HTTPError as exc:
        _discard(partial)
        return FetchResult(adapter, filename, False, "http_error",
                           f"{exc.code} {exc.reason}", url=url,
                           bytes_read=read, bytes_total=total,
                           expected_sha256=expected)
    except (URLError, ssl.SSLError, TimeoutError, OSError) as exc:
        _discard(partial)
        return FetchResult(adapter, filename, False, "unreachable",
                           f"{type(exc).__name__}: {exc}", url=url,
                           bytes_read=read, bytes_total=total,
                           expected_sha256=expected)
    except BaseException:
        # Ctrl+C is a BaseException, so the clauses above never saw it, and it
        # is the cancellation route the CLI actually offers. A half-written
        # model left in the cache fails hours into a render rather than at
        # load time, so the partial goes before the interrupt continues.
        _discard(partial)
        raise

    actual = digest.hexdigest()
    if total is not None and read != total:
        _discard(partial)
        return FetchResult(adapter, filename, False, "truncated",
                           f"read {read} bytes of {total}", url=url,
                           bytes_read=read, bytes_total=total,
                           sha256=actual, expected_sha256=expected)
    if actual != expected:
        _discard(partial)
        return FetchResult(adapter, filename, False, "hash_mismatch",
                           f"expected {expected}, got {actual}", url=url,
                           bytes_read=read, bytes_total=total,
                           sha256=actual, expected_sha256=expected)

    try:
        os.replace(partial, final)
    except OSError as exc:
        _discard(partial)
        return FetchResult(adapter, filename, False, "cache_unwritable",
                           str(exc), url=url, bytes_read=read,
                           bytes_total=total, sha256=actual,
                           expected_sha256=expected)

    _record_provenance(adapter, final, entry, endpoint, source)
    logger.info("Fetched %s (%d bytes) to %s", filename, read, final)
    return FetchResult(adapter, filename, True, "downloaded",
                       f"verified against the {adapter} manifest pin",
                       path=str(final), url=url, bytes_read=read,
                       bytes_total=total, sha256=actual,
                       expected_sha256=expected)


def _discard(partial: Path) -> None:
    try:
        if partial.is_dir():
            shutil.rmtree(partial, ignore_errors=True)
        elif partial.exists():
            partial.unlink()
    except OSError as exc:
        logger.warning("Could not remove the partial download %s: %s",
                       partial, exc)


def _record_provenance(
    adapter: str,
    path: Path,
    entry: AdapterManifestEntry,
    endpoint: str,
    env: Mapping[str, str],
) -> None:
    try:
        verification = verify_adapter_path(adapter, str(path), env=env)
        payload = model_provenance_from_verification(
            verification,
            repository=entry.repository,
            revision=entry.revision,
            cache_path=path,
            source="fetch_weight",
        )
        payload["endpoint"] = endpoint
        write_adapter_provenance(adapter, payload, env)
    except (OSError, KeyError, TypeError) as exc:
        # Provenance is evidence, not a gate: the bytes already matched the
        # manifest pin above, so a failed write must not discard the weight.
        logger.warning("Could not record %s provenance: %s", adapter, exc)
