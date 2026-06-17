"""Optional startup update check against GitHub Releases API.

Off by default. When enabled via settings, makes a single bounded HTTPS
GET to the GitHub API on startup. Never auto-downloads or auto-installs.
Works without the check when offline or when the API is unreachable.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
import threading
import time
from typing import Callable, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

GITHUB_LATEST_API = (
    "https://api.github.com/repos/SysAdminDoc/VideoSubtitleRemover"
    "/releases/latest"
)
TIMEOUT_SECONDS = 5
_VERSION_RE = re.compile(r"v?(\d+\.\d+\.\d+)")
API_VERSION = "2022-11-28"
USER_AGENT_TEMPLATE = (
    "VideoSubtitleRemover/{version} "
    "(+https://github.com/SysAdminDoc/VideoSubtitleRemover)"
)
DEFAULT_BACKOFF_SECONDS = 3600


def _parse_version(tag: str) -> Optional[Tuple[int, ...]]:
    m = _VERSION_RE.search(tag)
    if not m:
        return None
    return tuple(int(x) for x in m.group(1).split("."))


def _state_path() -> Path:
    base = (
        Path(os.environ.get("APPDATA", Path.home() / ".config"))
        / "VideoSubtitleRemoverPro"
    )
    return base / "update_check.json"


def _load_state(path: Optional[Path] = None) -> dict:
    p = Path(path) if path is not None else _state_path()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_state(state: dict, path: Optional[Path] = None) -> None:
    p = Path(path) if path is not None else _state_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".{p.name}.tmp")
        tmp.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
        os.replace(tmp, p)
    except OSError as exc:
        logger.debug("Update check state write skipped: %s", exc)


def _header_value(headers, name: str) -> str:
    getter = getattr(headers, "get", None)
    if callable(getter):
        value = getter(name)
        if value:
            return str(value)
    return ""


def _request_headers(current_version: str, state: dict) -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": USER_AGENT_TEMPLATE.format(version=current_version),
        "X-GitHub-Api-Version": API_VERSION,
    }
    etag = str(state.get("etag", "") or "").strip()
    last_modified = str(state.get("last_modified", "") or "").strip()
    if etag:
        headers["If-None-Match"] = etag
    if last_modified:
        headers["If-Modified-Since"] = last_modified
    return headers


def _retry_after_seconds(exc: HTTPError) -> int:
    retry_after = _header_value(exc.headers, "Retry-After")
    if retry_after:
        try:
            return max(1, int(float(retry_after)))
        except ValueError:
            pass
    reset = _header_value(exc.headers, "X-RateLimit-Reset")
    if reset:
        try:
            return max(1, int(float(reset) - time.time()))
        except ValueError:
            pass
    return DEFAULT_BACKOFF_SECONDS


def check_for_update(
    current_version: str,
    callback: Callable[[str, str], None],
    *,
    state_path: Optional[Path] = None,
) -> Optional[threading.Thread]:
    """Check GitHub for a newer release in a daemon thread.

    *callback(latest_tag, html_url)* is called from the background
    thread only when a newer version exists. The caller is responsible
    for marshalling into the GUI thread (e.g. ``root.after``).

    Returns the daemon :class:`threading.Thread` performing the check so
    callers (and tests) can ``join`` it for deterministic completion, or
    ``None`` when *current_version* is unparseable and no check is run.
    """
    current = _parse_version(current_version)
    if current is None:
        return None

    def _worker():
        try:
            state = _load_state(state_path)
            now = time.time()
            backoff_until = float(state.get("backoff_until", 0) or 0)
            if backoff_until > now:
                logger.debug(
                    "Update check skipped until %.0f after rate limit",
                    backoff_until,
                )
                return
            req = Request(
                GITHUB_LATEST_API,
                headers=_request_headers(current_version, state),
            )
            with urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                resp_headers = getattr(resp, "headers", {})
                state["etag"] = _header_value(resp_headers, "ETag")
                state["last_modified"] = _header_value(
                    resp_headers, "Last-Modified"
                )
                state["backoff_until"] = 0
                _save_state(state, state_path)
            tag = data.get("tag_name", "")
            html_url = data.get("html_url", "")
            latest = _parse_version(tag)
            if latest and latest > current:
                logger.info(
                    "Update available: %s (current %s)", tag, current_version
                )
                callback(tag, html_url)
            else:
                logger.debug(
                    "No update: latest=%s current=%s", tag, current_version
                )
        except HTTPError as exc:
            if exc.code == 304:
                logger.debug("No update: GitHub returned 304 Not Modified")
                state = _load_state(state_path)
                state["backoff_until"] = 0
                _save_state(state, state_path)
                return
            if exc.code in (403, 429):
                retry_seconds = _retry_after_seconds(exc)
                state = _load_state(state_path)
                state["backoff_until"] = time.time() + retry_seconds
                _save_state(state, state_path)
                logger.debug(
                    "Update check rate-limited for %s seconds", retry_seconds
                )
                return
            logger.debug("Update check skipped: HTTP %s", exc.code)
        except (URLError, OSError, ValueError, KeyError) as exc:
            logger.debug("Update check skipped: %s", exc)
        except Exception as exc:
            logger.debug("Update check failed: %s", exc)

    t = threading.Thread(target=_worker, daemon=True, name="update-check")
    t.start()
    return t
