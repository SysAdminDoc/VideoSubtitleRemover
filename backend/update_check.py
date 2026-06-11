"""Optional startup update check against GitHub Releases API.

Off by default. When enabled via settings, makes a single bounded HTTPS
GET to the GitHub API on startup. Never auto-downloads or auto-installs.
Works without the check when offline or when the API is unreachable.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Callable, Optional, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

GITHUB_LATEST_API = (
    "https://api.github.com/repos/SysAdminDoc/VideoSubtitleRemover"
    "/releases/latest"
)
TIMEOUT_SECONDS = 5
_VERSION_RE = re.compile(r"v?(\d+\.\d+\.\d+)")


def _parse_version(tag: str) -> Optional[Tuple[int, ...]]:
    m = _VERSION_RE.search(tag)
    if not m:
        return None
    return tuple(int(x) for x in m.group(1).split("."))


def check_for_update(
    current_version: str,
    callback: Callable[[str, str], None],
) -> None:
    """Check GitHub for a newer release in a daemon thread.

    *callback(latest_tag, html_url)* is called from the background
    thread only when a newer version exists. The caller is responsible
    for marshalling into the GUI thread (e.g. ``root.after``).
    """
    current = _parse_version(current_version)
    if current is None:
        return

    def _worker():
        try:
            req = Request(
                GITHUB_LATEST_API,
                headers={"Accept": "application/vnd.github+json"},
            )
            with urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
                import json
                data = json.loads(resp.read().decode("utf-8"))
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
        except (URLError, OSError, ValueError, KeyError) as exc:
            logger.debug("Update check skipped: %s", exc)
        except Exception as exc:
            logger.debug("Update check failed: %s", exc)

    t = threading.Thread(target=_worker, daemon=True, name="update-check")
    t.start()
