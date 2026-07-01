"""NLE round-trip sidecar export.

RM-76: when a user processes a file inside a DaVinci Resolve / Premiere
edit, they typically want the cleaned video to slot back in at the same
timecode as the original. We don't pretend to re-author the NLE
project; we emit a minimal sidecar that names the source, the cleaned
output, and the time range that was actually processed. The NLE
operator drops the sidecar into their bin or hand-substitutes the clip
using the timecode metadata.

Two formats are supported:
- A CMX 3600 EDL with one event spanning the processed range.
- An FCPXML 1.10 minimal stub naming the cleaned file as an asset.

Both are UTF-8 encoded and small (<2 KB for a 1-event sidecar). Neither
attempts to round-trip transitions or audio tracks; this is the
"hand it to the editor" interchange, not a full project re-author.
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import quote as _uri_quote
from xml.sax.saxutils import quoteattr

logger = logging.getLogger(__name__)


def _write_atomic(path: str, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, str(target))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _edl_comment_text(value: str) -> str:
    """Keep EDL comment metadata on one line."""
    return " ".join(str(value).replace("\r", "\n").splitlines()).strip()


def _xml_attr(value: str) -> str:
    return quoteattr(str(value))


def _ts_to_smpte(seconds: float, fps: float) -> str:
    """Return SMPTE HH:MM:SS:FF (drop-frame ignored; integer-fps only
    so 23.976 falls into 24 fps for the sidecar). The CMX 3600 spec
    accepts both ; and : separators; we use : to keep ASCII-only."""
    if fps <= 0:
        fps = 24.0
    total_frames = max(0, int(round(seconds * fps)))
    rate = max(1, int(round(fps)))
    hh = total_frames // (3600 * rate)
    rem = total_frames - hh * 3600 * rate
    mm = rem // (60 * rate)
    rem -= mm * 60 * rate
    ss = rem // rate
    ff = rem - ss * rate
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def write_edl(path: str, source: str, cleaned: str,
              fps: float, start_s: float, end_s: float,
              title: str = "VSR cleanup",
              segments: Optional[List[Tuple[float, float]]] = None,
              width: int = 0, height: int = 0) -> str:
    """Write a CMX 3600 EDL with one or more events. When `segments` is
    provided, each (start_s, end_s) pair becomes a numbered event;
    otherwise the single start_s/end_s pair produces a 1-event EDL.
    Returns the path written."""
    all_segments = segments if segments else [(start_s, end_s)]
    payload = []
    payload.append(f"TITLE: {title}")
    payload.append(f"FCM: NON-DROP FRAME")
    if width > 0 and height > 0:
        payload.append(f"* SOURCE DIMENSIONS: {width}x{height}")
    payload.append("")
    for idx, (seg_start, seg_end) in enumerate(all_segments, 1):
        src_in = _ts_to_smpte(seg_start, fps)
        src_out = _ts_to_smpte(seg_end, fps)
        payload.append(
            f"{idx:03d}  AX       V     C        "
            f"{src_in} {src_out} {src_in} {src_out}"
        )
        payload.append(f"* FROM CLIP NAME: {_edl_comment_text(Path(source).name)}")
        payload.append(f"* TO CLIP NAME:   {_edl_comment_text(Path(cleaned).name)}")
        payload.append(f"* CLEANED BY:     Video Subtitle Remover Pro")
        payload.append("")
    text = "\n".join(payload) + "\n"
    _write_atomic(path, text)
    return path


def write_fcpxml(path: str, source: str, cleaned: str,
                  fps: float, start_s: float, end_s: float,
                  segments: Optional[List[Tuple[float, float]]] = None,
                  width: int = 0, height: int = 0) -> str:
    """Write a minimal FCPXML 1.10 stub with one or more asset-clips
    referencing the cleaned file. When `segments` is provided, each
    (start_s, end_s) pair becomes an asset-clip in the spine.

    We use ascii-safe XML (no smart quotes, no em-dashes) so the file
    survives PyInstaller's text bundling and arbitrary editor parsers.
    """
    all_segments = segments if segments else [(start_s, end_s)]
    src_name = Path(source).stem
    resolved = str(Path(cleaned).resolve()).replace("\\", "/")
    cleaned_uri = "file:///" + _uri_quote(resolved.lstrip("/"), safe="/:")
    rate = max(1, int(round(fps if fps > 0 else 24.0)))
    fmt_width = width if width > 0 else 1920
    fmt_height = height if height > 0 else 1080
    total_start = min(s for s, _ in all_segments) if all_segments else start_s
    total_end = max(e for _, e in all_segments) if all_segments else end_s
    total_duration_frames = max(1, int(round((total_end - total_start) * rate)))
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    event_name = f"VSR Cleanup {now} UTC"
    project_name = f"{src_name} -- cleaned"
    clips = []
    for seg_start, seg_end in all_segments:
        dur_frames = max(1, int(round((seg_end - seg_start) * rate)))
        off_frames = max(0, int(round(seg_start * rate)))
        clips.append(
            f"            <asset-clip name={_xml_attr(src_name)} "
            f"ref=\"r1\" offset=\"{off_frames}/{rate}s\" "
            f"start=\"0s\" duration=\"{dur_frames}/{rate}s\"/>"
        )
    payload = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<!DOCTYPE fcpxml>\n"
        "<fcpxml version=\"1.10\">\n"
        "  <resources>\n"
        f"    <format id=\"r0\" frameDuration=\"1/{rate}s\" "
        f"width=\"{fmt_width}\" height=\"{fmt_height}\"/>\n"
        f"    <asset id=\"r1\" name={_xml_attr(src_name)} "
        f"start=\"0s\" duration=\"{total_duration_frames}/{rate}s\" "
        f"hasVideo=\"1\" hasAudio=\"1\" format=\"r0\" "
        f"src={_xml_attr(cleaned_uri)}/>\n"
        "  </resources>\n"
        "  <library>\n"
        f"    <event name={_xml_attr(event_name)}>\n"
        f"      <project name={_xml_attr(project_name)}>\n"
        f"        <sequence format=\"r0\">\n"
        f"          <spine>\n"
        + "\n".join(clips) + "\n"
        "          </spine>\n"
        "        </sequence>\n"
        "      </project>\n"
        "    </event>\n"
        "  </library>\n"
        "</fcpxml>\n"
    )
    _write_atomic(path, payload)
    return path


_SMPTE_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[:;](\d{1,3})"
)

_EDL_EVENT_RE = re.compile(
    r"^\s*\d{1,3}\s+\S+\s+[VA]\w*\s+\S+\s+"
    r"(\d{1,2}:\d{2}:\d{2}[:;]\d{1,3})\s+"
    r"(\d{1,2}:\d{2}:\d{2}[:;]\d{1,3})\s+"
    r"(\d{1,2}:\d{2}:\d{2}[:;]\d{1,3})\s+"
    r"(\d{1,2}:\d{2}:\d{2}[:;]\d{1,3})"
)


def _smpte_to_seconds(smpte: str, fps: float = 24.0) -> float:
    m = _SMPTE_RE.match(smpte.strip())
    if not m:
        return 0.0
    hh, mm, ss, ff = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    rate = max(1, int(round(fps)))
    return hh * 3600.0 + mm * 60.0 + ss + ff / rate


def parse_edl(
    path: str, fps: float = 24.0,
) -> List[Tuple[float, float]]:
    """Parse a CMX 3600 EDL and return (start_s, end_s) tuples for each
    event's source in/out range. Ignores non-video events and comment
    lines. Returns an empty list when the file cannot be read."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning(f"Cannot read EDL: {exc}")
        return []
    segments = []
    for line in text.splitlines():
        m = _EDL_EVENT_RE.match(line)
        if not m:
            continue
        src_in = _smpte_to_seconds(m.group(1), fps)
        src_out = _smpte_to_seconds(m.group(2), fps)
        if src_out > src_in:
            segments.append((src_in, src_out))
    return segments


def parse_fcpxml(path: str) -> List[Tuple[float, float]]:
    """Parse a minimal FCPXML and return (start_s, end_s) tuples for
    each asset-clip's offset+duration. Uses the stdlib xml parser so no
    extra dependency is needed. Returns an empty list on parse failure."""
    try:
        try:
            from defusedxml.ElementTree import parse as _safe_parse
            tree = _safe_parse(path)
        except ImportError:
            import xml.etree.ElementTree as ET
            tree = ET.parse(path)
    except Exception as exc:
        logger.warning(f"Cannot parse FCPXML: {exc}")
        return []

    def _rational_to_seconds(val: str) -> float:
        if not val:
            return 0.0
        val = val.strip().rstrip("s")
        if "/" in val:
            num, den = val.split("/", 1)
            try:
                return float(num) / float(den)
            except (ValueError, ZeroDivisionError):
                return 0.0
        try:
            return float(val)
        except ValueError:
            return 0.0

    segments = []
    for clip in tree.iter("asset-clip"):
        offset = _rational_to_seconds(clip.get("offset", "0s"))
        duration = _rational_to_seconds(clip.get("duration", "0s"))
        if duration > 0:
            segments.append((offset, offset + duration))
    return segments


def parse_nle_input(path: str, fps: float = 24.0) -> List[Tuple[float, float]]:
    """Auto-detect EDL vs FCPXML and parse time segments."""
    ext = Path(path).suffix.lower()
    if ext in (".xml", ".fcpxml"):
        return parse_fcpxml(path)
    return parse_edl(path, fps)
