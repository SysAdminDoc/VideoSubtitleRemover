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
from pathlib import Path
from typing import Optional


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
              title: str = "VSR cleanup") -> str:
    """Write a 1-event CMX 3600 EDL. `start_s`/`end_s` describe the
    in/out of the cleaned region within the original media; both EDL
    columns get the same range. Returns the path written."""
    src_in = _ts_to_smpte(start_s, fps)
    src_out = _ts_to_smpte(end_s, fps)
    payload = []
    payload.append(f"TITLE: {title}")
    payload.append(f"FCM: NON-DROP FRAME")
    payload.append("")
    payload.append(
        f"001  AX       V     C        "
        f"{src_in} {src_out} {src_in} {src_out}"
    )
    payload.append(f"* FROM CLIP NAME: {Path(source).name}")
    payload.append(f"* TO CLIP NAME:   {Path(cleaned).name}")
    payload.append(f"* CLEANED BY:     Video Subtitle Remover Pro")
    payload.append("")
    text = "\n".join(payload) + "\n"
    Path(path).write_text(text, encoding="utf-8")
    return path


def write_fcpxml(path: str, source: str, cleaned: str,
                  fps: float, start_s: float, end_s: float) -> str:
    """Write a minimal FCPXML 1.10 stub with one asset-clip referencing
    the cleaned file. Real FCPXMLs are huge; this stub only names the
    asset + range so a DaVinci / Premiere import re-creates a single
    clip the editor can hand-conform to the source slot.

    We use ascii-safe XML (no smart quotes, no em-dashes) so the file
    survives PyInstaller's text bundling and arbitrary editor parsers.
    """
    src_name = Path(source).stem
    cleaned_uri = "file://" + str(Path(cleaned).resolve()).replace("\\", "/")
    rate = max(1, int(round(fps if fps > 0 else 24.0)))
    duration_frames = max(1, int(round((end_s - start_s) * rate)))
    start_frames = max(0, int(round(start_s * rate)))
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    payload = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<!DOCTYPE fcpxml>\n"
        "<fcpxml version=\"1.10\">\n"
        "  <resources>\n"
        f"    <format id=\"r0\" frameDuration=\"1/{rate}s\" "
        f"width=\"1920\" height=\"1080\"/>\n"
        f"    <asset id=\"r1\" name=\"{src_name}\" "
        f"start=\"0s\" duration=\"{duration_frames}/{rate}s\" "
        f"hasVideo=\"1\" hasAudio=\"1\" format=\"r0\" src=\"{cleaned_uri}\"/>\n"
        "  </resources>\n"
        "  <library>\n"
        f"    <event name=\"VSR Cleanup {now} UTC\">\n"
        f"      <project name=\"{src_name} -- cleaned\">\n"
        f"        <sequence format=\"r0\">\n"
        f"          <spine>\n"
        f"            <asset-clip name=\"{src_name}\" "
        f"ref=\"r1\" offset=\"{start_frames}/{rate}s\" "
        f"start=\"0s\" duration=\"{duration_frames}/{rate}s\"/>\n"
        "          </spine>\n"
        "        </sequence>\n"
        "      </project>\n"
        "    </event>\n"
        "  </library>\n"
        "</fcpxml>\n"
    )
    Path(path).write_text(payload, encoding="utf-8")
    return path
