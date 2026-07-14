"""Probe, plan, and verify non-primary-video container payloads."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping


_MATROSKA = {".mkv", ".mka"}
_MP4 = {".mp4", ".mov", ".m4v"}
_WEBM = {".webm"}
_AVI = {".avi"}

_MP4_AUDIO_COPY = {"aac", "ac3", "eac3", "alac", "mp3"}
_WEBM_AUDIO_COPY = {"opus", "vorbis"}
_AVI_AUDIO_COPY = {"mp3", "ac3", "pcm_s16le", "pcm_s24le"}
_TEXT_SUBTITLES = {
    "ass", "ssa", "subrip", "srt", "text", "webvtt", "mov_text",
}
_DISPOSITION_FLAGS = (
    "default", "dub", "original", "comment", "lyrics", "karaoke",
    "forced", "hearing_impaired", "visual_impaired", "clean_effects",
    "attached_pic", "timed_thumbnails", "non_diegetic", "still_image",
)
_PRESERVED_TAGS = {"language", "title", "filename", "mimetype", "handler_name"}
_PRESERVED_FORMAT_TAGS = {"title", "artist", "album", "comment", "date", "genre"}


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _rotation(stream: Mapping[str, Any]) -> int:
    tags = stream.get("tags") if isinstance(stream.get("tags"), Mapping) else {}
    candidates = [tags.get("rotate")]
    side_data = stream.get("side_data_list")
    if isinstance(side_data, list):
        candidates.extend(
            item.get("rotation")
            for item in side_data
            if isinstance(item, Mapping)
        )
    for candidate in candidates:
        try:
            value = int(round(float(candidate))) % 360
        except (TypeError, ValueError):
            continue
        return value
    return 0


def probe_container_manifest(path: str | Path, timeout: float = 30.0) -> dict:
    payload = {
        "schema": "vsr.container_manifest.v1",
        "path": str(path),
        "available": False,
        "format": {},
        "streams": [],
        "chapters": [],
        "rotationDegrees": 0,
        "error": "",
    }
    if Path(path).is_dir() or shutil.which("ffprobe") is None:
        payload["error"] = "ffprobe is unavailable"
        return payload
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_streams", "-show_chapters",
                "-show_format", "-of", "json", str(path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            payload["error"] = (result.stderr or "ffprobe failed")[-1000:]
            return payload
        raw = json.loads(result.stdout or "{}")
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        payload["error"] = str(exc)
        return payload

    raw_format = raw.get("format") if isinstance(raw.get("format"), Mapping) else {}
    format_tags = (
        raw_format.get("tags") if isinstance(raw_format.get("tags"), Mapping) else {}
    )
    payload["format"] = {
        "name": str(raw_format.get("format_name") or ""),
        "duration": str(raw_format.get("duration") or ""),
        "tags": {str(key): str(value) for key, value in format_tags.items()},
    }
    streams = raw.get("streams") if isinstance(raw.get("streams"), list) else []
    for raw_stream in streams:
        if not isinstance(raw_stream, Mapping):
            continue
        tags = raw_stream.get("tags") if isinstance(raw_stream.get("tags"), Mapping) else {}
        disposition = (
            raw_stream.get("disposition")
            if isinstance(raw_stream.get("disposition"), Mapping)
            else {}
        )
        stream = {
            "index": _int(raw_stream.get("index"), -1),
            "codecType": str(raw_stream.get("codec_type") or ""),
            "codecName": str(raw_stream.get("codec_name") or ""),
            "codecTag": str(raw_stream.get("codec_tag_string") or ""),
            "width": _int(raw_stream.get("width")),
            "height": _int(raw_stream.get("height")),
            "tags": {str(key): str(value) for key, value in tags.items()},
            "disposition": {
                flag: bool(_int(disposition.get(flag)))
                for flag in _DISPOSITION_FLAGS
            },
            "rotationDegrees": _rotation(raw_stream),
        }
        payload["streams"].append(stream)
    chapters = raw.get("chapters") if isinstance(raw.get("chapters"), list) else []
    payload["chapters"] = [
        {
            "id": _int(chapter.get("id"), index),
            "startTime": str(chapter.get("start_time") or ""),
            "endTime": str(chapter.get("end_time") or ""),
            "tags": {
                str(key): str(value)
                for key, value in (
                    chapter.get("tags")
                    if isinstance(chapter.get("tags"), Mapping)
                    else {}
                ).items()
            },
        }
        for index, chapter in enumerate(chapters)
        if isinstance(chapter, Mapping)
    ]
    primary_video = next(
        (stream for stream in payload["streams"] if stream["codecType"] == "video"),
        None,
    )
    payload["rotationDegrees"] = (
        int(primary_video.get("rotationDegrees", 0)) if primary_video else 0
    )
    payload["available"] = True
    return payload


def _stream_action(
    stream: Mapping[str, Any],
    suffix: str,
    *,
    preserve_audio: bool,
    multi_audio: bool,
    audio_number: int,
    loudnorm_target: float,
    include_auxiliary: bool,
    force_audio_transcode: bool,
    primary_video_index: int,
) -> tuple[str, str, str]:
    stream_type = str(stream.get("codecType") or "")
    codec = str(stream.get("codecName") or "").lower()
    codec_tag = str(stream.get("codecTag") or "").lower()
    disposition = stream.get("disposition") or {}
    index = _int(stream.get("index"), -1)
    if stream_type == "video":
        if index == primary_video_index:
            return "replace", "", "primary video is replaced by processed frames"
        if bool(disposition.get("attached_pic")) and include_auxiliary:
            return "copy", codec, "compatible cover-art stream"
        return "drop", "", "alternate video streams are not processed"
    if stream_type == "audio":
        if not preserve_audio:
            return "drop", "", "audio preservation is disabled"
        if not multi_audio and audio_number > 0:
            return "drop", "", "multi-track audio preservation is disabled"
        if loudnorm_target:
            codec_out = "opus" if suffix in _WEBM else "aac"
            return "transcode", codec_out, "loudness normalization was requested"
        if force_audio_transcode:
            codec_out = "opus" if suffix in _WEBM else "aac"
            return "transcode", codec_out, "copy fallback requires a compatible codec"
        if suffix in _MATROSKA:
            return "copy", codec, "Matroska supports the source audio codec"
        if suffix in _MP4 and codec in _MP4_AUDIO_COPY:
            return "copy", codec, "MP4/MOV supports the source audio codec"
        if suffix in _WEBM and codec in _WEBM_AUDIO_COPY:
            return "copy", codec, "WebM supports the source audio codec"
        if suffix in _AVI and codec in _AVI_AUDIO_COPY:
            return "copy", codec, "AVI supports the source audio codec"
        codec_out = "opus" if suffix in _WEBM else ("mp3" if suffix in _AVI else "aac")
        return "transcode", codec_out, f"{suffix or 'target container'} requires a compatible audio codec"
    if not include_auxiliary:
        return "drop", "", "auxiliary-stream fallback is active"
    if stream_type == "subtitle":
        if suffix in _MATROSKA:
            return "copy", codec, "Matroska supports the source subtitle codec"
        if suffix in _MP4 and codec == "mov_text":
            return "copy", codec, "MP4/MOV supports mov_text subtitles"
        if suffix in _MP4 and codec in _TEXT_SUBTITLES:
            return "transcode", "mov_text", "text subtitles require MP4/MOV conversion"
        if suffix in _WEBM and codec == "webvtt":
            return "copy", codec, "WebM supports WebVTT subtitles"
        if suffix in _WEBM and codec in _TEXT_SUBTITLES:
            return "transcode", "webvtt", "text subtitles require WebVTT conversion"
        return "drop", "", f"{suffix or 'target container'} cannot preserve this subtitle codec"
    if stream_type == "attachment":
        if suffix in _MATROSKA:
            return "copy", codec, "Matroska supports attachments"
        return "drop", "", f"{suffix or 'target container'} does not support attachments"
    if stream_type == "data":
        if suffix in _MP4 and (codec == "tmcd" or codec_tag == "tmcd"):
            return "recreate", "tmcd", "MP4/MOV recreates the timecode track from video metadata"
        return "drop", "", f"{suffix or 'target container'} cannot safely copy this data stream"
    return "drop", "", f"unsupported stream type {stream_type or 'unknown'}"


def build_container_mux_plan(
    manifest: Mapping[str, Any],
    output_path: str | Path,
    *,
    preserve_audio: bool,
    multi_audio: bool,
    loudnorm_target: float = 0.0,
    include_auxiliary: bool = True,
    force_audio_transcode: bool = False,
    start_seconds: float = 0.0,
    end_seconds: float = 0.0,
) -> dict:
    suffix = Path(output_path).suffix.lower()
    streams = manifest.get("streams") if isinstance(manifest.get("streams"), list) else []
    primary_video = next(
        (stream for stream in streams if stream.get("codecType") == "video"),
        {},
    )
    primary_video_index = _int(primary_video.get("index"), -1)
    chapters = manifest.get("chapters") if isinstance(manifest.get("chapters"), list) else []
    range_start = max(0.0, float(start_seconds or 0.0))
    range_end = max(0.0, float(end_seconds or 0.0))
    selected_chapters = chapters
    if range_end > range_start:
        selected_chapters = [
            chapter for chapter in chapters
            if _float(chapter.get("endTime")) > range_start
            and _float(chapter.get("startTime")) < range_end
        ]
    plan = {
        "schema": "vsr.container_payload.v1",
        "status": "planned",
        "manifestAvailable": bool(manifest.get("available")),
        "fallbackAudio": bool(preserve_audio),
        "fallbackMultiAudio": bool(multi_audio),
        "sourceFormat": str((manifest.get("format") or {}).get("name") or ""),
        "outputContainer": suffix.lstrip(".") or "unknown",
        "streams": [],
        "chapters": {
            "sourceCount": len(selected_chapters),
            "sourceTotal": len(chapters),
            "action": "copy" if suffix in (_MATROSKA | _MP4) else "drop",
        },
        "timeRange": {
            "startSeconds": range_start,
            "endSeconds": range_end,
        },
        "globalMetadata": {
            "action": "copy",
            "keys": sorted(str(key) for key in ((manifest.get("format") or {}).get("tags") or {})),
        },
        "sourceTags": dict((manifest.get("format") or {}).get("tags") or {}),
        "rotation": {
            "sourceDegrees": _int(manifest.get("rotationDegrees")),
            "action": "bake-and-clear" if _int(manifest.get("rotationDegrees")) else "clear",
        },
        "warnings": [],
        "issues": [],
    }
    if not manifest.get("available"):
        plan["status"] = "degraded"
        plan["warnings"].append(
            "Source container inventory is unavailable; auxiliary streams cannot be proven preserved."
        )
        return plan

    excluded_chapters = len(chapters) - len(selected_chapters)
    if excluded_chapters:
        plan["warnings"].append(
            f"Excluded {excluded_chapters} chapter(s) outside the selected processing range."
        )

    audio_number = 0
    for stream in streams:
        action, output_codec, reason = _stream_action(
            stream,
            suffix,
            preserve_audio=preserve_audio,
            multi_audio=multi_audio,
            audio_number=audio_number,
            loudnorm_target=loudnorm_target,
            include_auxiliary=include_auxiliary,
            force_audio_transcode=force_audio_transcode,
            primary_video_index=primary_video_index,
        )
        if stream.get("codecType") == "audio":
            audio_number += 1
        record = {
            "sourceIndex": _int(stream.get("index"), -1),
            "type": str(stream.get("codecType") or "unknown"),
            "codec": str(
                stream.get("codecName")
                or (stream.get("codecTag") if stream.get("codecType") == "data" else "")
                or ""
            ),
            "action": action,
            "outputCodec": output_codec,
            "reason": reason,
            "tags": dict(stream.get("tags") or {}),
            "disposition": dict(stream.get("disposition") or {}),
        }
        plan["streams"].append(record)
        if action == "drop":
            plan["warnings"].append(
                f"Dropped {record['type']} stream {record['sourceIndex']} "
                f"({record['codec'] or 'unknown'}): {reason}."
            )
        elif action == "transcode":
            suffix_note = " Styling may change." if record["type"] == "subtitle" else ""
            plan["warnings"].append(
                f"Transcoded {record['type']} stream {record['sourceIndex']} "
                f"from {record['codec'] or 'unknown'} to {output_codec}: {reason}.{suffix_note}"
            )
        elif action == "recreate":
            plan["warnings"].append(
                f"Recreated {record['type']} stream {record['sourceIndex']} as "
                f"{output_codec}: {reason}; container-generated handler metadata may differ."
            )
    if plan["chapters"]["sourceCount"] and plan["chapters"]["action"] == "drop":
        plan["warnings"].append(
            f"Dropped {plan['chapters']['sourceCount']} chapters: target container does not support them."
        )
    return plan


def _disposition_value(disposition: Mapping[str, Any]) -> str:
    enabled = [flag for flag in _DISPOSITION_FLAGS if bool(disposition.get(flag))]
    return "+".join(enabled) if enabled else "0"


def build_container_mux_args(
    plan: Mapping[str, Any],
    *,
    input_index: int = 1,
    loudnorm_target: float = 0.0,
) -> list[str]:
    if not plan.get("manifestAvailable"):
        args: list[str] = []
        if plan.get("fallbackAudio"):
            selector = f"{input_index}:a?" if plan.get("fallbackMultiAudio") else f"{input_index}:a:0?"
            args += ["-map", selector, "-c:a", "aac"]
        args += [
            "-map_metadata", str(input_index), "-map_chapters", str(input_index),
            "-metadata:s:v:0", "rotate=0",
        ]
        return args
    mapped = [
        stream for stream in plan.get("streams", [])
        if stream.get("action") in {"copy", "transcode"}
    ]
    recreated = [
        stream for stream in plan.get("streams", [])
        if stream.get("action") == "recreate"
    ]
    args: list[str] = []
    audio = [stream for stream in mapped if stream.get("type") == "audio"]
    if loudnorm_target and audio:
        filters = []
        for output_index, stream in enumerate(audio):
            filters.append(
                f"[{input_index}:{stream['sourceIndex']}]"
                f"loudnorm=I={loudnorm_target}:TP=-1.5:LRA=11[a{output_index}]"
            )
        args += ["-filter_complex", ";".join(filters)]

    type_counts = {"audio": 0, "subtitle": 0, "attachment": 0, "data": 0, "video": 1}
    stream_chars = {
        "audio": "a", "subtitle": "s", "attachment": "t", "data": "d", "video": "v",
    }
    for stream in mapped:
        stream_type = str(stream.get("type") or "")
        char = stream_chars.get(stream_type)
        if char is None:
            continue
        output_index = type_counts[stream_type]
        type_counts[stream_type] += 1
        if stream_type == "audio" and loudnorm_target:
            args += ["-map", f"[a{output_index}]"]
        else:
            args += ["-map", f"{input_index}:{stream['sourceIndex']}"]
        codec = "copy" if stream.get("action") == "copy" else str(stream.get("outputCodec") or "copy")
        args += [f"-c:{char}:{output_index}", codec]
        args += [
            f"-map_metadata:s:{char}:{output_index}",
            f"{input_index}:s:{stream['sourceIndex']}",
            f"-disposition:{char}:{output_index}",
            _disposition_value(stream.get("disposition") or {}),
        ]
        tags = stream.get("tags") or {}
        for key in _PRESERVED_TAGS:
            if key not in tags:
                continue
            output_key = (
                "handler_name"
                if key == "title" and plan.get("outputContainer") in {"mp4", "mov", "m4v"}
                else key
            )
            args += [
                f"-metadata:s:{char}:{output_index}",
                f"{output_key}={tags[key]}",
            ]
    args += [
        "-map_metadata", str(input_index),
        "-map_metadata:s:v:0", f"{input_index}:s:v:0",
        "-map_chapters", (
            str(input_index)
            if (plan.get("chapters") or {}).get("action") == "copy"
            else "-1"
        ),
        "-metadata:s:v:0", "rotate=0",
    ]
    for output_index, stream in enumerate(recreated):
        char = "d" if stream.get("type") == "data" else "v"
        for key in _PRESERVED_TAGS:
            tags = stream.get("tags") or {}
            if key in tags:
                args += [
                    f"-metadata:s:{char}:{output_index}",
                    f"{key}={tags[key]}",
                ]
        args += [
            f"-disposition:{char}:{output_index}",
            _disposition_value(stream.get("disposition") or {}),
        ]
    return args


def validate_container_payload(plan: Mapping[str, Any], output_path: str | Path) -> dict:
    report = copy.deepcopy(dict(plan))
    manifest = probe_container_manifest(output_path)
    issues: list[str] = []
    if not manifest.get("available"):
        issues.append(f"output container probe failed: {manifest.get('error') or 'unknown error'}")
    else:
        expected_by_type: dict[str, list[Mapping[str, Any]]] = {}
        for stream in report.get("streams", []):
            if stream.get("action") in {"copy", "transcode", "recreate"}:
                expected_by_type.setdefault(str(stream.get("type")), []).append(stream)
        actual_by_type: dict[str, list[Mapping[str, Any]]] = {}
        for stream in manifest.get("streams", []):
            actual_by_type.setdefault(str(stream.get("codecType")), []).append(stream)
        for stream_type, expected_streams in expected_by_type.items():
            actual_streams = actual_by_type.get(stream_type, [])
            if stream_type == "video":
                actual_streams = actual_streams[1:]
            if len(actual_streams) != len(expected_streams):
                issues.append(
                    f"{stream_type} stream count is {len(actual_streams)}, expected {len(expected_streams)}"
                )
                continue
            for expected, actual in zip(expected_streams, actual_streams):
                expected_codec = (
                    expected.get("codec")
                    if expected.get("action") == "copy"
                    else expected.get("outputCodec")
                )
                actual_codec = actual.get("codecName") or actual.get("codecTag")
                if expected_codec and actual_codec != expected_codec:
                    issues.append(
                        f"{stream_type} stream {expected.get('sourceIndex')} codec is "
                        f"{actual_codec or 'missing'}, expected {expected_codec}"
                    )
                expected_tags = expected.get("tags") or {}
                actual_tags = actual.get("tags") or {}
                for key in _PRESERVED_TAGS:
                    if expected.get("action") == "recreate" and key == "handler_name":
                        continue
                    actual_value = actual_tags.get(key)
                    if key == "title" and not actual_value:
                        actual_value = actual_tags.get("handler_name")
                    if key in expected_tags and actual_value != expected_tags.get(key):
                        issues.append(
                            f"{stream_type} stream {expected.get('sourceIndex')} lost {key} metadata"
                        )
                expected_disposition = expected.get("disposition") or {}
                actual_disposition = actual.get("disposition") or {}
                for flag in _DISPOSITION_FLAGS:
                    if bool(expected_disposition.get(flag)) and not bool(actual_disposition.get(flag)):
                        issues.append(
                            f"{stream_type} stream {expected.get('sourceIndex')} lost {flag} disposition"
                        )
        chapters = report.get("chapters") or {}
        if chapters.get("action") == "copy" and len(manifest.get("chapters") or []) != int(chapters.get("sourceCount") or 0):
            issues.append(
                f"chapter count is {len(manifest.get('chapters') or [])}, expected {chapters.get('sourceCount') or 0}"
            )
        source_tags = (report.get("globalMetadata") or {}).get("keys") or []
        source_manifest_tags = {
            str(key).lower(): str(value)
            for key, value in ((manifest.get("format") or {}).get("tags") or {}).items()
        }
        original_tags = {
            str(key).lower(): str(value)
            for key, value in ((plan.get("sourceTags") or {})).items()
        }
        for key in _PRESERVED_FORMAT_TAGS & {str(item).lower() for item in source_tags}:
            if original_tags.get(key) and source_manifest_tags.get(key) != original_tags.get(key):
                issues.append(f"global {key} metadata was not preserved")
        if _int((report.get("rotation") or {}).get("sourceDegrees")) and _int(manifest.get("rotationDegrees")):
            issues.append("source rotation metadata was not cleared after pixels were decoded")
    report["status"] = (
        "failed" if issues
        else ("preserved" if report.get("manifestAvailable") else "degraded")
    )
    report["issues"] = issues
    return report
