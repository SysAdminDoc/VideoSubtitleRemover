"""SRT export and translation-workflow methods for SubtitleRemover."""

from __future__ import annotations

import logging
import math
import unicodedata
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from backend.detection_geometry import (
    DetectionGeometry,
    as_detection_geometry,
)
from backend.io import VideoFrameTiming, _write_text_atomic

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SrtTextObservation:
    """Recognized text tied to the tracker identity that produced it."""

    track_id: Optional[int]
    text: str
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None


@dataclass(frozen=True)
class SrtFrameObservation:
    """All recognized subtitle text observed on one source frame."""

    frame_idx: int
    detections: Tuple[SrtTextObservation, ...]

    @property
    def text(self) -> str:
        return " ".join(
            detection.text for detection in self.detections
            if detection.text
        ).strip()

    @property
    def confidence(self) -> float:
        if not self.detections:
            return 1.0
        weights = [
            max(1, len(_grapheme_clusters(_comparison_text(item.text))))
            for item in self.detections
        ]
        return sum(
            item.confidence * weight
            for item, weight in zip(self.detections, weights)
        ) / float(sum(weights))

    @property
    def track_ids(self) -> frozenset[int]:
        return frozenset(
            int(item.track_id)
            for item in self.detections
            if item.track_id is not None
        )


@dataclass(frozen=True)
class _CueObservation:
    frame_idx: int
    text: str
    confidence: float
    track_ids: frozenset[int]


def _display_text(text: str) -> str:
    """Normalize canonical Unicode forms while preserving visible content."""
    return " ".join(
        unicodedata.normalize("NFC", str(text or "")).split()
    ).strip()


def _comparison_text(text: str) -> str:
    """Return a compatibility-normalized form used only for matching."""
    return " ".join(
        unicodedata.normalize("NFKC", str(text or "")).casefold().split()
    ).strip()


def _grapheme_clusters(text: str) -> List[str]:
    """Conservatively segment Unicode graphemes without an optional package."""
    clusters: List[str] = []
    regional_run = 0
    for character in str(text or ""):
        codepoint = ord(character)
        category = unicodedata.category(character)
        is_regional = 0x1F1E6 <= codepoint <= 0x1F1FF
        is_extend = (
            category.startswith("M")
            or 0xFE00 <= codepoint <= 0xFE0F
            or 0xE0100 <= codepoint <= 0xE01EF
            or 0x1F3FB <= codepoint <= 0x1F3FF
        )
        if not clusters:
            clusters.append(character)
        elif (
            is_extend
            or character == "\u200d"
            or clusters[-1].endswith("\u200d")
        ):
            clusters[-1] += character
        elif is_regional and regional_run % 2 == 1:
            clusters[-1] += character
        else:
            clusters.append(character)
        regional_run = regional_run + 1 if is_regional else 0
    return clusters


def _edit_distance(first: Sequence[str], second: Sequence[str]) -> int:
    """Compute grapheme-level Levenshtein distance."""
    if len(first) > len(second):
        first, second = second, first
    previous = list(range(len(first) + 1))
    for row, right in enumerate(second, 1):
        current = [row]
        for column, left in enumerate(first, 1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (left != right),
            ))
        previous = current
    return previous[-1]


def _text_similarity(first: str, second: str) -> float:
    left = _grapheme_clusters(_comparison_text(first))
    right = _grapheme_clusters(_comparison_text(second))
    longest = max(len(left), len(right))
    if longest == 0:
        return 1.0
    return 1.0 - (_edit_distance(left, right) / float(longest))


def _near_equivalent(
    first: str,
    second: str,
    first_confidence: float,
    second_confidence: float,
) -> bool:
    left = _grapheme_clusters(_comparison_text(first))
    right = _grapheme_clusters(_comparison_text(second))
    if left == right:
        return True
    longest = max(len(left), len(right))
    if min(len(left), len(right)) == 0 or longest < 4:
        return False
    distance = _edit_distance(left, right)
    allowed = max(1, int(longest * 0.18))
    similarity_floor = 0.75 if longest <= 7 else 0.8
    if (
        distance > allowed
        or 1.0 - distance / float(longest) < similarity_floor
    ):
        return False
    # A single changed grapheme is a large semantic difference in short
    # captions. Only absorb it when at least one OCR reading was uncertain.
    if longest <= 7 and min(first_confidence, second_confidence) >= 0.85:
        return False
    return True


def _safe_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 1.0
    if not math.isfinite(confidence):
        confidence = 1.0
    return max(0.01, min(1.0, confidence))


def _strong_direction(text: str) -> str:
    for character in str(text or ""):
        direction = unicodedata.bidirectional(character)
        if direction in {"R", "AL"}:
            return "rtl"
        if direction == "L":
            return "ltr"
    return ""


def _ordered_text_detections(
    detections: Sequence[DetectionGeometry],
) -> List[DetectionGeometry]:
    """Order OCR lines top-to-bottom and respect strong RTL line direction."""
    lines: List[List[DetectionGeometry]] = []
    for detection in sorted(
        detections,
        key=lambda item: (item.bbox[1], item.bbox[0]),
    ):
        target = None
        for line in lines:
            top = min(item.bbox[1] for item in line)
            bottom = max(item.bbox[3] for item in line)
            line_height = max(1, bottom - top)
            item_height = max(1, detection.bbox[3] - detection.bbox[1])
            overlap = max(
                0,
                min(bottom, detection.bbox[3])
                - max(top, detection.bbox[1]),
            )
            if overlap / float(min(line_height, item_height)) >= 0.5:
                target = line
                break
        if target is None:
            lines.append([detection])
        else:
            target.append(detection)
    lines.sort(key=lambda line: min(item.bbox[1] for item in line))
    ordered: List[DetectionGeometry] = []
    for line in lines:
        direction = next(
            (
                value
                for item in line
                for value in [_strong_direction(item.text)]
                if value
            ),
            "ltr",
        )
        ordered.extend(sorted(
            line,
            key=lambda item: item.bbox[0],
            reverse=direction == "rtl",
        ))
    return ordered


def _entry_observation(entry: Any) -> Optional[_CueObservation]:
    if isinstance(entry, SrtFrameObservation):
        text = _display_text(entry.text)
        if not text:
            return None
        return _CueObservation(
            int(entry.frame_idx),
            text,
            _safe_confidence(entry.confidence),
            entry.track_ids,
        )
    try:
        frame_idx = int(entry[0])
        text = _display_text(entry[1])
    except (IndexError, TypeError, ValueError):
        return None
    if not text:
        return None
    confidence = 1.0
    track_ids: frozenset[int] = frozenset()
    try:
        if len(entry) > 2:
            confidence = _safe_confidence(entry[2])
        if len(entry) > 3:
            track_ids = frozenset(int(value) for value in entry[3])
    except (TypeError, ValueError):
        track_ids = frozenset()
    return _CueObservation(frame_idx, text, confidence, track_ids)


def _consensus_text(observations: Sequence[_CueObservation]) -> str:
    """Choose the variant with the strongest confidence-weighted support."""
    candidates: List[str] = []
    for observation in observations:
        if observation.text not in candidates:
            candidates.append(observation.text)
    best_text = candidates[0]
    best_rank = (-1.0, -1.0, -1.0, 0)
    for index, candidate in enumerate(candidates):
        canonical = _comparison_text(candidate)
        weighted_support = sum(
            observation.confidence
            * (_text_similarity(candidate, observation.text) ** 2)
            for observation in observations
        )
        exact_support = sum(
            observation.confidence
            for observation in observations
            if _comparison_text(observation.text) == canonical
        )
        peak_confidence = max(
            observation.confidence
            for observation in observations
            if _comparison_text(observation.text) == canonical
        )
        rank = (
            weighted_support,
            exact_support,
            peak_confidence,
            -index,
        )
        if rank > best_rank:
            best_text = candidate
            best_rank = rank
    return best_text


def _cluster_accepts(
    observations: Sequence[_CueObservation],
    incoming: _CueObservation,
) -> bool:
    candidate = _consensus_text(observations)
    if _comparison_text(candidate) == _comparison_text(incoming.text):
        return True
    existing_tracks = frozenset(
        track_id
        for observation in observations
        for track_id in observation.track_ids
    )
    if (
        existing_tracks
        and incoming.track_ids
        and existing_tracks.isdisjoint(incoming.track_ids)
    ):
        return False
    candidate_confidence = max(
        observation.confidence for observation in observations
    )
    return _near_equivalent(
        candidate,
        incoming.text,
        candidate_confidence,
        incoming.confidence,
    )


class _SrtMixin:
    """SRT writing, OCR fixes, and translation-preparation methods."""

    def _collect_srt_entry(
        self,
        frame: np.ndarray,
        frame_idx: int,
        detections: Sequence[Any],
    ) -> None:
        """Collect detector text, falling back only when recognition is absent."""
        geometry = [
            detection
            for value in detections or []
            for detection in [as_detection_geometry(value)]
            if detection is not None
        ]
        recognized = [item for item in geometry if item.text.strip()]
        if recognized:
            observations = []
            for detection in _ordered_text_detections(recognized):
                text = _display_text(detection.text)
                if text and getattr(self.config, "ocr_fix_enable", False):
                    text = self._apply_ocr_fixes(text)
                text = _display_text(text)
                if text:
                    observations.append(SrtTextObservation(
                        detection.track_id,
                        text,
                        _safe_confidence(detection.confidence),
                        detection.bbox,
                    ))
            if observations:
                self._srt_entries.append(SrtFrameObservation(
                    int(frame_idx), tuple(observations)
                ))
            return

        boxes = [detection.bbox for detection in geometry]
        try:
            text = self._read_text_for_boxes(frame, boxes)
        except Exception:
            logger.warning("SRT text collection failed", exc_info=True)
            text = ""
        if text and getattr(self.config, "ocr_fix_enable", False):
            text = self._apply_ocr_fixes(text)
        text = _display_text(text)
        if text:
            self._srt_entries.append(SrtFrameObservation(
                int(frame_idx),
                (SrtTextObservation(None, text, 1.0),),
            ))

    def _apply_ocr_fixes(self, text: str) -> str:
        """Apply the per-language OCR-fix replace list to detected SRT text.
        Loaded once per job and cached on the instance."""
        replacements = getattr(self, "_ocr_fix_replacements", None)
        if replacements is None:
            try:
                from backend.ocr_fix import load_ocr_fix_replacements
                replacements = load_ocr_fix_replacements(
                    getattr(self.config, "detection_lang", "en"))
            except Exception:
                logger.warning("OCR-fix list load failed", exc_info=True)
                replacements = {}
            self._ocr_fix_replacements = replacements
        if not replacements:
            return text
        try:
            from backend.ocr_fix import apply_ocr_fixes
            return apply_ocr_fixes(text, replacements)
        except Exception:
            logger.warning("OCR-fix application failed", exc_info=True)
            return text

    def _read_text_for_boxes(self, frame: np.ndarray,
                               boxes: List[Tuple[int, int, int, int]]) -> str:
        """Best-effort text extraction. Returns an empty string when the
        underlying engine doesn't expose a recognition path.
        """
        if not boxes:
            return ""
        # RapidOCR returns (poly, text, conf)
        if self.detector._rapid_model is not None:
            try:
                output = self.detector._rapid_model(frame)
                texts = []
                if isinstance(output, tuple) and output and output[0]:
                    for entry in output[0]:
                        if len(entry) >= 2 and entry[1]:
                            texts.append(entry[1])
                else:
                    txt_attr = getattr(output, 'txts', None)
                    if txt_attr:
                        texts.extend(t for t in txt_attr if t)
                return " ".join(texts).strip()
            except Exception:
                logger.warning("RapidOCR SRT extraction failed", exc_info=True)
        # PaddleOCR (line[1][0] is the recognised text)
        if self.detector._paddle_model is not None:
            try:
                results = self.detector._paddle_model.ocr(frame, cls=False)
                if results and results[0]:
                    return " ".join(line[1][0] for line in results[0] if line and line[1]).strip()
            except Exception:
                logger.warning("PaddleOCR SRT extraction failed", exc_info=True)
        # EasyOCR: readtext yields (bbox, text, conf)
        if self.detector._easyocr_reader is not None:
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rows = self.detector._easyocr_reader.readtext(frame_rgb)
                return " ".join(r[1] for r in rows if len(r) >= 2 and r[1]).strip()
            except Exception:
                logger.warning("EasyOCR SRT extraction failed", exc_info=True)
        return ""

    def _write_srt(
        self,
        path: str,
        fps: float,
        offset_frames: int = 0,
        *,
        frame_timing: Optional[VideoFrameTiming] = None,
    ):
        """Build confidence-weighted cues and retain exact source timing."""
        if not self._srt_entries:
            return
        # Below one frame per second the rate is broken container metadata
        # rather than a real timelapse: fps=0.001 would stretch a single
        # frame into a 1000-second cue. Keep the floor at 1.0.
        fps = fps if fps and fps > 1.0 else 30.0
        gap_tol = max(1, int(fps * 0.5))

        def ts(t) -> str:
            value = t if isinstance(t, Fraction) else Fraction(str(float(t)))
            if value >= 0:
                ms = (value.numerator * 1000 * 2 + value.denominator) // (
                    2 * value.denominator)
            else:
                ms = 0
            hh, rem = divmod(ms, 3600000)
            mm, rem = divmod(rem, 60000)
            ss, ms = divmod(rem, 1000)
            return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

        entries = [
            observation
            for entry in self._srt_entries
            for observation in [_entry_observation(entry)]
            if observation is not None
        ]
        entries.sort(key=lambda observation: observation.frame_idx)
        if not entries:
            return

        cues: List[Tuple[int, int, str]] = []
        current = [entries[0]]
        cur_start = entries[0].frame_idx
        cur_end = entries[0].frame_idx
        for observation in entries[1:]:
            frame_idx = observation.frame_idx
            if frame_timing is not None:
                previous_end = (
                    frame_timing.frame_time(cur_end + offset_frames, fps)
                    + frame_timing.frame_duration(
                        cur_end + offset_frames, fps)
                )
                current_start = frame_timing.frame_time(
                    frame_idx + offset_frames, fps)
                bridge_gap = current_start - previous_end <= 0.5
            else:
                bridge_gap = frame_idx - cur_end <= gap_tol
            if bridge_gap and _cluster_accepts(current, observation):
                cur_end = frame_idx
                current.append(observation)
            else:
                cues.append((cur_start, cur_end, _consensus_text(current)))
                current = [observation]
                cur_start = frame_idx
                cur_end = frame_idx
        cues.append((cur_start, cur_end, _consensus_text(current)))

        try:
            payload = []
            for i, (s, e, txt) in enumerate(cues, 1):
                if frame_timing is not None:
                    absolute_start = s + offset_frames
                    absolute_end = e + offset_frames
                    t_start = frame_timing.frame_time_fraction(
                        absolute_start, fps)
                    t_end = (
                        frame_timing.frame_time_fraction(absolute_end, fps)
                        + frame_timing.frame_duration_fraction(
                            absolute_end, fps)
                    )
                else:
                    t_start = (s + offset_frames) / fps
                    t_end = (e + offset_frames + 1) / fps
                payload.append(f"{i}\n{ts(t_start)} --> {ts(t_end)}\n{txt}\n\n")
            _write_text_atomic(Path(path), "".join(payload))
            logger.info(f"SRT written: {path} ({len(cues)} cues)")
        except Exception as exc:
            logger.warning(f"SRT write failed: {exc}", exc_info=True)

    def _prepare_translation_workflow(
        self,
        input_path: str,
        output_path: str,
        fps: float,
        offset_frames: int = 0,
        *,
        frame_timing: Optional[VideoFrameTiming] = None,
    ) -> None:
        """Resolve or generate the translated SRT before post-processing."""
        if not self.config.translation_enabled:
            return
        if self.config.restyle_subtitle:
            raise ValueError(
                "translation workflow cannot be combined with restyle_subtitle")
        if Path(output_path).is_dir():
            raise ValueError(
                "translation re-embedding requires encoded video output")

        from backend.subtitle_translation import (
            SubtitleTranslationError,
            provided_translation_evidence,
            render_segments_srt,
            translate_srt_file,
            translated_srt_path,
        )

        style_configured = bool(self.config.translation_style.strip())
        if self.config.translation_srt:
            translated_path = Path(self.config.translation_srt)
            report = provided_translation_evidence(
                translated_path,
                target_language=self.config.translation_target_lang,
            )
        else:
            source_kind = "provided-source-srt"
            if self.config.translation_source_srt:
                source_path = Path(self.config.translation_source_srt)
            else:
                source_path = (
                    Path(output_path).with_suffix(".srt")
                    if self.config.export_srt
                    else Path(output_path).with_name(
                        f"{Path(output_path).stem}.source.srt")
                )
                if self._srt_entries:
                    self._write_srt(
                        str(source_path),
                        fps,
                        0,
                        frame_timing=frame_timing,
                    )
                    source_kind = "ocr-srt"
                elif getattr(self, "_whisper_segments", None):
                    _write_text_atomic(
                        source_path,
                        render_segments_srt(self._whisper_segments),
                    )
                    source_kind = "whisper-srt"
                else:
                    raise SubtitleTranslationError(
                        "translation needs --translation-source-srt, OCR text, "
                        "or an enabled Whisper transcript")
            # RM-154: a WebVTT source stays WebVTT through translation, so
            # the sidecar keeps the cue settings, regions, and markup the
            # SRT model would have flattened.
            from backend.subtitle_translation import subtitle_format

            translated_path = translated_srt_path(
                output_path,
                self.config.translation_target_lang,
                suffix=f".{subtitle_format(source_path)}",
            )
            report = translate_srt_file(
                source_path,
                translated_path,
                provider_name=self.config.translation_provider,
                source_language=self.config.translation_source_lang,
                target_language=self.config.translation_target_lang,
                provider_options={
                    "command": self.config.translation_command,
                    "timeout": self.config.translation_timeout_seconds,
                },
                source_kind=source_kind,
            )
        report["styleConfigured"] = style_configured
        report["mediaSource"] = Path(input_path).name
        self.last_translation = report
        self._translation_burn_path = str(translated_path)
        logger.info(
            "Translation captions ready: %s (%s, %d cues)",
            translated_path,
            report.get("provider", "unknown"),
            int(report.get("cueCount", 0) or 0),
        )
