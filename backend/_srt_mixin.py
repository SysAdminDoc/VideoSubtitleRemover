"""SRT export and translation-workflow methods for SubtitleRemover."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from backend.io import VideoFrameTiming, _write_text_atomic

logger = logging.getLogger(__name__)


class _SrtMixin:
    """SRT writing, OCR fixes, and translation-preparation methods."""

    def _collect_srt_entry(self, frame: np.ndarray, frame_idx: int,
                             boxes: List[Tuple[int, int, int, int]]):
        """Extract text strings for the detected boxes and append to the SRT
        buffer. We re-use the detector's already-loaded model where possible.
        """
        try:
            text = self._read_text_for_boxes(frame, boxes)
        except Exception:
            logger.warning("SRT text collection failed", exc_info=True)
            text = ""
        if text and getattr(self.config, "ocr_fix_enable", False):
            text = self._apply_ocr_fixes(text)
        if text:
            self._srt_entries.append((frame_idx, text))

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
        """Collapse consecutive per-frame entries with the same text into SRT
        cues and write to disk. Gaps of up to 0.5s are bridged."""
        if not self._srt_entries:
            return
        fps = fps if fps and fps > 1.0 else 30.0
        gap_tol = max(1, int(fps * 0.5))

        def ts(t: float) -> str:
            ms = int(round(t * 1000))
            hh, rem = divmod(ms, 3600000)
            mm, rem = divmod(rem, 60000)
            ss, ms = divmod(rem, 1000)
            return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

        cues: List[Tuple[int, int, str]] = []
        cur_start, cur_end, cur_text = None, None, None
        for frame_idx, text in self._srt_entries:
            if cur_text is None:
                cur_start, cur_end, cur_text = frame_idx, frame_idx, text
                continue
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
            if text == cur_text and bridge_gap:
                cur_end = frame_idx
            else:
                cues.append((cur_start, cur_end, cur_text))
                cur_start, cur_end, cur_text = frame_idx, frame_idx, text
        if cur_text is not None:
            cues.append((cur_start, cur_end, cur_text))

        try:
            payload = []
            for i, (s, e, txt) in enumerate(cues, 1):
                if frame_timing is not None:
                    absolute_start = s + offset_frames
                    absolute_end = e + offset_frames
                    t_start = frame_timing.frame_time(absolute_start, fps)
                    t_end = (
                        frame_timing.frame_time(absolute_end, fps)
                        + frame_timing.frame_duration(absolute_end, fps)
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
                        offset_frames,
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
            translated_path = translated_srt_path(
                output_path, self.config.translation_target_lang)
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

