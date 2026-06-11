"""Subtitle detector cascade (VLM > RapidOCR > PaddleOCR > Surya >
EasyOCR > OpenCV fallback).

Extracted from processor.py as part of RFP-L-1. The cascade dispatches
through `backend.ocr_vlm.maybe_build_vlm_detector` first when the user
opted in via VSR_VLM_OCR / lang="manga", then walks the historical
order. Surya stays gated behind VSR_ALLOW_GPL (RM-B-2).

Vertical-text mode (RM-24) lives here as a wrapper that rotates the
frame, runs the underlying detector, then rotates the boxes back into
the source coordinate space.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _surya_allowed() -> bool:
    """Return True when the user has explicitly opted into the GPL Surya
    detector via VSR_ALLOW_GPL."""
    import os
    val = os.environ.get("VSR_ALLOW_GPL", "").strip().lower()
    return val in {"1", "true", "yes", "on"}


def _onnxruntime_has_provider(provider: str) -> bool:
    try:
        import onnxruntime as ort  # type: ignore
        return provider in ort.get_available_providers()
    except ImportError:
        return False
    except Exception as exc:
        logger.debug(f"ONNX Runtime provider probe failed: {exc}")
        return False


def _rapidocr_directml_params(device: str) -> Optional[Dict[str, Any]]:
    if device != "directml":
        return None
    if not _onnxruntime_has_provider("DmlExecutionProvider"):
        logger.info(
            "RapidOCR DirectML requested but DmlExecutionProvider is not "
            "available; using RapidOCR CPU provider."
        )
        return None
    return {
        "EngineConfig.onnxruntime.use_dml": True,
        "EngineConfig.onnxruntime.use_cuda": False,
        "EngineConfig.onnxruntime.use_cann": False,
        "EngineConfig.onnxruntime.use_coreml": False,
    }


def _build_rapidocr(rapid_cls, device: str):
    directml_params = _rapidocr_directml_params(device)
    if directml_params:
        try:
            return rapid_cls(params=directml_params), "DirectML"
        except Exception as exc:
            logger.warning(
                "RapidOCR DirectML provider init failed; retrying CPU "
                f"provider: {exc}"
            )
    return rapid_cls(), "CPU"


class SubtitleDetector:
    """Detects subtitle regions in video frames using text detection models."""

    def __init__(self, device: str = "cuda:0", lang: str = "en",
                 vertical: bool = False):
        self.device = device
        self.lang = lang
        self.vertical = bool(vertical)
        self._engine_name = "none"
        self._rapid_model = None
        self._paddle_model = None
        self._surya_det = None
        self._surya_processor = None
        self._easyocr_reader = None
        self._vlm_detector = None
        self._load_model()

    def _is_gpu_device(self) -> bool:
        return 'cuda' in self.device

    def _load_model(self):
        """Load detection model: VLM (opt-in) > RapidOCR > PaddleOCR > Surya >
        EasyOCR > OpenCV fallback."""
        try:
            from backend.ocr_vlm import maybe_build_vlm_detector
            vlm = maybe_build_vlm_detector(self.device, self.lang)
            if vlm is not None:
                self._vlm_detector = vlm
                self._engine_name = f"VLM ({vlm.name})"
                logger.info(f"VLM OCR detector active: {vlm.name}")
                return
        except Exception as exc:
            logger.debug(f"VLM detector probe failed: {exc}")
        self._vlm_detector = None

        # RapidOCR (ONNX PP-OCR, default)
        try:
            rapid_obj = None
            try:
                from rapidocr import RapidOCR as _RapidOCR
                rapid_obj, rapid_provider = _build_rapidocr(_RapidOCR, self.device)
            except ImportError:
                from rapidocr_onnxruntime import RapidOCR as _RapidOCR
                rapid_obj, rapid_provider = _build_rapidocr(_RapidOCR, self.device)
            if rapid_obj is not None:
                self._rapid_model = rapid_obj
                self._engine_name = (
                    "RapidOCR (DirectML)"
                    if rapid_provider == "DirectML"
                    else "RapidOCR"
                )
                logger.info(
                    f"RapidOCR loaded via ONNX Runtime {rapid_provider} "
                    f"provider (lang={self.lang})"
                )
                return
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"RapidOCR init failed: {e}")

        # PaddleOCR PP-OCRv5
        try:
            from paddleocr import PaddleOCR
            self._paddle_model = PaddleOCR(
                use_angle_cls=False,
                lang=self.lang,
                use_gpu='cuda' in self.device,
                show_log=False
            )
            self._engine_name = "PaddleOCR"
            logger.info(f"PaddleOCR loaded (lang={self.lang})")
            return
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"PaddleOCR init failed: {e}")

        # Surya (GPL, opt-in via VSR_ALLOW_GPL)
        if _surya_allowed():
            try:
                from surya.detection import DetectionPredictor
                self._surya_det = DetectionPredictor()
                self._engine_name = "Surya"
                logger.info("Surya text detection loaded (GPL opt-in via VSR_ALLOW_GPL)")
                return
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"Surya init failed: {e}")
        else:
            try:
                import surya.detection  # noqa: F401
                logger.warning(
                    "Surya is installed but skipped: it is GPL-licensed and "
                    "VSR ships MIT-clean. Set VSR_ALLOW_GPL=1 to opt in."
                )
            except ImportError:
                pass
            except Exception:
                pass

        # EasyOCR
        try:
            import easyocr
            gpu = self._is_gpu_device()
            easyocr_lang_map = {
                "ch": "ch_sim", "chinese_cht": "ch_tra",
                "ko": "ko", "ja": "ja", "en": "en",
                "fr": "fr", "de": "de", "es": "es", "pt": "pt",
                "ru": "ru", "ar": "ar", "hi": "hi", "it": "it",
            }
            mapped_lang = easyocr_lang_map.get(self.lang, self.lang)
            lang_list = [mapped_lang]
            if mapped_lang != "en":
                lang_list.append("en")
            self._easyocr_reader = easyocr.Reader(lang_list, gpu=gpu, verbose=False)
            self._engine_name = "EasyOCR"
            logger.info(f"EasyOCR loaded (lang={lang_list})")
            return
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"EasyOCR init failed: {e}")

        self._engine_name = "OpenCV fallback"
        logger.warning("No OCR engine available, using OpenCV fallback detection")

    def detect(self, frame: np.ndarray, threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
        """Detect text regions in a frame. Returns axis-aligned boxes.
        When self.vertical is set, rotates the frame 90 CCW before
        detection and rotates returned boxes back into the source
        coordinate space."""
        if self.vertical:
            h, w = frame.shape[:2]
            rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            rotated_boxes = self._detect_axis_aligned(rotated, threshold)
            out: List[Tuple[int, int, int, int]] = []
            for (rx1, ry1, rx2, ry2) in rotated_boxes:
                ox1 = max(0, ry1)
                oy1 = max(0, w - rx2)
                ox2 = min(h, ry2)
                oy2 = min(w, w - rx1)
                if ox2 > ox1 and oy2 > oy1:
                    out.append((ox1, oy1, ox2, oy2))
            return out
        return self._detect_axis_aligned(frame, threshold)

    def _detect_axis_aligned(self, frame: np.ndarray,
                              threshold: float) -> List[Tuple[int, int, int, int]]:
        vlm = getattr(self, "_vlm_detector", None)
        if vlm is not None:
            try:
                return vlm.detect(frame, threshold)
            except Exception as exc:
                logger.warning(f"VLM detector errored, falling back: {exc}")
        if self._rapid_model is not None:
            return self._detect_rapid(frame, threshold)
        elif self._paddle_model is not None:
            return self._detect_paddle(frame, threshold)
        elif self._surya_det is not None:
            return self._detect_surya(frame, threshold)
        elif self._easyocr_reader is not None:
            return self._detect_easyocr(frame, threshold)
        else:
            return self._fallback_detection(frame)

    def _detect_rapid(self, frame: np.ndarray, threshold: float) -> List[Tuple[int, int, int, int]]:
        try:
            output = self._rapid_model(frame)
            results = None
            if output is None:
                return []
            if isinstance(output, tuple) and len(output) >= 1:
                results = output[0]
            else:
                boxes_attr = getattr(output, 'boxes', None)
                scores_attr = getattr(output, 'scores', None)
                if boxes_attr is not None:
                    boxes = []
                    for i, poly in enumerate(boxes_attr):
                        conf = float(scores_attr[i]) if scores_attr is not None else 1.0
                        if conf >= threshold:
                            pts = np.array(poly, dtype=np.int32)
                            x1, y1 = pts.min(axis=0)
                            x2, y2 = pts.max(axis=0)
                            boxes.append((int(x1), int(y1), int(x2), int(y2)))
                    return boxes
                return []

            if not results:
                return []
            boxes = []
            for entry in results:
                if len(entry) < 3:
                    continue
                poly, _text, conf = entry[0], entry[1], entry[2]
                if conf is None:
                    conf = 1.0
                if float(conf) >= threshold:
                    pts = np.array(poly, dtype=np.int32)
                    x1, y1 = pts.min(axis=0)
                    x2, y2 = pts.max(axis=0)
                    boxes.append((int(x1), int(y1), int(x2), int(y2)))
            return boxes
        except Exception as e:
            logger.error(f"RapidOCR detection error: {e}")
            return self._fallback_detection(frame)

    def _detect_paddle(self, frame: np.ndarray, threshold: float) -> List[Tuple[int, int, int, int]]:
        try:
            results = self._paddle_model.ocr(frame, cls=False)
            boxes = []
            if results and results[0]:
                for line in results[0]:
                    if line[1][1] >= threshold:
                        pts = np.array(line[0], dtype=np.int32)
                        x1, y1 = pts.min(axis=0)
                        x2, y2 = pts.max(axis=0)
                        boxes.append((int(x1), int(y1), int(x2), int(y2)))
            return boxes
        except Exception as e:
            logger.error(f"PaddleOCR detection error: {e}")
            return self._fallback_detection(frame)

    def _detect_surya(self, frame: np.ndarray, threshold: float) -> List[Tuple[int, int, int, int]]:
        try:
            from PIL import Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            predictions = self._surya_det([pil_image])
            boxes = []
            if predictions and len(predictions) > 0:
                for bbox in predictions[0].bboxes:
                    if bbox.confidence >= threshold:
                        x1, y1, x2, y2 = [int(v) for v in bbox.bbox]
                        boxes.append((x1, y1, x2, y2))
            return boxes
        except Exception as e:
            logger.error(f"Surya detection error: {e}")
            return self._fallback_detection(frame)

    def _detect_easyocr(self, frame: np.ndarray, threshold: float) -> List[Tuple[int, int, int, int]]:
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._easyocr_reader.readtext(frame_rgb)
            boxes = []
            for (bbox, text, conf) in results:
                if conf >= threshold:
                    pts = np.array(bbox, dtype=np.int32)
                    x1, y1 = pts.min(axis=0)
                    x2, y2 = pts.max(axis=0)
                    boxes.append((int(x1), int(y1), int(x2), int(y2)))
            return boxes
        except Exception as e:
            logger.error(f"EasyOCR detection error: {e}")
            return self._fallback_detection(frame)

    def _fallback_detection(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """EI-1 percentile-based fallback for grey/mid-tone subtitles."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]
        p_bright = float(np.percentile(gray, 95))
        p_dark = float(np.percentile(gray, 5))
        median = float(np.median(gray))
        bright_thresh = int(max(median + 20, min(245, p_bright - 1)))
        dark_thresh = int(min(median - 20, max(10, p_dark + 1)))
        bright_thresh = max(0, min(255, bright_thresh))
        dark_thresh = max(0, min(255, dark_thresh))
        _, thresh_bright = cv2.threshold(gray, bright_thresh, 255, cv2.THRESH_BINARY)
        _, thresh_dark = cv2.threshold(gray, dark_thresh, 255, cv2.THRESH_BINARY_INV)
        combined = cv2.bitwise_or(thresh_bright, thresh_dark)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, w // 40), max(1, h // 80)))
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_boxes = []
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            in_subtitle_zone = (y > h * 0.6) or (y + ch < h * 0.15)
            if in_subtitle_zone and cw > w * 0.08 and ch < h * 0.15 and ch > 4:
                raw_boxes.append((x, y, x + cw, y + ch))
        return self._merge_boxes(raw_boxes, margin=10)

    @staticmethod
    def _merge_boxes(boxes: List[Tuple[int, int, int, int]],
                     margin: int = 10) -> List[Tuple[int, int, int, int]]:
        if not boxes:
            return []
        expanded = [(x1 - margin, y1 - margin, x2 + margin, y2 + margin)
                    for x1, y1, x2, y2 in boxes]
        merged = list(expanded)
        changed = True
        while changed:
            changed = False
            new_merged = []
            used = set()
            for i in range(len(merged)):
                if i in used:
                    continue
                ax1, ay1, ax2, ay2 = merged[i]
                for j in range(i + 1, len(merged)):
                    if j in used:
                        continue
                    bx1, by1, bx2, by2 = merged[j]
                    if ax1 <= bx2 and ax2 >= bx1 and ay1 <= by2 and ay2 >= by1:
                        ax1 = min(ax1, bx1)
                        ay1 = min(ay1, by1)
                        ax2 = max(ax2, bx2)
                        ay2 = max(ay2, by2)
                        used.add(j)
                        changed = True
                new_merged.append((ax1, ay1, ax2, ay2))
                used.add(i)
            merged = new_merged
        result = []
        for x1, y1, x2, y2 in merged:
            ux1 = max(0, x1 + margin)
            uy1 = max(0, y1 + margin)
            ux2 = x2 - margin
            uy2 = y2 - margin
            if ux2 > ux1 and uy2 > uy1:
                result.append((ux1, uy1, ux2, uy2))
        return result
