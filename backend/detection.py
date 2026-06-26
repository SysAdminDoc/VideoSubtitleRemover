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
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.import_safety import module_can_import as _module_can_import

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


def windows_ml_python_decision(run_smoke: bool = False) -> Dict[str, object]:
    """Return the guarded Windows ML probe decision for diagnostics."""
    try:
        from backend.onnx_model_info import collect_windows_ml_probe
        return collect_windows_ml_probe(run_smoke=run_smoke)
    except Exception as exc:
        return {
            "schema": "vsr.windows_ml_probe.v1",
            "decision": "blocked",
            "reason": f"Windows ML probe failed: {exc}",
        }


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


def _rapidocr_engine_preference() -> str:
    return os.environ.get("VSR_RAPIDOCR_ENGINE", "auto").strip().lower()


def _openvino_runtime_available() -> bool:
    return _module_can_import(
        "openvino",
        logger=logger,
        failure_context="RapidOCR OpenVINO engine probe skipped",
    )


def _rapidocr_openvino_params(rapid_module, device: str) -> Optional[Dict[str, Any]]:
    preference = _rapidocr_engine_preference()
    forced = preference in {"openvino", "ov"}
    if preference in {"onnx", "onnxruntime", "cpu"}:
        return None
    if "cuda" in str(device).lower() and not forced:
        return None
    if not forced and str(device).lower() not in {"cpu", "directml"}:
        return None
    engine_type = getattr(rapid_module, "EngineType", None)
    openvino_type = getattr(engine_type, "OPENVINO", None) if engine_type else None
    if openvino_type is None:
        if forced:
            logger.info(
                "RapidOCR OpenVINO requested, but the installed RapidOCR "
                "package does not expose EngineType.OPENVINO; using ONNX "
                "Runtime instead."
            )
        return None
    if not _openvino_runtime_available():
        if forced:
            logger.info(
                "RapidOCR OpenVINO requested, but openvino is not installed; "
                "using ONNX Runtime instead."
            )
        return None
    return {
        "Det.engine_type": openvino_type,
        "Cls.engine_type": openvino_type,
        "Rec.engine_type": openvino_type,
    }


def _build_rapidocr(rapid_cls, device: str, rapid_module=None):
    openvino_params = (
        _rapidocr_openvino_params(rapid_module, device)
        if rapid_module is not None else None
    )
    if openvino_params:
        try:
            return rapid_cls(params=openvino_params), "OpenVINO"
        except Exception as exc:
            logger.warning(
                "RapidOCR OpenVINO engine init failed; retrying ONNX "
                f"Runtime provider: {exc}"
            )
    directml_params = _rapidocr_directml_params(device)
    if directml_params:
        try:
            return rapid_cls(params=directml_params), "DirectML"
        except Exception as exc:
            logger.warning(
                "RapidOCR DirectML provider init failed; retrying CPU "
                f"provider: {exc}"
            )
    try:
        return rapid_cls(), "CPU"
    except TypeError:
        return rapid_cls(params={}), "CPU"


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
            if _module_can_import("rapidocr"):
                import rapidocr as _rapidocr_module
                rapid_obj, rapid_provider = _build_rapidocr(
                    _rapidocr_module.RapidOCR,
                    self.device,
                    _rapidocr_module,
                )
            elif _module_can_import("rapidocr_onnxruntime"):
                import rapidocr_onnxruntime as _rapidocr_module
                rapid_obj, rapid_provider = _build_rapidocr(
                    _rapidocr_module.RapidOCR,
                    self.device,
                    _rapidocr_module,
                )
            else:
                raise ImportError("RapidOCR unavailable or failed import probe")
            if rapid_obj is not None:
                self._rapid_model = rapid_obj
                self._engine_name = {
                    "DirectML": "RapidOCR (DirectML)",
                    "OpenVINO": "RapidOCR (OpenVINO)",
                }.get(rapid_provider, "RapidOCR")
                if rapid_provider == "OpenVINO":
                    logger.info(
                        f"RapidOCR loaded via OpenVINO engine (lang={self.lang})"
                    )
                else:
                    logger.info(
                        f"RapidOCR loaded via ONNX Runtime {rapid_provider} "
                        f"provider (lang={self.lang})"
                    )
                return
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"RapidOCR init failed: {e}")

        # PaddleOCR PP-OCRv6/3.x (2.x compatibility handled by paddle_compat)
        try:
            from backend.paddle_compat import build_paddleocr
            self._paddle_model = build_paddleocr(self.lang, self.device)
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
        if not _module_can_import("easyocr"):
            self._engine_name = "OpenCV fallback"
            logger.warning(
                "EasyOCR is unavailable or failed its import probe; "
                "using OpenCV fallback detection"
            )
            return
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
                ox1 = max(0, w - ry2)
                oy1 = max(0, rx1)
                ox2 = min(w, w - ry1)
                oy2 = min(h, rx2)
                if ox2 > ox1 and oy2 > oy1:
                    out.append((ox1, oy1, ox2, oy2))
            return out
        return self._detect_axis_aligned(frame, threshold)

    def detect_with_confidence(
        self, frame: np.ndarray, threshold: float = 0.5
    ) -> List[Tuple[int, int, int, int, float]]:
        """Like detect(), but each result is (x1, y1, x2, y2, confidence)."""
        if self.vertical:
            h, w = frame.shape[:2]
            rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            rotated = self._detect_axis_aligned_conf(rotated, threshold)
            out: List[Tuple[int, int, int, int, float]] = []
            for (rx1, ry1, rx2, ry2, conf) in rotated:
                ox1 = max(0, w - ry2)
                oy1 = max(0, rx1)
                ox2 = min(w, w - ry1)
                oy2 = min(h, rx2)
                if ox2 > ox1 and oy2 > oy1:
                    out.append((ox1, oy1, ox2, oy2, conf))
            return out
        return self._detect_axis_aligned_conf(frame, threshold)

    def _detect_axis_aligned_conf(
        self, frame: np.ndarray, threshold: float
    ) -> List[Tuple[int, int, int, int, float]]:
        if self._rapid_model is not None:
            return self._detect_rapid_conf(frame, threshold)
        boxes = self._detect_axis_aligned(frame, threshold)
        return [(x1, y1, x2, y2, 1.0) for (x1, y1, x2, y2) in boxes]

    def _detect_rapid_conf(
        self, frame: np.ndarray, threshold: float
    ) -> List[Tuple[int, int, int, int, float]]:
        try:
            output = self._rapid_model(frame)
            return self._rapid_output_to_boxes_conf(output, threshold)
        except Exception as e:
            logger.error(f"RapidOCR detection error: {e}")
            return [(x1, y1, x2, y2, 1.0)
                    for (x1, y1, x2, y2) in self._fallback_detection(frame)]

    @classmethod
    def _rapid_output_to_boxes_conf(
        cls, output, threshold: float
    ) -> List[Tuple[int, int, int, int, float]]:
        if output is None:
            return []
        results = output[0] if isinstance(output, tuple) and output else output
        if not results:
            return []
        boxes: List[Tuple[int, int, int, int, float]] = []
        polys = cls._rapid_field(
            results, "boxes", "dt_polys", "dt_boxes", "polys", "det_polys")
        if polys is not None:
            scores = cls._rapid_field(
                results, "scores", "rec_scores", "text_scores", "cls_scores")
            for index, poly in enumerate(polys):
                conf = cls._rapid_score_at(scores, index)
                if conf >= threshold:
                    parsed = cls._poly_to_box(poly)
                    if parsed is not None:
                        boxes.append(parsed + (conf,))
            return boxes
        for entry in results:
            parsed = cls._rapid_entry_to_box(entry, threshold)
            if parsed is not None:
                conf = 1.0
                if isinstance(entry, dict):
                    c = cls._rapid_field(
                        entry, "score", "confidence", "conf", "rec_score")
                    if c is not None:
                        try:
                            conf = float(c)
                        except (TypeError, ValueError):
                            pass
                elif hasattr(entry, '__len__') and len(entry) >= 3:
                    try:
                        conf = float(entry[2])
                    except (TypeError, ValueError):
                        pass
                boxes.append(parsed + (conf,))
        return boxes

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
            return self._rapid_output_to_boxes(output, threshold)
        except Exception as e:
            logger.error(f"RapidOCR detection error: {e}")
            return self._fallback_detection(frame)

    @classmethod
    def _rapid_output_to_boxes(cls, output, threshold: float) -> List[Tuple[int, int, int, int]]:
        if output is None:
            return []
        results = output[0] if isinstance(output, tuple) and output else output
        structured = cls._rapid_structured_boxes(results, threshold)
        if structured is not None:
            return structured
        if not results:
            return []
        boxes: List[Tuple[int, int, int, int]] = []
        for entry in results:
            parsed = cls._rapid_entry_to_box(entry, threshold)
            if parsed is not None:
                boxes.append(parsed)
        return boxes

    @classmethod
    def _rapid_structured_boxes(cls, results, threshold: float) -> Optional[List[Tuple[int, int, int, int]]]:
        polys = cls._rapid_field(
            results,
            "boxes", "dt_polys", "dt_boxes", "polys", "det_polys",
        )
        if polys is None:
            return None
        scores = cls._rapid_field(
            results,
            "scores", "rec_scores", "text_scores", "cls_scores",
        )
        boxes: List[Tuple[int, int, int, int]] = []
        for index, poly in enumerate(polys):
            conf = cls._rapid_score_at(scores, index)
            if conf >= threshold:
                parsed = cls._poly_to_box(poly)
                if parsed is not None:
                    boxes.append(parsed)
        return boxes

    @staticmethod
    def _rapid_field(results, *names):
        if isinstance(results, dict):
            for name in names:
                if name in results:
                    return results.get(name)
            return None
        for name in names:
            value = getattr(results, name, None)
            if value is not None:
                return value
        return None

    @classmethod
    def _rapid_entry_to_box(cls, entry, threshold: float) -> Optional[Tuple[int, int, int, int]]:
        if isinstance(entry, dict):
            poly = cls._rapid_field(entry, "box", "bbox", "poly", "points", "dt_poly")
            conf = cls._rapid_score_at(
                [cls._rapid_field(entry, "score", "confidence", "conf", "rec_score")],
                0,
            )
        else:
            try:
                if len(entry) < 1:
                    return None
                poly = entry[0]
                conf = entry[2] if len(entry) >= 3 else 1.0
            except TypeError:
                poly = cls._rapid_field(entry, "box", "bbox", "poly", "points", "dt_poly")
                conf = cls._rapid_field(entry, "score", "confidence", "conf", "rec_score")
        if conf is None:
            conf = 1.0
        try:
            if float(conf) < threshold:
                return None
        except (TypeError, ValueError):
            return None
        return cls._poly_to_box(poly)

    @staticmethod
    def _rapid_score_at(scores, index: int) -> float:
        if scores is None:
            return 1.0
        try:
            score = scores[index]
        except (IndexError, KeyError, TypeError):
            return 1.0
        if isinstance(score, dict):
            score = score.get("score", score.get("confidence", score.get("conf")))
        elif isinstance(score, (list, tuple)) and score:
            score = score[-1]
        if score is None:
            return 1.0
        try:
            return float(score)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _poly_to_box(poly) -> Optional[Tuple[int, int, int, int]]:
        try:
            pts = np.array(poly, dtype=np.float32)
        except (TypeError, ValueError):
            return None
        if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 2:
            return None
        xy = pts[:, :2]
        x1, y1 = xy.min(axis=0)
        x2, y2 = xy.max(axis=0)
        if x2 <= x1 or y2 <= y1:
            return None
        return (int(x1), int(y1), int(x2), int(y2))

    def _detect_paddle(self, frame: np.ndarray, threshold: float) -> List[Tuple[int, int, int, int]]:
        try:
            from backend.paddle_compat import extract_paddle_boxes
            return extract_paddle_boxes(self._paddle_model, frame, threshold)
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


def _classify_script(text: str) -> str:
    """Classify the dominant script family of recognized text."""
    if not text:
        return "unknown"
    counts = {"cjk": 0, "latin": 0, "cyrillic": 0, "arabic": 0,
              "devanagari": 0, "thai": 0, "hangul": 0, "other": 0}
    for ch in text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
            counts["cjk"] += 1
        elif 0x3040 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF:
            counts["cjk"] += 1
        elif 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF:
            counts["hangul"] += 1
        elif 0x0041 <= cp <= 0x024F:
            counts["latin"] += 1
        elif 0x0400 <= cp <= 0x04FF:
            counts["cyrillic"] += 1
        elif 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F:
            counts["arabic"] += 1
        elif 0x0900 <= cp <= 0x097F:
            counts["devanagari"] += 1
        elif 0x0E00 <= cp <= 0x0E7F:
            counts["thai"] += 1
        elif not ch.isspace():
            counts["other"] += 1
    best = max(counts, key=counts.get)
    if counts[best] == 0:
        return "unknown"
    return best


_SCRIPT_TO_LANG = {
    "latin": "en",
    "cjk": "ch",
    "hangul": "ko",
    "cyrillic": "ru",
    "arabic": "ar",
    "devanagari": "hi",
    "thai": "th",
}


def probe_language(frame: np.ndarray,
                   region: Optional[Tuple[int, int, int, int]] = None,
                   device: str = "cpu") -> Tuple[str, float, str]:
    """Probe a frame (optionally cropped to region) and return
    (lang_code, confidence, script_name).

    Uses the default RapidOCR engine to recognize text in the region
    and classifies the dominant script to suggest a language code.
    Returns ("en", 0.0, "unknown") when no text is detected.
    """
    if region:
        x1, y1, x2, y2 = region
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return ("en", 0.0, "unknown")
        crop = frame[y1:y2, x1:x2]
    else:
        crop = frame

    texts = []
    confs = []
    try:
        try:
            from rapidocr import RapidOCR
        except ImportError:
            from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()
        output = ocr(crop)
        results = output[0] if isinstance(output, tuple) and output else output
        if results:
            for entry in (results if isinstance(results, list) else []):
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    text = str(entry[1]) if not isinstance(entry[1], str) else entry[1]
                    conf = float(entry[2]) if len(entry) >= 3 else 0.5
                    texts.append(text)
                    confs.append(conf)
                elif isinstance(entry, dict):
                    text = str(entry.get("text", entry.get("rec_text", "")))
                    conf = float(entry.get("score", entry.get("confidence", 0.5)))
                    texts.append(text)
                    confs.append(conf)
    except Exception as exc:
        logger.debug(f"Language probe OCR failed: {exc}")
        return ("en", 0.0, "unknown")

    if not texts:
        return ("en", 0.0, "unknown")

    combined = " ".join(texts)
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    script = _classify_script(combined)
    lang = _SCRIPT_TO_LANG.get(script, "en")
    return (lang, avg_conf, script)
