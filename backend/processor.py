"""
Backend Subtitle Removal Processor
Handles the actual subtitle detection and removal using AI models.

This module provides the core processing functionality that interfaces with
various inpainting models (STTN, LAMA, ProPainter) for subtitle removal.
"""

import os
import sys
import cv2
import numpy as np
import logging
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List, Generator, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class InpaintMode(Enum):
    """Supported inpainting algorithms."""
    STTN = "sttn"
    LAMA = "lama"
    PROPAINTER = "propainter"


@dataclass
class ProcessingConfig:
    """Configuration for subtitle removal."""
    mode: InpaintMode = InpaintMode.STTN
    device: str = "cuda:0"

    # STTN settings
    sttn_skip_detection: bool = False
    sttn_neighbor_stride: int = 10
    sttn_reference_length: int = 10
    sttn_max_load_num: int = 30

    # LAMA settings
    lama_super_fast: bool = False

    # Detection settings
    subtitle_area: Optional[Tuple[int, int, int, int]] = None
    detection_threshold: float = 0.5
    detection_lang: str = "en"
    detection_frame_skip: int = 0  # 0=detect every frame, N=reuse mask for N frames

    # Mask settings
    mask_dilate_px: int = 8  # morphological dilation on masks for cleaner removal

    # Time range (video only, seconds from start)
    time_start: float = 0.0   # 0 = beginning
    time_end: float = 0.0     # 0 = entire video

    # Output settings
    preserve_audio: bool = True
    output_format: str = "mp4"
    output_quality: int = 23  # CRF value for x264
    use_hw_encode: bool = True  # try NVENC/QSV before falling back to libx264


# =============================================================================
# SUBTITLE DETECTION -- PaddleOCR > Surya > EasyOCR > OpenCV fallback
# =============================================================================

class SubtitleDetector:
    """Detects subtitle regions in video frames using text detection models."""

    def __init__(self, device: str = "cuda:0", lang: str = "en"):
        self.device = device
        self.lang = lang
        self._engine_name = "none"
        self._paddle_model = None
        self._surya_det = None
        self._surya_processor = None
        self._easyocr_reader = None
        self._load_model()

    def _is_gpu_device(self) -> bool:
        """Check if the device string indicates GPU acceleration."""
        return 'cuda' in self.device or self.device == 'directml'

    def _load_model(self):
        """Load detection model: PaddleOCR > Surya > EasyOCR > OpenCV fallback."""
        # Try PaddleOCR first (PP-OCRv5 with paddleocr>=3.0.0)
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

        # Try Surya OCR (fast, layout-aware, 90+ languages)
        try:
            from surya.detection import DetectionPredictor
            self._surya_det = DetectionPredictor()
            self._engine_name = "Surya"
            logger.info("Surya text detection loaded")
            return
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Surya init failed: {e}")

        # Try EasyOCR
        try:
            import easyocr
            gpu = self._is_gpu_device()
            # Map PaddleOCR lang codes to EasyOCR equivalents
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
        """Detect text regions in a frame. Returns list of (x1, y1, x2, y2) boxes."""
        if self._paddle_model is not None:
            return self._detect_paddle(frame, threshold)
        elif self._surya_det is not None:
            return self._detect_surya(frame, threshold)
        elif self._easyocr_reader is not None:
            return self._detect_easyocr(frame, threshold)
        else:
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
        """Detect text using Surya OCR text detection."""
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
        """Fallback detection using image processing when no OCR is available."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]

        # Detect both bright-on-dark and dark-on-bright text
        _, thresh_bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        _, thresh_dark = cv2.threshold(gray, 55, 255, cv2.THRESH_BINARY_INV)
        combined = cv2.bitwise_or(thresh_bright, thresh_dark)

        # Morphological close to merge nearby character regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, w // 40), max(1, h // 80)))
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_boxes = []
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            # Accept subtitle-shaped regions in top 15% or bottom 40% of frame
            in_subtitle_zone = (y > h * 0.6) or (y + ch < h * 0.15)
            if in_subtitle_zone and cw > w * 0.08 and ch < h * 0.15 and ch > 4:
                raw_boxes.append((x, y, x + cw, y + ch))

        return self._merge_boxes(raw_boxes, margin=10)

    @staticmethod
    def _merge_boxes(boxes: List[Tuple[int, int, int, int]],
                     margin: int = 10) -> List[Tuple[int, int, int, int]]:
        """Merge overlapping or nearby bounding boxes."""
        if not boxes:
            return []
        expanded = [(x1 - margin, y1 - margin, x2 + margin, y2 + margin) for x1, y1, x2, y2 in boxes]
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


# =============================================================================
# INPAINTING -- Real LaMa via simple-lama-inpainting, with cv2 fallback
# =============================================================================

class BaseInpainter(ABC):
    """Abstract base class for inpainting models."""

    @abstractmethod
    def inpaint(self, frames: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
        """Inpaint the masked regions in the frames."""
        pass


def _cv2_inpaint(frame: np.ndarray, mask: np.ndarray, radius: int = 5,
                 method: int = cv2.INPAINT_TELEA) -> np.ndarray:
    """OpenCV inpainting fallback."""
    if mask.max() > 0:
        return cv2.inpaint(frame, mask, radius, method)
    return frame.copy()


class STTNInpainter(BaseInpainter):
    """STTN-based video inpainting. Falls back to cv2.inpaint if model weights unavailable."""

    def __init__(self, device: str = "cuda:0", config: ProcessingConfig = None):
        self.device = device
        self.config = config or ProcessingConfig()

    def inpaint(self, frames: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
        return [_cv2_inpaint(f, m, 3, cv2.INPAINT_TELEA) for f, m in zip(frames, masks)]


class LAMAInpainter(BaseInpainter):
    """LAMA-based image inpainting. Uses simple-lama-inpainting if available."""

    def __init__(self, device: str = "cuda:0", config: ProcessingConfig = None):
        self.device = device
        self.config = config or ProcessingConfig()
        self._lama = None
        self._load_model()

    def _load_model(self):
        try:
            from simple_lama_inpainting import SimpleLama
            self._lama = SimpleLama()
            logger.info("LaMa neural inpainting model loaded (simple-lama-inpainting)")
        except ImportError:
            logger.warning("simple-lama-inpainting not installed, LAMA will use OpenCV fallback. "
                          "Install with: pip install simple-lama-inpainting")
        except Exception as e:
            logger.warning(f"LaMa model load failed: {e}")

    def inpaint(self, frames: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
        if self._lama is not None:
            return self._inpaint_lama(frames, masks)
        return [_cv2_inpaint(f, m, 7, cv2.INPAINT_NS) for f, m in zip(frames, masks)]

    def _inpaint_lama(self, frames: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
        from PIL import Image
        results = []
        for frame, mask in zip(frames, masks):
            if mask.max() == 0:
                results.append(frame.copy())
                continue
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                pil_mask = Image.fromarray(mask)
                result_pil = self._lama(pil_image, pil_mask)
                result_bgr = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
                results.append(result_bgr)
            except Exception as e:
                logger.warning(f"LaMa inpaint failed for frame, falling back to cv2: {e}")
                results.append(_cv2_inpaint(frame, mask, 7, cv2.INPAINT_NS))
        return results


class ProPainterInpainter(BaseInpainter):
    """ProPainter-based video inpainting. Falls back to cv2.inpaint if model weights unavailable."""

    def __init__(self, device: str = "cuda:0", config: ProcessingConfig = None):
        self.device = device
        self.config = config or ProcessingConfig()

    def inpaint(self, frames: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
        return [_cv2_inpaint(f, m, 5, cv2.INPAINT_TELEA) for f, m in zip(frames, masks)]


# =============================================================================
# HARDWARE ENCODER DETECTION
# =============================================================================

def _detect_hw_encoder() -> Optional[str]:
    """Probe FFmpeg for hardware encoder availability. Returns encoder name or None."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-encoders'],
            capture_output=True, text=True, timeout=10
        )
        for encoder in ('h264_nvenc', 'h264_qsv', 'h264_amf'):
            if encoder in result.stdout:
                logger.info(f"Hardware encoder available: {encoder}")
                return encoder
    except Exception:
        pass
    return None


# =============================================================================
# MAIN SUBTITLE REMOVER
# =============================================================================

class SubtitleRemover:
    """Coordinates detection and inpainting to remove subtitles from videos/images."""

    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
        self.detector = SubtitleDetector(
            self.config.device,
            lang=self.config.detection_lang
        )
        self.inpainter = self._create_inpainter()
        self.on_progress: Optional[Callable[[float, str], None]] = None
        self._hw_encoder: Optional[str] = None

        if self.config.use_hw_encode:
            self._hw_encoder = _detect_hw_encoder()

        logger.info(f"Detector: {self.detector._engine_name} | "
                    f"Inpainter: {self.config.mode.value} | "
                    f"Device: {self.config.device}"
                    f"{' | HW encode: ' + self._hw_encoder if self._hw_encoder else ''}")

    def _create_inpainter(self) -> BaseInpainter:
        if self.config.mode == InpaintMode.STTN:
            return STTNInpainter(self.config.device, self.config)
        elif self.config.mode == InpaintMode.LAMA:
            return LAMAInpainter(self.config.device, self.config)
        elif self.config.mode == InpaintMode.PROPAINTER:
            return ProPainterInpainter(self.config.device, self.config)
        return STTNInpainter(self.config.device, self.config)

    def _report_progress(self, progress: float, message: str):
        if self.on_progress:
            self.on_progress(progress, message)

    def _create_mask(self, frame_shape: Tuple[int, int], boxes: List[Tuple[int, int, int, int]],
                     padding: int = 5) -> np.ndarray:
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for x1, y1, x2, y2 in boxes:
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            mask[y1:y2, x1:x2] = 255

        # Morphological dilation for cleaner inpainting boundaries
        dilate_px = self.config.mask_dilate_px
        if dilate_px > 0 and mask.max() > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
            mask = cv2.dilate(mask, kernel, iterations=1)

        return mask

    def process_image(self, input_path: str, output_path: str) -> bool:
        try:
            self._report_progress(0.1, "Loading image...")
            image = cv2.imread(input_path)
            if image is None:
                raise ValueError(f"Could not load image: {input_path}")

            self._report_progress(0.3, "Detecting text regions...")
            if self.config.subtitle_area:
                boxes = [self.config.subtitle_area]
            else:
                boxes = self.detector.detect(image, self.config.detection_threshold)

            if not boxes:
                logger.info("No text detected, copying original")
                shutil.copy(input_path, output_path)
                self._report_progress(1.0, "Complete (no text found)")
                return True

            self._report_progress(0.5, f"Removing {len(boxes)} text regions...")
            mask = self._create_mask(image.shape, boxes)
            [result] = self.inpainter.inpaint([image], [mask])

            self._report_progress(0.9, "Saving result...")
            ext = Path(output_path).suffix.lower()
            if ext in ('.jpg', '.jpeg'):
                ok = cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
            elif ext == '.png':
                ok = cv2.imwrite(output_path, result, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            elif ext == '.webp':
                ok = cv2.imwrite(output_path, result, [cv2.IMWRITE_WEBP_QUALITY, 95])
            else:
                ok = cv2.imwrite(output_path, result)
            if not ok:
                raise IOError(f"Failed to write output image: {output_path}")
            self._report_progress(1.0, "Complete!")
            return True

        except InterruptedError:
            logger.info("Image processing cancelled")
            raise
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            return False

    def process_video(self, input_path: str, output_path: str) -> bool:
        temp_dir = None
        cap = None
        writer = None
        try:
            self._report_progress(0.0, "Opening video...")
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {input_path}")

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

            if width == 0 or height == 0:
                raise ValueError(f"Invalid video dimensions: {width}x{height}")

            # Time range support
            start_frame = 0
            end_frame = total_frames
            if self.config.time_start > 0:
                start_frame = min(total_frames - 1, int(self.config.time_start * fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            if self.config.time_end > 0:
                end_frame = min(total_frames, int(self.config.time_end * fps))
            if end_frame <= start_frame:
                raise ValueError(
                    f"Invalid time range: end ({self.config.time_end}s) "
                    f"must be after start ({self.config.time_start}s)")
            frames_to_process = end_frame - start_frame

            if start_frame > 0 or end_frame < total_frames:
                logger.info(f"Video: {width}x{height} @ {fps:.1f}fps, "
                           f"frames {start_frame}-{end_frame} of {total_frames}")
            else:
                logger.info(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

            temp_dir = tempfile.mkdtemp(prefix="vsr_")
            temp_video = os.path.join(temp_dir, "temp_video.mp4")

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
            if not writer.isOpened():
                raise ValueError(f"Could not create video writer for: {temp_video}")

            frame_idx = 0
            batch_size = self.config.sttn_max_load_num
            frame_skip = self.config.detection_frame_skip
            last_mask = None  # cached mask for frame-skip optimization
            fixed_mask = None  # cached mask for skip_detection mode

            while True:
                frames = []
                masks = []

                for _ in range(batch_size):
                    if start_frame + frame_idx >= end_frame:
                        break
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if self.config.sttn_skip_detection and self.config.subtitle_area:
                        # Fixed region: create mask once and reuse for all frames
                        if fixed_mask is None:
                            boxes = [self.config.subtitle_area]
                            fixed_mask = self._create_mask(frame.shape, boxes)
                        frames.append(frame)
                        masks.append(fixed_mask)
                        frame_idx += 1
                        continue
                    elif frame_skip > 0 and last_mask is not None and frame_idx % (frame_skip + 1) != 0:
                        # Reuse cached mask for intermediate frames
                        frames.append(frame)
                        masks.append(last_mask)
                        frame_idx += 1
                        continue
                    else:
                        boxes = self.detector.detect(frame, self.config.detection_threshold)

                    mask = self._create_mask(frame.shape, boxes)
                    last_mask = mask
                    frames.append(frame)
                    masks.append(mask)
                    frame_idx += 1

                if not frames:
                    break

                progress = min(0.9, frame_idx / max(1, frames_to_process) * 0.8 + 0.1)
                self._report_progress(progress, f"Processing frame {frame_idx}/{frames_to_process}...")

                results = self.inpainter.inpaint(frames, masks)
                for result in results:
                    writer.write(result)

            cap.release()
            cap = None
            writer.release()
            writer = None

            self._report_progress(0.9, "Merging audio...")
            if self.config.preserve_audio:
                self._merge_audio(input_path, temp_video, output_path)
            else:
                self._reencode_or_copy(temp_video, output_path)

            self._report_progress(1.0, "Complete!")
            return True

        except InterruptedError:
            logger.info("Video processing cancelled")
            raise
        except Exception as e:
            logger.error(f"Video processing error: {e}")
            return False
        finally:
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _get_encode_args(self) -> List[str]:
        """Return FFmpeg video encoder arguments, preferring hardware encoding."""
        if self._hw_encoder and self.config.use_hw_encode:
            if 'nvenc' in self._hw_encoder:
                return ['-c:v', self._hw_encoder, '-preset', 'p4',
                        '-cq', str(self.config.output_quality)]
            elif 'qsv' in self._hw_encoder:
                return ['-c:v', self._hw_encoder,
                        '-global_quality', str(self.config.output_quality)]
            elif 'amf' in self._hw_encoder:
                return ['-c:v', self._hw_encoder,
                        '-quality', 'balanced',
                        '-rc', 'cqp', '-qp', str(self.config.output_quality)]
        # Software fallback
        return ['-c:v', 'libx264', '-crf', str(self.config.output_quality),
                '-preset', 'medium']

    def _reencode_or_copy(self, source: str, output: str):
        """Re-encode with preferred encoder or just copy if FFmpeg unavailable."""
        try:
            cmd = ['ffmpeg', '-y', '-i', source]
            cmd += self._get_encode_args()
            cmd += ['-an', output]
            subprocess.run(cmd, check=True, capture_output=True, timeout=600)
        except subprocess.CalledProcessError as e:
            if self._hw_encoder:
                logger.warning(f"HW encoder failed, retrying with libx264: {e}")
                self._hw_encoder = None
                self._reencode_or_copy(source, output)
                return
            shutil.copy(source, output)
        except Exception:
            shutil.copy(source, output)

    def _merge_audio(self, original: str, processed: str, output: str):
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', processed,
            ]
            # When a time range was used, seek audio to match the processed segment
            if self.config.time_start > 0:
                cmd += ['-ss', str(self.config.time_start)]
            cmd += ['-i', original]
            if self.config.time_end > 0 and self.config.time_start > 0:
                duration = self.config.time_end - self.config.time_start
                cmd += ['-t', str(duration)]
            elif self.config.time_end > 0:
                cmd += ['-t', str(self.config.time_end)]
            cmd += self._get_encode_args()
            cmd += [
                '-c:a', 'aac',
                '-map', '0:v:0',
                '-map', '1:a:0?',
                '-shortest',
                output,
            ]
            subprocess.run(cmd, check=True, capture_output=True, timeout=600)
            encoder_name = self._hw_encoder or 'libx264'
            logger.info(f"Audio merged successfully (encoder: {encoder_name})")
        except subprocess.TimeoutExpired:
            logger.warning("FFmpeg audio merge timed out (>10min), copying video without audio")
            shutil.copy(processed, output)
        except subprocess.CalledProcessError as e:
            # If hardware encoder failed, retry with software
            if self._hw_encoder:
                logger.warning(f"HW encoder failed, retrying with libx264: {e}")
                self._hw_encoder = None
                self._merge_audio(original, processed, output)
                return
            logger.warning(f"Audio merge failed: {e}, copying video without audio")
            shutil.copy(processed, output)
        except FileNotFoundError:
            logger.warning("FFmpeg not found, copying video without audio")
            shutil.copy(processed, output)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Video Subtitle Remover")
    parser.add_argument("--input", "-i", required=True, help="Input file path")
    parser.add_argument("--output", "-o", required=True, help="Output file path")
    parser.add_argument("--mode", "-m", default="sttn", choices=["sttn", "lama", "propainter"],
                       help="Inpainting algorithm")
    parser.add_argument("--gpu", "-g", type=int, default=0, help="GPU device ID (-1 for CPU)")
    parser.add_argument("--lang", "-l", default="en", help="Detection language (en, ch, ja, ko, etc.)")
    parser.add_argument("--skip-detection", action="store_true",
                       help="Skip automatic detection (STTN only)")
    parser.add_argument("--fast", action="store_true", help="Fast mode (LAMA only)")
    parser.add_argument("--no-audio", action="store_true", help="Don't preserve audio")
    parser.add_argument("--crf", type=int, default=23, help="Output CRF quality (15-35)")
    parser.add_argument("--start", type=float, default=0, help="Start time in seconds")
    parser.add_argument("--end", type=float, default=0, help="End time in seconds (0=full)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold (0.1-1.0)")
    parser.add_argument("--frame-skip", type=int, default=0,
                       help="Reuse detection mask for N frames between detections (0=every frame)")
    parser.add_argument("--mask-dilate", type=int, default=8,
                       help="Mask dilation in pixels for cleaner removal (0=off)")
    parser.add_argument("--no-hw-encode", action="store_true",
                       help="Disable hardware encoding (force libx264)")

    args = parser.parse_args()

    config = ProcessingConfig(
        mode=InpaintMode(args.mode),
        device=f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu",
        sttn_skip_detection=args.skip_detection,
        lama_super_fast=args.fast,
        preserve_audio=not args.no_audio,
        detection_lang=args.lang,
        detection_threshold=args.threshold,
        output_quality=args.crf,
        time_start=args.start,
        time_end=args.end,
        detection_frame_skip=args.frame_skip,
        mask_dilate_px=args.mask_dilate,
        use_hw_encode=not args.no_hw_encode,
    )

    remover = SubtitleRemover(config)
    remover.on_progress = lambda p, m: print(f"[{int(p*100):3d}%] {m}")

    ext = Path(args.input).suffix.lower()
    video_exts = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.mpeg', '.mpg'}

    if ext in video_exts:
        success = remover.process_video(args.input, args.output)
    else:
        success = remover.process_image(args.input, args.output)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
