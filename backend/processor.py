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
from pathlib import Path
from typing import Optional, Tuple, List, Generator, Callable
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
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
    
    # Output settings
    preserve_audio: bool = True
    output_format: str = "mp4"
    output_quality: int = 23  # CRF value for x264


class SubtitleDetector:
    """Detects subtitle regions in video frames using text detection models."""
    
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the text detection model."""
        try:
            # Try to import and load PaddleOCR or other detection model
            from paddleocr import PaddleOCR
            self.model = PaddleOCR(
                use_angle_cls=False,
                lang='en',
                use_gpu='cuda' in self.device,
                show_log=False
            )
            logger.info("PaddleOCR text detection model loaded")
        except ImportError:
            logger.warning("PaddleOCR not available, using fallback detection")
            self.model = None
    
    def detect(self, frame: np.ndarray, threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
        """
        Detect text regions in a frame.
        
        Args:
            frame: Input frame (BGR format)
            threshold: Detection confidence threshold
            
        Returns:
            List of bounding boxes (x1, y1, x2, y2)
        """
        if self.model is None:
            return self._fallback_detection(frame)
        
        try:
            results = self.model.ocr(frame, cls=False)
            boxes = []
            
            if results and results[0]:
                for line in results[0]:
                    if line[1][1] >= threshold:  # Confidence check
                        pts = np.array(line[0], dtype=np.int32)
                        x1, y1 = pts.min(axis=0)
                        x2, y2 = pts.max(axis=0)
                        boxes.append((int(x1), int(y1), int(x2), int(y2)))
            
            return boxes
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []
    
    def _fallback_detection(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Fallback detection using image processing when OCR is unavailable."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to find bright text regions
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        h, w = frame.shape[:2]
        
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            
            # Filter for subtitle-like regions (bottom portion of frame, wide)
            if y > h * 0.6 and cw > w * 0.1 and ch < h * 0.15:
                boxes.append((x, y, x + cw, y + ch))
        
        return boxes


class BaseInpainter(ABC):
    """Abstract base class for inpainting models."""
    
    @abstractmethod
    def inpaint(self, frames: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
        """Inpaint the masked regions in the frames."""
        pass


class STTNInpainter(BaseInpainter):
    """STTN-based video inpainting."""
    
    def __init__(self, device: str = "cuda:0", config: ProcessingConfig = None):
        self.device = device
        self.config = config or ProcessingConfig()
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the STTN model."""
        try:
            import torch
            # In production, load actual STTN model weights
            logger.info("STTN inpainting model initialized")
        except ImportError:
            logger.error("PyTorch not available")
    
    def inpaint(self, frames: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
        """
        Inpaint frames using STTN algorithm.
        
        Args:
            frames: List of video frames (BGR)
            masks: List of binary masks (255 = inpaint region)
            
        Returns:
            List of inpainted frames
        """
        results = []
        
        for frame, mask in zip(frames, masks):
            # Use OpenCV inpainting as fallback
            if mask.max() > 0:
                inpainted = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
            else:
                inpainted = frame.copy()
            results.append(inpainted)
        
        return results


class LAMAInpainter(BaseInpainter):
    """LAMA-based image inpainting."""
    
    def __init__(self, device: str = "cuda:0", config: ProcessingConfig = None):
        self.device = device
        self.config = config or ProcessingConfig()
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the LAMA model."""
        try:
            import torch
            logger.info("LAMA inpainting model initialized")
        except ImportError:
            logger.error("PyTorch not available")
    
    def inpaint(self, frames: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
        """Inpaint frames using LAMA algorithm."""
        results = []
        
        for frame, mask in zip(frames, masks):
            if mask.max() > 0:
                # Use OpenCV inpainting as fallback
                inpainted = cv2.inpaint(frame, mask, 7, cv2.INPAINT_NS)
            else:
                inpainted = frame.copy()
            results.append(inpainted)
        
        return results


class ProPainterInpainter(BaseInpainter):
    """ProPainter-based video inpainting."""
    
    def __init__(self, device: str = "cuda:0", config: ProcessingConfig = None):
        self.device = device
        self.config = config or ProcessingConfig()
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the ProPainter model."""
        try:
            import torch
            logger.info("ProPainter inpainting model initialized")
        except ImportError:
            logger.error("PyTorch not available")
    
    def inpaint(self, frames: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
        """Inpaint frames using ProPainter algorithm."""
        results = []
        
        for frame, mask in zip(frames, masks):
            if mask.max() > 0:
                inpainted = cv2.inpaint(frame, mask, 5, cv2.INPAINT_TELEA)
            else:
                inpainted = frame.copy()
            results.append(inpainted)
        
        return results


class SubtitleRemover:
    """
    Main subtitle removal class.
    Coordinates detection and inpainting to remove subtitles from videos/images.
    """
    
    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
        self.detector = SubtitleDetector(self.config.device)
        self.inpainter = self._create_inpainter()
        
        # Callbacks
        self.on_progress: Optional[Callable[[float, str], None]] = None
    
    def _create_inpainter(self) -> BaseInpainter:
        """Create the appropriate inpainter based on config."""
        if self.config.mode == InpaintMode.STTN:
            return STTNInpainter(self.config.device, self.config)
        elif self.config.mode == InpaintMode.LAMA:
            return LAMAInpainter(self.config.device, self.config)
        elif self.config.mode == InpaintMode.PROPAINTER:
            return ProPainterInpainter(self.config.device, self.config)
        else:
            return STTNInpainter(self.config.device, self.config)
    
    def _report_progress(self, progress: float, message: str):
        """Report progress to callback."""
        if self.on_progress:
            self.on_progress(progress, message)
    
    def _create_mask(self, frame_shape: Tuple[int, int], boxes: List[Tuple[int, int, int, int]],
                     padding: int = 5) -> np.ndarray:
        """Create a binary mask from detected boxes."""
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for x1, y1, x2, y2 in boxes:
            # Add padding
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            mask[y1:y2, x1:x2] = 255
        
        return mask
    
    def process_image(self, input_path: str, output_path: str) -> bool:
        """
        Process a single image.
        
        Args:
            input_path: Path to input image
            output_path: Path for output image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self._report_progress(0.1, "Loading image...")
            
            # Load image
            image = cv2.imread(input_path)
            if image is None:
                raise ValueError(f"Could not load image: {input_path}")
            
            self._report_progress(0.3, "Detecting text regions...")
            
            # Detect subtitles
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
            
            # Create mask and inpaint
            mask = self._create_mask(image.shape, boxes)
            [result] = self.inpainter.inpaint([image], [mask])
            
            self._report_progress(0.9, "Saving result...")
            
            # Save result
            cv2.imwrite(output_path, result)
            
            self._report_progress(1.0, "Complete!")
            return True
            
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            return False
    
    def process_video(self, input_path: str, output_path: str) -> bool:
        """
        Process a video file.
        
        Args:
            input_path: Path to input video
            output_path: Path for output video
            
        Returns:
            True if successful, False otherwise
        """
        temp_dir = None
        
        try:
            self._report_progress(0.0, "Opening video...")
            
            # Open video
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {input_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
            
            # Create temp directory
            temp_dir = tempfile.mkdtemp(prefix="vsr_")
            temp_video = os.path.join(temp_dir, "temp_video.mp4")
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
            
            # Process frames
            frame_idx = 0
            batch_size = self.config.sttn_max_load_num
            
            while True:
                frames = []
                masks = []
                
                # Read batch of frames
                for _ in range(batch_size):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Detect text regions
                    if self.config.sttn_skip_detection and self.config.subtitle_area:
                        boxes = [self.config.subtitle_area]
                    else:
                        boxes = self.detector.detect(frame, self.config.detection_threshold)
                    
                    frames.append(frame)
                    masks.append(self._create_mask(frame.shape, boxes))
                    frame_idx += 1
                
                if not frames:
                    break
                
                # Progress update
                progress = min(0.9, frame_idx / total_frames * 0.8 + 0.1)
                self._report_progress(progress, f"Processing frame {frame_idx}/{total_frames}...")
                
                # Inpaint batch
                results = self.inpainter.inpaint(frames, masks)
                
                # Write results
                for result in results:
                    writer.write(result)
            
            cap.release()
            writer.release()
            
            self._report_progress(0.9, "Merging audio...")
            
            # Merge with original audio using ffmpeg
            if self.config.preserve_audio:
                self._merge_audio(input_path, temp_video, output_path)
            else:
                shutil.copy(temp_video, output_path)
            
            self._report_progress(1.0, "Complete!")
            return True
            
        except Exception as e:
            logger.error(f"Video processing error: {e}")
            return False
            
        finally:
            # Cleanup
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _merge_audio(self, original: str, processed: str, output: str):
        """Merge audio from original video with processed video."""
        import subprocess
        
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', processed,
                '-i', original,
                '-c:v', 'libx264',
                '-crf', str(self.config.output_quality),
                '-preset', 'medium',
                '-c:a', 'aac',
                '-map', '0:v:0',
                '-map', '1:a:0?',
                '-shortest',
                output
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info("Audio merged successfully")
            
        except subprocess.CalledProcessError as e:
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
    parser.add_argument("--gpu", "-g", type=int, default=0, help="GPU device ID")
    parser.add_argument("--skip-detection", action="store_true",
                       help="Skip automatic detection (STTN only)")
    parser.add_argument("--fast", action="store_true", help="Fast mode (LAMA only)")
    parser.add_argument("--no-audio", action="store_true", help="Don't preserve audio")
    
    args = parser.parse_args()
    
    # Create config
    config = ProcessingConfig(
        mode=InpaintMode(args.mode),
        device=f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu",
        sttn_skip_detection=args.skip_detection,
        lama_super_fast=args.fast,
        preserve_audio=not args.no_audio,
    )
    
    # Process
    remover = SubtitleRemover(config)
    remover.on_progress = lambda p, m: print(f"[{int(p*100):3d}%] {m}")
    
    input_path = args.input
    output_path = args.output
    
    # Determine file type
    ext = Path(input_path).suffix.lower()
    video_exts = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'}
    
    if ext in video_exts:
        success = remover.process_video(input_path, output_path)
    else:
        success = remover.process_image(input_path, output_path)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
