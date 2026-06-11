"""Video I/O, ffprobe helpers, capture wrappers, and atomic file ops.

Extracted from processor.py as part of RFP-L-1. Holds every helper that
touches the filesystem or shells out to ffmpeg/ffprobe but is not part
of the per-frame inpainting pipeline:

- ``_open_capture``: dispatch entry point used by ``SubtitleRemover``.
  Routes directories to ``_FrameSequenceCapture``, trusted ``.vpy`` input
  to the VapourSynth bridge when ``VSR_VAPOURSYNTH=1``, and NVIDIA users
  to PyNvVideoCodec when opted in.
- ``_FrameSequenceCapture`` / ``_PrefetchReader`` /
  ``_LosslessIntermediateWriter``: cv2.VideoCapture-shaped adapters.
- ffprobe helpers (codec banner, audio stream count, duration,
  keyframe set, interlace probe).
- Atomic output helpers (``_allocate_temp_output_path`` etc.).

The module imports ``backend.processor`` symbols lazily where needed
(e.g. the registry-import dance) to avoid a top-level cycle.
"""

from __future__ import annotations

import logging
import os
import queue
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ffprobe helpers
# ---------------------------------------------------------------------------


def _probe_codec_for_log(path: str) -> Optional[str]:
    """RM-74: read the video codec for the diagnostic log banner."""
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name,width,height,r_frame_rate',
            '-of', 'csv=p=0', path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _probe_audio_stream_count(path: str) -> int:
    """B-4: count audio streams via ffprobe; default to 1 on failure."""
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'a',
            '-show_entries', 'stream=index',
            '-of', 'csv=p=0', path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if result.returncode == 0:
            lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
            return max(1, len(lines))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 1


def _probe_duration_seconds(path: str) -> float:
    """F-6: container duration in seconds via ffprobe, 0.0 on failure."""
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if result.returncode == 0:
            return max(0.0, float(result.stdout.strip()))
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return 0.0


def _ffmpeg_subprocess_timeout(duration_seconds: float,
                                base: float = 180.0,
                                factor: float = 4.0,
                                cap: float = 24 * 3600.0) -> float:
    """Budget an ffmpeg subprocess timeout that scales with content length."""
    if duration_seconds <= 0:
        return base + 600.0
    return max(base, min(cap, base + duration_seconds * factor))


def _probe_keyframe_indices(video_path: str) -> Optional[set]:
    """v3.12: list decode-order I-frame indices via ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'frame=key_frame',
            '-of', 'csv=print_section=0', video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            if result.stderr:
                logger.debug(f"ffprobe keyframe scan stderr: {result.stderr.strip()[:400]}")
            return None
        keyframe_indices = set()
        frame_idx = 0
        for raw in result.stdout.splitlines():
            line = raw.strip()
            if not line:
                continue
            first = line.split(',', 1)[0].strip()
            if first in ('0', '1'):
                if first == '1':
                    keyframe_indices.add(frame_idx)
                frame_idx += 1
        return keyframe_indices if keyframe_indices else None
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        logger.warning("ffprobe keyframe scan timed out; falling back to pHash skip")
        return None
    except Exception as exc:
        logger.warning(f"ffprobe keyframe scan failed: {exc}")
        return None


def _probe_is_interlaced(video_path: str) -> bool:
    """v3.12: idet filter probe; True when majority of 200-frame sample
    reports interlaced content."""
    try:
        cmd = [
            'ffmpeg', '-hide_banner', '-nostats', '-i', video_path,
            '-vf', 'idet', '-frames:v', '200', '-an', '-f', 'null', '-',
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        stderr = result.stderr
        import re as _re
        m = _re.search(r'Multi frame detection:.*TFF:\s*(\d+).*BFF:\s*(\d+).*Progressive:\s*(\d+)',
                        stderr, _re.DOTALL)
        if m:
            tff, bff, prog = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return (tff + bff) > prog
    except Exception:
        pass
    return False


def _deinterlace_to_temp(src: str, temp_dir: str) -> str:
    """v3.12: run `ffmpeg -vf yadif` and return the temp progressive path."""
    dst = os.path.join(temp_dir, "deinterlaced.mp4")
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-nostats',
        '-i', src,
        '-vf', 'yadif=1',
        '-c:v', 'libx264', '-crf', '16', '-preset', 'veryfast',
        '-c:a', 'copy', dst,
    ]
    timeout = _ffmpeg_subprocess_timeout(_probe_duration_seconds(src))
    subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
    return dst


# ---------------------------------------------------------------------------
# Atomic file helpers
# ---------------------------------------------------------------------------


def _ensure_output_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _path_key(path) -> str:
    return str(Path(path).resolve(strict=False)).casefold()


def _choose_available_output_path(base_path: Path, reserved: Optional[set] = None) -> Path:
    """Avoid overwriting an existing file or a path reserved earlier in the batch."""
    reserved = reserved or set()
    candidate = base_path
    counter = 2
    while candidate.exists() or _path_key(candidate) in reserved:
        candidate = base_path.with_name(f"{base_path.stem}({counter}){base_path.suffix}")
        counter += 1
    return candidate


def _write_text_atomic(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
        )
        temp_path = Path(temp_name)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temp_path, path)
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def _allocate_temp_output_path(path) -> Path:
    """Create a sibling temp file path that preserves the final suffix."""
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{final_path.stem}.",
        suffix=final_path.suffix or ".tmp",
        dir=str(final_path.parent),
    )
    os.close(fd)
    return Path(temp_name)


def _cleanup_temp_output(path) -> None:
    if not path:
        return
    try:
        Path(path).unlink()
    except OSError:
        pass


def _promote_temp_output(temp_path, final_path) -> None:
    _ensure_output_parent(str(final_path))
    os.replace(Path(temp_path), Path(final_path))


def _copy_file_atomic(source: str, output: str) -> None:
    temp_output = _allocate_temp_output_path(output)
    try:
        shutil.copy2(source, temp_output)
        _promote_temp_output(temp_output, output)
    finally:
        _cleanup_temp_output(temp_output)


# ---------------------------------------------------------------------------
# Capture wrappers
# ---------------------------------------------------------------------------


class _FrameSequenceCapture:
    """cv2.VideoCapture-shaped adapter for a directory of images."""

    SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

    def __init__(self, dir_path: str, fps: float = 24.0):
        self._dir = Path(dir_path)
        if not self._dir.is_dir():
            raise ValueError(f"Not a directory: {dir_path}")
        self._files: List[Path] = sorted(
            p for p in self._dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTS
        )
        if not self._files:
            raise ValueError(
                f"No supported image files in {dir_path} "
                f"(expected one of {sorted(self.SUPPORTED_EXTS)})"
            )
        first = cv2.imread(str(self._files[0]))
        if first is None:
            raise ValueError(f"Could not read first frame: {self._files[0]}")
        self._h, self._w = first.shape[:2]
        self._fps = max(1.0, float(fps))
        self._pos = 0

    def isOpened(self) -> bool:
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._w)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._h)
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self._files))
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self._pos)
        return 0.0

    def set(self, prop, value) -> bool:
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self._pos = max(0, min(len(self._files), int(value)))
            return True
        return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._pos >= len(self._files):
            return False, None
        frame = cv2.imread(str(self._files[self._pos]))
        self._pos += 1
        if frame is None:
            return False, None
        if frame.shape[:2] != (self._h, self._w):
            frame = cv2.resize(
                frame, (self._w, self._h), interpolation=cv2.INTER_AREA
            )
        return True, frame

    def release(self) -> None:
        return None


def _open_capture(path: str, hw_accel: str = "off", *,
                  input_fps: float = 24.0):
    """Open a frame source. Directory -> ``_FrameSequenceCapture``,
    trusted ``.vpy`` -> VapourSynth bridge when ``VSR_VAPOURSYNTH=1``,
    VSR_PYNVVIDEOCODEC=1 -> PyNvVideoCodec, else cv2.VideoCapture
    (optionally HW-accelerated).
    """
    if Path(path).is_dir():
        logger.info(f"Frame-sequence input detected at {path} (fps={input_fps})")
        return _FrameSequenceCapture(path, fps=input_fps)
    if path.lower().endswith(".vpy"):
        try:
            from backend.vapoursynth_bridge import try_open_vpy
            cap = try_open_vpy(path)
            if cap is not None:
                logger.info(f"VapourSynth bridge active for {path}")
                return cap
        except Exception as exc:
            logger.debug(f"VapourSynth bridge failed: {exc}")
    if os.environ.get("VSR_PYNVVIDEOCODEC", "").strip().lower() in {"1", "true", "yes", "on"}:
        try:
            from backend.decode_accel import try_open_pynv
            pynv = try_open_pynv(path)
            if pynv is not None:
                return pynv
        except Exception as exc:
            logger.debug(f"PyNvVideoCodec probe failed: {exc}")
    if hw_accel in (None, "", "off"):
        return cv2.VideoCapture(path)
    accel_map = {
        "any": getattr(cv2, "VIDEO_ACCELERATION_ANY", 1),
        "auto": getattr(cv2, "VIDEO_ACCELERATION_ANY", 1),
        "d3d11": getattr(cv2, "VIDEO_ACCELERATION_D3D11", 2),
        "vaapi": getattr(cv2, "VIDEO_ACCELERATION_VAAPI", 3),
        "mfx": getattr(cv2, "VIDEO_ACCELERATION_MFX", 4),
    }
    accel_value = accel_map.get(hw_accel, accel_map["any"])
    try:
        cap = cv2.VideoCapture(
            path,
            cv2.CAP_FFMPEG,
            [cv2.CAP_PROP_HW_ACCELERATION, accel_value],
        )
        if cap.isOpened():
            ok, _frame = cap.read()
            if ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return cap
            cap.release()
            logger.warning(
                f"HW-accelerated decode '{hw_accel}' opened but returned no "
                f"frames (known cv2/FFmpeg issue); falling back to software."
            )
    except Exception as exc:
        logger.warning(
            f"HW-accelerated decode '{hw_accel}' raised {exc}; "
            f"falling back to software."
        )
    return cv2.VideoCapture(path)


class _PrefetchReader:
    """Background frame reader that wraps a cv2.VideoCapture so the
    detect+inpaint critical path overlaps with decode I/O. Ownership:
    once wrapped, the underlying capture is owned by the worker
    thread; the main thread must never call ``.set/.get/.read`` on it
    directly until ``.release()`` returns.
    """

    _STOP = object()

    def __init__(self, cap, *, max_frames: int, queue_size: int = 16):
        self._cap = cap
        self._max = max(0, int(max_frames))
        self._q: "queue.Queue" = queue.Queue(maxsize=max(2, int(queue_size)))
        self._stop = threading.Event()
        self._exhausted = False
        self._thread = threading.Thread(
            target=self._loop, name="vsr-prefetch", daemon=True,
        )
        self._thread.start()

    def _loop(self) -> None:
        try:
            for _ in range(self._max):
                if self._stop.is_set():
                    break
                ret, frame = self._cap.read()
                if not ret:
                    break
                while not self._stop.is_set():
                    try:
                        self._q.put((True, frame), timeout=0.25)
                        break
                    except queue.Full:
                        continue
        except Exception as exc:
            logger.warning(f"Prefetch reader crashed: {exc}")
        finally:
            try:
                self._q.put(self._STOP, timeout=1.0)
            except queue.Full:
                pass

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._exhausted:
            return False, None
        item = self._q.get()
        if item is self._STOP:
            self._exhausted = True
            return False, None
        return item

    def release(self) -> None:
        self._stop.set()
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        self._thread.join(timeout=2.0)
        try:
            self._cap.release()
        except Exception:
            pass

    def isOpened(self) -> bool:
        return self._cap.isOpened()

    def get(self, prop):
        return self._cap.get(prop)


class _LosslessIntermediateWriter:
    """Streams BGR frames through ``ffmpeg -c:v ffv1`` so the final
    encode is the only lossy step in the pipeline. Falls back to
    ``cv2.VideoWriter('mp4v')`` when ffmpeg is missing."""

    def __init__(self, path: str, width: int, height: int, fps: float):
        self._path = path
        self._width = int(width)
        self._height = int(height)
        try:
            fps_f = float(fps)
        except (TypeError, ValueError):
            fps_f = 30.0
        if not np.isfinite(fps_f) or fps_f <= 0.0:
            fps_f = 30.0
        self._fps = fps_f
        self._proc: Optional[subprocess.Popen] = None
        self._fallback: Optional[cv2.VideoWriter] = None
        self._opened = False
        self._lossless = False
        self._open()

    def _open(self):
        if shutil.which("ffmpeg") is None:
            self._open_fallback("ffmpeg not on PATH")
            return
        try:
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostats",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{self._width}x{self._height}",
                "-r", f"{self._fps:.6f}",
                "-i", "-",
                "-c:v", "ffv1", "-level", "3", "-coder", "1",
                "-context", "1", "-g", "1", "-slices", "16",
                "-slicecrc", "1",
                self._path,
            ]
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
            self._opened = True
            self._lossless = True
            logger.info(
                f"Intermediate writer: FFV1 lossless via ffmpeg stdin "
                f"({self._width}x{self._height} @ {self._fps:.2f} fps)"
            )
        except Exception as exc:
            self._open_fallback(f"ffmpeg Popen failed: {exc}")

    def _open_fallback(self, reason: str) -> None:
        logger.warning(
            f"Lossless intermediate unavailable ({reason}); "
            f"falling back to mp4v writer."
        )
        fallback_path = self._path
        if fallback_path.lower().endswith(".mkv"):
            fallback_path = fallback_path[:-4] + ".mp4"
            self._path = fallback_path
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._fallback = cv2.VideoWriter(
            fallback_path, fourcc, self._fps, (self._width, self._height)
        )
        self._opened = self._fallback.isOpened()
        self._lossless = False

    @property
    def path(self) -> str:
        return self._path

    @property
    def lossless(self) -> bool:
        return self._lossless

    def isOpened(self) -> bool:
        return self._opened

    def write(self, frame: np.ndarray) -> None:
        if frame is None:
            return
        if self._proc is not None and self._proc.stdin is not None:
            try:
                self._proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError) as exc:
                stderr_excerpt = ""
                if self._proc.stderr is not None:
                    try:
                        stderr_excerpt = self._proc.stderr.read().decode(
                            "utf-8", errors="replace")[-400:]
                    except Exception:
                        pass
                logger.error(
                    f"FFV1 ffmpeg stdin broke after writing frames: {exc}"
                    + (f"\nffmpeg stderr: {stderr_excerpt}" if stderr_excerpt else "")
                )
                raise
        elif self._fallback is not None:
            self._fallback.write(frame)

    def release(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.stdin is not None:
                    self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=300)
            except subprocess.TimeoutExpired:
                logger.warning("FFV1 ffmpeg flush timeout; killing")
                self._proc.kill()
                try:
                    self._proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
            if self._proc.stderr is not None:
                try:
                    self._proc.stderr.close()
                except Exception:
                    pass
            self._proc = None
        if self._fallback is not None:
            try:
                self._fallback.release()
            except Exception:
                pass
            self._fallback = None
