"""
Video Subtitle Remover Pro
A professional Windows application for AI-powered subtitle removal from videos and images.
Based on: https://github.com/YaoFANGUK/video-subtitle-remover

Author: SysAdminDoc
Version: 3.4.0
"""

import os
import sys
import json
import threading
import subprocess
import time
import logging
import logging.handlers
import traceback
from pathlib import Path
from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# =============================================================================
# LOGGING SETUP -- file + stream, crash handler
# =============================================================================

APP_NAME = "Video Subtitle Remover Pro"
APP_VERSION = "3.4.0"
APP_AUTHOR = "SysAdminDoc"

LOG_DIR = Path(os.environ.get("APPDATA", Path.home())) / "VideoSubtitleRemoverPro"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "vsr_pro.log"
SETTINGS_FILE = LOG_DIR / "settings.json"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=2, encoding='utf-8'),
    ]
)
logger = logging.getLogger(__name__)


def crash_handler(exc_type, exc_value, exc_tb):
    """Global crash handler -- log to file and show MessageBox."""
    msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
    logger.critical(f"UNHANDLED EXCEPTION:\n{msg}")
    try:
        import tkinter.messagebox as mb
        mb.showerror("Fatal Error",
                     f"{APP_NAME} crashed.\n\n{exc_value}\n\nLog: {LOG_FILE}")
    except Exception:
        pass
    sys.__excepthook__(exc_type, exc_value, exc_tb)


sys.excepthook = crash_handler

# GUI Imports
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from tkinter import font as tkfont
except ImportError:
    logger.error("Tkinter not found. Please install Python with Tkinter support.")
    sys.exit(1)

try:
    from PIL import Image, ImageTk, ImageDraw, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("Pillow not installed. Image preview will be limited.")

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

# Color Theme (Professional Dark Theme)
class Theme:
    # Primary colors
    BG_DARK = "#020617"           # Slate 950
    BG_SECONDARY = "#0f172a"      # Slate 900
    BG_TERTIARY = "#1e293b"       # Slate 800
    BG_CARD = "#0f172a"           # Card background
    BG_LOG = "#0a0f1a"            # Log panel background

    # Accent colors
    GREEN_PRIMARY = "#22c55e"      # Green 500
    GREEN_HOVER = "#16a34a"        # Green 600
    GREEN_MUTED = "#166534"        # Green 800

    BLUE_PRIMARY = "#60a5fa"       # Blue 400
    BLUE_HOVER = "#3b82f6"         # Blue 500
    BLUE_MUTED = "#1e40af"         # Blue 800

    # Text colors
    TEXT_PRIMARY = "#f8fafc"       # Slate 50
    TEXT_SECONDARY = "#94a3b8"     # Slate 400
    TEXT_MUTED = "#64748b"         # Slate 500
    TEXT_DISABLED = "#475569"      # Slate 600

    # Status colors
    SUCCESS = "#22c55e"
    WARNING = "#f59e0b"
    ERROR = "#ef4444"
    INFO = "#60a5fa"

    # Border colors
    BORDER = "#334155"             # Slate 700
    BORDER_FOCUS = "#60a5fa"

    # Progress colors
    PROGRESS_BG = "#1e293b"
    PROGRESS_FILL = "#22c55e"


class InpaintMode(Enum):
    STTN = "STTN"
    LAMA = "LAMA"
    PROPAINTER = "ProPainter"


class ProcessingStatus(Enum):
    IDLE = "idle"
    LOADING = "loading"
    DETECTING = "detecting"
    PROCESSING = "processing"
    MERGING = "merging"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class ProcessingConfig:
    """Configuration for subtitle removal processing."""
    mode: InpaintMode = InpaintMode.STTN
    use_gpu: bool = True
    gpu_id: int = 0

    # STTN settings
    sttn_skip_detection: bool = False
    sttn_neighbor_stride: int = 10
    sttn_reference_length: int = 10
    sttn_max_load_num: int = 30

    # LAMA settings
    lama_super_fast: bool = False

    # Region settings
    subtitle_area: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2

    # Detection settings
    detection_lang: str = "en"
    detection_threshold: float = 0.5

    # Time range (video only, seconds)
    time_start: float = 0.0
    time_end: float = 0.0

    # Output settings
    output_format: str = "mp4"
    preserve_audio: bool = True
    output_quality: int = 23  # CRF value (15-35, lower = better quality)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "use_gpu": self.use_gpu,
            "gpu_id": self.gpu_id,
            "sttn_skip_detection": self.sttn_skip_detection,
            "sttn_neighbor_stride": self.sttn_neighbor_stride,
            "sttn_reference_length": self.sttn_reference_length,
            "sttn_max_load_num": self.sttn_max_load_num,
            "lama_super_fast": self.lama_super_fast,
            "subtitle_area": list(self.subtitle_area) if self.subtitle_area else None,
            "detection_lang": self.detection_lang,
            "detection_threshold": self.detection_threshold,
            "output_format": self.output_format,
            "preserve_audio": self.preserve_audio,
            "output_quality": self.output_quality,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ProcessingConfig':
        mode_str = data.get("mode", "STTN")
        try:
            mode = InpaintMode(mode_str)
        except ValueError:
            mode = InpaintMode.STTN
        return cls(
            mode=mode,
            use_gpu=data.get("use_gpu", True),
            gpu_id=data.get("gpu_id", 0),
            sttn_skip_detection=data.get("sttn_skip_detection", False),
            sttn_neighbor_stride=data.get("sttn_neighbor_stride", 10),
            sttn_reference_length=data.get("sttn_reference_length", 10),
            sttn_max_load_num=data.get("sttn_max_load_num", 30),
            lama_super_fast=data.get("lama_super_fast", False),
            subtitle_area=tuple(data["subtitle_area"]) if data.get("subtitle_area") else None,
            detection_lang=data.get("detection_lang", "en"),
            detection_threshold=data.get("detection_threshold", 0.5),
            output_format=data.get("output_format", "mp4"),
            preserve_audio=data.get("preserve_audio", True),
            output_quality=data.get("output_quality", 23),
        )


@dataclass
class QueueItem:
    """Represents an item in the processing queue."""
    id: str
    file_path: str
    output_path: str
    config: ProcessingConfig
    status: ProcessingStatus = ProcessingStatus.IDLE
    progress: float = 0.0
    message: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


# =============================================================================
# SETTINGS PERSISTENCE
# =============================================================================

def load_settings() -> ProcessingConfig:
    """Load saved settings from disk."""
    try:
        if SETTINGS_FILE.exists():
            data = json.loads(SETTINGS_FILE.read_text(encoding='utf-8'))
            logger.info(f"Settings loaded from {SETTINGS_FILE}")
            return ProcessingConfig.from_dict(data)
    except Exception as e:
        logger.warning(f"Could not load settings: {e}")
    return ProcessingConfig()


def save_settings(config: ProcessingConfig):
    """Save settings to disk."""
    try:
        SETTINGS_FILE.write_text(json.dumps(config.to_dict(), indent=2), encoding='utf-8')
        logger.info(f"Settings saved to {SETTINGS_FILE}")
    except Exception as e:
        logger.warning(f"Could not save settings: {e}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_app_dir() -> Path:
    """Get the application directory."""
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent
    return Path(__file__).parent


def detect_gpu() -> List[dict]:
    """Detect available GPUs."""
    gpus = []

    # Try NVIDIA GPU detection
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 3:
                        try:
                            gpu_idx = int(parts[0].strip())
                            gpu_mem = f"{int(parts[2].strip())} MB"
                        except ValueError:
                            continue
                        gpus.append({
                            "index": gpu_idx,
                            "name": parts[1].strip(),
                            "memory": gpu_mem,
                            "type": "NVIDIA"
                        })
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # If no NVIDIA GPU, check for DirectML support
    if not gpus:
        try:
            import torch_directml
            gpus.append({
                "index": 0,
                "name": "DirectML Device",
                "memory": "Unknown",
                "type": "DirectML"
            })
        except ImportError:
            pass

    return gpus


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {int(s)}s"
    else:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)}h {int(m)}m"


def format_size(bytes_size: int) -> str:
    """Format bytes into human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f} PB"


def is_video_file(path: str) -> bool:
    """Check if file is a supported video format."""
    video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.mpeg', '.mpg'}
    return Path(path).suffix.lower() in video_extensions


def is_image_file(path: str) -> bool:
    """Check if file is a supported image format."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    return Path(path).suffix.lower() in image_extensions


def detect_ai_engines() -> dict:
    """Probe which AI engines are available."""
    engines = {"detection": [], "inpainting": []}
    try:
        import paddleocr
        engines["detection"].append("PaddleOCR")
    except ImportError:
        pass
    try:
        import easyocr
        engines["detection"].append("EasyOCR")
    except ImportError:
        pass
    if not engines["detection"]:
        engines["detection"].append("OpenCV fallback")
    try:
        from simple_lama_inpainting import SimpleLama
        engines["inpainting"].append("LaMa (neural)")
    except ImportError:
        pass
    engines["inpainting"].append("OpenCV")
    return engines


def get_file_info(path: str) -> str:
    """Get a short info string for a file (type + size)."""
    p = Path(path)
    try:
        size = format_size(p.stat().st_size)
    except OSError:
        size = "?"
    ext = p.suffix.lower()
    if is_video_file(path):
        return f"Video ({ext}) - {size}"
    elif is_image_file(path):
        return f"Image ({ext}) - {size}"
    return f"{ext} - {size}"


# =============================================================================
# CUSTOM WIDGETS
# =============================================================================

def _get_dpi_scale(root) -> float:
    """Get the DPI scaling factor relative to 96 DPI baseline."""
    try:
        return root.winfo_fpixels('1i') / 96.0
    except Exception:
        return 1.0


def _scaled(root, px: int) -> int:
    """Scale a pixel value by the current DPI factor."""
    return int(px * _get_dpi_scale(root))


class Tooltip:
    """Simple hover tooltip for any widget."""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self._tip = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")

    def _show(self, event):
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        # Truncate very long tooltip text
        display_text = self.text if len(self.text) <= 120 else self.text[:117] + "..."
        label = tk.Label(self._tip, text=display_text, font=("Segoe UI", 9),
                        bg=Theme.BG_TERTIARY, fg=Theme.TEXT_PRIMARY,
                        relief="solid", bd=1, padx=6, pady=3, wraplength=400)
        label.pack()
        self._tip.update_idletasks()
        # Position: clamp to screen bounds
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        sw = self._tip.winfo_screenwidth()
        sh = self._tip.winfo_screenheight()
        tw = self._tip.winfo_reqwidth()
        th = self._tip.winfo_reqheight()
        if x + tw > sw:
            x = sw - tw - 4
        if y + th > sh:
            y = self.widget.winfo_rooty() - th - 4
        self._tip.wm_geometry(f"+{x}+{y}")

    def _hide(self, event):
        if self._tip:
            self._tip.destroy()
            self._tip = None


class ModernButton(tk.Canvas):
    """A modern styled button with hover effects."""

    def __init__(self, parent, text="Button", command=None, width=120, height=36,
                 bg=Theme.GREEN_PRIMARY, hover_bg=Theme.GREEN_HOVER, fg=Theme.TEXT_PRIMARY,
                 corner_radius=8, font_size=10, style="primary", **kwargs):
        super().__init__(parent, width=width, height=height, highlightthickness=0,
                        bg=parent.cget('bg') if hasattr(parent, 'cget') else Theme.BG_DARK)

        self.text = text
        self.command = command
        self.width = width
        self.height = height
        self.corner_radius = corner_radius
        self.font_size = font_size
        self.enabled = True

        # Style-based colors
        if style == "primary":
            self.bg_color = Theme.GREEN_PRIMARY
            self.hover_color = Theme.GREEN_HOVER
            self.fg_color = "#ffffff"
        elif style == "secondary":
            self.bg_color = Theme.BG_TERTIARY
            self.hover_color = Theme.BORDER
            self.fg_color = Theme.TEXT_PRIMARY
        elif style == "accent":
            self.bg_color = Theme.BLUE_PRIMARY
            self.hover_color = Theme.BLUE_HOVER
            self.fg_color = "#ffffff"
        elif style == "danger":
            self.bg_color = Theme.ERROR
            self.hover_color = "#dc2626"
            self.fg_color = "#ffffff"
        else:
            self.bg_color = bg
            self.hover_color = hover_bg
            self.fg_color = fg

        self.current_bg = self.bg_color
        self._draw()

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)
        self.bind("<ButtonRelease-1>", self._on_release)

    def _draw(self):
        self.delete("all")

        # Draw rounded rectangle
        self._create_rounded_rect(2, 2, self.width - 2, self.height - 2,
                                  self.corner_radius, fill=self.current_bg)

        # Draw text
        text_color = self.fg_color if self.enabled else Theme.TEXT_DISABLED
        self.create_text(self.width // 2, self.height // 2, text=self.text,
                        fill=text_color, font=("Segoe UI", self.font_size, "bold"))

    def _create_rounded_rect(self, x1, y1, x2, y2, r, **kwargs):
        points = [
            x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
            x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
            x1, y2, x1, y2 - r, x1, y1 + r, x1, y1
        ]
        return self.create_polygon(points, smooth=True, **kwargs)

    def _on_enter(self, event):
        if self.enabled:
            self.current_bg = self.hover_color
            self._draw()
            self.config(cursor="hand2")

    def _on_leave(self, event):
        if self.enabled:
            self.current_bg = self.bg_color
            self._draw()
            self.config(cursor="")

    def _on_click(self, event):
        if self.enabled:
            self.current_bg = self.hover_color
            self._draw()

    def _on_release(self, event):
        if self.enabled and self.command:
            # Only fire if mouse is still inside the button
            if 0 <= event.x <= self.width and 0 <= event.y <= self.height:
                self.command()

    def set_enabled(self, enabled: bool):
        self.enabled = enabled
        self.current_bg = self.bg_color if enabled else Theme.BG_TERTIARY
        self._draw()

    def set_text(self, text: str):
        self.text = text
        self._draw()


class ModernProgressBar(tk.Canvas):
    """A modern styled progress bar."""

    def __init__(self, parent, width=400, height=8, bg=Theme.PROGRESS_BG,
                 fill=Theme.PROGRESS_FILL, corner_radius=4, **kwargs):
        super().__init__(parent, width=width, height=height, highlightthickness=0,
                        bg=parent.cget('bg') if hasattr(parent, 'cget') else Theme.BG_DARK)

        self.bar_width = width
        self.bar_height = height
        self.corner_radius = corner_radius
        self.bg_color = bg
        self.fill_color = fill
        self.progress = 0.0

        self._draw()

    def _draw(self):
        self.delete("all")
        r = self.corner_radius

        # Background
        self._create_rounded_rect(0, 0, self.bar_width, self.bar_height, r, fill=self.bg_color)

        # Progress fill
        if self.progress > 0:
            fill_width = max(r * 2, int(self.bar_width * self.progress))
            self._create_rounded_rect(0, 0, fill_width, self.bar_height, r, fill=self.fill_color)

    def _create_rounded_rect(self, x1, y1, x2, y2, r, **kwargs):
        points = [
            x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
            x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
            x1, y2, x1, y2 - r, x1, y1 + r, x1, y1
        ]
        return self.create_polygon(points, smooth=True, **kwargs)

    def set_progress(self, value: float):
        self.progress = max(0.0, min(1.0, value))
        self._draw()

    def set_color(self, color: str):
        self.fill_color = color
        self._draw()

    def resize(self, width: int, height: int = None):
        """Resize the progress bar (for DPI/layout changes)."""
        self.bar_width = width
        if height:
            self.bar_height = height
        self.config(width=self.bar_width, height=self.bar_height)
        self._draw()


class ModernEntry(tk.Frame):
    """A modern styled entry field."""

    def __init__(self, parent, width=300, placeholder="", **kwargs):
        super().__init__(parent, bg=Theme.BG_TERTIARY, highlightthickness=1,
                        highlightbackground=Theme.BORDER, highlightcolor=Theme.BORDER_FOCUS)

        self.placeholder = placeholder
        self.placeholder_active = True

        self.entry = tk.Entry(self, width=width // 10, bg=Theme.BG_TERTIARY,
                             fg=Theme.TEXT_MUTED, insertbackground=Theme.TEXT_PRIMARY,
                             font=("Segoe UI", 10), relief="flat", bd=8)
        self.entry.pack(fill="x", padx=2, pady=2)

        if placeholder:
            self.entry.insert(0, placeholder)
            self.entry.bind("<FocusIn>", self._on_focus_in)
            self.entry.bind("<FocusOut>", self._on_focus_out)

    def _on_focus_in(self, event):
        if self.placeholder_active:
            self.entry.delete(0, "end")
            self.entry.config(fg=Theme.TEXT_PRIMARY)
            self.placeholder_active = False

    def _on_focus_out(self, event):
        if not self.entry.get():
            self.entry.insert(0, self.placeholder)
            self.entry.config(fg=Theme.TEXT_MUTED)
            self.placeholder_active = True

    def get(self) -> str:
        if self.placeholder_active:
            return ""
        return self.entry.get()

    def set(self, value: str):
        self.entry.delete(0, "end")
        self.entry.insert(0, value)
        self.entry.config(fg=Theme.TEXT_PRIMARY)
        self.placeholder_active = False


class DragDropFrame(tk.Frame):
    """A frame that accepts drag and drop files."""

    def __init__(self, parent, on_drop: Callable[[List[str]], None],
                 width=400, height=200, **kwargs):
        super().__init__(parent, bg=Theme.BG_SECONDARY, highlightthickness=2,
                        highlightbackground=Theme.BORDER, highlightcolor=Theme.BLUE_PRIMARY)

        self.on_drop = on_drop
        self.configure(height=height)
        self.pack_propagate(False)
        self.grid_propagate(False)

        # Inner content
        inner = tk.Frame(self, bg=Theme.BG_SECONDARY)
        inner.place(relx=0.5, rely=0.5, anchor="center")

        # Icon (using text as fallback)
        icon_label = tk.Label(inner, text="+", font=("Segoe UI", 32, "bold"),
                             bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED)
        icon_label.pack(pady=(0, 6))

        # Main text
        main_text = tk.Label(inner, text="Drag & Drop Files Here",
                            font=("Segoe UI", 12, "bold"), bg=Theme.BG_SECONDARY,
                            fg=Theme.TEXT_PRIMARY)
        main_text.pack()

        # Sub text
        sub_text = tk.Label(inner, text="Click = files  |  Right-click = folder  |  Drag & drop both",
                           font=("Segoe UI", 9), bg=Theme.BG_SECONDARY,
                           fg=Theme.TEXT_MUTED)
        sub_text.pack(pady=(5, 0))

        # Bind click (left = files, right = folder)
        self.bind("<Button-1>", self._on_click)
        self.bind("<Button-3>", self._on_right_click)
        for child in self.winfo_children():
            child.bind("<Button-1>", self._on_click)
            child.bind("<Button-3>", self._on_right_click)
            for subchild in child.winfo_children():
                subchild.bind("<Button-1>", self._on_click)
                subchild.bind("<Button-3>", self._on_right_click)

        # Try to enable native drag-drop (Windows)
        try:
            self._setup_dnd()
        except Exception:
            pass

    def _setup_dnd(self):
        """Setup native drag and drop if available."""
        try:
            import tkinterdnd2
            self.drop_target_register(tkinterdnd2.DND_FILES)
            self.dnd_bind('<<Drop>>', self._handle_drop)
        except ImportError:
            pass

    def _handle_drop(self, event):
        files = self.tk.splitlist(event.data)
        # Accept both files and folders
        valid = [f for f in files if is_video_file(f) or is_image_file(f) or Path(f).is_dir()]
        if valid:
            self.on_drop(valid)

    def _on_click(self, event):
        filetypes = [
            ("All Supported", "*.mp4;*.avi;*.mkv;*.mov;*.wmv;*.flv;*.webm;*.jpg;*.jpeg;*.png;*.bmp"),
            ("Video Files", "*.mp4;*.avi;*.mkv;*.mov;*.wmv;*.flv;*.webm"),
            ("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff;*.webp"),
            ("All Files", "*.*")
        ]
        files = filedialog.askopenfilenames(
            title="Select Files to Process",
            filetypes=filetypes
        )
        if files:
            self.on_drop(list(files))

    def _on_right_click(self, event):
        folder = filedialog.askdirectory(title="Select Folder to Process")
        if folder:
            self.on_drop([folder])


class QueueItemWidget(tk.Frame):
    """Widget representing a single queue item."""

    def __init__(self, parent, item: QueueItem, on_remove: Callable,
                 on_select: Callable = None, **kwargs):
        super().__init__(parent, bg=Theme.BG_TERTIARY, highlightthickness=1,
                        highlightbackground=Theme.BORDER)

        self.item = item
        self.on_remove = on_remove
        self.on_select = on_select

        # Main container with padding
        container = tk.Frame(self, bg=Theme.BG_TERTIARY)
        container.pack(fill="x", padx=12, pady=10)

        # Top row: filename and status
        top_row = tk.Frame(container, bg=Theme.BG_TERTIARY)
        top_row.pack(fill="x")

        # Filename
        filename = Path(item.file_path).name
        if len(filename) > 40:
            filename = filename[:37] + "..."
        self.name_label = tk.Label(top_row, text=filename, font=("Segoe UI", 10, "bold"),
                                   bg=Theme.BG_TERTIARY, fg=Theme.TEXT_PRIMARY,
                                   cursor="hand2")
        self.name_label.pack(side="left")
        self.name_label.bind("<Button-1>", lambda e: self.on_select(self.item) if self.on_select else None)
        self.name_label.bind("<Double-Button-1>", lambda e: self._open_output())
        self.name_label.bind("<Button-3>", lambda e: self.on_select(self.item, show_mask=True) if self.on_select else None)
        Tooltip(self.name_label, item.file_path)

        # Status badge
        self.status_label = tk.Label(top_row, text=item.status.value.upper(),
                                     font=("Segoe UI", 8, "bold"), bg=Theme.BG_TERTIARY,
                                     fg=self._get_status_color())
        self.status_label.pack(side="right")

        # Remove button
        remove_btn = tk.Label(top_row, text="x", font=("Segoe UI", 10, "bold"),
                             bg=Theme.BG_TERTIARY, fg=Theme.TEXT_MUTED, cursor="hand2")
        remove_btn.pack(side="right", padx=(0, 10))
        remove_btn.bind("<Button-1>", lambda e: self.on_remove(self.item.id))
        remove_btn.bind("<Enter>", lambda e: remove_btn.config(fg=Theme.ERROR))
        remove_btn.bind("<Leave>", lambda e: remove_btn.config(fg=Theme.TEXT_MUTED))

        # File info row
        file_info = get_file_info(item.file_path)
        self.info_label = tk.Label(container, text=file_info, font=("Segoe UI", 8),
                                   bg=Theme.BG_TERTIARY, fg=Theme.TEXT_MUTED, anchor="w")
        self.info_label.pack(fill="x", pady=(2, 0))

        # Progress bar (resizes with container)
        self.progress_bar = ModernProgressBar(container, width=300, height=6,
                                              fill=self._get_status_color())
        self.progress_bar.pack(fill="x", pady=(6, 4))
        self.progress_bar.set_progress(item.progress)
        def _resize_bar(event):
            if event.width > 20:
                self.progress_bar.resize(event.width)
        container.bind("<Configure>", _resize_bar)

        # Bottom row: message + elapsed time
        bottom_row = tk.Frame(container, bg=Theme.BG_TERTIARY)
        bottom_row.pack(fill="x")

        self.message_label = tk.Label(bottom_row, text=item.message or "Waiting...",
                                      font=("Segoe UI", 9), bg=Theme.BG_TERTIARY,
                                      fg=Theme.TEXT_MUTED, anchor="w")
        self.message_label.pack(side="left", fill="x", expand=True)

        self.time_label = tk.Label(bottom_row, text="", font=("Segoe UI", 8),
                                   bg=Theme.BG_TERTIARY, fg=Theme.TEXT_MUTED, anchor="e")
        self.time_label.pack(side="right")

    def _open_output(self):
        """Open the output file if processing is complete."""
        if self.item.status == ProcessingStatus.COMPLETE and Path(self.item.output_path).exists():
            try:
                os.startfile(self.item.output_path)
            except Exception:
                pass

    def _get_status_color(self) -> str:
        status_colors = {
            ProcessingStatus.IDLE: Theme.TEXT_MUTED,
            ProcessingStatus.LOADING: Theme.BLUE_PRIMARY,
            ProcessingStatus.DETECTING: Theme.BLUE_PRIMARY,
            ProcessingStatus.PROCESSING: Theme.GREEN_PRIMARY,
            ProcessingStatus.MERGING: Theme.WARNING,
            ProcessingStatus.COMPLETE: Theme.SUCCESS,
            ProcessingStatus.ERROR: Theme.ERROR,
            ProcessingStatus.CANCELLED: Theme.TEXT_MUTED,
        }
        return status_colors.get(self.item.status, Theme.TEXT_MUTED)

    def update_item(self, item: QueueItem):
        self.item = item
        self.status_label.config(text=item.status.value.upper(), fg=self._get_status_color())
        self.progress_bar.set_progress(item.progress)
        self.progress_bar.set_color(self._get_status_color())
        self.message_label.config(text=item.message or "Waiting...")

        # Elapsed time
        elapsed_text = ""
        if item.started_at:
            end = item.completed_at or datetime.now()
            elapsed = (end - item.started_at).total_seconds()
            elapsed_text = format_time(elapsed)
        self.time_label.config(text=elapsed_text)


# =============================================================================
# LOG PANEL HANDLER -- routes log messages into a tk.Text widget
# =============================================================================

class TextWidgetHandler(logging.Handler):
    """Logging handler that writes to a tk.Text widget."""

    def __init__(self, text_widget: tk.Text):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record) + '\n'
        try:
            self.text_widget.after(0, self._append, msg, record.levelno)
        except Exception:
            pass

    def _append(self, msg, levelno):
        self.text_widget.config(state="normal")
        tag = "info"
        if levelno >= logging.ERROR:
            tag = "error"
        elif levelno >= logging.WARNING:
            tag = "warning"
        self.text_widget.insert("end", msg, tag)
        self.text_widget.see("end")
        self.text_widget.config(state="disabled")


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class VideoSubtitleRemoverApp:
    """Main application class."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(f"{APP_NAME} v{APP_VERSION}")
        self.root.geometry("1100x800")
        self.root.minsize(900, 650)
        self.root.configure(bg=Theme.BG_DARK)

        # Set window icon
        try:
            self.root.iconbitmap(get_app_dir() / "assets" / "icon.ico")
        except Exception:
            pass

        # State
        self.config = load_settings()
        self.queue: List[QueueItem] = []
        self.queue_widgets: dict = {}
        self.is_processing = False
        self.cancel_event = threading.Event()
        self.queue_lock = threading.Lock()
        self.gpus = detect_gpu()
        self.ai_engines = detect_ai_engines()
        self._elapsed_timer_id = None
        self._output_dir: Optional[Path] = None  # None = use input_dir/output/

        # Variables
        self.mode_var = tk.StringVar(value=self.config.mode.value)
        self.gpu_var = tk.StringVar()
        self.skip_detection_var = tk.BooleanVar(value=self.config.sttn_skip_detection)
        self.lama_fast_var = tk.BooleanVar(value=self.config.lama_super_fast)
        self.preserve_audio_var = tk.BooleanVar(value=self.config.preserve_audio)
        self.lang_var = tk.StringVar(value=self.config.detection_lang)

        # Build UI
        self._setup_styles()
        self._build_ui()

        # GPU setup
        if self.gpus:
            self.gpu_var.set(f"{self.gpus[0]['name']} ({self.gpus[0]['memory']})")
        else:
            self.gpu_var.set("CPU Mode")
            self.config.use_gpu = False

        # Attach log panel handler
        handler = TextWidgetHandler(self.log_text)
        handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                                                datefmt='%H:%M:%S'))
        logging.getLogger().addHandler(handler)

        # Restore subtitle_area label if saved
        if self.config.subtitle_area:
            x1, y1, x2, y2 = self.config.subtitle_area
            self.region_label.config(
                text=f"Subtitle Region: ({x1}, {y1}) to ({x2}, {y2})",
                fg=Theme.GREEN_PRIMARY)

        # Save settings on close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        """Stop processing, save settings, and close."""
        if self.is_processing:
            self.is_processing = False
            self.cancel_event.set()
            self._stop_elapsed_timer()
        self._sync_config_from_ui()
        save_settings(self.config)
        self.root.destroy()

    def _sync_config_from_ui(self):
        """Sync config object from current UI state."""
        try:
            self.config.mode = InpaintMode(self.mode_var.get())
        except ValueError:
            pass
        self.config.sttn_skip_detection = self.skip_detection_var.get()
        self.config.lama_super_fast = self.lama_fast_var.get()
        self.config.preserve_audio = self.preserve_audio_var.get()
        self.config.detection_lang = self.lang_var.get()
        # Threshold slider stores as int percent, convert to float
        pct = getattr(self.config, '_detection_threshold_pct', 50)
        self.config.detection_threshold = pct / 100.0
        # Time range
        try:
            self.config.time_start = float(self.time_start_entry.get() or 0)
        except ValueError:
            self.config.time_start = 0.0
        try:
            self.config.time_end = float(self.time_end_entry.get() or 0)
        except ValueError:
            self.config.time_end = 0.0
        # GPU sync
        selection = self.gpu_var.get()
        for gpu in self.gpus:
            if f"{gpu['name']} ({gpu['memory']})" == selection:
                self.config.gpu_id = gpu['index']
                break

    def _setup_styles(self):
        """Configure ttk styles."""
        style = ttk.Style()
        style.theme_use('clam')

        # Combobox style
        style.configure("Dark.TCombobox",
                       fieldbackground=Theme.BG_TERTIARY,
                       background=Theme.BG_TERTIARY,
                       foreground=Theme.TEXT_PRIMARY,
                       arrowcolor=Theme.TEXT_SECONDARY,
                       bordercolor=Theme.BORDER,
                       darkcolor=Theme.BG_TERTIARY,
                       lightcolor=Theme.BG_TERTIARY,
                       insertcolor=Theme.TEXT_PRIMARY)

        style.map("Dark.TCombobox",
                 fieldbackground=[('readonly', Theme.BG_TERTIARY)],
                 selectbackground=[('readonly', Theme.BLUE_MUTED)],
                 selectforeground=[('readonly', Theme.TEXT_PRIMARY)])

        # Theme the combobox dropdown popup listbox
        self.root.option_add('*TCombobox*Listbox.background', Theme.BG_TERTIARY)
        self.root.option_add('*TCombobox*Listbox.foreground', Theme.TEXT_PRIMARY)
        self.root.option_add('*TCombobox*Listbox.selectBackground', Theme.BLUE_MUTED)
        self.root.option_add('*TCombobox*Listbox.selectForeground', Theme.TEXT_PRIMARY)

        # Checkbutton style
        style.configure("Dark.TCheckbutton",
                       background=Theme.BG_SECONDARY,
                       foreground=Theme.TEXT_PRIMARY,
                       indicatorcolor=Theme.BG_TERTIARY,
                       indicatorbackground=Theme.BG_TERTIARY)

        style.map("Dark.TCheckbutton",
                 background=[('active', Theme.BG_SECONDARY)],
                 indicatorcolor=[('selected', Theme.GREEN_PRIMARY)])

        # Scrollbar style
        style.configure("Dark.Vertical.TScrollbar",
                        background=Theme.BG_TERTIARY,
                        troughcolor=Theme.BG_SECONDARY,
                        bordercolor=Theme.BG_SECONDARY,
                        arrowcolor=Theme.TEXT_MUTED)
        style.map("Dark.Vertical.TScrollbar",
                 background=[('active', Theme.BORDER)])

    def _build_ui(self):
        """Build the main user interface."""
        # Main container
        main_container = tk.Frame(self.root, bg=Theme.BG_DARK)
        main_container.pack(fill="both", expand=True, padx=20, pady=20)

        # Header
        self._build_header(main_container)

        # Content area (two columns)
        content = tk.Frame(main_container, bg=Theme.BG_DARK)
        content.pack(fill="both", expand=True, pady=(20, 0))

        # Use grid for proportional column sizing (left ~55%, right ~45%)
        content.columnconfigure(0, weight=55)
        content.columnconfigure(1, weight=45)
        content.rowconfigure(0, weight=1)

        # Left column - Input & Settings
        left_col = tk.Frame(content, bg=Theme.BG_DARK)
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        self._build_input_section(left_col)
        self._build_settings_section(left_col)

        # Right column - Queue & Preview
        right_col = tk.Frame(content, bg=Theme.BG_DARK)
        right_col.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        self._build_queue_section(right_col)

        # Log panel
        self._build_log_panel(main_container)

        # Footer
        self._build_footer(main_container)

    def _build_header(self, parent):
        """Build the header section."""
        header = tk.Frame(parent, bg=Theme.BG_DARK)
        header.pack(fill="x")

        # Top row: title + version
        top_row = tk.Frame(header, bg=Theme.BG_DARK)
        top_row.pack(fill="x")

        title = tk.Label(top_row, text="Video Subtitle Remover",
                        font=("Segoe UI", 20, "bold"), bg=Theme.BG_DARK,
                        fg=Theme.TEXT_PRIMARY)
        title.pack(side="left")

        pro_badge = tk.Label(top_row, text=" PRO", font=("Segoe UI", 10, "bold"),
                            bg=Theme.GREEN_PRIMARY, fg="#ffffff", padx=6, pady=1)
        pro_badge.pack(side="left", padx=(8, 0))

        version = tk.Label(top_row, text=f"v{APP_VERSION}",
                          font=("Segoe UI", 9), bg=Theme.BG_DARK,
                          fg=Theme.TEXT_MUTED)
        version.pack(side="left", padx=(8, 0))

        # GPU + engine status on the right (same row)
        if self.gpus:
            gpu_text = f"{self.gpus[0]['type']}: {self.gpus[0]['name']}"
            gpu_color = Theme.GREEN_PRIMARY
        else:
            gpu_text = "CPU Mode"
            gpu_color = Theme.WARNING

        det_names = ", ".join(self.ai_engines["detection"])
        inp_names = ", ".join(self.ai_engines["inpainting"])
        has_neural = "LaMa (neural)" in self.ai_engines["inpainting"]
        status_text = f"{gpu_text}  |  {det_names}  |  {inp_names}"

        tk.Label(top_row, text=status_text, font=("Segoe UI", 8),
                bg=Theme.BG_DARK,
                fg=Theme.GREEN_PRIMARY if has_neural else gpu_color).pack(side="right")

    def _build_input_section(self, parent):
        """Build the file input section."""
        section = tk.Frame(parent, bg=Theme.BG_SECONDARY, highlightthickness=1,
                          highlightbackground=Theme.BORDER)
        section.pack(fill="x", pady=(0, 15))

        # Section header
        header = tk.Frame(section, bg=Theme.BG_SECONDARY)
        header.pack(fill="x", padx=15, pady=(15, 10))

        tk.Label(header, text="INPUT FILES", font=("Segoe UI", 10, "bold"),
                bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY).pack(side="left")

        # Drag & drop area
        self.drop_area = DragDropFrame(section, self._on_files_dropped, height=130)
        self.drop_area.pack(fill="x", padx=15, pady=(0, 10))

        # Output directory row
        out_row = tk.Frame(section, bg=Theme.BG_SECONDARY)
        out_row.pack(fill="x", padx=15, pady=(0, 15))

        tk.Label(out_row, text="Output:", font=("Segoe UI", 9),
                bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED).pack(side="left")

        self.output_dir_label = tk.Label(out_row, text="Same as input / output /",
                                          font=("Segoe UI", 9), bg=Theme.BG_SECONDARY,
                                          fg=Theme.TEXT_SECONDARY, anchor="w")
        self.output_dir_label.pack(side="left", padx=(6, 0), fill="x", expand=True)

        reset_btn = tk.Label(out_row, text="Reset", font=("Segoe UI", 8),
                             bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED, cursor="hand2")
        reset_btn.pack(side="right", padx=(6, 0))
        reset_btn.bind("<Button-1>", lambda e: self._reset_output_dir())

        choose_btn = tk.Label(out_row, text="Browse", font=("Segoe UI", 8),
                              bg=Theme.BG_SECONDARY, fg=Theme.BLUE_PRIMARY, cursor="hand2")
        choose_btn.pack(side="right")
        choose_btn.bind("<Button-1>", lambda e: self._choose_output_dir())

    def _build_settings_section(self, parent):
        """Build the settings section."""
        section = tk.Frame(parent, bg=Theme.BG_SECONDARY, highlightthickness=1,
                          highlightbackground=Theme.BORDER)
        section.pack(fill="both", expand=True)

        # Section header
        header = tk.Frame(section, bg=Theme.BG_SECONDARY)
        header.pack(fill="x", padx=15, pady=(15, 10))

        tk.Label(header, text="PROCESSING SETTINGS", font=("Segoe UI", 10, "bold"),
                bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY).pack(side="left")

        # Settings container
        settings = tk.Frame(section, bg=Theme.BG_SECONDARY)
        settings.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        # Row 1: Algorithm selection
        row1 = tk.Frame(settings, bg=Theme.BG_SECONDARY)
        row1.pack(fill="x", pady=(0, 12))

        tk.Label(row1, text="Inpainting Algorithm", font=("Segoe UI", 10),
                bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY).pack(side="left")

        mode_combo = ttk.Combobox(row1, textvariable=self.mode_var, width=20,
                                 values=[m.value for m in InpaintMode],
                                 style="Dark.TCombobox", state="readonly")
        mode_combo.pack(side="right")
        mode_combo.bind("<<ComboboxSelected>>", self._on_mode_changed)

        # Algorithm description
        self.algo_desc = tk.Label(settings, text=self._get_algo_description(),
                                 font=("Segoe UI", 9), bg=Theme.BG_SECONDARY,
                                 fg=Theme.TEXT_MUTED, justify="left", anchor="w")
        self.algo_desc.pack(fill="x", pady=(0, 12))
        # Dynamic wraplength based on actual widget width
        def _update_wrap(event):
            self.algo_desc.config(wraplength=max(100, event.width - 20))
        self.algo_desc.bind("<Configure>", _update_wrap)

        # Row 2: GPU selection
        if self.gpus:
            row2 = tk.Frame(settings, bg=Theme.BG_SECONDARY)
            row2.pack(fill="x", pady=(0, 12))

            tk.Label(row2, text="GPU Device", font=("Segoe UI", 10),
                    bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY).pack(side="left")

            gpu_options = [f"{g['name']} ({g['memory']})" for g in self.gpus]
            gpu_combo = ttk.Combobox(row2, textvariable=self.gpu_var, width=30,
                                    values=gpu_options, style="Dark.TCombobox",
                                    state="readonly")
            gpu_combo.pack(side="right")
            gpu_combo.bind("<<ComboboxSelected>>", self._on_gpu_changed)

        # Checkboxes frame
        checks_frame = tk.Frame(settings, bg=Theme.BG_SECONDARY)
        checks_frame.pack(fill="x", pady=(0, 12))

        # Skip detection checkbox
        self.skip_check = tk.Checkbutton(checks_frame, text="Skip subtitle detection (faster, STTN only)",
                                        variable=self.skip_detection_var, font=("Segoe UI", 10),
                                        bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY,
                                        selectcolor=Theme.BG_TERTIARY, activebackground=Theme.BG_SECONDARY,
                                        activeforeground=Theme.TEXT_PRIMARY)
        self.skip_check.pack(anchor="w")
        Tooltip(self.skip_check, "Use a fixed subtitle region instead of per-frame detection. Requires Set Region.")

        # LAMA fast mode checkbox
        self.lama_check = tk.Checkbutton(checks_frame, text="LAMA Super Fast mode (lower quality)",
                                        variable=self.lama_fast_var, font=("Segoe UI", 10),
                                        bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY,
                                        selectcolor=Theme.BG_TERTIARY, activebackground=Theme.BG_SECONDARY,
                                        activeforeground=Theme.TEXT_PRIMARY)
        self.lama_check.pack(anchor="w")
        Tooltip(self.lama_check, "Faster but lower quality inpainting. LAMA mode only.")

        # Preserve audio checkbox
        tk.Checkbutton(checks_frame, text="Preserve original audio",
                      variable=self.preserve_audio_var, font=("Segoe UI", 10),
                      bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY,
                      selectcolor=Theme.BG_TERTIARY, activebackground=Theme.BG_SECONDARY,
                      activeforeground=Theme.TEXT_PRIMARY).pack(anchor="w")

        # Language & Region row
        lang_row = tk.Frame(settings, bg=Theme.BG_SECONDARY)
        lang_row.pack(fill="x", pady=(0, 12))

        tk.Label(lang_row, text="Detection Language", font=("Segoe UI", 10),
                bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY).pack(side="left")

        SUPPORTED_LANGS = ["en", "ch", "ja", "ko", "fr", "de", "es", "pt", "ru", "ar", "hi", "it"]
        lang_combo = ttk.Combobox(lang_row, textvariable=self.lang_var, width=8,
                                  values=SUPPORTED_LANGS, style="Dark.TCombobox", state="readonly")
        lang_combo.pack(side="right")

        # Subtitle region selector button
        region_row = tk.Frame(settings, bg=Theme.BG_SECONDARY)
        region_row.pack(fill="x", pady=(0, 8))

        self.region_label = tk.Label(region_row, text="Subtitle Region: Auto-detect",
                                     font=("Segoe UI", 9), bg=Theme.BG_SECONDARY,
                                     fg=Theme.TEXT_MUTED, anchor="w")
        self.region_label.pack(side="left", fill="x", expand=True)

        region_reset = tk.Label(region_row, text="Reset", font=("Segoe UI", 8),
                                bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED, cursor="hand2")
        region_reset.pack(side="right", padx=(6, 0))
        region_reset.bind("<Button-1>", lambda e: self._reset_region())

        region_btn = tk.Label(region_row, text="Set Region", font=("Segoe UI", 8),
                              bg=Theme.BG_SECONDARY, fg=Theme.BLUE_PRIMARY, cursor="hand2")
        region_btn.pack(side="right")
        region_btn.bind("<Button-1>", lambda e: self._open_region_selector())

        # Advanced settings toggle
        adv_frame = tk.Frame(settings, bg=Theme.BG_SECONDARY)
        adv_frame.pack(fill="x", pady=(10, 0))

        self.adv_visible = False
        self.adv_toggle = tk.Label(adv_frame, text="> Advanced Settings",
                                  font=("Segoe UI", 10), bg=Theme.BG_SECONDARY,
                                  fg=Theme.BLUE_PRIMARY, cursor="hand2")
        self.adv_toggle.pack(anchor="w")
        self.adv_toggle.bind("<Button-1>", self._toggle_advanced)

        # Advanced settings panel
        self.adv_panel = tk.Frame(settings, bg=Theme.BG_SECONDARY)

        # STTN settings
        sttn_frame = tk.LabelFrame(self.adv_panel, text="STTN Settings",
                                  font=("Segoe UI", 9, "bold"), bg=Theme.BG_SECONDARY,
                                  fg=Theme.TEXT_SECONDARY, bd=1,
                                  highlightbackground=Theme.BORDER, highlightcolor=Theme.BORDER)
        sttn_frame.pack(fill="x", pady=(10, 5))

        self._create_slider(sttn_frame, "Neighbor Stride", 5, 30,
                           self.config.sttn_neighbor_stride, "sttn_neighbor_stride")
        self._create_slider(sttn_frame, "Reference Length", 5, 30,
                           self.config.sttn_reference_length, "sttn_reference_length")
        self._create_slider(sttn_frame, "Max Load Frames", 10, 100,
                           self.config.sttn_max_load_num, "sttn_max_load_num")

        # Detection settings
        det_frame = tk.LabelFrame(self.adv_panel, text="Detection",
                                   font=("Segoe UI", 9, "bold"), bg=Theme.BG_SECONDARY,
                                   fg=Theme.TEXT_SECONDARY, bd=1,
                                  highlightbackground=Theme.BORDER, highlightcolor=Theme.BORDER)
        det_frame.pack(fill="x", pady=(5, 5))

        self._create_slider(det_frame, "Threshold", 10, 90,
                           int(self.config.detection_threshold * 100), "_detection_threshold_pct")
        Tooltip(det_frame, "Detection confidence 10-90%. Lower = more text found, higher = fewer false positives.")

        # Output quality settings
        quality_frame = tk.LabelFrame(self.adv_panel, text="Output Quality",
                                      font=("Segoe UI", 9, "bold"), bg=Theme.BG_SECONDARY,
                                      fg=Theme.TEXT_SECONDARY, bd=1,
                                  highlightbackground=Theme.BORDER, highlightcolor=Theme.BORDER)
        quality_frame.pack(fill="x", pady=(5, 5))

        self._create_slider(quality_frame, "CRF (lower=better)", 15, 35,
                           self.config.output_quality, "output_quality")

        # Video time range
        time_frame = tk.LabelFrame(self.adv_panel, text="Video Time Range",
                                    font=("Segoe UI", 9, "bold"), bg=Theme.BG_SECONDARY,
                                    fg=Theme.TEXT_SECONDARY, bd=1,
                                  highlightbackground=Theme.BORDER, highlightcolor=Theme.BORDER)
        time_frame.pack(fill="x", pady=(5, 5))

        time_inner = tk.Frame(time_frame, bg=Theme.BG_SECONDARY)
        time_inner.pack(fill="x", padx=10, pady=5)

        tk.Label(time_inner, text="Start (sec):", font=("Segoe UI", 9),
                bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY).pack(side="left")
        self.time_start_entry = tk.Entry(time_inner, width=6, bg=Theme.BG_TERTIARY,
                                          fg=Theme.TEXT_PRIMARY, font=("Segoe UI", 9),
                                          insertbackground=Theme.TEXT_PRIMARY, relief="flat", bd=4)
        self.time_start_entry.insert(0, "0")
        self.time_start_entry.pack(side="left", padx=(4, 12))

        tk.Label(time_inner, text="End (sec):", font=("Segoe UI", 9),
                bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY).pack(side="left")
        self.time_end_entry = tk.Entry(time_inner, width=6, bg=Theme.BG_TERTIARY,
                                        fg=Theme.TEXT_PRIMARY, font=("Segoe UI", 9),
                                        insertbackground=Theme.TEXT_PRIMARY, relief="flat", bd=4)
        self.time_end_entry.insert(0, "0")
        self.time_end_entry.pack(side="left", padx=(4, 0))

        tk.Label(time_inner, text="(0 = full)", font=("Segoe UI", 8),
                bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED).pack(side="left", padx=(8, 0))

        # Update checkbox states based on mode
        self._update_mode_options()

    def _create_slider(self, parent, label, min_val, max_val, default, attr_name):
        """Create a labeled slider."""
        frame = tk.Frame(parent, bg=Theme.BG_SECONDARY)
        frame.pack(fill="x", padx=10, pady=5)

        tk.Label(frame, text=label, font=("Segoe UI", 9),
                bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY, width=15,
                anchor="w").pack(side="left")

        value_label = tk.Label(frame, text=str(default), font=("Segoe UI", 9, "bold"),
                              bg=Theme.BG_SECONDARY, fg=Theme.GREEN_PRIMARY, width=4)
        value_label.pack(side="right")

        def update_value(val):
            int_val = int(float(val))
            value_label.config(text=str(int_val))
            setattr(self.config, attr_name, int_val)

        scale = tk.Scale(frame, from_=min_val, to=max_val, orient="horizontal",
                        bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY,
                        troughcolor=Theme.BG_TERTIARY, highlightthickness=0,
                        activebackground=Theme.GREEN_PRIMARY,
                        sliderrelief="flat", bd=0,
                        showvalue=False, command=update_value)
        scale.set(default)
        scale.pack(side="left", fill="x", expand=True, padx=(10, 10))

    def _toggle_advanced(self, event=None):
        """Toggle advanced settings visibility."""
        self.adv_visible = not self.adv_visible
        if self.adv_visible:
            self.adv_toggle.config(text="v Advanced Settings")
            self.adv_panel.pack(fill="x")
        else:
            self.adv_toggle.config(text="> Advanced Settings")
            self.adv_panel.pack_forget()

    def _build_queue_section(self, parent):
        """Build the processing queue section."""
        section = tk.Frame(parent, bg=Theme.BG_SECONDARY, highlightthickness=1,
                          highlightbackground=Theme.BORDER)
        section.pack(fill="both", expand=True)

        # Section header
        header = tk.Frame(section, bg=Theme.BG_SECONDARY)
        header.pack(fill="x", padx=15, pady=(15, 10))

        tk.Label(header, text="PROCESSING QUEUE", font=("Segoe UI", 10, "bold"),
                bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY).pack(side="left")

        self.queue_count = tk.Label(header, text="0 items",
                                   font=("Segoe UI", 9), bg=Theme.BG_SECONDARY,
                                   fg=Theme.TEXT_MUTED)
        self.queue_count.pack(side="right")

        # Overall batch progress bar
        batch_bar_frame = tk.Frame(section, bg=Theme.BG_SECONDARY)
        batch_bar_frame.pack(fill="x", padx=15, pady=(0, 8))

        self.batch_progress = ModernProgressBar(batch_bar_frame, width=300, height=4,
                                                 fill=Theme.BLUE_PRIMARY)
        self.batch_progress.pack(side="left", fill="x", expand=True)
        def _resize_batch(event):
            if event.width > 60:
                self.batch_progress.resize(event.width - 60)
        batch_bar_frame.bind("<Configure>", _resize_batch)

        self.batch_label = tk.Label(batch_bar_frame, text="", font=("Segoe UI", 8),
                                    bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED)
        self.batch_label.pack(side="right", padx=(8, 0))

        # Queue container with scrollbar
        queue_container = tk.Frame(section, bg=Theme.BG_SECONDARY)
        queue_container.pack(fill="both", expand=True, padx=15, pady=(0, 10))

        # Canvas for scrolling
        self.queue_canvas = tk.Canvas(queue_container, bg=Theme.BG_SECONDARY,
                                     highlightthickness=0)
        scrollbar = ttk.Scrollbar(queue_container, orient="vertical",
                                 command=self.queue_canvas.yview,
                                 style="Dark.Vertical.TScrollbar")

        self.queue_frame = tk.Frame(self.queue_canvas, bg=Theme.BG_SECONDARY)

        self.queue_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.queue_canvas.pack(side="left", fill="both", expand=True)

        self.queue_window = self.queue_canvas.create_window((0, 0), window=self.queue_frame,
                                                            anchor="nw")

        self.queue_frame.bind("<Configure>", self._on_queue_configure)
        self.queue_canvas.bind("<Configure>", self._on_canvas_configure)

        # Mousewheel scrolling
        self.queue_canvas.bind("<Enter>", self._bind_mousewheel)
        self.queue_canvas.bind("<Leave>", self._unbind_mousewheel)

        # Empty state
        self.empty_label = tk.Label(self.queue_frame, text="No files in queue\nDrag & drop or browse to add files",
                                   font=("Segoe UI", 10), bg=Theme.BG_SECONDARY,
                                   fg=Theme.TEXT_MUTED, justify="center")
        self.empty_label.pack(pady=40)

        # Preview area (shows thumbnail of selected item)
        self._preview_frame = tk.Frame(section, bg=Theme.BG_TERTIARY)
        self._preview_frame.pack(fill="x", padx=15, pady=(0, 8))
        self._preview_label = tk.Label(self._preview_frame, bg=Theme.BG_TERTIARY,
                                       text="Click filename to preview | Right-click for detection mask",
                                       font=("Segoe UI", 8), fg=Theme.TEXT_MUTED)
        self._preview_label.pack(pady=4)
        self._preview_photo = None  # prevent GC

        # Control buttons
        btn_frame = tk.Frame(section, bg=Theme.BG_SECONDARY)
        btn_frame.pack(fill="x", padx=15, pady=(0, 15))

        self.start_btn = ModernButton(btn_frame, text="Start Processing", width=140,
                                     height=36, command=self._start_processing,
                                     style="primary", font_size=9)
        self.start_btn.pack(side="left")

        self.open_output_btn = ModernButton(btn_frame, text="Output", width=70,
                                            height=36, command=self._open_output_folder,
                                            style="accent", font_size=9)
        self.open_output_btn.pack(side="left", padx=(6, 0))

        self.retry_btn = ModernButton(btn_frame, text="Retry", width=60,
                                      height=36, command=self._retry_failed,
                                      style="secondary", font_size=9)
        self.retry_btn.pack(side="right")

        self.clear_btn = ModernButton(btn_frame, text="Clear", width=60,
                                     height=36, command=self._clear_queue,
                                     style="secondary", font_size=9)
        self.clear_btn.pack(side="right", padx=(0, 6))

    def _bind_mousewheel(self, event):
        self.queue_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, event):
        self.queue_canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event):
        self.queue_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_queue_configure(self, event):
        self.queue_canvas.configure(scrollregion=self.queue_canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.queue_canvas.itemconfig(self.queue_window, width=event.width)

    def _build_log_panel(self, parent):
        """Build the embedded, collapsible log panel."""
        log_section = tk.Frame(parent, bg=Theme.BG_SECONDARY, highlightthickness=1,
                               highlightbackground=Theme.BORDER)
        log_section.pack(fill="x", pady=(15, 0))

        # Header with toggle + clear
        log_header = tk.Frame(log_section, bg=Theme.BG_SECONDARY)
        log_header.pack(fill="x", padx=15, pady=(10, 0))

        self._log_visible = True
        self._log_toggle_label = tk.Label(log_header, text="v LOG", font=("Segoe UI", 9, "bold"),
                                          bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY,
                                          cursor="hand2")
        self._log_toggle_label.pack(side="left")
        self._log_toggle_label.bind("<Button-1>", lambda e: self._toggle_log_panel())

        open_log_btn = tk.Label(log_header, text="Open Log File", font=("Segoe UI", 8),
                                bg=Theme.BG_SECONDARY, fg=Theme.BLUE_PRIMARY, cursor="hand2")
        open_log_btn.pack(side="right")
        open_log_btn.bind("<Button-1>", lambda e: os.startfile(str(LOG_FILE)) if LOG_FILE.exists() else None)

        clear_log_btn = tk.Label(log_header, text="Clear", font=("Segoe UI", 8),
                                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED, cursor="hand2")
        clear_log_btn.pack(side="right", padx=(0, 12))
        clear_log_btn.bind("<Button-1>", lambda e: self._clear_log())

        # Log body (collapsible)
        self._log_body = tk.Frame(log_section, bg=Theme.BG_LOG)
        self._log_body.pack(fill="x", padx=15, pady=(5, 10))

        self.log_text = tk.Text(self._log_body, height=5, bg=Theme.BG_LOG,
                                fg=Theme.TEXT_MUTED, font=("Consolas", 9),
                                relief="flat", bd=4, state="disabled",
                                wrap="word", insertbackground=Theme.TEXT_PRIMARY)
        log_scroll = ttk.Scrollbar(self._log_body, orient="vertical", command=self.log_text.yview,
                                   style="Dark.Vertical.TScrollbar")
        self.log_text.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side="right", fill="y")
        self.log_text.pack(side="left", fill="both", expand=True)

        # Tag colors
        self.log_text.tag_configure("info", foreground=Theme.TEXT_MUTED)
        self.log_text.tag_configure("warning", foreground=Theme.WARNING)
        self.log_text.tag_configure("error", foreground=Theme.ERROR)

    def _toggle_log_panel(self):
        """Toggle log panel visibility."""
        self._log_visible = not self._log_visible
        if self._log_visible:
            self._log_body.pack(fill="x", padx=15, pady=(5, 10))
            self._log_toggle_label.config(text="v LOG")
        else:
            self._log_body.pack_forget()
            self._log_toggle_label.config(text="> LOG")

    def _clear_log(self):
        """Clear the log panel."""
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

    def _build_footer(self, parent):
        """Build the footer section."""
        footer = tk.Frame(parent, bg=Theme.BG_DARK)
        footer.pack(fill="x", pady=(10, 0))

        # Status bar
        self.status_label = tk.Label(footer, text="Ready", font=("Segoe UI", 9),
                                    bg=Theme.BG_DARK, fg=Theme.TEXT_MUTED, anchor="w")
        self.status_label.pack(side="left")

        # Credits
        credits = tk.Label(footer, text=f"{APP_AUTHOR}  |  Based on YaoFANGUK/video-subtitle-remover",
                          font=("Segoe UI", 9), bg=Theme.BG_DARK, fg=Theme.TEXT_MUTED)
        credits.pack(side="right")

    def _get_algo_description(self) -> str:
        """Get description for current algorithm."""
        descriptions = {
            "STTN": "STTN: Best for real-world videos. Fast processing, supports skip detection mode.",
            "LAMA": "LAMA: Best quality for images and animations. Moderate speed, detailed inpainting.",
            "ProPainter": "ProPainter: Best for high-motion videos. Slow, high VRAM usage.",
        }
        return descriptions.get(self.mode_var.get(), "")

    def _on_mode_changed(self, event=None):
        """Handle algorithm mode change."""
        self.config.mode = InpaintMode(self.mode_var.get())
        self.algo_desc.config(text=self._get_algo_description())
        self._update_mode_options()

    def _update_mode_options(self):
        """Update checkbox states based on selected mode."""
        mode = self.mode_var.get()

        # Skip detection only for STTN
        if mode == "STTN":
            self.skip_check.config(state="normal")
        else:
            self.skip_detection_var.set(False)
            self.skip_check.config(state="disabled")

        # LAMA fast only for LAMA
        if mode == "LAMA":
            self.lama_check.config(state="normal")
        else:
            self.lama_fast_var.set(False)
            self.lama_check.config(state="disabled")

    def _on_gpu_changed(self, event=None):
        """Handle GPU device selection change."""
        selection = self.gpu_var.get()
        for i, gpu in enumerate(self.gpus):
            label = f"{gpu['name']} ({gpu['memory']})"
            if label == selection:
                self.config.gpu_id = gpu['index']
                self.config.use_gpu = True
                logger.info(f"GPU set to: {gpu['name']} (index {gpu['index']})")
                break

    def _choose_output_dir(self):
        """Let user pick a custom output directory."""
        d = filedialog.askdirectory(title="Select Output Directory")
        if d:
            self._output_dir = Path(d)
            display = str(self._output_dir)
            if len(display) > 50:
                display = "..." + display[-47:]
            self.output_dir_label.config(text=display, fg=Theme.GREEN_PRIMARY)
            logger.info(f"Output directory: {self._output_dir}")

    def _reset_output_dir(self):
        """Reset output directory to default (input_dir/output/)."""
        self._output_dir = None
        self.output_dir_label.config(text="Same as input / output /", fg=Theme.TEXT_SECONDARY)

    def _open_region_selector(self):
        """Open a window to draw a subtitle region rectangle on the first frame."""
        # Get first file from queue or ask user to pick one
        source_path = None
        for item in self.queue:
            source_path = item.file_path
            break

        if not source_path:
            source_path = filedialog.askopenfilename(
                title="Select a video/image to define subtitle region",
                filetypes=[("All Supported", "*.mp4;*.avi;*.mkv;*.mov;*.jpg;*.jpeg;*.png")]
            )
        if not source_path:
            return

        # Load first frame
        try:
            import cv2 as _cv2
            if is_video_file(source_path):
                cap = _cv2.VideoCapture(source_path)
                try:
                    ret, frame = cap.read()
                    if not ret:
                        logger.error("Could not read video frame for region selection")
                        return
                finally:
                    cap.release()
            else:
                frame = _cv2.imread(source_path)
                if frame is None:
                    logger.error("Could not read image for region selection")
                    return
        except Exception as e:
            logger.error(f"Region selector error: {e}")
            return

        if not PIL_AVAILABLE:
            self._update_status("Pillow required for region selector")
            return

        frame_rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
        orig_h, orig_w = frame_rgb.shape[:2]

        # Scale to fit screen (80% of screen size max)
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        max_w = min(800, int(screen_w * 0.8))
        max_h = min(500, int(screen_h * 0.7))
        scale = min(max_w / orig_w, max_h / orig_h, 1.0)
        disp_w, disp_h = int(orig_w * scale), int(orig_h * scale)

        img = Image.fromarray(frame_rgb).resize((disp_w, disp_h), Image.LANCZOS)

        # Create Toplevel window
        win = tk.Toplevel(self.root)
        win.title("Draw Subtitle Region (click and drag)")
        win.configure(bg=Theme.BG_DARK)
        win.resizable(False, False)
        win.geometry(f"{disp_w}x{disp_h + 40}")

        photo = ImageTk.PhotoImage(img)
        canvas = tk.Canvas(win, width=disp_w, height=disp_h, highlightthickness=0)
        canvas.pack()
        canvas.create_image(0, 0, anchor="nw", image=photo)
        canvas._photo = photo  # prevent GC

        rect_id = [None]
        start = [0, 0]

        def on_press(event):
            start[0], start[1] = event.x, event.y
            if rect_id[0]:
                canvas.delete(rect_id[0])
            rect_id[0] = canvas.create_rectangle(event.x, event.y, event.x, event.y,
                                                   outline="#22c55e", width=2)

        def on_drag(event):
            if rect_id[0]:
                canvas.coords(rect_id[0], start[0], start[1], event.x, event.y)

        def on_release(event):
            x1 = int(min(start[0], event.x) / scale)
            y1 = int(min(start[1], event.y) / scale)
            x2 = int(max(start[0], event.x) / scale)
            y2 = int(max(start[1], event.y) / scale)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)
            if (x2 - x1) > 10 and (y2 - y1) > 5:
                self.config.subtitle_area = (x1, y1, x2, y2)
                self.region_label.config(
                    text=f"Subtitle Region: ({x1}, {y1}) to ({x2}, {y2})",
                    fg=Theme.GREEN_PRIMARY)
                logger.info(f"Subtitle region set: ({x1}, {y1}, {x2}, {y2})")
            win.destroy()

        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)
        win.bind("<Escape>", lambda e: win.destroy())

        hint = tk.Label(win, text="Drag to select subtitle area  |  Escape to cancel",
                        font=("Segoe UI", 9), bg=Theme.BG_DARK, fg=Theme.TEXT_MUTED)
        hint.pack(pady=8)

        win.transient(self.root)
        win.grab_set()

    def _reset_region(self):
        """Reset subtitle region to auto-detect."""
        self.config.subtitle_area = None
        self.region_label.config(text="Subtitle Region: Auto-detect", fg=Theme.TEXT_MUTED)

    def _on_files_dropped(self, files: List[str]):
        """Handle dropped files."""
        for file_path in files:
            if Path(file_path).is_dir():
                self._add_folder_to_queue(file_path)
            else:
                self._add_to_queue(file_path)

    def _add_folder_to_queue(self, folder_path: str):
        """Recursively add all supported files from a folder."""
        folder = Path(folder_path)
        count = 0
        for f in sorted(folder.rglob("*")):
            if f.is_file() and (is_video_file(str(f)) or is_image_file(str(f))):
                self._add_to_queue(str(f))
                count += 1
        if count:
            logger.info(f"Added {count} files from folder: {folder.name}")
        else:
            logger.warning(f"No supported files found in: {folder_path}")

    def _add_to_queue(self, file_path: str):
        """Add a file to the processing queue."""
        # Check file exists and is valid
        if not Path(file_path).is_file():
            logger.warning(f"File not found: {file_path}")
            return
        if not (is_video_file(file_path) or is_image_file(file_path)):
            logger.warning(f"Unsupported file type: {file_path}")
            return

        # Queue size limit
        if len(self.queue) >= 500:
            logger.warning("Queue full (500 items max)")
            return

        # Prevent duplicate files in queue
        normalized = str(Path(file_path).resolve())
        with self.queue_lock:
            for existing in self.queue:
                if str(Path(existing.file_path).resolve()) == normalized:
                    logger.warning(f"Already in queue: {Path(file_path).name}")
                    return

        # Generate unique ID
        item_id = f"{int(time.time() * 1000)}_{len(self.queue)}"

        # Generate output path (auto-increment to avoid overwriting)
        # Dir creation deferred to processing time
        input_path = Path(file_path)
        output_dir = self._output_dir or (input_path.parent / "output")
        output_path = output_dir / f"{input_path.stem}_no_sub{input_path.suffix}"
        counter = 2
        while output_path.exists():
            output_path = output_dir / f"{input_path.stem}_no_sub({counter}){input_path.suffix}"
            counter += 1

        # Create config copy
        config = ProcessingConfig(
            mode=self.config.mode,
            use_gpu=self.config.use_gpu,
            gpu_id=self.config.gpu_id,
            sttn_skip_detection=self.skip_detection_var.get(),
            sttn_neighbor_stride=self.config.sttn_neighbor_stride,
            sttn_reference_length=self.config.sttn_reference_length,
            sttn_max_load_num=self.config.sttn_max_load_num,
            lama_super_fast=self.lama_fast_var.get(),
            preserve_audio=self.preserve_audio_var.get(),
            output_quality=self.config.output_quality,
            detection_lang=self.lang_var.get(),
            detection_threshold=getattr(self.config, '_detection_threshold_pct', 50) / 100.0,
            subtitle_area=self.config.subtitle_area,
            time_start=float(self.time_start_entry.get() or 0),
            time_end=float(self.time_end_entry.get() or 0),
        )

        # Create queue item
        item = QueueItem(
            id=item_id,
            file_path=file_path,
            output_path=str(output_path),
            config=config,
            message="Queued for processing"
        )

        with self.queue_lock:
            self.queue.append(item)
        self._update_queue_display()
        self._update_status(f"Added: {Path(file_path).name}")
        logger.info(f"Queued: {Path(file_path).name} ({get_file_info(file_path)})")

    def _remove_from_queue(self, item_id: str):
        """Remove an item from the queue."""
        with self.queue_lock:
            # Don't remove items that are currently being processed
            item = next((i for i in self.queue if i.id == item_id), None)
            if item and item.status in (ProcessingStatus.LOADING, ProcessingStatus.DETECTING,
                                         ProcessingStatus.PROCESSING, ProcessingStatus.MERGING):
                return
            self.queue = [i for i in self.queue if i.id != item_id]
        self._update_queue_display()

    def _clear_queue(self):
        """Clear all items from the queue."""
        if self.is_processing:
            self._update_status("Cannot clear queue while processing")
            return

        with self.queue_lock:
            self.queue.clear()
        self._update_queue_display()
        self._update_status("Queue cleared")

    def _update_queue_display(self):
        """Update the queue display. Only rebuilds widgets that changed."""
        with self.queue_lock:
            current_ids = {item.id for item in self.queue}

        # Remove widgets for items no longer in queue
        stale_ids = [wid for wid in self.queue_widgets if wid not in current_ids]
        for wid in stale_ids:
            self.queue_widgets[wid].destroy()
            del self.queue_widgets[wid]

        # Update count
        self.queue_count.config(text=f"{len(self.queue)} item{'s' if len(self.queue) != 1 else ''}")

        if not self.queue:
            # Clear any remaining children and show empty state
            for widget in self.queue_frame.winfo_children():
                widget.destroy()
            self.queue_widgets.clear()
            self.empty_label = tk.Label(self.queue_frame,
                                       text="No files in queue\nDrag & drop or browse to add files",
                                       font=("Segoe UI", 10), bg=Theme.BG_SECONDARY,
                                       fg=Theme.TEXT_MUTED, justify="center")
            self.empty_label.pack(pady=40)
        else:
            # Remove empty label if present
            for child in self.queue_frame.winfo_children():
                if child not in self.queue_widgets.values():
                    child.destroy()

            # Add widgets for new items only
            for item in self.queue:
                if item.id not in self.queue_widgets:
                    widget = QueueItemWidget(self.queue_frame, item, self._remove_from_queue,
                                             on_select=self._show_preview)
                    widget.pack(fill="x", pady=(0, 8))
                    self.queue_widgets[item.id] = widget

    def _update_status(self, message: str):
        """Update the status bar."""
        self.status_label.config(text=message)

    def _open_output_folder(self):
        """Open the output folder for the most recently completed item."""
        completed = [i for i in self.queue if i.status == ProcessingStatus.COMPLETE]
        if completed:
            output_dir = str(Path(completed[-1].output_path).parent)
            try:
                os.startfile(output_dir)
            except Exception:
                logger.warning(f"Could not open folder: {output_dir}")
        else:
            self._update_status("No completed items to open")

    def _show_preview(self, item: QueueItem, show_mask: bool = False):
        """Show thumbnail preview. Side-by-side before/after for completed items.
        If show_mask=True, run detection and overlay red boxes on the frame."""
        if not PIL_AVAILABLE:
            self._preview_label.config(text="Install Pillow for previews", image="")
            return

        try:
            import cv2 as _cv2

            def load_first_frame_raw(path):
                """Load first frame as BGR numpy array."""
                if is_image_file(path):
                    return _cv2.imread(path)
                elif is_video_file(path):
                    cap = _cv2.VideoCapture(path)
                    try:
                        ret, frame = cap.read()
                        return frame if ret else None
                    finally:
                        cap.release()
                return None

            def to_pil(bgr_frame):
                return Image.fromarray(_cv2.cvtColor(bgr_frame, _cv2.COLOR_BGR2RGB))

            raw_frame = load_first_frame_raw(item.file_path)
            if raw_frame is None:
                self._preview_label.config(text="Could not read file", image="")
                return

            try:
                max_w = max(200, self._preview_frame.winfo_width() - 20)
            except Exception:
                max_w = 390
            max_h = 120

            # Mask preview mode -- show detected regions as red rectangles
            if show_mask:
                self._preview_label.config(text="Detecting...", image="")
                self._preview_label.update_idletasks()
                try:
                    from backend.processor import SubtitleDetector
                    det = SubtitleDetector(lang=self.lang_var.get())
                    threshold = getattr(self.config, '_detection_threshold_pct', 50) / 100.0
                    if self.config.subtitle_area:
                        boxes = [self.config.subtitle_area]
                    else:
                        boxes = det.detect(raw_frame, threshold)
                    # Draw red rectangles on frame
                    vis = raw_frame.copy()
                    for (x1, y1, x2, y2) in boxes:
                        _cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    img = to_pil(vis)
                    count_text = f" ({len(boxes)} regions)" if boxes else " (none found)"
                    img.thumbnail((max_w, max_h), Image.LANCZOS)
                    self._preview_photo = ImageTk.PhotoImage(img)
                    self._preview_label.config(
                        image=self._preview_photo,
                        text=f"{det._engine_name}: {len(boxes)} detected" if boxes else "No text detected")
                except Exception as e:
                    self._preview_label.config(text=f"Detection error: {e}", image="")
                return

            input_img = to_pil(raw_frame)

            # Check if completed and output exists -- show before/after
            output_img = None
            if item.status == ProcessingStatus.COMPLETE and Path(item.output_path).exists():
                out_frame = load_first_frame_raw(item.output_path)
                if out_frame is not None:
                    output_img = to_pil(out_frame)

            if output_img:
                half_w = max_w // 2 - 2
                input_img.thumbnail((half_w, max_h), Image.LANCZOS)
                output_img.thumbnail((half_w, max_h), Image.LANCZOS)
                total_w = input_img.width + output_img.width + 4
                total_h = max(input_img.height, output_img.height)
                composite = Image.new("RGB", (total_w, total_h), (15, 23, 42))
                composite.paste(input_img, (0, 0))
                composite.paste(output_img, (input_img.width + 4, 0))
                draw = ImageDraw.Draw(composite)
                draw.line([(input_img.width + 1, 0), (input_img.width + 1, total_h)],
                          fill="#22c55e", width=2)
                self._preview_photo = ImageTk.PhotoImage(composite)
                self._preview_label.config(image=self._preview_photo, text="")
            else:
                input_img.thumbnail((max_w, max_h), Image.LANCZOS)
                self._preview_photo = ImageTk.PhotoImage(input_img)
                self._preview_label.config(image=self._preview_photo, text="")
        except Exception as e:
            self._preview_label.config(text=f"Preview error: {e}", image="")

    def _retry_failed(self):
        """Reset failed/cancelled items so they can be reprocessed."""
        if self.is_processing:
            self._update_status("Cannot retry while processing")
            return
        count = 0
        with self.queue_lock:
            for item in self.queue:
                if item.status in (ProcessingStatus.ERROR, ProcessingStatus.CANCELLED):
                    item.status = ProcessingStatus.IDLE
                    item.progress = 0.0
                    item.message = "Queued for retry"
                    item.error = None
                    item.started_at = None
                    item.completed_at = None
                    count += 1
        if count:
            self._update_queue_display()
            # Force-refresh all widgets to show reset state
            for item in self.queue:
                if item.message == "Queued for retry" and item.id in self.queue_widgets:
                    self.queue_widgets[item.id].update_item(item)
            self._update_status(f"Reset {count} item{'s' if count != 1 else ''} for retry")
        else:
            self._update_status("No failed items to retry")

    def _set_settings_locked(self, locked: bool):
        """Lock or unlock settings controls during processing."""
        state = "disabled" if locked else "normal"
        try:
            self.skip_check.config(state=state)
            self.lama_check.config(state=state)
        except Exception:
            pass

    def _start_processing(self):
        """Start processing the queue."""
        if not self.queue:
            self._update_status("Add files to the queue first")
            return

        if self.is_processing:
            self._stop_processing()
            return

        self.is_processing = True
        self.cancel_event.clear()
        self._set_settings_locked(True)
        self.start_btn.set_text("Stop Processing")
        self.start_btn.bg_color = Theme.ERROR
        self.start_btn.hover_color = "#dc2626"
        self.start_btn._draw()

        # Start elapsed timer
        self._start_elapsed_timer()

        # Start processing thread
        threading.Thread(target=self._process_queue, daemon=True).start()

    def _stop_processing(self):
        """Stop the current processing."""
        self.is_processing = False
        self.cancel_event.set()
        self._stop_elapsed_timer()
        self._set_settings_locked(False)

        self.start_btn.set_text("Start Processing")
        self.start_btn.bg_color = Theme.GREEN_PRIMARY
        self.start_btn.hover_color = Theme.GREEN_HOVER
        self.start_btn._draw()
        self.root.title(f"{APP_NAME} v{APP_VERSION}")
        self._update_status("Processing stopped")

    def _start_elapsed_timer(self):
        """Start a timer that updates elapsed times on in-progress queue items."""
        def tick():
            if not self.is_processing:
                return
            try:
                for widget in list(self.queue_widgets.values()):
                    if widget.item.started_at and not widget.item.completed_at:
                        elapsed = (datetime.now() - widget.item.started_at).total_seconds()
                        widget.time_label.config(text=format_time(elapsed))
            except Exception:
                pass
            self._elapsed_timer_id = self.root.after(1000, tick)
        self._elapsed_timer_id = self.root.after(1000, tick)

    def _stop_elapsed_timer(self):
        if self._elapsed_timer_id:
            self.root.after_cancel(self._elapsed_timer_id)
            self._elapsed_timer_id = None

    def _process_queue(self):
        """Process all items in the queue."""
        with self.queue_lock:
            items_to_process = [i for i in self.queue
                                if i.status not in (ProcessingStatus.COMPLETE,
                                                     ProcessingStatus.ERROR,
                                                     ProcessingStatus.CANCELLED)]

        total = len(items_to_process)
        for idx, item in enumerate(items_to_process):
            if self.cancel_event.is_set():
                # Mark ALL remaining items as cancelled
                for remaining in items_to_process[idx:]:
                    remaining.status = ProcessingStatus.CANCELLED
                    remaining.message = "Cancelled"
                    self._update_item_display(remaining)
                break

            # Update batch progress + window title
            self.root.after(0, self._update_batch_progress, idx, total)
            self._process_item(item)

        # Final batch state
        self.root.after(0, self._update_batch_progress, total, total)
        # Reset button
        self.root.after(0, self._on_processing_complete)

    def _process_item(self, item: QueueItem):
        """Process a single queue item using the backend processor."""
        try:
            item.status = ProcessingStatus.LOADING
            item.started_at = datetime.now()
            item.progress = 0.0
            item.message = "Initializing..."
            self._update_item_display(item)

            from backend.processor import (
                SubtitleRemover as BackendRemover,
                ProcessingConfig as BackendConfig,
                InpaintMode as BackendInpaintMode,
            )

            # Map GUI enum values to backend enum values
            mode_map = {
                "STTN": BackendInpaintMode.STTN,
                "LAMA": BackendInpaintMode.LAMA,
                "ProPainter": BackendInpaintMode.PROPAINTER,
            }

            device = f"cuda:{item.config.gpu_id}" if item.config.use_gpu else "cpu"

            backend_config = BackendConfig(
                mode=mode_map.get(item.config.mode.value, BackendInpaintMode.STTN),
                device=device,
                sttn_skip_detection=item.config.sttn_skip_detection,
                sttn_neighbor_stride=item.config.sttn_neighbor_stride,
                sttn_reference_length=item.config.sttn_reference_length,
                sttn_max_load_num=item.config.sttn_max_load_num,
                lama_super_fast=item.config.lama_super_fast,
                preserve_audio=item.config.preserve_audio,
                output_quality=item.config.output_quality,
                detection_lang=getattr(item.config, 'detection_lang', 'en'),
                detection_threshold=getattr(item.config, 'detection_threshold', 0.5),
                subtitle_area=item.config.subtitle_area,
                time_start=getattr(item.config, 'time_start', 0.0),
                time_end=getattr(item.config, 'time_end', 0.0),
            )

            remover = BackendRemover(backend_config)

            def on_progress(progress: float, message: str):
                if self.cancel_event.is_set():
                    raise InterruptedError("Processing cancelled")
                # Map backend progress to GUI status
                if progress < 0.3:
                    item.status = ProcessingStatus.DETECTING
                elif progress < 0.9:
                    item.status = ProcessingStatus.PROCESSING
                elif progress < 1.0:
                    item.status = ProcessingStatus.MERGING
                else:
                    item.status = ProcessingStatus.COMPLETE
                item.progress = progress
                item.message = message
                self._update_item_display(item)

            remover.on_progress = on_progress

            # Ensure output directory exists
            Path(item.output_path).parent.mkdir(parents=True, exist_ok=True)

            # Run the actual processing
            file_name = Path(item.file_path).name
            logger.info(f"Processing: {file_name} with {item.config.mode.value}")

            if is_video_file(item.file_path):
                success = remover.process_video(item.file_path, item.output_path)
            elif is_image_file(item.file_path):
                success = remover.process_image(item.file_path, item.output_path)
            else:
                raise ValueError(f"Unsupported file type: {Path(item.file_path).suffix}")

            if success:
                item.status = ProcessingStatus.COMPLETE
                item.progress = 1.0
                item.message = "Complete!"
                item.completed_at = datetime.now()
                elapsed = (item.completed_at - item.started_at).total_seconds()
                logger.info(f"Completed: {file_name} in {format_time(elapsed)}")
            else:
                item.status = ProcessingStatus.ERROR
                item.message = "Processing failed"
                logger.error(f"Failed: {file_name}")
            self._update_item_display(item)

        except InterruptedError:
            item.status = ProcessingStatus.CANCELLED
            item.message = "Cancelled"
            self._update_item_display(item)
            logger.info(f"Cancelled: {Path(item.file_path).name}")
        except Exception as e:
            item.status = ProcessingStatus.ERROR
            item.error = str(e)
            item.message = f"Error: {str(e)}"
            self._update_item_display(item)
            logger.error(f"Processing error for {item.file_path}: {e}")

    def _update_batch_progress(self, current: int, total: int):
        """Update the overall batch progress bar and window title."""
        if total > 0:
            progress = current / total
            self.batch_progress.set_progress(progress)
            self.batch_label.config(text=f"{current}/{total}")
            self.root.title(f"[{current}/{total}] {APP_NAME} v{APP_VERSION}")
        else:
            self.batch_progress.set_progress(0)
            self.batch_label.config(text="")

    def _update_item_display(self, item: QueueItem):
        """Update the display for a queue item."""
        def update():
            if item.id in self.queue_widgets:
                self.queue_widgets[item.id].update_item(item)
            self._update_status(f"Processing: {Path(item.file_path).name} - {item.message}")

        self.root.after(0, update)

    def _on_processing_complete(self):
        """Handle processing completion."""
        self.is_processing = False
        self.cancel_event.clear()
        self._stop_elapsed_timer()
        self._set_settings_locked(False)
        self.start_btn.set_text("Start Processing")
        self.start_btn.bg_color = Theme.GREEN_PRIMARY
        self.start_btn.hover_color = Theme.GREEN_HOVER
        self.start_btn._draw()
        self.root.title(f"{APP_NAME} v{APP_VERSION}")
        self.batch_progress.set_progress(0)
        self.batch_label.config(text="")

        complete = sum(1 for item in self.queue if item.status == ProcessingStatus.COMPLETE)
        errors = sum(1 for item in self.queue if item.status == ProcessingStatus.ERROR)

        summary = f"Done: {complete} succeeded, {errors} failed"
        self._update_status(summary)
        logger.info(summary)
        self._notify_completion(complete, errors)

    def _notify_completion(self, complete: int, errors: int):
        """Flash taskbar + play sound when batch processing finishes."""
        # Flash the taskbar icon to draw attention
        try:
            import ctypes
            import ctypes.wintypes
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())

            class FLASHWINFO(ctypes.Structure):
                _fields_ = [
                    ('cbSize', ctypes.wintypes.UINT),
                    ('hwnd', ctypes.wintypes.HWND),
                    ('dwFlags', ctypes.wintypes.DWORD),
                    ('uCount', ctypes.wintypes.UINT),
                    ('dwTimeout', ctypes.wintypes.DWORD),
                ]

            FLASHW_ALL = 0x03
            FLASHW_TIMERNOFG = 0x0C
            fwi = FLASHWINFO(
                ctypes.sizeof(FLASHWINFO), hwnd,
                FLASHW_ALL | FLASHW_TIMERNOFG, 5, 0)
            ctypes.windll.user32.FlashWindowEx(ctypes.byref(fwi))
        except Exception:
            pass
        # Completion sound
        try:
            import winsound
            if errors == 0:
                winsound.MessageBeep(winsound.MB_OK)
            else:
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        except Exception:
            pass

    def run(self):
        """Run the application."""
        # Center window
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

        logger.info(f"{APP_NAME} v{APP_VERSION} started")
        logger.info(f"Log file: {LOG_FILE}")
        self.root.mainloop()


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    # High DPI support on Windows -- Per-Monitor V2 for best multi-monitor support
    try:
        from ctypes import windll
        # Try Per-Monitor V2 first (Windows 10 1703+), then fall back
        try:
            windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    app = VideoSubtitleRemoverApp()
    app.run()


if __name__ == "__main__":
    main()
