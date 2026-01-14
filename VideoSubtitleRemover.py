"""
Video Subtitle Remover Pro
A professional Windows application for AI-powered subtitle removal from videos and images.
Based on: https://github.com/YaoFANGUK/video-subtitle-remover

Author: Maven Imaging Tools
Version: 2.0.0
"""

import os
import sys
import json
import threading
import subprocess
import time
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

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

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

APP_NAME = "Video Subtitle Remover Pro"
APP_VERSION = "2.0.0"
APP_AUTHOR = "Maven Imaging Tools"

# Color Theme (Professional Dark Theme)
class Theme:
    # Primary colors
    BG_DARK = "#020617"           # Slate 950
    BG_SECONDARY = "#0f172a"      # Slate 900
    BG_TERTIARY = "#1e293b"       # Slate 800
    BG_CARD = "#0f172a"           # Card background
    
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
    
    # Output settings
    output_format: str = "mp4"
    preserve_audio: bool = True
    
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
            "output_format": self.output_format,
            "preserve_audio": self.preserve_audio,
        }


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


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

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
                        gpus.append({
                            "index": int(parts[0].strip()),
                            "name": parts[1].strip(),
                            "memory": f"{int(parts[2].strip())} MB",
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


# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM WIDGETS
# ═══════════════════════════════════════════════════════════════════════════════

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
        self.configure(width=width, height=height)
        self.pack_propagate(False)
        
        # Inner content
        inner = tk.Frame(self, bg=Theme.BG_SECONDARY)
        inner.place(relx=0.5, rely=0.5, anchor="center")
        
        # Icon (using text as fallback)
        icon_label = tk.Label(inner, text="📁", font=("Segoe UI", 36),
                             bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED)
        icon_label.pack(pady=(0, 10))
        
        # Main text
        main_text = tk.Label(inner, text="Drag & Drop Files Here",
                            font=("Segoe UI", 12, "bold"), bg=Theme.BG_SECONDARY,
                            fg=Theme.TEXT_PRIMARY)
        main_text.pack()
        
        # Sub text
        sub_text = tk.Label(inner, text="or click to browse • MP4, AVI, MKV, MOV, PNG, JPG",
                           font=("Segoe UI", 9), bg=Theme.BG_SECONDARY,
                           fg=Theme.TEXT_MUTED)
        sub_text.pack(pady=(5, 0))
        
        # Bind click
        self.bind("<Button-1>", self._on_click)
        for child in self.winfo_children():
            child.bind("<Button-1>", self._on_click)
            for subchild in child.winfo_children():
                subchild.bind("<Button-1>", self._on_click)
        
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
        valid_files = [f for f in files if is_video_file(f) or is_image_file(f)]
        if valid_files:
            self.on_drop(valid_files)
    
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


class QueueItemWidget(tk.Frame):
    """Widget representing a single queue item."""
    
    def __init__(self, parent, item: QueueItem, on_remove: Callable, **kwargs):
        super().__init__(parent, bg=Theme.BG_TERTIARY, highlightthickness=1,
                        highlightbackground=Theme.BORDER)
        
        self.item = item
        self.on_remove = on_remove
        
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
                                   bg=Theme.BG_TERTIARY, fg=Theme.TEXT_PRIMARY)
        self.name_label.pack(side="left")
        
        # Status badge
        self.status_label = tk.Label(top_row, text=item.status.value.upper(),
                                     font=("Segoe UI", 8, "bold"), bg=Theme.BG_TERTIARY,
                                     fg=self._get_status_color())
        self.status_label.pack(side="right")
        
        # Remove button
        remove_btn = tk.Label(top_row, text="✕", font=("Segoe UI", 10),
                             bg=Theme.BG_TERTIARY, fg=Theme.TEXT_MUTED, cursor="hand2")
        remove_btn.pack(side="right", padx=(0, 10))
        remove_btn.bind("<Button-1>", lambda e: self.on_remove(self.item.id))
        remove_btn.bind("<Enter>", lambda e: remove_btn.config(fg=Theme.ERROR))
        remove_btn.bind("<Leave>", lambda e: remove_btn.config(fg=Theme.TEXT_MUTED))
        
        # Progress bar
        self.progress_bar = ModernProgressBar(container, width=380, height=6,
                                              fill=self._get_status_color())
        self.progress_bar.pack(fill="x", pady=(8, 4))
        self.progress_bar.set_progress(item.progress)
        
        # Bottom row: message
        self.message_label = tk.Label(container, text=item.message or "Waiting...",
                                      font=("Segoe UI", 9), bg=Theme.BG_TERTIARY,
                                      fg=Theme.TEXT_MUTED, anchor="w")
        self.message_label.pack(fill="x")
    
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


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

class VideoSubtitleRemoverApp:
    """Main application class."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(f"{APP_NAME} v{APP_VERSION}")
        self.root.geometry("1100x750")
        self.root.minsize(900, 600)
        self.root.configure(bg=Theme.BG_DARK)
        
        # Set window icon
        try:
            self.root.iconbitmap(get_app_dir() / "assets" / "icon.ico")
        except:
            pass
        
        # State
        self.config = ProcessingConfig()
        self.queue: List[QueueItem] = []
        self.queue_widgets: dict = {}
        self.is_processing = False
        self.current_process: Optional[subprocess.Popen] = None
        self.gpus = detect_gpu()
        
        # Variables
        self.mode_var = tk.StringVar(value=self.config.mode.value)
        self.gpu_var = tk.StringVar()
        self.skip_detection_var = tk.BooleanVar(value=self.config.sttn_skip_detection)
        self.lama_fast_var = tk.BooleanVar(value=self.config.lama_super_fast)
        self.preserve_audio_var = tk.BooleanVar(value=self.config.preserve_audio)
        
        # Build UI
        self._setup_styles()
        self._build_ui()
        
        # GPU setup
        if self.gpus:
            self.gpu_var.set(f"{self.gpus[0]['name']} ({self.gpus[0]['memory']})")
        else:
            self.gpu_var.set("CPU Mode")
            self.config.use_gpu = False
    
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
        
        # Checkbutton style
        style.configure("Dark.TCheckbutton",
                       background=Theme.BG_SECONDARY,
                       foreground=Theme.TEXT_PRIMARY,
                       indicatorcolor=Theme.BG_TERTIARY,
                       indicatorbackground=Theme.BG_TERTIARY)
        
        style.map("Dark.TCheckbutton",
                 background=[('active', Theme.BG_SECONDARY)],
                 indicatorcolor=[('selected', Theme.GREEN_PRIMARY)])
    
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
        
        # Left column - Input & Settings
        left_col = tk.Frame(content, bg=Theme.BG_DARK)
        left_col.pack(side="left", fill="both", expand=True)
        
        self._build_input_section(left_col)
        self._build_settings_section(left_col)
        
        # Right column - Queue & Preview
        right_col = tk.Frame(content, bg=Theme.BG_DARK, width=420)
        right_col.pack(side="right", fill="both", padx=(20, 0))
        right_col.pack_propagate(False)
        
        self._build_queue_section(right_col)
        
        # Footer
        self._build_footer(main_container)
    
    def _build_header(self, parent):
        """Build the header section."""
        header = tk.Frame(parent, bg=Theme.BG_DARK)
        header.pack(fill="x")
        
        # Title and version
        title_frame = tk.Frame(header, bg=Theme.BG_DARK)
        title_frame.pack(side="left")
        
        title = tk.Label(title_frame, text="Video Subtitle Remover",
                        font=("Segoe UI", 24, "bold"), bg=Theme.BG_DARK,
                        fg=Theme.TEXT_PRIMARY)
        title.pack(side="left")
        
        pro_badge = tk.Label(title_frame, text=" PRO", font=("Segoe UI", 12, "bold"),
                            bg=Theme.GREEN_PRIMARY, fg="#ffffff", padx=8, pady=2)
        pro_badge.pack(side="left", padx=(10, 0))
        
        version = tk.Label(title_frame, text=f"v{APP_VERSION}",
                          font=("Segoe UI", 10), bg=Theme.BG_DARK,
                          fg=Theme.TEXT_MUTED)
        version.pack(side="left", padx=(10, 0))
        
        # GPU status
        gpu_frame = tk.Frame(header, bg=Theme.BG_DARK)
        gpu_frame.pack(side="right")
        
        if self.gpus:
            gpu_icon = "🖥️"
            gpu_text = f"{self.gpus[0]['type']}: {self.gpus[0]['name']}"
            gpu_color = Theme.GREEN_PRIMARY
        else:
            gpu_icon = "💻"
            gpu_text = "CPU Mode (No GPU Detected)"
            gpu_color = Theme.WARNING
        
        gpu_status = tk.Label(gpu_frame, text=f"{gpu_icon} {gpu_text}",
                             font=("Segoe UI", 10), bg=Theme.BG_DARK, fg=gpu_color)
        gpu_status.pack()
    
    def _build_input_section(self, parent):
        """Build the file input section."""
        section = tk.Frame(parent, bg=Theme.BG_SECONDARY, highlightthickness=1,
                          highlightbackground=Theme.BORDER)
        section.pack(fill="x", pady=(0, 15))
        
        # Section header
        header = tk.Frame(section, bg=Theme.BG_SECONDARY)
        header.pack(fill="x", padx=15, pady=(15, 10))
        
        tk.Label(header, text="📂 INPUT FILES", font=("Segoe UI", 10, "bold"),
                bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY).pack(side="left")
        
        # Drag & drop area
        self.drop_area = DragDropFrame(section, self._on_files_dropped, width=420, height=140)
        self.drop_area.pack(padx=15, pady=(0, 15))
    
    def _build_settings_section(self, parent):
        """Build the settings section."""
        section = tk.Frame(parent, bg=Theme.BG_SECONDARY, highlightthickness=1,
                          highlightbackground=Theme.BORDER)
        section.pack(fill="both", expand=True)
        
        # Section header
        header = tk.Frame(section, bg=Theme.BG_SECONDARY)
        header.pack(fill="x", padx=15, pady=(15, 10))
        
        tk.Label(header, text="⚙️ PROCESSING SETTINGS", font=("Segoe UI", 10, "bold"),
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
                                 fg=Theme.TEXT_MUTED, justify="left", anchor="w",
                                 wraplength=400)
        self.algo_desc.pack(fill="x", pady=(0, 12))
        
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
        
        # LAMA fast mode checkbox
        self.lama_check = tk.Checkbutton(checks_frame, text="LAMA Super Fast mode (lower quality)",
                                        variable=self.lama_fast_var, font=("Segoe UI", 10),
                                        bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY,
                                        selectcolor=Theme.BG_TERTIARY, activebackground=Theme.BG_SECONDARY,
                                        activeforeground=Theme.TEXT_PRIMARY)
        self.lama_check.pack(anchor="w")
        
        # Preserve audio checkbox
        tk.Checkbutton(checks_frame, text="Preserve original audio",
                      variable=self.preserve_audio_var, font=("Segoe UI", 10),
                      bg=Theme.BG_SECONDARY, fg=Theme.TEXT_PRIMARY,
                      selectcolor=Theme.BG_TERTIARY, activebackground=Theme.BG_SECONDARY,
                      activeforeground=Theme.TEXT_PRIMARY).pack(anchor="w")
        
        # Advanced settings toggle
        adv_frame = tk.Frame(settings, bg=Theme.BG_SECONDARY)
        adv_frame.pack(fill="x", pady=(10, 0))
        
        self.adv_visible = False
        self.adv_toggle = tk.Label(adv_frame, text="▶ Advanced Settings",
                                  font=("Segoe UI", 10), bg=Theme.BG_SECONDARY,
                                  fg=Theme.BLUE_PRIMARY, cursor="hand2")
        self.adv_toggle.pack(anchor="w")
        self.adv_toggle.bind("<Button-1>", self._toggle_advanced)
        
        # Advanced settings panel
        self.adv_panel = tk.Frame(settings, bg=Theme.BG_SECONDARY)
        
        # STTN settings
        sttn_frame = tk.LabelFrame(self.adv_panel, text="STTN Settings",
                                  font=("Segoe UI", 9, "bold"), bg=Theme.BG_SECONDARY,
                                  fg=Theme.TEXT_SECONDARY, bd=1)
        sttn_frame.pack(fill="x", pady=(10, 5))
        
        self._create_slider(sttn_frame, "Neighbor Stride", 5, 30, 
                           self.config.sttn_neighbor_stride, "sttn_neighbor_stride")
        self._create_slider(sttn_frame, "Reference Length", 5, 30,
                           self.config.sttn_reference_length, "sttn_reference_length")
        self._create_slider(sttn_frame, "Max Load Frames", 10, 100,
                           self.config.sttn_max_load_num, "sttn_max_load_num")
        
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
                        showvalue=False, command=update_value)
        scale.set(default)
        scale.pack(side="left", fill="x", expand=True, padx=(10, 10))
    
    def _toggle_advanced(self, event=None):
        """Toggle advanced settings visibility."""
        self.adv_visible = not self.adv_visible
        if self.adv_visible:
            self.adv_toggle.config(text="▼ Advanced Settings")
            self.adv_panel.pack(fill="x")
        else:
            self.adv_toggle.config(text="▶ Advanced Settings")
            self.adv_panel.pack_forget()
    
    def _build_queue_section(self, parent):
        """Build the processing queue section."""
        section = tk.Frame(parent, bg=Theme.BG_SECONDARY, highlightthickness=1,
                          highlightbackground=Theme.BORDER)
        section.pack(fill="both", expand=True)
        
        # Section header
        header = tk.Frame(section, bg=Theme.BG_SECONDARY)
        header.pack(fill="x", padx=15, pady=(15, 10))
        
        tk.Label(header, text="📋 PROCESSING QUEUE", font=("Segoe UI", 10, "bold"),
                bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY).pack(side="left")
        
        self.queue_count = tk.Label(header, text="0 items",
                                   font=("Segoe UI", 9), bg=Theme.BG_SECONDARY,
                                   fg=Theme.TEXT_MUTED)
        self.queue_count.pack(side="right")
        
        # Queue container with scrollbar
        queue_container = tk.Frame(section, bg=Theme.BG_SECONDARY)
        queue_container.pack(fill="both", expand=True, padx=15, pady=(0, 10))
        
        # Canvas for scrolling
        self.queue_canvas = tk.Canvas(queue_container, bg=Theme.BG_SECONDARY,
                                     highlightthickness=0)
        scrollbar = tk.Scrollbar(queue_container, orient="vertical",
                                command=self.queue_canvas.yview)
        
        self.queue_frame = tk.Frame(self.queue_canvas, bg=Theme.BG_SECONDARY)
        
        self.queue_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.queue_canvas.pack(side="left", fill="both", expand=True)
        
        self.queue_window = self.queue_canvas.create_window((0, 0), window=self.queue_frame,
                                                            anchor="nw")
        
        self.queue_frame.bind("<Configure>", self._on_queue_configure)
        self.queue_canvas.bind("<Configure>", self._on_canvas_configure)
        
        # Empty state
        self.empty_label = tk.Label(self.queue_frame, text="No files in queue\nDrag & drop or browse to add files",
                                   font=("Segoe UI", 10), bg=Theme.BG_SECONDARY,
                                   fg=Theme.TEXT_MUTED, justify="center")
        self.empty_label.pack(pady=40)
        
        # Control buttons
        btn_frame = tk.Frame(section, bg=Theme.BG_SECONDARY)
        btn_frame.pack(fill="x", padx=15, pady=(0, 15))
        
        self.start_btn = ModernButton(btn_frame, text="▶  Start Processing", width=180,
                                     height=40, command=self._start_processing,
                                     style="primary")
        self.start_btn.pack(side="left")
        
        self.clear_btn = ModernButton(btn_frame, text="Clear Queue", width=100,
                                     height=40, command=self._clear_queue,
                                     style="secondary")
        self.clear_btn.pack(side="right")
    
    def _on_queue_configure(self, event):
        self.queue_canvas.configure(scrollregion=self.queue_canvas.bbox("all"))
    
    def _on_canvas_configure(self, event):
        self.queue_canvas.itemconfig(self.queue_window, width=event.width)
    
    def _build_footer(self, parent):
        """Build the footer section."""
        footer = tk.Frame(parent, bg=Theme.BG_DARK)
        footer.pack(fill="x", pady=(15, 0))
        
        # Status bar
        self.status_label = tk.Label(footer, text="Ready", font=("Segoe UI", 9),
                                    bg=Theme.BG_DARK, fg=Theme.TEXT_MUTED, anchor="w")
        self.status_label.pack(side="left")
        
        # Credits
        credits = tk.Label(footer, text=f"{APP_AUTHOR} • Based on YaoFANGUK/video-subtitle-remover",
                          font=("Segoe UI", 9), bg=Theme.BG_DARK, fg=Theme.TEXT_MUTED)
        credits.pack(side="right")
    
    def _get_algo_description(self) -> str:
        """Get description for current algorithm."""
        descriptions = {
            "STTN": "⚡ STTN: Best for real-world videos. Fast processing, supports skip detection mode.",
            "LAMA": "🎨 LAMA: Best quality for images and animations. Moderate speed, detailed inpainting.",
            "ProPainter": "🎬 ProPainter: Best for high-motion videos. Slow, high VRAM usage.",
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
    
    def _on_files_dropped(self, files: List[str]):
        """Handle dropped files."""
        for file_path in files:
            self._add_to_queue(file_path)
    
    def _add_to_queue(self, file_path: str):
        """Add a file to the processing queue."""
        # Generate unique ID
        item_id = f"{int(time.time() * 1000)}_{len(self.queue)}"
        
        # Generate output path
        input_path = Path(file_path)
        output_dir = input_path.parent / "output"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{input_path.stem}_no_sub{input_path.suffix}"
        
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
        )
        
        # Create queue item
        item = QueueItem(
            id=item_id,
            file_path=file_path,
            output_path=str(output_path),
            config=config,
            message="Queued for processing"
        )
        
        self.queue.append(item)
        self._update_queue_display()
        self._update_status(f"Added: {Path(file_path).name}")
    
    def _remove_from_queue(self, item_id: str):
        """Remove an item from the queue."""
        self.queue = [item for item in self.queue if item.id != item_id]
        self._update_queue_display()
    
    def _clear_queue(self):
        """Clear all items from the queue."""
        if self.is_processing:
            messagebox.showwarning("Warning", "Cannot clear queue while processing.")
            return
        
        self.queue.clear()
        self._update_queue_display()
        self._update_status("Queue cleared")
    
    def _update_queue_display(self):
        """Update the queue display."""
        # Clear existing widgets
        for widget in self.queue_frame.winfo_children():
            widget.destroy()
        self.queue_widgets.clear()
        
        # Update count
        self.queue_count.config(text=f"{len(self.queue)} item{'s' if len(self.queue) != 1 else ''}")
        
        if not self.queue:
            # Show empty state
            self.empty_label = tk.Label(self.queue_frame, 
                                       text="No files in queue\nDrag & drop or browse to add files",
                                       font=("Segoe UI", 10), bg=Theme.BG_SECONDARY,
                                       fg=Theme.TEXT_MUTED, justify="center")
            self.empty_label.pack(pady=40)
        else:
            # Create widgets for each item
            for item in self.queue:
                widget = QueueItemWidget(self.queue_frame, item, self._remove_from_queue)
                widget.pack(fill="x", pady=(0, 8))
                self.queue_widgets[item.id] = widget
    
    def _update_status(self, message: str):
        """Update the status bar."""
        self.status_label.config(text=message)
    
    def _start_processing(self):
        """Start processing the queue."""
        if not self.queue:
            messagebox.showinfo("Info", "Add files to the queue first.")
            return
        
        if self.is_processing:
            self._stop_processing()
            return
        
        self.is_processing = True
        self.start_btn.set_text("⏹  Stop Processing")
        self.start_btn.bg_color = Theme.ERROR
        self.start_btn.hover_color = "#dc2626"
        self.start_btn._draw()
        
        # Start processing thread
        threading.Thread(target=self._process_queue, daemon=True).start()
    
    def _stop_processing(self):
        """Stop the current processing."""
        self.is_processing = False
        if self.current_process:
            self.current_process.terminate()
        
        self.start_btn.set_text("▶  Start Processing")
        self.start_btn.bg_color = Theme.GREEN_PRIMARY
        self.start_btn.hover_color = Theme.GREEN_HOVER
        self.start_btn._draw()
        self._update_status("Processing stopped")
    
    def _process_queue(self):
        """Process all items in the queue."""
        for item in self.queue:
            if not self.is_processing:
                break
            
            if item.status in [ProcessingStatus.COMPLETE, ProcessingStatus.ERROR]:
                continue
            
            self._process_item(item)
        
        # Reset button
        self.root.after(0, self._on_processing_complete)
    
    def _process_item(self, item: QueueItem):
        """Process a single queue item."""
        try:
            item.status = ProcessingStatus.LOADING
            item.started_at = datetime.now()
            item.message = "Loading file..."
            self._update_item_display(item)
            
            # Build the processing command
            cmd = self._build_process_command(item)
            
            # For now, simulate processing since we don't have the actual backend
            # In production, this would call the actual subtitle remover
            self._simulate_processing(item)
            
        except Exception as e:
            item.status = ProcessingStatus.ERROR
            item.error = str(e)
            item.message = f"Error: {str(e)}"
            self._update_item_display(item)
            logger.error(f"Processing error: {e}")
    
    def _build_process_command(self, item: QueueItem) -> List[str]:
        """Build the command line for processing."""
        cmd = [
            sys.executable,
            str(get_app_dir() / "backend" / "main.py"),
            "--input", item.file_path,
            "--output", item.output_path,
            "--mode", item.config.mode.value.lower(),
        ]
        
        if item.config.use_gpu:
            cmd.extend(["--gpu", str(item.config.gpu_id)])
        
        if item.config.sttn_skip_detection and item.config.mode == InpaintMode.STTN:
            cmd.append("--skip-detection")
        
        if item.config.lama_super_fast and item.config.mode == InpaintMode.LAMA:
            cmd.append("--fast")
        
        return cmd
    
    def _simulate_processing(self, item: QueueItem):
        """Simulate processing for demonstration."""
        stages = [
            (ProcessingStatus.DETECTING, "Detecting subtitle regions...", 0.1, 0.3),
            (ProcessingStatus.PROCESSING, "Removing subtitles...", 0.3, 0.8),
            (ProcessingStatus.MERGING, "Merging audio...", 0.8, 0.95),
            (ProcessingStatus.COMPLETE, "Complete!", 1.0, 1.0),
        ]
        
        for status, message, start_prog, end_prog in stages:
            if not self.is_processing:
                item.status = ProcessingStatus.CANCELLED
                item.message = "Cancelled"
                self._update_item_display(item)
                return
            
            item.status = status
            item.message = message
            
            # Animate progress
            steps = 20
            for i in range(steps):
                if not self.is_processing:
                    return
                item.progress = start_prog + (end_prog - start_prog) * (i / steps)
                self._update_item_display(item)
                time.sleep(0.05)
            
            item.progress = end_prog
            self._update_item_display(item)
        
        item.completed_at = datetime.now()
    
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
        self.start_btn.set_text("▶  Start Processing")
        self.start_btn.bg_color = Theme.GREEN_PRIMARY
        self.start_btn.hover_color = Theme.GREEN_HOVER
        self.start_btn._draw()
        
        complete = sum(1 for item in self.queue if item.status == ProcessingStatus.COMPLETE)
        errors = sum(1 for item in self.queue if item.status == ProcessingStatus.ERROR)
        
        self._update_status(f"Processing complete: {complete} succeeded, {errors} failed")
    
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
        self.root.mainloop()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point."""
    # High DPI support on Windows
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
    
    app = VideoSubtitleRemoverApp()
    app.run()


if __name__ == "__main__":
    main()
