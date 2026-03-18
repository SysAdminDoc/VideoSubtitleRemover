# VideoSubtitleRemover - Working Notes

## Tech Stack
- **GUI**: Python/tkinter (~2250 lines), dark theme, tooltips, before/after preview, mask preview, region selector
- **Backend**: (~570 lines) PaddleOCR/EasyOCR/OpenCV detection + simple-lama-inpainting + cv2 fallback
- **Audio**: FFmpeg subprocess (10min timeout)
- **Build**: PyInstaller, GitHub Actions CI/CD

## Key File Paths
- `VideoSubtitleRemover.py` -- Main GUI
- `backend/processor.py` -- Detection + inpainting pipeline
- Settings: `%APPDATA%\VideoSubtitleRemoverPro\settings.json`
- Logs: `%APPDATA%\VideoSubtitleRemoverPro\vsr_pro.log`

## Architecture
- **Detection chain**: PaddleOCR > EasyOCR > OpenCV fallback (automatic)
- **EasyOCR language mapping**: PaddleOCR "ch" -> EasyOCR "ch_sim" (backend `_load_model`)
- **Inpainting**: LAMA uses `simple-lama-inpainting` (neural), STTN/ProPainter use cv2.inpaint
- GUI/backend InpaintMode enums have different values -- mapped via `mode_map` dict
- `subtitle_area` persisted in settings.json as list, restored as tuple
- `_detection_threshold_pct` is a pseudo-attribute (int 10-90), converted to float in `_sync_config_from_ui`
- Time range: `time_start`/`time_end` in seconds, backend seeks to start frame and stops at end frame
- Mask preview: right-click queue item filename runs detection and draws red boxes on preview
- Image output quality: JPEG 95%, PNG compression 3, WebP 95% (auto by extension)
- Queue capped at 500 items
- FFmpeg audio merge has 10min timeout
- ALL source files are pure ASCII

## Encoding Rules (CRITICAL)
- ALL .py files must be pure ASCII -- no em-dashes, no box-drawing chars, no unicode decorators
- .bat files must be ASCII only
- setup.py uses `os.system('')` to enable ANSI colors on Windows

## Version History
- **3.3.0** -- Detection threshold slider, video time-range fields, mask preview (right-click), image quality preservation (JPEG/PNG/WebP), CLI --start/--end/--threshold flags, committed + pushed + branch protection enabled
- **3.2.0** -- Comprehensive audit: 13 bugs fixed, tooltips, CI/CD easyocr, FFmpeg timeout, queue cap, ASCII-only source
- **3.1.0** -- Before/after comparison, AI engine badges, Open Log File, right-click folder
- **3.0.0** -- Real LAMA inpainting, EasyOCR fallback, multi-language, region selector, folder input, CI/CD
- **2.x** -- Bug fixes, settings persistence, log panel, queue features
- **2.0.0** -- Initial release

## Gotchas
- STTN/ProPainter still use cv2.inpaint -- only LAMA has real neural inpainting
- InpaintMode enums differ between GUI and backend -- NEVER unify them
- EasyOCR uses different language codes than PaddleOCR -- mapping in backend `_load_model`
- `_detection_threshold_pct` is stored as int (10-90) on config, converted to float (0.1-0.9) in sync
- Region selector uses `import cv2 as _cv2` -- must use `_cv2` consistently in that scope
- `subtitle_area` stored as list in JSON, converted to tuple on load
- VideoCapture in region selector and _show_preview uses try/finally for leak-safe release
- `time_start`/`time_end` of 0 means "full video" -- backend checks `> 0` before seeking

## Current Status
- v3.3.0, committed + pushed, branch protection enabled
