# VideoSubtitleRemover - Working Notes

## Tech Stack
- **GUI**: Python/tkinter (~2150 lines), dark theme, tooltips, before/after preview, region selector
- **Backend**: (~535 lines) PaddleOCR/EasyOCR/OpenCV detection + simple-lama-inpainting + cv2 fallback
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
- Settings lock during processing only affects skip_check + lama_check (scoped, not recursive)
- `_on_close` sets cancel_event before destroying root
- `_process_queue` marks ALL remaining items as cancelled (not just next)
- Output dirs created at processing time (not queue-add time)
- Queue capped at 500 items
- FFmpeg audio merge has 10min timeout to prevent deadlock
- ALL Python source files are pure ASCII (no unicode section dividers)

## Encoding Rules (CRITICAL)
- ALL .py files must be pure ASCII -- no em-dashes, no box-drawing chars, no unicode decorators
- .bat files must be ASCII only
- setup.py uses `os.system('')` to enable ANSI colors on Windows

## Version History
- **3.2.0** -- Comprehensive audit: 13 bugs fixed (race conditions, resource leaks, crash paths), tooltips, bigger preview, CI/CD easyocr, FFmpeg timeout, queue cap, ASCII-only source, .gitignore expanded
- **3.1.0** -- Before/after comparison, AI engine badges, Open Log File, right-click folder
- **3.0.0** -- Real LAMA inpainting, EasyOCR fallback, multi-language, region selector, folder input, CI/CD
- **2.x** -- Bug fixes, settings persistence, log panel, queue features
- **2.0.0** -- Initial release

## Gotchas
- STTN/ProPainter still use cv2.inpaint -- only LAMA has real neural inpainting
- InpaintMode enums differ between GUI and backend -- NEVER unify them
- EasyOCR uses different language codes than PaddleOCR -- mapping in backend `_load_model`
- Region selector uses `import cv2 as _cv2` -- must use `_cv2` consistently in that scope
- `subtitle_area` stored as list in JSON, converted to tuple on load
- `_set_settings_locked` is intentionally scoped to only skip_check + lama_check -- DO NOT walk full widget tree
- Tooltip class binds with `add="+"` to not clobber existing bindings
- VideoCapture in region selector uses try/finally for leak-safe release
- FFmpeg subprocess has 600s timeout -- prevents pipe deadlock on verbose output

## Current Status
- v3.2.0, production-ready, zero non-ASCII in source
