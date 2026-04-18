# Assets Folder

This directory stores optional UI and packaging assets used by the desktop app.

- `icon.ico`, `icon.png`, `favicon.ico`, and `banner.png` are picked up by the
  launchers, About dialog, and EXE packaging flow when present.
- Keep source artwork lightweight and repository-safe. Large generated exports
  can stay outside the repo until they are finalized.
- Missing assets are handled gracefully. The app and build scripts fall back to
  text branding instead of failing.
