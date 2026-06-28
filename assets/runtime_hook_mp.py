"""PyInstaller runtime hook for multiprocessing-safe Windows launches."""

import multiprocessing

multiprocessing.freeze_support()
