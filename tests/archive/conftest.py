"""Archived GUI tests: kept for reference, excluded from every run.

These exercised the tkinter layer (widget behavior, dialog layout, smoke
runs of the assembled app). They churned with every visual change while
catching little, so they were retired from the suite on 2026-08-20. The
backend, controller-logic, i18n-catalog, and release-gate tests remain in
``tests/``. To resurrect one, move it back up a directory; nothing here is
imported by the live suite.
"""

collect_ignore_glob = ["*.py"]
