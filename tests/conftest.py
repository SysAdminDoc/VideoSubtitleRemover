"""Keep the suite out of the real user profile (RM-368).

Settings, the log, the saved queue, crash-resume checkpoints, the update
check and the TensorRT cache all resolve from %APPDATA%. Tests that drive the
CLI or the queue used to write straight into the developer's own profile:
stray `.done` markers that can make a real job look finished, a test queue
of `0.mp4` and `1.mp4` in place of the saved one, and paused test jobs
beside real ones.

This runs before any test module imports product code, so paths computed
at import time (``gui.config.LOG_DIR``) land here too, and child processes
inherit the environment.
"""

from __future__ import annotations

import atexit
import os
from pathlib import Path
import shutil
import tempfile
import time

_PREFIX = "vsr-test-profile-"
# The single-instance lock stays open until the interpreter exits, after
# every atexit handler, so a run's own cleanup can leave `state.lock` behind.
# Sweep what earlier runs left; 12 hours is far past any run's length, so a
# concurrent run's profile is never touched.
for _stale in Path(tempfile.gettempdir()).glob(_PREFIX + "*"):
    try:
        if time.time() - _stale.stat().st_mtime > 12 * 3600:
            shutil.rmtree(_stale, ignore_errors=True)
    except OSError:
        pass

_PROFILE_ROOT = tempfile.mkdtemp(prefix=_PREFIX)
os.environ["VSR_TEST_PROFILE_ROOT"] = _PROFILE_ROOT
os.environ["APPDATA"] = os.path.join(_PROFILE_ROOT, "AppData", "Roaming")
os.makedirs(os.environ["APPDATA"], exist_ok=True)
atexit.register(shutil.rmtree, _PROFILE_ROOT, ignore_errors=True)
