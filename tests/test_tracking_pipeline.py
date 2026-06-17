import importlib
import sys
import types
import unittest
from contextlib import contextmanager

import numpy as np


class _FakeKalmanFilter:
    def __init__(self, state_size, measurement_size):
        self.statePost = np.zeros((state_size, 1), dtype=np.float32)
        self.transitionMatrix = np.eye(state_size, dtype=np.float32)

    def predict(self):
        self.statePost = self.transitionMatrix.dot(self.statePost)
        return self.statePost

    def correct(self, measurement):
        self.statePost[:4, 0] = measurement[:, 0]
        return self.statePost


@contextmanager
def _fresh_tracking_module():
    old_tracking = sys.modules.pop("backend.tracking", None)
    old_cv2 = sys.modules.get("cv2")
    had_cv2 = "cv2" in sys.modules
    fake_cv2 = types.SimpleNamespace(KalmanFilter=_FakeKalmanFilter)
    sys.modules["cv2"] = fake_cv2
    try:
        yield importlib.import_module("backend.tracking")
    finally:
        sys.modules.pop("backend.tracking", None)
        if old_tracking is not None:
            sys.modules["backend.tracking"] = old_tracking
        if had_cv2:
            sys.modules["cv2"] = old_cv2
        else:
            sys.modules.pop("cv2", None)


class SubtitleTrackerTests(unittest.TestCase):
    def test_tracker_survives_single_frame_miss_and_recovers_identity(self):
        with _fresh_tracking_module() as tracking:
            tracker = tracking.SubtitleTracker(iou_threshold=0.2, max_age=2)
            initial = tracker.update([(10, 10, 30, 20)])
            missed = tracker.update([])
            recovered = tracker.update([(11, 10, 31, 20)])

            self.assertEqual(len(initial), 1)
            self.assertEqual(len(missed), 1)
            self.assertEqual(len(recovered), 1)
            self.assertEqual(len(tracker._tracks), 1)
            self.assertEqual(tracker._tracks[0].age, 0)
            self.assertEqual(tracker._tracks[0].hits, 2)
            self.assertEqual(tracker.categorize(min_chyron_hits=2), ["chyron"])


if __name__ == "__main__":
    unittest.main()
