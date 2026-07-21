"""RM-132: source frame-rate probe for NLE timecode->frame conversion.

The NLE ingest path used a bare ``VideoCapture`` probe that silently fell
back to 24 fps on any failure, mis-mapping every SMPTE timecode on 25/30/
50/60 fps sources.  ``probe_video_fps`` prefers ffprobe (accurate for VFR
and non-24 rates), falls back to OpenCV, and returns ``None`` on total
failure so the caller can warn before substituting a default.
"""

import types
import unittest
from unittest import mock

from backend import io as vsr_io
from backend import nle_sidecar


def _proc(stdout: str, returncode: int = 0):
    return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")


class ProbeVideoFpsTests(unittest.TestCase):
    def test_prefers_ffprobe_avg_frame_rate(self):
        payload = '{"streams": [{"avg_frame_rate": "30/1", "r_frame_rate": "60/1"}]}'
        with mock.patch.object(vsr_io.shutil, "which", return_value="ffprobe"), \
                mock.patch.object(vsr_io, "run_process", return_value=_proc(payload)):
            self.assertAlmostEqual(vsr_io.probe_video_fps("clip.mp4"), 30.0)

    def test_falls_back_to_r_frame_rate_when_avg_missing(self):
        payload = '{"streams": [{"avg_frame_rate": "0/0", "r_frame_rate": "25/1"}]}'
        with mock.patch.object(vsr_io.shutil, "which", return_value="ffprobe"), \
                mock.patch.object(vsr_io, "run_process", return_value=_proc(payload)):
            self.assertAlmostEqual(vsr_io.probe_video_fps("clip.mp4"), 25.0)

    def test_falls_back_to_opencv_when_ffprobe_absent(self):
        fake_cap = mock.Mock()
        fake_cap.isOpened.return_value = True
        fake_cap.get.return_value = 59.94
        fake_cv2 = types.SimpleNamespace(
            VideoCapture=mock.Mock(return_value=fake_cap),
            CAP_PROP_FPS=5,
        )
        with mock.patch.object(vsr_io.shutil, "which", return_value=None), \
                mock.patch.dict("sys.modules", {"cv2": fake_cv2}):
            self.assertAlmostEqual(vsr_io.probe_video_fps("clip.mp4"), 59.94)
        fake_cap.release.assert_called_once()

    def test_returns_none_when_all_probes_fail(self):
        fake_cap = mock.Mock()
        fake_cap.isOpened.return_value = False
        fake_cv2 = types.SimpleNamespace(
            VideoCapture=mock.Mock(return_value=fake_cap),
            CAP_PROP_FPS=5,
        )
        with mock.patch.object(vsr_io.shutil, "which", return_value=None), \
                mock.patch.dict("sys.modules", {"cv2": fake_cv2}):
            self.assertIsNone(vsr_io.probe_video_fps("clip.mp4"))


class NleFrameMappingTests(unittest.TestCase):
    """A frame-bearing SMPTE timecode must resolve against the real fps."""

    _EDL = (
        "TITLE: demo\n"
        "001  AX       V     C        "
        "00:00:01:15 00:00:03:00 00:00:01:15 00:00:03:00\n"
    )

    def _write_edl(self, tmp_path):
        p = tmp_path / "demo.edl"
        p.write_text(self._EDL, encoding="utf-8")
        return str(p)

    def test_30fps_maps_frame_component_correctly(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            edl = self._write_edl(Path(d))
            seg30 = nle_sidecar.parse_nle_input(edl, fps=30.0)
            seg24 = nle_sidecar.parse_nle_input(edl, fps=24.0)
        # 00:00:01:15 -> 1 + 15/30 = 1.5s at 30fps, 1 + 15/24 = 1.625s at 24fps.
        self.assertEqual(len(seg30), 1)
        self.assertAlmostEqual(seg30[0][0], 1.5, places=4)
        self.assertAlmostEqual(seg24[0][0], 1.625, places=4)
        self.assertNotAlmostEqual(seg30[0][0], seg24[0][0], places=3)


if __name__ == "__main__":
    unittest.main()
