import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

from backend import nle_sidecar


class NleSidecarEncodingTests(unittest.TestCase):
    def test_edl_and_fcpxml_allow_non_ascii_filenames(self):
        source_name = "\u65e5\u672c\u8a9e_source.mp4"
        cleaned_name = "\u6e05\u7406\u6e08\u307f.mp4"
        with tempfile.TemporaryDirectory() as tmpdir:
            edl_path = str(Path(tmpdir) / "out.edl")
            fcpxml_path = str(Path(tmpdir) / "out.fcpxml")
            cleaned_path = str(Path(tmpdir) / cleaned_name)
            written_edl = nle_sidecar.write_edl(
                edl_path,
                source=f"/clips/{source_name}",
                cleaned=cleaned_path,
                fps=24.0,
                start_s=1.0,
                end_s=3.0,
            )
            written_fcpxml = nle_sidecar.write_fcpxml(
                fcpxml_path,
                source=f"/clips/{source_name}",
                cleaned=cleaned_path,
                fps=24.0,
                start_s=1.0,
                end_s=3.0,
            )
            edl_text = Path(written_edl).read_text(encoding="utf-8")
            fcpxml_text = Path(written_fcpxml).read_text(encoding="utf-8")
        self.assertIn(source_name, edl_text)
        self.assertIn(cleaned_name, edl_text)
        self.assertIn(source_name[:-4], fcpxml_text)

    def test_fcpxml_escapes_attribute_filenames(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fcpxml_path = Path(tmpdir) / "out.fcpxml"
            written = nle_sidecar.write_fcpxml(
                str(fcpxml_path),
                source="/clips/rough cut & review's clip.mp4",
                cleaned=str(Path(tmpdir) / "clean & final.mp4"),
                fps=24.0,
                start_s=0.0,
                end_s=1.0,
            )

            root = ElementTree.parse(written).getroot()

        asset = root.find("./resources/asset")
        self.assertIsNotNone(asset)
        self.assertEqual(asset.attrib["name"], "rough cut & review's clip")
        self.assertIn("clean%20%26%20final.mp4", asset.attrib["src"])
        self.assertTrue(asset.attrib["src"].startswith("file:///"))

    def test_edl_clip_comments_are_single_line(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            edl_path = Path(tmpdir) / "out.edl"
            written = nle_sidecar.write_edl(
                str(edl_path),
                source="/clips/source\nname.mp4",
                cleaned=str(Path(tmpdir) / "cleaned.mp4"),
                fps=24.0,
                start_s=0.0,
                end_s=1.0,
            )

            lines = Path(written).read_text(encoding="utf-8").splitlines()

        from_lines = [line for line in lines if line.startswith("* FROM CLIP NAME:")]
        self.assertEqual(from_lines, ["* FROM CLIP NAME: source name.mp4"])


class NleIngestTests(unittest.TestCase):
    def test_edl_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            edl_path = str(Path(tmpdir) / "out.edl")
            nle_sidecar.write_edl(
                edl_path, source="src.mp4", cleaned="clean.mp4",
                fps=24.0, start_s=10.0, end_s=30.0,
            )
            segments = nle_sidecar.parse_edl(edl_path, fps=24.0)
        self.assertEqual(len(segments), 1)
        self.assertAlmostEqual(segments[0][0], 10.0, places=1)
        self.assertAlmostEqual(segments[0][1], 30.0, places=1)

    def test_fcpxml_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = str(Path(tmpdir) / "out.fcpxml")
            nle_sidecar.write_fcpxml(
                xml_path, source="src.mp4", cleaned="clean.mp4",
                fps=30.0, start_s=5.0, end_s=15.0,
            )
            segments = nle_sidecar.parse_fcpxml(xml_path)
        self.assertEqual(len(segments), 1)
        self.assertGreater(segments[0][1], segments[0][0])

    def test_parse_nle_input_auto_detects_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            edl_path = str(Path(tmpdir) / "timeline.edl")
            nle_sidecar.write_edl(
                edl_path, source="src.mp4", cleaned="clean.mp4",
                fps=24.0, start_s=0.0, end_s=60.0,
            )
            segments = nle_sidecar.parse_nle_input(edl_path, fps=24.0)
        self.assertEqual(len(segments), 1)

    def test_missing_file_returns_empty(self):
        segments = nle_sidecar.parse_edl("/nonexistent/file.edl")
        self.assertEqual(segments, [])

    def test_smpte_to_seconds(self):
        self.assertAlmostEqual(
            nle_sidecar._smpte_to_seconds("01:00:00:00", 24.0), 3600.0)
        self.assertAlmostEqual(
            nle_sidecar._smpte_to_seconds("00:01:30:12", 24.0), 90.5)


class MultiSegmentNleTests(unittest.TestCase):
    def test_edl_multi_segment_round_trip(self):
        segments = [(5.0, 10.0), (20.0, 30.0), (45.0, 55.0)]
        with tempfile.TemporaryDirectory() as tmpdir:
            edl_path = str(Path(tmpdir) / "multi.edl")
            nle_sidecar.write_edl(
                edl_path, source="src.mp4", cleaned="clean.mp4",
                fps=24.0, start_s=5.0, end_s=55.0,
                segments=segments, width=1920, height=1080,
            )
            parsed = nle_sidecar.parse_edl(edl_path, fps=24.0)
            text = Path(edl_path).read_text(encoding="utf-8")
        self.assertEqual(len(parsed), 3)
        self.assertAlmostEqual(parsed[0][0], 5.0, places=1)
        self.assertAlmostEqual(parsed[1][0], 20.0, places=1)
        self.assertAlmostEqual(parsed[2][1], 55.0, places=1)
        self.assertIn("1920x1080", text)

    def test_fcpxml_multi_segment_round_trip(self):
        segments = [(0.0, 5.0), (10.0, 20.0)]
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = str(Path(tmpdir) / "multi.fcpxml")
            nle_sidecar.write_fcpxml(
                xml_path, source="src.mp4", cleaned="clean.mp4",
                fps=30.0, start_s=0.0, end_s=20.0,
                segments=segments, width=3840, height=2160,
            )
            parsed = nle_sidecar.parse_fcpxml(xml_path)
            root = ElementTree.parse(xml_path).getroot()
        self.assertEqual(len(parsed), 2)
        fmt = root.find("./resources/format")
        self.assertEqual(fmt.attrib["width"], "3840")
        self.assertEqual(fmt.attrib["height"], "2160")

    def test_edl_single_segment_backward_compatible(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            edl_path = str(Path(tmpdir) / "single.edl")
            nle_sidecar.write_edl(
                edl_path, source="src.mp4", cleaned="clean.mp4",
                fps=24.0, start_s=10.0, end_s=30.0,
            )
            parsed = nle_sidecar.parse_edl(edl_path, fps=24.0)
        self.assertEqual(len(parsed), 1)
        self.assertAlmostEqual(parsed[0][0], 10.0, places=1)

    def test_fcpxml_dimensions_default_when_not_provided(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = str(Path(tmpdir) / "default.fcpxml")
            nle_sidecar.write_fcpxml(
                xml_path, source="src.mp4", cleaned="clean.mp4",
                fps=24.0, start_s=0.0, end_s=5.0,
            )
            root = ElementTree.parse(xml_path).getroot()
        fmt = root.find("./resources/format")
        self.assertEqual(fmt.attrib["width"], "1920")
        self.assertEqual(fmt.attrib["height"], "1080")


if __name__ == "__main__":
    unittest.main()
