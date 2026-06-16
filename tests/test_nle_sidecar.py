import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
