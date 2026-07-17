from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _assert_exc_info(testcase: unittest.TestCase, text: str, phrase: str):
    index = text.find(phrase)
    testcase.assertNotEqual(index, -1, f"Missing log phrase: {phrase}")
    snippet = text[index:index + 320]
    testcase.assertIn("exc_info=True", snippet, phrase)


class ProcessingExceptionLoggingTests(unittest.TestCase):
    def test_processor_processing_fallbacks_keep_tracebacks(self):
        text = _read("backend/processor.py")
        for phrase in (
            "RapidOCR SRT extraction failed",
            "PaddleOCR SRT extraction failed",
            "EasyOCR SRT extraction failed",
            "Quality ROI metric calculation failed",
            "Detection denoise fell back",
            "Video writer release failed",
            "Matte writer cleanup failed",
            "Prefetch reader release failed",
            "Input capture release failed",
            "Whisper temp cleanup failed",
            "HDR encode argument generation failed",
            "FFmpeg re-encode failed; salvaging intermediate",
        ):
            _assert_exc_info(self, text, phrase)

    def test_gui_processing_callbacks_keep_tracebacks(self):
        text = _read("gui/processing_controller.py")
        for phrase in (
            "Live preview callback failed",
            "Processing error for",
        ):
            _assert_exc_info(self, text, phrase)

    def test_inpainter_fallbacks_keep_tracebacks(self):
        targets = {
            "backend/inpainters/auto.py": (
                "AUTO ProPainter idle GC cleanup failed",
                "AUTO ProPainter idle CUDA cleanup failed",
            ),
            "backend/inpainters/propainter.py": (
                "ProPainter LaMa residual refinement failed",
            ),
            "backend/inpainters/lama.py": (
                "LaMa-ONNX tile inference failed",
                "Batched LaMa fell back to per-frame",
                "Tiled LaMa fell back to full-frame",
                "LaMa inpaint failed for frame",
                "LaMa PyTorch tile inference failed",
            ),
        }
        for path, phrases in targets.items():
            text = _read(path)
            for phrase in phrases:
                _assert_exc_info(self, text, phrase)


if __name__ == "__main__":
    unittest.main()
