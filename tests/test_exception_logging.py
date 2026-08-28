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
        # RM-349: this was a hand-written list of the processor's files, so
        # moving the frame loop into its own mixin took "Detection denoise
        # fell back" out of scope without failing anything. The processor is
        # `processor.py` plus every `_*_mixin.py` beside it; deriving the
        # list means the next split cannot quietly drop coverage either.
        sources = sorted((ROOT / "backend").glob("_*_mixin.py"))
        self.assertGreaterEqual(len(sources), 6, [p.name for p in sources])
        text = _read("backend/processor.py") + "".join(
            path.read_text(encoding="utf-8") for path in sources)
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
                "Tiled LaMa-ONNX fell back to full-frame",
                "Tiled LaMa cv2.dnn fell back to full-frame",
                "Batched LaMa fell back to per-frame",
                "Tiled LaMa fell back to full-frame",
            ),
        }
        for path, phrases in targets.items():
            text = _read(path)
            for phrase in phrases:
                _assert_exc_info(self, text, phrase)


if __name__ == "__main__":
    unittest.main()
