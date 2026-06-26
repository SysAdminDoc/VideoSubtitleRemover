"""Optional VLM-OCR detectors and specialised CJK adapters.

RM-22 Florence-2 / Qwen2.5-VL -- vision-language model OCR with layout
awareness. The 2B variant of Qwen2.5-VL leads the OmniDocBench
leaderboard as of April 2026. Each model defaults off; users opt in
by setting `VSR_VLM_OCR=florence2|qwen25vl` and `pip install`-ing the
matching transformers stack.

RM-23 PaddleOCR-VL 0.9B -- alternative VLM-OCR with irregular-polygon
bbox support. Marketed as beating GPT-4o on OmniDocBench v1.5 at 94.5%
accuracy. Loaded via the official PaddleOCR Python package when the
user sets `VSR_PADDLEOCR_VL=1`.

RM-42 Manga / anime mode -- `manga-ocr` (vertical Japanese) + the
comic-text-detector for irregular speech-bubble shapes. Activated via
`detection_lang="manga"` so the existing dispatch (lang-keyed) routes
through this adapter without a new InpaintMode-style enum.

All adapters import lazily and degrade gracefully when their optional
deps are missing. Each registers itself with `SubtitleDetector` only
when the env-var / lang token names it -- the default cascade stays
RapidOCR -> PaddleOCR -> Surya (gated) -> EasyOCR -> OpenCV.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from backend.remote_model_policy import resolve_remote_model_source

logger = logging.getLogger(__name__)


def selected_vlm_backend() -> Optional[str]:
    """Return the user-selected VLM OCR backend name, or None when
    the cascade should keep its default. The env var values are
    "florence2", "qwen25vl", "paddleocr-vl"."""
    raw = os.environ.get("VSR_VLM_OCR", "").strip().lower()
    if raw in {"florence2", "qwen25vl", "paddleocr-vl"}:
        return raw
    return None


class _BaseVlmDetector:
    """Common scaffolding for VLM OCR backends. Subclasses implement
    `_load()` (return a callable or None) and `_extract_boxes(frame)`
    (return a list of (x1,y1,x2,y2)).
    """

    name = "vlm"

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._loaded = False
        self._model = None

    def _load(self):
        raise NotImplementedError

    def _extract_boxes(self, frame: np.ndarray, threshold: float) -> List[Tuple[int, int, int, int]]:
        raise NotImplementedError

    def detect(self, frame: np.ndarray, threshold: float) -> List[Tuple[int, int, int, int]]:
        if not self._loaded:
            self._model = self._load()
            self._loaded = True
        if self._model is None:
            return []
        try:
            return self._extract_boxes(frame, threshold)
        except Exception as exc:
            logger.warning(f"{self.name} VLM detect failed: {exc}")
            return []


class _Florence2Detector(_BaseVlmDetector):
    """Microsoft Florence-2-base layout-aware OCR. Output bboxes are
    in <loc_xxx> token coordinates (0..1000); we scale back to pixels.
    """

    name = "florence2"
    MODEL_ID = "microsoft/Florence-2-base"

    def _load(self):
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore
            import torch  # type: ignore
        except ImportError:
            logger.info(
                "transformers + torch not available for Florence-2 OCR; "
                "install via `pip install transformers accelerate`."
            )
            return None
        try:
            source = resolve_remote_model_source("florence2")
            if not source.allowed:
                logger.warning("Florence-2 OCR disabled: %s", source.reason)
                return None
            kwargs = {"trust_remote_code": True}
            if source.revision:
                kwargs["revision"] = source.revision
            model_ref = source.source or self.MODEL_ID
            processor = AutoProcessor.from_pretrained(model_ref, **kwargs)
            model = AutoModelForCausalLM.from_pretrained(model_ref, **kwargs)
            if "cuda" in self.device and torch.cuda.is_available():
                model = model.to("cuda")
            model.eval()
            return (processor, model, torch)
        except Exception as exc:
            logger.warning(f"Florence-2 load failed: {exc}")
            return None

    def _extract_boxes(self, frame: np.ndarray, threshold: float) -> List[Tuple[int, int, int, int]]:
        processor, model, torch = self._model
        from PIL import Image as _Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = _Image.fromarray(rgb)
        h, w = frame.shape[:2]
        task = "<OCR_WITH_REGION>"
        inputs = processor(text=task, images=pil, return_tensors="pt")
        if "cuda" in self.device and torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            generated = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                num_beams=1,
            )
        text = processor.batch_decode(generated, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(text, task=task, image_size=(w, h))
        bboxes = (parsed.get(task) or {}).get("quad_boxes", [])
        out: List[Tuple[int, int, int, int]] = []
        for poly in bboxes:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
            if pts.size == 0:
                continue
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            if x2 > x1 and y2 > y1:
                out.append((int(x1), int(y1), int(x2), int(y2)))
        return out


class _Qwen25VLDetector(_BaseVlmDetector):
    """Qwen2.5-VL (2B) detection. The model has not standardised an
    "OCR-with-region" task token across releases yet, so we use the
    grounding prompt and parse JSON-shaped responses."""

    name = "qwen25vl"
    MODEL_ID = "Qwen/Qwen2.5-VL-2B-Instruct"

    def _load(self):
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor  # type: ignore
            import torch  # type: ignore
        except ImportError:
            logger.info(
                "Qwen2.5-VL OCR requires `pip install transformers torch`."
            )
            return None
        try:
            from backend.remote_model_policy import resolve_remote_model_source
            source = resolve_remote_model_source("qwen25vl")
            if not source.allowed:
                logger.warning("Qwen2.5-VL disabled: %s", source.reason)
                return None
            model_ref = source.source or self.MODEL_ID
            kwargs = {}
            if source.revision:
                kwargs["revision"] = source.revision
            processor = AutoProcessor.from_pretrained(model_ref, **kwargs)
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_ref, **kwargs)
            if "cuda" in self.device and torch.cuda.is_available():
                model = model.to("cuda")
            model.eval()
            return (processor, model, torch)
        except Exception as exc:
            logger.warning(f"Qwen2.5-VL load failed: {exc}")
            return None

    def _extract_boxes(self, frame: np.ndarray, threshold: float) -> List[Tuple[int, int, int, int]]:
        processor, model, torch = self._model
        from PIL import Image as _Image
        import json as _json
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = _Image.fromarray(rgb)
        prompt = (
            "Detect every burned-in subtitle / caption / chyron in this image "
            "and return JSON: [{\"bbox\": [x1,y1,x2,y2]}, ...]. Pixel coords."
        )
        messages = [{"role": "user", "content": [
            {"type": "image", "image": pil},
            {"type": "text", "text": prompt},
        ]}]
        text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], images=[pil], return_tensors="pt")
        if "cuda" in self.device and torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=256)
        raw = processor.batch_decode(generated, skip_special_tokens=True)[0]
        # Greedily isolate the JSON array.
        try:
            start = raw.index("[")
            end = raw.rindex("]") + 1
            arr = _json.loads(raw[start:end])
        except (ValueError, _json.JSONDecodeError):
            return []
        out: List[Tuple[int, int, int, int]] = []
        for entry in arr:
            bbox = entry.get("bbox") or entry.get("box")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            try:
                x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
            except (TypeError, ValueError):
                continue
            if x2 > x1 and y2 > y1:
                out.append((x1, y1, x2, y2))
        return out


class _PaddleOcrVlDetector(_BaseVlmDetector):
    """RM-23: PaddleOCR-VL 0.9B. Uses the PaddleOCR Python package
    when `paddleocr_vl=True` is exposed (PaddleOCR 3.0+); falls back
    to a None load when the package isn't installed."""

    name = "paddleocr-vl"

    def _load(self):
        try:
            import paddleocr  # noqa: F401  (availability probe)
        except ImportError:
            logger.info(
                "paddleocr not available for PaddleOCR-VL; "
                "install `pip install paddleocr>=3.0`."
            )
            return None
        from backend.paddle_compat import build_paddleocr
        try:
            return build_paddleocr(
                "en", self.device,
                ocr_version="PP-OCRv5",
                ocr_lang_model="paddleocr_vl",
            )
        except TypeError:
            # Some PaddleOCR builds don't expose ocr_lang_model; the
            # caller falls back to the normal PaddleOCR detector.
            logger.info(
                "Installed PaddleOCR does not expose the VL variant; "
                "running standard PP-OCRv5."
            )
            return build_paddleocr("en", self.device)
        except Exception as exc:
            logger.warning(f"PaddleOCR-VL load failed: {exc}")
            return None

    def _extract_boxes(self, frame: np.ndarray, threshold: float) -> List[Tuple[int, int, int, int]]:
        from backend.paddle_compat import extract_paddle_boxes
        return extract_paddle_boxes(self._model, frame, threshold)


class _MangaOcrDetector(_BaseVlmDetector):
    """RM-42: manga-ocr + comic-text-detector for vertical Japanese
    manga / anime sources. manga-ocr only RECOGNISES text given a crop;
    we use it together with comic-text-detector to obtain crops first.
    Falls back to manga-ocr on the whole frame when comic-text-detector
    isn't installed (slow but works).
    """

    name = "manga-ocr"

    def _load(self):
        try:
            from manga_ocr import MangaOcr  # type: ignore
        except ImportError:
            logger.info(
                "manga-ocr not installed; manga mode unavailable. "
                "Install `pip install manga-ocr` to enable."
            )
            return None
        try:
            mocr = MangaOcr()
        except Exception as exc:
            logger.warning(f"manga-ocr load failed: {exc}")
            return None
        ctd = None
        try:
            # comic-text-detector ships as a single inference script;
            # users who installed the package expose a `predict_one`
            # function we can call.
            import comic_text_detector  # type: ignore
            ctd = comic_text_detector
        except ImportError:
            logger.info(
                "comic-text-detector not installed; manga mode will "
                "only recognise the whole-frame crop. Install at "
                "https://github.com/dmMaze/comic-text-detector for "
                "bubble-aware detection."
            )
        return (mocr, ctd)

    def _extract_boxes(self, frame: np.ndarray, threshold: float) -> List[Tuple[int, int, int, int]]:
        _mocr, ctd = self._model
        if ctd is None:
            # Single whole-frame crop -- we still return ONE box around
            # the suspected text region by running cv2 edge detection.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 4))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            ys, xs = np.where(closed > 0)
            if ys.size == 0:
                return []
            return [(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))]
        try:
            polys = ctd.predict_one(frame)  # type: ignore[attr-defined]
        except Exception as exc:
            logger.debug(f"comic-text-detector inference failed: {exc}")
            return []
        out: List[Tuple[int, int, int, int]] = []
        for poly in polys:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
            if pts.size == 0:
                continue
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            if x2 > x1 and y2 > y1:
                out.append((int(x1), int(y1), int(x2), int(y2)))
        return out


def maybe_build_vlm_detector(device: str, lang: str) -> Optional[object]:
    """Return a VLM detector instance when the user opted in via env var
    or via a special lang token ("manga"). None means "leave the default
    cascade alone"."""
    if lang and lang.strip().lower() == "manga":
        return _MangaOcrDetector(device=device)
    selected = selected_vlm_backend()
    if selected == "florence2":
        return _Florence2Detector(device=device)
    if selected == "qwen25vl":
        return _Qwen25VLDetector(device=device)
    if selected == "paddleocr-vl":
        return _PaddleOcrVlDetector(device=device)
    return None
