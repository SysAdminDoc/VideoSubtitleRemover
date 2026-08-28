"""Optional VLM-OCR detectors and specialised CJK adapters.

RM-22 Florence-2 / Qwen2.5-VL -- vision-language model OCR with layout
awareness. The 2B variant of Qwen2.5-VL leads the OmniDocBench
leaderboard as of April 2026. Each model defaults off; users opt in
by setting `VSR_VLM_OCR=florence2|qwen25vl` and `pip install`-ing the
matching transformers stack.

RM-23 PaddleOCR-VL 0.9B -- alternative VLM-OCR with irregular-polygon
bbox support. Marketed as beating GPT-4o on OmniDocBench v1.5 at 94.5%
accuracy. Loaded via the official PaddleOCR Python package when the
user sets `VSR_VLM_OCR=paddleocr-vl`.

RM-113 PaddleOCR-VL-1.5 llama.cpp -- CPU/edge VLM-OCR tier using a
local `llama-server` OpenAI-compatible endpoint. Activated by
`VSR_PADDLEOCR_VL=1`; returns None during construction if the server or
PaddleOCRVL entrypoint is unavailable so the normal OCR cascade keeps
working.

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
import json
import ipaddress
import socket
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import urljoin, urlsplit

import cv2
import numpy as np

from backend.remote_model_policy import resolve_remote_model_source
from backend.detection_geometry import DetectionGeometry
from backend.safe_image import safe_imwrite

logger = logging.getLogger(__name__)

_TRUE_FLAG_VALUES = {"1", "true", "yes", "on"}
_PADDLEOCR_VL_LLAMA_VALUES = {
    "paddleocr-vl-llama",
    "paddleocr-vl-llamacpp",
    "paddleocr-vl15",
    "paddleocr-vl-1.5",
}
_PADDLEOCR_VL_LLAMA_DEFAULT_URL = "http://127.0.0.1:8080/v1"
_PADDLEOCR_VL_REMOTE_ACK_ENV = "VSR_ALLOW_REMOTE_VLM"
_PADDLEOCR_VL_SKIP_PROBE_ENV = "VSR_PADDLEOCR_VL_SKIP_SERVER_PROBE"
Box = Tuple[int, int, int, int]


def _env_truthy(env: Mapping[str, str], name: str) -> bool:
    return str(env.get(name, "") or "").strip().lower() in _TRUE_FLAG_VALUES


def selected_vlm_backend() -> Optional[str]:
    """Return the user-selected VLM OCR backend name, or None when
    the cascade should keep its default. The env var values are
    "florence2", "qwen25vl", "paddleocr-vl", and
    "paddleocr-vl-llama". The legacy `VSR_PADDLEOCR_VL=1` flag selects
    the llama.cpp-backed PaddleOCR-VL-1.5 path."""
    raw = os.environ.get("VSR_VLM_OCR", "").strip().lower()
    if raw in {"florence2", "qwen25vl", "paddleocr-vl"}:
        return raw
    if raw in _PADDLEOCR_VL_LLAMA_VALUES:
        return "paddleocr-vl-llama"
    if _env_truthy(os.environ, "VSR_PADDLEOCR_VL"):
        return "paddleocr-vl-llama"
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

    def _extract_geometry(
        self, frame: np.ndarray, threshold: float
    ) -> List[DetectionGeometry]:
        output: List[DetectionGeometry] = []
        for box in self._extract_boxes(frame, threshold):
            detection = DetectionGeometry.from_box(box)
            if detection is not None:
                output.append(detection)
        return output

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
            raise

    def detect_with_geometry(
        self, frame: np.ndarray, threshold: float
    ) -> List[DetectionGeometry]:
        if not self._loaded:
            self._model = self._load()
            self._loaded = True
        if self._model is None:
            return []
        try:
            return list(self._extract_geometry(frame, threshold))
        except Exception as exc:
            logger.warning(f"{self.name} VLM geometry detect failed: {exc}")
            raise


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

    def _extract_geometry(
        self, frame: np.ndarray, threshold: float
    ) -> List[DetectionGeometry]:
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
        out: List[DetectionGeometry] = []
        for poly in bboxes:
            detection = DetectionGeometry.from_polygon(poly, frame.shape)
            if detection is not None:
                out.append(detection)
        return out

    def _extract_boxes(
        self, frame: np.ndarray, threshold: float
    ) -> List[Tuple[int, int, int, int]]:
        return [
            detection.bbox
            for detection in self._extract_geometry(frame, threshold)
        ]


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

    def _extract_geometry(
        self, frame: np.ndarray, threshold: float
    ) -> List[DetectionGeometry]:
        processor, model, torch = self._model
        from PIL import Image as _Image
        import json as _json
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = _Image.fromarray(rgb)
        prompt = (
            "Detect every burned-in subtitle / caption / chyron in this image "
            "and return JSON: [{\"polygon\": [[x,y], ...]}, ...]. "
            "Use pixel coords and preserve rotated text quadrilaterals."
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
        if not isinstance(arr, list):
            return []
        out: List[DetectionGeometry] = []
        for entry in arr:
            if isinstance(entry, dict):
                coords = entry.get("polygon") or entry.get("poly")
                polygon = coords is not None
                if coords is None:
                    coords = entry.get("bbox") or entry.get("box")
                confidence = entry.get(
                    "confidence", entry.get("score", 1.0))
            elif isinstance(entry, (list, tuple)):
                # Some models emit a bare [x1, y1, x2, y2] per detection
                # instead of {"bbox": [...]}; accept that shape too rather
                # than dropping every box.
                coords = entry
                polygon = False
                confidence = 1.0
            else:
                continue
            if not isinstance(coords, (list, tuple)):
                continue
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = 1.0
            if confidence < threshold:
                continue
            detection = DetectionGeometry.from_polygon(
                coords,
                frame.shape,
                confidence=confidence,
            )
            if not polygon:
                detection = DetectionGeometry.from_box(
                    coords,
                    frame.shape,
                    confidence=confidence,
                )
                if detection is None:
                    detection = DetectionGeometry.from_polygon(
                        coords,
                        frame.shape,
                        confidence=confidence,
                    )
            if detection is not None:
                out.append(detection)
        return out

    def _extract_boxes(
        self, frame: np.ndarray, threshold: float
    ) -> List[Tuple[int, int, int, int]]:
        return [
            detection.bbox
            for detection in self._extract_geometry(frame, threshold)
        ]


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

    def _extract_geometry(
        self, frame: np.ndarray, threshold: float
    ) -> List[DetectionGeometry]:
        from backend.paddle_compat import extract_paddle_geometry
        return extract_paddle_geometry(self._model, frame, threshold)

    def _extract_boxes(
        self, frame: np.ndarray, threshold: float
    ) -> List[Tuple[int, int, int, int]]:
        return [
            detection.bbox
            for detection in self._extract_geometry(frame, threshold)
        ]


def _normalise_vl_server_url(raw: str) -> str:
    value = str(raw or "").strip() or _PADDLEOCR_VL_LLAMA_DEFAULT_URL
    return value.rstrip("/")


class VlmEndpointPolicyError(ValueError):
    """Raised when a llama.cpp endpoint crosses the privacy boundary."""

    failure_class = "policy_blocked"


@dataclass(frozen=True)
class VlmEndpointPolicy:
    """Validated endpoint facts used immediately before network access."""

    url: str
    scheme: str
    host: str
    port: int
    addresses: Tuple[str, ...]
    loopback: bool
    remote: bool
    acknowledged: bool


def _ip_is_loopback(value: str) -> bool:
    """Treat IPv4-mapped loopback addresses as loopback too."""
    address = ipaddress.ip_address(str(value).split("%", 1)[0])
    mapped = getattr(address, "ipv4_mapped", None)
    return bool(mapped.is_loopback if mapped is not None else address.is_loopback)


def _resolve_vlm_endpoint(host: str, port: int) -> Tuple[str, ...]:
    """Resolve every stream address for an endpoint without connecting."""
    try:
        records = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError) as exc:
        raise VlmEndpointPolicyError(
            "The VLM endpoint hostname could not be resolved."
        ) from exc
    addresses = []
    for record in records:
        try:
            address = str(record[4][0]).split("%", 1)[0]
            ipaddress.ip_address(address)
        except (IndexError, TypeError, ValueError):
            continue
        if address not in addresses:
            addresses.append(address)
    if not addresses:
        raise VlmEndpointPolicyError(
            "The VLM endpoint did not resolve to an IP address."
        )
    return tuple(addresses)


def validate_vlm_server_endpoint(
    server_url: str,
    env: Optional[Mapping[str, str]] = None,
) -> VlmEndpointPolicy:
    """Resolve and enforce the local-by-default VLM endpoint policy.

    HTTP is accepted only when every resolved address is loopback. Remote
    endpoints require HTTPS and an explicit acknowledgement that frame pixels
    leave this computer. Callers invoke this again immediately before each
    probe or inference request so a DNS change cannot bypass the decision.
    """
    source_env = os.environ if env is None else env
    value = _normalise_vl_server_url(server_url)
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise VlmEndpointPolicyError("The VLM endpoint URL is malformed.") from exc
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise VlmEndpointPolicyError(
            "The VLM endpoint must use HTTP or HTTPS."
        )
    if parsed.username is not None or parsed.password is not None:
        raise VlmEndpointPolicyError(
            "Credentials are not allowed in the VLM endpoint URL."
        )
    if not parsed.hostname:
        raise VlmEndpointPolicyError("The VLM endpoint URL needs a hostname.")
    if parsed.query or parsed.fragment:
        raise VlmEndpointPolicyError(
            "The VLM endpoint URL cannot contain a query or fragment."
        )
    resolved_port = int(port or (443 if scheme == "https" else 80))
    addresses = _resolve_vlm_endpoint(parsed.hostname, resolved_port)
    loopback = all(_ip_is_loopback(address) for address in addresses)
    remote = not loopback
    acknowledged = _env_truthy(source_env, _PADDLEOCR_VL_REMOTE_ACK_ENV)
    if scheme == "http" and remote:
        raise VlmEndpointPolicyError(
            "Remote VLM endpoints must use HTTPS because video frames leave "
            "this computer."
        )
    if remote and not acknowledged:
        raise VlmEndpointPolicyError(
            "Remote VLM processing is blocked. Set VSR_ALLOW_REMOTE_VLM=1 "
            "only after accepting that video frames leave this computer."
        )
    return VlmEndpointPolicy(
        url=value,
        scheme=scheme,
        host=parsed.hostname,
        port=resolved_port,
        addresses=addresses,
        loopback=loopback,
        remote=remote,
        acknowledged=acknowledged,
    )


def vlm_endpoint_privacy_status(
    env: Optional[Mapping[str, str]] = None,
) -> dict:
    """Return endpoint privacy status without exposing its host or address."""
    source_env = os.environ if env is None else env
    server_url = str(source_env.get(
        "VSR_PADDLEOCR_VL_SERVER_URL",
        _PADDLEOCR_VL_LLAMA_DEFAULT_URL,
    ))
    try:
        policy = validate_vlm_server_endpoint(server_url, source_env)
    except VlmEndpointPolicyError as exc:
        return {
            "schema": "vsr.vlm_endpoint_privacy.v1",
            "allowed": False,
            "remote": None,
            "loopback": False,
            "acknowledged": _env_truthy(
                source_env, _PADDLEOCR_VL_REMOTE_ACK_ENV
            ),
            "transport": "",
            "message": str(exc),
        }
    if policy.remote:
        message = (
            "Remote VLM enabled. Video frames are sent to an HTTPS service "
            "outside this computer."
        )
    else:
        message = (
            "Local VLM endpoint. Frame pixels stay on this computer."
        )
    return {
        "schema": "vsr.vlm_endpoint_privacy.v1",
        "allowed": True,
        "remote": policy.remote,
        "loopback": policy.loopback,
        "acknowledged": policy.acknowledged,
        "transport": policy.scheme,
        "message": message,
    }


def _llama_cpp_models_url(server_url: str) -> str:
    return f"{_normalise_vl_server_url(server_url)}/models"


class _VlmRedirectHandler(urlrequest.HTTPRedirectHandler):
    """Apply the same privacy policy to every probe redirect."""

    def __init__(self, env: Mapping[str, str]):
        super().__init__()
        self.env = env

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urljoin(req.full_url, newurl)
        validate_vlm_server_endpoint(target, self.env)
        return super().redirect_request(
            req, fp, code, msg, headers, target
        )


def _llama_cpp_server_reachable(
    server_url: str,
    timeout: float = 0.75,
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    source_env = os.environ if env is None else env
    try:
        policy = validate_vlm_server_endpoint(server_url, source_env)
    except VlmEndpointPolicyError as exc:
        logger.info(
            "PaddleOCR-VL server endpoint rejected by privacy policy: %s",
            exc,
        )
        return False
    if _env_truthy(source_env, _PADDLEOCR_VL_SKIP_PROBE_ENV):
        return True
    models_url = _llama_cpp_models_url(policy.url)
    try:
        req = urlrequest.Request(models_url, method="GET")
        opener = urlrequest.build_opener(_VlmRedirectHandler(source_env))
        with opener.open(req, timeout=timeout) as response:
            return 200 <= int(getattr(response, "status", 200)) < 500
    except urlerror.HTTPError as exc:
        return 200 <= exc.code < 500
    except (OSError, TimeoutError, ValueError) as exc:
        logger.info(
            "PaddleOCR-VL llama.cpp server unavailable at %s: %s",
            models_url,
            exc,
        )
        return False


def _result_payload(result: Any) -> Any:
    if isinstance(result, (dict, list, tuple)):
        return result
    for attr in ("json", "to_json", "as_dict", "dict"):
        data = getattr(result, attr, None)
        if data is None:
            continue
        if callable(data):
            try:
                data = data()
            except TypeError:
                continue
        if isinstance(data, str):
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                continue
        return data
    return None


def _box_from_coords(coords: Any, frame_shape: Tuple[int, int, int]) -> Optional[Box]:
    h, w = frame_shape[:2]
    try:
        pts = np.array(coords, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if pts.size < 4:
        return None
    if pts.ndim == 1 and pts.size == 4:
        x1, y1, x2, y2 = [float(v) for v in pts[:4]]
    else:
        try:
            pts = pts.reshape(-1, 2)
        except ValueError:
            return None
        if pts.shape[1] < 2:
            return None
        x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
        x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
    x1 = max(0, min(w, int(round(x1))))
    y1 = max(0, min(h, int(round(y1))))
    x2 = max(0, min(w, int(round(x2))))
    y2 = max(0, min(h, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _score_at(scores: Any, idx: int, fallback: float = 1.0) -> float:
    try:
        if scores is not None and not isinstance(scores, dict):
            return float(scores[idx])
    except (IndexError, KeyError, TypeError, ValueError):
        try:
            return float(scores)
        except (TypeError, ValueError):
            pass
    except (AttributeError, OverflowError):
        pass
    return fallback


def _score_from_dict(data: Mapping[str, Any]) -> float:
    for key in ("score", "confidence", "rec_score", "layout_score"):
        if key in data:
            try:
                return float(data[key])
            except (TypeError, ValueError):
                return 1.0
    return 1.0


def _collect_vl_geometry(
    payload: Any,
    threshold: float,
    frame_shape: Tuple[int, int, int],
) -> List[DetectionGeometry]:
    detections: List[DetectionGeometry] = []
    if isinstance(payload, dict):
        data = payload.get("res", payload)
        handled_keys = {
            "bbox", "box", "rect", "coordinate", "poly", "polygon",
            "rec_boxes", "dt_boxes", "boxes", "bboxes",
            "rec_polys", "dt_polys", "polys", "quad_boxes", "points",
            "rec_scores", "scores", "layout_scores", "confidence",
            "score", "rec_score", "layout_score",
        }
        score = _score_from_dict(data)
        for key in ("bbox", "box", "rect", "coordinate", "poly", "polygon"):
            if key in data and score >= threshold:
                is_polygon = key in {"poly", "polygon"}
                detection = (
                    DetectionGeometry.from_polygon(
                        data[key], frame_shape, confidence=score)
                    if is_polygon else DetectionGeometry.from_box(
                        data[key], frame_shape, confidence=score)
                )
                if detection is None and not is_polygon:
                    detection = DetectionGeometry.from_polygon(
                        data[key], frame_shape, confidence=score)
                if detection is not None:
                    detections.append(detection)
        scores = None
        for score_key in (
            "rec_scores", "scores", "layout_scores", "confidence"):
            if score_key in data and data[score_key] is not None:
                scores = data[score_key]
                break
        for key in ("rec_boxes", "dt_boxes", "boxes", "bboxes"):
            values = data.get(key)
            if isinstance(values, (list, tuple, np.ndarray)):
                for idx, value in enumerate(values):
                    if _score_at(scores, idx) < threshold:
                        continue
                    detection = DetectionGeometry.from_box(
                        value,
                        frame_shape,
                        confidence=_score_at(scores, idx),
                    )
                    if detection is not None:
                        detections.append(detection)
        for key in ("rec_polys", "dt_polys", "polys", "quad_boxes", "points"):
            values = data.get(key)
            if isinstance(values, (list, tuple, np.ndarray)):
                for idx, value in enumerate(values):
                    if _score_at(scores, idx) < threshold:
                        continue
                    score_value = _score_at(scores, idx)
                    detection = DetectionGeometry.from_polygon(
                        value,
                        frame_shape,
                        confidence=score_value,
                    )
                    if detection is None:
                        detection = DetectionGeometry.from_box(
                            value,
                            frame_shape,
                            confidence=score_value,
                        )
                    if detection is not None:
                        detections.append(detection)
        for key, value in data.items():
            if key in handled_keys:
                continue
            if isinstance(value, (dict, list, tuple)):
                detections.extend(
                    _collect_vl_geometry(value, threshold, frame_shape))
        return detections
    if isinstance(payload, (list, tuple, np.ndarray)):
        try:
            array = np.asarray(payload)
            if array.ndim == 2 and array.shape[1] == 4 and array.shape[0] > 1:
                for row in array:
                    detection = DetectionGeometry.from_box(
                        row.tolist(), frame_shape, confidence=1.0)
                    if detection is not None:
                        detections.append(detection)
                return detections
        except (TypeError, ValueError):
            pass
        detection = DetectionGeometry.from_polygon(payload, frame_shape)
        if detection is None:
            detection = DetectionGeometry.from_box(payload, frame_shape)
        if detection is not None:
            return [detection]
        for value in payload:
            detections.extend(
                _collect_vl_geometry(value, threshold, frame_shape))
    return detections


def _collect_vl_boxes(payload: Any, threshold: float,
                      frame_shape: Tuple[int, int, int]) -> List[Box]:
    """Compatibility wrapper that intentionally drops polygon vertices."""
    return [
        detection.bbox
        for detection in _collect_vl_geometry(payload, threshold, frame_shape)
    ]


def _dedupe_geometry(
    detections: List[DetectionGeometry],
) -> List[DetectionGeometry]:
    out: List[DetectionGeometry] = []
    seen = set()
    for detection in detections:
        key = (detection.bbox, detection.polygon)
        if key in seen:
            continue
        seen.add(key)
        out.append(detection)
    return out


def _dedupe_boxes(boxes: List[Box]) -> List[Box]:
    out: List[Box] = []
    seen = set()
    for box in boxes:
        if box in seen:
            continue
        seen.add(box)
        out.append(box)
    return out


def _vl_result_sequence(results: Any) -> List[Any]:
    if results is None:
        return []
    if isinstance(results, (dict, str, bytes)):
        return [results]
    try:
        return list(results)
    except TypeError:
        return [results]


class _PaddleOcrVlLlamaCppDetector(_BaseVlmDetector):
    """PaddleOCR-VL-1.5 through a local llama.cpp OpenAI-compatible
    server. The Python process only runs the document parser and talks
    to the local CPU/edge server, so this tier does not require CUDA.
    """

    name = "paddleocr-vl-1.5-llama.cpp"

    def __init__(self, device: str = "cpu", env: Optional[Mapping[str, str]] = None):
        super().__init__(device="cpu")
        self.env = os.environ if env is None else env
        self.server_url = _normalise_vl_server_url(
            str(self.env.get(
                "VSR_PADDLEOCR_VL_SERVER_URL",
                _PADDLEOCR_VL_LLAMA_DEFAULT_URL,
            ))
        )

    def _warm_load(self) -> bool:
        self._model = self._load()
        self._loaded = True
        return self._model is not None

    def _load(self):
        if not _llama_cpp_server_reachable(self.server_url, env=self.env):
            logger.info(
                "PaddleOCR-VL-1.5 llama.cpp detector disabled; start "
                "llama-server and expose %s before setting VSR_PADDLEOCR_VL=1.",
                self.server_url,
            )
            return None
        try:
            from paddleocr import PaddleOCRVL  # type: ignore
        except ImportError:
            logger.info(
                "PaddleOCR-VL-1.5 requires PaddleOCR with PaddleOCRVL; "
                "install a compatible paddleocr package."
            )
            return None
        except Exception as exc:
            logger.warning("PaddleOCR-VL import failed: %s", exc)
            return None
        kwargs = {
            "vl_rec_backend": "llama-cpp-server",
            "vl_rec_server_url": self.server_url,
        }
        try:
            return PaddleOCRVL(**kwargs)
        except TypeError:
            logger.warning(
                "Installed PaddleOCRVL does not expose llama.cpp server "
                "kwargs; falling back to the default OCR cascade."
            )
            return None
        except Exception as exc:
            logger.warning("PaddleOCR-VL-1.5 llama.cpp load failed: %s", exc)
            return None

    def _extract_geometry(
        self, frame: np.ndarray, threshold: float
    ) -> List[DetectionGeometry]:
        try:
            validate_vlm_server_endpoint(self.server_url, self.env)
        except VlmEndpointPolicyError as exc:
            logger.warning(
                "PaddleOCR-VL frame request blocked by privacy policy: %s",
                exc,
            )
            raise
        fd, path = tempfile.mkstemp(prefix="vsr_paddleocr_vl_", suffix=".png")
        os.close(fd)
        try:
            if not safe_imwrite(path, frame):
                raise RuntimeError(
                    "could not stage the PaddleOCR-VL inference frame"
                )
            if hasattr(self._model, "predict"):
                results = self._model.predict(path)
            else:
                results = self._model(path)
        finally:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass
        payloads = [_result_payload(result) for result in _vl_result_sequence(results)]
        detections: List[DetectionGeometry] = []
        for payload in payloads:
            detections.extend(
                _collect_vl_geometry(payload, threshold, frame.shape))
        return _dedupe_geometry(detections)

    def _extract_boxes(self, frame: np.ndarray, threshold: float) -> List[Box]:
        return [
            detection.bbox
            for detection in self._extract_geometry(frame, threshold)
        ]


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

    def _extract_geometry(
        self, frame: np.ndarray, threshold: float
    ) -> List[DetectionGeometry]:
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
            detection = DetectionGeometry.from_box(
                (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                frame.shape,
            )
            return [detection] if detection is not None else []
        try:
            polys = ctd.predict_one(frame)  # type: ignore[attr-defined]
        except Exception as exc:
            logger.warning(f"comic-text-detector inference failed: {exc}")
            raise
        out: List[DetectionGeometry] = []
        for poly in polys:
            detection = DetectionGeometry.from_polygon(poly, frame.shape)
            if detection is not None:
                out.append(detection)
        return out

    def _extract_boxes(
        self, frame: np.ndarray, threshold: float
    ) -> List[Tuple[int, int, int, int]]:
        return [
            detection.bbox
            for detection in self._extract_geometry(frame, threshold)
        ]


# The four names the GUI engine picker and --ocr-engine accept with a
# "vlm-" prefix. Order matters only for documentation.
VLM_ENGINE_NAMES = (
    "florence2", "qwen25vl", "paddleocr-vl", "paddleocr-vl-llama")


def build_named_vlm_detector(name: str, device: str) -> Optional[object]:
    """Build one specific VLM detector, or None for an unknown name.

    Shared by the env-var opt-in and the explicit ``vlm-<name>`` engine
    selection, so both go through the same llama.cpp warm-load gate.
    """
    if name == "florence2":
        return _Florence2Detector(device=device)
    if name == "qwen25vl":
        return _Qwen25VLDetector(device=device)
    if name == "paddleocr-vl":
        return _PaddleOcrVlDetector(device=device)
    if name == "paddleocr-vl-llama":
        detector = _PaddleOcrVlLlamaCppDetector(device=device)
        if detector._warm_load():
            return detector
        return None
    return None


def maybe_build_vlm_detector(device: str, lang: str) -> Optional[object]:
    """Return a VLM detector instance when the user opted in via env var
    or via a special lang token ("manga"). None means "leave the default
    cascade alone"."""
    if lang and lang.strip().lower() == "manga":
        return _MangaOcrDetector(device=device)
    selected = selected_vlm_backend()
    if selected is None:
        return None
    return build_named_vlm_detector(selected, device)
