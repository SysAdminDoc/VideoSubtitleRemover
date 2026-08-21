import http.server
import socket
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from urllib import request as urlrequest

import cv2
import numpy as np

from backend import ocr_vlm


def _address_record(address: str):
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    sockaddr = (address, 0, 0, 0) if family == socket.AF_INET6 else (address, 0)
    return (family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", sockaddr)


class _ServerContext:
    def __init__(self, handler):
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            daemon=True,
        )

    def __enter__(self):
        self.thread.start()
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    def __exit__(self, exc_type, exc, traceback):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)
        return False


class VlmEndpointPrivacyTests(unittest.TestCase):
    def test_http_accepts_only_loopback_and_mapped_loopback(self):
        records = {
            "127.0.0.1": [_address_record("127.0.0.1")],
            "::1": [_address_record("::1")],
            "localhost": [
                _address_record("127.0.0.1"),
                _address_record("::1"),
            ],
            "::ffff:127.0.0.1": [
                _address_record("::ffff:127.0.0.1")
            ],
        }

        with mock.patch.object(
            ocr_vlm.socket,
            "getaddrinfo",
            side_effect=lambda host, *_args, **_kwargs: records[host],
        ):
            for url in (
                "http://127.0.0.1:8080/v1",
                "http://[::1]:8080/v1",
                "http://localhost:8080/v1",
                "http://[::ffff:127.0.0.1]:8080/v1",
            ):
                with self.subTest(url=url):
                    policy = ocr_vlm.validate_vlm_server_endpoint(url, {})
                    self.assertTrue(policy.loopback)
                    self.assertFalse(policy.remote)

    def test_remote_and_mapped_remote_require_https_and_acknowledgement(self):
        remote_records = [_address_record("203.0.113.8")]
        mapped_records = [_address_record("::ffff:192.0.2.9")]

        def resolve(host, *_args, **_kwargs):
            return mapped_records if ":" in host else remote_records

        with mock.patch.object(
            ocr_vlm.socket, "getaddrinfo", side_effect=resolve
        ):
            for host in ("203.0.113.8", "[::ffff:192.0.2.9]"):
                with self.subTest(host=host, transport="http"):
                    with self.assertRaisesRegex(
                        ocr_vlm.VlmEndpointPolicyError,
                        "must use HTTPS",
                    ):
                        ocr_vlm.validate_vlm_server_endpoint(
                            f"http://{host}:8080/v1",
                            {"VSR_ALLOW_REMOTE_VLM": "1"},
                        )
                with self.subTest(host=host, acknowledged=False):
                    with self.assertRaisesRegex(
                        ocr_vlm.VlmEndpointPolicyError,
                        "VSR_ALLOW_REMOTE_VLM",
                    ):
                        ocr_vlm.validate_vlm_server_endpoint(
                            f"https://{host}/v1",
                            {},
                        )
                with self.subTest(host=host, acknowledged=True):
                    policy = ocr_vlm.validate_vlm_server_endpoint(
                        f"https://{host}/v1",
                        {"VSR_ALLOW_REMOTE_VLM": "1"},
                    )
                    self.assertTrue(policy.remote)
                    self.assertTrue(policy.acknowledged)

    def test_malformed_credentialed_and_unsupported_urls_are_rejected(self):
        cases = (
            "file:///tmp/model",
            "ftp://127.0.0.1/model",
            "http://user:secret@127.0.0.1:8080/v1",
            "http:///v1",
            "http://127.0.0.1:bad/v1",
            "http://127.0.0.1:8080/v1?token=secret",
            "http://127.0.0.1:8080/v1#fragment",
            "not-a-url",
        )
        with mock.patch.object(
            ocr_vlm.socket,
            "getaddrinfo",
            return_value=[_address_record("127.0.0.1")],
        ):
            for url in cases:
                with self.subTest(url=url):
                    with self.assertRaises(ocr_vlm.VlmEndpointPolicyError):
                        ocr_vlm.validate_vlm_server_endpoint(url, {})

    def test_dns_is_resolved_again_and_rebinding_blocks_http(self):
        with mock.patch.object(
            ocr_vlm.socket,
            "getaddrinfo",
            side_effect=[
                [_address_record("127.0.0.1")],
                [_address_record("203.0.113.9")],
            ],
        ) as resolver:
            first = ocr_vlm.validate_vlm_server_endpoint(
                "http://vlm.internal:8080/v1",
                {},
            )
            self.assertTrue(first.loopback)
            with self.assertRaisesRegex(
                ocr_vlm.VlmEndpointPolicyError,
                "must use HTTPS",
            ):
                ocr_vlm.validate_vlm_server_endpoint(
                    "http://vlm.internal:8080/v1",
                    {},
                )
        self.assertEqual(resolver.call_count, 2)

    def test_probe_rejects_redirect_outside_approved_boundary(self):
        requests = []

        class RedirectHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append((self.command, self.path))
                self.send_response(302)
                self.send_header(
                    "Location",
                    "http://203.0.113.10/v1/models",
                )
                self.end_headers()

            def log_message(self, _format, *_args):
                pass

        with _ServerContext(RedirectHandler) as endpoint:
            self.assertFalse(ocr_vlm._llama_cpp_server_reachable(
                f"{endpoint}/v1",
                timeout=2.0,
                env={},
            ))

        self.assertEqual(requests, [("GET", "/v1/models")])

    def test_probe_bypass_never_bypasses_endpoint_policy(self):
        skip = {"VSR_PADDLEOCR_VL_SKIP_SERVER_PROBE": "1"}
        with mock.patch.object(
            ocr_vlm.socket,
            "getaddrinfo",
            return_value=[_address_record("203.0.113.11")],
        ), mock.patch.object(ocr_vlm.urlrequest, "build_opener") as opener:
            self.assertFalse(ocr_vlm._llama_cpp_server_reachable(
                "http://vlm.example/v1",
                env=skip,
            ))
            opener.assert_not_called()

        with mock.patch.object(
            ocr_vlm.socket,
            "getaddrinfo",
            return_value=[_address_record("127.0.0.1")],
        ), mock.patch.object(ocr_vlm.urlrequest, "build_opener") as opener:
            self.assertTrue(ocr_vlm._llama_cpp_server_reachable(
                "http://127.0.0.1:8080/v1",
                env=skip,
            ))
            opener.assert_not_called()

    def test_inference_rechecks_dns_before_model_request(self):
        detector = ocr_vlm._PaddleOcrVlLlamaCppDetector(
            env={"VSR_PADDLEOCR_VL_SERVER_URL": "http://vlm.internal/v1"}
        )
        model = mock.Mock()
        detector._model = model
        frame = np.zeros((8, 8, 3), dtype=np.uint8)

        with mock.patch.object(
            ocr_vlm.socket,
            "getaddrinfo",
            side_effect=[
                [_address_record("127.0.0.1")],
                [_address_record("203.0.113.12")],
            ],
        ):
            initial = ocr_vlm.validate_vlm_server_endpoint(
                detector.server_url,
                detector.env,
            )
            self.assertTrue(initial.loopback)
            self.assertEqual(detector._extract_boxes(frame, 0.5), [])

        model.predict.assert_not_called()

    def test_captured_frame_request_is_a_full_png_payload(self):
        captured = {}

        class CaptureHandler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                captured["method"] = self.command
                captured["path"] = self.path
                captured["content_type"] = self.headers.get("Content-Type")
                captured["body"] = self.rfile.read(length)
                self.send_response(200)
                self.end_headers()

            def log_message(self, _format, *_args):
                pass

        with _ServerContext(CaptureHandler) as endpoint:
            detector = ocr_vlm._PaddleOcrVlLlamaCppDetector(
                env={"VSR_PADDLEOCR_VL_SERVER_URL": f"{endpoint}/v1"}
            )

            class CapturingPaddleClient:
                def predict(self, path):
                    payload = Path(path).read_bytes()
                    request = urlrequest.Request(
                        f"{endpoint}/frame",
                        data=payload,
                        headers={"Content-Type": "image/png"},
                        method="POST",
                    )
                    with urlrequest.urlopen(request, timeout=2.0):
                        pass
                    return []

            detector._model = CapturingPaddleClient()
            frame = np.zeros((9, 11, 3), dtype=np.uint8)
            frame[:, :, 0] = 17
            frame[:, :, 1] = 83
            frame[:, :, 2] = 191
            self.assertEqual(detector._extract_boxes(frame, 0.5), [])

        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["path"], "/frame")
        self.assertEqual(captured["content_type"], "image/png")
        self.assertTrue(captured["body"].startswith(b"\x89PNG\r\n\x1a\n"))
        decoded = cv2.imdecode(
            np.frombuffer(captured["body"], dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        np.testing.assert_array_equal(decoded, frame)

    def test_privacy_status_warns_without_disclosing_endpoint(self):
        env = {
            "VSR_PADDLEOCR_VL_SERVER_URL": "https://vlm.example/v1",
            "VSR_ALLOW_REMOTE_VLM": "1",
        }
        with mock.patch.object(
            ocr_vlm.socket,
            "getaddrinfo",
            return_value=[_address_record("203.0.113.13")],
        ):
            status = ocr_vlm.vlm_endpoint_privacy_status(env)

        self.assertTrue(status["allowed"])
        self.assertTrue(status["remote"])
        self.assertIn("Video frames are sent", status["message"])
        self.assertNotIn("vlm.example", str(status))

    def test_gui_notice_keeps_remote_frame_transfer_visible(self):
        from gui.settings_controller import AdvancedSettingsControllerMixin
        from gui.theme import Theme

        updates = {}

        class Label:
            def config(self, **kwargs):
                updates.update(kwargs)

        host = SimpleNamespace(
            config=SimpleNamespace(
                detection_engine="vlm-paddleocr-vl-llama"
            ),
            vlm_privacy_label=Label(),
            backend_status={
                "vlm_endpoint_privacy": {
                    "allowed": True,
                    "remote": True,
                    "message": "remote",
                }
            },
        )

        AdvancedSettingsControllerMixin._refresh_vlm_privacy_notice(host)

        self.assertIn("Video frames are sent", updates["text"])
        self.assertEqual(updates["fg"], Theme.WARNING)


if __name__ == "__main__":
    unittest.main()
