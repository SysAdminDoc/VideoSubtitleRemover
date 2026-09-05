"""RM-354: the optional LaMa weights must be obtainable, and only verified.

Every test here drives ``backend.model_fetch.fetch_weight`` against a real
local HTTP server over a real socket, so the download, the streaming hash, the
partial-file discipline and the atomic rename are exercised rather than
described. The served bytes are a fixture, so the fixture registers its own
manifest entry with its own true digest; the shipped ``lama-onnx`` and
``opencv-lama`` pins are checked separately by
``ShippedLamaPinTests``, which is what stops the real pins rotting.
"""

from __future__ import annotations

import hashlib
import http.server
import json
import os
import socket
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from backend import model_fetch
from backend.adapter_manifest import (
    ADAPTER_MANIFEST,
    AdapterManifestEntry,
    adapter_provenance_path,
    get_manifest_entry,
)

_FIXTURE_NAME = "vsr_test_weight.onnx"
_FIXTURE_ADAPTER = "vsr-test-weight"
_FIXTURE_BODY = b"onnx-shaped fixture payload " * 4096


class _ServerContext:
    """Serve one payload from 127.0.0.1 on an ephemeral port."""

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


def _body_handler(body: bytes, *, chunk: int = 8192):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):  # noqa: N802 - stdlib callback name
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            for start in range(0, len(body), chunk):
                self.wfile.write(body[start:start + chunk])

        def log_message(self, *_args):
            return

    return Handler


def _missing_handler():
    class Handler(http.server.BaseHTTPRequestHandler):
        # HTTP/1.0 so the error response closes the connection instead of
        # leaving urllib waiting out a keep-alive it will never get more on.
        protocol_version = "HTTP/1.0"

        def do_GET(self):  # noqa: N802 - stdlib callback name
            self.send_error(404, "not here")

        def log_message(self, *_args):
            return

    return Handler


class _FixtureManifest:
    """Register a real manifest entry for the fixture payload.

    The entry is genuine manifest data, not a stubbed collaborator: the code
    under test reads it through ``get_manifest_entry`` exactly as it reads the
    shipped adapters.
    """

    def __init__(self, body: bytes, filename: str = _FIXTURE_NAME):
        self.entry = AdapterManifestEntry(
            name=_FIXTURE_ADAPTER,
            env_vars=(),
            expected_filenames=(filename,),
            sha256={filename: hashlib.sha256(body).hexdigest()},
            license="test-fixture",
            source_url="http://127.0.0.1/fixture",
            preferred_format="ONNX",
            repository="vsr/test-weight",
            revision="0" * 40,
        )

    def __enter__(self):
        ADAPTER_MANIFEST[_FIXTURE_ADAPTER] = self.entry
        model_fetch.FETCHABLE_WEIGHTS = model_fetch.FETCHABLE_WEIGHTS + (
            (_FIXTURE_ADAPTER, self.entry.expected_filenames[0]),
        )
        return self.entry

    def __exit__(self, exc_type, exc, traceback):
        ADAPTER_MANIFEST.pop(_FIXTURE_ADAPTER, None)
        model_fetch.FETCHABLE_WEIGHTS = tuple(
            item for item in model_fetch.FETCHABLE_WEIGHTS
            if item[0] != _FIXTURE_ADAPTER
        )
        return False


class FetchWeightTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="vsr-fetch-")
        self.addCleanup(self._tmp.cleanup)
        self.appdata = Path(self._tmp.name) / "AppData"
        self.env = {"APPDATA": str(self.appdata)}
        self.cache = self.appdata / "VideoSubtitleRemoverPro" / "models"

    def test_empty_cache_downloads_verifies_and_lands_where_lama_looks(self):
        with _FixtureManifest(_FIXTURE_BODY) as entry:
            with _ServerContext(_body_handler(_FIXTURE_BODY)) as endpoint:
                self.assertFalse(self.cache.exists())
                seen = []
                result = model_fetch.fetch_weight(
                    _FIXTURE_ADAPTER,
                    env=self.env,
                    endpoint=endpoint,
                    progress=lambda read, total: seen.append((read, total)),
                )

        self.assertTrue(result.ok, result.detail)
        self.assertEqual(result.reason, "downloaded")
        self.assertEqual(result.sha256, entry.sha256[_FIXTURE_NAME])
        self.assertEqual(result.bytes_read, len(_FIXTURE_BODY))

        landed = self.cache / _FIXTURE_NAME
        self.assertEqual(Path(result.path), landed)
        self.assertEqual(landed.read_bytes(), _FIXTURE_BODY)
        self.assertEqual(
            hashlib.sha256(landed.read_bytes()).hexdigest(),
            entry.sha256[_FIXTURE_NAME],
        )

        # The download is worthless unless the inpainter's own search finds
        # it. lama.py resolves %APPDATA%/VideoSubtitleRemoverPro/models first.
        from backend.inpainters import lama as lama_module

        with mock.patch.dict(os.environ, {"APPDATA": str(self.appdata)},
                             clear=False):
            os.environ.pop("VSR_OPENCV_LAMA", None)
            found = lama_module._find_opencv_lama_weight()
        expected_dir = model_fetch.model_cache_dir(self.env)
        self.assertEqual(expected_dir, self.cache)
        self.assertIsNone(
            found,
            "the fixture is not a real LaMa filename, so discovery must not "
            "claim it; the directory match is asserted above instead",
        )

        self.assertGreater(len(seen), 1, "progress reported only once")
        self.assertEqual(seen[0][0], 0)
        self.assertEqual(seen[-1], (len(_FIXTURE_BODY), len(_FIXTURE_BODY)))
        self.assertEqual(
            [read for read, _ in seen],
            sorted(read for read, _ in seen),
            "progress must be monotonic",
        )

    def test_a_real_lama_filename_is_discovered_after_it_is_fetched(self):
        body = b"opencv lama fixture " * 2048
        with _FixtureManifest(body, filename="inpainting_lama_2025jan.onnx"):
            with _ServerContext(_body_handler(body)) as endpoint:
                result = model_fetch.fetch_weight(
                    _FIXTURE_ADAPTER,
                    "inpainting_lama_2025jan.onnx",
                    env=self.env,
                    endpoint=endpoint,
                )
        self.assertTrue(result.ok, result.detail)

        from backend.inpainters import lama as lama_module

        with mock.patch.dict(os.environ, {"APPDATA": str(self.appdata)},
                             clear=False):
            os.environ.pop("VSR_OPENCV_LAMA", None)
            found = lama_module._find_opencv_lama_weight()
        self.assertEqual(
            Path(found or ""),
            self.cache / "inpainting_lama_2025jan.onnx",
            "a fetched weight the inpainter cannot find is not a fix",
        )

    def test_wrong_bytes_are_rejected_and_nothing_is_left_behind(self):
        with _FixtureManifest(_FIXTURE_BODY):
            with _ServerContext(_body_handler(b"substituted payload")) as url:
                result = model_fetch.fetch_weight(
                    _FIXTURE_ADAPTER, env=self.env, endpoint=url,
                )

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "hash_mismatch")
        self.assertNotEqual(result.sha256, result.expected_sha256)
        self.assertFalse((self.cache / _FIXTURE_NAME).exists())
        self.assertFalse((self.cache / f"{_FIXTURE_NAME}.part").exists())

    def test_cancelling_removes_the_partial_file(self):
        calls = []

        def _cancel() -> bool:
            calls.append(1)
            return len(calls) > 1

        with _FixtureManifest(_FIXTURE_BODY):
            with _ServerContext(_body_handler(_FIXTURE_BODY)) as url:
                result = model_fetch.fetch_weight(
                    _FIXTURE_ADAPTER, env=self.env, endpoint=url,
                    cancel=_cancel,
                )

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "cancelled")
        self.assertFalse((self.cache / _FIXTURE_NAME).exists())
        self.assertFalse((self.cache / f"{_FIXTURE_NAME}.part").exists())

    def test_offline_reports_unreachable_without_a_partial_file(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            dead_port = probe.getsockname()[1]

        with _FixtureManifest(_FIXTURE_BODY):
            result = model_fetch.fetch_weight(
                _FIXTURE_ADAPTER, env=self.env,
                endpoint=f"http://127.0.0.1:{dead_port}",
            )

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "unreachable")
        self.assertFalse((self.cache / f"{_FIXTURE_NAME}.part").exists())

    def test_a_missing_remote_file_is_an_http_error_not_a_silent_pass(self):
        with _FixtureManifest(_FIXTURE_BODY):
            with _ServerContext(_missing_handler()) as url:
                result = model_fetch.fetch_weight(
                    _FIXTURE_ADAPTER, env=self.env, endpoint=url,
                )

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "http_error")
        self.assertIn("404", result.detail)
        self.assertFalse((self.cache / f"{_FIXTURE_NAME}.part").exists())

    def test_a_verified_file_already_in_the_cache_is_not_downloaded_again(self):
        with _FixtureManifest(_FIXTURE_BODY):
            self.cache.mkdir(parents=True, exist_ok=True)
            (self.cache / _FIXTURE_NAME).write_bytes(_FIXTURE_BODY)
            # No server is running, so any network access fails the test.
            result = model_fetch.fetch_weight(
                _FIXTURE_ADAPTER, env=self.env,
                endpoint="https://model-fetch.invalid",
            )

        self.assertTrue(result.ok, result.detail)
        self.assertEqual(result.reason, "already_present")

    def test_provenance_records_the_commit_and_endpoint_that_served_it(self):
        with _FixtureManifest(_FIXTURE_BODY) as entry:
            with _ServerContext(_body_handler(_FIXTURE_BODY)) as endpoint:
                result = model_fetch.fetch_weight(
                    _FIXTURE_ADAPTER, env=self.env, endpoint=endpoint,
                )
                self.assertTrue(result.ok, result.detail)
                path = adapter_provenance_path(_FIXTURE_ADAPTER, self.env)
                payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["adapter"], _FIXTURE_ADAPTER)
        self.assertEqual(payload["commit"], entry.revision)
        self.assertEqual(payload["repository"], entry.repository)
        self.assertEqual(payload["source"], "fetch_weight")
        self.assertEqual(payload["endpoint"], endpoint)


class FetchInterruptTests(unittest.TestCase):
    """Ctrl+C is the cancellation route the CLI offers, and it is not Exception."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="vsr-fetch-int-")
        self.addCleanup(self._tmp.cleanup)
        self.appdata = Path(self._tmp.name) / "AppData"
        self.env = {"APPDATA": str(self.appdata)}
        self.cache = self.appdata / "VideoSubtitleRemoverPro" / "models"

    def test_a_keyboard_interrupt_still_removes_the_partial_file(self):
        calls = []

        def _cancel() -> bool:
            calls.append(1)
            if len(calls) > 1:
                raise KeyboardInterrupt()
            return False

        with _FixtureManifest(_FIXTURE_BODY):
            with _ServerContext(_body_handler(_FIXTURE_BODY)) as url:
                with self.assertRaises(KeyboardInterrupt):
                    model_fetch.fetch_weight(
                        _FIXTURE_ADAPTER, env=self.env, endpoint=url,
                        cancel=_cancel,
                    )

        self.assertFalse(
            (self.cache / f"{_FIXTURE_NAME}.part").exists(),
            "an interrupted download must not leave a truncated model behind",
        )
        self.assertFalse((self.cache / _FIXTURE_NAME).exists())


class RedirectPolicyTests(unittest.TestCase):
    """The transport rule has to hold for every hop, not just the first."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="vsr-fetch-redir-")
        self.addCleanup(self._tmp.cleanup)
        self.appdata = Path(self._tmp.name) / "AppData"
        self.env = {"APPDATA": str(self.appdata)}
        self.cache = self.appdata / "VideoSubtitleRemoverPro" / "models"

    @staticmethod
    def _redirect_handler(target: str):
        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def do_GET(self):  # noqa: N802 - stdlib callback name
                self.send_response(302)
                self.send_header("Location", target)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *_args):
                return

        return Handler

    def test_a_redirect_to_plain_http_off_the_loopback_is_refused(self):
        with _FixtureManifest(_FIXTURE_BODY):
            handler = self._redirect_handler(
                "http://models.example.com/weight.onnx")
            with _ServerContext(handler) as url:
                result = model_fetch.fetch_weight(
                    _FIXTURE_ADAPTER, env=self.env, endpoint=url,
                )
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "unreachable")
        self.assertIn("refusing redirect", result.detail)
        self.assertFalse((self.cache / f"{_FIXTURE_NAME}.part").exists())

    def test_a_redirect_that_keeps_the_transport_is_followed(self):
        # Hugging Face redirects a resolve URL to its CDN, so refusing every
        # redirect would break the real fetch. Loopback http is the allowed
        # case here for the same reason the endpoint check allows it.
        with _FixtureManifest(_FIXTURE_BODY):
            with _ServerContext(_body_handler(_FIXTURE_BODY)) as content_url:
                handler = self._redirect_handler(
                    f"{content_url}/redirected.bin")
                with _ServerContext(handler) as entry_url:
                    result = model_fetch.fetch_weight(
                        _FIXTURE_ADAPTER, env=self.env, endpoint=entry_url,
                    )
        self.assertTrue(result.ok, result.detail)
        self.assertEqual(result.reason, "downloaded")


class FetchPolicyTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="vsr-fetch-policy-")
        self.addCleanup(self._tmp.cleanup)
        self.env = {"APPDATA": str(Path(self._tmp.name) / "AppData")}

    def test_a_plain_http_endpoint_off_the_loopback_is_refused(self):
        with _FixtureManifest(_FIXTURE_BODY):
            result = model_fetch.fetch_weight(
                _FIXTURE_ADAPTER, env=self.env,
                endpoint="http://models.example.com",
            )
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "bad_endpoint")

    def test_an_unknown_adapter_is_reported_rather_than_raised(self):
        result = model_fetch.fetch_weight("no-such-adapter", env=self.env)
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "unknown_adapter")

    def test_a_moving_reference_is_refused(self):
        entry = AdapterManifestEntry(
            name="vsr-unpinned",
            env_vars=(),
            expected_filenames=("weight.onnx",),
            sha256={"weight.onnx": "0" * 64},
            repository="vsr/unpinned",
            revision="main",
        )
        ADAPTER_MANIFEST["vsr-unpinned"] = entry
        self.addCleanup(ADAPTER_MANIFEST.pop, "vsr-unpinned", None)
        result = model_fetch.fetch_weight(
            "vsr-unpinned", "weight.onnx", env=self.env,
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "not_pinned")
        self.assertIn("immutable commit", result.detail)

    def test_a_filename_outside_the_manifest_is_refused(self):
        result = model_fetch.fetch_weight(
            "opencv-lama", "something_else.onnx", env=self.env,
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "not_pinned")

    def test_the_endpoint_env_var_overrides_the_default(self):
        self.assertEqual(model_fetch.default_endpoint({}),
                         model_fetch.DEFAULT_ENDPOINT)
        self.assertEqual(
            model_fetch.default_endpoint(
                {model_fetch.ENDPOINT_ENV_VAR: "https://mirror.example/"}
            ),
            "https://mirror.example",
        )


class ShippedLamaPinTests(unittest.TestCase):
    """The pins users actually download from must stay complete and immutable."""

    def test_every_fetchable_weight_is_pinned_to_a_commit_and_a_digest(self):
        self.assertTrue(model_fetch.FETCHABLE_WEIGHTS)
        for adapter, filename in model_fetch.FETCHABLE_WEIGHTS:
            with self.subTest(adapter=adapter, filename=filename):
                entry = get_manifest_entry(adapter)
                self.assertIn(filename, entry.expected_filenames)
                self.assertTrue(entry.repository)
                self.assertEqual(
                    len(entry.revision), 40,
                    "a fetch must name an immutable commit, not a branch",
                )
                digest = entry.sha256.get(filename, "")
                self.assertEqual(len(digest), 64)
                self.assertEqual(digest, digest.lower())

    def test_the_pinned_digests_match_the_vendored_weight_hashes(self):
        from backend.model_hashes import KNOWN_WEIGHT_HASHES

        for adapter, filename in model_fetch.FETCHABLE_WEIGHTS:
            with self.subTest(adapter=adapter, filename=filename):
                entry = get_manifest_entry(adapter)
                self.assertEqual(
                    entry.sha256[filename],
                    KNOWN_WEIGHT_HASHES[filename],
                    "the manifest and the vendored hash table disagree, so "
                    "one of them would reject a good download",
                )

    @unittest.skipUnless(
        os.environ.get("VSR_MODEL_FETCH_TESTS", "").strip().lower()
        in {"1", "true", "yes", "on"},
        "opt-in: reaches Hugging Face. Set VSR_MODEL_FETCH_TESTS=1.",
    )
    def test_the_pins_still_match_what_upstream_serves(self):
        """Catch a wrong or rotated pin without downloading 300 MB.

        The other pin test only proves the manifest and the vendored hash
        table agree with each other, and both are hand-maintained here, so it
        cannot see a digest that was wrong when it was written or a file the
        upstream repository has since replaced. Hugging Face returns the LFS
        digest in X-Linked-Etag on a HEAD, so the real answer costs one
        request per file rather than the whole download.
        """
        import urllib.error
        import urllib.request

        for adapter, filename in model_fetch.FETCHABLE_WEIGHTS:
            with self.subTest(adapter=adapter, filename=filename):
                entry = get_manifest_entry(adapter)
                url = model_fetch.resolve_url(
                    entry, filename, model_fetch.DEFAULT_ENDPOINT)

                class _NoRedirect(urllib.request.HTTPRedirectHandler):
                    def redirect_request(self, *_args, **_kwargs):
                        return None

                opener = urllib.request.build_opener(_NoRedirect())
                request = urllib.request.Request(url, method="HEAD")
                try:
                    with opener.open(request, timeout=30) as response:
                        headers = response.headers
                except urllib.error.HTTPError as exc:
                    # The digest rides on the 302 that points at storage, and
                    # refusing to follow it turns that response into this.
                    headers = exc.headers
                served = headers.get("X-Linked-Etag", "").strip('"')
                self.assertEqual(
                    served, entry.sha256[filename],
                    f"{filename} upstream now serves {served!r}; the pin in "
                    f"adapter_manifest.py would reject every real download",
                )

    def test_the_resolve_url_names_the_commit_not_a_branch(self):
        entry = get_manifest_entry("opencv-lama")
        url = model_fetch.resolve_url(
            entry, "inpainting_lama_2025jan.onnx",
            model_fetch.DEFAULT_ENDPOINT,
        )
        self.assertEqual(
            url,
            "https://huggingface.co/opencv/inpainting_lama/resolve/"
            "aee6d22f0a13e5e35af1c9a1c3afd62841fc6f3f/"
            "inpainting_lama_2025jan.onnx",
        )


@unittest.skipUnless(
    os.environ.get("VSR_MODEL_FETCH_TESTS", "").strip().lower()
    in {"1", "true", "yes", "on"},
    "opt-in: downloads 92 MB from Hugging Face. Set VSR_MODEL_FETCH_TESTS=1.",
)
class RealFetchEndToEndTests(unittest.TestCase):
    """The whole route, with the real weight, on demand.

    The deterministic tests above cover the download mechanism against a local
    server, but they cannot prove that a fetched weight actually inpaints,
    because the fixture payload is not an ONNX model. This one starts from an
    empty model cache, fetches the pinned OpenCV LaMa weight for real, and runs
    a LaMa job on a reference clip. It is opt-in because it moves 92 MB.
    """

    def test_an_empty_cache_can_reach_a_finished_lama_render(self):
        import subprocess
        import sys

        repo_root = Path(__file__).resolve().parents[1]
        clip = repo_root / "tests" / "clips" / "static_dialogue.mkv"
        self.assertTrue(clip.is_file(), f"missing reference clip {clip}")

        with tempfile.TemporaryDirectory(prefix="vsr-real-fetch-") as tmp:
            env = dict(os.environ)
            env["APPDATA"] = str(Path(tmp) / "AppData")
            env.pop("VSR_LAMA_ONNX", None)
            env.pop("VSR_OPENCV_LAMA", None)
            env.pop("VSR_ENABLE_PYTORCH_LAMA", None)
            output = Path(tmp) / "out.mp4"

            before = subprocess.run(
                [sys.executable, "-m", "backend.cli", "--input", str(clip),
                 "--output", str(output), "--mode", "lama"],
                cwd=str(repo_root), env=env, capture_output=True, text=True,
            )
            self.assertNotEqual(
                before.returncode, 0,
                "an empty cache must fail closed, or this test proves nothing",
            )
            self.assertFalse(output.exists())

            fetched = subprocess.run(
                [sys.executable, "-m", "backend.cli",
                 "--fetch-model", "opencv-lama"],
                cwd=str(repo_root), env=env, capture_output=True, text=True,
            )
            self.assertEqual(fetched.returncode, 0, fetched.stderr)

            after = subprocess.run(
                [sys.executable, "-m", "backend.cli", "--input", str(clip),
                 "--output", str(output), "--mode", "lama"],
                cwd=str(repo_root), env=env, capture_output=True, text=True,
            )
            self.assertEqual(after.returncode, 0, after.stderr[-2000:])
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
