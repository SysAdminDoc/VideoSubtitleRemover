import os
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace




def _has_display() -> bool:
    """Return True if a GUI display is available."""
    if sys.platform == "win32":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class RemoteModelPolicyTests(unittest.TestCase):
    def test_code_executing_remote_adapter_requires_full_commit_sha(self):
        from backend.remote_model_policy import resolve_remote_model_source

        tag = resolve_remote_model_source(
            "cotracker3", {"VSR_COTRACKER_REF": "v1.2.3"})
        short_sha = resolve_remote_model_source(
            "cotracker3", {"VSR_COTRACKER_REF": "deadbeef"})
        full_sha = resolve_remote_model_source(
            "cotracker3", {"VSR_COTRACKER_REF": "a" * 40})

        self.assertFalse(tag.allowed)
        self.assertFalse(short_sha.allowed)
        self.assertIn("tags and branches are mutable", tag.reason)
        self.assertTrue(full_sha.allowed)
        self.assertEqual(full_sha.reason, "approved immutable remote commit")

    def test_non_executing_adapter_can_use_version_tag(self):
        from backend.remote_model_policy import resolve_remote_model_source

        source = resolve_remote_model_source(
            "qwen25vl", {"VSR_QWEN25VL_REVISION": "v1.2.3"})

        self.assertTrue(source.allowed)


class ModelHashVerificationTests(unittest.TestCase):
    """RM-49: verify_weight_file should return True for a match,
    False for a mismatch, and True (with a debug log) when no vendored
    hash exists for the filename."""

    def test_verify_match(self):
        from backend.model_hashes import verify_weight_file, hash_file
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "weights.pt"
            p.write_bytes(b"hello world")
            expected = hash_file(p)
            self.assertTrue(verify_weight_file(p, expected_hash=expected))

    def test_verify_mismatch(self):
        from backend.model_hashes import verify_weight_file
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "weights.pt"
            p.write_bytes(b"hello world")
            self.assertFalse(verify_weight_file(p, expected_hash="0" * 64))

    def test_verify_unknown_filename_returns_true(self):
        from backend.model_hashes import verify_weight_file
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "not-tracked.bin"
            p.write_bytes(b"some bytes")
            # No vendored hash entry; verifier returns True (no-op).
            self.assertTrue(verify_weight_file(p))

    def test_verify_missing_file_returns_false(self):
        from backend.model_hashes import verify_weight_file
        result = verify_weight_file(Path("/nonexistent/weights.pt"),
                                      expected_hash="0" * 64)
        self.assertFalse(result)


class AdapterManifestVerificationTests(unittest.TestCase):
    """#109: optional adapter model paths carry provenance and can fail
    closed on unknown or mismatched weights before a loader deserializes
    the file."""

    def _entry(self, filename: str, sha256=None):
        from backend.adapter_manifest import AdapterManifestEntry
        return AdapterManifestEntry(
            name="unit-adapter",
            env_vars=("VSR_UNIT_MODEL",),
            expected_filenames=(filename,),
            sha256={} if sha256 is None else {filename: sha256},
            license="test-license",
            source_url="https://example.invalid/model",
            preferred_format="ONNX",
            remote_code_required=False,
        )

    def test_unknown_hash_allowed_for_legacy_adapter(self):
        from unittest import mock
        from backend import adapter_manifest as manifest
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "model.onnx"
            p.write_bytes(b"model")
            with mock.patch.dict(
                manifest.ADAPTER_MANIFEST,
                {"unit-adapter": self._entry(p.name)},
                clear=False,
            ):
                result = manifest.verify_adapter_path("unit-adapter", str(p))
        self.assertTrue(result.allowed)
        self.assertEqual(result.hash_status, "unknown")
        payload = result.as_dict()
        self.assertEqual(payload["preferredFormat"], "ONNX")
        self.assertEqual(payload["license"], "test-license")

    def test_strict_unknown_hash_fails_without_unsafe_override(self):
        from unittest import mock
        from backend import adapter_manifest as manifest
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "model.onnx"
            p.write_bytes(b"model")
            with mock.patch.dict(
                manifest.ADAPTER_MANIFEST,
                {"unit-adapter": self._entry(p.name)},
                clear=False,
            ):
                result = manifest.verify_adapter_path(
                    "unit-adapter", str(p), strict_unknown=True, env={}
                )
                override = manifest.verify_adapter_path(
                    "unit-adapter",
                    str(p),
                    strict_unknown=True,
                    env={manifest.UNSAFE_OVERRIDE_ENV: "1"},
                )
        self.assertFalse(result.allowed)
        self.assertEqual(result.hash_status, "unknown")
        self.assertTrue(override.allowed)
        self.assertEqual(override.hash_status, "unsafe_override")
        self.assertTrue(override.unsafe_override)

    def test_hash_mismatch_fails_unless_override_is_explicit(self):
        from unittest import mock
        from backend import adapter_manifest as manifest
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "model.onnx"
            p.write_bytes(b"model")
            with mock.patch.dict(
                manifest.ADAPTER_MANIFEST,
                {"unit-adapter": self._entry(p.name, "0" * 64)},
                clear=False,
            ):
                result = manifest.verify_adapter_path("unit-adapter", str(p))
                override = manifest.verify_adapter_path(
                    "unit-adapter",
                    str(p),
                    env={manifest.UNSAFE_OVERRIDE_ENV: "true"},
                )
        self.assertFalse(result.allowed)
        self.assertEqual(result.hash_status, "mismatch")
        self.assertTrue(result.actual_sha256)
        self.assertTrue(override.allowed)
        self.assertEqual(override.hash_status, "unsafe_override")

    def test_release_manifest_reports_configured_adapter_status(self):
        from unittest import mock
        from backend import adapter_manifest as manifest
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "model.onnx"
            p.write_bytes(b"model")
            with mock.patch.dict(
                manifest.ADAPTER_MANIFEST,
                {"unit-adapter": self._entry(p.name)},
                clear=True,
            ):
                statuses = manifest.release_manifest_status(
                    env={"VSR_UNIT_MODEL": str(p)}
                )
        self.assertEqual(len(statuses), 1)
        self.assertEqual(statuses[0]["name"], "unit-adapter")
        self.assertEqual(statuses[0]["configuredEnvVar"], "VSR_UNIT_MODEL")
        self.assertEqual(statuses[0]["hashStatus"], "unknown")

    def test_onnx_loader_refuses_mismatched_manifest_before_session(self):
        from unittest import mock
        from backend import adapter_manifest as manifest
        from backend import inpainters_onnx
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "model.onnx"
            p.write_bytes(b"model")
            session = mock.Mock()
            fake_ort = SimpleNamespace(InferenceSession=session)
            entry = manifest.AdapterManifestEntry(
                name="lama-onnx",
                env_vars=("VSR_LAMA_ONNX",),
                expected_filenames=(p.name,),
                sha256={p.name: "0" * 64},
                license="test-license",
                source_url="https://example.invalid/model",
                preferred_format="ONNX",
                remote_code_required=False,
            )
            with mock.patch.dict(
                manifest.ADAPTER_MANIFEST, {"lama-onnx": entry}, clear=False
            ), mock.patch.dict(sys.modules, {"onnxruntime": fake_ort}):
                result = inpainters_onnx._maybe_session(
                    str(p), ["CPUExecutionProvider"], "lama-onnx"
                )
        self.assertIsNone(result)
        session.assert_not_called()


class AdapterConformanceMatrixTests(unittest.TestCase):
    def test_conformance_matrix_schema_and_structure(self):
        from backend.adapter_manifest import (
            collect_adapter_conformance_matrix,
            CONFORMANCE_MATRIX_SCHEMA,
        )
        matrix = collect_adapter_conformance_matrix(env={})
        self.assertEqual(matrix["schema"], CONFORMANCE_MATRIX_SCHEMA)
        self.assertGreater(matrix["adapterCount"], 0)
        self.assertFalse(matrix["unsafeOverride"])
        self.assertEqual(
            matrix["summary"]["notConfigured"],
            matrix["adapterCount"],
        )
        adapter = matrix["adapters"][0]
        self.assertIn("name", adapter)
        self.assertIn("envVars", adapter)
        self.assertIn("license", adapter)
        self.assertIn("availability", adapter)
        self.assertIn("hasPinnedHash", adapter)
        self.assertEqual(adapter["availability"], "not-configured")

    def test_conformance_matrix_detects_configured_adapter(self):
        from backend.adapter_manifest import collect_adapter_conformance_matrix
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_model = os.path.join(tmpdir, "lama_fp32.onnx")
            Path(fake_model).write_bytes(b"fake")
            matrix = collect_adapter_conformance_matrix(
                env={"VSR_LAMA_ONNX": fake_model},
            )
        lama = next(a for a in matrix["adapters"] if a["name"] == "lama-onnx")
        self.assertTrue(lama["configured"])
        self.assertTrue(lama["pathExists"])
        self.assertEqual(lama["availability"], "ready")

    def test_conformance_format_is_human_readable(self):
        from backend.adapter_manifest import (
            collect_adapter_conformance_matrix,
            format_adapter_conformance_matrix,
        )
        matrix = collect_adapter_conformance_matrix(env={})
        text = format_adapter_conformance_matrix(matrix)
        self.assertIn("Adapter Conformance Matrix", text)
        self.assertIn("lama-onnx", text)
        self.assertIn("Total:", text)



if __name__ == "__main__":
    unittest.main()


def _read_msaa_properties(hwnd: int) -> dict:
    """Read an HWND's accessible properties back through MSAA.

    RM-282 annotates through IAccPropServices; reading with the client API
    is the only proof the annotation actually reached the window rather
    than being accepted and dropped.
    """
    import ctypes
    from ctypes import wintypes

    class _GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", ctypes.c_ulong),
            ("Data2", ctypes.c_ushort),
            ("Data3", ctypes.c_ushort),
            ("Data4", ctypes.c_ubyte * 8),
        ]

    class _VARIANT(ctypes.Structure):
        _fields_ = [
            ("vt", ctypes.c_ushort),
            ("r1", ctypes.c_ushort),
            ("r2", ctypes.c_ushort),
            ("r3", ctypes.c_ushort),
            ("data", ctypes.c_byte * 16),
        ]

    iid_accessible = _GUID(
        0x618736E0, 0x3C3D, 0x11CF,
        (ctypes.c_ubyte * 8)(0x81, 0x0C, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71))
    oleacc = ctypes.oledll.oleacc
    oleacc.AccessibleObjectFromWindow.argtypes = [
        wintypes.HWND, wintypes.DWORD, ctypes.POINTER(_GUID),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    acc = ctypes.c_void_p()
    oleacc.AccessibleObjectFromWindow(
        hwnd, 0xFFFFFFFC, ctypes.byref(iid_accessible), ctypes.byref(acc))
    vtable = ctypes.cast(
        acc, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    child_self = _VARIANT()
    child_self.vt = 3  # VT_I4 with a zero payload is CHILDID_SELF

    def _string_at(index):
        proto = ctypes.WINFUNCTYPE(
            ctypes.HRESULT, ctypes.c_void_p, _VARIANT,
            ctypes.POINTER(ctypes.c_wchar_p))
        out = ctypes.c_wchar_p()
        try:
            proto(vtable[index])(acc, child_self, ctypes.byref(out))
        except OSError:
            return None
        return out.value

    def _int_at(index):
        proto = ctypes.WINFUNCTYPE(
            ctypes.HRESULT, ctypes.c_void_p, _VARIANT,
            ctypes.POINTER(_VARIANT))
        out = _VARIANT()
        try:
            proto(vtable[index])(acc, child_self, ctypes.byref(out))
        except OSError:
            return None
        return int.from_bytes(bytes(out.data[:4]), "little")

    # IAccessible: IUnknown 0-2, IDispatch 3-6, get_accParent 7,
    # get_accChildCount 8, get_accChild 9, get_accName 10, get_accValue 11,
    # get_accDescription 12, get_accRole 13, get_accState 14, get_accHelp 15.
    return {
        "name": _string_at(10),
        "value": _string_at(11),
        "description": _string_at(12),
        "role": _int_at(13),
        "help": _string_at(15),
    }


class WindowsShellIntegrationTests(unittest.TestCase):
    """RM-160 / RM-166: two shipped Windows features that never ran.

    The screen-reader announcer called CreateObject("CUIAutomation8") -- an
    unregistered class string -- and would have called RaiseNotificationEvent
    on a client element that has no such method. The taskbar progress passed
    a GUID instance where comtypes wants an interface class, raising
    TypeError on every launch. Both failures were swallowed, so the features
    looked present and did nothing for several releases.
    """

    @unittest.skipUnless(sys.platform == "win32", "Windows-only shell APIs")
    def test_uia_announcement_provider_binds(self):
        from backend import a11y

        a11y._PROBED = False
        a11y._PROVIDER = None
        try:
            provider = a11y._probe_provider()
            self.assertIsNotNone(
                provider,
                "UIAutomationCore announcement entry points must bind; a "
                "silent None is how this stayed broken",
            )
            self.assertTrue(hasattr(provider, "UiaRaiseNotificationEvent"))
            self.assertTrue(hasattr(provider, "UiaHostProviderFromHwnd"))
        finally:
            a11y._PROBED = False
            a11y._PROVIDER = None

    @unittest.skipUnless(sys.platform == "win32", "Windows-only shell APIs")
    def test_announce_does_not_raise_and_is_silent_without_text(self):
        from backend import a11y

        a11y._PROBED = False
        a11y._PROVIDER = None
        try:
            a11y.announce("")
            a11y.announce("batch complete")
            a11y.announce("fatal error", importance="high")
        finally:
            a11y._PROBED = False
            a11y._PROVIDER = None

    @unittest.skipUnless(sys.platform == "win32", "Windows-only shell APIs")
    def test_taskbar_progress_acquires_a_real_com_object(self):
        import ctypes

        from gui.widgets import TaskbarProgress

        hwnd = ctypes.windll.user32.GetDesktopWindow()
        taskbar = TaskbarProgress(hwnd)
        try:
            self.assertIsNotNone(
                taskbar._taskbar,
                "ITaskbarList3 must be created; a None here is the dead "
                "feature this test exists to catch",
            )
            taskbar.set_state(TaskbarProgress.STATE_NORMAL)
            taskbar.set_value(1, 2)
            taskbar.clear()
        finally:
            taskbar.close()
        self.assertIsNone(taskbar._taskbar)
        taskbar.close()

    @unittest.skipUnless(sys.platform == "win32", "Windows-only shell APIs")
    def test_dynamic_annotation_services_bind(self):
        """RM-282: a silent None here is a dead accessibility feature."""
        from backend import a11y

        a11y._ACC_PROBED = False
        a11y._ACC_SERVICES = None
        try:
            services = a11y._acc_prop_services()
            self.assertIsNotNone(
                services,
                "IAccPropServices must be creatable; without it every "
                "custom control stays an anonymous pane",
            )
        finally:
            a11y._ACC_PROBED = False
            a11y._ACC_SERVICES = None

    @unittest.skipUnless(sys.platform == "win32", "Windows-only shell APIs")
    def test_annotation_lands_on_a_real_window_and_reads_back(self):
        """RM-282: annotate a real HWND, then read it back through MSAA."""
        import ctypes
        from ctypes import wintypes

        from backend import a11y

        user32 = ctypes.windll.user32
        user32.CreateWindowExW.restype = wintypes.HWND
        user32.CreateWindowExW.argtypes = [
            wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR,
            wintypes.DWORD, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE,
            ctypes.c_void_p,
        ]
        hwnd = user32.CreateWindowExW(
            0, "STATIC", "probe", 0x80000000,  # WS_POPUP, never shown
            0, 0, 10, 10, None, None, None, None,
        )
        self.assertTrue(hwnd, "could not create the probe window")
        try:
            applied = a11y.annotate_hwnd(
                hwnd,
                role="slider",
                name="Mask dilation",
                value="6 pixels",
                description="Grows the detected mask",
                help_text="Larger values cover more of the stroke",
            )
            self.assertTrue(applied, "no MSAA property was accepted")
            props = _read_msaa_properties(hwnd)
            self.assertEqual(props["name"], "Mask dilation")
            self.assertEqual(props["value"], "6 pixels")
            self.assertEqual(props["description"], "Grows the detected mask")
            self.assertEqual(
                props["help"], "Larger values cover more of the stroke")
            self.assertEqual(props["role"], 0x33)  # ROLE_SYSTEM_SLIDER
        finally:
            user32.DestroyWindow(hwnd)

    def test_every_widget_role_maps_to_a_distinct_msaa_role(self):
        """A role the map does not know must not silently become a button."""
        from backend import a11y

        for role, expected in (
            ("button", 0x2B),
            ("checkbox", 0x2C),
            ("radio button", 0x2D),
            ("slider", 0x33),
            ("progressbar", 0x30),
            ("status", 0x29),
            ("dialog", 0x12),
        ):
            with self.subTest(role=role):
                self.assertEqual(a11y._MSAA_ROLES[role], expected)
        self.assertNotIn("mystery control", a11y._MSAA_ROLES)
        self.assertEqual(a11y._MSAA_ROLE_DEFAULT, 0x14)  # grouping

    def test_annotation_is_skipped_when_nothing_changed(self):
        """The sync runs on every hover; unchanged metadata must not re-call."""
        from backend import a11y

        class FakeWidget:
            def winfo_id(self):
                return 0  # no real window, so annotate_hwnd short-circuits

        widget = FakeWidget()
        a11y.set_accessible_metadata(
            widget, role="button", label="Start", state="enabled")
        self.assertFalse(a11y.annotate_widget(widget))
        a11y.set_accessible_metadata(
            widget, role="button", label="Start", state="disabled")
        self.assertEqual(
            widget._vsr_a11y_hwnd_applied[2], "disabled",
            "a changed snapshot must be re-applied, not cached away",
        )

    def test_tooltip_text_becomes_accessible_help(self):
        """RM-282: a tooltip is invisible to a reader without this."""
        from backend import a11y

        class FakeWidget:
            def winfo_id(self):
                return 0

        widget = FakeWidget()
        a11y.set_tooltip_help(widget, "Runs every queued item")
        self.assertEqual(widget._vsr_a11y_help, "Runs every queued item")
        a11y.set_accessible_metadata(widget, role="button", label="Start")
        self.assertEqual(
            widget._vsr_a11y_hwnd_applied[5], "Runs every queued item",
            "widget metadata must not erase the tooltip help",
        )

    def test_annotating_an_unannotatable_object_degrades_silently(self):
        from backend import a11y

        class Hostile:
            def winfo_id(self):
                raise RuntimeError("no window")

        widget = Hostile()
        a11y.set_accessible_metadata(widget, role="button", label="Start")
        self.assertFalse(a11y.annotate_widget(widget))
