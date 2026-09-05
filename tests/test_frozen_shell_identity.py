"""RM-346: long-path support and a taskbar identity for the installed app.

Two things the frozen build silently lost against a source run. CPython ships
``longPathAware`` in ``python.exe``'s manifest, so a developer can open a media
path past 260 characters while the installed application cannot, and the value
is cached per process on first use so it cannot be set at runtime. And a frozen
Python application with no AppUserModelID inherits the interpreter's identity,
so Windows groups and pins the wrong thing.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import VideoSubtitleRemover as entry
from backend.release_verification import (
    frozen_manifest_status,
    read_embedded_manifest,
)

_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST = _ROOT / "installer" / "VideoSubtitleRemoverPro.exe.manifest"


class ManifestSourceTests(unittest.TestCase):
    def test_the_manifest_declares_long_path_awareness(self):
        import xml.etree.ElementTree as ET

        # Parsed rather than matched: the file's own comment names
        # longPathAware too, and a regex finds that first.
        tree = ET.parse(_MANIFEST)
        found = [
            element for element in tree.iter()
            if element.tag.rsplit("}", 1)[-1] == "longPathAware"
        ]
        self.assertEqual(len(found), 1, "expected exactly one declaration")
        self.assertEqual((found[0].text or "").strip().lower(), "true")

    def test_the_spec_points_at_that_manifest(self):
        spec = (_ROOT / "VideoSubtitleRemoverPro.spec").read_text(
            encoding="utf-8")
        self.assertIn("manifest=", spec)
        self.assertIn("installer/VideoSubtitleRemoverPro.exe.manifest", spec)

    def test_the_manifest_is_well_formed_xml(self):
        import xml.etree.ElementTree as ET

        # PyInstaller embeds this verbatim. Malformed XML produces an
        # executable Windows refuses to start, which no other test would see.
        ET.parse(_MANIFEST)


@unittest.skipUnless(sys.platform == "win32", "manifest resources are Windows")
class ManifestReadbackTests(unittest.TestCase):
    """The reader has to actually detect the flag, not just return a dict."""

    def test_it_finds_long_path_awareness_in_a_binary_that_declares_it(self):
        # CPython ships longPathAware in its own manifest, so this is a
        # positive control that does not depend on a build having run.
        text = read_embedded_manifest(sys.executable)
        self.assertTrue(text, "no manifest resource read from python.exe")
        match = re.search(r"longPathAware[^<]*>([^<]*)<", text, re.I)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1).strip().lower(), "true")

    def test_a_missing_file_reports_nothing_rather_than_raising(self):
        with tempfile.TemporaryDirectory(prefix="vsr-manifest-") as tmp:
            status = frozen_manifest_status(tmp)
        self.assertFalse(status["available"])
        self.assertFalse(status["longPathAware"])
        self.assertEqual(read_embedded_manifest(Path(tmp) / "absent.exe"), "")

    def test_a_file_with_no_manifest_resource_is_reported_unreadable(self):
        with tempfile.TemporaryDirectory(prefix="vsr-manifest-") as tmp:
            fake = Path(tmp) / "VideoSubtitleRemoverPro.exe"
            fake.write_bytes(b"MZ" + b"\x00" * 512)
            status = frozen_manifest_status(tmp)
        self.assertTrue(status["available"])
        self.assertFalse(status["readable"])
        self.assertFalse(status["longPathAware"])


class ManifestDetectionTests(unittest.TestCase):
    """The detector must read the declaration, not any mention of it."""

    def test_a_comment_naming_the_setting_is_not_a_declaration(self):
        from backend.release_verification import _manifest_flag_is_true

        xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<assembly xmlns="urn:schemas-microsoft-com:asm.v1" '
            'manifestVersion="1.0">'
            "<!-- this build deliberately does not set longPathAware -->"
            "</assembly>"
        )
        self.assertFalse(
            _manifest_flag_is_true(xml, "longPathAware"),
            "a substring match would call this a declaration",
        )

    def test_a_declaration_after_a_comment_that_names_it_is_still_found(self):
        from backend.release_verification import _manifest_flag_is_true

        # This is the shape of the shipped manifest, whose own comment
        # explains longPathAware above the element. A substring detector reads
        # the comment first and reports the setting as absent or empty, which
        # would fail a correct build.
        xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<assembly xmlns="urn:schemas-microsoft-com:asm.v1" '
            'manifestVersion="1.0">'
            "<!-- CPython ships longPathAware in its own manifest -->"
            '<application xmlns="urn:schemas-microsoft-com:asm.v3">'
            "<windowsSettings>"
            '<longPathAware xmlns="http://schemas.microsoft.com/SMI/2016/'
            'WindowsSettings">true</longPathAware>'
            "</windowsSettings></application></assembly>"
        )
        self.assertTrue(_manifest_flag_is_true(xml, "longPathAware"))

    def test_a_false_declaration_is_reported_false(self):
        from backend.release_verification import _manifest_flag_is_true

        xml = (
            '<assembly xmlns="urn:schemas-microsoft-com:asm.v1" '
            'manifestVersion="1.0"><application '
            'xmlns="urn:schemas-microsoft-com:asm.v3"><windowsSettings>'
            "<longPathAware>false</longPathAware>"
            "</windowsSettings></application></assembly>"
        )
        self.assertFalse(_manifest_flag_is_true(xml, "longPathAware"))

    def test_unparseable_xml_is_reported_false_rather_than_raising(self):
        from backend.release_verification import _manifest_flag_is_true

        self.assertFalse(_manifest_flag_is_true("<assembly", "longPathAware"))

    def test_pyinstaller_transform_preserves_the_declaration(self):
        """The spec hands PyInstaller a file; PyInstaller rewrites it.

        create_application_manifest reformats the document and appends its own
        trustInfo and dependency blocks, so the value that reaches the
        executable is not the file on disk. This runs the real transform.
        """
        from PyInstaller.utils.win32 import winmanifest

        from backend.release_verification import (
            _manifest_element_text,
            _manifest_flag_is_true,
        )

        produced = winmanifest.create_application_manifest(
            _MANIFEST.read_bytes(), False, False)
        text = (produced.decode("utf-8", "replace")
                if isinstance(produced, bytes) else str(produced))
        self.assertTrue(_manifest_flag_is_true(text, "longPathAware"))
        self.assertIn(
            "PerMonitorV2", _manifest_element_text(text, "dpiAwareness"))


@unittest.skipUnless(sys.platform == "win32", "resource writing is Windows")
class FrozenManifestStatusTests(unittest.TestCase):
    """Drive frozen_manifest_status against real executables.

    The detector tests above call the parsing helper directly. That is not
    enough: a substring-matching status function passes every one of them
    while reporting a correct build as non-compliant, which is how this was
    nearly shipped. These build actual PE files and read them back.
    """

    def _exe_with_manifest(self, directory: Path, manifest: str) -> Path:
        from PyInstaller.utils.win32 import winresource

        target = directory / "VideoSubtitleRemoverPro.exe"
        target.write_bytes(Path(sys.executable).read_bytes())
        try:
            winresource.add_or_update_resource(
                str(target), manifest.encode("utf-8"), 24, [1], [0])
        except Exception as exc:  # noqa: BLE001 - environment, not the subject
            self.skipTest(f"cannot rewrite PE resources here: {exc}")
        return target

    def test_a_real_exe_declaring_it_is_reported_compliant(self):
        manifest = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<assembly xmlns="urn:schemas-microsoft-com:asm.v1" '
            'manifestVersion="1.0">'
            '<application xmlns="urn:schemas-microsoft-com:asm.v3">'
            "<windowsSettings>"
            '<longPathAware xmlns="http://schemas.microsoft.com/SMI/2016/'
            'WindowsSettings">true</longPathAware>'
            "</windowsSettings></application></assembly>"
        )
        with tempfile.TemporaryDirectory(prefix="vsr-pe-") as tmp:
            self._exe_with_manifest(Path(tmp), manifest)
            status = frozen_manifest_status(tmp)
        self.assertTrue(status["available"])
        self.assertTrue(status["readable"])
        self.assertTrue(status["longPathAware"])

    def test_a_real_exe_only_mentioning_it_in_a_comment_is_not(self):
        manifest = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<assembly xmlns="urn:schemas-microsoft-com:asm.v1" '
            'manifestVersion="1.0">'
            "<!-- this build deliberately does not set longPathAware -->"
            '<application xmlns="urn:schemas-microsoft-com:asm.v3">'
            "<windowsSettings/></application></assembly>"
        )
        with tempfile.TemporaryDirectory(prefix="vsr-pe-") as tmp:
            self._exe_with_manifest(Path(tmp), manifest)
            status = frozen_manifest_status(tmp)
        self.assertTrue(status["readable"])
        self.assertFalse(
            status["longPathAware"],
            "a comment is not a declaration; a substring check says it is",
        )

    def test_the_shipped_manifest_survives_the_real_pyinstaller_transform(self):
        from PyInstaller.utils.win32 import winmanifest

        produced = winmanifest.create_application_manifest(
            _MANIFEST.read_bytes(), False, False)
        text = (produced.decode("utf-8", "replace")
                if isinstance(produced, bytes) else str(produced))
        with tempfile.TemporaryDirectory(prefix="vsr-pe-") as tmp:
            self._exe_with_manifest(Path(tmp), text)
            status = frozen_manifest_status(tmp)
        self.assertTrue(
            status["longPathAware"],
            "this is the exact manifest the build embeds",
        )
        self.assertTrue(status["dpiAware"])


class ManifestGateTests(unittest.TestCase):
    """Recording the value is not asserting it.

    frozen_manifest_status writes longPathAware into the evidence, but only
    _validation_errors can fail a build. An earlier version of this work
    recorded the flag and gated on nothing, so a spec edit that silently
    stopped embedding the manifest still produced a green release.
    """

    def test_a_build_without_the_declaration_is_rejected(self):
        from backend.release_verification import _validation_errors

        errors = list(_validation_errors({
            "frozenManifest": {
                "available": True, "readable": True, "longPathAware": False,
            },
        }))
        self.assertTrue(
            any("longPathAware" in error for error in errors),
            f"nothing failed the build; errors were {errors}",
        )

    def test_a_build_with_the_declaration_raises_no_manifest_error(self):
        from backend.release_verification import _validation_errors

        errors = list(_validation_errors({
            "frozenManifest": {
                "available": True, "readable": True, "longPathAware": True,
            },
        }))
        self.assertEqual(
            [error for error in errors if "longPathAware" in error], [])

    def test_an_unreadable_manifest_is_rejected(self):
        from backend.release_verification import _validation_errors

        errors = list(_validation_errors({
            "frozenManifest": {"available": True, "readable": False},
        }))
        self.assertTrue(any("readable" in error for error in errors))

    def test_missing_evidence_is_rejected_rather_than_assumed_fine(self):
        from backend.release_verification import _validation_errors

        errors = list(_validation_errors({}))
        self.assertTrue(
            any("manifest evidence" in error for error in errors),
            "absent evidence must not read as a pass",
        )

    def test_no_frozen_executable_does_not_double_report(self):
        from backend.release_verification import _validation_errors

        errors = list(_validation_errors({
            "frozenManifest": {"available": False},
        }))
        self.assertEqual(
            [error for error in errors if "manifest" in error.lower()], [],
            "the launcher checks already report a missing executable",
        )


class AppUserModelIdTests(unittest.TestCase):
    def test_the_identity_is_the_product_not_the_interpreter(self):
        self.assertEqual(
            entry.APP_USER_MODEL_ID, "SysAdminDoc.VideoSubtitleRemoverPro")

    @unittest.skipUnless(sys.platform == "win32", "shell32 is Windows")
    def test_setting_it_calls_shell32_with_that_identity(self):
        seen = []

        class _Shell32:
            def SetCurrentProcessExplicitAppUserModelID(self, value):
                seen.append(value)
                return 0

        class _Windll:
            shell32 = _Shell32()

        with mock.patch.dict(sys.modules):
            import ctypes

            with mock.patch.object(ctypes, "windll", _Windll(), create=True):
                self.assertTrue(entry._set_app_user_model_id())
        self.assertEqual(seen, [entry.APP_USER_MODEL_ID])

    @unittest.skipUnless(sys.platform == "win32", "shell32 is Windows")
    def test_a_failing_call_does_not_stop_the_application_starting(self):
        class _Shell32:
            def SetCurrentProcessExplicitAppUserModelID(self, value):
                raise OSError("no such export")

        class _Windll:
            shell32 = _Shell32()

        import ctypes

        with mock.patch.object(ctypes, "windll", _Windll(), create=True):
            self.assertFalse(
                entry._set_app_user_model_id(),
                "the only cost of failure is taskbar grouping",
            )

    def test_it_runs_before_any_window_exists(self):
        source = (_ROOT / "VideoSubtitleRemover.py").read_text(encoding="utf-8")
        call = source.index("    _set_app_user_model_id()")
        # Anchored on the construction that presents UI, not on the import of
        # the same name at the top of the module.
        window = source.index("VideoSubtitleRemoverApp(instance_guard=guard)")
        self.assertLess(
            call, window,
            "SetCurrentProcessExplicitAppUserModelID must precede the first "
            "window or Windows has already grouped the process",
        )


class LongPathBehaviourTests(unittest.TestCase):
    """Prove the interpreter running these tests can use a long path.

    This is the behaviour the manifest buys the frozen build. It skips rather
    than fails where the machine policy is off, because the manifest alone is
    not sufficient: Windows requires LongPathsEnabled as well.
    """

    def _long_path_enabled(self) -> bool:
        if sys.platform != "win32":
            return True
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SYSTEM\CurrentControlSet\Control\FileSystem",
            ) as key:
                value, _ = winreg.QueryValueEx(key, "LongPathsEnabled")
                return bool(value)
        except OSError:
            return False

    def test_a_path_past_260_characters_round_trips(self):
        if not self._long_path_enabled():
            self.skipTest(
                "LongPathsEnabled is off on this host; the manifest alone "
                "does not lift the limit"
            )
        with tempfile.TemporaryDirectory(prefix="vsr-longpath-") as tmp:
            root = Path(tmp)
            deep = root
            while len(str(deep)) < 300:
                deep = deep / ("segment" + "x" * 20)
            deep.mkdir(parents=True, exist_ok=True)
            target = deep / "clip.txt"
            payload = "a deep media path\n"
            target.write_text(payload, encoding="utf-8")
            self.assertGreater(len(str(target)), 260)
            self.assertEqual(target.read_text(encoding="utf-8"), payload)
            self.assertTrue(os.path.isfile(target))


if __name__ == "__main__":
    unittest.main()
