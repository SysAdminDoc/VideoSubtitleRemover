import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from backend import release_staging


ROOT = Path(__file__).resolve().parents[1]
VERSION = "9.9.9"


def _evidence(version=VERSION, **overrides):
    payload = {
        "app": {"name": "Video Subtitle Remover Pro", "version": version},
        "versionChecks": {"appVersion": version, "documents": []},
        "errors": [],
        "installer": {"validPortableExecutable": True, "sha256": "0" * 64},
        "installerSmoke": {"ran": True, "passed": True},
        "smokeLaunch": {"passed": True},
    }
    payload.update(overrides)
    return payload


def _make_inputs(root: Path, *, version=VERSION, evidence=None):
    dist = root / "dist" / "VideoSubtitleRemoverPro"
    (dist / "sub").mkdir(parents=True)
    (dist / "VideoSubtitleRemoverPro.exe").write_bytes(b"MZ frozen payload")
    (dist / "sub" / "asset.dat").write_bytes(b"asset")
    installer = root / "stage-installer.exe"
    installer.write_bytes(b"MZ installer payload")
    evidence_dir = root / "evidence"
    evidence_dir.mkdir()
    (evidence_dir / "release-verification.json").write_text(
        json.dumps(evidence if evidence is not None else _evidence(version)),
        encoding="utf-8",
    )
    for name in release_staging.EVIDENCE_FILES:
        path = evidence_dir / name
        if not path.exists():
            path.write_text("{}\n", encoding="utf-8")
    return dist, installer, evidence_dir


class ReleaseAssetNamingTests(unittest.TestCase):
    def test_every_asset_name_derives_from_the_version(self):
        names = release_staging.expected_assets("4.1.2")
        self.assertIn("VideoSubtitleRemoverPro-4.1.2-Setup.exe", names)
        self.assertIn("VideoSubtitleRemoverPro-4.1.2-Windows-x64.zip", names)
        self.assertIn("SHA256SUMS.txt", names)
        for evidence in release_staging.EVIDENCE_FILES:
            self.assertIn(evidence, names)
        self.assertEqual(len(names), len(set(names)))


class ReleaseStagingTests(unittest.TestCase):
    def test_stage_promotes_one_hashed_versioned_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dist, installer, evidence_dir = _make_inputs(root)
            release_root = root / "release"
            report = release_staging.stage_release(
                VERSION,
                dist_dir=dist,
                installer_path=installer,
                evidence_dir=evidence_dir,
                release_root=release_root,
            )
            self.assertTrue(report["valid"])
            target = release_root / VERSION
            self.assertEqual(
                sorted(item.name for item in target.iterdir()),
                list(release_staging.expected_assets(VERSION)),
            )
            verify = release_staging.verify_release_dir(target, VERSION)
            self.assertTrue(verify["valid"], verify["problems"])
            # No temporary stage directories are left behind.
            self.assertEqual(
                [item.name for item in release_root.iterdir()], [VERSION])
            archive = target / release_staging.portable_asset_name(VERSION)
            with zipfile.ZipFile(archive) as handle:
                self.assertIn(
                    "VideoSubtitleRemoverPro/VideoSubtitleRemoverPro.exe",
                    handle.namelist(),
                )
                self.assertIn(
                    "VideoSubtitleRemoverPro/sub/asset.dat", handle.namelist())

    def test_version_drift_in_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dist, installer, evidence_dir = _make_inputs(
                root, evidence=_evidence("3.22.0"))
            with self.assertRaises(release_staging.ReleaseStagingError) as ctx:
                release_staging.stage_release(
                    VERSION,
                    dist_dir=dist,
                    installer_path=installer,
                    evidence_dir=evidence_dir,
                    release_root=root / "release",
                )
            self.assertIn("3.22.0", str(ctx.exception))
            self.assertFalse((root / "release" / VERSION).exists())

    def test_failed_smoke_blocks_promotion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dist, installer, evidence_dir = _make_inputs(
                root,
                evidence=_evidence(installerSmoke={"ran": True, "passed": False}),
            )
            with self.assertRaises(release_staging.ReleaseStagingError) as ctx:
                release_staging.stage_release(
                    VERSION,
                    dist_dir=dist,
                    installer_path=installer,
                    evidence_dir=evidence_dir,
                    release_root=root / "release",
                )
            self.assertIn("installer payload smoke", str(ctx.exception))

    def test_verification_errors_block_promotion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dist, installer, evidence_dir = _make_inputs(
                root, evidence=_evidence(errors=["SBOM is not derived"]))
            with self.assertRaises(release_staging.ReleaseStagingError):
                release_staging.stage_release(
                    VERSION,
                    dist_dir=dist,
                    installer_path=installer,
                    evidence_dir=evidence_dir,
                    release_root=root / "release",
                )

    def test_incomplete_evidence_set_blocks_promotion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dist, installer, evidence_dir = _make_inputs(root)
            (evidence_dir / "sbom.cdx.json").unlink()
            with self.assertRaises(release_staging.ReleaseStagingError) as ctx:
                release_staging.stage_release(
                    VERSION,
                    dist_dir=dist,
                    installer_path=installer,
                    evidence_dir=evidence_dir,
                    release_root=root / "release",
                )
            self.assertIn("sbom.cdx.json", str(ctx.exception))

    def test_restaging_replaces_the_previous_set_atomically(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dist, installer, evidence_dir = _make_inputs(root)
            release_root = root / "release"
            release_staging.stage_release(
                VERSION,
                dist_dir=dist,
                installer_path=installer,
                evidence_dir=evidence_dir,
                release_root=release_root,
            )
            target = release_root / VERSION
            (target / "leftover-from-old-build.zip").write_bytes(b"stale")
            self.assertFalse(
                release_staging.verify_release_dir(target, VERSION)["valid"])
            installer.write_bytes(b"MZ rebuilt installer payload")
            release_staging.stage_release(
                VERSION,
                dist_dir=dist,
                installer_path=installer,
                evidence_dir=evidence_dir,
                release_root=release_root,
            )
            report = release_staging.verify_release_dir(target, VERSION)
            self.assertTrue(report["valid"], report["problems"])
            self.assertFalse((target / "leftover-from-old-build.zip").exists())


class ReleaseVerificationTests(unittest.TestCase):
    def test_mismatched_checksum_is_reported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dist, installer, evidence_dir = _make_inputs(root)
            release_root = root / "release"
            release_staging.stage_release(
                VERSION,
                dist_dir=dist,
                installer_path=installer,
                evidence_dir=evidence_dir,
                release_root=release_root,
            )
            target = release_root / VERSION
            (target / release_staging.installer_asset_name(VERSION)).write_bytes(
                b"tampered")
            report = release_staging.verify_release_dir(target, VERSION)
            self.assertFalse(report["valid"])
            self.assertTrue(
                any("checksum mismatch" in item for item in report["problems"]))

    def test_extra_asset_is_reported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dist, installer, evidence_dir = _make_inputs(root)
            release_root = root / "release"
            release_staging.stage_release(
                VERSION,
                dist_dir=dist,
                installer_path=installer,
                evidence_dir=evidence_dir,
                release_root=release_root,
            )
            target = release_root / VERSION
            (target / "VideoSubtitleRemoverPro-3.21.0-Windows-x64.zip").write_bytes(
                b"older release")
            report = release_staging.verify_release_dir(target, VERSION)
            self.assertFalse(report["valid"])
            self.assertTrue(any(
                "unexpected asset" in item for item in report["problems"]))

    def test_missing_directory_is_reported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = release_staging.verify_release_dir(
                Path(tmpdir) / "absent", VERSION)
            self.assertFalse(report["valid"])


class PublicationGuidanceTests(unittest.TestCase):
    def test_guidance_is_draft_immutable_and_explicitly_unsigned(self):
        lines = "\n".join(release_staging.publication_guidance("1.2.3"))
        self.assertIn("--draft", lines)
        self.assertIn("immutable", lines.lower())
        self.assertIn("UNSIGNED", lines)
        self.assertNotIn("sign the", lines.lower())
        self.assertIn("build/release/1.2.3/", lines)


class BuildScriptContractTests(unittest.TestCase):
    def test_build_script_stages_and_verifies_before_promoting(self):
        script = (ROOT / "build_exe.bat").read_text(encoding="utf-8")
        self.assertIn("backend.release_staging stage", script)
        self.assertIn("backend.release_staging verify", script)
        self.assertIn("backend.release_staging guidance", script)
        stage_index = script.index("backend.release_staging stage")
        promote_index = script.index('copy /Y "!RELEASE_DIR!\\!APP_VERSION!')
        self.assertLess(stage_index, promote_index)


if __name__ == "__main__":
    unittest.main()
