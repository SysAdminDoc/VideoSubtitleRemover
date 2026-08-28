import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

from backend import release_staging


ROOT = Path(__file__).resolve().parents[1]
VERSION = "9.9.9"
PROFILE = "cpu"


def _frozen_provider(profile=PROFILE, **overrides):
    """The frozen build's own answer about which provider it activates."""
    from backend.build_profile import declared_provider

    provider = declared_provider(profile)
    payload = {
        "schema": "vsr.frozen_provider_smoke.v1",
        "available": True,
        "ran": True,
        "passed": True,
        "profile": profile,
        "profileSource": "stamp",
        "declaredProvider": provider,
        "availableProviders": [provider, "CPUExecutionProvider"],
        "activeProviders": [provider],
        "fellBack": False,
        "error": "",
    }
    payload.update(overrides)
    return payload


def _evidence(version=VERSION, profile=PROFILE, **overrides):
    payload = {
        "app": {"name": "Video Subtitle Remover Pro", "version": version},
        "versionChecks": {"appVersion": version, "documents": []},
        "errors": [],
        "installer": {"validPortableExecutable": True, "sha256": "0" * 64},
        "installerSmoke": {"ran": True, "passed": True},
        "smokeLaunch": {"passed": True},
        "frozenProviderSmoke": _frozen_provider(profile),
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


def _stage(root, dist, installer, evidence_dir, *, profile=PROFILE,
           version=VERSION):
    return release_staging.stage_release(
        version,
        dist_dir=dist,
        # A lane that ships no installer refuses one on purpose, so the
        # helper hands over only what that lane actually publishes.
        installer_path=(
            installer if release_staging.ships_installer(profile) else None),
        evidence_dir=evidence_dir,
        release_root=root / "release",
        profile=profile,
    )


class ReleaseAssetNamingTests(unittest.TestCase):
    def test_every_asset_name_derives_from_the_version_and_the_lane(self):
        names = release_staging.expected_assets("4.1.2", "cpu")
        self.assertIn("VideoSubtitleRemoverPro-4.1.2-cpu-Setup.exe", names)
        self.assertIn(
            "VideoSubtitleRemoverPro-4.1.2-cpu-Windows-x64.zip", names)
        self.assertIn("SHA256SUMS.txt", names)
        for evidence in release_staging.EVIDENCE_FILES:
            self.assertIn(evidence, names)
        self.assertEqual(len(names), len(set(names)))

    def test_the_two_published_lanes_never_share_a_filename(self):
        """RM-350: the download was one generically named build."""
        cpu = set(release_staging.expected_assets("4.1.2", "cpu"))
        nvidia = set(release_staging.expected_assets("4.1.2", "nvidia"))
        shared = cpu & nvidia
        self.assertEqual(
            shared,
            {"SHA256SUMS.txt", *release_staging.EVIDENCE_FILES},
            "only the per-lane evidence filenames may repeat",
        )
        self.assertIn(
            "VideoSubtitleRemoverPro-4.1.2-nvidia-Windows-x64.zip", nvidia)

    def test_an_unknown_lane_is_refused_rather_than_named(self):
        for candidate in ("", "cuda", "gpu", "CPU-ish", None):
            with self.subTest(profile=candidate):
                with self.assertRaises(release_staging.ReleaseStagingError):
                    release_staging.installer_asset_name("4.1.2", candidate)

    def test_a_zip_only_lane_expects_no_installer_asset(self):
        """RM-350: the CUDA payload is past what NSIS can package."""
        names = release_staging.expected_assets("4.1.2", "nvidia")
        self.assertIn(
            "VideoSubtitleRemoverPro-4.1.2-nvidia-Windows-x64.zip", names)
        self.assertNotIn(
            "VideoSubtitleRemoverPro-4.1.2-nvidia-Setup.exe", names)
        self.assertFalse(release_staging.ships_installer("nvidia"))
        self.assertTrue(release_staging.ships_installer("cpu"))

    def test_the_reason_the_cuda_lane_has_no_installer_is_written_down(self):
        source = (ROOT / "backend" / "release_staging.py").read_text(
            encoding="utf-8")
        self.assertIn("32-bit", source)
        self.assertIn("2 GB", source)
        self.assertIn("mmapping file", source)

    def test_directml_is_a_profile_but_not_a_published_artifact(self):
        from backend.dependency_profiles import SUPPORTED_PROFILES

        self.assertIn("directml", SUPPORTED_PROFILES)
        self.assertNotIn("directml", release_staging.PUBLISHED_PROFILES)
        self.assertEqual(
            release_staging.PUBLISHED_PROFILES, ("cpu", "nvidia"))


class ProfileClaimTests(unittest.TestCase):
    """RM-350: the filename is a claim; two records have to back it."""

    def test_a_cpu_payload_cannot_be_promoted_under_an_nvidia_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dist, installer, evidence_dir = _make_inputs(
                root, evidence=_evidence(profile="cpu"))
            with self.assertRaises(release_staging.ReleaseStagingError) as ctx:
                _stage(root, dist, installer, evidence_dir, profile="nvidia")
            message = str(ctx.exception)
            self.assertIn("nvidia", message)
            self.assertIn("cpu", message)
            self.assertFalse((root / "release" / VERSION).exists())

    def test_a_build_that_fell_back_to_cpu_cannot_claim_cuda(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frozen = _frozen_provider(
                "nvidia",
                activeProviders=["CPUExecutionProvider"],
                fellBack=True,
                passed=False,
            )
            dist, installer, evidence_dir = _make_inputs(
                root, evidence=_evidence(frozenProviderSmoke=frozen))
            with self.assertRaises(release_staging.ReleaseStagingError) as ctx:
                _stage(root, dist, installer, evidence_dir, profile="nvidia")
            self.assertIn("CUDAExecutionProvider", str(ctx.exception))

    def test_an_inferred_profile_is_not_evidence_of_what_was_built(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frozen = _frozen_provider(
                PROFILE, profileSource="installed-provider")
            dist, installer, evidence_dir = _make_inputs(
                root, evidence=_evidence(frozenProviderSmoke=frozen))
            with self.assertRaises(release_staging.ReleaseStagingError) as ctx:
                _stage(root, dist, installer, evidence_dir)
            self.assertIn("stamped", str(ctx.exception))

    def test_a_build_with_no_frozen_provider_evidence_is_refused(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            evidence = _evidence()
            evidence.pop("frozenProviderSmoke")
            dist, installer, evidence_dir = _make_inputs(
                root, evidence=evidence)
            with self.assertRaises(release_staging.ReleaseStagingError) as ctx:
                _stage(root, dist, installer, evidence_dir)
            self.assertIn("frozen provider evidence", str(ctx.exception))

    def test_a_smoke_that_never_ran_is_refused(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frozen = _frozen_provider(
                PROFILE, ran=False, passed=False, activeProviders=[])
            dist, installer, evidence_dir = _make_inputs(
                root, evidence=_evidence(frozenProviderSmoke=frozen))
            with self.assertRaises(release_staging.ReleaseStagingError) as ctx:
                _stage(root, dist, installer, evidence_dir)
            self.assertIn("did not run", str(ctx.exception))

    def test_a_matching_nvidia_build_promotes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dist, installer, evidence_dir = _make_inputs(
                root, evidence=_evidence(profile="nvidia"))
            report = _stage(root, dist, installer, evidence_dir,
                            profile="nvidia")
            self.assertTrue(report["valid"])
            self.assertEqual(report["profile"], "nvidia")
            target = root / "release" / VERSION / "nvidia"
            self.assertTrue(
                (target
                 / f"VideoSubtitleRemoverPro-{VERSION}-nvidia-Windows-x64.zip"
                 ).is_file())
            self.assertFalse(
                (target / f"VideoSubtitleRemoverPro-{VERSION}-nvidia-Setup.exe"
                 ).exists())

    def test_a_zip_only_lane_refuses_an_installer_it_cannot_ship(self):
        """Passing one means the caller believes something untrue."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dist, installer, evidence_dir = _make_inputs(
                root, evidence=_evidence(profile="nvidia"))
            with self.assertRaises(release_staging.ReleaseStagingError) as ctx:
                release_staging.stage_release(
                    VERSION,
                    dist_dir=dist,
                    installer_path=installer,
                    evidence_dir=evidence_dir,
                    release_root=root / "release",
                    profile="nvidia",
                )
            self.assertIn("ships no installer", str(ctx.exception))

    def test_a_lane_that_does_ship_an_installer_still_requires_one(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dist, _installer, evidence_dir = _make_inputs(root)
            with self.assertRaises(release_staging.ReleaseStagingError) as ctx:
                release_staging.stage_release(
                    VERSION,
                    dist_dir=dist,
                    installer_path=None,
                    evidence_dir=evidence_dir,
                    release_root=root / "release",
                    profile="cpu",
                )
            self.assertIn("Installer artifact is missing", str(ctx.exception))


class ReleaseStagingTests(unittest.TestCase):
    def test_stage_promotes_one_hashed_versioned_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dist, installer, evidence_dir = _make_inputs(root)
            release_root = root / "release"
            report = _stage(root, dist, installer, evidence_dir)
            self.assertTrue(report["valid"])
            target = release_root / VERSION / PROFILE
            self.assertEqual(
                sorted(item.name for item in target.iterdir()),
                list(release_staging.expected_assets(VERSION, PROFILE)),
            )
            verify = release_staging.verify_release_dir(
                target, VERSION, PROFILE)
            self.assertTrue(verify["valid"], verify["problems"])
            # No temporary stage directories are left behind.
            self.assertEqual(
                [item.name for item in release_root.iterdir()], [VERSION])
            archive = target / release_staging.portable_asset_name(
                VERSION, PROFILE)
            with zipfile.ZipFile(archive) as handle:
                self.assertIn(
                    "VideoSubtitleRemoverPro/VideoSubtitleRemoverPro.exe",
                    handle.namelist(),
                )
                self.assertIn(
                    "VideoSubtitleRemoverPro/sub/asset.dat", handle.namelist())

    def test_two_lanes_of_one_version_live_side_by_side(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            release_root = root / "release"
            for profile in ("cpu", "nvidia"):
                sub = root / profile
                sub.mkdir()
                dist, installer, evidence_dir = _make_inputs(
                    sub, evidence=_evidence(profile=profile))
                _stage(root, dist, installer, evidence_dir,
                       profile=profile)
            self.assertEqual(
                sorted(item.name for item in
                       (release_root / VERSION).iterdir()),
                ["cpu", "nvidia"],
            )
            self.assertEqual(
                release_staging.missing_published_profiles(
                    VERSION, release_root),
                [],
            )

    def test_a_release_missing_a_lane_says_which_one(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dist, installer, evidence_dir = _make_inputs(root)
            release_root = root / "release"
            _stage(root, dist, installer, evidence_dir)
            self.assertEqual(
                release_staging.missing_published_profiles(
                    VERSION, release_root),
                ["nvidia"],
            )

    def test_pruning_stale_artifacts_does_not_corrupt_the_report(self):
        """`--prune-stale` used to rebind the profile to a filename.

        The release was already promoted by then, so the operator saw
        "Release staging failed" and exit 1 for a release that had in fact
        succeeded, and the report named a ZIP as its lane.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dist, installer, evidence_dir = _make_inputs(root)
            release_root = root / "release"
            release_root.mkdir()
            stale = release_root / "VideoSubtitleRemoverPro-3.39.0-Setup.exe"
            stale.write_bytes(b"MZ from an older build")
            report = release_staging.stage_release(
                VERSION,
                dist_dir=dist,
                installer_path=installer,
                evidence_dir=evidence_dir,
                release_root=release_root,
                profile=PROFILE,
                prune_stale=True,
            )
            self.assertTrue(report["valid"])
            self.assertEqual(report["profile"], PROFILE)
            self.assertIn(stale.name, report["prunedStaleArtifacts"])
            self.assertFalse(stale.exists())
            self.assertEqual(
                report["assets"],
                list(release_staging.expected_assets(VERSION, PROFILE)),
            )

    def test_version_drift_in_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dist, installer, evidence_dir = _make_inputs(
                root, evidence=_evidence("3.22.0"))
            with self.assertRaises(release_staging.ReleaseStagingError) as ctx:
                _stage(root, dist, installer, evidence_dir)
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
                _stage(root, dist, installer, evidence_dir)
            self.assertIn("installer payload smoke", str(ctx.exception))

    def test_verification_errors_block_promotion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dist, installer, evidence_dir = _make_inputs(
                root, evidence=_evidence(errors=["SBOM is not derived"]))
            with self.assertRaises(release_staging.ReleaseStagingError):
                _stage(root, dist, installer, evidence_dir)

    def test_incomplete_evidence_set_blocks_promotion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dist, installer, evidence_dir = _make_inputs(root)
            (evidence_dir / "sbom.cdx.json").unlink()
            with self.assertRaises(release_staging.ReleaseStagingError) as ctx:
                _stage(root, dist, installer, evidence_dir)
            self.assertIn("sbom.cdx.json", str(ctx.exception))

    def test_restaging_replaces_the_previous_set_atomically(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dist, installer, evidence_dir = _make_inputs(root)
            release_root = root / "release"
            _stage(root, dist, installer, evidence_dir)
            target = release_root / VERSION / PROFILE
            (target / "leftover-from-old-build.zip").write_bytes(b"stale")
            self.assertFalse(
                release_staging.verify_release_dir(
                    target, VERSION, PROFILE)["valid"])
            installer.write_bytes(b"MZ rebuilt installer payload")
            _stage(root, dist, installer, evidence_dir)
            report = release_staging.verify_release_dir(
                target, VERSION, PROFILE)
            self.assertTrue(report["valid"], report["problems"])
            self.assertFalse((target / "leftover-from-old-build.zip").exists())

    def test_restaging_one_lane_leaves_the_other_alone(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            release_root = root / "release"
            for profile in ("cpu", "nvidia"):
                sub = root / profile
                sub.mkdir()
                dist, installer, evidence_dir = _make_inputs(
                    sub, evidence=_evidence(profile=profile))
                _stage(root, dist, installer, evidence_dir,
                       profile=profile)
            nvidia_before = sorted(
                (release_root / VERSION / "nvidia").iterdir())
            sub = root / "cpu"
            _stage(root, sub / "dist" / "VideoSubtitleRemoverPro",
                   sub / "stage-installer.exe", sub / "evidence",
                   profile="cpu")
            self.assertEqual(
                sorted((release_root / VERSION / "nvidia").iterdir()),
                nvidia_before,
            )


class ReleaseVerificationTests(unittest.TestCase):
    def test_mismatched_checksum_is_reported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dist, installer, evidence_dir = _make_inputs(root)
            release_root = root / "release"
            _stage(root, dist, installer, evidence_dir)
            target = release_root / VERSION / PROFILE
            (target / release_staging.installer_asset_name(
                VERSION, PROFILE)).write_bytes(b"tampered")
            report = release_staging.verify_release_dir(
                target, VERSION, PROFILE)
            self.assertFalse(report["valid"])
            self.assertTrue(
                any("checksum mismatch" in item for item in report["problems"]))

    def test_extra_asset_is_reported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dist, installer, evidence_dir = _make_inputs(root)
            release_root = root / "release"
            _stage(root, dist, installer, evidence_dir)
            target = release_root / VERSION / PROFILE
            (target / "VideoSubtitleRemoverPro-3.21.0-cpu-Windows-x64.zip"
             ).write_bytes(b"older release")
            report = release_staging.verify_release_dir(
                target, VERSION, PROFILE)
            self.assertFalse(report["valid"])
            self.assertTrue(any(
                "unexpected asset" in item for item in report["problems"]))

    def test_the_other_lanes_assets_are_not_accepted_here(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dist, installer, evidence_dir = _make_inputs(root)
            release_root = root / "release"
            _stage(root, dist, installer, evidence_dir)
            target = release_root / VERSION / PROFILE
            report = release_staging.verify_release_dir(
                target, VERSION, "nvidia")
            self.assertFalse(report["valid"])
            self.assertTrue(any(
                "nvidia" in item for item in report["problems"]))

    def test_missing_directory_is_reported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = release_staging.verify_release_dir(
                Path(tmpdir) / "absent", VERSION, PROFILE)
            self.assertFalse(report["valid"])


class PublicationGuidanceTests(unittest.TestCase):
    def test_guidance_is_draft_immutable_and_explicitly_unsigned(self):
        lines = "\n".join(release_staging.publication_guidance("1.2.3"))
        self.assertIn("--draft", lines)
        self.assertIn("immutable", lines.lower())
        self.assertIn("UNSIGNED", lines)
        self.assertNotIn("sign the", lines.lower())
        self.assertIn("build/release/1.2.3/", lines)

    def test_guidance_names_every_lane_that_has_to_ship(self):
        lines = "\n".join(release_staging.publication_guidance("1.2.3"))
        for profile in release_staging.PUBLISHED_PROFILES:
            self.assertIn(profile, lines)
        self.assertIn("DirectML", lines)

    def test_guidance_says_which_lanes_ship_no_installer(self):
        lines = "\n".join(release_staging.publication_guidance("1.2.3"))
        for profile in release_staging.PUBLISHED_PROFILES:
            if not release_staging.ships_installer(profile):
                self.assertIn("portable ZIP and no installer", lines)
                self.assertIn("2 GB", lines)


class FrozenProviderVerificationTests(unittest.TestCase):
    """`release_verification`'s half of the name-versus-provider gate.

    Staging refuses a mismatched artifact, and that is tested. This is the
    other end: the evidence writer's own error list, which could be deleted
    whole with every test still green.
    """

    def setUp(self):
        from backend import release_verification

        self.module = release_verification

    def _problems(self, **overrides):
        frozen = {
            "available": True,
            "ran": True,
            "passed": True,
            "profile": "nvidia",
            "profileSource": "stamp",
            "declaredProvider": "CUDAExecutionProvider",
            "activeProviders": ["CUDAExecutionProvider"],
            "fellBack": False,
            "error": "",
        }
        frozen.update(overrides)
        return list(self.module._frozen_provider_problems(
            {"frozenProviderSmoke": frozen}))

    def test_a_matching_build_reports_nothing(self):
        self.assertEqual(self._problems(), [])

    def test_a_build_that_ran_on_the_cpu_under_a_cuda_stamp_is_reported(self):
        problems = self._problems(
            activeProviders=["CPUExecutionProvider"], passed=False)
        self.assertTrue(problems)
        self.assertTrue(any("CUDAExecutionProvider" in item
                            for item in problems), problems)

    def test_a_fallback_is_reported(self):
        problems = self._problems(fellBack=True)
        self.assertTrue(any("fell back" in item for item in problems),
                        problems)

    def test_an_inferred_profile_is_reported(self):
        problems = self._problems(profileSource="installed-provider")
        self.assertTrue(any("stamped" in item for item in problems), problems)

    def test_an_unstamped_build_is_reported(self):
        problems = self._problems(profile="")
        self.assertTrue(any("no dependency profile stamp" in item
                            for item in problems), problems)

    def test_an_unrecognised_profile_does_not_silence_the_check(self):
        """`declared_provider` returns "" for a profile it does not know, and
        every comparison below used to short-circuit on that: a stamp saying
        CUDA with the CPU actually running produced no problem at all."""
        problems = self._problems(
            profile="cuda", activeProviders=["CPUExecutionProvider"])
        self.assertTrue(problems)
        self.assertTrue(any("unrecognised profile" in item
                            for item in problems), problems)
        self.assertTrue(any("CUDAExecutionProvider" in item
                            for item in problems), problems)

    def test_a_missing_exe_is_not_reported_as_a_failure(self):
        """A dist folder with no executable is the same non-event here as it
        is for the launch smoke; staging still refuses to promote it."""
        self.assertEqual(
            self._problems(available=False, ran=False, passed=False), [])

    def test_an_exe_that_was_there_and_did_not_answer_is_reported(self):
        problems = self._problems(
            ran=False, passed=False, error="the smoke wrote no result")
        self.assertTrue(any("did not run" in item for item in problems),
                        problems)

    def test_missing_evidence_is_reported(self):
        self.assertTrue(list(self.module._frozen_provider_problems({})))


class InstallerlessLaneTests(unittest.TestCase):
    """Strict verification for a lane that ships no installer."""

    def test_the_flag_is_wired_from_the_command_line(self):
        import argparse
        import inspect

        from backend import release_verification

        signature = inspect.signature(
            release_verification.write_release_evidence)
        self.assertIn("ships_installer", signature.parameters)
        self.assertIs(
            signature.parameters["ships_installer"].default, True,
            "a lane ships an installer unless it says otherwise",
        )
        source = inspect.getsource(release_verification.main)
        self.assertIn("--no-installer", source)
        self.assertIn("ships_installer=not args.no_installer", source)
        self.assertIsInstance(argparse.ArgumentParser(), argparse.ArgumentParser)

    def test_strict_mode_only_demands_an_installer_from_a_lane_with_one(self):
        import inspect

        from backend import release_verification

        source = inspect.getsource(release_verification.write_release_evidence)
        installer_check = source.index("NSIS installer artifact")
        guard = source.rindex("if ships_installer:", 0, installer_check)
        self.assertLess(guard, installer_check)
        # The shipped-executable smoke is NOT inside that guard: a ZIP-only
        # lane still has to prove its payload runs.
        smoke_check = source.index("Installer payload smoke did not pass")
        self.assertGreater(
            smoke_check - guard,
            installer_check - guard,
            "the smoke must stay outside the ships_installer guard",
        )


class FrozenCudaRuntimeTests(unittest.TestCase):
    """RM-350: the CUDA lane's runtime has to be findable once frozen.

    `onnxruntime.preload_dlls()` locates the CUDA and cuDNN libraries by
    asking `importlib.metadata` where the torch distribution is installed. A
    PyInstaller bundle carries no distribution metadata, so that lookup finds
    nothing, the search falls back to the directory above onnxruntime, and
    the CUDA provider cannot resolve cuBLAS. ONNX Runtime keeps reporting
    CUDA as available and quietly runs the session on the CPU, which is the
    "why is my GPU idle" report the separate CUDA download exists to answer.

    The frozen provider smoke caught this on a real build before it was
    published. These tests keep the fix honest without needing a GPU.
    """

    def setUp(self):
        from backend import onnxruntime_cuda

        self.module = onnxruntime_cuda

    def test_a_source_checkout_asks_for_no_directory(self):
        """Outside a frozen build the automatic lookup works."""
        self.assertFalse(getattr(sys, "frozen", False))
        self.assertIsNone(self.module.frozen_cuda_dll_directory())

    def test_a_frozen_build_points_at_the_bundled_torch_lib(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = Path(tmpdir)
            (bundle / "torch" / "lib").mkdir(parents=True)
            with mock.patch.object(sys, "frozen", True, create=True), \
                    mock.patch.object(sys, "_MEIPASS", str(bundle),
                                      create=True):
                self.assertEqual(
                    Path(self.module.frozen_cuda_dll_directory()),
                    bundle / "torch" / "lib",
                )

    def test_a_frozen_build_without_torch_asks_for_no_directory(self):
        """The CPU lane has no torch, and must not name a path that is not
        there: preload_dlls raises on a directory that does not exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(sys, "frozen", True, create=True), \
                    mock.patch.object(sys, "_MEIPASS", tmpdir, create=True):
                self.assertIsNone(self.module.frozen_cuda_dll_directory())

    def test_the_preload_passes_that_directory_through(self):
        recorded = {}

        class _Ort:
            __version__ = "1.29.0"

            @staticmethod
            def preload_dlls(directory=None):
                recorded["directory"] = directory

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = Path(tmpdir)
            (bundle / "torch" / "lib").mkdir(parents=True)
            self.module.reset_onnxruntime_cuda_preload_status()
            try:
                with mock.patch.object(sys, "frozen", True, create=True), \
                        mock.patch.object(sys, "_MEIPASS", str(bundle),
                                          create=True):
                    status = self.module.preload_onnxruntime_cuda_dlls_if_needed(
                        _Ort(), ["CUDAExecutionProvider"])
            finally:
                self.module.reset_onnxruntime_cuda_preload_status()

        self.assertEqual(
            Path(recorded["directory"]), bundle / "torch" / "lib")
        self.assertTrue(status["succeeded"])
        self.assertEqual(
            Path(status["lastDirectory"]), bundle / "torch" / "lib")

    def test_an_explicit_directory_still_wins(self):
        recorded = {}

        class _Ort:
            __version__ = "1.29.0"

            @staticmethod
            def preload_dlls(directory=None):
                recorded["directory"] = directory

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = Path(tmpdir)
            (bundle / "torch" / "lib").mkdir(parents=True)
            (bundle / "elsewhere").mkdir()
            self.module.reset_onnxruntime_cuda_preload_status()
            try:
                with mock.patch.object(sys, "frozen", True, create=True), \
                        mock.patch.object(sys, "_MEIPASS", str(bundle),
                                          create=True):
                    self.module.preload_onnxruntime_cuda_dlls_if_needed(
                        _Ort(), ["CUDAExecutionProvider"],
                        directory=str(bundle / "elsewhere"))
            finally:
                self.module.reset_onnxruntime_cuda_preload_status()
        self.assertEqual(
            Path(recorded["directory"]), bundle / "elsewhere")

    def test_a_cpu_session_never_goes_looking(self):
        calls = []

        class _Ort:
            __version__ = "1.29.0"

            @staticmethod
            def preload_dlls(directory=None):
                calls.append(directory)

        self.module.reset_onnxruntime_cuda_preload_status()
        try:
            self.module.preload_onnxruntime_cuda_dlls_if_needed(
                _Ort(), ["CPUExecutionProvider"])
        finally:
            self.module.reset_onnxruntime_cuda_preload_status()
        self.assertEqual(calls, [])


class BuildScriptContractTests(unittest.TestCase):
    def setUp(self):
        self.script = (ROOT / "build_exe.bat").read_text(encoding="utf-8")

    def test_build_script_stages_and_verifies_before_promoting(self):
        self.assertIn("backend.release_staging stage", self.script)
        self.assertIn("backend.release_staging verify", self.script)
        self.assertIn("backend.release_staging guidance", self.script)
        stage_index = self.script.index("backend.release_staging stage")
        promote_index = self.script.index(
            'copy /Y "!RELEASE_ROOT!\\!APP_VERSION!\\!PROFILE!')
        self.assertLess(stage_index, promote_index)

    def test_the_build_takes_a_lane_and_passes_it_everywhere(self):
        self.assertIn('set "PROFILE=%~1"', self.script)
        self.assertIn('set "VSR_DEPENDENCY_PROFILE=!PROFILE!"', self.script)
        self.assertIn('--profile "!PROFILE!"', self.script)
        # Two lanes must not share a dist tree, a work tree, or a scratch
        # evidence directory.
        self.assertIn('--distpath "dist\\!PROFILE!"', self.script)
        self.assertIn('--workpath "build\\!PROFILE!"', self.script)
        self.assertIn('set "RELEASE_DIR=!CD!\\build\\stage\\!PROFILE!"',
                      self.script)

    def test_a_non_cpu_lane_builds_from_its_own_environment(self):
        """Building the CUDA artifact must not mutate the venv the tests
        run in, which is why it was parked as a release-session item."""
        self.assertIn('set "PYTHON=venv-!PROFILE!\\Scripts\\python.exe"',
                      self.script)
        self.assertIn("dependency_profiles\\!PROFILE!.txt", self.script)

    def test_the_cuda_lane_keeps_the_torch_runtime_the_provider_needs(self):
        """RM-319: onnxruntime_providers_cuda.dll loads cuBLAS out of the
        torch cu130 wheel, so stripping torch would produce the CPU-only
        payload under a CUDA name that this item exists to stop."""
        self.assertIn('if /I "!PROFILE!"=="nvidia" set "NEEDS_TORCH=1"',
                      self.script)
        spec = (ROOT / "VideoSubtitleRemoverPro.spec").read_text(
            encoding="utf-8")
        self.assertIn("needs_torch = build_profile == 'nvidia'", spec)
        self.assertIn("not needs_torch", spec)

    def test_the_build_skips_packaging_it_cannot_do(self):
        """The batch asks the staging module rather than repeating the list."""
        self.assertIn("ships_installer", self.script)
        self.assertIn('set "SHIP_INSTALLER=0"', self.script)
        self.assertIn("goto skip_installer", self.script)
        self.assertIn("--no-installer", self.script)

    def test_no_label_comment_sits_inside_a_parenthesised_block(self):
        """`::` is a label, not a comment.

        cmd parses a label inside a parenthesised block as a drive
        reference and fails the whole block with "The system cannot find
        the drive specified", which is what a `::` comment added to the
        ZIP-only branch did: the build got as far as producing the frozen
        payload and then died before writing any evidence. Comments inside
        a block have to be REM.
        """
        offenders = []
        depth = 0
        for number, line in enumerate(self.script.splitlines(), 1):
            if depth > 0 and line.strip().startswith("::"):
                offenders.append(number)
            depth = max(0, depth + line.count("(") - line.count(")"))
        self.assertEqual(
            offenders, [],
            "build_exe.bat lines with a `::` comment inside a block: "
            + ", ".join(str(item) for item in offenders),
        )

    def test_the_lane_is_stamped_into_the_bundle(self):
        spec = (ROOT / "VideoSubtitleRemoverPro.spec").read_text(
            encoding="utf-8")
        self.assertIn("write_build_profile", spec)
        self.assertIn("build_profile_path", spec)


if __name__ == "__main__":
    unittest.main()
