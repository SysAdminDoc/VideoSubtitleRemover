import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

from backend import release_verification


ROOT = Path(__file__).resolve().parents[1]


class ReleaseVerificationTests(unittest.TestCase):
    def _copy_release_inputs(self, dist_dir: Path):
        for name in release_verification.DOCUMENTS:
            source = ROOT / name
            if source.exists():
                (dist_dir / name).write_bytes(source.read_bytes())
        for name in release_verification.LAUNCHERS:
            source = release_verification.FROZEN_LAUNCHER_SOURCE_DIR / name
            (dist_dir / name).write_bytes(source.read_bytes())

    def _patched_environment(self):
        patches = [
            mock.patch(
                "backend.release_verification.collect_dependency_versions",
                return_value=[{"name": "Pillow", "version": "12.3.0"}],
            ),
            mock.patch(
                "backend.release_verification.release_manifest_status",
                return_value=[{"name": "adapter-policy", "ok": True}],
            ),
            mock.patch(
                "backend.release_verification.release_remote_model_status",
                return_value=[{"name": "remote-model-policy", "ok": True}],
            ),
            mock.patch(
                "backend.release_verification.rapidocr_release_provenance",
                return_value={
                    "package": {"name": "rapidocr", "version": "3.9.0"},
                    "model_count": 3,
                    "missing": False,
                    "packaging_compatible": True,
                },
            ),
            mock.patch(
                "backend.release_verification._tool_version",
                return_value={"available": True, "version": "test-tool"},
            ),
            mock.patch(
                "backend.release_verification._ffmpeg_encoder_status",
                return_value={"available": True, "hasLibvvenc": True},
            ),
            mock.patch(
                "backend.release_verification.probe_ffmpeg_security",
                return_value={
                    "raw": "ffmpeg version 8.1.2",
                    "version": "8.1.2",
                    "parsed": True,
                    "vulnerable": False,
                    "fixed_in": "",
                    "advisories": [],
                    "reason": "no known-vulnerable release floor matched",
                    "available": True,
                    "path": "ffmpeg",
                },
            ),
            mock.patch(
                "backend.release_verification.collect_ffmpeg_capability_profiles",
                return_value={
                    "schema": "vsr.ffmpeg_profiles.v1",
                    "profiles": [
                        {"name": "basic", "available": True, "reason": "ready"},
                        {
                            "name": "advanced_quality",
                            "available": False,
                            "reason": "missing filters: libvmaf",
                        },
                        {
                            "name": "speech_fallback",
                            "available": False,
                            "reason": "missing filters: whisper",
                        },
                        {"name": "modern_codec", "available": True, "reason": "ready"},
                    ],
                },
            ),
            mock.patch(
                "backend.release_verification.collect_onnxruntime_provider_status",
                return_value={
                    "schema": "vsr.onnxruntime_providers.v1",
                    "cuda": {
                        "packageChannel": "cuda12-pypi-stable",
                        "providerAvailable": True,
                        "preloadStatus": {
                            "schema": "vsr.onnxruntime_cuda_preload.v1",
                            "needed": True,
                            "attempted": True,
                            "available": True,
                            "succeeded": True,
                            "callCount": 1,
                            "lastProviders": ["CUDAExecutionProvider"],
                            "error": "",
                        },
                    },
                    "directml": {"providerAvailable": False},
                    "warnings": [],
                },
            ),
            mock.patch(
                "backend.release_verification.collect_opencv_wheel_status",
                return_value={
                    "schema": "vsr.opencv_wheels.v1",
                    "conflict": False,
                    "imported": {
                        "owner": "opencv-python",
                        "version": "4.12.0",
                        "file": "cv2/__init__.py",
                    },
                    "warnings": [],
                },
            ),
            mock.patch(
                "backend.release_verification.collect_opencv_dnn_ocr_status",
                return_value={
                    "schema": "vsr.opencv_dnn_ocr.v1",
                    "eligible": True,
                    "opencvVersion": "5.0.0",
                    "errors": [],
                },
            ),
            mock.patch(
                "backend.release_verification.collect_rapidocr_engine_status",
                return_value={
                    "schema": "vsr.rapidocr_engines.v1",
                    "preferredEngine": "openvino",
                    "preferredProvider": "OpenVINO CPU",
                    "engines": {
                        "openvino": {"available": True},
                        "onnxruntime": {"available": True},
                    },
                    "warnings": [],
                },
            ),
            mock.patch(
                "backend.release_verification._run_smoke",
                return_value={"ran": True, "passed": True, "returncode": 0},
            ),
            mock.patch(
                "backend.release_verification._ffmpeg_subprocess_smoke",
                return_value={
                    "schema": "vsr.ffmpeg_subprocess_smoke.v1",
                    "ran": True,
                    "passed": True,
                    "ffmpegPath": "/usr/bin/ffmpeg",
                    "ffprobePath": "/usr/bin/ffprobe",
                    "ffmpegAvailable": True,
                    "ffprobeAvailable": True,
                    "generate": {"ran": True, "passed": True, "error": ""},
                    "probe": {"ran": True, "passed": True, "error": "", "codec": "rawvideo", "width": 32, "height": 32, "frames": 1},
                    "transcode": {"ran": True, "passed": True, "error": ""},
                    "env": {"PATH": "/usr/bin", "frozen": False},
                    "error": "",
                },
            ),
            mock.patch(
                "backend.release_verification.opencv_libpng_status",
                return_value={
                    "vulnerable": False,
                    "libpng_version": "1.6.54",
                    "fixed_version": "1.6.54",
                },
            ),
        ]
        return patches

    def test_parse_hidden_imports_extracts_unique_module_names(self):
        hidden = release_verification.parse_hidden_imports(
            "--hidden-import cv2 --hidden-import=numpy "
            "--hidden-import tkinter.ttk --hidden-import 'rapidocr' "
            "--hidden-import cv2"
        )
        self.assertEqual(hidden, ("cv2", "numpy", "rapidocr", "tkinter.ttk"))

    def test_parse_runtime_hooks_normalizes_windows_paths(self):
        hooks = release_verification.parse_runtime_hooks(
            "--runtime-hook assets\\runtime_hook_mp.py "
            "--runtime-hook=assets/runtime_hook_mp.py"
        )
        self.assertEqual(hooks, ("assets/runtime_hook_mp.py",))

    def test_parse_excluded_modules_records_lean_release_profile(self):
        excluded = release_verification.parse_excluded_modules(
            "--exclude-module torch --exclude-module=paddleocr "
            "--exclude-module 'easyocr' --exclude-module torch"
        )
        self.assertEqual(excluded, ("easyocr", "paddleocr", "torch"))

    def test_artifact_sbom_excludes_unbundled_environment_packages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dist = root / "dist"
            internal = dist / "_internal"
            internal.mkdir(parents=True)
            (dist / "VideoSubtitleRemoverPro.exe").write_bytes(b"MZapp")
            (internal / "native.dll").write_bytes(b"native-payload")
            analysis = root / "Analysis-00.toc"
            analysis.write_text(
                repr(([
                    (
                        "bundled_pkg.core",
                        str(root / "site-packages" / "bundled_pkg" / "core.py"),
                        "PYMODULE",
                    ),
                    (
                        "native.dll",
                        str(root / "site-packages" / "bundled_pkg" / "native.dll"),
                        "BINARY",
                    ),
                ],)),
                encoding="utf-8",
            )
            sbom = release_verification.build_cyclonedx_sbom(
                [
                    {"name": "bundled-pkg", "version": "1.2.3"},
                    {"name": "excluded-pkg", "version": "9.9.9"},
                    {"name": "PyInstaller", "version": "6.21.0"},
                ],
                dist_dir=dist,
                analysis_path=analysis,
            )

        properties = {
            item["name"]: item["value"]
            for item in sbom["metadata"]["properties"]
        }
        self.assertEqual(properties["vsr:artifactDerived"], "true")
        libraries = {
            item["name"]: item["scope"]
            for item in sbom["components"]
            if item["type"] == "library"
        }
        self.assertEqual(libraries["bundled-pkg"], "required")
        self.assertEqual(libraries["PyInstaller"], "excluded")
        self.assertNotIn("excluded-pkg", libraries)
        native = next(
            item for item in sbom["components"]
            if item["type"] == "file" and item["name"] == "native.dll"
        )
        self.assertEqual(native["hashes"][0]["alg"], "SHA-256")
        self.assertEqual(len(native["hashes"][0]["content"]), 64)

    def test_dependency_audit_uses_only_required_frozen_libraries(self):
        sbom = {
            "components": [
                {
                    "type": "library", "name": "runtime-lib",
                    "version": "1.2.3", "scope": "required",
                },
                {
                    "type": "library", "name": "build-tool",
                    "version": "4.5.6", "scope": "excluded",
                },
                {"type": "file", "name": "native.dll", "scope": "required"},
            ],
        }
        completed = mock.Mock(
            returncode=0,
            stdout=json.dumps({
                "dependencies": [
                    {"name": "runtime-lib", "version": "1.2.3", "vulns": []},
                ],
            }),
            stderr="",
        )
        with mock.patch(
            "backend.release_verification.run_process", return_value=completed,
        ) as run:
            result = release_verification._audit_frozen_dependencies(sbom)

        self.assertTrue(result["passed"])
        self.assertEqual(result["requirements"], ["runtime-lib==1.2.3"])
        command = run.call_args.args[0]
        self.assertIn("--strict", command)
        self.assertIn("--no-deps", command)

    def test_build_release_evidence_records_local_release_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            dist_dir = Path(tmp) / "VideoSubtitleRemoverPro"
            dist_dir.mkdir()
            self._copy_release_inputs(dist_dir)

            with ExitStack() as stack:
                for patch in self._patched_environment():
                    stack.enter_context(patch)
                evidence, hidden_payload, sbom, advisories = release_verification.build_release_evidence(
                    dist_dir=dist_dir,
                    hidden_imports=(
                        "--hidden-import cv2 --hidden-import rapidocr_onnxruntime"
                    ),
                    runtime_hooks="--runtime-hook assets\\runtime_hook_mp.py",
                    collect_data="--collect-data rapidocr",
                )

        self.assertEqual(evidence["schema"], "vsr.release_verification.v1")
        self.assertEqual(evidence["errors"], [])
        self.assertIn("cv2", evidence["hiddenImports"])
        self.assertIn("rapidocr_onnxruntime", evidence["hiddenImports"])
        self.assertEqual(evidence["runtimeHooks"], ["assets/runtime_hook_mp.py"])
        self.assertTrue(all(item["bundled"] for item in evidence["documents"]))
        self.assertTrue(all(item["bundled"] for item in evidence["launchers"]))
        self.assertTrue(all(item["nativeExecutable"] for item in evidence["launchers"]))
        self.assertTrue(evidence["smokeLaunch"]["passed"])
        self.assertEqual(evidence["sbom"]["componentCount"], 1)
        self.assertEqual(evidence["advisories"]["file"], "release-advisories.json")
        self.assertEqual(evidence["advisories"]["blocking"], 0)
        self.assertEqual(
            evidence["releaseTools"]["pyinstallerRuntimeHooks"][0]["name"],
            "assets/runtime_hook_mp.py",
        )
        self.assertTrue(
            evidence["releaseTools"]["pyinstallerRuntimeHooks"][0]["sourceExists"]
        )
        self.assertEqual(
            evidence["releaseTools"]["ffmpegProfiles"]["schema"],
            "vsr.ffmpeg_profiles.v1",
        )
        self.assertEqual(
            evidence["releaseTools"]["onnxRuntimeProviders"]["schema"],
            "vsr.onnxruntime_providers.v1",
        )
        self.assertTrue(
            evidence["releaseTools"]["onnxRuntimeProviders"]["cuda"][
                "preloadStatus"
            ]["succeeded"]
        )
        self.assertEqual(
            evidence["releaseTools"]["opencvWheels"]["schema"],
            "vsr.opencv_wheels.v1",
        )
        self.assertEqual(
            evidence["releaseTools"]["opencvDnnOcr"]["schema"],
            "vsr.opencv_dnn_ocr.v1",
        )
        self.assertTrue(evidence["releaseTools"]["opencvDnnOcr"]["eligible"])
        self.assertEqual(
            evidence["releaseTools"]["rapidocrEngines"]["schema"],
            "vsr.rapidocr_engines.v1",
        )
        self.assertEqual(
            evidence["releaseTools"]["rapidocrEngines"]["preferredProvider"],
            "OpenVINO CPU",
        )
        self.assertFalse(evidence["releaseTools"]["referenceCorpus"]["ran"])
        self.assertEqual(
            evidence["releaseTools"]["ffmpegSubprocessSmoke"]["schema"],
            "vsr.ffmpeg_subprocess_smoke.v1",
        )
        self.assertTrue(evidence["releaseTools"]["ffmpegSubprocessSmoke"]["passed"])
        self.assertTrue(evidence["releaseTools"]["dependencyProfile"]["valid"])
        self.assertEqual(
            len(evidence["releaseTools"]["dependencyProfile"]["constraintSha256"]),
            64,
        )
        self.assertTrue(evidence["rapidocrModels"]["packaging_compatible"])
        self.assertEqual(hidden_payload["schema"], "vsr.release_hidden_imports.v1")
        self.assertEqual(hidden_payload["runtimeHooks"], ["assets/runtime_hook_mp.py"])
        self.assertEqual(sbom["bomFormat"], "CycloneDX")
        self.assertEqual(sbom["components"][0]["name"], "Pillow")
        self.assertEqual(advisories["schema"], "vsr.release_advisories.v1")
        self.assertEqual(advisories["summary"]["blocking"], 0)

    def test_source_bootstrap_launcher_is_rejected_from_frozen_dist(self):
        with tempfile.TemporaryDirectory() as tmp:
            dist_dir = Path(tmp)
            self._copy_release_inputs(dist_dir)
            (dist_dir / "Run_VSR_Pro.bat").write_text(
                "python setup.py --repair\nvenv\\Scripts\\python.exe VideoSubtitleRemover.py\n",
                encoding="utf-8",
            )
            with ExitStack() as stack:
                for patch in self._patched_environment():
                    stack.enter_context(patch)
                evidence, _, _, _ = release_verification.build_release_evidence(
                    dist_dir=dist_dir,
                    runtime_hooks="--runtime-hook assets\\runtime_hook_mp.py",
                )

        self.assertIn(
            "Bundled launcher is not frozen-native: Run_VSR_Pro.bat "
            "(source references: setup.py, videosubtitleremover.py, venv\\)",
            evidence["errors"],
        )

    def test_frozen_smoke_runs_every_entry_point_from_path_with_spaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            dist_dir = Path(tmp) / "dist"
            dist_dir.mkdir()
            (dist_dir / "VideoSubtitleRemoverPro.exe").write_bytes(b"MZ")
            self._copy_release_inputs(dist_dir)
            completed = mock.Mock(returncode=0, stdout="ok", stderr="")
            with mock.patch(
                "backend.release_verification.run_process",
                return_value=completed,
            ) as run, mock.patch(
                "backend.release_verification.shutil.which",
                side_effect=lambda name: "pwsh.exe" if name == "pwsh" else None,
            ):
                smoke = release_verification._run_smoke(dist_dir)

        self.assertTrue(smoke["ran"])
        self.assertTrue(smoke["passed"])
        self.assertEqual(
            [item["name"] for item in smoke["entryPoints"]],
            ["VideoSubtitleRemoverPro.exe", *release_verification.LAUNCHERS],
        )
        self.assertEqual(run.call_count, 4)
        for call in run.call_args_list:
            self.assertIn("Video Subtitle Remover Pro", str(call.kwargs["cwd"]))
            self.assertEqual(call.kwargs["env"]["VSR_LAUNCHER_WAIT"], "1")
            self.assertEqual(call.kwargs["env"]["VSR_LAUNCHER_SMOKE"], "1")

    def test_installer_artifact_and_extracted_payload_are_proven(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            installer = root / "VideoSubtitleRemoverPro-Setup.exe"
            installed = root / "installed" / "VideoSubtitleRemoverPro.exe"
            installed.parent.mkdir()
            installer.write_bytes(b"MZinstaller")
            installed.write_bytes(b"MZinstalled")
            completed = mock.Mock(returncode=0, stdout="ok", stderr="")
            with mock.patch(
                "backend.release_verification.run_process",
                return_value=completed,
            ) as run:
                status = release_verification._installer_status(installer)
                smoke = release_verification._run_installer_smoke(installed)

        self.assertTrue(status["validPortableExecutable"])
        self.assertEqual(len(status["sha256"]), 64)
        self.assertTrue(smoke["passed"])
        self.assertEqual(run.call_args.args[0], [str(installed), "--smoke-test"])
        self.assertEqual(run.call_args.kwargs["env"]["VSR_LAUNCHER_SMOKE"], "1")

    def test_write_release_evidence_writes_json_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dist_dir = root / "dist"
            evidence_dir = root / "evidence"
            dist_dir.mkdir()
            self._copy_release_inputs(dist_dir)

            with ExitStack() as stack:
                for patch in self._patched_environment():
                    stack.enter_context(patch)
                release_verification.write_release_evidence(
                    dist_dir=dist_dir,
                    evidence_dir=evidence_dir,
                    hidden_imports="--hidden-import cv2",
                    runtime_hooks="--runtime-hook assets\\runtime_hook_mp.py",
                    run_smoke=False,
                )

            release_json = json.loads(
                (evidence_dir / "release-verification.json").read_text(encoding="utf-8")
            )
            hidden_json = json.loads(
                (evidence_dir / "release-hidden-imports.json").read_text(encoding="utf-8")
            )
            advisory_json = json.loads(
                (evidence_dir / "release-advisories.json").read_text(encoding="utf-8")
            )
            sbom_json = json.loads(
                (evidence_dir / "sbom.cdx.json").read_text(encoding="utf-8")
            )

        self.assertEqual(release_json["schema"], "vsr.release_verification.v1")
        self.assertEqual(hidden_json["hiddenImports"], ["cv2"])
        self.assertEqual(hidden_json["runtimeHooks"], ["assets/runtime_hook_mp.py"])
        self.assertEqual(advisory_json["schema"], "vsr.release_advisories.v1")
        self.assertEqual(sbom_json["specVersion"], "1.5")

    def test_release_evidence_can_run_reference_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            dist_dir = Path(tmp) / "dist"
            dist_dir.mkdir()
            self._copy_release_inputs(dist_dir)

            corpus_result = {
                "schema": "vsr.reference_corpus.v1",
                "ran": True,
                "passed": True,
                "clipCount": 10,
                "failures": [],
                "error": "",
            }
            with ExitStack() as stack:
                for patch in self._patched_environment():
                    stack.enter_context(patch)
                stack.enter_context(mock.patch(
                    "backend.release_verification._run_reference_corpus",
                    return_value=corpus_result,
                ))
                evidence, _hidden_payload, _sbom, _advisories = (
                    release_verification.build_release_evidence(
                        dist_dir=dist_dir,
                        run_reference_corpus=True,
                        runtime_hooks="--runtime-hook assets\\runtime_hook_mp.py",
                    )
                )

        self.assertEqual(
            evidence["releaseTools"]["referenceCorpus"],
            corpus_result,
        )
        self.assertEqual(evidence["errors"], [])

    def test_release_evidence_reports_reference_corpus_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            dist_dir = Path(tmp) / "dist"
            dist_dir.mkdir()
            self._copy_release_inputs(dist_dir)

            corpus_result = {
                "schema": "vsr.reference_corpus.v1",
                "ran": True,
                "passed": False,
                "clipCount": 10,
                "failures": [{"filename": "clip.mkv", "failures": ["hash"]}],
                "error": "",
            }
            with ExitStack() as stack:
                for patch in self._patched_environment():
                    stack.enter_context(patch)
                stack.enter_context(mock.patch(
                    "backend.release_verification._run_reference_corpus",
                    return_value=corpus_result,
                ))
                evidence, _hidden_payload, _sbom, _advisories = (
                    release_verification.build_release_evidence(
                        dist_dir=dist_dir,
                        run_reference_corpus=True,
                        runtime_hooks="--runtime-hook assets\\runtime_hook_mp.py",
                    )
                )

        self.assertIn("Reference corpus regression failed", evidence["errors"])

    def test_strict_release_fails_on_blocking_advisory(self):
        with tempfile.TemporaryDirectory() as tmp:
            dist_dir = Path(tmp) / "dist"
            dist_dir.mkdir()
            self._copy_release_inputs(dist_dir)

            patches = self._patched_environment()
            patches[0] = mock.patch(
                "backend.release_verification.collect_dependency_versions",
                return_value=[{"name": "torch", "version": "2.5.1"}],
            )

            with ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                with self.assertRaises(SystemExit) as ctx:
                    release_verification.write_release_evidence(
                        dist_dir=dist_dir,
                        evidence_dir=Path(tmp) / "evidence",
                        runtime_hooks="--runtime-hook assets\\runtime_hook_mp.py",
                        strict=True,
                        run_smoke=False,
                    )

        self.assertIn("CVE-2025-32434", str(ctx.exception))

    def test_release_advisories_include_onnxruntime_cuda_warnings(self):
        deps = [{"name": "onnxruntime-gpu", "version": "1.17.3"}]
        with mock.patch(
            "backend.release_verification.collect_onnxruntime_provider_status",
            return_value={
                "schema": "vsr.onnxruntime_providers.v1",
                "cuda": {
                    "packageVersion": "1.17.3",
                    "packageChannel": "legacy-cuda-package",
                },
                "warnings": [
                    {
                        "id": "ORT-CUDA-LEGACY-PACKAGE",
                        "severity": "medium",
                        "message": "legacy CUDA package",
                    }
                ],
            },
        ), mock.patch(
            "backend.release_verification.probe_ffmpeg_security",
            return_value={"vulnerable": False},
        ), mock.patch(
            "backend.release_verification.opencv_libpng_status",
            return_value={
                "vulnerable": False,
                "libpng_version": "1.6.57",
                "fixed_version": "1.6.54",
            },
        ):
            advisories = release_verification.collect_release_advisories(deps)

        self.assertEqual(advisories["summary"]["blocking"], 0)
        ids = {item["id"] for item in advisories["advisories"]}
        self.assertIn("ORT-CUDA-LEGACY-PACKAGE", ids)

    def test_ffmpeg_version_classifier_flags_vulnerable_lines(self):
        from backend import ffmpeg_profiles

        for banner in (
            "ffmpeg version 8.1.1-full_build-www.gyan.dev",
            "ffmpeg version 8.1.0",
            "ffmpeg version 8.0.2",
            "ffmpeg version n8.0.0",
        ):
            status = ffmpeg_profiles.classify_ffmpeg_security(banner)
            self.assertTrue(status["vulnerable"], banner)
            self.assertIn(status["fixed_in"], ("8.1.2", "8.0.3"))
            self.assertIn("CVE-2026-8461", status["advisories"])

        for banner in (
            "ffmpeg version 8.1.2-full_build",
            "ffmpeg version 8.0.3",
            "ffmpeg version 8.1.9",
            "ffmpeg version 8.0.4",
        ):
            status = ffmpeg_profiles.classify_ffmpeg_security(banner)
            self.assertFalse(status["vulnerable"], banner)
            self.assertTrue(status["safe"], banner)
            self.assertEqual(status["classification"], "safe")

        unsupported = ffmpeg_profiles.classify_ffmpeg_security(
            "ffmpeg version 7.1.5"
        )
        self.assertEqual(unsupported["classification"], "unsupported")
        self.assertFalse(unsupported["safe"])

        for banner in ("ffmpeg version 8.2.0", "ffmpeg version 9.0"):
            status = ffmpeg_profiles.classify_ffmpeg_security(banner)
            self.assertEqual(status["classification"], "unknown", banner)
            self.assertFalse(status["safe"], banner)

        snapshot = ffmpeg_profiles.classify_ffmpeg_security(
            "ffmpeg version N-119876-g1a2b3c"
        )
        self.assertFalse(snapshot["parsed"])
        self.assertFalse(snapshot["vulnerable"])
        self.assertEqual(snapshot["classification"], "unknown")

    def test_ffmpeg_security_advisory_blocks_vulnerable_runtime(self):
        advisory = release_verification.ffmpeg_security_advisory({
            "vulnerable": True,
            "version": "8.1.1",
            "fixed_in": "8.1.2",
            "advisories": ["CVE-2026-8461", "CVE-2026-30999"],
        })
        self.assertIsNotNone(advisory)
        self.assertTrue(advisory["blocking"])
        self.assertEqual(advisory["severity"], "high")
        self.assertEqual(advisory["fixedIn"], "8.1.2")
        self.assertIsNone(
            release_verification.ffmpeg_security_advisory({"vulnerable": False})
        )

    def test_ffmpeg_security_advisory_blocks_unsupported_and_unknown(self):
        from backend import ffmpeg_profiles

        unsupported = release_verification.ffmpeg_security_advisory(
            ffmpeg_profiles.classify_ffmpeg_security("ffmpeg version 7.1.5")
        )
        unknown = release_verification.ffmpeg_security_advisory(
            ffmpeg_profiles.classify_ffmpeg_security(
                "ffmpeg version N-119876-g1a2b3c"
            )
        )

        self.assertEqual(unsupported["id"], "FFMPEG-UNSUPPORTED-BRANCH")
        self.assertTrue(unsupported["blocking"])
        self.assertEqual(unknown["id"], "FFMPEG-UNCLASSIFIED-VERSION")
        self.assertTrue(unknown["blocking"])

    def test_pyinstaller_below_floor_blocks(self):
        deps = [{"name": "pyinstaller", "version": "6.6.0"}]
        with mock.patch(
            "backend.release_verification.probe_ffmpeg_security",
            return_value={"vulnerable": False},
        ), mock.patch(
            "backend.release_verification.nsis_security_advisory",
            return_value=None,
        ):
            advisories = release_verification.collect_release_advisories(deps)
        ids = {a["id"]: a for a in advisories["advisories"]}
        self.assertIn("CVE-2025-59042", ids)
        self.assertTrue(ids["CVE-2025-59042"]["blocking"])

    def test_pyinstaller_at_floor_passes(self):
        deps = [{"name": "pyinstaller", "version": "6.10.0"}]
        with mock.patch(
            "backend.release_verification.probe_ffmpeg_security",
            return_value={"vulnerable": False},
        ), mock.patch(
            "backend.release_verification.nsis_security_advisory",
            return_value=None,
        ):
            advisories = release_verification.collect_release_advisories(deps)
        ids = {a["id"] for a in advisories["advisories"]}
        self.assertNotIn("CVE-2025-59042", ids)

    def test_nsis_advisory_flags_old_toolchain(self):
        advisory = release_verification.nsis_security_advisory("3.11")
        self.assertIsNotNone(advisory)
        self.assertTrue(advisory["blocking"])
        self.assertEqual(advisory["fixedIn"], "3.12")
        self.assertEqual(advisory["id"], "NSIS-2026-LOW-IL-TEMP")
        self.assertIsNone(release_verification.nsis_security_advisory("3.12"))
        self.assertIsNone(release_verification.nsis_security_advisory(""))

    def test_pillow_below_security_floor_blocks(self):
        deps = [{"name": "Pillow", "version": "12.2.0"}]
        with mock.patch(
            "backend.release_verification.probe_ffmpeg_security",
            return_value={"vulnerable": False},
        ), mock.patch(
            "backend.release_verification.nsis_security_advisory",
            return_value=None,
        ):
            advisories = release_verification.collect_release_advisories(deps)
        ids = {item["id"]: item for item in advisories["advisories"]}
        self.assertTrue(ids["PILLOW-12.3-SECURITY-FLOOR"]["blocking"])
        self.assertEqual(
            ids["PILLOW-12.3-SECURITY-FLOOR"]["fixedIn"], "12.3.0"
        )

    def test_strict_release_fails_on_vulnerable_ffmpeg(self):
        deps = [{"name": "Pillow", "version": "12.3.0"}]
        with mock.patch(
            "backend.release_verification.probe_ffmpeg_security",
            return_value={
                "vulnerable": True,
                "version": "8.1.1",
                "fixed_in": "8.1.2",
                "advisories": ["CVE-2026-8461"],
            },
        ):
            advisories = release_verification.collect_release_advisories(deps)
        self.assertGreaterEqual(advisories["summary"]["blocking"], 1)
        ids = {item["id"] for item in advisories["advisories"]}
        self.assertIn("CVE-2026-8461", ids)

    def test_ffmpeg_subprocess_smoke_passes_with_real_tools(self):
        smoke = release_verification._ffmpeg_subprocess_smoke(timeout=30.0)
        if not smoke["ffmpegAvailable"] or not smoke["ffprobeAvailable"]:
            self.skipTest("ffmpeg/ffprobe not installed")
        self.assertTrue(smoke["ran"])
        self.assertTrue(smoke["passed"])
        self.assertTrue(smoke["generate"]["passed"])
        self.assertTrue(smoke["probe"]["passed"])
        self.assertEqual(smoke["probe"]["width"], 32)
        self.assertEqual(smoke["probe"]["height"], 32)
        self.assertTrue(smoke["transcode"]["passed"])

    def test_ffmpeg_subprocess_smoke_fails_without_ffmpeg(self):
        with mock.patch("backend.release_verification.shutil.which", return_value=None):
            smoke = release_verification._ffmpeg_subprocess_smoke()
        self.assertFalse(smoke["ran"])
        self.assertFalse(smoke["passed"])
        self.assertIn("not on PATH", smoke["error"])

    def test_release_evidence_reports_ffmpeg_smoke_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            dist_dir = Path(tmp) / "dist"
            dist_dir.mkdir()
            self._copy_release_inputs(dist_dir)

            failed_smoke = {
                "schema": "vsr.ffmpeg_subprocess_smoke.v1",
                "ran": True,
                "passed": False,
                "error": "ffprobe failed on generated fixture",
            }
            patches = self._patched_environment()
            with ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                stack.enter_context(mock.patch(
                    "backend.release_verification._ffmpeg_subprocess_smoke",
                    return_value=failed_smoke,
                ))
                evidence, _, _, _ = release_verification.build_release_evidence(
                    dist_dir=dist_dir,
                    runtime_hooks="--runtime-hook assets\\runtime_hook_mp.py",
                )

        self.assertIn(
            "FFmpeg subprocess smoke failed: ffprobe failed on generated fixture",
            evidence["errors"],
        )

    def test_opencv_libpng_blocks_until_runtime_is_fixed(self):
        deps = [{"name": "opencv-python", "version": "4.13.0.92"}]
        with mock.patch(
            "backend.release_verification.probe_ffmpeg_security",
            return_value={"vulnerable": False},
        ), mock.patch(
            "backend.release_verification.opencv_libpng_status",
            return_value={
                "vulnerable": True,
                "libpng_version": "1.6.43",
                "fixed_version": "1.6.54",
            },
        ):
            vulnerable = release_verification.collect_release_advisories(deps)
        with mock.patch(
            "backend.release_verification.probe_ffmpeg_security",
            return_value={"vulnerable": False},
        ), mock.patch(
            "backend.release_verification.opencv_libpng_status",
            return_value={
                "vulnerable": False,
                "libpng_version": "1.6.54",
                "fixed_version": "1.6.54",
            },
        ):
            fixed = release_verification.collect_release_advisories(deps)

        self.assertEqual(vulnerable["advisories"][0]["id"], "CVE-2026-22801")
        self.assertTrue(vulnerable["advisories"][0]["blocking"])
        self.assertFalse(vulnerable["advisories"][0]["allowed"])
        self.assertEqual(fixed["advisories"], [])


class LocalBuildScriptTests(unittest.TestCase):
    def setUp(self):
        self.bat = (ROOT / "build_exe.bat").read_text(encoding="utf-8")
        self.nsi = (ROOT / "installer" / "vsr.nsi").read_text(encoding="utf-8")

    def test_build_script_generates_local_release_evidence(self):
        self.assertIn("-m backend.release_verification", self.bat)
        self.assertIn("--hidden-imports", self.bat)
        self.assertIn("--runtime-hooks", self.bat)
        self.assertIn("--collect-data", self.bat)
        self.assertIn("--analysis-path", self.bat)
        self.assertIn("--run-dependency-audit", self.bat)
        self.assertIn("--quality strict", self.bat)
        self.assertIn("-m unittest discover -s tests -q", self.bat)
        self.assertIn(
            "-m PyInstaller --noconfirm --clean VideoSubtitleRemoverPro.spec",
            self.bat,
        )
        self.assertIn("installer\\vsr.nsi", self.bat)
        self.assertIn("/DVSR_SMOKE_BUILD=1", self.bat)
        self.assertIn("--installer-smoke-executable", self.bat)
        self.assertIn("--runtime-hook assets\\runtime_hook_mp.py", self.bat)
        self.assertNotIn("pause", self.bat.lower())
        self.assertIn("release-verification.json", self.bat)
        self.assertIn("release-hidden-imports.json", self.bat)
        self.assertIn("release-advisories.json", self.bat)
        self.assertIn("sbom.cdx.json", self.bat)
        self.assertIn("pip-audit.json", self.bat)
        self.assertIn("call :maybe_collect_data rapidocr", self.bat)
        self.assertIn("--hidden-import backend.opencv_ocr", self.bat)
        self.assertIn("call :maybe_collect_data rapidocr_onnxruntime", self.bat)
        self.assertIn("--add-data locale;locale", self.bat)
        self.assertIn("Run_VSR_Pro.bat", self.bat)
        self.assertIn("Run_VSR_Pro_Debug.bat", self.bat)
        self.assertIn("Run_VSR_Pro.ps1", self.bat)
        self.assertIn('"assets\\frozen\\%%F"', self.bat)
        self.assertNotIn(
            "README.md LICENSE CHANGELOG.md Run_VSR_Pro.bat",
            self.bat,
        )

    def test_frozen_launcher_assets_do_not_bootstrap_source(self):
        forbidden = ("setup.py", "videosubtitleremover.py", "venv\\")
        for name in release_verification.LAUNCHERS:
            text = (
                release_verification.FROZEN_LAUNCHER_SOURCE_DIR / name
            ).read_text(encoding="utf-8").lower()
            self.assertIn("videosubtitleremoverpro.exe", text)
            for token in forbidden:
                self.assertNotIn(token, text)

    def test_pytorch_lama_hidden_import_is_explicit_opt_in(self):
        default_line = next(
            line for line in self.bat.splitlines()
            if line.startswith('set "HIDDEN_IMPORTS=')
        )
        self.assertNotIn("simple_lama_inpainting", default_line)
        self.assertIn("VSR_ENABLE_PYTORCH_LAMA", self.bat)
        self.assertIn("call :maybe_hidden_import simple_lama_inpainting", self.bat)

    def test_no_torch_directml_hidden_import(self):
        self.assertNotIn("torch_directml", self.bat)
        self.assertNotIn("torch-directml", self.bat)

    def test_nsis_floor_and_running_app_guard_use_bundled_capabilities(self):
        self.assertIn("0x030C0000", self.nsi)
        self.assertIn("OpenMutexW", self.nsi)
        self.assertNotIn("FindProcDLL::FindProc", self.nsi)
        self.assertIn("VSR_SMOKE_BUILD", self.nsi)
        self.assertIn("RequestExecutionLevel user", self.nsi)

    def test_entrypoint_and_runtime_hook_freeze_support_before_heavy_imports(self):
        entry = (ROOT / "VideoSubtitleRemover.py").read_text(encoding="utf-8")
        self.assertLess(
            entry.index("multiprocessing.freeze_support()"),
            entry.index("import logging"),
        )
        hook = (ROOT / "assets" / "runtime_hook_mp.py").read_text(encoding="utf-8")
        self.assertIn("multiprocessing.freeze_support()", hook)


if __name__ == "__main__":
    unittest.main()
