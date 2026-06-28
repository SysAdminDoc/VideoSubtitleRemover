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
        for name in release_verification.DOCUMENTS + release_verification.LAUNCHERS:
            source = ROOT / name
            if source.exists():
                (dist_dir / name).write_bytes(source.read_bytes())

    def _patched_environment(self):
        patches = [
            mock.patch(
                "backend.release_verification.collect_dependency_versions",
                return_value=[{"name": "Pillow", "version": "12.2.0"}],
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
            evidence["releaseTools"]["rapidocrEngines"]["schema"],
            "vsr.rapidocr_engines.v1",
        )
        self.assertEqual(
            evidence["releaseTools"]["rapidocrEngines"]["preferredProvider"],
            "OpenVINO CPU",
        )
        self.assertFalse(evidence["releaseTools"]["referenceCorpus"]["ran"])
        self.assertTrue(evidence["rapidocrModels"]["packaging_compatible"])
        self.assertEqual(hidden_payload["schema"], "vsr.release_hidden_imports.v1")
        self.assertEqual(hidden_payload["runtimeHooks"], ["assets/runtime_hook_mp.py"])
        self.assertEqual(sbom["bomFormat"], "CycloneDX")
        self.assertEqual(sbom["components"][0]["name"], "Pillow")
        self.assertEqual(advisories["schema"], "vsr.release_advisories.v1")
        self.assertEqual(advisories["summary"]["blocking"], 0)

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
        ):
            advisories = release_verification.collect_release_advisories(deps)

        self.assertEqual(advisories["summary"]["blocking"], 0)
        ids = {item["id"] for item in advisories["advisories"]}
        self.assertIn("ORT-CUDA-LEGACY-PACKAGE", ids)

    def test_opencv_libpng_exception_is_removed_when_runtime_is_fixed(self):
        deps = [{"name": "opencv-python", "version": "4.13.0.92"}]
        with mock.patch(
            "backend.release_verification.opencv_libpng_status",
            return_value={
                "vulnerable": True,
                "libpng_version": "1.6.43",
                "fixed_version": "1.6.54",
            },
        ):
            vulnerable = release_verification.collect_release_advisories(deps)
        with mock.patch(
            "backend.release_verification.opencv_libpng_status",
            return_value={
                "vulnerable": False,
                "libpng_version": "1.6.54",
                "fixed_version": "1.6.54",
            },
        ):
            fixed = release_verification.collect_release_advisories(deps)

        self.assertEqual(vulnerable["advisories"][0]["id"], "CVE-2026-22801")
        self.assertTrue(vulnerable["advisories"][0]["allowed"])
        self.assertEqual(fixed["advisories"], [])


class LocalBuildScriptTests(unittest.TestCase):
    def setUp(self):
        self.bat = (ROOT / "build_exe.bat").read_text(encoding="utf-8")

    def test_build_script_generates_local_release_evidence(self):
        self.assertIn("-m backend.release_verification", self.bat)
        self.assertIn("--hidden-imports", self.bat)
        self.assertIn("--runtime-hooks", self.bat)
        self.assertIn("--collect-data", self.bat)
        self.assertIn("--runtime-hook assets\\runtime_hook_mp.py", self.bat)
        self.assertNotIn("pause", self.bat.lower())
        self.assertIn("release-verification.json", self.bat)
        self.assertIn("release-hidden-imports.json", self.bat)
        self.assertIn("release-advisories.json", self.bat)
        self.assertIn("sbom.cdx.json", self.bat)
        self.assertIn("call :maybe_collect_data rapidocr", self.bat)
        self.assertIn("call :maybe_collect_data rapidocr_onnxruntime", self.bat)
        self.assertIn("Run_VSR_Pro.bat", self.bat)
        self.assertIn("Run_VSR_Pro_Debug.bat", self.bat)
        self.assertIn("Run_VSR_Pro.ps1", self.bat)

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
