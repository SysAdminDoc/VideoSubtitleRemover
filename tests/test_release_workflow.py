import json
import tempfile
import unittest
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
                return_value={"package": "rapidocr", "modelCount": 2, "missing": []},
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
                "backend.release_verification._run_smoke",
                return_value={"ran": True, "passed": True, "returncode": 0},
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

    def test_build_release_evidence_records_local_release_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            dist_dir = Path(tmp) / "VideoSubtitleRemoverPro"
            dist_dir.mkdir()
            self._copy_release_inputs(dist_dir)

            patches = self._patched_environment()
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
                evidence, hidden_payload, sbom = release_verification.build_release_evidence(
                    dist_dir=dist_dir,
                    hidden_imports=(
                        "--hidden-import cv2 --hidden-import rapidocr_onnxruntime"
                    ),
                    collect_data="--collect-data rapidocr",
                )

        self.assertEqual(evidence["schema"], "vsr.release_verification.v1")
        self.assertEqual(evidence["errors"], [])
        self.assertIn("cv2", evidence["hiddenImports"])
        self.assertIn("rapidocr_onnxruntime", evidence["hiddenImports"])
        self.assertTrue(all(item["bundled"] for item in evidence["documents"]))
        self.assertTrue(all(item["bundled"] for item in evidence["launchers"]))
        self.assertTrue(evidence["smokeLaunch"]["passed"])
        self.assertEqual(evidence["sbom"]["componentCount"], 1)
        self.assertEqual(hidden_payload["schema"], "vsr.release_hidden_imports.v1")
        self.assertEqual(sbom["bomFormat"], "CycloneDX")
        self.assertEqual(sbom["components"][0]["name"], "Pillow")

    def test_write_release_evidence_writes_json_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dist_dir = root / "dist"
            evidence_dir = root / "evidence"
            dist_dir.mkdir()
            self._copy_release_inputs(dist_dir)

            patches = self._patched_environment()
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
                release_verification.write_release_evidence(
                    dist_dir=dist_dir,
                    evidence_dir=evidence_dir,
                    hidden_imports="--hidden-import cv2",
                    run_smoke=False,
                )

            release_json = json.loads(
                (evidence_dir / "release-verification.json").read_text(encoding="utf-8")
            )
            hidden_json = json.loads(
                (evidence_dir / "release-hidden-imports.json").read_text(encoding="utf-8")
            )
            sbom_json = json.loads(
                (evidence_dir / "sbom.cdx.json").read_text(encoding="utf-8")
            )

        self.assertEqual(release_json["schema"], "vsr.release_verification.v1")
        self.assertEqual(hidden_json["hiddenImports"], ["cv2"])
        self.assertEqual(sbom_json["specVersion"], "1.5")


class LocalBuildScriptTests(unittest.TestCase):
    def setUp(self):
        self.bat = (ROOT / "build_exe.bat").read_text(encoding="utf-8")

    def test_build_script_generates_local_release_evidence(self):
        self.assertIn("-m backend.release_verification", self.bat)
        self.assertIn("--hidden-imports", self.bat)
        self.assertIn("--collect-data", self.bat)
        self.assertIn("release-verification.json", self.bat)
        self.assertIn("release-hidden-imports.json", self.bat)
        self.assertIn("sbom.cdx.json", self.bat)
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


if __name__ == "__main__":
    unittest.main()
