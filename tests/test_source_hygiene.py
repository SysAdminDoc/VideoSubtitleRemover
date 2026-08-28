import ast
import json
from pathlib import Path
import subprocess
import sys
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "venv",
}


def _is_excluded(path: Path) -> bool:
    try:
        relative = path.relative_to(ROOT)
    except ValueError:
        return True
    return any(part in EXCLUDED_DIRS for part in relative.parts)


def _source_files():
    for suffix in ("*.py", "*.bat"):
        for path in ROOT.rglob(suffix):
            if not _is_excluded(path) and path.is_file():
                yield path


def _doc_files():
    for rel in ("README.md", "docs/architecture.md"):
        path = ROOT / rel
        if path.is_file():
            yield path


class SourceHygieneTests(unittest.TestCase):
    def test_cli_main_is_a_small_orchestrator_with_named_phases(self):
        tree = ast.parse(
            (ROOT / "backend" / "cli.py").read_text(encoding="utf-8")
        )
        functions = {
            node.name: node for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        for name in (
            "_build_parser",
            "_handle_utility_actions",
            "_prepare_cli_args",
            "_build_processing_config",
            "_apply_cli_config_overlays",
            "_run_soft_subtitle_modes",
            "_run_processing",
        ):
            self.assertIn(name, functions)

        main = functions["main"]
        self.assertLessEqual(main.end_lineno - main.lineno + 1, 80)
        nested = [
            node for node in ast.walk(main)
            if isinstance(node, ast.FunctionDef) and node is not main
        ]
        self.assertEqual(nested, [])

    def test_settings_builder_delegates_to_named_groups(self):
        # The UI builder methods were extracted from app.py into the
        # LayoutBuildMixin module; they compose onto VideoSubtitleRemoverApp.
        tree = ast.parse(
            (ROOT / "gui" / "layout_build.py").read_text(encoding="utf-8")
        )
        mixin = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "LayoutBuildMixin"
        )
        methods = {
            node.name: node for node in mixin.body
            if isinstance(node, ast.FunctionDef)
        }
        for name in (
            "_build_profile_settings_group",
            "_build_workflow_settings_group",
            "_build_sttn_settings_group",
            "_build_detection_settings_group",
            "_build_output_settings_group",
            "_build_range_settings_group",
            "_build_performance_settings_groups",
            "_build_accessibility_storage_settings",
        ):
            self.assertIn(name, methods)
        entry = methods["_build_settings_section"]
        self.assertLessEqual(entry.end_lineno - entry.lineno + 1, 60)

    def test_ruff_baseline_and_release_gate_are_explicit(self):
        config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        lint = config["tool"]["ruff"]["lint"]
        # RM-323: B905 is part of the baseline now. A bare zip() in the
        # inpaint path silently processed the shorter prefix and left the
        # rest of the video untouched.
        self.assertEqual(lint["select"], ["E4", "E7", "E9", "F", "B905"])
        self.assertNotIn("ignore", lint)
        self.assertEqual(
            lint["per-file-ignores"],
            {
                "VideoSubtitleRemover.py": ["E402"],
                "backend/processor.py": ["E402"],
            },
        )

        build_script = (ROOT / "build_exe.bat").read_text(encoding="ascii")
        self.assertIn('"ruff==0.15.20"', build_script)
        self.assertIn(
            "-m ruff check backend gui scripts VideoSubtitleRemover.py --no-cache",
            build_script,
        )
        self.assertIn('"%PYTHON%" scripts\\generate_cli_reference.py', build_script)

    def test_generated_cli_and_config_reference_is_current(self):
        check = subprocess.run(
            [sys.executable, "scripts/generate_cli_reference.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(check.returncode, 0, check.stderr or check.stdout)

        dump = subprocess.run(
            [sys.executable, "-m", "backend.processor", "--dump-cli-reference"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(dump.returncode, 0, dump.stderr or dump.stdout)
        payload = json.loads(dump.stdout)
        self.assertEqual(payload["schema"], "vsr.cli_reference.v1")
        internal = [
            option["flags"][0]
            for option in payload["options"]
            if option["internal"]
        ]
        self.assertEqual(internal, ["--dump-cli-reference"])
        for option in payload["options"]:
            self.assertIn(option["category"], payload["categories"])
            self.assertIsInstance(option["deprecated"], bool)
            self.assertIn("default", option)
            self.assertIn("range", option)

        help_result = subprocess.run(
            [sys.executable, "-m", "backend.processor", "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        for category in payload["categories"]:
            if category != "Diagnostics and automation":
                self.assertIn(category + ":", help_result.stdout)
        self.assertIn("Diagnostics and automation:", help_result.stdout)

    # RM-334: an exempt launch says so on the spot, with a reason. The
    # marker has to be in the comment block attached to the call, so it
    # reads as part of the code rather than as a list somewhere else that
    # nobody updates. This used to be a nine-line lookback, which let the
    # reason written for one launch silently cover a different launch
    # further down the same function.
    POLICY_EXEMPT_MARKER = "subprocess-policy-exempt:"

    # setup.py bootstraps the environment that makes `backend` importable in
    # the first place, so it cannot import from it. Every other root-level
    # module is in scope. Matched on the path from the repo root, not on the
    # basename: a `setup.py` in any subpackage is not this one.
    POLICY_EXEMPT_FILES = {"setup.py"}

    # Everything in subprocess that starts a child process. The first
    # version of this listed run and Popen, so check_output, check_call,
    # call, getoutput and getstatusoutput all walked past it.
    SUBPROCESS_LAUNCHERS = frozenset({
        "run", "Popen", "call", "check_call", "check_output",
        "getoutput", "getstatusoutput",
    })
    # The os-level equivalents. os.system takes a shell string, which is
    # the exact thing the policy exists to validate. os.startfile is not
    # here: it hands a path to the registered handler and has no argument
    # list, so there is nothing for the policy to validate and an exemption
    # comment on each of the eight "open the output folder" buttons would
    # say nothing.
    OS_LAUNCHERS = frozenset({
        "system", "popen", "spawnl", "spawnle", "spawnlp", "spawnlpe",
        "spawnv", "spawnve", "spawnvp", "spawnvpe", "execl", "execle",
        "execlp", "execlpe", "execv", "execve", "execvp", "execvpe",
    })

    @staticmethod
    def _launcher_names(tree, module: str, attributes):
        """Local names that reach `module`, and bare names bound to a launcher.

        `import subprocess as sp` and `from subprocess import check_output as
        _out` both used to slip past a gate that only knew the literal name
        `subprocess`.
        """
        modules = set()
        bare = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == module or alias.name.startswith(
                            module + "."):
                        modules.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module == module:
                for alias in node.names:
                    if alias.name in attributes:
                        bare.add(alias.asname or alias.name)
            elif isinstance(node, ast.Assign):
                # `_sp = subprocess` rebinds the module under a new name.
                if (isinstance(node.value, ast.Name)
                        and node.value.id in modules):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            modules.add(target.id)
        return modules, bare

    def _raw_launch_sites(self, roots):
        """Every raw child-process launch, minus the ones that explain
        themselves."""
        policy = ROOT / "backend" / "subprocess_policy.py"
        offenders = []
        for root in roots:
            base = ROOT / root
            paths = (
                sorted(base.rglob("*.py")) if base.is_dir() else [base])
            for path in paths:
                if path == policy or _is_excluded(path):
                    continue
                relative = path.relative_to(ROOT).as_posix()
                if relative in self.POLICY_EXEMPT_FILES:
                    continue
                source = path.read_text(encoding="utf-8")
                lines = source.split("\n")
                tree = ast.parse(source, filename=str(path))

                def _exempt(lineno: int, lines=lines) -> bool:
                    # Walk up through the comment block directly above the
                    # call and stop at the first line of real code.
                    index = lineno - 2
                    while index >= 0:
                        stripped = lines[index].strip()
                        if not stripped:
                            index -= 1
                            continue
                        if not stripped.startswith("#"):
                            return False
                        if self.POLICY_EXEMPT_MARKER in stripped:
                            return True
                        index -= 1
                    return False

                sp_modules, sp_bare = self._launcher_names(
                    tree, "subprocess", self.SUBPROCESS_LAUNCHERS)
                os_modules, os_bare = self._launcher_names(
                    tree, "os", self.OS_LAUNCHERS)

                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call):
                        continue
                    func = node.func
                    hit = False
                    if isinstance(func, ast.Attribute):
                        owner = func.value
                        if isinstance(owner, ast.Name):
                            hit = (
                                (owner.id in sp_modules
                                 and func.attr in self.SUBPROCESS_LAUNCHERS)
                                or (owner.id in os_modules
                                    and func.attr in self.OS_LAUNCHERS)
                            )
                    elif isinstance(func, ast.Name):
                        hit = func.id in sp_bare or func.id in os_bare
                    if hit and not _exempt(node.lineno):
                        offenders.append(
                            f"{path.relative_to(ROOT)}:{node.lineno}")
        return offenders

    def test_launches_go_through_subprocess_policy(self):
        """RM-334: the gate covers gui/ and tools/, not only backend/.

        Eleven launch sites outside backend/ were bypassing the shared
        policy's argument validation, including the ones that shell out to
        nvidia-smi and to the release smoke.
        """
        offenders = self._raw_launch_sites(
            ("backend", "gui", "tools", "scripts"))
        self.assertEqual(
            offenders,
            [],
            "Raw child-process launch outside subprocess_policy: "
            + ", ".join(offenders),
        )

    def test_the_gate_catches_a_raw_launch_and_honours_an_exemption(self):
        import tempfile as _tempfile

        raw = (
            "import subprocess\n"
            "def go():\n"
            "    return subprocess.run(['x'])\n"
        )
        exempt = (
            "import subprocess\n"
            "def go():\n"
            "    # subprocess-policy-exempt: this is a test fixture\n"
            "    return subprocess.run(['x'])\n"
        )
        with _tempfile.TemporaryDirectory(dir=str(ROOT)) as tmpdir:
            folder = Path(tmpdir)
            (folder / "raw.py").write_text(raw, encoding="utf-8")
            self.assertTrue(
                self._raw_launch_sites((folder.relative_to(ROOT),)))
            (folder / "raw.py").write_text(exempt, encoding="utf-8")
            self.assertEqual(
                self._raw_launch_sites((folder.relative_to(ROOT),)), [])

    def test_the_gate_catches_the_launches_it_used_to_walk_past(self):
        """Each of these starts a child process and none of them is
        subprocess.run or subprocess.Popen."""
        import tempfile as _tempfile

        cases = {
            "aliased module": (
                "import subprocess as sp\n"
                "def go():\n"
                "    return sp.run(['x'])\n"
            ),
            "rebound module": (
                "import subprocess\n"
                "_sp = subprocess\n"
                "def go():\n"
                "    return _sp.Popen(['x'])\n"
            ),
            "check_output": (
                "import subprocess\n"
                "def go():\n"
                "    return subprocess.check_output(['x'])\n"
            ),
            "bare call import": (
                "from subprocess import check_call\n"
                "def go():\n"
                "    return check_call(['x'])\n"
            ),
            "renamed bare import": (
                "from subprocess import getoutput as _out\n"
                "def go():\n"
                "    return _out('x')\n"
            ),
            "os.system": (
                "import os\n"
                "def go():\n"
                "    return os.system('x')\n"
            ),
            "os.popen": (
                "import os\n"
                "def go():\n"
                "    return os.popen('x').read()\n"
            ),
            "os.execv": (
                "import os\n"
                "def go():\n"
                "    return os.execv('/bin/x', ['x'])\n"
            ),
        }
        with _tempfile.TemporaryDirectory(dir=str(ROOT)) as tmpdir:
            folder = Path(tmpdir)
            for label, body in cases.items():
                with self.subTest(case=label):
                    (folder / "raw.py").write_text(body, encoding="utf-8")
                    self.assertTrue(
                        self._raw_launch_sites((folder.relative_to(ROOT),)),
                        label,
                    )

    def test_an_exemption_does_not_cover_a_later_unrelated_launch(self):
        """The marker used to reach nine lines down the file."""
        import tempfile as _tempfile

        body = (
            "import subprocess\n"
            "def go():\n"
            "    # subprocess-policy-exempt: the probe below is validated\n"
            "    first = subprocess.run(['probe'])\n"
            "    a = 1\n"
            "    return subprocess.run(['unrelated'], shell=True), first, a\n"
        )
        with _tempfile.TemporaryDirectory(dir=str(ROOT)) as tmpdir:
            folder = Path(tmpdir)
            (folder / "raw.py").write_text(body, encoding="utf-8")
            offenders = self._raw_launch_sites((folder.relative_to(ROOT),))
            self.assertEqual(len(offenders), 1, offenders)
            self.assertTrue(offenders[0].endswith(":6"), offenders)

    def test_a_multi_line_reason_still_counts_as_attached(self):
        """The window is the comment block, not a line count, so a reason
        that needs four lines to explain itself is not cut in half."""
        import tempfile as _tempfile

        body = (
            "import subprocess\n"
            "def go():\n"
            "    # subprocess-policy-exempt: this one is special because\n"
            "    # the reason takes several lines to write down, and\n"
            "    # truncating it would make the exemption unreadable\n"
            "    # rather than making the launch any safer.\n"
            "    return subprocess.run(['x'])\n"
        )
        with _tempfile.TemporaryDirectory(dir=str(ROOT)) as tmpdir:
            folder = Path(tmpdir)
            (folder / "raw.py").write_text(body, encoding="utf-8")
            self.assertEqual(
                self._raw_launch_sites((folder.relative_to(ROOT),)), [])

    def test_the_setup_exemption_is_that_one_file_and_not_every_setup_py(self):
        import tempfile as _tempfile

        body = (
            "import subprocess\n"
            "def go():\n"
            "    return subprocess.run(['x'])\n"
        )
        with _tempfile.TemporaryDirectory(dir=str(ROOT)) as tmpdir:
            folder = Path(tmpdir)
            (folder / "setup.py").write_text(body, encoding="utf-8")
            self.assertTrue(
                self._raw_launch_sites((folder.relative_to(ROOT),)),
                "a setup.py in a subpackage is not the bootstrap one",
            )

    def test_the_setup_exemption_states_its_reason(self):
        """setup.py is exempt because it creates the venv backend lives in."""
        self.assertEqual(self.POLICY_EXEMPT_FILES, {"setup.py"})
        self.assertTrue((ROOT / "setup.py").is_file())
        source = (ROOT / "tests" / "test_source_hygiene.py").read_text(
            encoding="utf-8")
        self.assertIn("bootstraps the environment", source)

    def test_python_and_batch_sources_are_ascii_only(self):
        offenders = []
        for path in _source_files():
            data = path.read_bytes()
            for lineno, line in enumerate(data.splitlines(), 1):
                if any(byte > 0x7F for byte in line):
                    rel = path.relative_to(ROOT)
                    offenders.append(f"{rel}:{lineno}")
                    break

        self.assertEqual(
            offenders,
            [],
            "Non-ASCII bytes found in source files: " + ", ".join(offenders),
        )

    def test_language_support_docs_do_not_repeat_legacy_claim(self):
        stale_phrases = ("12-language support", "12 language support")
        offenders = []
        for path in _doc_files():
            text = path.read_text(encoding="utf-8").lower()
            for phrase in stale_phrases:
                if phrase in text:
                    offenders.append(f"{path.relative_to(ROOT)}:{phrase}")
        self.assertEqual(offenders, [])

        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("52 selectable OCR language codes", readme)


if __name__ == "__main__":
    unittest.main()
