"""SDK-selection checks; no PyTorch, TensorRT, or GPU installation is required.

Run with ``python3 tests/toolchains/test_drive_tensorrt_repository.py``.
Set BAZEL to a Bazel 8 binary to also exercise the repository rule with fake SDKs.
Bazel may need network access to resolve its standard module dependencies.
"""

import ast
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
RULE = REPO / "toolchains/drive_tensorrt_repository.bzl"


def named_call(source, name):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg == "name" and ast.literal_eval(keyword.value) == name:
                    return node
    raise AssertionError(f"Missing declaration: {name}")


class TensorRTSelectionTest(unittest.TestCase):
    def test_drive_rule_has_no_download_path(self):
        tree = ast.parse(RULE.read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertFalse(
            any(
                isinstance(node.func, ast.Attribute)
                and node.func.attr in ("download", "download_and_extract", "execute")
                for node in calls
            )
        )
        rule = next(
            node
            for node in calls
            if isinstance(node.func, ast.Name) and node.func.id == "repository_rule"
        )
        environ = next(k.value for k in rule.keywords if k.arg == "environ")
        self.assertEqual(ast.literal_eval(environ), ["TORCHTRT_TENSORRT_ROOT"])

    def test_modules_keep_drive_and_sbsa_separate(self):
        for path in ("MODULE.bazel", "toolchains/ci_workspaces/MODULE.bazel.tmpl"):
            with self.subTest(path=path):
                source = (REPO / path).read_text()
                sbsa = named_call(source, "tensorrt_sbsa")
                self.assertEqual(sbsa.func.id, "http_archive")
                attrs = {k.arg: ast.literal_eval(k.value) for k in sbsa.keywords}
                self.assertEqual(
                    attrs["build_file"], "@//third_party/tensorrt/archive:BUILD"
                )
                self.assertEqual(attrs["strip_prefix"], "TensorRT-11.3.0.99")
                drive = named_call(source, "tensorrt_driveos")
                self.assertEqual(drive.func.id, "drive_tensorrt_repository")
                attrs = {k.arg: ast.literal_eval(k.value) for k in drive.keywords}
                self.assertEqual(
                    attrs,
                    {
                        "name": "tensorrt_driveos",
                        "build_file": "@//third_party/tensorrt/local:BUILD",
                    },
                )
                # The local TensorRT BUILD also refers to the DRIVE CUDA repository.
                self.assertEqual(
                    named_call(source, "cuda_driveos").func.id, "drive_cuda_repository"
                )

    def test_executorch_consumers_select_local_drive_sdk(self):
        for path, target, normal in (
            ("cpp/BUILD", "tensorrt_executorch_backend", ":sbsa"),
            ("core/runtime/BUILD", "tensorrt_binding_names", ":sbsa"),
            (
                "py/torch-tensorrt-executorch-runtime/native/BUILD.bazel",
                "delegate_native",
                ":aarch64_linux",
            ),
        ):
            with self.subTest(path=path):
                target_call = named_call((REPO / path).read_text(), target)
                selections = [
                    ast.literal_eval(node.args[0])
                    for node in ast.walk(target_call)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "select"
                ]
                trt = next(
                    s
                    for s in selections
                    if "@tensorrt_sbsa//:nvinfer" in s.get(normal, [])
                )
                self.assertIn("@tensorrt_driveos//:nvinfer", trt[":driveos"])
                self.assertNotIn("@tensorrt_sbsa//:nvinfer", trt[":driveos"])
                self.assertNotIn(
                    "driveos", (REPO / "third_party/tensorrt/archive/BUILD").read_text()
                )


@unittest.skipUnless(
    os.environ.get("BAZEL") or shutil.which("bazel"),
    "Set BAZEL to run repository integration tests",
)
class DriveRepositoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="torchtrt-drive-tensorrt-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.workspace = Path(cls.temp.name) / "workspace"
        cls.workspace.mkdir()
        cls.output = Path(cls.temp.name) / "bazel-output"
        cls.bazel = os.environ.get("BAZEL") or shutil.which("bazel")
        shutil.copyfile(RULE, cls.workspace / RULE.name)
        (cls.workspace / "MODULE.bazel").write_text(
            'module(name = "drive_sdk_test")\n'
            'drive_trt = use_repo_rule("//:drive_tensorrt_repository.bzl", "drive_tensorrt_repository")\n'
            'drive_trt(name = "tensorrt_driveos", build_file = "//:sdk.BUILD")\n'
        )
        (cls.workspace / "BUILD.bazel").write_text('filegroup(name = "unrelated")\n')
        # Only repository creation is tested here, not C++ compilation or ABI compatibility.
        (cls.workspace / "sdk.BUILD").write_text(
            'filegroup(name = "sdk", srcs = ["include/aarch64-linux-gnu/NvInfer.h", '
            '"lib/aarch64-linux-gnu/libnvinfer.so"])\n'
        )

    def query(self, root, target="@tensorrt_driveos//:sdk"):
        env = os.environ.copy()
        env.pop("TORCHTRT_TENSORRT_ROOT", None)
        result = subprocess.run(
            [
                self.bazel,
                "--batch",
                "--ignore_all_rc_files",
                f"--output_base={self.output}",
                f"--output_user_root={self.output.parent / 'bazel-user'}",
                "query",
                "--incompatible_autoload_externally=",
                "--lockfile_mode=off",
                f"--repo_env=TORCHTRT_TENSORRT_ROOT={root}",
                target,
            ],
            cwd=self.workspace,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=90,
        )
        return result

    def make_sdk(self, name, files):
        root = self.workspace.parent / name
        for relative in files:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        return root

    def test_missing_root_fails(self):
        result = self.query("")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("Set TORCHTRT_TENSORRT_ROOT", result.stdout)

    def test_missing_header_fails(self):
        root = self.make_sdk("no-headers", ["lib/aarch64-linux-gnu/libnvinfer.so"])
        result = self.query(root)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("is missing include/aarch64-linux-gnu/NvInfer.h", result.stdout)

    def test_missing_library_fails(self):
        root = self.make_sdk("no-libraries", ["include/aarch64-linux-gnu/NvInfer.h"])
        result = self.query(root)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("is missing lib/aarch64-linux-gnu/libnvinfer.so", result.stdout)

    def test_local_sdk_is_exposed_and_tracks_root_changes(self):
        for name in ("first-sdk", "second-sdk"):
            with self.subTest(name=name):
                root = self.make_sdk(
                    name,
                    [
                        "include/aarch64-linux-gnu/NvInfer.h",
                        "lib/aarch64-linux-gnu/libnvinfer.so",
                    ],
                )
                result = self.query(root)
                self.assertEqual(result.returncode, 0, result.stdout)
                generated = next((self.output / "external").glob("*tensorrt_driveos"))
                for directory in ("include", "lib"):
                    self.assertTrue((generated / directory).is_symlink())
                    self.assertEqual(
                        (generated / directory).resolve(), root / directory
                    )
                self.assertEqual(
                    (generated / "BUILD.bazel").read_text(),
                    (self.workspace / "sdk.BUILD").read_text(),
                )

    def test_unrelated_target_does_not_require_drive_sdk(self):
        result = self.query("", "//:unrelated")
        self.assertEqual(result.returncode, 0, result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
