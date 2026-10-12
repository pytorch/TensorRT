"""SDK-selection checks; no PyTorch, TensorRT, or GPU installation is required.

Run with ``python3 tests/toolchains/test_drive_tensorrt_repository.py``.
"""

import ast
from pathlib import Path
import unittest

REPO = Path(__file__).resolve().parents[2]


def named_call(source, name):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg == "name" and ast.literal_eval(keyword.value) == name:
                    return node
    raise AssertionError(f"Missing declaration: {name}")


class TensorRTSelectionTest(unittest.TestCase):
    def test_drive_uses_existing_ngc_local_repository_layout(self):
        ngc = named_call(
            (REPO / "docker/MODULE.bazel.ngc").read_text(), "tensorrt_sbsa"
        )
        ngc_attrs = {k.arg: ast.literal_eval(k.value) for k in ngc.keywords}
        for path in ("MODULE.bazel", "toolchains/ci_workspaces/MODULE.bazel.tmpl"):
            with self.subTest(path=path):
                source = (REPO / path).read_text()
                drive = named_call(source, "tensorrt_driveos")
                self.assertEqual(drive.func.id, "new_local_repository")
                attrs = {k.arg: ast.literal_eval(k.value) for k in drive.keywords}
                self.assertEqual(
                    attrs,
                    {
                        "name": "tensorrt_driveos",
                        "path": "/usr/",
                        "build_file": "@//third_party/tensorrt/local:BUILD",
                    },
                )
                self.assertEqual(attrs["path"], ngc_attrs["path"])
                self.assertEqual(attrs["build_file"], ngc_attrs["build_file"])
                self.assertNotIn("drive_tensorrt_repository", source)

    def test_drive_does_not_require_custom_tensorrt_root_variable(self):
        for path in (".bazelrc", "cmake/Modules/FindTensorRT.cmake", "README.md"):
            with self.subTest(path=path):
                self.assertNotIn("TORCHTRT_TENSORRT_ROOT", (REPO / path).read_text())

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

    def test_drive_allocator_uses_upstream_cuda_extension(self):
        target = named_call(
            (REPO / "cpp/BUILD").read_text(),
            "tensorrt_executorch_cuda_device_allocator",
        )
        deps = next(k.value for k in target.keywords if k.arg == "deps")
        selections = ast.literal_eval(deps.args[0])
        self.assertEqual(selections[":driveos"], selections[":sbsa"])
        self.assertIn("@executorch//:extension_cuda", selections[":driveos"])
        self.assertNotIn(
            "@executorch//:executorch_cuda_allocator", selections[":driveos"]
        )
        extension = named_call(
            (REPO / "third_party/executorch/BUILD").read_text(), "extension_cuda"
        )
        self.assertEqual(extension.func.id, "cc_library")


if __name__ == "__main__":
    unittest.main(verbosity=2)
