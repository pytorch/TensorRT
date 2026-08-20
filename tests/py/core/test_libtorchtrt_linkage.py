import importlib.metadata
import importlib.util
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path


@unittest.skipUnless(sys.platform.startswith("linux"), "ELF linkage test")
class TestLibTorchTensorRTLinkage(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch_spec = importlib.util.find_spec("torch")
        torchtrt_spec = importlib.util.find_spec("torch_tensorrt")
        if (
            torch_spec is None
            or torch_spec.origin is None
            or torchtrt_spec is None
            or torchtrt_spec.submodule_search_locations is None
        ):
            raise unittest.SkipTest("torch and torch_tensorrt must be installed")

        cls.torch_lib_dir = Path(torch_spec.origin).parent / "lib"
        cls.trt_lib_dirs = [
            distribution.locate_file(path).parent
            for distribution in importlib.metadata.distributions()
            if distribution.metadata["Name"] in {"tensorrt-cu13-libs"}
            for path in distribution.files or []
            if path.name.startswith("libnvinfer_plugin.so")
        ]
        cls.libtorchtrt = (
            Path(next(iter(torchtrt_spec.submodule_search_locations)))
            / "lib"
            / "libtorchtrt.so"
        )
        if not cls.libtorchtrt.is_file():
            raise unittest.SkipTest(
                f"installed torch_tensorrt package has no native library at "
                f"{cls.libtorchtrt}"
            )

        distribution_names = {
            name.replace("_", "-").lower()
            for name in importlib.metadata.packages_distributions().get(
                "torch_tensorrt", []
            )
        }
        cls.preload_module = (
            "tensorrt_rtx" if "torch-tensorrt-rtx" in distribution_names else ""
        )

    def test_does_not_directly_need_pytorch_transitive_dsos(self) -> None:
        dynamic_section = subprocess.run(
            ["readelf", "-d", self.libtorchtrt],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        needed = re.findall(
            r"\(NEEDED\).*Shared library: \[([^\]]+)\]", dynamic_section
        )
        forbidden_prefixes = ("libcudnn", "libcusparseLt", "libnvshmem_host")
        direct_transitive_dependencies = [
            soname for soname in needed if soname.startswith(forbidden_prefixes)
        ]
        self.assertEqual(direct_transitive_dependencies, [])

    def test_loads_in_fresh_process_without_importing_torch(self) -> None:
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [
                str(self.torch_lib_dir),
                *map(str, self.trt_lib_dirs),
                *filter(None, env.get("LD_LIBRARY_PATH", "").split(os.pathsep)),
            ]
        )
        child = """
import ctypes
import importlib
import os
import sys

assert "torch" not in sys.modules
if sys.argv[2]:
    importlib.import_module(sys.argv[2])
    assert "torch" not in sys.modules
ctypes.CDLL(sys.argv[1], mode=os.RTLD_NOW | ctypes.RTLD_GLOBAL)
assert "torch" not in sys.modules
"""
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                child,
                str(self.libtorchtrt),
                self.preload_module,
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
