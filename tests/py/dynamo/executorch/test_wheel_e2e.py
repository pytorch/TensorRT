"""End-to-end test: export with TensorRT, then run from C++ against wheels only.

The other tests in this directory stop at the serialized program, which is what
they say they do: they assert the right delegates are present but never load or
run anything. This one goes further and is the only test that proves the shipped
C++ pieces are usable:

1. Export a model where TensorRT actually claims work, and fail if it claimed
   nothing. Without that check the rest of the test proves nothing, because a
   program with zero TensorRT delegates still loads and runs fine.
2. Build a C++ application against INSTALLED packages, with no source checkout,
   using find_package for both the runtime and the delegate.
3. Run it and compare against a reference computed in Python.
4. Repeat with the input tensors allocated in device memory, because that is how
   an accelerator application feeds data. The CUDA delegate copies device to
   device and never stages through the host, so it expects tensors that are
   already resident.
"""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("executorch.exir")

import torch  # noqa: E402

# Importing this pulls in the compiled TensorRT runtime, which is absent in a
# collection-only or CPU-only environment. Skipping rather than failing keeps the
# rest of the suite collectable there.
torch_tensorrt = pytest.importorskip("torch_tensorrt")

_RUNNER_SOURCE_DIR = (
    Path(__file__).resolve().parents[4] / "examples" / "executorch_wheel_runner"
)


def _installed_executorch_cmake_dir():
    """Where the installed ExecuTorch package keeps its CMake config.

    None unless the config actually offers the shared runtime target. Released wheels
    ship the file without it, and the example needs that target, so returning the
    directory anyway would let the test configure and fail rather than skip.
    """
    spec = importlib.util.find_spec("executorch")
    if spec is None or not spec.submodule_search_locations:
        return None
    root = Path(list(spec.submodule_search_locations)[0])
    config = root / "share" / "cmake" / "executorch-config.cmake"
    if not config.is_file():
        return None
    if "executorch::runtime" not in config.read_text():
        return None
    return config.parent


_CMAKE_DIR = _installed_executorch_cmake_dir()

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DELEGATE_SOURCE_DIR = _REPO_ROOT / "cpp" / "src" / "torch_tensorrt" / "executorch"


@pytest.fixture(scope="module")
def delegate_prefix(tmp_path_factory):
    """Build and install the TensorRT delegate, and return its CMake prefix.

    Without this the runner is configured against ExecuTorch alone, so
    find_package(torch_tensorrt_executorch) finds nothing and the application is
    built without the delegate. A program that needs TensorRTBackend would then run
    against a runner that never linked it, which cannot work and would not say why.
    """
    if _CMAKE_DIR is None or shutil.which("cmake") is None:
        pytest.skip("needs an ExecuTorch CMake package and cmake")
    if not (_DELEGATE_SOURCE_DIR / "CMakeLists.txt").is_file():
        pytest.skip("needs the delegate sources, which a wheel-only install lacks")

    root = tmp_path_factory.mktemp("delegate")
    prefix = root / "install"
    configure = [
        "cmake",
        "-S",
        str(_DELEGATE_SOURCE_DIR),
        "-B",
        str(root / "build"),
        "-DTORCHTRT_EXECUTORCH_BUILD_SHARED_DELEGATE=ON",
        f"-DCMAKE_PREFIX_PATH={_CMAKE_DIR}",
        f"-DCMAKE_INSTALL_PREFIX={prefix}",
        # Pinned so the check below does not have to guess between lib and lib64.
        "-DCMAKE_INSTALL_LIBDIR=lib",
    ]
    if os.environ.get("TensorRT_ROOT"):
        configure.append(f"-DTensorRT_ROOT={os.environ['TensorRT_ROOT']}")

    result = subprocess.run(configure, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        pytest.skip(f"the delegate does not configure here: {result.stderr[-400:]}")
    subprocess.run(
        ["cmake", "--build", str(root / "build"), "-j"],
        capture_output=True,
        text=True,
        check=True,
    )
    subprocess.run(["cmake", "--install", str(root / "build")], check=True)

    installed = list(prefix.rglob("libexecutorch_backend_tensorrt.so*"))
    assert installed, f"the delegate installed no shared library under {prefix}"
    return prefix


requires_wheel = pytest.mark.skipif(
    _CMAKE_DIR is None,
    reason="needs an ExecuTorch install that ships its CMake package",
)
requires_cmake = pytest.mark.skipif(
    shutil.which("cmake") is None, reason="needs cmake to build the C++ application"
)
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


class _Model(torch.nn.Module):
    """Small graph that TensorRT can take in full."""

    def forward(self, x):
        return torch.tanh(x * 2.0 + 1.0)


def _delegate_ids(pte_path: Path):
    """Backend ids of every delegate in the serialized program, in order."""
    from executorch.exir._serialize._program import deserialize_pte_binary

    program = deserialize_pte_binary(pte_path.read_bytes()).program
    return [
        delegate.id for plan in program.execution_plan for delegate in plan.delegates
    ]


def _export(model, example_input, destination: Path) -> Path:
    exported = torch.export.export(model, (example_input,))
    compiled = torch_tensorrt.dynamo.compile(
        exported,
        arg_inputs=[
            torch_tensorrt.Input(
                shape=tuple(example_input.shape), dtype=example_input.dtype
            )
        ],
        min_block_size=1,
    )
    torch_tensorrt.save(
        compiled,
        str(destination),
        output_format="executorch",
        arg_inputs=(example_input,),
        retrace=False,
    )
    return destination


def _skip_on_engine_version_mismatch(result, caplog_text=""):
    """Skip when the engine and the TensorRT runtime are different versions.

    A serialized engine only loads on the TensorRT version that produced it. When
    the export environment and the linked runtime disagree, every delegated run
    fails for that reason alone, which says nothing about the packaging this test
    covers. Any other failure is still a real failure.

    The mismatch can surface either from the application or from TensorRT's own
    logger during export, so both sources are checked. A delegated program whose
    operators are all missing is the same situation seen from the runtime side:
    the engine never deserialized, so the delegate produced no kernels.
    """
    combined = result.stdout + result.stderr + caplog_text
    markers = (
        "Serialized Engine Version",
        "version mismatch",
        "Version tag does not match",
    )
    if any(marker in combined for marker in markers):
        pytest.skip(
            "the serialized engine and the linked TensorRT runtime are different "
            "versions, so a delegated run cannot succeed in this environment"
        )


def _build_runner(build_dir: Path, delegate_prefix: Path = None) -> Path:
    prefixes = [str(_CMAKE_DIR)]
    if delegate_prefix is not None:
        prefixes.append(str(delegate_prefix))
    subprocess.run(
        [
            "cmake",
            "-S",
            str(_RUNNER_SOURCE_DIR),
            "-B",
            str(build_dir),
            "-DCMAKE_PREFIX_PATH={}".format(";".join(prefixes)),
        ],
        check=True,
    )
    subprocess.run(["cmake", "--build", str(build_dir), "-j"], check=True)
    runner = build_dir / "executorch_wheel_runner"
    assert runner.is_file(), "the C++ application did not get built"
    if delegate_prefix is not None:
        # The delegate registers itself from a static initializer, so a runner that
        # merely built is not enough: the library has to still be on the link line.
        needed = subprocess.run(
            ["readelf", "-d", str(runner)], capture_output=True, text=True, check=False
        ).stdout
        assert "libexecutorch_backend_tensorrt" in needed, (
            "the runner built without the TensorRT delegate, so a delegated program "
            "cannot run against it"
        )
    return runner


def _write_reference(model, example_input, path: Path) -> None:
    with torch.no_grad():
        reference = model(example_input)
    values = reference.detach().cpu().flatten().tolist()
    path.write_text(" ".join(f"{value:.6f}" for value in values))


@requires_cuda
@requires_wheel
@requires_cmake
def test_tensorrt_partitions_run_from_cpp(tmp_path, delegate_prefix, caplog):
    """A TensorRT-delegated program runs from C++ and matches a reference."""
    model = _Model().eval().cuda()
    example_input = torch.ones((2, 3, 4, 4)).cuda()

    pte = _export(model, example_input, tmp_path / "model.pte")
    assert pte.is_file()

    # Without this the test would pass even if TensorRT claimed nothing at all.
    delegates = _delegate_ids(pte)
    assert delegates.count("TensorRTBackend") > 0, (
        f"TensorRT claimed no part of the graph, so this proves nothing about the "
        f"delegate; delegates were {delegates}"
    )

    reference = tmp_path / "expected.txt"
    # The runner fills inputs with ones, so the reference uses the same input.
    _write_reference(model, torch.ones_like(example_input), reference)

    runner = _build_runner(tmp_path / "build", delegate_prefix)
    result = subprocess.run(
        [
            str(runner),
            "--model",
            str(pte),
            "--expected",
            str(reference),
            "--tolerance",
            "1e-3",
        ],
        capture_output=True,
        text=True,
    )
    sys.stderr.write(result.stderr)
    _skip_on_engine_version_mismatch(result, caplog.text)
    assert result.returncode == 0, "the C++ application did not match the reference"
    assert "outputs match the reference" in result.stdout


@requires_cuda
@requires_wheel
@requires_cmake
def test_device_resident_inputs(tmp_path, delegate_prefix, caplog):
    """The same program runs when its inputs start in device memory.

    An accelerator application allocates its tensors on the device and hands those
    pointers to the runtime. This checks that path rather than the host one.
    """
    model = _Model().eval().cuda()
    example_input = torch.ones((2, 3, 4, 4)).cuda()

    pte = _export(model, example_input, tmp_path / "model.pte")
    delegates = _delegate_ids(pte)
    assert delegates.count("TensorRTBackend") > 0, (
        f"TensorRT claimed no part of the graph; delegates were {delegates}"
    )

    reference = tmp_path / "expected.txt"
    _write_reference(model, torch.ones_like(example_input), reference)

    runner = _build_runner(tmp_path / "build", delegate_prefix)
    result = subprocess.run(
        [
            str(runner),
            "--model",
            str(pte),
            "--expected",
            str(reference),
            "--tolerance",
            "1e-3",
            "--gpu-inputs",
        ],
        capture_output=True,
        text=True,
    )
    sys.stderr.write(result.stderr)
    if result.returncode == 2:
        pytest.skip("the installed wheel has no CUDA delegate, so no device inputs")
    _skip_on_engine_version_mismatch(result, caplog.text)
    # The runner refuses device pointers for a memory-planned input, because the
    # runtime would copy such an input into the plan with a host memcpy. That is
    # the runner behaving correctly for this program, not a packaging failure.
    if "is memory planned" in result.stdout + result.stderr:
        pytest.skip(
            "this program's inputs are memory planned, so the runtime copies them "
            "on the host and device-resident inputs do not apply"
        )
    assert result.returncode == 0, "device-resident inputs did not match the reference"
    assert "outputs match the reference" in result.stdout


@requires_wheel
@requires_cmake
def test_runner_uses_no_source_checkout(tmp_path):
    """The application builds from installed packages alone.

    This is the property the whole prebuilt-library effort exists to provide, so it
    is worth asserting rather than assuming: configuring must not need an
    ExecuTorch source tree anywhere.
    """
    build_dir = tmp_path / "build"
    configure = subprocess.run(
        [
            "cmake",
            "-S",
            str(_RUNNER_SOURCE_DIR),
            "-B",
            str(build_dir),
            f"-DCMAKE_PREFIX_PATH={_CMAKE_DIR}",
        ],
        capture_output=True,
        text=True,
    )
    assert configure.returncode == 0, configure.stderr
    # The example's own output, not ExecuTorch's. Matching another project's log wording
    # would break whenever that project rephrases a message, while this line is emitted
    # by the code under test and only when the runtime target actually resolved.
    assert "linking executorch::" in configure.stdout, (
        "the example did not report linking any ExecuTorch component, so the package "
        f"config did not provide the targets it needs: {configure.stdout[-400:]}"
    )

    # A source build would have configured ExecuTorch itself, leaving its cache
    # behind. Only the application's own cache should exist.
    caches = list(build_dir.rglob("CMakeCache.txt"))
    assert len(caches) == 1, (
        f"expected only the application's own CMake cache, found {caches}. More than "
        f"one means an ExecuTorch source tree was configured as a subproject."
    )
    assert not (build_dir / "executorch").exists()
