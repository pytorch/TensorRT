#!/usr/bin/env bash
# Verify that an application can consume the ExecuTorch TensorRT delegate from
# INSTALLED packages, with no source checkout.
#
# The existing reference-runner check builds ExecuTorch from source via
# add_subdirectory, which is the right thing to verify for that path but says
# nothing about whether the installed package is usable. This script covers the
# other half: install the delegate, then build and run an application that finds
# everything through find_package.
#
# Exports its own model so it does not depend on another script's scratch
# directory, and asserts TensorRT actually claimed part of the graph: without that
# a program with no TensorRT delegate would still load and run, and the check would
# prove nothing.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
work_dir="$(mktemp -d)"
trap 'rm -rf "${work_dir}"' EXIT

python_executable="${PYTHON_EXECUTABLE:-python}"
pte_path="${work_dir}/model.pte"
expected_path="${work_dir}/expected.txt"

# The example needs CMake 3.28, which is what the ExecuTorch package it consumes
# requires. Say so plainly rather than letting a version error
# surface from three layers down.
if ! command -v cmake >/dev/null 2>&1; then
  echo "SKIP: cmake is not available, cannot build the example" >&2
  exit 0
fi
cmake_version="$(cmake --version | head -1 | awk '{print $3}')"
if [ "$(printf '%s\n3.28\n' "${cmake_version}" | sort -V | head -1)" != "3.28" ]; then
  echo "SKIP: cmake ${cmake_version} is older than the 3.28 the example needs" >&2
  exit 0
fi
echo "using cmake ${cmake_version}"

echo "=== locating the installed ExecuTorch package ==="
executorch_cmake_dir="$(
  cd "${work_dir}" && "${python_executable}" - <<'PY'
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.find_spec("executorch")
locations = list(getattr(spec, "submodule_search_locations", []) or []) if spec else []
if not locations:
    print("SKIP no installed ExecuTorch package")
    raise SystemExit(0)
package = Path(locations[0])
config = package / "share" / "cmake" / "executorch-config.cmake"
if not config.is_file():
    print("SKIP the installed package ships no CMake config")
    raise SystemExit(0)
# A released wheel predating the separately shipped runtime has the config but no
# linkable runtime, so there is nothing for an application to consume yet. That is
# a missing feature upstream rather than a failure of this check.
if "executorch::runtime" not in config.read_text():
    print("SKIP the installed package offers no shared runtime target")
    raise SystemExit(0)
print(config.parent)
PY
)"

case "${executorch_cmake_dir}" in
  SKIP*)
    echo "${executorch_cmake_dir#SKIP }: nothing to verify against, skipping"
    exit 0
    ;;
esac
echo "found: ${executorch_cmake_dir}"

echo "=== exporting a model with a TensorRT delegate ==="
(cd "${work_dir}" && "${python_executable}" - "${pte_path}" "${expected_path}" <<'PY'
import sys

import torch
import torch_tensorrt

pte_path, expected_path = sys.argv[1], sys.argv[2]


class Model(torch.nn.Module):
    def forward(self, x):
        return torch.tanh(x * 2.0 + 1.0)


model = Model().eval().cuda()
example = torch.ones((2, 3, 4, 4)).cuda()
compiled = torch_tensorrt.dynamo.compile(
    torch.export.export(model, (example,)),
    arg_inputs=[torch_tensorrt.Input(shape=tuple(example.shape), dtype=example.dtype)],
    min_block_size=1,
)
torch_tensorrt.save(
    compiled, pte_path, output_format="executorch", arg_inputs=(example,), retrace=False
)

from executorch.exir._serialize._program import deserialize_pte_binary

with open(pte_path, "rb") as handle:
    program = deserialize_pte_binary(handle.read()).program
ids = [d.id for plan in program.execution_plan for d in plan.delegates]
if ids.count("TensorRTBackend") < 1:
    sys.exit(f"TensorRT claimed no part of the graph, so this proves nothing: {ids}")
print("delegates:", ids)

# The application fills inputs with ones, so the reference uses the same input.
with torch.no_grad():
    reference = model(torch.ones_like(example))
with open(expected_path, "w") as handle:
    handle.write(" ".join(f"{v:.6f}" for v in reference.detach().cpu().flatten().tolist()))
PY
)


echo "=== locating the TensorRT SDK ==="
# The delegate links TensorRT directly, and the installed library carries only
# $ORIGIN entries, so the SDK has to be supplied at configure time and its
# library directory again when the application links. On distributions where
# TensorRT is a system library this is already satisfied and the search finds
# nothing, which is why a failure here is not fatal.
tensorrt_root="${TensorRT_ROOT:-}"
if [ -z "${tensorrt_root}" ] && command -v bazel >/dev/null 2>&1; then
  output_base="$(bazel info output_base 2>/dev/null || true)"
  if [ -n "${output_base}" ]; then
    trt_header="$(
      find -L "${output_base}/external" \
        \( -path "*/+*tensorrt/include/NvInfer.h" \
           -o -path "*/tensorrt/include/NvInfer.h" \) \
        -print -quit 2>/dev/null || true
    )"
    if [ -n "${trt_header}" ]; then
      tensorrt_root="$(dirname "$(dirname "${trt_header}")")"
    fi
  fi
fi

delegate_cmake_args=()
consumer_cmake_args=()
if [ -n "${tensorrt_root}" ]; then
  echo "using the TensorRT SDK at ${tensorrt_root}"
  delegate_cmake_args+=("-DTensorRT_ROOT=${tensorrt_root}")
  consumer_cmake_args+=(
    # --disable-new-dtags so the path lands in DT_RPATH rather than DT_RUNPATH. A
    # DT_RUNPATH on the application is not used when resolving what its own libraries
    # need, so the delegate's transitive TensorRT dependency would go unfound at load
    # time even though the link succeeded. The delegate target itself already forces
    # DT_RPATH for the same reason.
    "-DCMAKE_EXE_LINKER_FLAGS=-L${tensorrt_root}/lib -Wl,-rpath,${tensorrt_root}/lib -Wl,--disable-new-dtags"
  )
else
  echo "no separate TensorRT SDK found, assuming it is installed system wide"
fi

echo "=== installing the shared delegate ==="
delegate_prefix="${work_dir}/delegate-install"
cmake -S "${repo_root}/cpp/src/torch_tensorrt/executorch" \
  -B "${work_dir}/delegate-build" \
  -DTORCHTRT_EXECUTORCH_BUILD_SHARED_DELEGATE=ON \
  -DCMAKE_PREFIX_PATH="${executorch_cmake_dir}" \
  -DCMAKE_INSTALL_PREFIX="${delegate_prefix}" \
  -DCMAKE_INSTALL_LIBDIR=lib \
  "${delegate_cmake_args[@]}"
cmake --build "${work_dir}/delegate-build" -j"${MAX_JOBS:-$(nproc)}"
cmake --install "${work_dir}/delegate-build"

# Pinned to lib above, but check both names anyway: the default library directory
# is lib64 on several distributions, and a wrong path here would report a failed
# install after a completely successful build.
delegate_library="$(
  find "${delegate_prefix}" -name "libexecutorch_backend_tensorrt.so*" -type f | head -1
)"
if [ -z "${delegate_library}" ]; then
  echo "ERROR: the delegate did not install a shared library under ${delegate_prefix}" >&2
  find "${delegate_prefix}" -type f | head -20 >&2
  exit 1
fi
echo "installed: ${delegate_library#"${delegate_prefix}"/}"

echo "=== building the application against installed packages only ==="
app_build="${work_dir}/app-build"
cmake -S "${repo_root}/examples/executorch_wheel_runner" \
  -B "${app_build}" \
  -DCMAKE_PREFIX_PATH="${executorch_cmake_dir};${delegate_prefix}" \
  "${consumer_cmake_args[@]}"
cmake --build "${app_build}" -j"${MAX_JOBS:-$(nproc)}"

# A source build would have configured ExecuTorch itself, leaving a second cache
# behind. Exactly one means nothing was built from source.
cache_count="$(find "${app_build}" -name CMakeCache.txt | wc -l)"
if [ "${cache_count}" -ne 1 ]; then
  echo "ERROR: expected one CMake cache, found ${cache_count}; something was built from source" >&2
  exit 1
fi

# The delegate registers itself from a static initializer, so nothing in the
# application references it and a linker may drop it. If that happens the backend
# is missing at runtime, so check the dependency is recorded.
app_path="${app_build}/executorch_wheel_runner"
if command -v readelf >/dev/null 2>&1; then
  if ! readelf -d "${app_path}" | grep -q "libexecutorch_backend_tensorrt"; then
    echo "ERROR: the delegate was dropped from the application's dependencies" >&2
    exit 1
  fi
  echo "the delegate is recorded in the application's dependencies"
fi

echo "=== running it ==="
"${app_path}" --model "${pte_path}" --expected "${expected_path}" --tolerance 1e-3

echo "SUCCESS: the delegate is usable from an installed package"
