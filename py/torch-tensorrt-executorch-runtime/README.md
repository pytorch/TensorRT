# torch-tensorrt-executorch-runtime

This wheel is the TensorRT delegate for the ExecuTorch runtime: the plugin that runs the TensorRT
engines stored in a `.pte` file. It holds one shared library,
`lib/libexecutorch_backend_tensorrt.so`, and a CMake package for it. The ExecuTorch runtime that
the delegate plugs into comes from the `executorch` wheel.

Its layout is close to the one ExecuTorch uses for its own delegates:

```text
executorch/                            torch_tensorrt_executorch_runtime/
  lib/libexecutorch_backend_cuda.so      lib/libexecutorch_backend_tensorrt.so
  share/cmake/executorch-config.cmake    lib/cmake/executorch_backend_tensorrt/executorch_backend_tensorrt-config.cmake
```

The [ExecuTorch deployment guide](https://docs.pytorch.org/TensorRT/user_guide/runtime_performance/executorch.html)
is the main documentation: how to install this wheel together with Torch-TensorRT, export a model,
and run it from Python and C++. To install, follow its
[Installation](https://docs.pytorch.org/TensorRT/user_guide/runtime_performance/executorch.html#installation)
section.

## Version matching

This wheel has to match its Torch-TensorRT wheel: the same PyTorch, ExecuTorch, CUDA and TensorRT,
built with the same C++ ABI (the compiler's binary conventions). It needs CUDA 13. Builds
exist for `cu132` and `cu134`. The Linux wheels are tagged `manylinux_2_28_x86_64` and
`manylinux_2_35_aarch64`, so aarch64 needs glibc 2.35 or newer. The Windows x86-64 wheel, so far
published only as a nightly build, holds `executorch_backend_tensorrt.dll` (a Windows shared
library, or DLL) instead.

It pins the exact ExecuTorch build it was compiled against, including its CUDA label, such as
`+cu132`. The delegate needs `libexecutorch_extension_cuda.so`, which only ExecuTorch's CUDA wheels
ship, so a CPU build of ExecuTorch would install and then fail on import. The exact pin rules that
out.

It names PyTorch with no version, because ExecuTorch leaves that choice open too. Installed on its
own, it can get a PyTorch older than the ExecuTorch it pins, and that fails at the first run with a
confusing error about a tensor. Installing it as the deployment guide shows avoids that.

## Use it from Python

Import the package once, before you load a program. The import registers the delegate with
ExecuTorch. Everything after it is ExecuTorch's own runtime API:

```python
from pathlib import Path

import torch
import torch_tensorrt_executorch_runtime  # noqa: F401
from executorch.runtime import Runtime

program = Runtime.get().load_program(Path("model.pte"))
forward = program.load_method("forward")
outputs = forward.execute((torch.ones((2, 3, 4, 4)),))
```

ExecuTorch's own delegates register because they are linked into its Python extension. A delegate
in a separate wheel cannot join that link, and ExecuTorch has no hook to discover one, so this
package loads the delegate when it is imported. It replaces nothing: ExecuTorch keeps its own
delegates and kernels.

- If the delegate cannot load, the import raises `DelegateCompatibilityError` and says what to
  install. You get the error at the import, not later as a program that cannot find its backend.
- `register()` does the same step again, for the rare case where you do not control the import
  order. Calling it twice is harmless.
- Set `TORCH_TENSORRT_SKIP_DELEGATE_REGISTRATION=1` to import the package without loading the
  delegate. That is for tools that need only its metadata, such as the path query below.

Load programs through `executorch.runtime`, ExecuTorch's public Python API. Do not use the
underscore-prefixed loader inside its extension. It is private and can change without notice.
Load your programs from one thread: loading from several threads at once can crash the process,
because ExecuTorch's logging and the bindings' output redirection race.

## Use it from C++

A C++ application links the delegate the same way it links one of ExecuTorch's own delegates:

```cmake
find_package(executorch REQUIRED COMPONENTS backend_cuda kernels_optimized)
find_package(executorch_backend_tensorrt REQUIRED)
target_link_libraries(my_app PRIVATE
  executorch::runtime
  executorch::backend_cuda
  executorch::backend_tensorrt
  executorch::kernels_optimized
)
```

Point CMake at both wheels. ExecuTorch's location comes from its package metadata, because
`executorch.__file__` is not set:

```bash
ET_CMAKE=$(python -c 'import importlib.metadata as m, pathlib; print(pathlib.Path(str(m.distribution("executorch").locate_file("executorch"))) / "share" / "cmake")')
DELEGATE=$(TORCH_TENSORRT_SKIP_DELEGATE_REGISTRATION=1 python -c 'import pathlib, torch_tensorrt_executorch_runtime as r; print(pathlib.Path(r.__file__).parent)')
cmake -DCMAKE_PREFIX_PATH="${ET_CMAKE};${DELEGATE}" ...
```

The deployment guide explains each target and shows a complete program. The
[delegate README](https://github.com/pytorch/TensorRT/blob/main/cpp/src/torch_tensorrt/executorch/README.md)
lists the rules a C++ caller has to keep (threads, buffers, streams), the load errors, and the
known issues.

- **There is no header.** The delegate registers itself from a static initializer when its library
  loads. The CMake target links the library with `--no-as-needed`, because your code references no
  symbol in it, and the linker would otherwise drop it and leave the backend unregistered.
- **You do not need to find TensorRT.** The delegate finds it in the TensorRT wheel next to it.
- **CMake 3.28 or newer** is needed for the recipe above: on older versions the ExecuTorch package
  does not provide the `executorch::` targets. CMake is not part of these wheels, so install it
  yourself. The section below shows a way for older CMake.
- **No header for optimization profiles (nightly only).** Choosing a profile needs
  `OptimizationProfileGuard` from `TensorRTBackend.h`, which only the source tree and
  `libtorchtrt.tar.gz` carry. On Linux you can compile against it, with the TensorRT headers, and
  link this wheel's delegate. On Windows the wheel's library does not export the guard. Without it,
  every engine runs its profile 0.

On Linux, linking the target also records the wheel's library directory in your binary, so it
finds the delegate with no library path set. That is right for an application built against an
installed wheel, and wrong for anything you ship to other machines. To turn it off, set both of
these before `find_package`, then ship the delegate yourself or point a library path at it:

```cmake
set(EXECUTORCH_BACKEND_TENSORRT_EMBED_RUNPATH OFF)
set(CMAKE_SKIP_BUILD_RPATH ON)
find_package(executorch_backend_tensorrt REQUIRED)
```

The second line is needed because CMake records the directory on its own when you link an imported
library.

### CMake older than 3.28

This package itself needs only CMake 3.19. On 3.19 to 3.27, find ExecuTorch without components and
link the variables it sets instead of its targets:

```cmake
find_package(executorch REQUIRED)
find_package(executorch_backend_tensorrt REQUIRED)
target_include_directories(my_app PRIVATE ${EXECUTORCH_INCLUDE_DIRS})
target_compile_definitions(my_app PRIVATE ${EXECUTORCH_COMPILE_DEFINITIONS})
target_link_libraries(my_app PRIVATE
  ${EXECUTORCH_LIBRARIES}
  executorch::backend_tensorrt
)
```

## Runtime libraries

The wheel bundles no ExecuTorch, PyTorch, TensorRT or CUDA libraries. The delegate finds them in
the wheels installed next to it, through these runtime search paths:

- `$ORIGIN`
- `$ORIGIN/../../executorch/lib`
- `$ORIGIN/../../tensorrt_libs`
- `$ORIGIN/../../nvidia/cu13/lib`

`$ORIGIN` is this package's `lib` directory. The delegate links no PyTorch, so there is no
`torch/lib` entry. For a TensorRT or CUDA installation outside these places, make its `lib`
directory visible to the dynamic loader, for example with `LD_LIBRARY_PATH`.

NVIDIA publishes its CUDA runtime wheels for Linux only. On Windows the wheel does not depend on
one, and the delegate loads the CUDA runtime from a CUDA 13 toolkit install instead: when the
package is imported, it adds the `bin` directory of `CUDA_PATH` or `CUDA_HOME` to the DLL search
path.

## Build from source

You need this only to work on the delegate itself. The steps below are for Linux.

> [!IMPORTANT]
> Build with `--no-build-isolation`. The delegate links the ExecuTorch runtime from the
> `executorch` wheel installed at build time, and it must use the same PyTorch as the matching
> Torch-TensorRT build. An isolated build may download a newer PyTorch or ExecuTorch that is not
> ABI compatible.

What you need:

- A matching CUDA 13 PyTorch and Torch-TensorRT installed.
- `bazelisk` or `bazel` on `PATH`. The build calls Bazel to compile the delegate, and Bazel fetches
  TensorRT itself, so you need no local TensorRT SDK.
- On x86-64, a developer toolset like the one in the release containers. Without it, the C++
  runtime helpers the delegate uses are not built into the library, and the check after linking
  refuses the result: the library would need a newer C++ runtime than its platform tag promises.
  An ordinary development machine usually has no such toolset, so build in the release container,
  or take a wheel that a continuous integration run built. aarch64 does not have this problem.

First render `MODULE.bazel` for your machine. The checked-in copy names one CUDA path, and on a
machine with another CUDA the fetch fails, saying that path does not exist. The command below
overwrites the checked-in file, so do not commit the result. Substitute your own CUDA version:

```bash
export CUDA_HOME=/usr/local/cuda-13.2
export CU_VERSION=cu132
export CHANNEL=nightly
export BUILD_VERSION=0.3.0.dev0
export TORCH_INSTALL_PATH="$(python -c 'import torch; print(torch.__path__[0])')"
export TENSORRT_CUDA_VERSION_UPPER_BOUND=14.0
export TENSORRT_RTX_CUDA_VERSION_UPPER_BOUND=14.0
envsubst < toolchains/ci_workspaces/MODULE.bazel.tmpl > MODULE.bazel
```

Do not run `packaging/pre_build_script.sh` for this. It is for the release containers: it also
installs packages, and on a development machine it can replace your PyTorch with one from another
index.

Then install the build dependencies and build the wheel. Use the ExecuTorch version that
`pyproject.toml` pins:

```bash
python -m pip install pyyaml patchelf tensorrt-cu13 \
  --extra-index-url "https://download.pytorch.org/whl/nightly/${CU_VERSION}" \
  --extra-index-url https://pypi.nvidia.com/ \
  "executorch==1.6.0.dev20261021"
export TORCH_TENSORRT_EXECUTORCH_RUNTIME_VERSION="0.3.0.dev0+${CU_VERSION}"
python -m pip wheel --no-build-isolation --no-deps \
  --wheel-dir dist py/torch-tensorrt-executorch-runtime
```

- `tensorrt-cu13` is needed at build time so the wheel can record an exact `tensorrt-cu13==`
  requirement, read from the installed package. It needs NVIDIA's index, because on PyPI its
  `tensorrt-cu13-libs` dependency is a source package that downloads the real wheel from NVIDIA's
  index while it builds.
- The delegate compiles and links against the installed ExecuTorch wheel, which ships the headers,
  the prebuilt runtime and a CMake package. It must be a CUDA wheel. No ExecuTorch source checkout
  is involved.
- `TORCH_TENSORRT_EXECUTORCH_RUNTIME_VERSION` sets this wheel's version. Without it, the version
  is the base version in `version.txt` plus `.dev0` and the Git revision. Nightly builds keep this
  wheel's own base version, with the main wheel's date and CUDA suffix, such as
  `0.3.0.dev20261023+cu132`. A release uses the base version and the CUDA suffix, such as
  `0.2.0+cu132`.
- A release build of the main wheel (`RELEASE=1`) reads the same variable and records the version
  for its error messages. When `TORCH_TENSORRT_EXECUTORCH_RUNTIME_CHANNEL` also names the channel
  that carries the pinned ExecuTorch (`nightly` for a development build, `test` for a release), the
  main wheel's `executorch` extra requires exactly this version of this wheel.

Install the result with PyTorch's indexes for your CUDA version. The ExecuTorch build it pins is
on one of them: a development build on the nightly index, a release on the test index and later
on the stable one.

```bash
python -m pip install dist/torch_tensorrt_executorch_runtime-*.whl \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu132 \
  --extra-index-url https://download.pytorch.org/whl/test/cu132
```

### Rebuilding and editable installs

To replace an installed wheel of the same version with a rebuilt one, use
`python -m pip install --no-deps --force-reinstall` and the wheel's path. The delegate is a shared
library, not a Python extension, so `build_ext --inplace` does not rebuild it.

For editable development, with the same dependencies and version setting:

```bash
python -m pip install --no-build-isolation --no-deps \
  --editable py/torch-tensorrt-executorch-runtime
```

Run this again after changing native code or CMake files. Restart any process that loaded the
delegate, because rebuilding does not update a loaded library. Python changes are visible at once.
For strict editable mode, add `--config-settings editable_mode=strict`.

### Environment variables

- `TORCH_TENSORRT_SKIP_DELEGATE_REGISTRATION`: import the package without loading the delegate.
- `TORCH_TENSORRT_ALLOW_UNPINNED_EXECUTORCH`: build against an ExecuTorch other than the pinned one.
- `TORCH_TENSORRT_EXECUTORCH_DEBUG`: build the delegate with debug information.

Each accepts `1`, `true`, `yes` or `on`, in any case. Any other value, or no value, means off.
