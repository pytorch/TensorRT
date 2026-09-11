# Torch-TensorRT ExecuTorch Runtime Wheel

This directory builds `torch-tensorrt-executorch-runtime`. The Linux wheel
contains one shared library, `lib/libexecutorch_backend_tensorrt.so`, holding the
TensorRT delegate and nothing else. The ExecuTorch runtime it registers with
comes from the `executorch` wheel.

The layout follows the one ExecuTorch uses for its own backends, so the delegate
is an out-of-tree sibling of them rather than a special case:

```
executorch/                            torch_tensorrt_executorch_runtime/
  lib/libexecutorch_backend_cuda.so      lib/libexecutorch_backend_tensorrt.so
  share/cmake/executorch-config.cmake    lib/cmake/torchtrt_executorch/torchtrt_executorch-config.cmake
```

Python users just import the package. A C++ app links it the same way it links
one of ExecuTorch's own backends:

```cmake
find_package(executorch REQUIRED COMPONENTS backend_cuda)
find_package(torchtrt_executorch REQUIRED)
target_link_libraries(my_app PRIVATE
  executorch::runtime executorch::backend_cuda torchtrt::executorch_backend)
```

Point CMake at both wheels. The example above calls `find_package(executorch)`
as well, and that package lives in its own distribution. ExecuTorch is a
namespace package, so `executorch.__file__` is `None` and has to be located
through its distribution metadata instead:

```bash
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import importlib.metadata as m, torch_tensorrt_executorch_runtime as r, pathlib; print(str(m.distribution("executorch").locate_file("executorch")) + ";" + str(pathlib.Path(r.__file__).parent))')" ...
```

CMake 3.28 or newer is required, not because of this package but because the
`backend_cuda` component it pairs with rejects anything older: earlier versions
write the `$ORIGIN` token in a runtime search path incorrectly.

There is nothing to include. The delegate has no public header: it registers
itself with ExecuTorch's backend registry from a static initializer inside the
shared library, and everything after that is ExecuTorch's own runtime API. The
CMake target links the library with `--no-as-needed`, because nothing in a
consumer references a symbol the delegate defines, and the default would drop it
and leave the backend unregistered.

The wheel must use the same Python, PyTorch, ExecuTorch, CUDA, TensorRT, and
C++ ABI as its matching Torch-TensorRT wheel. This delegate requires CUDA 13;
the build matrix currently covers `cu130` and `cu132`. Ordinary Torch-TensorRT
release and JetPack builds retain their separate CUDA 12 support.

## Runtime libraries

The wheel bundles no ExecuTorch, PyTorch, c10, TensorRT, or CUDA shared
libraries. The delegate carries origin-relative runtime search paths, exactly as
the build sets them:

- `$ORIGIN`
- `$ORIGIN/../../executorch/lib`
- `$ORIGIN/../../tensorrt_libs`
- `$ORIGIN/../../nvidia/cu13/lib`

There is no `$ORIGIN/../torch/lib` entry, because the delegate links no torch,
and no `$ORIGIN/../../nvidia/cuda_runtime/lib` entry, because that is the CUDA 12
layout and this package requires CUDA 13.

`$ORIGIN` is this package's `lib` directory; the three `../../` entries reach sibling
distributions, because `libexecutorch.so`, the TensorRT libraries, and the CUDA
runtime belong to other wheels. These packages are installed transitively with
the matching `torch-tensorrt` wheel. For a system TensorRT or CUDA installation
outside these standard locations, its `lib` directory must be available through
the system dynamic loader configuration or `LD_LIBRARY_PATH`.

The shared Linux build tags this wheel with `wheel tags` after the native
checks: `manylinux_2_28_x86_64` or `manylinux_2_39_aarch64`, with `py3-none`.
This updates wheel metadata and RECORD without bundling external libraries or
changing the delegate. Payload, dependency metadata, and installed-library
resolution are checked before the shared artifact is uploaded.

## Build

> [!IMPORTANT]
> Build this wheel with `--no-build-isolation`. The delegate links the
> prebuilt runtime out of the ExecuTorch wheel that is installed at build time,
> and it must use the exact PyTorch installation the matching Torch-TensorRT
> artifacts were built against. An isolated build may download a newer,
> ABI-incompatible PyTorch or ExecuTorch.

The build shells out to Bazel to compile the delegate, so `bazelisk` or `bazel`
must be on `PATH`. TensorRT itself arrives through Bazel's `@tensorrt` external
repository, so no local SDK path is needed.

The example below assumes Linux with a matching CUDA 13 PyTorch and
Torch-TensorRT installation. Substitute the channel for your CUDA throughout, such
as `cu134` for CUDA 13.4.

```bash
python -m pip install pyyaml patchelf tensorrt-cu13 \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu130 \
  --extra-index-url https://pypi.nvidia.com/ \
  "executorch==1.5.0.dev20260911"
export TORCH_TENSORRT_EXECUTORCH_RUNTIME_VERSION="0.2.0.dev0+cu130"
python -m pip wheel --no-build-isolation --no-deps \
  --wheel-dir dist py/torch-tensorrt-executorch-runtime
```

Install the matching full `torch-tensorrt` wheel before building the companion.
The exact main-wheel dependency comes from that installed distribution, with
its local CUDA suffix removed. `TORCH_TENSORRT_EXECUTORCH_RUNTIME_VERSION`
sets only the companion's own version. When unset, the companion uses its own
base version plus a development suffix and Git revision. The shared build keeps
the main wheel's date and CUDA suffix while retaining the companion's independent
base version. Its CMake version also describes the companion, not the main wheel.

`tensorrt-cu13` is needed at build time so the wheel can record an exact
`tensorrt-cu13==` requirement: that version is read from the installed
distribution, which the Bazel-provided libraries alone do not carry. It needs
`--extra-index-url https://pypi.nvidia.com/` above, because the PyPI
`tensorrt-cu13` sdist is a stub that downloads the real wheel from NVIDIA's index
and fails metadata generation without it.

The delegate compiles and links entirely against the installed ExecuTorch
wheel, which ships the headers, the prebuilt runtime, and a CMake package. A
CUDA wheel is required: the CPU wheel ships no CUDA extension, and ExecuTorch
releases up to 1.4.1 ship no linkable runtime at all. ExecuTorch is not built
from source for this wheel, so no source checkout or `EXECUTORCH_SOURCE_DIR` is
involved.

## Registration

Loading the delegate adds `TensorRTBackend` to the backend registry that the
installed ExecuTorch runtime owns. It replaces nothing: the stock runtime keeps
its own backends and kernels, and XNNPACK and CPU fallback regions behave
exactly as they do without this wheel.

Registration happens in the delegate's static initializer, so the library has
to be loaded before a delegated program is loaded. Importing this package does
that, and nothing else: there is no API to call.

```python
import torch_tensorrt_executorch_runtime  # noqa: F401
from executorch.extension.pybindings.portable_lib import _load_for_executorch

program = _load_for_executorch("model.pte")
outputs = program.run_method("forward", (tensor,))
```

ExecuTorch's own delegates register because they are linked into its pybindings
extension, so loading that extension pulls them in. A delegate shipped in a
separate wheel cannot join that link, and ExecuTorch has no discovery hook for
out-of-tree backends, so this package performs the equivalent step at import
time. Set `TORCH_TENSORRT_SKIP_DELEGATE_REGISTRATION=1` to import it without
loading the delegate; that is for tooling that wants the metadata only.

Loading and running a program is ExecuTorch's API, not this package's. Tensor
placement, method lookup and output devices are all documented by ExecuTorch. A
program exported with `skip_h2d_for_method_inputs` keeps its inputs on the
device, because nothing here copies them.

The device copies around the delegate need the program's device-tagged
memory-planned arenas backed by real device memory, which ExecuTorch does only
through its Module API. The examples therefore use `_load_for_executorch`
and `run_method`. The program loader in `executorch.runtime` plans these arenas
on the host and is not suitable for these delegated programs.

## Use

A **CUDA** build of `executorch` is required at runtime, not just to build. The delegate carries a
`DT_NEEDED` on `libexecutorch_extension_cuda.so`, which only ExecuTorch's CUDA wheels ship, and
those live on the PyTorch nightly index. `install_requires` names the version without a local
label, and a specifier written that way admits any label, so a `+cpu` wheel satisfies it and then
fails to load at import. Adding a CUDA local label would rule that out, because PEP 440 only
ignores labels when the specifier omits them. It would also bind this wheel to one CUDA train,
so the requirement stays label-free and the import reports the mismatch instead.

To install the wheel built above, use the same CUDA index as the build. Build it for the
CUDA train you run on: the requirement is label-free, but the delegate links the CUDA 13 runtime.
Its exact development-version requirements already permit the required prereleases.
CUDA and TensorRT packages may resolve from PyPI or NVIDIA's index; the extra
index applies to the whole dependency solve.

```bash
python -m pip install dist/torch_tensorrt_executorch_runtime-*.whl \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu130
```

```python
import torch
import torch_tensorrt_executorch_runtime  # noqa: F401
from executorch.extension.pybindings.portable_lib import _load_for_executorch

program = _load_for_executorch("model.pte")
outputs = program.run_method("forward", (torch.ones((2, 3, 4, 4)),))
```
