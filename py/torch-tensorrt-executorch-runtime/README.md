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
  share/cmake/executorch-config.cmake    lib/cmake/executorch_backend_tensorrt/executorch_backend_tensorrt-config.cmake
```

Python users just import the package. A C++ app links it the same way it links
one of ExecuTorch's own backends:

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

The optimized-kernel library supplies the `et_copy` host/device copy operators.

Linking that target also records the wheel's own library directory in your binary,
so it finds the delegate without a library path being set. That is right for an
application built against an installed wheel and wrong for anything you intend to
redistribute, so it can be turned off:

```cmake
set(EXECUTORCH_BACKEND_TENSORRT_EMBED_RUNPATH OFF)
find_package(executorch_backend_tensorrt REQUIRED)
```

Set it before `find_package`, and ship the delegate yourself, or point a library
path at it at run time.
For device-resident exports with `alloc_graph_output=False`, C++ Module callers
must provide a CUDA output tensor with `Module::set_output` before execution.
Python callers cannot, so that arrangement is C++ only.

If you want one program that runs from both, turn off allocation for the inputs
and leave it on for the outputs. The caller then supplies the input buffer, which
is the copy worth avoiding, and the program's own device arena owns the output.
Measured on two architectures, that arrangement runs from Python and from C++ with
no boundary copies in the program and results identical to eager. Turning
allocation off for both works only from C++. Leaving it on for both is refused by the
export step, because a program that allocates its own inputs has nothing for a caller
to hand in.

Point CMake at both wheels. The example above calls `find_package(executorch)`
as well, and that package lives in its own distribution. ExecuTorch is a
namespace package, so `executorch.__file__` is `None` and has to be located
through its distribution metadata instead. The variable in front is what keeps
this a path query: importing this package loads the delegate, which raises when
the delegate cannot be loaded, and a path does not need it loaded.

```bash
cmake -DCMAKE_PREFIX_PATH="$(TORCH_TENSORRT_SKIP_DELEGATE_REGISTRATION=1 python -c 'import importlib.metadata as m, torch_tensorrt_executorch_runtime as r, pathlib; print(str(pathlib.Path(str(m.distribution("executorch").locate_file("executorch"))) / "share" / "cmake") + ";" + str(pathlib.Path(r.__file__).parent))')" ...
```

CMake 3.28 or newer is required for the example above, not because of this
package but because the `backend_cuda` component it pairs with rejects anything
older: earlier versions write the `$ORIGIN` token in a runtime search path
incorrectly.

Bring your own CMake. It is not part of these wheels, and a freshly imaged Jetson
has none at all, which is easy to miss because the Python side needs none.

You do not need to find TensorRT. The delegate needs it, and it records where to
look relative to its own location, so a linker that reads that record resolves it
from the sibling wheel without being told. That is also why the example links no
TensorRT target: your application uses the delegate's interface, not TensorRT's.

This package itself needs only 3.19. On 3.19 through 3.27 you can still use it,
by asking ExecuTorch for no components and linking the variables it gives you
instead of its targets:

```cmake
find_package(executorch REQUIRED)
find_package(executorch_backend_tensorrt REQUIRED)
target_include_directories(app PRIVATE ${EXECUTORCH_INCLUDE_DIRS})
target_compile_definitions(app PRIVATE ${EXECUTORCH_COMPILE_DEFINITIONS})
target_link_libraries(app PRIVATE
  ${EXECUTORCH_LIBRARIES}
  executorch::backend_tensorrt
)
```

Without an imported target to carry them, the include directories and the compile
definitions have to be applied by hand as well, which is what the three variables
above are for.

There is nothing to include. The delegate has no public header: it registers
itself with ExecuTorch's backend registry from a static initializer inside the
shared library, and everything after that is ExecuTorch's own runtime API. The
CMake target links the library with `--no-as-needed`, because nothing in a
consumer references a symbol the delegate defines, and the default would drop it
and leave the backend unregistered.

The wheel must use the same Python, PyTorch, ExecuTorch, CUDA, TensorRT, and
C++ ABI as its matching Torch-TensorRT wheel. This delegate requires CUDA 13;
the build matrix currently covers `cu130`, `cu132` and `cu134`, on both architectures. Ordinary Torch-TensorRT
release and JetPack builds retain their separate CUDA 12 support.

## Install

One command. The `executorch` extra brings this wheel, a CUDA build of ExecuTorch,
Torch-TensorRT and PyTorch. Swap `cu132` for the CUDA version you run:

```bash
python -m pip install --pre "torch-tensorrt[executorch]" \
  --index-url https://download.pytorch.org/whl/nightly/cu132 \
  --extra-index-url https://pypi.org/simple \
  --extra-index-url https://pypi.nvidia.com
```

All three indexes are needed, and so is `--pre`. Without `--pre` pip picks the stable
Torch-TensorRT from the public index, which is far older and does not carry this extra.
Without NVIDIA's index, the inference library resolves to a source distribution and pip
spends around twenty minutes trying to build it before failing.

You do not name PyTorch or this wheel yourself. PyTorch arrives as a dependency of
Torch-TensorRT, and this wheel arrives through the extra. Building it by hand is only
needed to work on it, and those instructions are below.

On a fresh environment that command was measured installing:

| package | version |
| --- | --- |
| torch-tensorrt | 2.15.0.dev20260919+cu132 |
| torch-tensorrt-executorch-runtime | 0.2.0.dev20260919+cu132 |
| executorch | 1.6.0.dev20260915+cu132 |
| torch | 2.15.0.dev20260919+cu132 |
| tensorrt-cu13-libs | 11.3.0.99 |
| nvidia-cuda-runtime | 13.4.92 |

ExecuTorch lands on an older date than the rest because this wheel pins the exact build
it was compiled against. That is deliberate, and the next section says why.

A CUDA build of ExecuTorch is required, not only to build against. The delegate
needs a library that only ExecuTorch's CUDA wheels carry, so a processor-only build
installs and then fails on import. This wheel pins the exact ExecuTorch build it was
compiled against, label and all, which is what stops that happening quietly.

## Use it from Python

Import the package once, anywhere before you load a program. Importing is what
registers the delegate, and nothing else about your code changes:

```python
from pathlib import Path

import torch
import torch_tensorrt_executorch_runtime  # noqa: F401
from executorch.runtime import Runtime

program = Runtime.get().load_program(Path("model.pte"))
forward = program.load_method("forward")
outputs = forward.execute((torch.ones((2, 3, 4, 4)),))
```

If the delegate cannot be loaded, that import raises, rather than letting the
failure surface later as a program that will not load.

A program whose graph is split between TensorRT and ExecuTorch's own CUDA backend
needs nothing extra here: both backends are registered, and the program says which
parts go where.

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
checks: `manylinux_2_28_x86_64` or `manylinux_2_35_aarch64`, with `py3-none`.
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

On x86 the build also needs a developer toolset, the kind the release containers
carry. Without one, the C++ runtime helpers the delegate references are not
absorbed into the library, so the check that runs after linking refuses the result
and names the symbol version it found. That refusal is correct: the library would
demand a newer C++ runtime than its own platform tag promises. An ordinary
development machine usually has no such toolset, so build there through the same
container the release uses, or take the wheel a continuous integration run already
produced. Arm does not have this problem, and its tag is higher for that reason.

The build reads the CUDA location out of the repository's `MODULE.bazel`, and the
copy checked in names one specific version. On a machine with a different CUDA, the
fetch fails saying that path does not exist. That file is generated from a template, so
render it for your machine before building. The template takes seven values:

```bash
export CUDA_HOME=/usr/local/cuda-13.2
export CU_VERSION=cu132
export CHANNEL=nightly
export BUILD_VERSION=0.2.0.dev0
export TORCH_INSTALL_PATH="$(python -c 'import torch; print(torch.__path__[0])')"
export TENSORRT_CUDA_VERSION_UPPER_BOUND=14.0
export TENSORRT_RTX_CUDA_VERSION_UPPER_BOUND=14.0
envsubst < toolchains/ci_workspaces/MODULE.bazel.tmpl > MODULE.bazel
```

Render it yourself rather than running `packaging/pre_build_script.sh`. That script is
for the release containers: it also installs packages, and on a development machine it
can replace the PyTorch you already have with one from another index, which leaves you
worse off than when you started.

```bash
python -m pip install pyyaml patchelf tensorrt-cu13 \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu130 \
  --extra-index-url https://pypi.nvidia.com/ \
  "executorch==1.6.0.dev20260915"
export TORCH_TENSORRT_EXECUTORCH_RUNTIME_VERSION="0.2.0.dev0+cu130"
python -m pip wheel --no-build-isolation --no-deps \
  --wheel-dir dist py/torch-tensorrt-executorch-runtime
```

Install the matching full `torch-tensorrt` wheel before building the companion.

A wheel built here, or taken from a continuous integration run, pins the exact
`torch-tensorrt` build it was made against, label included. That build is on no
index until it is published, so such a wheel installs only alongside the sibling
from the same run, not on its own. Installing both from the same nightly channel,
which is what a release looks like, resolves normally.
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

### Rebuilding and editable installs

After rebuilding a wheel, install it with `python -m pip install --no-deps
--force-reinstall` followed by its path. This replaces an installed wheel with
the same version. The delegate is a shared library, not a Python extension;
`build_ext --inplace` does not rebuild it.

For editable development, use the same matching dependencies and version setting:

```bash
python -m pip install --no-build-isolation --no-deps \
  --editable py/torch-tensorrt-executorch-runtime
```

Repeat this command after native or CMake changes. Stop native consumers before
rebuilding and restart them afterward; rebuilding does not update a loaded
library. Python source changes are visible directly. For strict editable mode,
add `--config-settings editable_mode=strict` and keep the generated link directory.
The native library and generated CMake files live beside the source package in
its generated library directory.

`TORCH_TENSORRT_EXECUTORCH_DEBUG`, `TORCH_TENSORRT_ALLOW_UNPINNED_EXECUTORCH`,
and `TORCH_TENSORRT_SKIP_DELEGATE_REGISTRATION` accept `1`, `true`, `yes`, or
`on`, ignoring case. All other values, including unset, empty, `0`, and `false`,
are false.

## Registration

Loading the delegate adds `TensorRTBackend` to the backend registry that the
installed ExecuTorch runtime owns. It replaces nothing: the stock runtime keeps
its own backends and kernels, and XNNPACK and CPU fallback regions behave
exactly as they do without this wheel.

Registration happens in the delegate's static initializer, so the library has
to be loaded before a delegated program is loaded. Importing this package does
that, and nothing else, so an import is all a normal program needs. There is a
`register()` for the rare case where the import order is not yours to choose, and
calling it twice is harmless.

```python
from pathlib import Path

import torch
import torch_tensorrt_executorch_runtime  # noqa: F401
from executorch.runtime import Runtime

program = Runtime.get().load_program(Path("model.pte"))
outputs = program.load_method("forward").execute((torch.randn(1, 3, 224, 224),))
```

ExecuTorch's own delegates register because they are linked into its pybindings
extension, so loading that extension pulls them in. A delegate shipped in a
separate wheel cannot join that link, and ExecuTorch has no discovery hook for
out-of-tree backends, so this package performs the equivalent step at import
time. Set `TORCH_TENSORRT_SKIP_DELEGATE_REGISTRATION=1` to import it without
loading the delegate; that is for tooling that wants the metadata only, and the
path recipe above is exactly that case.

Because registering is the point, an import that cannot register raises
`DelegateCompatibilityError` naming what to install, rather than returning a
module that registered nothing and leaving the program to fail later with a
backend it cannot find.

Loading and running a program is ExecuTorch's API, not this package's. Tensor
placement, method lookup and output devices are all documented by ExecuTorch. A
program exported with `skip_h2d_for_method_inputs` keeps its inputs on the
device, because nothing here copies them.

Such a program does not check that promise, and it will not tell you when you
break it. Hand it a host tensor and it runs, returns the right answer, and
stages the copy for you, once per call. So a program you believe is copying
nothing can be copying on every run with nothing to show it. The other
direction is refused, so the leniency goes one way only. If the copies are why
you exported this way, measure them rather than trusting the flag.

Load programs through `executorch.runtime`, which is ExecuTorch's public Python
runtime API. Do not use the underscore-prefixed loader inside its pybindings
extension: it is private and can change without notice. The two were measured
against each other on every arrangement here, host and device, including a graph
split with ExecuTorch's own CUDA backend, and they agree to the digit.

The choice that does matter is who owns the output buffer. Leave allocation on for
the graph outputs if Python will run the program, because a Python caller has no
way to hand one in. Turn it off only for a C++ consumer that supplies the address
itself.

## Use

A **CUDA** build of `executorch` is required at runtime, not just to build. The delegate carries a
`DT_NEEDED` on `libexecutorch_extension_cuda.so`, which only ExecuTorch's CUDA wheels ship, and
those live on the PyTorch nightly index. `install_requires` names the version including its local
label, so only the CUDA build the delegate linked satisfies it. A label-free specifier admits any
label, which let a `+cpu` wheel resolve and then fail to load at import. Carrying the label rules
that out, because PEP 440 only ignores labels when the specifier omits them. It does bind the wheel
to one CUDA train, which is correct: the delegate links that train's runtime.

To install the wheel built above, use the same CUDA index as the build. Build it for the
CUDA train you run on: the requirement names that train, because the delegate links its runtime.
Its exact development-version requirements already permit the required prereleases.
CUDA and TensorRT packages may resolve from PyPI or NVIDIA's index; the extra
index applies to the whole dependency solve.

```bash
python -m pip install dist/torch_tensorrt_executorch_runtime-*.whl \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu130
```

```python
from pathlib import Path

import torch
import torch_tensorrt_executorch_runtime  # noqa: F401
from executorch.runtime import Runtime

program = Runtime.get().load_program(Path("model.pte"))
forward = program.load_method("forward")
outputs = forward.execute((torch.ones((2, 3, 4, 4)),))
```
