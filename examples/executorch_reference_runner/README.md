# Torch-TensorRT ExecuTorch Reference Runner

This directory contains minimal C++ reference runners for loading and
executing Torch-TensorRT compiled models saved in ExecuTorch `.pte` format:

| Target | Shows |
| ------ | ----- |
| `example_executorch_runner` (`main.cpp`) | The low-level `Program` / `Method` loading sequence |
| `example_executorch_multi_profile_runner` (`multi_profile_main.cpp`) | Selecting a TensorRT optimization profile per call through the high-level `Module` API |
| `example_executorch_multi_profile_benchmark` (`multi_profile_benchmark.cpp`) | What per-call profile switching costs |

The `.pte` file contains an ExecuTorch program with embedded TensorRT engine
payloads. The runner links the TensorRT ExecuTorch backend, loads the `.pte`
with the ExecuTorch C++ runtime, prepares input tensors, calls `execute()`, and
prints output shapes and sample values.

This is reference code. It fills all inputs with `1.0f`; replace that input
setup with your application's real input buffers.

## Input Model

You can also generate a sample `.pte` from the Torch-TensorRT source tree:

```bash
python examples/torchtrt_executorch_example/export_static_shape.py --model_path=model.pte

# Two-profile Gemma-3 engine for the multi-profile runner below. Defaults to a
# mini Gemma-3 that needs no download and exports in about a minute, most of it
# spent serializing the engine into the .pte. Add --weights google/gemma-3-1b-it
# for the real 1B model -- but that .pte is about 2 GB and serialization scales
# with engine size, so budget hours rather than minutes for it. The export
# script documents the measured rate.
python examples/torchtrt_executorch_example/export_multi_profile.py --model_path=model_gemma3_multi_profile.pte
```

## Build The Reference Runner

A normal reference runner build does not need separate steps for
`libexecutorch_core.a` and `libexecutorch_trt_backend.a`. The runner CMake adds
both ExecuTorch and the Torch-TensorRT ExecuTorch source package, and linking
`torchtrt::executorch_backend` makes the backend archive a dependency of
`example_executorch_runner`.

The `libtorchtrt.tar.gz` package also includes a prebuilt reference runner:

```text
torch_tensorrt/bin/example_executorch_runner
```

```bash
# Get the ExecuTorch source code. Set EXECUTORCH_REF to a branch or tag;
# leave it unset for the latest main.
EXECUTORCH_REF="${EXECUTORCH_REF:-main}"
case "${EXECUTORCH_REF}" in
  latest|latest-main|latest_main|"latest main")
    EXECUTORCH_REF="main"
    ;;
esac
git clone --depth 1 --branch "${EXECUTORCH_REF}" --recurse-submodules --shallow-submodules \
  https://github.com/pytorch/executorch.git executorch

# download the libtorchtrt.tar.gz
tar xvf libtorchtrt.tar.gz

export EXECUTORCH_SOURCE_DIR="${PWD}/executorch"
# tarball untared path
export TORCH_TENSORRT_ROOT="${PWD}/torch_tensorrt"
export TORCHTRT_EXECUTORCH_SOURCE_DIR="${TORCH_TENSORRT_ROOT}/src/torch_tensorrt/executorch"
export TensorRT_ROOT=/path/to/extracted/TensorRT
export LD_LIBRARY_PATH="${TensorRT_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cmake -S "${TORCH_TENSORRT_ROOT}/examples/executorch_reference_runner" \
  -B build-executorch-reference-runner \
  -DEXECUTORCH_SOURCE_DIR="${EXECUTORCH_SOURCE_DIR}" \
  -DTORCHTRT_EXECUTORCH_SOURCE_DIR="${TORCHTRT_EXECUTORCH_SOURCE_DIR}" \
  -DTensorRT_ROOT="${TensorRT_ROOT}"

cmake --build build-executorch-reference-runner --target example_executorch_runner -j
```

Expected artifact:

```text
build-executorch-reference-runner/example_executorch_runner
```

The build also creates the executorch core and tensorrt backend archive as a dependency:

```text
build-executorch-reference-runner/executorch/libexecutorch_core.a
build-executorch-reference-runner/lib/libexecutorch_trt_backend.a
```

## Load And Run A `.pte` Model

Run the reference runner against a Torch-TensorRT compiled ExecuTorch model:

```bash
./build-executorch-reference-runner/example_executorch_runner --model_path=/path/to/model.pte --num_runs=1
```

The runner demonstrates this C++ loading sequence:

```text
executorch::runtime::runtime_init()
FileDataLoader::from(model_path)
Program::load(loader)
program.method_meta(method_name)
allocate planned ExecuTorch memory
program.load_method(method_name, memory_manager)
method.set_input(...)
method.execute()
method.get_outputs(...)
```

Loading the method initializes the TensorRT ExecuTorch backend for any
Torch-TensorRT delegate subgraphs embedded in the `.pte`. The Python
`torch_tensorrt` package is needed when exporting the `.pte`; it is not needed
by this native runner at inference time.

## Selecting An Optimization Profile

A TensorRT engine can hold several optimization profiles: one weight set, one
engine, several kernel tunings, each valid over a different input-shape range.
Scope an `OptimizationProfileGuard` around the call to pick one.

A profile is identified by its index in the list declared at export time. The
library defines no index constants; name them yourself to match the exporter, as
`export_multi_profile.py` declares decode first and prefill second:

```cpp
#include <torch_tensorrt/executorch/TensorRTBackend.h>

using torch_tensorrt::executorch_backend::OptimizationProfileGuard;

constexpr int32_t kDecodeProfile = 0;
constexpr int32_t kPrefillProfile = 1;

executorch::extension::Module module("model_gemma3_multi_profile.pte");
{
  OptimizationProfileGuard profile_guard(kPrefillProfile);
  auto result = module.forward(prefill_inputs);
}
{
  OptimizationProfileGuard profile_guard(kDecodeProfile);
  auto result = module.forward(decode_inputs);
}
```

The guard records an index for the calling thread and nothing else — it does not
inspect the `Module`, `Method`, or delegate handles, and does not call TensorRT.
Each TensorRT delegate reads it inside its own `execute()` and switches there.
Construct it on the thread that calls `forward()`. With no guard in scope, every
delegate runs profile 0.

To have each delegate choose from the input shapes instead of being told an
index, use the named constructor:

```cpp
auto profile_guard = OptimizationProfileGuard::automatic();
```

The index reaches every TensorRT delegate in the method, and each resolves it
against its own profile list. Nothing makes index 1 mean the same thing in two
engines, so pin by index only when the `.pte` holds one TensorRT engine or when
its engines were compiled from a single profile list. An engine with just one
profile runs profile 0 and logs that the pin did nothing; a multi-profile engine
that lacks the index fails the execution.

Build and run:

```bash
cmake --build build-executorch-reference-runner --target example_executorch_multi_profile_runner -j
./build-executorch-reference-runner/example_executorch_multi_profile_runner \
  --model_path=model_gemma3_multi_profile.pte
```

After the correctness walkthrough it times decode on each profile, the same
comparison `examples/dynamo/multi_optimization_profiles.py` makes through the
Python runtime:

```
Per-call latency (ms), batch=1
call                        active profile        ms
----------------------------------------------------
decode (seq=1)                     prefill     6.415
decode (seq=1)                      decode     4.981
prefill (seq=128)                  prefill     8.438

Giving decode its own profile: 1.29x faster per token (+1.434 ms)
```

The profile is pinned around each timing loop rather than per call, so profile
switches stay out of the measurement. Prefill appears once because the decode
profile does not accept a 128-token input at all — prefill has only one profile
it can run on.

### What Selecting A Profile Is Worth

`multi_profile_benchmark.cpp` times the same prefill/decode loop twice against
one engine: once with every call pinned to the prefill profile (it accepts
`seq == 1` too, so decode runs on prefill-tuned kernels, which is what a
single-profile engine gives you), and once with each phase pinned to its own
profile.

```bash
cmake --build build-executorch-reference-runner --target example_executorch_multi_profile_benchmark -j
./build-executorch-reference-runner/example_executorch_multi_profile_benchmark \
  --model_path=model_gemma3_multi_profile.pte
```

On the real `google/gemma-3-1b-it` (exported with `--weights
google/gemma-3-1b-it`) on an idle A40, decode is **1.29x faster** on its own
profile (6.42 ms down to 4.97 ms per token) while a switch costs ~3.6 ms,
charged to whichever call switches. That breaks even after about five decode
steps. End to end, one prefill plus 16 decode steps drops from 112.0 ms to
96.2 ms (14.1% faster), and a 64-step round from 420.6 ms to 336.2 ms (20.1%).

Both numbers shrink with the model. The mini Gemma-3 exported by default is
small enough that decode gains only 0.02 ms (1.12x) against a 0.48 ms switch, so
it takes ~46 decode steps to break even and a 16-step round is actually 4-5%
slower with switching. Use it to exercise the API, and `--weights` to see what
the feature is worth.

When comparing wall-clock rounds, keep blocks long (`--block_rounds=8`). With
short blocks the prefill-only configuration inherits the decode profile from the
preceding switching block and pays a switch it would never pay in production,
which inflates switching's margin.

Read the `min` and `p10` columns. The two configurations are interleaved in
short blocks so that other tenants on the GPU perturb both equally, and since
interference only ever adds time, the low percentiles are the signal; the median
and `p90` tell you how busy the machine was, not what switching cost.
