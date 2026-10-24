# Torch-TensorRT ExecuTorch Reference Runners

Small example programs that load and run a `.pte` file written by Torch-TensorRT. They are
reference code with fixed example inputs, so replace those with your real inputs.

| Program | Source | Shows |
| --- | --- | --- |
| `example_executorch_runner` | `main.cpp` | The low-level `Program` and `Method` loading sequence, and running on a green context stream |
| `kv_cache_decode_check` | `kv_cache_decode_check.cpp` | That a key-value (KV) cache in memory the caller owns keeps its updates from call to call |
| `example_executorch_multi_profile_runner` | `multi_profile_main.cpp` | Picking a TensorRT optimization profile for each call through the `Module` API (nightly only) |
| `example_executorch_multi_profile_benchmark` | `multi_profile_benchmark.cpp` | What picking the right profile is worth (nightly only) |
| `load_model.py` | Python | Loading and running a `.pte` with ExecuTorch's Python runtime |
| `load_model_device_resident.py` | Python | The same for a program that keeps its inputs and outputs on the GPU |

The [ExecuTorch deployment guide](https://docs.pytorch.org/TensorRT/user_guide/runtime_performance/executorch.html)
explains the concepts used here.

## Get a model

The export scripts are in `examples/torchtrt_executorch_example/` of the Torch-TensorRT
repository. They need Torch-TensorRT with its `executorch` extra, installed as the guide's
[Installation](https://docs.pytorch.org/TensorRT/user_guide/runtime_performance/executorch.html#installation)
section shows.

```bash
# A minimal program, for example_executorch_runner and load_model.py.
python examples/torchtrt_executorch_example/export_static_shape.py --model_path=model.pte

# A program split between TensorRT and ExecuTorch's CUDA backend, for the green context example.
python examples/torchtrt_executorch_example/export_coalesced.py --model_path=coalesced.pte

# A one-layer decoder step, for kv_cache_decode_check.
python examples/torchtrt_executorch_example/export_kv_cache_decode.py --model_path=kv_cache_decode.pte

# One engine with two optimization profiles, for the two multi-profile programs (nightly only).
python examples/torchtrt_executorch_example/export_multi_profile.py --model_path=model_gemma3_multi_profile.pte
```

`export_coalesced.py` and `export_device_resident.py` also need a CUDA toolkit, because the CUDA
backend compiles its part with `nvcc`. Both write `aoti_cuda_blob.ptd` next to their `.pte`. Here
that is harmless, because neither CUDA part has weights, but give your own models one directory
each, as the guide explains. `export_multi_profile.py` needs `transformers>=5.4.0`,
which the extra does not install. By default it exports a small Gemma-3 with random weights, which
needs no download and takes about a minute. With `--weights google/gemma-3-1b-it` it exports the
real 1B model, whose `.pte` is about 2 GB and takes hours to write. That model is gated on Hugging
Face, so accept its license and log in first.

## Build the C++ programs

You need:

- A CUDA toolkit. The build turns on ExecuTorch's CUDA backend, which this build needs even for a
  program that uses only TensorRT: here the backend registers the allocator for the program's GPU
  memory.
- PyTorch, installed in the Python environment that CMake finds. The CUDA backend needs it to
  configure, but the programs do not link libtorch.
- A TensorRT SDK. Point `TensorRT_ROOT` at it.
- `libtorchtrt.tar.gz` from your Torch-TensorRT release, which carries the delegate source and
  these examples.
- An ExecuTorch source checkout at the commit that your Torch-TensorRT release pins in its
  `MODULE.bazel`. The tarball does not carry that file, so look it up in the release's source on
  GitHub. The default below is the commit pinned in the same source tree as this README.

```bash
# The ExecuTorch commit pinned in MODULE.bazel.
EXECUTORCH_REF="${EXECUTORCH_REF:-5e379f610e034da507b48da61464679609517b52}"
git clone --filter=blob:none --no-checkout https://github.com/pytorch/executorch.git executorch
pushd executorch
git fetch --depth 1 origin "${EXECUTORCH_REF}"
git checkout FETCH_HEAD
git submodule update --init --recursive --depth 1
popd

tar xvf libtorchtrt.tar.gz

export TORCH_TENSORRT_ROOT="${PWD}/torch_tensorrt"
export TensorRT_ROOT=/path/to/extracted/TensorRT
export LD_LIBRARY_PATH="${TensorRT_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cmake -S "${TORCH_TENSORRT_ROOT}/examples/executorch_reference_runner" \
  -B build-executorch-reference-runner \
  -DEXECUTORCH_SOURCE_DIR="${PWD}/executorch" \
  -DEXECUTORCH_BACKEND_TENSORRT_SOURCE_DIR="${TORCH_TENSORRT_ROOT}/src/torch_tensorrt/executorch" \
  -DTensorRT_ROOT="${TensorRT_ROOT}"

cmake --build build-executorch-reference-runner -j \
  --target example_executorch_runner kv_cache_decode_check
```

This builds the two programs that every release has into `build-executorch-reference-runner/`, and
the delegate's static archive, `build-executorch-reference-runner/lib/libexecutorch_trt_backend.a`,
as a dependency. Naming the targets keeps the build from compiling every default ExecuTorch target
too.

The two multi-profile programs exist only in nightly builds, so build them from a source checkout of
`main`: point `TORCH_TENSORRT_ROOT` at the checkout, leave out
`-DEXECUTORCH_BACKEND_TENSORRT_SOURCE_DIR` (the same `CMakeLists.txt` then finds the delegate source
on its own), and add `example_executorch_multi_profile_runner` and
`example_executorch_multi_profile_benchmark` to the targets.

This flow covers Linux x86-64 and Arm servers (SBSA). Windows and JetPack are not part of it.
`libtorchtrt.tar.gz` and the Torch-TensorRT wheels for Linux also carry a prebuilt
`torch_tensorrt/bin/example_executorch_runner`. It registers the GPU allocator itself, but it has no
CUDA backend, and of the CPU operators only the copies at the edges of the program, so it runs only
programs that TensorRT takes in full, such as the one `export_static_shape.py` writes. To find the
copy in your Python environment:

```bash
python -c 'import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec("torch_tensorrt").origin).parent / "bin")'
```

## Run a program from C++

```bash
./build-executorch-reference-runner/example_executorch_runner --model_path=model.pte --num_runs=1
```

`main.cpp` shows the low-level loading sequence:

```text
executorch::runtime::runtime_init()
FileDataLoader::from(model_path)
Program::load(loader)
program.method_meta(method_name)
allocate the method's planned memory
program.load_method(method_name, memory_manager)
method.set_input(...)
method.execute()
method.get_outputs(...)
```

Loading the method initializes the TensorRT delegate for each engine in the `.pte`. The runner does
not need the Python `torch_tensorrt` package, which is only needed to export. It takes no data file,
so it runs a program split with the CUDA backend only when that part has no weights, like the one
`export_coalesced.py` writes, and it cannot run a program saved with `external_engine_data`.

### Run on a green context stream

A CUDA green context owns a fixed number of the GPU's SMs (streaming multiprocessors). Running the
program on a stream of one keeps the whole model inside that part of the GPU. Both the TensorRT
delegate and ExecuTorch's CUDA backend honor it, so test it with the coalesced program:

```bash
./build-executorch-reference-runner/example_executorch_runner \
  --model_path=coalesced.pte --num_runs=1 --green_context_sms=8
```

`--green_context_sms=N` creates the stream inside a green context with at least N SMs. If the
device cannot provide that, the runner stops with exit status 2 and says how many SMs the device
has, instead of quietly using an ordinary stream. So a passing run really used a green context.
`0`, the default, uses an ordinary stream.

TensorRT warns on such a run that the engine was built without green context constraints, which it
says may cause crashes or undefined behavior. Torch-TensorRT has no option to build engines with
those constraints.

## Run a program from Python

```bash
python examples/executorch_reference_runner/load_model.py --model_path=model.pte --num_runs=1
```

For a program exported to keep its inputs and outputs on the GPU, `load_model_device_resident.py`
passes a CUDA input and checks that a CUDA output comes back with the right values:

```bash
python examples/torchtrt_executorch_example/export_device_resident.py --model_path=device_resident.pte
python examples/executorch_reference_runner/load_model_device_resident.py --model_path=device_resident.pte
```

## Check a KV cache the caller owns

`kv_cache_decode_check` tests a decoder step whose KV cache stays in a buffer the caller owns. The
engine writes the cache in place, and the writes have to persist from one `execute()` call to the
next.

```bash
./build-executorch-reference-runner/kv_cache_decode_check --model_path=kv_cache_decode.pte
```

It runs a decode step at position 1 twice: once on a fresh cache, and once after a step at
position 0. Attention at position 1 reads positions 0 and 1, so the two results differ only if
the cache write at position 0 persisted. The program zeroes the cache itself on each load, because
ExecuTorch does not keep the initial value of a buffer the program changes. It prints
`[kv-check] PASS` and returns 0 when the results differ, and fails when they are the same.

It does all of this twice, printing `caller stream: none` and then `caller stream: own`. The first
pass runs on `cudaStreamPerThread`, and the second on a stream the program creates and selects with
`CallerStreamGuard`. Both have to pass. It needs a CUDA device.

## Pick an optimization profile

**Nightly only.**

An engine can hold several optimization profiles: one set of weights, tuned for several input
shape ranges. Scope an `OptimizationProfileGuard` around a call to pick one:

```cpp
#include <executorch/extension/module/module.h>
#include <torch_tensorrt/executorch/TensorRTBackend.h>

using torch_tensorrt::executorch_backend::OptimizationProfileGuard;

constexpr int32_t kDecodeProfile = 0;  // export_multi_profile.py declares decode first
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

A profile is named by its index in the list declared at export, so define your own constants that
match the exporter. The guard only records the index for the calling thread. Each TensorRT engine
reads it inside its own `execute()` and switches there, so create the guard on the thread that
calls `forward()`. With no guard, every engine runs profile 0. To let each engine pick from the
input shapes instead, use `auto profile_guard = OptimizationProfileGuard::automatic();`.

The index reaches every TensorRT engine in the method, and each looks it up in its own list of
profiles. So pin by index only when the `.pte` holds one engine, or when its engines were compiled
from one list of profiles. An engine with one profile runs profile 0 for any pin above 0 and logs
that the pin did nothing. An engine with several profiles that lacks the index fails the call, and
a negative index always fails. The delegate README in `cpp/src/torch_tensorrt/executorch/` has
the full rules.

```bash
./build-executorch-reference-runner/example_executorch_multi_profile_runner \
  --model_path=model_gemma3_multi_profile.pte
```

After checking that each profile gives correct results, the program times decode on each profile,
and prefill on the one profile that accepts its 128-token input. It then prints how much faster
decode is on its own profile. The profile is pinned around each timing loop, so switching is not
part of the measurement. These numbers come from the ExecuTorch delegate, so do not compare them
with the [multiple optimization profiles tutorial](https://docs.pytorch.org/TensorRT/tutorials/runtime_opt/multi_optimization_profiles.html),
which uses a different runtime and a different baseline.

### What picking a profile is worth

`multi_profile_benchmark.cpp` runs the same prefill and decode loop twice on one engine: once with
every call pinned to the prefill profile, which also accepts one token, and once with each phase
on its own profile. Both runs use the same engine, so the difference is only the choice of
profile.

```bash
./build-executorch-reference-runner/example_executorch_multi_profile_benchmark \
  --model_path=model_gemma3_multi_profile.pte
```

- The call that switches profiles pays for the switch. Switching wins only when the decode gain,
  added up over a round's decode steps, is larger than that cost.
- The small default model only exercises the API. Export with `--weights google/gemma-3-1b-it` to
  see what the feature is worth.
- To compare whole rounds, use long blocks (`--block_rounds=8`). With short blocks, the
  prefill-only run inherits the decode profile from the block before it, and pays a switch it would
  never pay in production.
- Read the `min` and `p10` (10th percentile) columns. The two runs alternate in short blocks, so
  other work on the GPU slows both equally. Interference only ever adds time, so the low
  percentiles are the signal. The median and `p90` show how busy the machine was.
