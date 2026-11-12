.. _executorch_deployment:

ExecuTorch Deployment
=====================

`ExecuTorch <https://pytorch.org/executorch/>`_ is PyTorch's runtime for edge devices. It runs a
model from one ``.pte`` file, with a small C++ library that does not need libtorch, PyTorch's own
C++ library. That suits a robot, a drone, or any device where a full PyTorch install is too heavy.

Torch-TensorRT can write that file. TensorRT compiles the parts of the model it supports into
engines, and the engines are stored inside the ``.pte``. At run time, the TensorRT delegate (the
ExecuTorch plugin for those parts) hands each engine back to TensorRT. Operators that TensorRT
cannot compile can go to ExecuTorch's own CUDA backend in the same file, so the whole model stays
on the GPU.

Use ExecuTorch when you want to:

* run the model from a C++ application with no Python and no libtorch,
* ship the program and its TensorRT engines in one file (a coalesced program, described below,
  adds one data file), or
* run the model on your own CUDA stream, for example one that keeps it inside part of the GPU.

.. note::

    The delegate runs on Linux (x86-64 and aarch64), and on Windows x86-64. The
    prebuilt delegate needs CUDA 13. On Windows, the delegate loads the CUDA runtime from a CUDA 13
    toolkit install, so make sure ``CUDA_PATH`` or ``CUDA_HOME`` points at one.

    By default, TensorRT builds each engine for the operating system and the GPU it is exported on,
    so export your programs on the kind of machine that will run them. On Windows, ExecuTorch's
    CUDA backend cannot export a program, and its Python runtime there does not include that
    backend, so a coalesced program (see below) is a Linux setup.

----

Installation
-------------

One command installs everything: PyTorch, Torch-TensorRT, TensorRT, the CUDA build of
ExecuTorch, and the TensorRT delegate. Use a fresh virtual environment.

Releases
^^^^^^^^^

.. code-block:: bash

    pip install "torch_tensorrt[executorch]"

This installs the CUDA 13.2 builds. On Windows, PyPI has only the CPU builds of PyTorch and
ExecuTorch, so add ``--extra-index-url https://download.pytorch.org/whl/cu132``.

Nightly builds
^^^^^^^^^^^^^^^

To pick up a change before it ships, or to report a problem against ``main``:

.. code-block:: bash

    pip install --pre "torch_tensorrt[executorch]" --index-url https://download.pytorch.org/whl/nightly/cu132

Swap ``cu132`` for ``cu134`` on CUDA 13.4.

What these commands install
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* PyTorch and TensorRT, which Torch-TensorRT depends on.
* ExecuTorch, the CUDA build. That is all you need to export.
* ``torch-tensorrt-executorch-runtime``, one shared library holding the TensorRT delegate,
  which you need to run a ``.pte``. It is the one built together with your Torch-TensorRT wheel,
  for the same CUDA version. It requires the exact ExecuTorch build it was compiled against, so
  you get that one rather than the newest.

A wheel you build yourself does not bring the delegate. Build it as well, as the runtime wheel
README shows.

The delegate is built for the standard Linux wheels (x86-64 and aarch64), and for Windows x86-64.
On aarch64 the delegate needs glibc 2.35 or newer, and the Torch-TensorRT wheel glibc 2.39
(for example Ubuntu 24.04).
The JetPack, TensorRT-RTX and Python-only builds have no delegate. To export a coalesced program you also
need a CUDA toolkit, because ExecuTorch's CUDA backend compiles the leftover operators with
``nvcc``.

----

Export a Model
---------------

Compile with ``torch_tensorrt.dynamo.compile``, then save with ``output_format="executorch"``:

.. code-block:: python

    import torch
    import torch_tensorrt


    class MyModel(torch.nn.Module):
        def forward(self, x):
            return torch.relu(x) + 1


    model = MyModel().eval().cuda()
    example_inputs = (torch.randn(2, 3, 4, 4, device="cuda"),)

    exported = torch.export.export(model, example_inputs)
    trt_gm = torch_tensorrt.dynamo.compile(
        exported,
        arg_inputs=example_inputs,
        min_block_size=1,
    )
    torch_tensorrt.save(trt_gm, "model.pte", output_format="executorch")

``save()`` writes the compiled graph as it is, so it needs no example inputs.

TensorRT has no float64. Compile a float64 model with ``truncate_double=True``. Without it, the
engine expects float32 while the program keeps float64 tensors, and every call fails with
``Error::InvalidProgram``. An output that the program declares an alias of an input, such as an
updated KV cache, must have that input's type, or loading the program fails with
``Error::InvalidProgram``.

:ref:`executorch_save` lists every option that ``save()`` takes for this format. It also shows
how to put several methods in one file, and how to let the engine of a large language model (LLM)
update its key-value (KV) cache in place.

Run the leftover operators on the CUDA backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TensorRT cannot convert every PyTorch operator. By default the ones it rejects run on
ExecuTorch's CPU kernels, with a copy to the CPU and back on every call, and an operator with no
CPU kernel, such as ``erfinv``, stops the program from loading. Pass a ``CudaPartitioner``, and
ExecuTorch's CUDA backend compiles them instead, so nothing leaves the GPU:

.. code-block:: python

    from executorch.backends.cuda.cuda_backend import CudaBackend
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner


    class CoalescedModel(torch.nn.Module):
        def forward(self, x):
            return torch.cos(torch.erfinv(torch.tanh(x)))


    coalesced_model = CoalescedModel().eval().cuda()
    coalesced_inputs = (torch.randn(64, 64, device="cuda"),)

    coalesced_gm = torch_tensorrt.dynamo.compile(
        torch.export.export(coalesced_model, coalesced_inputs),
        arg_inputs=coalesced_inputs,
        min_block_size=1,
    )
    torch_tensorrt.save(
        coalesced_gm,
        "coalesced.pte",
        output_format="executorch",
        partitioners=[
            CudaPartitioner([CudaBackend.generate_method_name_compile_spec("forward")])
        ],
    )

TensorRT takes what it can first, and the ``CudaPartitioner`` takes the rest. TensorRT cannot
take ``erfinv``, so this program's delegates are
``['TensorRTBackend', 'CudaBackend', 'TensorRTBackend']``. These docs call such a program
*coalesced*.

The CUDA backend compiles its part with ``nvcc``, the CUDA compiler, so the export needs a CUDA
toolkit. The CUDA backend also writes the weights of its part to a data file,
``aoti_cuda_blob.ptd``, next to the ``.pte``. Pass that file when you load the program, as shown
in `Run from Python`_ and `Run from C++`_. If the CUDA part has weights, loading fails without it.

.. warning::

    Every coalesced export names its weight file ``aoti_cuda_blob.ptd``, so a second export into
    the same directory overwrites the first one's weights. The first model can then return wrong
    answers with no error, or fail to load. Save each model into its own directory.

Keep plugins that run PyTorch out of the engine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some converters build a TensorRT plugin that runs PyTorch when the engine runs: ``aten.index_put``
with ``accumulate=True`` uses Torch-TensorRT's ScatterAdd plugin, and a plugin made with
``torch_tensorrt.dynamo.conversion.plugins.custom_op`` that does not use an AOT kernel runs the
PyTorch op from Python. The ExecuTorch runtime does not load PyTorch, so ``save()`` refuses such an
engine and names the operators. It finds them through converters registered with
``requires_pytorch_plugin=True``, so a custom converter that adds such a plugin must set that flag.
``save()`` also refuses an engine that does not record this. If an older Torch-TensorRT saved it,
compile the model again. If you wrapped it in ``TorchTensorRTModule`` yourself, pass
``pytorch_plugin_ops`` (``[]`` for an engine built with ``fallback_pytorch_plugin_ops=True``). Compile with ``fallback_pytorch_plugin_ops=True`` to keep those
operators out of TensorRT. A ``torch.cond`` with one of them in a branch then runs in PyTorch too.
With the setting, an accumulating ``index_put`` with one 1-D index stays in TensorRT instead,
through TensorRT's own ScatterElements plugin, which needs no PyTorch. This needs float32, float16,
bfloat16, int32 or int64 data. The input and the values it receives (the input's shape with the
indexed dimension set to the index length) must stay under ``2**31`` elements, so every dynamic size needs an upper bound in the exported program, such as
``torch.export.Dim(max=...)``, that keeps them there. Otherwise the operator runs in PyTorch.
TensorRT cannot check index values inside the engine, so an out-of-range index is not reported
there, where PyTorch raises.

.. code-block:: python

    trt_gm = torch_tensorrt.dynamo.compile(
        exported,
        arg_inputs=example_inputs,
        fallback_pytorch_plugin_ops=True,
    )

Keep inputs and outputs on the GPU
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, the program copies its inputs from the CPU to the GPU before the first delegate, and
its outputs back after the last one. That way you can call it with CPU tensors. If your data is
already on the GPU, those copies are wasted time. Turn them off at export:

.. code-block:: python

    from executorch.exir import ExecutorchBackendConfig
    from executorch.exir.passes import MemoryPlanningPass
    from executorch.exir.passes.propagate_device_config import PropagateDeviceConfig

    torch_tensorrt.save(
        trt_gm,
        "device_resident.pte",
        output_format="executorch",
        backend_config=ExecutorchBackendConfig(
            propagate_device_config=PropagateDeviceConfig(
                skip_h2d_for_method_inputs=True,
                skip_d2h_for_method_outputs=True,
            ),
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False, alloc_graph_output=False
            ),
        ),
    )

* **The two skip flags work only with** ``enable_non_cpu_memory_planning``, which is on by
  default. With either flag set, turning it off makes the export raise an error.
* **Do not let the program allocate its inputs:** pass ``alloc_graph_input=False``. Otherwise
  the program keeps its own input buffer, and every call fails with error ``0x10``
  (``NotSupported``). The log then says ``Planned-buffer copy needs host memory on both sides``.
* **Do not let it allocate its outputs either:** ``alloc_graph_output=False`` keeps the outputs
  out of the program's memory. From Python, the runtime allocates GPU memory for each output, and
  the next call does not overwrite an output you still hold, unless you pass
  ``clone_outputs=False``. From C++, you must give the method your own GPU buffer once with
  ``Module::set_output``, before the first run. If you keep outputs planned instead, the next call
  overwrites them, so Python's ``execute()`` copies them on every call unless you pass
  ``clone_outputs=False``.

Pass CUDA tensors to a program exported this way. ExecuTorch's Python runtime refuses a CPU
tensor. From C++, a CPU tensor costs a copy on every call when the first delegate to use that
input is TensorRT, and fails when it is the CUDA backend.

Keep the engines in a separate file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default the engines are stored inside the ``.pte``, so the file is as large as the model. With
``external_engine_data``, they go to an ExecuTorch data file (``.ptd``) next to the ``.pte``
instead. Give the name without the extension:

.. code-block:: python

    torch_tensorrt.save(
        trt_gm,
        "small.pte",
        output_format="executorch",
        external_engine_data="small",
    )

This writes a small ``small.pte`` and a ``small.ptd`` next to it. Pass the ``.ptd`` file itself,
not its directory, when you load the program. Without it, loading the method fails. The Python runtime takes one data file per
program, so for a coalesced program pass ``external_engine_data="aoti_cuda_blob"``: the engines
then go into the CUDA backend's file.

Treat a ``.ptd`` file as trusted input, the same as a ``.pte``. The engines in it are GPU code.

.. warning::

    The ``.ptd`` name does not follow the ``.pte`` name, and ``save()`` replaces any file with that
    name. Save two programs into one directory with the same ``external_engine_data``, and the
    ``.ptd`` holds only the second program's engines, so the first fails to load. Name the file
    after its program, as above, or give each program its own directory.

----

Run from Python
----------------

Import the delegate package once, before you load a program. The import is what registers the
delegate. Everything after it is ExecuTorch's own API:

.. code-block:: python

    from pathlib import Path

    import torch
    import torch_tensorrt_executorch_runtime  # noqa: F401
    from executorch.runtime import Runtime

    program = Runtime.get().load_program(Path("model.pte"))
    forward = program.load_method("forward")
    outputs = forward.execute((torch.randn(2, 3, 4, 4),))

If the delegate cannot load, the import raises an error right away, instead of a program failing
to load later. Torch-TensorRT itself is needed only to export, not to run. A program exported as
above takes CPU tensors.

If the export wrote a data file, such as ``aoti_cuda_blob.ptd`` from a coalesced export, pass it.
The runtime does not look for it on its own, and if the file holds weights or engines, loading
fails without it:

.. code-block:: python

    program = Runtime.get().load_program(
        Path("coalesced.pte"), data_path=Path("aoti_cuda_blob.ptd")
    )
    outputs = program.load_method("forward").execute((torch.ones(64, 64),))

A program that keeps its inputs and outputs on the GPU takes CUDA tensors and returns CUDA tensors:

.. code-block:: python

    program = Runtime.get().load_program(Path("device_resident.pte"))
    outputs = program.load_method("forward").execute(
        (torch.ones(2, 3, 4, 4, device="cuda"),)
    )

.. note::

    ``torch_tensorrt.load(path, format="executorch")`` is deprecated. It copies CUDA inputs to the
    CPU, so it runs only programs that take CPU inputs and have no data file. Use the runtime API
    above instead.

----

Run from C++
-------------

The wheels ship a prebuilt delegate and a CMake package, so a C++ application can link it without
building anything. The steps below are for Linux. Put the program shown further down in
``main.cpp``, the ``model.pte`` from `Export a Model`_ in the same folder, and this next to them as
``CMakeLists.txt``:

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.28)
    project(my_app CXX)
    set(CMAKE_CXX_STANDARD 17)

    find_package(executorch REQUIRED COMPONENTS backend_cuda kernels_optimized)
    find_package(executorch_backend_tensorrt REQUIRED)

    add_executable(my_app main.cpp)
    target_link_libraries(my_app PRIVATE
      executorch::runtime
      executorch::backend_cuda
      executorch::backend_tensorrt
      executorch::kernels_optimized
    )

* ``kernels_optimized`` provides the copy operators that move data between the CPU and the GPU at
  the edges of the program. Without it, loading fails with
  ``kernel 'et_copy::_h2d_copy.out' not found``.
* ``backend_cuda`` registers the allocator for the program's GPU memory. A program that uses only
  TensorRT needs it too. Without it, loading fails with ``No device allocator registered``.

The two CMake packages are inside two Python packages, so point CMake at both. ExecuTorch's
location comes from its package metadata, because ``executorch.__file__`` is not set. Build and
run in the Python environment where you installed the packages:

.. code-block:: bash

    ET_CMAKE=$(python -c 'import importlib.metadata as m, pathlib; print(pathlib.Path(str(m.distribution("executorch").locate_file("executorch"))) / "share" / "cmake")')
    DELEGATE=$(TORCH_TENSORRT_SKIP_DELEGATE_REGISTRATION=1 python -c 'import pathlib, torch_tensorrt_executorch_runtime as r; print(pathlib.Path(r.__file__).parent)')
    cmake -S . -B build -DCMAKE_PREFIX_PATH="${ET_CMAKE};${DELEGATE}"
    cmake --build build
    ./build/my_app

The variable in the second line lets Python import the delegate package to read its path, without
loading the delegate.

You need CMake 3.28 or newer: on older versions the ExecuTorch package does not provide the
``executorch::`` targets used above. The runtime wheel README shows a way for CMake 3.19 to 3.27.
CMake is not in these wheels, so install it yourself. You do not need to find TensorRT: the
delegate finds it in the TensorRT wheel on its own.

There is no header to include. The delegate registers itself when its library loads, and the rest
is the ordinary ExecuTorch C++ API:

.. code-block:: cpp

    #include <cstdio>

    #include <executorch/extension/module/module.h>
    #include <executorch/extension/tensor/tensor.h>

    using namespace executorch::extension;

    int main() {
      Module module("model.pte");

      std::vector<float> data(2 * 3 * 4 * 4, 1.0f);
      auto input = make_tensor_ptr({2, 3, 4, 4}, std::move(data));

      const auto outputs = module.forward(input);
      if (!outputs.ok()) {
        printf("forward failed\n");
        return 1;
      }
      printf("first output value: %f\n",
             outputs->at(0).toTensor().const_data_ptr<float>()[0]);
      return 0;
    }

If the export wrote a data file, pass it to the constructor:

.. code-block:: cpp

    Module module("coalesced.pte", "aoti_cuda_blob.ptd");

On Linux, linking the target also records the wheel's library directory in your binary, so the
application finds the delegate with no library path set. That is right for a local build, but not
for anything you ship to other machines. To turn it off, set both of these before
``find_package``, and then ship the delegate yourself:

.. code-block:: cmake

    set(EXECUTORCH_BACKEND_TENSORRT_EMBED_RUNPATH OFF)
    set(CMAKE_SKIP_BUILD_RPATH ON)

To build the delegate from source instead, follow the
`delegate README <https://github.com/pytorch/TensorRT/blob/main/cpp/src/torch_tensorrt/executorch/README.md>`_.
The same README lists the rules a C++ caller has to keep (threads, buffers, streams), the load
errors, and the known issues. ``examples/executorch_reference_runner/`` is a complete C++
example.

A program exported as in `Keep inputs and outputs on the GPU`_, with its outputs left to the caller
by ``alloc_graph_output=False``, does not allocate them. Give it a buffer on the engine's GPU for
each output once, before the first call, and every call writes there. Without it, the first call
fails. Pass a GPU ``input``, made the same way from ``input_tensor_meta(0)`` and filled with your
data, and reuse it from call to call. Set any options from `Setting a runtime option`_ before
``method_meta``, which loads the program. ``cudaMalloc`` needs the CUDA toolkit: add
``find_package(CUDAToolkit REQUIRED)`` and ``CUDA::cudart`` to the CMake file. The snippet assumes
the engine's GPU is the current device.
For output 0 (the statements go in ``main()``):

.. code-block:: cpp

    #include <cuda_runtime.h>

    #include <executorch/extension/module/module.h>
    #include <executorch/extension/tensor/tensor.h>

    using namespace executorch::extension;
    using namespace executorch::runtime;

    Module module("device_resident.pte");

    const auto meta = module.method_meta("forward")->output_tensor_meta(0);
    void* data = nullptr;
    cudaMalloc(&data, meta->nbytes());
    auto output = from_blob(data, {meta->sizes().begin(), meta->sizes().end()}, meta->scalar_type());
    if (module.set_output(output, 0) != Error::Ok) {
      return 1;
    }

    for (int step = 0; step < num_steps; ++step) {
      // Each call writes into data.
      if (!module.forward(input).ok()) {
        return 1;
      }
    }

``examples/executorch_module_consumer/main.cpp`` makes the same calls for every input and output.

----

Runtime Options
----------------

Some of the options below are saved in the ``.pte`` at export, and some are set when the program
runs.

Setting a runtime option
^^^^^^^^^^^^^^^^^^^^^^^^^

A *load option* applies to one program. In C++, pass it to ``Module::load`` before the first call:

.. code-block:: cpp

    #include <executorch/extension/module/module.h>

    using namespace executorch::extension;
    using namespace executorch::runtime;

    Module module("model.pte");
    BackendOptions<1> options;
    options.set_option("use_cuda_graphs", false);
    LoadBackendOptionsMap backend_options;
    backend_options.set_options("TensorRTBackend", options.view());
    const Error error = module.load(backend_options);

In Python, pass it to ``load_program``:

.. code-block:: python

    program = Runtime.get().load_program(
        Path("model.pte"),
        backend_options={"TensorRTBackend": {"use_cuda_graphs": False}},
    )

A *process option* applies to every program loaded after it is set. In C++:

.. code-block:: cpp

    #include <executorch/runtime/backend/interface.h>

    using namespace executorch::runtime;

    BackendOptions<1> options;
    options.set_option("use_shared_activation_scratch", true);
    const Error error = set_option("TensorRTBackend", options.view());

``Error::NotFound`` from ``set_option`` means the delegate is not linked into your binary.

In Python:

.. code-block:: python

    Runtime.get().backend_registry.set_option(
        "TensorRTBackend", {"use_shared_activation_scratch": True}
    )

.. _executorch_single_stream:

Choosing the CUDA stream
^^^^^^^^^^^^^^^^^^^^^^^^^

With no stream chosen, both delegates run on ``cudaStreamPerThread``, the default stream of the
calling thread. So a coalesced program run from one thread is already in the right order.

To run every delegate on a stream of your own, put one guard around the whole call:

.. code-block:: cpp

    #include <executorch/extension/cuda/caller_stream.h>

    using namespace executorch::extension;

    cuda::CallerStreamGuard guard(stream);
    module.forward(input);

The guard is in the ``extension_cuda`` library, so add it to the CMake recipe above. Your own CUDA
calls, such as creating the stream, also need the CUDA runtime library:

.. code-block:: cmake

    find_package(executorch REQUIRED COMPONENTS backend_cuda extension_cuda kernels_optimized)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(my_app PRIVATE executorch::extension_cuda CUDA::cudart)

* One guard reaches every delegate, because they all share one ``extension_cuda`` library.
* The stream must be on the same device as the engine.
* The CUDA backend refuses a caller stream for a method that uses its own CUDA graph option.
* **Synchronize the stream before you read GPU outputs.** The TensorRT delegate always waits for
  its work to finish, but the CUDA backend can return while its work is still queued.
* If another thread uses the same stream at the same time, for example to launch a CUDA graph and
  synchronize the stream, compile with ``max_aux_streams=0``. NVIDIA drivers 580 and 595 were seen
  to drop a stream's wait on an event in that case, and TensorRT joins its auxiliary streams back to
  your stream with such a wait at the end of each call. When that wait is dropped, the call returns
  wrong outputs with no error, and with the shared activation scratch on, the next call that shares
  it can start while those kernels still use it. Replayed CUDA graphs are not affected.

That stream can belong to a CUDA green context. A green context owns a fixed number of the GPU's SMs
(streaming multiprocessors), so the model stays inside that part of the GPU and the rest is free for
other work. Create the stream with ``cuGreenCtxStreamCreate`` and put the same guard around the
call. ``cuGreenCtxStreamCreate`` is a CUDA driver function, so also link ``CUDA::cuda_driver``. The
green context does not need to be made current. TensorRT then warns that the engine was built
without green context constraints, which it says may cause crashes or undefined behavior.
Torch-TensorRT has no option to build engines with those constraints.

The reference runner in ``examples/executorch_reference_runner/`` can do this for you. Build it as
its README shows (the prebuilt copy described below cannot run this program), then pass the number
of SMs:

.. code-block:: bash

    example_executorch_runner --model_path=coalesced.pte --green_context_sms=8

If the device cannot provide the partition, the runner stops with exit status 2 instead of using
an ordinary stream. ``0``, the default, uses an ordinary stream. The runner takes no data file, so
it runs a coalesced program only when the CUDA part has no weights, like the one
``export_coalesced.py`` writes, and it cannot run a program saved with ``external_engine_data``.

A prebuilt copy of the runner ships as ``torch_tensorrt/bin/example_executorch_runner``, in
``libtorchtrt.tar.gz`` and in the Torch-TensorRT wheels for Linux, except the JetPack and
TensorRT-RTX builds. It has no CUDA backend, and its only CPU operators are the copies at the
edges of the program. So it runs only programs that TensorRT takes in full, not coalesced ones.

CUDA graph replay
^^^^^^^^^^^^^^^^^^

With replay on, the delegate records an engine's kernel launches once as a CUDA graph, and then
replays the whole engine with a single launch. This helps engines with fixed shapes that launch
many short kernels, where the CPU work of launching them is a large part of each call. It is off
by default. Turn it on at export:

.. code-block:: python

    torch_tensorrt.save(
        trt_gm,
        "model.pte",
        output_format="executorch",
        use_cuda_graphs=True,
    )

``save()`` and ``torch_tensorrt.executorch.export()`` take ``True``, ``False``, or ``None`` (the
default, which saves no choice). The choice applies to every TensorRT engine in the program. At
run time, the first of these that is set decides:

#. the ``use_cuda_graphs`` load option,
#. a process option set to ``False``,
#. the choice saved in the program,
#. a process option set to ``True``.

With none of them set, replay is off.

A replay reads and writes the buffers it was recorded on. Keep each input and output in the same
device buffer from call to call, as the memory planning of ``torch_tensorrt.save()`` does, and
replays copy nothing. With the recipe in `Keep inputs and outputs on the GPU`_, memory planning
does not own them: from Python pass ``clone_outputs=False``, or from C++ give each output once with
``Module::set_output``, and reuse the same input tensors. These are still copied on every call:
host buffers (unless the GPU reads host memory directly), buffers off a 256-byte boundary,
non-row-major inputs, and an output that shares memory with an input. A graph goes on the caller's
buffers once a call brings the same buffers as the call before it. Until then the engine records on
GPU buffers of its own and copies the inputs in and the outputs out on every call after the first.
When the buffers of a graph on the caller's buffers change, that call records again on the new ones.
When they change a second time, for any input shapes, that graph records on its own buffers and
copies on every call, also after the shapes or the profile change. The plain graph and the graph of
each green context count their own changes. With few kernels, or with large inputs and outputs,
those copies can cost more than the launches they save. So buffers that move on every call, or
alternate between two sets, cost one recording for each set of shapes in each graph, as buffers that
stay put do, and both first replay on the 3rd call. Each growth of the shared activation scratch in
between adds a call without a graph and a recording, and moves the first replay later.

CUDA allows a green context handle on one CPU thread at a time. The delegate takes a lock around
its own calls on the handle, but your threads do not take that lock, so do not use the same handle
from another thread while a call runs on one of its streams.

A graph runs its kernels in the CUDA context it was recorded in, so a green context's calls keep a
graph of their own, recorded there, and its replays stay on the green context's SMs. Kernels that
TensorRT runs on auxiliary streams it creates itself stay there too with TensorRT 11.4 and newer.
TensorRT 11.1 and older create those streams outside the green context, so their kernels can run on
any SM; with those versions, compile with ``max_aux_streams=0`` to keep all of them on the green
context's SMs. TensorRT 11.2 and 11.3 were not checked; with them, compile with ``max_aux_streams=0``
and destroy the engine before its green contexts. An engine keeps such a graph for up to 4 green
contexts, and one for plain streams, so a caller that uses both replays on each, as long as every caller passes the same input shapes: a
call with other shapes starts every graph over. A fifth green context takes the place of the one
called longest ago, and a replaced one starts over with a warm-up call, so one call on each of more
than 4 green contexts in turn never replays.

Destroy the engine before the regular CUDA context it was loaded in, and before the green contexts
it ran on. A call after the load context is destroyed can crash, also when a new context was given
the destroyed one's handle. TensorRT requires a green context to outlive every CUDA graph recorded
in it, and the engine keeps such a graph until the engine is destroyed or a newer green context
takes its place. With TensorRT 11.4 and newer, an engine with auxiliary streams needs this order
even with replay off, since TensorRT makes those streams in the green context.

An engine loaded while a green context is current is created in the device's primary context, which
it keeps until it is destroyed, so that green context may be destroyed first if the engine never
runs on its streams. This needs a CUDA 12.5 or newer driver. With an older one, the engine is
created in that green context.

Some calls never replay:

* An engine whose input shapes change on every call never replays.
* An engine with aliased outputs, a GPU without stream-ordered memory, and a driver older than
  CUDA 12.5 never replay. They take the ordinary path.
* A green context stream while a regular context other than its device's primary one is current,
  and a plain stream while a green context is current, take the ordinary path.
* A stream of another device fails, with or without replay. So does a stream of a regular CUDA
  context other than the one the engine was loaded in, except that, while the stream's own context is
  current, the engine's own graph replays on it. When that caller's buffers are not the
  ones the graph was recorded on, and the graph is not already on buffers of its own, the engine
  records again in its own context, on buffers of its own, and from then on every call of the engine
  that replays on a plain stream, its own included, copies the inputs in and the outputs out until
  its shapes or its profile change. When that recording cannot go ahead (the engine's own context
  has not run the shapes yet, or the recording fails), the call fails with an error that says why.
  The engine's own calls keep their graph, unless that call brought new shapes, which drop it. The
  buffers the delegate keeps for such a call are allocated in the engine's context, so that other
  context may be destroyed first.
* An engine on the shared activation scratch fails on a stream of another device or of another
  regular CUDA context than the one it was loaded in, with or without replay. Such a call fails
  before it sets its shapes or claims the shared scratch, and the engine keeps its graphs. Green
  context streams of its device run when it was loaded in the device's primary context. The scratch
  and its completion event are one per device, made in the context of the device's first pooled
  call, so load every pooled engine on a device in the same regular context. A pooled engine loaded
  in another one fails before it runs. Keep that context alive while the process uses the scratch:
  the scratch, and every larger buffer it grows into, is made in it. With a driver older than CUDA
  12.5, that context is a green context when the first pooled engine was loaded while one was
  current, so keep that green context alive too. The check on the stream uses cuStreamGetCtx_v2,
  which CUDA 12.5 added, so with an older driver a call on a stream of another device or context
  still fails, but only later, after it claims the scratch.

An engine on the shared activation scratch below also replays. It records again whenever the
shared buffer grows.

.. warning::

    While a graph is being recorded, no other thread may create or destroy a TensorRT engine or
    execution context outside this delegate, or call a whole device sync such as
    ``cudaDeviceSynchronize()`` or ``torch.cuda.synchronize()``. Loading a program with ExecuTorch's
    CUDA backend can make such a sync too. Recording can happen again:

    * after a change of shapes, buffers or profile,
    * after the shared activation scratch grows, which another engine can cause,
    * in each new or replaced green context,
    * on the first call from another regular CUDA context with other buffers.

    So loading everything first does not remove the risk. Leave replay off when other threads may
    do any of these things.

Shared activation scratch
^^^^^^^^^^^^^^^^^^^^^^^^^^

Every TensorRT execution context keeps its own scratch memory for intermediate results, for as
long as it lives. A model split into many small engines pays for that many times, and can run out
of GPU memory on the number of engines alone. With the ``use_shared_activation_scratch`` process
option, all contexts on a device share one buffer instead. The buffer grows to the largest size
any call has needed, so you save the sum of the separate sizes minus the largest one.

Set the option before you load the methods that should use the shared buffer. Methods loaded
before it keep their own scratch. Load every method that uses it on a device in the same regular
CUDA context, and keep that context alive while the process uses the buffer, with replay on or off.
`CUDA graph replay`_ has the detail.

The cost is parallelism. Two calls that share the buffer never run at the same time on one device:
the delegate takes a per-device lock from claiming the buffer to queuing the work, and each call's
stream waits for the call before it on a CUDA event and on a small flag in GPU memory. A GPU without
64-bit stream memory operations has no such flag, so there the delegate holds the lock until the
call has finished. The buffer never shrinks. Do not call ``cudaDeviceReset()`` on a device that ran
an engine with the shared buffer. Recording a call into your own CUDA graph is not supported, with
or without the shared buffer.

Shared engines
^^^^^^^^^^^^^^^

When the same program is loaded twice in one process, for example once per robot arm, both loads
share one TensorRT engine. The engine's weights are then in GPU memory once. Each load still gets
its own execution context and buffers. Loads share an engine when they have the same engine
bytes, the same device, and the same weight streaming budget.

This is on by default. On an 8 GB Jetson Orin Nano, loading one robot policy a second time took
806 MiB more GPU memory without sharing, and 122 MiB more with it. To find a match, every load
hashes the engine bytes, which takes about 0.15 seconds per GB on a Jetson AGX Thor. The hash
does not resist deliberate collisions, so every program in the process must be trusted.

To keep one program's engines private, pass the ``use_shared_engines`` load option as ``false``.

Weight streaming
^^^^^^^^^^^^^^^^^

An engine compiled with ``enable_weight_streaming=True`` can keep part of its weights in CPU
memory and stream them to the GPU when needed. The budget is the number of bytes of weights that
stay on the GPU. Save one with ``weight_streaming_budget_per_engine``, or set one at run time
with the ``weight_streaming_budget`` load option, which takes the number of bytes as a decimal
string, or as an int below 2 GiB. The budget applies to each engine separately, so a program with N engines can keep up to N times
the budget on the GPU. With no budget, TensorRT chooses one from the free memory of the device.

Optimization profiles
^^^^^^^^^^^^^^^^^^^^^^

An engine can hold several optimization profiles: one set of weights, tuned for several ranges
of input shapes. The usual case is an LLM, with one profile for decoding one token at a time and
one for a long prompt. Declare the profiles with ``torch_tensorrt.Input(profiles=[...])``, as
:ref:`multi_optimization_profiles_tutorial` shows. A C++ application then picks the profile for
each call with ``OptimizationProfileGuard``:

.. code-block:: cpp

    #include <torch_tensorrt/executorch/TensorRTBackend.h>

    using torch_tensorrt::executorch_backend::OptimizationProfileGuard;

    constexpr int32_t kDecodeProfile = 0;  // same order as the profiles at export
    constexpr int32_t kPrefillProfile = 1;

    {
      OptimizationProfileGuard guard(kPrefillProfile);
      module.forward(prefill_inputs);
    }
    {
      OptimizationProfileGuard guard(kDecodeProfile);
      module.forward(decode_inputs);
    }

To let each engine choose from the input shapes instead, create the guard with
``OptimizationProfileGuard::automatic()``. With no guard, every engine runs profile 0, and Python
cannot choose a profile. The header is
``cpp/include/torch_tensorrt/executorch/TensorRTBackend.h`` in the source tree, and
``torch_tensorrt/include/torch_tensorrt/executorch/TensorRTBackend.h`` in ``libtorchtrt.tar.gz``.
The wheel ships no header. On Linux, an application can still compile against the header from
the source, which also needs the TensorRT headers, and link the wheel's delegate. On Windows, the
wheel's library does not export the guard.

----

Comparison: .pte vs .pt2
-------------------------

A ``.pt2`` file is the other way to run a Torch-TensorRT model without Python: AOTInductor,
PyTorch's ahead-of-time compiler, packages the whole model, and :ref:`aot_inductor` describes it.

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Feature
     - ``.pte`` (ExecuTorch)
     - ``.pt2`` (AOTInductor)
   * - Python load
     - ``executorch.runtime.Runtime``
     - ``torch._inductor.aoti_load_package``
   * - C++ load
     - ``executorch::extension::Module``
     - ``AOTIModelPackageLoader``
   * - libtorch at runtime
     - Not required
     - Required
   * - Operators TensorRT rejects
     - ExecuTorch CUDA backend, or the CPU
     - Compiled by AOTInductor
   * - Platform
     - Linux (x86-64 and aarch64) and Windows x86-64, with CUDA 13
     - Linux

----

Examples
---------

In ``examples/torchtrt_executorch_example/``:

* ``export_static_shape.py`` writes a minimal ``.pte``.
* ``export_dynamic_shape.py`` exports a model whose batch size can change.
* ``export_coalesced.py`` splits one graph between the TensorRT delegate and the CUDA backend.
* ``export_device_resident.py`` does the same with no copies at the edges of the program.
* ``export_kv_cache_decode.py`` exports a decoder step whose KV cache stays in memory you own.
* ``export_multi_profile.py`` exports one engine with two optimization profiles.
* ``export_lerobot_pi05.py`` deploys a robot policy: LeRobot's pi0.5, a vision-language-action
  model with 3.6 billion parameters, as one program that predicts a chunk of 50 actions.

``examples/executorch_reference_runner/`` loads and runs a ``.pte`` from Python and from C++,
including optimization profiles and the green context option.
