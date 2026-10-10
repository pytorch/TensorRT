.. _executorch_deployment:

ExecuTorch Deployment
=====================

**ExecuTorch** is PyTorch's runtime for edge devices. It loads a single ``.pte`` file, which
holds the model's program and all of its compiled payloads, and runs it from a small C++
library that does not link libtorch. That makes it a good fit for a robot, a drone, or any
device where a full PyTorch install is too heavy.

Torch-TensorRT writes that file for you. TensorRT compiles the operators it can convert into
engines, the engines are stored inside the ``.pte``, and a delegate hands each one back to
TensorRT at run time. Operators TensorRT does not take can be compiled by ExecuTorch's own
CUDA backend in the same program, so the whole model stays on the GPU.

**When to use ExecuTorch**

* Deploying to an edge device or a C++ application with no Python and no libtorch.
* Shipping one file that holds the program, the TensorRT engines, and any fallback kernels.
* Driving the model from your own CUDA stream, including a green-context stream that pins the
  work to part of the GPU.

.. note::

    The ExecuTorch delegate is Linux only, on x86_64 and aarch64, and it requires CUDA 13.
    Ordinary Torch-TensorRT builds keep their separate CUDA 12 support.

----

Installation
-------------

One command installs everything. The ``executorch`` extra brings the delegate wheel and a
CUDA build of ExecuTorch, and Torch-TensorRT brings PyTorch. Take the release build unless
you need something that has not shipped yet.

Release builds
^^^^^^^^^^^^^^^

.. code-block:: bash

    pip install "torch-tensorrt[executorch]" \
      --index-url https://download.pytorch.org/whl/cu132 \
      --extra-index-url https://pypi.org/simple \
      --extra-index-url https://pypi.nvidia.com

Swap ``cu132`` for the channel that matches your CUDA, such as ``cu134`` for CUDA 13.4. Keep
PyTorch, ExecuTorch and Torch-TensorRT on the same channel.

.. note::

    Torch-TensorRT 2.15 is the first release whose ``executorch`` extra pulls the delegate
    wheel. On 2.14 the extra installs ExecuTorch but not the delegate, so use a nightly
    until 2.15 is out.

Nightly builds
^^^^^^^^^^^^^^^

Nightlies carry delegate changes before a release does. Use one to pick up something that
has just landed, or to report a problem against current ``main``:

.. code-block:: bash

    pip install --pre "torch-tensorrt[executorch]" \
      --index-url https://download.pytorch.org/whl/nightly/cu132 \
      --extra-index-url https://pypi.org/simple \
      --extra-index-url https://pypi.nvidia.com

``--pre`` is required here. Without it pip ignores the nightly and takes the stable
Torch-TensorRT from the public index instead.

What the extra installs
^^^^^^^^^^^^^^^^^^^^^^^^

All three indexes are needed either way. Without NVIDIA's index the inference library
resolves to a source distribution, and pip spends around twenty minutes trying to build it
before failing.

The extra installs a companion wheel, ``torch-tensorrt-executorch-runtime``. It ships one
shared library holding the TensorRT delegate, which registers itself with the ExecuTorch
runtime from the ``executorch`` distribution rather than bundling a runtime of its own.

A CUDA build of ExecuTorch is required at run time, not only to build against. A processor
only build installs and then fails on import. Use a fresh virtual environment so the install
cannot disturb a working stack.

To export a coalesced program you also need a CUDA toolkit, because ExecuTorch's CUDA backend
compiles the leftover operators with ``nvcc``.

----

Compile and Save
-----------------

The workflow is the standard ``ir="dynamo"`` path, with two extra arguments to
``torch_tensorrt.save``:

* ``output_format="executorch"`` selects the ``.pte`` writer.
* ``retrace=False`` is recommended. It keeps the compiled graph as it is instead of
  re-exporting it, so each TensorRT engine is still there when ExecuTorch's partitioner runs.

.. code-block:: python

    import torch
    import torch_tensorrt

    model = MyModel().eval().cuda()
    example_inputs = (torch.randn(2, 3, 4, 4, device="cuda"),)

    exported = torch.export.export(model, example_inputs)
    trt_gm = torch_tensorrt.dynamo.compile(
        exported,
        arg_inputs=example_inputs,
        min_block_size=1,
    )

    torch_tensorrt.save(
        trt_gm,
        "model.pte",
        output_format="executorch",
        arg_inputs=example_inputs,
        retrace=False,
    )

:ref:`executorch_save` covers the rest of the export surface: dynamic shapes, several methods
in one file, a zero-copy KV cache, and the two-step
``torch_tensorrt.executorch.export()`` path for programs that need work before they are
written to disk.

Coalescing the leftover operators onto the CUDA backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TensorRT has no converter for every ATen operator. By default the operators it rejects run on
the CPU, which costs a copy in each direction on every call. Pass a ``CudaPartitioner`` and
they are compiled by ExecuTorch's CUDA backend instead, so nothing leaves the GPU:

.. code-block:: python

    from executorch.backends.cuda.cuda_backend import CudaBackend
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner

    torch_tensorrt.save(
        trt_gm,
        "coalesced.pte",
        output_format="executorch",
        arg_inputs=example_inputs,
        retrace=False,
        partitioners=[
            CudaPartitioner([CudaBackend.generate_method_name_compile_spec("forward")])
        ],
    )

The TensorRT partitioner always runs first and the ``CudaPartitioner`` picks up the rest. For
a model such as ``cos(erfinv(tanh(x)))``, where TensorRT cannot take ``erfinv``, the delegate
list in the saved program reads ``['TensorRTBackend', 'CudaBackend', 'TensorRTBackend']``.

.. warning::

    The CUDA backend names its external weight file per device, not per model, so saving two
    coalesced programs into one directory overwrites the first one's weights. The first
    program still loads, still reports finding its weights, and returns a wrong answer with
    no error. Give each export its own directory.

Keeping inputs and outputs on the GPU
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default ExecuTorch inserts a host-to-device copy before the first delegate and a
device-to-host copy after the last one, so a method is safe to call with CPU tensors. For a
pipeline whose data is already on the GPU those copies are pure overhead:

.. code-block:: python

    from executorch.exir import ExecutorchBackendConfig
    from executorch.exir.passes import MemoryPlanningPass
    from executorch.exir.passes.propagate_device_config import PropagateDeviceConfig

    torch_tensorrt.save(
        trt_gm,
        "device_resident.pte",
        output_format="executorch",
        arg_inputs=example_inputs,
        retrace=False,
        backend_config=ExecutorchBackendConfig(
            propagate_device_config=PropagateDeviceConfig(
                skip_h2d_for_method_inputs=True,
                skip_d2h_for_method_outputs=True,
            ),
            enable_non_cpu_memory_planning=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        ),
    )

Three things are easy to miss:

* **Both skip flags need** ``enable_non_cpu_memory_planning=True``. Copy insertion happens
  during device-aware memory planning, and asking for a skip with it off raises.
* **Inputs must be unplanned**, through ``MemoryPlanningPass(alloc_graph_input=False)``.
  Without it the program reserves its own input buffer and the runtime fills it from the
  caller's memory, which puts the copy straight back.
* **Leave the outputs planned if Python will run the program.** The program's own device
  arena then owns the output. Add ``alloc_graph_output=False`` only for a C++ consumer that
  supplies the output address itself with ``Module::set_output``; a Python caller has no way
  to hand one in.

The choice is baked into the file. A program exported this way wants CUDA tensors. Handing it
a host tensor still returns the right answer, but it stages a copy on every call, which is
the cost the export existed to remove, and nothing warns you.

----

Python Inference
-----------------

Import the delegate package once, anywhere before a program is loaded. The import is what
registers the backend, and nothing else about your code changes. Loading and running is
ExecuTorch's own API:

.. code-block:: python

    from pathlib import Path

    import torch
    import torch_tensorrt_executorch_runtime  # noqa: F401
    from executorch.runtime import Runtime

    program = Runtime.get().load_program(Path("model.pte"))
    forward = program.load_method("forward")
    outputs = forward.execute((torch.ones(2, 3, 4, 4),))

If the delegate cannot be loaded, that import raises straight away, rather than letting the
failure surface later as a program that will not load.

A coalesced program needs nothing extra here. Both backends are registered, and the program
records which parts go where. Torch-TensorRT itself is not needed at inference time; it is an
export-time dependency.

.. note::

    ``torch_tensorrt.load(path, format="executorch")`` still works but is deprecated. It
    copies CUDA inputs to the CPU and supports embedded weights only. New applications should
    use the Runtime API above, and a device-resident program has to.

----

C++ Inference
--------------

The wheels ship a prebuilt delegate and a CMake package, so a C++ application can link them
without building anything from source:

.. code-block:: cmake

    find_package(executorch REQUIRED COMPONENTS backend_cuda extension_cuda kernels_optimized)
    find_package(executorch_backend_tensorrt REQUIRED)

    target_link_libraries(my_app PRIVATE
      executorch::runtime
      executorch::backend_cuda
      executorch::backend_tensorrt
      executorch::extension_cuda
      executorch::kernels_optimized
    )

``kernels_optimized`` supplies the ``et_copy`` operators that move data across the method
boundary. ``backend_cuda`` registers the device allocator those copies use, so it is needed
even by a program that carries only the TensorRT delegate. Without it the program loads, the
engine initializes, and the first instruction fails with
``_h2d_copy: no device allocator registered``. ``extension_cuda`` provides
``CallerStreamGuard``, used below to choose the CUDA stream.

The two packages live in two distributions, so point CMake at both. ExecuTorch is a namespace
package, so its path has to come from its distribution metadata rather than from
``__file__``:

.. code-block:: bash

    cmake -DCMAKE_PREFIX_PATH="$(TORCH_TENSORRT_SKIP_DELEGATE_REGISTRATION=1 python -c 'import importlib.metadata as m, torch_tensorrt_executorch_runtime as r, pathlib; print(str(pathlib.Path(str(m.distribution("executorch").locate_file("executorch"))) / "share" / "cmake") + ";" + str(pathlib.Path(r.__file__).parent))')" ...

The environment variable in front keeps that a path query. Importing the package normally
loads the delegate, and a path does not need it loaded.

CMake 3.28 or newer is required for the form above, because the ``backend_cuda`` component
rejects older versions: they write the ``$ORIGIN`` token in a runtime search path
incorrectly. CMake is not part of these wheels, and a freshly imaged Jetson has none at all.

There is no header to include for the delegate. It registers itself with ExecuTorch's backend
registry from a static initializer inside the shared library, and everything after that is
the ordinary ExecuTorch C++ API:

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

Linking the target also records the wheel's own library directory in your binary, so the
application finds the delegate with no library path set. That is right for an application
built against an installed wheel and wrong for anything you intend to redistribute, so it can
be turned off. Set both of these before ``find_package``, then ship the delegate yourself:

.. code-block:: cmake

    set(EXECUTORCH_BACKEND_TENSORRT_EMBED_RUNPATH OFF)
    set(CMAKE_SKIP_BUILD_RPATH ON)

You do not need to find TensorRT. The delegate records where to look relative to its own
location, so the loader resolves it from the sibling wheel without being told, and your
application uses the delegate's interface rather than TensorRT's.

Building the delegate from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The delegate source ships inside ``libtorchtrt.tar.gz`` as
``torch_tensorrt/src/torch_tensorrt/executorch/``. Turn on ExecuTorch's CUDA backend and the
extensions a ``Module`` app uses, add the delegate next to ExecuTorch, and link the target it
provides:

.. code-block:: cmake

    set(EXECUTORCH_BUILD_CUDA ON CACHE BOOL "" FORCE)
    set(EXECUTORCH_BUILD_EXTENSION_TENSOR ON CACHE BOOL "" FORCE)
    set(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON CACHE BOOL "" FORCE)
    set(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR ON CACHE BOOL "" FORCE)
    set(EXECUTORCH_BUILD_EXTENSION_MODULE ON CACHE BOOL "" FORCE)
    set(EXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP ON CACHE BOOL "" FORCE)

    add_subdirectory("executorch")
    add_subdirectory("torch_tensorrt/src/torch_tensorrt/executorch")

    target_link_libraries(my_runner PRIVATE
      executorch
      executorch::backends
      executorch::extensions
      executorch::kernels
      executorch::backend_tensorrt
    )

Building it needs CUDA Toolkit 12.5 or newer. ``libextension_cuda`` stays a shared library on
purpose, so that every CUDA-capable delegate in the process reads the same caller-stream
state. A static copy would give each delegate its own.

``examples/executorch_reference_runner/`` is a complete runnable C++ example, and
``libtorchtrt.tar.gz`` also carries it prebuilt as
``torch_tensorrt/bin/example_executorch_runner``.

----

Runtime Performance
--------------------

Choosing the CUDA stream
^^^^^^^^^^^^^^^^^^^^^^^^^

With no stream chosen, both delegates run on ``cudaStreamPerThread``, the default stream of the
calling thread. A coalesced program run from one thread is therefore ordered correctly as it
is, with nothing to add.

To run every delegate on a stream of your own, for example a green-context stream, scope a
guard over the whole execution:

.. code-block:: cpp

    #include <executorch/extension/cuda/caller_stream.h>

    using namespace executorch::extension;

    cuda::CallerStreamGuard guard(stream);
    module.forward(input);

One guard reaches every CUDA-capable delegate, because they all resolve the same shared
``libextension_cuda``, the ``extension_cuda`` component linked above. The stream must be on
the engine's device. The CUDA backend refuses a caller stream for a method that uses its own
CUDA graphs.

Only the TensorRT delegate always waits for its work before it returns. ExecuTorch's CUDA
backend can return once its work is queued, so ``execute()`` on a coalesced program can return
before the work finishes. When the guard sets a stream of your own, synchronize that stream
before reading GPU outputs on the host or from another stream.
:ref:`Running a coalesced .pte <executorch_single_stream>` describes the same rule for a
decode loop.

Green contexts
^^^^^^^^^^^^^^^

Because both delegates honour the caller's stream, that stream can be a CUDA green-context
stream. A green context holds a fixed number of streaming multiprocessors, so the model stays
inside that partition and the rest of the GPU is free for other work. Create the stream with
``cuGreenCtxStreamCreate`` and scope the same guard over it. The confinement rides the stream,
so the green context does not have to be made current.

The reference runner has this built in. Build it with the CUDA delegate enabled and pass the
SM count:

.. code-block:: bash

    example_executorch_runner --model_path=coalesced.pte --green_context_sms=8

It refuses with a distinct status rather than falling back when a green context cannot be
created, and says how many SMs the device has, so a passing run always means one was really
used. ``0``, the default, uses an ordinary stream.

CUDA graph replay
^^^^^^^^^^^^^^^^^^

Replay is off by default. With it on, the delegate records an engine's kernel launches once
as a CUDA graph and then replays the whole engine with a single launch. It helps engines with
fixed shapes that launch many short kernels, where the CPU launch work is a large share of
each call.

.. code-block:: python

    torch_tensorrt.save(
        trt_gm,
        "model.pte",
        output_format="executorch",
        arg_inputs=example_inputs,
        retrace=False,
        use_cuda_graphs=True,
    )

Both ``save()`` and ``torch_tensorrt.executorch.export()`` accept ``use_cuda_graphs=True``
(on), ``False`` (off) or ``None`` (no baked choice). The choice applies to every method's
TensorRT delegates and is read when an engine loads.

It is not free. Each engine keeps one stable device buffer per input and output, and every
replayed call copies each input in and each output out. For an engine with few kernels, or
with large inputs and outputs, those copies can cost more than the launches they save. Shapes
that change on every call never replay and still pay the copies.

Some engines never replay, even with replay on. An engine on the shared activation scratch
described below always runs without a graph, so turning on both options gives no replay.
Engines with aliased outputs, GPUs without stream-ordered memory, and drivers older than
CUDA 12.5 also run without one, as does a call on a green-context stream. The delegate logs
the reason once.

.. warning::

    Recording is unsafe next to some other work. While a recording runs, creating or
    destroying a TensorRT engine or execution context elsewhere in the process is unsafe, and
    so is a whole-device sync such as ``cudaDeviceSynchronize`` or ``torch.cuda.synchronize()``.
    Recording can recur throughout an engine's lifetime, because every input shape change
    starts another cycle, so loading everything up front does not end the risk. Leave replay
    off when other threads may do any of that.

A C++ host can refuse a program's saved request for one load:

.. code-block:: cpp

    #include <executorch/extension/module/module.h>

    using namespace executorch::extension;
    using namespace executorch::runtime;

    Error load_without_cuda_graphs(Module& module) {
      BackendOptions<1> options;
      const Error stored = options.set_option("use_cuda_graphs", false);
      if (stored != Error::Ok) {
        return stored;
      }
      LoadBackendOptionsMap by_backend;
      const Error mapped = by_backend.set_options("TensorRTBackend", options.view());
      if (mapped != Error::Ok) {
        return mapped;
      }
      return module.load(by_backend);
    }

Shared activation scratch
^^^^^^^^^^^^^^^^^^^^^^^^^^

A TensorRT execution context allocates its own activation scratch and holds it for as long as
the context lives. A model lowered to many small engines pays that cost once per engine, and
can run out of device memory on the engine count alone. This option backs all of a device's
contexts from one buffer instead, grown to the largest requirement any call has asked for:

.. code-block:: cpp

    #include <executorch/runtime/backend/interface.h>

    using namespace executorch::runtime;

    Error enable_shared_activation_scratch() {
      BackendOptions<1> options;
      const Error stored = options.set_option("use_shared_activation_scratch", true);
      if (stored != Error::Ok) {
        return stored;
      }
      return set_option("TensorRTBackend", options.view());
    }

Set it before loading the methods whose contexts should use the pool. The memory reclaimed is
the sum of the separate requirements less the largest of them. ``Error::NotFound`` means no
backend is registered under that name, which is what a binary that has not linked the
delegate gets.

The cost is parallelism. Contexts that share one buffer do not run at the same time on the
device: the backend holds a per-device lock from the claim on the buffer through the enqueue,
so two calls on one device are serialized at submission. The pool never shrinks, and a device
the backend has run a pooled engine on must not be reset with ``cudaDeviceReset()``.

Shared engines
^^^^^^^^^^^^^^^

Loading the same program twice in one process, for example once per robot arm, used to
deserialize its engine twice and hold its weights in device memory twice. Handles loaded from
the same engine bytes, for the same device and the same weight streaming request, now share
one engine. Each handle still gets its own execution context, buffers and lock.

This is on by default. Measured on an 8 GB Jetson Orin Nano, loading one robot policy twice
grew reported GPU memory by 806 MiB without sharing and by 122 MiB with it. Finding a match
means hashing the engine bytes on every load, which costs roughly 0.15 s per GB on a Jetson
AGX Thor. Pass the ``use_shared_engines`` load option as ``false``, the same way as
``use_cuda_graphs`` above, to keep one module's engines private.

----

Comparison: .pte vs .pt2
-------------------------

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
   * - Non-TRT ops
     - ExecuTorch CUDA backend, or CPU fallback
     - Compiled by AOTInductor
   * - Platform
     - Linux, CUDA 13
     - Linux

----

Examples
---------

* ``examples/torchtrt_executorch_example/export_static_shape.py`` writes a minimal ``.pte``.
* ``examples/torchtrt_executorch_example/export_coalesced.py`` splits one graph across the
  TensorRT delegate and ExecuTorch's CUDA backend.
* ``examples/torchtrt_executorch_example/export_device_resident.py`` does the same with no
  copies at the method boundary.
* ``examples/executorch_reference_runner/`` loads and runs a ``.pte`` from Python and from
  C++, including the green-context option.
