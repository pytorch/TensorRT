.. _executorch_deployment:

ExecuTorch Deployment
=====================

**ExecuTorch** is PyTorch's runtime for edge devices. It loads one ``.pte`` file, holding the
program and every compiled payload, from a small C++ library that does not link libtorch.
That suits a robot, a drone, or any device where a full PyTorch install is too heavy.

Torch-TensorRT writes that file. TensorRT compiles the operators it can convert into engines,
the engines are stored inside the ``.pte``, and a delegate hands each one back to TensorRT at
run time. Operators TensorRT will not take go to ExecuTorch's own CUDA backend in the same
program, so the whole model stays on the GPU.

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

Nightly builds
^^^^^^^^^^^^^^^

To pick up a change before it ships, or to report a problem against ``main``:

.. code-block:: bash

    pip install --pre --upgrade "torch-tensorrt[executorch]" \
      --index-url https://download.pytorch.org/whl/nightly/cu132 \
      --extra-index-url https://pypi.org/simple \
      --extra-index-url https://pypi.nvidia.com

* ``--pre`` is required. Without it pip takes the stable release instead.
* ``--upgrade`` is required if a nightly is already installed, or pip keeps the old one and
  tells you nothing.

You get the newest delegate nightly plus the one ExecuTorch build it was compiled against.
That ExecuTorch is often a few days older than the newest on the channel. This is correct,
not a stale resolve, and asking for a newer one by hand will not resolve at all.

What the extra installs
^^^^^^^^^^^^^^^^^^^^^^^^

* ``torch-tensorrt-executorch-runtime``, one shared library holding the TensorRT delegate.
  It names the ExecuTorch it was built against, 1.6 for this release, so pip picks a
  matching one. Install a different one by hand and the delegate refuses it at import.
* A CUDA build of ExecuTorch. A CPU-only build installs and then fails on import.

Naming NVIDIA's index is what gets you a prebuilt TensorRT. Leave it out and pip falls
back to a placeholder package on PyPI that fetches the same files during the install
step. That works when the machine can reach ``pypi.nvidia.com``, but it is slower, it
breaks in an offline or wheels-only install, and the error it gives is hard to read.

Use a fresh virtual environment. To export a coalesced program you also need a CUDA toolkit,
because ExecuTorch's CUDA backend compiles the leftover operators with ``nvcc``.

----

Compile and Save
-----------------

The usual ``ir="dynamo"`` path, with two extra arguments to ``torch_tensorrt.save``:

* ``output_format="executorch"`` selects the ``.pte`` writer.
* ``retrace=False`` keeps the compiled graph as it is. Re-exporting would drop the TensorRT
  engines before ExecuTorch's partitioner sees them.

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

:ref:`executorch_save` covers the rest: dynamic shapes, several methods in one file, a
zero-copy KV cache, and the two-step ``torch_tensorrt.executorch.export()`` path for
programs that need work before they are written.

Coalescing the leftover operators onto the CUDA backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TensorRT has no converter for every ATen operator, and by default the ones it rejects run on
the CPU, costing a copy each way on every call. Pass a ``CudaPartitioner`` and ExecuTorch's
CUDA backend compiles them instead, so nothing leaves the GPU:

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

TensorRT partitions first and ``CudaPartitioner`` takes the rest. For ``cos(erfinv(tanh(x)))``,
where TensorRT cannot take ``erfinv``, the saved program's delegate list reads
``['TensorRTBackend', 'CudaBackend', 'TensorRTBackend']``.

.. warning::

    The CUDA backend names its external weight file per device, not per model, so saving two
    coalesced programs into one directory overwrites the first one's weights. That program
    still loads, still reports finding its weights, and returns a wrong answer with no error.
    Give each export its own directory.

Keeping inputs and outputs on the GPU
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default ExecuTorch copies host to device before the first delegate and back after the
last, so a method is safe to call with CPU tensors. If your data is already on the GPU, those
copies are pure overhead:

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

* **Both skip flags need** ``enable_non_cpu_memory_planning=True``, because the copies are
  inserted during device-aware memory planning. Asking for a skip with it off raises.
* **Inputs must be unplanned**, through ``MemoryPlanningPass(alloc_graph_input=False)``.
  Otherwise the program reserves its own input buffer and the runtime fills it from the
  caller, which puts the copy straight back.
* **Leave outputs planned if Python will run the program**, so the program's device arena
  owns the output. Add ``alloc_graph_output=False`` only for a C++ caller that supplies the
  address with ``Module::set_output``. Python cannot.

The choice is baked into the file, so feed such a program CUDA tensors. A host tensor still
gives the right answer, but it copies on every call, which is the cost you exported to
avoid, and nothing warns you.

----

Python Inference
-----------------

Import the delegate package once, before any program is loaded. That import is what
registers the backend. Everything after it is ExecuTorch's own API:

.. code-block:: python

    from pathlib import Path

    import torch
    import torch_tensorrt_executorch_runtime  # noqa: F401
    from executorch.runtime import Runtime

    program = Runtime.get().load_program(Path("model.pte"))
    forward = program.load_method("forward")
    outputs = forward.execute((torch.ones(2, 3, 4, 4),))

If the delegate cannot load, that import raises at once, instead of surfacing later as a
program that will not load.

A coalesced program needs nothing extra: both backends are registered and the program records
which parts go where. Torch-TensorRT is only needed to export, not to run.

.. note::

    ``torch_tensorrt.load(path, format="executorch")`` still works but is deprecated. It
    copies CUDA inputs to the CPU and supports embedded weights only. Use the Runtime API
    above instead. A device-resident program requires it.

----

C++ Inference
--------------

The wheels ship a prebuilt delegate and a CMake package, so a C++ application links them
without building anything:

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

* ``kernels_optimized`` supplies the ``et_copy`` operators that move data across the method
  boundary.
* ``backend_cuda`` registers the device allocator those copies use, so even a TensorRT-only
  program needs it. Without it the program loads and then the first instruction fails with
  ``_h2d_copy: no device allocator registered``.
* ``extension_cuda`` provides ``CallerStreamGuard``, used below to choose the CUDA stream.

The two packages live in two distributions, so point CMake at both. ExecuTorch is a
namespace package, so take its path from the distribution metadata, not from ``__file__``:

.. code-block:: bash

    cmake -DCMAKE_PREFIX_PATH="$(TORCH_TENSORRT_SKIP_DELEGATE_REGISTRATION=1 python -c 'import importlib.metadata as m, torch_tensorrt_executorch_runtime as r, pathlib; print(str(pathlib.Path(str(m.distribution("executorch").locate_file("executorch"))) / "share" / "cmake") + ";" + str(pathlib.Path(r.__file__).parent))')" ...

The environment variable keeps that a path query, so the delegate is not loaded just to
print a path.

CMake 3.28 or newer is required. Older versions write the ``$ORIGIN`` token in a runtime
search path incorrectly, so ``backend_cuda`` rejects them. CMake is not in these wheels, and
a freshly imaged Jetson has none.

There is no header to include. The delegate registers itself when its library loads, and the
rest is the ordinary ExecuTorch C++ API:

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

Linking also records the wheel's library directory in your binary, so the application finds
the delegate with no library path set. That suits a local build and not anything you
redistribute. To turn it off, set both of these before ``find_package`` and ship the
delegate yourself:

.. code-block:: cmake

    set(EXECUTORCH_BACKEND_TENSORRT_EMBED_RUNPATH OFF)
    set(CMAKE_SKIP_BUILD_RPATH ON)

You do not need to find TensorRT. The delegate locates it from the sibling wheel on its own,
and your application talks to the delegate rather than to TensorRT.

Building the delegate from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The source ships inside ``libtorchtrt.tar.gz`` as
``torch_tensorrt/src/torch_tensorrt/executorch/``. Turn on ExecuTorch's CUDA backend and the
extensions a ``Module`` app needs, add the delegate beside ExecuTorch, and link its target:

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

Building it needs CUDA Toolkit 12.5 or newer. Keep ``libextension_cuda`` shared rather than
static, so every delegate in the process reads the same caller-stream state.

``examples/executorch_reference_runner/`` is a complete C++ example. ``libtorchtrt.tar.gz``
also carries it prebuilt as ``torch_tensorrt/bin/example_executorch_runner``.

----

Runtime Performance
--------------------

Choosing the CUDA stream
^^^^^^^^^^^^^^^^^^^^^^^^^

With no stream chosen, both delegates use ``cudaStreamPerThread``, the calling thread's
default stream, so a coalesced program run from one thread is already ordered correctly.

To put every delegate on a stream of your own, scope a guard over the whole execution:

.. code-block:: cpp

    #include <executorch/extension/cuda/caller_stream.h>

    using namespace executorch::extension;

    cuda::CallerStreamGuard guard(stream);
    module.forward(input);

* One guard reaches every delegate, because they share one ``libextension_cuda``.
* The stream must be on the engine's device.
* The CUDA backend refuses a caller stream for a method that uses its own CUDA graphs.
* **Synchronize your stream before reading outputs.** The TensorRT delegate always waits for
  its work, but ExecuTorch's CUDA backend can return once the work is only queued, so
  ``execute()`` on a coalesced program can return early.
  :ref:`Running a coalesced .pte <executorch_single_stream>` has the same rule for a decode
  loop.

That stream can be a green context, which holds a fixed number of streaming multiprocessors,
so the model stays inside that partition and the rest of the GPU is free. Create it with
``cuGreenCtxStreamCreate`` and scope the same guard over it. The limit rides the stream, so
the green context does not have to be made current.

The reference runner has this built in. Pass the SM count:

.. code-block:: bash

    example_executorch_runner --model_path=coalesced.pte --green_context_sms=8

If a green context cannot be created it fails with a distinct status instead of falling back,
so a passing run really used one. ``0``, the default, uses an ordinary stream.

CUDA graph replay
^^^^^^^^^^^^^^^^^^

Replay is off by default. With it on, the delegate records an engine's kernel launches once
and then replays the whole engine with a single launch. It helps fixed-shape engines that
launch many short kernels, where CPU launch work dominates each call.

.. code-block:: python

    torch_tensorrt.save(
        trt_gm,
        "model.pte",
        output_format="executorch",
        arg_inputs=example_inputs,
        retrace=False,
        use_cuda_graphs=True,
    )

``save()`` and ``torch_tensorrt.executorch.export()`` both take ``use_cuda_graphs=True``,
``False``, or ``None`` for no baked choice. It applies to every method's TensorRT delegates
and is read when an engine loads.

It is not free. Each engine keeps a device buffer per input and output, and every replayed
call copies each one in and out. With few kernels, or large inputs and outputs, those copies
can cost more than the launches they save.

Some engines never replay and still pay the copies: changing shapes, aliased outputs, the
shared activation scratch below, a green-context stream, GPUs without stream-ordered memory,
and drivers older than CUDA 12.5. The delegate logs the reason once.

.. warning::

    While a recording runs, no other thread may create or destroy a TensorRT engine or
    execution context, or call a whole-device sync such as ``cudaDeviceSynchronize`` or
    ``torch.cuda.synchronize()``. Every input shape change starts another recording, so
    loading everything up front does not end the risk. Leave replay off when other threads
    may do any of that.

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

Every TensorRT execution context holds its own activation scratch for as long as it lives, so
a model split into many small engines can run out of device memory on the engine count alone.
This option backs all of a device's contexts from one buffer, grown to the largest
requirement seen so far:

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

Set it before loading the methods that should use the pool. You save the sum of the separate
requirements less the largest one. ``Error::NotFound`` means the delegate is not linked in.

The cost is parallelism. The backend holds a per-device lock from claiming the buffer through
the enqueue, so two calls on one device are serialized at submission. The pool never shrinks,
and a device that has run a pooled engine must not be reset with ``cudaDeviceReset()``.

Shared engines
^^^^^^^^^^^^^^^

Loading the same program twice in one process, say once per robot arm, used to hold its
weights in device memory twice. Handles built from the same engine bytes, on the same device
and with the same weight streaming request, now share one engine. Each still gets its own
execution context, buffers and lock.

This is on by default. On an 8 GB Jetson Orin Nano, loading one robot policy twice grew
reported GPU memory by 806 MiB without sharing and by 122 MiB with it. The match is found by
hashing the engine bytes on every load, roughly 0.15 s per GB on a Jetson AGX Thor. To keep a
module's engines private, pass the ``use_shared_engines`` load option as ``false``, the same
way as ``use_cuda_graphs`` above.

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
