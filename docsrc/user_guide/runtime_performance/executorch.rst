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

    The delegate runs on Linux (x86-64 and aarch64), and on Windows x86-64 (not in 2.15). The
    prebuilt delegate needs CUDA 13. On Windows, the delegate loads the CUDA runtime from a CUDA 13
    toolkit install, so make sure ``CUDA_PATH`` or ``CUDA_HOME`` points at one.

    By default, TensorRT builds each engine for the operating system and the GPU it is exported on,
    so export your programs on the kind of machine that will run them. On Windows, ExecuTorch's
    CUDA backend cannot export a program, and its Python runtime there does not include that
    backend, so a coalesced program (see below) is a Linux setup.

----

Installation
-------------

Use a fresh virtual environment, and pick the command for the Torch-TensorRT you want. Swap
``cu132`` for ``cu134`` on CUDA 13.4.

Torch-TensorRT 2.15
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    pip install --upgrade "torch-tensorrt[executorch]==2.15.*" "torch-tensorrt-executorch-runtime==0.2.*" \
      --index-url https://download.pytorch.org/whl/cu132 \
      --extra-index-url https://download.pytorch.org/whl/test/cu132 \
      --extra-index-url https://download.pytorch.org/whl/nightly/cu132 \
      --extra-index-url https://pypi.org/simple

* No release brings the delegate through its extra yet, so the command names it. It comes from
  the test channel, where the release was first published as a release candidate.
* The delegate pins a development build of ExecuTorch, which only the nightly channel carries, and
  only for a limited time. Once that build is gone from the nightly channel, this command fails.
* So far the 2.15 delegate is published for Linux only.

Nightly builds
^^^^^^^^^^^^^^^

To pick up a change before it ships, or to report a problem against ``main``. Here the extra
brings the delegate:

.. code-block:: bash

    pip install --pre --upgrade "torch-tensorrt[executorch]" \
      --index-url https://download.pytorch.org/whl/nightly/cu132 \
      --extra-index-url https://pypi.org/simple

* ``--pre`` is required. Without it pip takes the stable release instead.
* ``--upgrade`` is required if a nightly is already installed, or pip keeps the old one and
  tells you nothing.
* PyPI stays in the list because TensorRT is not published on the nightly channel.
* On Windows, install PyTorch first, with
  ``pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu132``.
  Then add ``--extra-index-url https://download.pytorch.org/whl/nightly/cpu`` to the command.
  ExecuTorch's Windows nightly needs torchao, and only that index carries its Windows build.
  With PyTorch already installed, pip does not take the CPU build of PyTorch from that index.
* Keep PyTorch, ExecuTorch and Torch-TensorRT on the same channel.

What the commands install
^^^^^^^^^^^^^^^^^^^^^^^^^^

* ExecuTorch, the CUDA build, from the ``executorch`` extra. That is all you need to export.
* ``torch-tensorrt-executorch-runtime``, one shared library holding the TensorRT delegate,
  which you need to run a ``.pte``. It is the one built together with your Torch-TensorRT wheel,
  for the same CUDA version. It requires the exact ExecuTorch build it was compiled against, so
  you get that one rather than the newest.

The extra brings the delegate for a nightly wheel whose ExecuTorch pin is a development build,
as on main today. A release does not bring it yet, because the delegate is not published on the
stable channel, and neither does a wheel you build yourself. If it is missing when you load a
``.pte``, the error message says how to get the one that matches your wheel.

The plain form, ``pip install "torch-tensorrt[executorch]"`` with
``--index-url https://download.pytorch.org/whl/cu132 --extra-index-url https://pypi.org/simple``,
is enough once a release brings the delegate through its extra. The PyTorch channel is needed in
every form because PyPI has only a CPU build of ExecuTorch. PyPI stays in the list for TensorRT
and the other dependencies.

The delegate is built for the standard Linux wheels (x86-64 and aarch64), and for Windows x86-64
in builds newer than 2.15. On aarch64 it needs glibc 2.35 or newer, the same as TensorRT itself.
The TensorRT-RTX and Python-only builds have no delegate. To export a coalesced program you also
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

.. note::

    In Torch-TensorRT 2.15, ``save()`` re-exports the graph by default, so on 2.15 add
    ``retrace=False`` to every ``save()`` call on this page.

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
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        ),
    )

* **The two skip flags work only with** ``enable_non_cpu_memory_planning``, which is on by
  default. With either flag set, turning it off makes the export raise an error.
* **Do not let the program allocate its inputs:** pass ``alloc_graph_input=False``. Otherwise
  the program keeps its own input buffer, and handing it a GPU tensor fails with error ``0x10``
  (``NotSupported``). The log then says ``Error copying tensor data at input 0``.
* **The outputs can stay in the program's own GPU memory**, as above. To receive them in memory
  you own instead, also pass ``alloc_graph_output=False``. Then each output needs a GPU buffer
  before the call: in C++ you set it with ``Module::set_output``, and the Python runtime allocates
  one for you.

Pass CUDA tensors to a program exported this way. A CPU tensor costs a copy on every call when the
first delegate to use that input is TensorRT, and fails when it is the CUDA backend. In builds
newer than 2.15, the Python runtime refuses it in both cases.

Keep the engines in a separate file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Not in 2.15.**

By default the engines are stored inside the ``.pte``, so the file is as large as the model. With
``external_engine_data``, they go to an ExecuTorch data file (``.ptd``) next to the ``.pte``
instead. Give the name without the extension:

.. code-block:: python

    torch_tensorrt.save(
        trt_gm,
        "small.pte",
        output_format="executorch",
        external_engine_data="engines",
    )

This writes a small ``small.pte`` and an ``engines.ptd`` next to it. Pass the ``.ptd`` when you
load the program. Without it, loading the method fails. The Python runtime takes one data file per
program, so for a coalesced program pass ``external_engine_data="aoti_cuda_blob"``: the engines
then go into the CUDA backend's file.

Treat a ``.ptd`` file as trusted input, the same as a ``.pte``. The engines in it are GPU code.

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
building anything. The steps below are for Linux.

.. code-block:: cmake

    find_package(executorch REQUIRED COMPONENTS backend_cuda kernels_optimized)
    find_package(executorch_backend_tensorrt REQUIRED)

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
location comes from its package metadata, because ``executorch.__file__`` is not set:

.. code-block:: bash

    ET_CMAKE=$(python -c 'import importlib.metadata as m, pathlib; print(pathlib.Path(str(m.distribution("executorch").locate_file("executorch"))) / "share" / "cmake")')
    DELEGATE=$(TORCH_TENSORRT_SKIP_DELEGATE_REGISTRATION=1 python -c 'import pathlib, torch_tensorrt_executorch_runtime as r; print(pathlib.Path(r.__file__).parent)')
    cmake -DCMAKE_PREFIX_PATH="${ET_CMAKE};${DELEGATE}" ...

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

In Python, pass it to ``load_program``. **Not in 2.15:** the ExecuTorch build that Torch-TensorRT
2.15 installs has no ``backend_options`` argument.

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

In Python, also **not in 2.15**:

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
``libtorchtrt.tar.gz`` and in the Torch-TensorRT wheels for Linux. It has no CUDA backend, and of
the CPU operators only the copies at the edges of the program, so it runs only programs that
TensorRT takes in full, not coalesced ones.

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

Replay is not free. Each engine keeps its own GPU buffer for every input and output, and every
replayed call copies the inputs in and the outputs out. With few kernels, or with large inputs
and outputs, those copies can cost more than the launches they save.

Some calls never replay:

* An engine whose input shapes change on every call never replays, and it still pays the copies.
* An engine with aliased outputs, a GPU without stream-ordered memory, and a driver older than
  CUDA 12.5 never replay. They take the ordinary path, without the extra copies.
* A call on a green context stream, or on a stream of another CUDA context, takes the ordinary
  path for that call.

An engine on the shared activation scratch below also replays. It records again whenever the
shared buffer grows. In Torch-TensorRT 2.15, an engine on the shared scratch
never replays.

.. warning::

    While a graph is being recorded, no other thread may create or destroy a TensorRT engine or
    execution context outside this delegate, or call a whole device sync such as
    ``cudaDeviceSynchronize()`` or ``torch.cuda.synchronize()``. Loading a program with
    ExecuTorch's CUDA backend can make such a sync too. Recording happens again after every shape
    change, so loading everything first does not remove the risk. Leave replay off when other
    threads may do any of these things.

Shared activation scratch
^^^^^^^^^^^^^^^^^^^^^^^^^^

Every TensorRT execution context keeps its own scratch memory for intermediate results, for as
long as it lives. A model split into many small engines pays for that many times, and can run out
of GPU memory on the number of engines alone. With the ``use_shared_activation_scratch`` process
option, all contexts on a device share one buffer instead. The buffer grows to the largest size
any call has needed, so you save the sum of the separate sizes minus the largest one.

Set the option before you load the methods that should use the shared buffer. Methods loaded
before it keep their own scratch.

The cost is parallelism. Two calls that share the buffer never run at the same time on one
device: the delegate takes a per-device lock from claiming the buffer to queuing the work. The
buffer never shrinks. Do not call ``cudaDeviceReset()`` on a device that ran an engine with the
shared buffer. Recording a call into your own CUDA graph is not supported, with or without the
shared buffer.

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
From Python that needs a build newer than 2.15, as `Setting a runtime option`_ explains, so Python loads
with Torch-TensorRT 2.15 always share.

Weight streaming
^^^^^^^^^^^^^^^^^

An engine compiled with ``enable_weight_streaming=True`` can keep part of its weights in CPU
memory and stream them to the GPU when needed. The budget is the number of bytes of weights that
stay on the GPU. Save one with ``weight_streaming_budget_per_engine``, or set one at run time
with the ``weight_streaming_budget`` load option, which takes the number of bytes as a decimal
string, or as an int below 2 GiB. From Python, that load option needs a build newer than 2.15.
The budget applies to each engine separately, so a program with N engines can keep up to N times
the budget on the GPU. With no budget, TensorRT chooses one from the free memory of the device.

Optimization profiles
^^^^^^^^^^^^^^^^^^^^^^

**Not in 2.15.**

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
     - Linux (x86-64 and aarch64) and Windows x86-64 (not in 2.15), with CUDA 13
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
* ``export_multi_profile.py`` exports one engine with two optimization profiles (not in 2.15).

``examples/executorch_reference_runner/`` loads and runs a ``.pte`` from Python and from C++,
including optimization profiles (not in 2.15) and the green context option.
