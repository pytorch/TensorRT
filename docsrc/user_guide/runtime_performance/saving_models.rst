.. _saving_models:

Saving models compiled with Torch-TensorRT
==========================================

.. note::

    ``torch.compile(backend="torch_tensorrt")`` uses **JIT compilation** — engines are
    built on first call and cannot be saved directly. To serialize a TRT engine to disk,
    use the AOT path: ``torch_tensorrt.compile(model, ir="dynamo", ...)`` followed by
    ``torch_tensorrt.save()``. See `Saving torch.compile models`_ below for details.

Saving models compiled with Torch-TensorRT can be done using `torch_tensorrt.save` API.

Dynamo IR
-------------

The output type of `ir=dynamo` compilation of Torch-TensorRT is `torch.fx.GraphModule` object by default.
We can save this object in either `TorchScript` (`torch.jit.ScriptModule`), `ExportedProgram` (`torch.export.ExportedProgram`), `PT2` or ExecuTorch formats by
specifying the `output_format` flag. Here are the options `output_format` will accept

* `exported_program` : This is the default. We perform transformations on the graphmodule first and use `torch.export.save` to save the module.
* `torchscript` : We trace the graphmodule via `torch.jit.trace` and save it via `torch.jit.save`.
* `PT2 Format` : This is a next generation runtime for PyTorch models, allowing them to run in Python and in C++
* `executorch` : We lower the graphmodule to an ExecuTorch ``.pte`` program, with the TensorRT engines stored inside it. Requires the ``executorch`` extra; see :ref:`executorch_deployment`.

a) ExportedProgram
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here's an example usage

.. code-block:: python

    import torch
    import torch_tensorrt

    model = MyModel().eval().cuda()
    inputs = [torch.randn((1, 3, 224, 224)).cuda()]
    # trt_ep is a torch.fx.GraphModule object
    trt_gm = torch_tensorrt.compile(model, ir="dynamo", arg_inputs=inputs)
    torch_tensorrt.save(trt_gm, "trt.ep", arg_inputs=inputs)

    # Later, you can load it and run inference
    model = torch.export.load("trt.ep").module()
    model(*inputs)


Saving Models with Dynamic Shapes
""""""""""""""""""""""""""""""""""

When saving models compiled with dynamic shapes, you have two methods to preserve
the dynamic shape specifications:

**Method 1: Using torch.export.Dim (explicit)**

Provide explicit ``dynamic_shapes`` parameter following torch.export's pattern:

.. code-block:: python

    import torch
    import torch_tensorrt

    model = MyModel().eval().cuda()
    example_input = torch.randn((2, 3, 224, 224)).cuda()

    # Define dynamic batch dimension
    dyn_batch = torch.export.Dim("batch", min=1, max=32)
    dynamic_shapes = {"x": {0: dyn_batch}}

    # Export with dynamic shapes
    exp_program = torch.export.export(
        model, (example_input,),
        dynamic_shapes=dynamic_shapes,
        strict=False
    )

    # Compile with dynamic input specifications
    trt_gm = torch_tensorrt.dynamo.compile(
        exp_program,
        arg_inputs=[torch_tensorrt.Input(
            min_shape=(1, 3, 224, 224),
            opt_shape=(8, 3, 224, 224),
            max_shape=(32, 3, 224, 224),
        )]
    )

    # Save with dynamic_shapes to preserve dynamic behavior
    torch_tensorrt.save(
        trt_gm,
        "trt_dynamic.ep",
        arg_inputs=[example_input],
        dynamic_shapes=dynamic_shapes  # Same as used during export
    )

    # Load and use with different batch sizes
    loaded_model = torch_tensorrt.load("trt_dynamic.ep").module()
    output_bs4 = loaded_model(torch.randn(4, 3, 224, 224).cuda())
    output_bs16 = loaded_model(torch.randn(16, 3, 224, 224).cuda())

**Method 2: Using torch_tensorrt.Input**

Pass ``torch_tensorrt.Input`` objects with min/opt/max shapes directly, and the
dynamic shapes will be inferred automatically:

.. code-block:: python

    import torch
    import torch_tensorrt

    model = MyModel().eval().cuda()

    # Define Input with dynamic shapes
    inputs = [
        torch_tensorrt.Input(
            min_shape=(1, 3, 224, 224),
            opt_shape=(8, 3, 224, 224),
            max_shape=(32, 3, 224, 224),
            dtype=torch.float32,
            name="x"  # Optional: provides better dimension naming
        )
    ]

    # Compile with Torch-TensorRT
    trt_gm = torch_tensorrt.compile(model, ir="dynamo", arg_inputs=inputs)

    # Save with Input objects - dynamic_shapes inferred automatically!
    torch_tensorrt.save(
        trt_gm,
        "trt_dynamic.ep",
        arg_inputs=inputs  # Dynamic shapes inferred from Input objects
    )

    # Load and use with different batch sizes
    loaded_model = torch_tensorrt.load("trt_dynamic.ep").module()
    output_bs4 = loaded_model(torch.randn(4, 3, 224, 224).cuda())
    output_bs16 = loaded_model(torch.randn(16, 3, 224, 224).cuda())


b) Torchscript
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    import torch_tensorrt

    model = MyModel().eval().cuda()
    inputs = [torch.randn((1, 3, 224, 224)).cuda()]
    # trt_gm is a torch.fx.GraphModule object
    trt_gm = torch_tensorrt.compile(model, ir="dynamo", arg_inputs=inputs)
    torch_tensorrt.save(trt_gm, "trt.ts", output_format="torchscript", arg_inputs=inputs)

    # Later, you can load it and run inference
    model = torch.jit.load("trt.ts").cuda()
    model(*inputs)


Torchscript IR
-------------

In Torch-TensorRT 1.X versions, the primary way to compile and run inference with Torch-TensorRT is using Torchscript IR.
For `ir=ts`, this behavior stays the same in 2.X versions as well.

.. code-block:: python

  import torch
  import torch_tensorrt

  model = MyModel().eval().cuda()
  inputs = [torch.randn((1, 3, 224, 224)).cuda()]
  trt_ts = torch_tensorrt.compile(model, ir="ts", arg_inputs=inputs) # Output is a ScriptModule object
  torch.jit.save(trt_ts, "trt_model.ts")

  # Later, you can load it and run inference
  model = torch.jit.load("trt_model.ts").cuda()
  model(*inputs)


Loading the models
--------------------

We can load torchscript or exported_program models using `torch.jit.load` and `torch.export.load` APIs from PyTorch directly.
Alternatively, we provide a light wrapper `torch_tensorrt.load(file_path)` which can load either of the above model types.

Here's an example usage

.. code-block:: python

    import torch
    import torch_tensorrt

    # file_path can be trt.ep or trt.ts file obtained via saving the model (refer to the above section)
    inputs = [torch.randn((1, 3, 224, 224)).cuda()]
    model = torch_tensorrt.load(<file_path>).module()
    model(*inputs)

b) PT2 Format (AOTInductor)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PT2 packages the model using `AOTInductor <https://docs.pytorch.org/docs/main/torch.compiler_aot_inductor.html>`_,
which compiles non-TRT subgraphs into native CUDA kernels. The resulting ``.pt2`` file
can be loaded in Python or C++ without a Torch-TensorRT runtime dependency.

.. code-block:: python

    import torch
    import torch_tensorrt

    model = MyModel().eval().cuda()
    inputs = [torch.randn((1, 3, 224, 224)).cuda()]
    trt_gm = torch_tensorrt.compile(model, ir="dynamo", arg_inputs=inputs)
    torch_tensorrt.save(trt_gm, "trt.pt2", arg_inputs=inputs,
                        output_format="aot_inductor", retrace=True)

    # Load without Torch-TensorRT at inference time
    model = torch._inductor.aoti_load_package("trt.pt2")
    model(*inputs)

For dynamic shapes, C++ deployment, and a full comparison with the ExportedProgram format,
see :ref:`aot_inductor`.


.. _executorch_save:

c) ExecuTorch (.pte)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. seealso::

    :ref:`executorch_deployment` is the end to end guide. It shows how to install the packages,
    run the saved program from Python and from C++, and tune it at run time. This section covers
    the saving step.

``output_format="executorch"`` writes an ExecuTorch ``.pte`` program. The TensorRT engines are
stored in it, and the TensorRT delegate runs them at run time. There are two ways to write one:

* ``torch_tensorrt.save()`` does everything in one call and writes the file. Start here.
* ``torch_tensorrt.executorch.export()`` stops before the file is written, and returns
  ExecuTorch's Edge program, the form the program has before it becomes a file. Use it to inspect
  or change the program first, or to put several methods in one file.

Default: ``save()``
"""""""""""""""""""

.. code-block:: python

    import torch
    import torch_tensorrt

    model = MyModel().eval().cuda()
    inputs = [torch.randn((1, 3, 224, 224)).cuda()]
    trt_gm = torch_tensorrt.compile(model, ir="dynamo", arg_inputs=inputs)
    torch_tensorrt.save(trt_gm, "trt.pte", output_format="executorch")

``save()`` writes the ``.pte``, and any ``.ptd`` data files next to it. It writes one method,
``forward``, plus any constant methods you pass.

.. note::

    Saving the graph as it is by default is nightly only. In Torch-TensorRT 2.15, ``save()``
    re-exports the graph by default, so on 2.15 add ``retrace=False`` to every ``save()`` call shown
    here.

These keyword arguments of ``save()`` apply only to this format. With any other format they log a
warning and are ignored. With this format, an unknown keyword raises a ``TypeError``.

* ``partitioners``: more partitioners, run after TensorRT's. For example, a ``CudaPartitioner``
  sends the operators TensorRT rejects to ExecuTorch's CUDA backend. See
  :ref:`executorch_deployment` for that recipe.
* ``compile_specs``: extra compile specs for TensorRT's partitioner. You rarely need it: the
  options below set their own specs, and passing one of those specs here raises a ``ValueError``.
* ``backend_config``: an ``ExecutorchBackendConfig`` for ExecuTorch's ``to_executorch()``, for
  example to keep inputs and outputs on the GPU.
* ``constant_methods``: extra methods that return a constant, such as
  ``{"get_max_seq_len": 2048}`` for an LLM runner.
* ``transform_passes``: extra Edge transform passes to run before lowering.
* ``compile_config``: an ``EdgeCompileConfig``. The default turns off Edge IR validation
  (``_check_ir_validity=False``), because the TensorRT engine placeholder does not pass it. A
  config you pass is used as it is, so set ``_check_ir_validity=False`` on it yourself.
* ``use_cuda_graphs``: save CUDA graph replay on or off for every engine.
* ``weight_streaming_budget_per_engine``: how many bytes of each engine's weights stay on the GPU,
  for an engine compiled with ``enable_weight_streaming=True``.
* ``external_engine_data``: a file name, without ``.ptd``, to store the engines in instead of the
  ``.pte``. **Nightly only.**
* ``zero_copy_kv``: let an engine update its KV cache in place. See
  :ref:`Zero-copy KV cache <executorch_zero_copy_kv>`.
* ``generate_etrecord``: also write an ETRecord, a debug record for the ExecuTorch Developer Tools
  ``Inspector``, as ``<name>_etrecord.bin``. It does not work yet for a program with TensorRT
  engines: writing the record fails with a ``SerializeError``.

:ref:`executorch_deployment` explains the run time side of ``use_cuda_graphs``,
``weight_streaming_budget_per_engine`` and ``external_engine_data``.

For example:

.. code-block:: python

    from executorch.exir import EdgeCompileConfig

    torch_tensorrt.save(
        trt_gm,
        "trt.pte",
        output_format="executorch",
        constant_methods={"get_max_seq_len": 2048},
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )

Advanced: ``export()``
""""""""""""""""""""""

``torch_tensorrt.executorch.export()`` takes a TensorRT-compiled ``GraphModule``, an
``ExportedProgram`` that already holds engines, or a dict that maps method names to such
programs. It runs TensorRT's partitioner first, then yours in order, and returns ExecuTorch's
``EdgeProgramManager``. You then finalize and write the program yourself. Here
``extra_partitioners``, ``passes``, ``edge_config`` and ``backend_config`` stand for your own
values of the options above:

.. code-block:: python

    import torch_tensorrt.executorch

    edge = torch_tensorrt.executorch.export(
        trt_gm,
        partitioners=extra_partitioners,
        transform_passes=passes,
        compile_config=edge_config,
        constant_methods={"get_vocab_size": 256},
    )

    # Inspect edge.exported_program() or apply more Edge transforms here.
    program = edge.to_executorch(config=backend_config)
    with open("trt.pte", "wb") as output:
        program.write_to_file(output)
    program.write_tensor_data_to_file(".")

It takes the same options as ``save()``, except ``backend_config``, which goes to
``to_executorch()``. ``partitioners``, ``compile_specs`` and ``transform_passes`` can also be a dict
keyed by method name.

**Several methods in one .pte**

Pass a dict of method name to ``ExportedProgram`` to keep separate entry points, such as prefill
and decode, in one program. Give each method its own partitioner instances. The CUDA backend
reads its method name from its compile specs, so it needs one ``CudaPartitioner`` per method,
each with that method's name:

.. code-block:: python

    from executorch.backends.cuda.cuda_backend import CudaBackend
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner

    edge = torch_tensorrt.executorch.export(
        {"prefill": prefill_program, "decode": decode_program},
        partitioners={
            "prefill": [
                CudaPartitioner([CudaBackend.generate_method_name_compile_spec("prefill")])
            ],
            "decode": [
                CudaPartitioner([CudaBackend.generate_method_name_compile_spec("decode")])
            ],
        },
    )

``export()`` refuses one partitioner instance shared by methods when its specs name a method. The
methods share no state: two methods do not share a buffer unless you plan it that way.

.. warning::

    **The returned Edge program shares its weights with the programs you passed in.** Do not
    change a weight in place in a transform pass, for example with ``weight.data.mul_(scale)``,
    and do not change a source program after ``export()``. Either change shows up in the other
    program too, with no error. Build a new tensor instead.

    Engines are not shared. ``export()`` copies each engine into the Edge program once, so a large
    engine is in host memory twice while both programs exist.

.. _executorch_zero_copy_kv:

**Zero-copy KV cache**

An LLM's TensorRT engine can update its KV cache in place, through an input and an output that
alias the same buffer. By default ExecuTorch does not know that. It hands the engine a copy of
the cache, and then copies the engine's result back: two cache-sized copies for every token.
``zero_copy_kv=True`` removes both, so the engine writes the cache directly.

With ``save()``, one flag does it:

.. code-block:: python

    torch_tensorrt.save(trt_gm, "decode.pte", output_format="executorch", zero_copy_kv=True)

With ``export()``, it takes two more calls, one at each end of the Edge program:

.. code-block:: python

    from torch_tensorrt.executorch import check_zero_copy_kv, export, zero_copy_backend_config

    edge = export({"prefill": prefill_program, "decode": decode_program}, zero_copy_kv=True)
    program = edge.to_executorch(zero_copy_backend_config())
    check_zero_copy_kv(program)

.. warning::

    **Nothing makes you call both.** If you finalize with a config that did not come from
    ``zero_copy_backend_config``, ``to_executorch()`` still writes a program, but the engine then
    writes a copy that is thrown away, so the cache never updates. That is wrong output, not a
    crash. ``check_zero_copy_kv`` refuses such a program before you write the file.
    ``save(..., zero_copy_kv=True)`` does both steps for you.

* ``zero_copy_backend_config(config)`` starts from your own config, if you pass one, and keeps
  every field of it. It refuses ``skip_h2d_for_method_inputs``. The caches have to be planned in
  GPU memory, so ``enable_non_cpu_memory_planning=False`` makes finalizing fail.
* If you build a new config from the returned one, for example with ``dataclasses.replace``,
  call ``zero_copy_backend_config`` again on the new one.
* It is opt-in because the ``.pte`` needs a runtime that knows about it. A runner built before
  this feature would break on it.
* To share one cache between two methods, such as prefill and decode, both methods' buffers have
  to be planned at the same place. Pass your own ``memory_planning_pass`` for that.
  ``zero_copy_backend_config`` keeps it. That planner must record which device each memory area
  is on, which ExecuTorch's ``apply_algo`` does when it runs with
  ``enable_non_cpu_memory_planning=True``. Otherwise ``check_zero_copy_kv`` refuses the program.

The :ref:`API reference <torch_tensorrt_executorch_py>` describes every refusal in detail.


Saving torch.compile models
-----------------------------

``torch.compile(backend="torch_tensorrt")`` is a **JIT** path — TRT engines are built
lazily on the first call and live in memory only. There is no direct way to call
``torch_tensorrt.save()`` on a ``torch.compile``-compiled model.

**Workaround: switch to the AOT path**

Replace ``torch.compile`` with ``torch_tensorrt.compile(ir="dynamo")`` to get a
serializable ``torch.fx.GraphModule``:

.. code-block:: python

    import torch
    import torch_tensorrt

    model = MyModel().eval().cuda()
    inputs = [torch.randn((1, 3, 224, 224)).cuda()]

    # JIT path — NOT serializable
    # jit_model = torch.compile(model, backend="torch_tensorrt", ...)

    # AOT path — produces a serializable GraphModule
    trt_gm = torch_tensorrt.compile(
        model,
        ir="dynamo",
        arg_inputs=inputs,
    )

    # Save to disk
    torch_tensorrt.save(trt_gm, "model.ep", arg_inputs=inputs)

    # Reload and run — no recompilation needed
    loaded = torch_tensorrt.load("model.ep").module()
    output = loaded(*inputs)

The ``ir="dynamo"`` path supports all the same compilation options as
``torch.compile(backend="torch_tensorrt")``. The key differences are:

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Feature
     - ``torch.compile`` (JIT)
     - ``ir="dynamo"`` (AOT)
   * - Compilation timing
     - On first call
     - Explicit compile step
   * - Auto-recompile on shape change
     - Yes
     - No (fixed shapes unless dynamic Input used)
   * - Serializable to disk
     - No
     - Yes (``torch_tensorrt.save``)
   * - C++ deployment
     - No
     - Yes (via ExportedProgram or PT2 format)
