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
We can save this object in either `TorchScript` (`torch.jit.ScriptModule`), `ExportedProgram` (`torch.export.ExportedProgram`) or `PT2` formats by
specifying the `output_format` flag. Here are the options `output_format` will accept

* `exported_program` : This is the default. We perform transformations on the graphmodule first and use `torch.export.save` to save the module.
* `torchscript` : We trace the graphmodule via `torch.jit.trace` and save it via `torch.jit.save`.
* `PT2 Format` : This is a next generation runtime for PyTorch models, allowing them to run in Python and in C++
* `executorch` : We lower the graphmodule to an ExecuTorch ``.pte`` program, delegating the TensorRT engines to the ExecuTorch backend. Linux-only; requires the ``executorch`` package.

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

The ``executorch`` output format lowers the compiled module to an ExecuTorch
``.pte`` program, delegating the TensorRT engines to the Torch-TensorRT ExecuTorch
backend. It requires the ``executorch`` package (``pip install
"torch_tensorrt[executorch]"``) and is Linux-only.

There are two ways to produce a ``.pte``, and they suit different needs:

* **Use** ``torch_tensorrt.save()``. This is the default, and the right choice
  whenever a ``.pte`` file is all that is needed. It runs the whole pipeline in one
  call and writes the file.
* **Use** ``torch_tensorrt.executorch.export()`` only when the program has to be
  changed before it is written to disk. It stops at the Edge program, hands it back
  for inspection or customization, and leaves serialization to the caller.

Start with ``save()``. Reach for ``export()`` when something below is required.

Default: ``save()``
"""""""""""""""""""

.. code-block:: python

    import torch
    import torch_tensorrt
    import torch_tensorrt.executorch

    model = MyModel().eval().cuda()
    inputs = [torch.randn((1, 3, 224, 224)).cuda()]
    trt_gm = torch_tensorrt.compile(model, ir="dynamo", arg_inputs=inputs)
    torch_tensorrt.save(
        trt_gm, "trt.pte", output_format="executorch",
        retrace=False, arg_inputs=inputs,
    )

``save`` writes both the ``.pte`` and any external ``.ptd`` tensor-data files.

Advanced: ``export()``
""""""""""""""""""""""

Stop at the standard Edge program boundary when the program needs work before it is
serialized: inspecting the delegated graph, adding metadata, controlling final memory
planning, or carrying more than one method. The caller then serializes it:

.. code-block:: python

    edge = torch_tensorrt.executorch.export(
        trt_gm,
        arg_inputs=inputs,
        retrace=False,
        partitioners=extra_partitioners,
        transform_passes=passes,
        compile_config=edge_config,
        constant_methods={"get_vocab_size": 256},
    )

    # Inspect edge.exported_program() or apply additional Edge transforms here.
    program = edge.to_executorch(config=backend_config)
    with open("trt.pte", "wb") as output:
        program.write_to_file(output)
    program.write_tensor_data_to_file(".")

``executorch.export`` accepts a TensorRT-compiled ``GraphModule``, an
engine-bearing ``ExportedProgram``, or a mapping of independently exported
methods. It always applies the TensorRT partitioner first, followed by caller
``partitioners`` in order, and returns ExecuTorch's native
``EdgeProgramManager``. Use ``save`` for the one-shot path where Torch-TensorRT
manages Edge lowering, finalization, and persistence.

**Several methods in one .pte**

Pass a mapping of method name to ``ExportedProgram`` to keep independent entry
points, such as a separate prefill and decode, in one program. Give each method
its own partitioner instances when a partitioner carries method-specific state: a
partitioner holds its compile specs from construction, so one instance whose specs
name a method would tag every method sharing it with that same name. Reusing such
an instance across methods is rejected.

Sharing an instance whose specs name no method is not rejected, because some
backends are built to share one. A backend that instead reads its own method name
from its specs, such as the CUDA backend below, raises
``Could not find method name in compile specs`` during lowering, so give it one
instance per method carrying that method's name spec.

.. code-block:: python

    from executorch.backends.cuda.cuda_backend import CudaBackend
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner

    edge = torch_tensorrt.executorch.export(
        {"prefill": prefill_program, "decode": decode_program},
        partitioners={
            "prefill": [
                CudaPartitioner(
                    [CudaBackend.generate_method_name_compile_spec("prefill")]
                )
            ],
            "decode": [
                CudaPartitioner(
                    [CudaBackend.generate_method_name_compile_spec("decode")]
                )
            ],
        },
    )

Each method is validated before any program is rewritten, so an error in one
method leaves the others untouched. A method mapping preserves independent entry
points but does not by itself give them shared mutable state.

.. warning::

    **The returned Edge program shares tensor storage with the programs you
    passed in.** Only structure is copied: the graph, the graph signature, the
    ``state_dict`` keys, and node metadata. Weights and every other tensor
    payload are shared by reference.

    TensorRT engines are not shared. Export never deep copies an engine object,
    since that would serialize and deserialize it, but every engine is decoded
    once into a byte buffer the returned program owns. Plan for the bytes of a
    multi-gigabyte engine to be resident twice while both programs are alive.

    Two consequences to plan for:

    * A transform pass must not modify a shared payload in place. An in-place
      edit such as ``weight.data.mul_(scale)`` changes the program you passed in,
      and every other Edge program exported from it. Build a new tensor and
      rebind it instead of mutating the existing one.
    * Modifying a source program after calling ``export`` is also visible in the
      Edge program. Finish preparing a program before exporting it.

    Neither case raises an error or a warning, so treat every shared payload as
    read-only.

.. _executorch_zero_copy_kv:

**Zero-copy KV cache**

When a TensorRT engine has aliased I/O -- a KV cache it updates through an
aliased binding -- running the engine over the cache already is the update.
ExecuTorch does not know that, so by default it pays for the update twice per
execution: it hands the delegate an ``_h2d_copy`` staging copy of the buffer
instead of the buffer itself, then copies the engine's aliased output back into
the buffer afterwards. For a KV cache both copies are cache-sized, per token.

``zero_copy_kv=True`` removes them, so the engine writes the caller's buffer
directly. Through the two-step ``export`` + ``to_executorch`` path it takes two
calls, one at each end of the Edge boundary:

.. code-block:: python

    from torch_tensorrt.executorch import export, zero_copy_backend_config

    edge = export(
        {"prefill": prefill_program, "decode": decode_program},
        partitioners={"prefill": [CudaPartitioner([])], "decode": [CudaPartitioner([])]},
        zero_copy_kv=True,
    )

    # zero_copy_backend_config composes onto your own config; every other field
    # (memory planning, passes) is preserved.
    program = edge.to_executorch(zero_copy_backend_config(backend_config))

It is opt-in rather than automatic because the resulting ``.pte`` needs a
runtime that understands a delegate whose aliased outputs are elided. Producing
one silently would break a runner built before this feature.

.. warning::

    **Both calls are required.** Exporting with ``zero_copy_kv=True`` and then
    finalizing without ``zero_copy_backend_config`` does not raise on its own:
    the engine writes a per-call staging copy that is discarded, and the cache
    never updates. For a KV cache that is wrong output, not a crash. Pass the
    finalized program to ``torch_tensorrt.executorch.check_zero_copy_kv``, which
    reads the graph back and refuses one whose caches are still staged, before
    writing the ``.pte``::

        program = edge.to_executorch(zero_copy_backend_config(backend_config))
        torch_tensorrt.executorch.check_zero_copy_kv(program)

    ``torch_tensorrt.save`` owns both ends and runs that check itself, so there
    is nothing to pair on that path.

``torch_tensorrt.save`` finalizes the program itself, so a single
``zero_copy_kv=True`` covers both steps:

.. code-block:: python

    torch_tensorrt.save(
        trt_gm, "decode.pte", output_format="executorch",
        arg_inputs=inputs, retrace=False,
        zero_copy_kv=True,
    )

It installs ``zero_copy_backend_config`` for you, so do not hand it one as
``backend_config`` as well: the pass would be installed twice and finalization
raises. The two entry points are alternatives, not a pair.

Two further responsibilities are the caller's, and neither raises:

* **One CUDA stream for every delegate**, if the ``.pte`` is coalesced -- and the
  synchronization it calls for, which zero-copy makes load-bearing. Getting this
  wrong is a race, not a deterministic error: it is intermittent and can surface
  as wrong results *or* as an illegal memory access. See
  :ref:`Running a coalesced .pte <executorch_single_stream>`.

* **Sharing one cache between methods.** Zero-copy is per method: it makes each
  method's engine write that method's buffer. Giving a prefill and a decode
  method *the same* cache is a memory-planning question -- their mutable buffers
  have to land at the same arena offsets -- which ExecuTorch's memory planner
  owns and which is deployment-specific. Supply your own
  ``memory_planning_pass`` for it; ``zero_copy_backend_config`` preserves it.

**Coalesced TensorRT + CUDA .pte**

To run the ops TensorRT does not take on ExecuTorch's CUDA (AOTInductor) backend
instead of leaving them non-delegated, pass a ``CudaPartitioner`` via
``partitioners=``. It is appended after the TensorRT partitioner, so TensorRT
claims what it can and the ``CudaPartitioner`` picks up the rest as a catch-all:

.. code-block:: python

    from executorch.backends.cuda.cuda_backend import CudaBackend
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner

    torch_tensorrt.save(
        trt_gm, "trt.pte", output_format="executorch",
        retrace=False, arg_inputs=inputs,
        partitioners=[
            CudaPartitioner(
                [CudaBackend.generate_method_name_compile_spec("forward")]
            )
        ],
    )

This needs ExecuTorch's CUDA backend and a CUDA toolkit (nvcc/ptxas) at export
time, and produces a ``.pte`` that requires a CUDA runtime at load. Any external
CUDA weights are written as ``.ptd`` data file(s) next to the ``.pte``; the runtime
must be pointed at those data files to load them.

.. warning::

    The CUDA backend names its external weight blob per-device (e.g.
    ``aoti_cuda_blob.ptd``), not per-model, so saving two different coalesced
    ``.pte`` into the same directory overwrites the blob and the first ``.pte``
    will fail to load. Save each coalesced model into its own directory.

.. _executorch_single_stream:

**Running a coalesced .pte: use a single CUDA stream**

A coalesced ``.pte`` runs on more than one backend delegate (the TensorRT delegate
plus the CUDA backend). By default each backend enqueues its GPU work on its *own*
CUDA stream, and delegate execution is asynchronous -- a delegate returns after
*enqueuing* its work, not after it completes. Because separate CUDA streams are not
ordered relative to one another, at a delegate boundary the consuming delegate can
begin before the producing delegate's writes have finished, reading incomplete
data. This is a race: it is intermittent and can surface as wrong results or an
illegal memory access.

The runtime does not impose a shared stream across delegates, so it is the
**runner's responsibility** to run all delegates on one CUDA stream. Create a
single stream and scope ``executorch::extension::cuda::CallerStreamGuard`` over it
for the duration of execution. That one guard reaches every CUDA-capable
delegate: they resolve a single shared ``libextension_cuda``, so the TensorRT
backend and the CUDA backend read the same caller-stream storage. All GPU work is
then enqueued in order and every cross-boundary dependency is satisfied, while
execution stays asynchronous.

If the runner reads a delegate's outputs between calls (for example, an
autoregressive decode loop), synchronize the shared stream before reading: the
work may still be in flight when ``execute()`` returns, and a host-side copy on
the default stream will not wait for a non-blocking stream. A model that threads
its aliased outputs through the delegate is insulated from this in practice:
reflecting each aliased output into its delegate output makes the delegate wait
for the engine before it returns. Under
:ref:`zero-copy KV <executorch_zero_copy_kv>` those outputs are elided, so there
is nothing to reflect and the delegate returns with the engine still running --
the synchronization is then the only thing making a host read see the new values.
**ExecuTorch lowering options**

``torch_tensorrt.save`` takes these extra keyword arguments. They are only
consulted for the ``executorch`` format; passing them with any other
``output_format`` logs a warning and is otherwise ignored. ``constant_methods``,
``transform_passes`` and ``compile_config`` are forwarded to ExecuTorch's
``to_edge_transform_and_lower(...)``; the rest are consumed at other points in
``save``.

* ``constant_methods`` — a ``dict`` of extra constant methods to embed in the
  ``.pte`` (e.g. ``{"get_max_seq_len": 2048}`` for an LLM runner).
* ``transform_passes`` — additional edge-dialect transform passes to run before
  lowering.
* ``compile_config`` — an ``EdgeCompileConfig``. When omitted, Torch-TensorRT
  supplies a default with ``_check_ir_validity=False`` (the TensorRT
  ``execute_engine`` placeholder graph does not pass edge-IR validation). A
  caller-supplied config is forwarded **verbatim**, so if you pass your own and
  your graph carries TensorRT engines, set ``_check_ir_validity=False`` explicitly.
* ``backend_config`` — an ``ExecutorchBackendConfig`` forwarded to
  ``to_executorch(...)``.
* ``zero_copy_kv`` — a ``bool`` (default ``False``, single-method only). Lets the
  TensorRT engine update an aliased KV cache in place. ``save`` owns both ends of
  the Edge boundary, so this one argument covers what the two-call path spells
  out; see :ref:`Zero-copy KV cache <executorch_zero_copy_kv>`.
* ``generate_etrecord`` — a ``bool`` (default ``False``). When ``True``, an
  `ETRecord <https://pytorch.org/executorch/stable/etrecord.html>`_ is written
  next to the ``.pte`` as ``<base>_etrecord.bin`` (e.g. ``trt.pte`` →
  ``trt_etrecord.bin``) for use with the ExecuTorch Developer Tools ``Inspector``.

.. code-block:: python

    from executorch.exir import EdgeCompileConfig

    torch_tensorrt.save(
        trt_gm, "trt.pte", output_format="executorch",
        retrace=False, arg_inputs=inputs,
        constant_methods={"get_max_seq_len": 2048},
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
        generate_etrecord=True,
    )

The ETRecord sidecar can be parsed back and paired with a runtime ETDump in the
Developer Tools ``Inspector``:

.. code-block:: python

    from executorch.devtools import Inspector
    from executorch.devtools.etrecord import parse_etrecord

    etrecord = parse_etrecord("trt_etrecord.bin")
    inspector = Inspector(etdump_path="etdump.etdp", etrecord=etrecord)
    inspector.print_data_tabular()


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
