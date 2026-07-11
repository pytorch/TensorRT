.. _distributed_inference:

Distributed Inference
=====================

Torch-TensorRT supports two distributed inference patterns:

* **Data parallel** — the same TRT-compiled model runs independently on multiple GPUs,
  each processing a different shard of the batch.
* **Tensor parallel** — the model is sharded across GPUs using ``torch.distributed``
  and DTensors; Torch-TensorRT compiles each per-GPU shard.

Both patterns produce standard TRT engines; Torch-TensorRT handles only the compilation
step — data movement and process coordination remain the responsibility of the
distributed framework you choose.

----

Data Parallel Inference
-----------------------

The simplest path is to replicate the compiled model on each GPU process using
`Accelerate <https://huggingface.co/docs/accelerate>`_ or PyTorch DDP. Each process
compiles its own TRT engine independently.

.. code-block:: python

    # Run with: torchrun --nproc_per_node=<N> script.py
    import torch
    import torch_tensorrt
    from accelerate import PartialState

    distributed_state = PartialState()
    device = distributed_state.device  # GPU assigned to this rank

    model = MyModel().eval().to(device)
    inputs = [torch.randn(1, 3, 224, 224).to(device)]

    with distributed_state.split_between_processes(inputs) as local_inputs:
        trt_model = torch_tensorrt.compile(
            model,
            ir="dynamo",
            arg_inputs=local_inputs,
            use_explicit_typing=True,  # enabled_precisions deprecated; cast model/inputs to target dtype
            min_block_size=1,
        )
        output = trt_model(*local_inputs)

See the `data_parallel_gpt2.py <https://github.com/pytorch/TensorRT/blob/main/examples/distributed_inference/data_parallel_gpt2.py>`_
and `data_parallel_stable_diffusion.py <https://github.com/pytorch/TensorRT/blob/main/examples/distributed_inference/data_parallel_stable_diffusion.py>`_
examples for complete Accelerate-based workflows with GPT-2 and Stable Diffusion.

----

Tensor Parallel Inference
--------------------------

Tensor parallelism shards model weights across GPUs. Each GPU holds a slice of every
weight tensor and participates in collective operations (all-reduce, all-gather,
reduce-scatter) to execute the full model forward pass.

Torch-TensorRT supports two compilation workflows for tensor-parallel models:

1. **torch.compile** (JIT) — uses ``torch._dynamo`` to trace the model at runtime.
   Works with DTensor-parallelized models directly.
2. **torch.export** (AOT) — ahead-of-time export, TRT compilation, and save/load.
   Requires manual weight slicing (DTensor is not yet supported by ``torch.export``).

----

Workflow 1: torch.compile (JIT)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest path for tensor-parallel inference. Shard the model with
``parallelize_module`` (DTensor), compile with ``torch.compile``, and wrap
inference in ``distributed_context`` for safe NCCL lifecycle management:

.. code-block:: python

    import os
    import torch
    import torch.distributed as dist
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
        parallelize_module,
    )
    import torch_tensorrt

    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{dist.get_rank()}")
    tp_mesh = dist.device_mesh.init_device_mesh("cuda", (dist.get_world_size(),))

    model = MyModel().eval().half().to(device)
    parallelize_module(
        model,
        tp_mesh,
        {
            "attn.q_proj": ColwiseParallel(),
            "attn.o_proj": RowwiseParallel(),
            "mlp.gate_proj": ColwiseParallel(),
            "mlp.down_proj": RowwiseParallel(),
        },
    )

    trt_model = torch.compile(
        model,
        backend="torch_tensorrt",
        dynamic=True,
        options={
            "use_explicit_typing": True,
            "use_fp32_acc": True,
            "use_python_runtime": False,  # C++ runtime
            "min_block_size": 1,
        },
    )

    # Warmup — triggers engine build
    _ = trt_model(input_ids, position_ids=position_ids)
    dist.barrier()

    # Use distributed_context to manage the NCCL lifecycle.
    # On __exit__ it releases NCCL communicators from TRT engines,
    # making dist.destroy_process_group() safe to call afterward.
    with torch_tensorrt.distributed.distributed_context(dist.group.WORLD, trt_model) as model:
        output = model(input_ids, position_ids=position_ids)

    dist.destroy_process_group()
    os._exit(0)  # bypass Python GC destructor races

**Key points:**

* **Use** ``distributed_context`` **for safe teardown** — TRT engines hold a raw
  pointer to the NCCL communicator.  ``distributed_context(group, module)`` tracks
  all multi-device engines and calls ``release_nccl_comm()`` on ``__exit__``,
  detaching the communicator so ``dist.destroy_process_group()`` doesn't cause a
  use-after-free.  Always follow the block with ``dist.destroy_process_group()``
  and ``os._exit(0)``.
* **Automatic distributed tracing** — when ``dist.init_process_group()`` has been
  called and ``world_size > 1``, Torch-TensorRT detects the active distributed context
  and automatically switches the ``torch.compile`` backend tracer from the default
  ``torch._dynamo`` path to ``aot_autograd``. You do not need to set
  ``use_distributed_mode_trace=True`` explicitly.
* ``dynamic=True`` enables dynamic sequence lengths — TRT builds a single engine that
  handles varying input shapes without recompiling.
* NCCL all-reduce ops from ``RowwiseParallel`` are fused and converted to native TRT
  ``DistCollective`` layers (TRT 10.16+) or TRT-LLM plugin layers (TRT < 10.16).
* The warmup forward pass should use ``torch._dynamo.mark_dynamic()`` to match the
  generate loop and avoid a recompile.
* **Non-default TP subgroup** — if the TRT engine should use a subgroup communicator
  (e.g. tensor-parallel inside a data-parallel job), pass the subgroup instead of
  ``dist.group.WORLD``:

  .. code-block:: python

      tp_group = dist.new_group(ranks=[0, 1])
      with torch_tensorrt.distributed.distributed_context(tp_group, trt_model) as model:
          output = model(inp)

For a complete LLM example, see
`tensor_parallel_llama_llm.py <https://github.com/pytorch/TensorRT/blob/main/tools/llm/tensor_parallel_llama_llm.py>`_
and the multinode variant
`tensor_parallel_llama_multinode.py <https://github.com/pytorch/TensorRT/blob/main/tools/llm/tensor_parallel_llama_multinode.py>`_.

----

Workflow 2: torch.export (AOT) with Save / Load
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For deployment scenarios where you want to compile once and load many times (e.g.
serving), use the export → compile → save → load workflow.

**Limitation:** ``torch.export`` does not currently support ``DTensor``-parallelized
models (sharding propagation fails on symbolic reshapes). The workaround is to manually
slice weights per-rank and insert explicit ``_c10d_functional.all_reduce`` ops for
row-parallel layers.

Step 1: Export and Save
""""""""""""""""""""""""

.. code-block:: python

    import torch
    import torch.distributed as dist
    import torch_tensorrt

    # Manual row-parallel wrapper (replaces DTensor RowwiseParallel)
    class RowParallelLinear(torch.nn.Module):
        def __init__(self, linear, group_name):
            super().__init__()
            self.linear = linear
            self.group_name = group_name

        def forward(self, x):
            out = self.linear(x)
            out = torch.ops._c10d_functional.all_reduce(out, "sum", self.group_name)
            out = torch.ops._c10d_functional.wait_tensor(out)
            return out

    # Slice weights for this rank and wrap row-parallel layers
    group_name = dist.distributed_c10d._get_default_group().group_name
    # ... slice column-parallel weights on dim 0, row-parallel on dim 1 ...
    model.o_proj = RowParallelLinear(model.o_proj, group_name)

    # Export (no DTensor → export succeeds)
    ep = torch.export.export(model, args=(input_ids,), strict=False)

    # Compile with TRT
    trt_model = torch_tensorrt.dynamo.compile(
        ep,
        inputs=[input_ids],
        use_explicit_typing=True,
        use_fp32_acc=True,
        use_python_runtime=False,
        min_block_size=1,
        use_distributed_mode_trace=True,
    )

    # Save per-rank engine
    torch_tensorrt.save(trt_model, f"/engines/model_rank{rank}.ep",
                        inputs=[input_ids], retrace=False)

Step 2: Load and Run
""""""""""""""""""""""

**Default world group** (most common — all ranks share one TP group):

.. code-block:: python

    import os
    import torch
    import torch.distributed as dist
    import torch_tensorrt
    from torch_tensorrt.distributed import setup_nccl_for_torch_tensorrt
    from torch_tensorrt.distributed._nccl_utils import initialize_nccl_comm

    dist.init_process_group(backend="nccl")
    setup_nccl_for_torch_tensorrt()
    initialize_nccl_comm()  # eagerly create NCCL communicator for TRT
    rank = dist.get_rank()

    # Load the per-rank engine
    loaded = torch_tensorrt.load(f"/engines/model_rank{rank}.ep")
    trt_model = loaded.module()

    with torch_tensorrt.distributed.distributed_context(dist.group.WORLD, trt_model) as model:
        output = model(input_ids)

    dist.destroy_process_group()
    os._exit(0)

**Non-default TP subgroup** (e.g. tensor-parallel inside a data-parallel job):

Use ``distributed_context(group, module)`` to pin the group on all TRT engines
in the loaded module and get the configured model back as the context value:

.. code-block:: python

    import os
    import torch
    import torch.distributed as dist
    import torch_tensorrt
    from torch_tensorrt.distributed import setup_nccl_for_torch_tensorrt
    from torch_tensorrt.distributed._nccl_utils import initialize_nccl_comm

    dist.init_process_group(backend="nccl")
    setup_nccl_for_torch_tensorrt()
    initialize_nccl_comm()

    tp_group = dist.new_group(ranks=[0, 1])  # tensor-parallel ranks
    rank = dist.get_rank()

    loaded = torch_tensorrt.load(f"/engines/model_rank{rank}.ep")
    raw_model = loaded.module()

    with torch_tensorrt.distributed.distributed_context(tp_group, raw_model) as trt_model:
        output = trt_model(input_ids)

    dist.destroy_process_group()
    os._exit(0)

**What happens under the hood:**

* ``_c10d_functional.all_reduce`` + ``wait_tensor`` are fused into
  ``tensorrt_fused_nccl_all_reduce_op`` by the ``fuse_distributed_ops`` lowering pass.
* The fused op is converted to a native TRT ``DistCollective`` layer inside the engine.
* At load time, the C++ runtime auto-resolves the NCCL process group name from the
  c10d registry.  ``initialize_nccl_comm()`` eagerly creates PyTorch's lazy NCCL
  communicator is initialized before TRT tries to bind to it.
* The engine is serialized with ``requires_native_multidevice=True``, which tells the C++ runtime to bind
  the NCCL communicator on first execution.

For a complete example, see
`tensor_parallel_llama_export.py <https://github.com/pytorch/TensorRT/blob/main/tools/llm/tensor_parallel_llama_export.py>`_.

----

Multinode Inference
--------------------

For tensor parallelism across multiple nodes (one GPU per node), use
``torchtrtrun`` — a ``torchrun``-compatible launcher included in Torch-TensorRT
that automatically sets up NCCL before spawning worker processes.

.. code-block:: bash

    # Node 0 (rank 0):
    torchtrtrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \
      --rdzv_endpoint=<node0-ip>:29500 \
      tensor_parallel_llama_multinode.py

    # Node 1 (rank 1):
    torchtrtrun --nproc_per_node=1 --nnodes=2 --node_rank=1 \
      --rdzv_endpoint=<node0-ip>:29500 \
      tensor_parallel_llama_multinode.py

Single-node multi-GPU also works:

.. code-block:: bash

    torchtrtrun --nproc_per_node=2 tensor_parallel_llama_llm.py

.. note::

   **TRT 10.x NCCL library workaround** — In TRT 10.x,
   ``IExecutionContext::setCommunicator`` calls ``dlopen("libnccl.so")``
   at runtime via TRT's internal ``libLoader``. This happens inside the C++
   runtime before any Python code executes, so setting ``LD_LIBRARY_PATH``
   or ``LD_PRELOAD`` inside the script is too late. This is fixed in TRT 11.0.

   ``torchtrtrun`` works around this by:

   1. Finding the ``nvidia-nccl`` pip package (``nvidia.nccl``).
   2. Creating a ``libnccl.so → libnccl.so.2`` symlink if missing (pip only
      ships ``libnccl.so.2``).
   3. Prepending the NCCL lib directory to ``LD_LIBRARY_PATH``.
   4. Setting ``LD_PRELOAD`` to ``libnccl.so.2`` so TRT's ``dlopen`` finds
      the library already resident in the process.
   5. Spawning worker processes with ``RANK``, ``LOCAL_RANK``, ``WORLD_SIZE``,
      ``MASTER_ADDR``, and ``MASTER_PORT`` set.

**Manual setup (without** ``torchtrtrun`` **):**

If you prefer to launch processes yourself, replicate the NCCL setup manually:

.. code-block:: bash

    # Step 1 — create the libnccl.so symlink
    python -c "
    from torch_tensorrt.distributed import setup_nccl_for_torch_tensorrt
    setup_nccl_for_torch_tensorrt()
    "

    # Step 2 — find the NCCL lib directory
    NCCL_LIB=$(python -c "
    from torch_tensorrt.distributed._nccl_utils import get_nccl_library_path
    print(get_nccl_library_path())
    ")

    # Step 3 — launch with LD_PRELOAD and LD_LIBRARY_PATH set before the process starts
    # Node 0:
    LD_PRELOAD="$NCCL_LIB/libnccl.so.2" LD_LIBRARY_PATH="$NCCL_LIB:$LD_LIBRARY_PATH" \
      RANK=0 WORLD_SIZE=2 MASTER_ADDR=<node0-ip> MASTER_PORT=29500 \
      python tensor_parallel_llama_multinode.py

    # Node 1:
    LD_PRELOAD="$NCCL_LIB/libnccl.so.2" LD_LIBRARY_PATH="$NCCL_LIB:$LD_LIBRARY_PATH" \
      RANK=1 WORLD_SIZE=2 MASTER_ADDR=<node0-ip> MASTER_PORT=29500 \
      python tensor_parallel_llama_multinode.py

**Important considerations for multinode:**

* Set a long NCCL timeout (e.g. 2 hours) via
  ``dist.init_process_group(timeout=datetime.timedelta(hours=2))``
  to prevent watchdog timeouts during TRT engine building.
* Add a ``dist.barrier()`` after the warmup forward pass so all ranks finish
  engine building before starting inference.
* If using NGC or a system-installed NCCL (``libnccl.so`` already on
  ``LD_LIBRARY_PATH``), the ``torchtrtrun`` setup step is skipped automatically.

----

NCCL Collective Ops in TRT Engines
-------------------------------------

Torch-TensorRT compiles NCCL collective ops (all-reduce, all-gather, reduce-scatter)
directly into the TRT engine binary. There are two backend paths, selected automatically
at import time based on what is available in your TRT build.

**Selection priority (highest to lowest):**

1. **Native TRT DistCollective** (TRT 10.16+ — preferred)
2. **TRT-LLM plugin** (TRT < 10.16 fallback)
3. **PyTorch fallback** (ops remain outside the TRT subgraph)

Check which path is active in your environment:

.. code-block:: python

    from torch_tensorrt._features import ENABLED_FEATURES
    print(ENABLED_FEATURES.native_trt_collectives)  # True → native path
    print(ENABLED_FEATURES.trtllm_for_nccl)         # True → TRT-LLM path

----

Path 1: Native TRT DistCollective (TRT 10.16+)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The preferred path. No external libraries needed — NCCL collectives are first-class
TRT layers compiled into the engine. Requires:

* TensorRT ≥ 10.16
* Torch-TensorRT built with ``ENABLE_TRT_NCCL_COLLECTIVES=ON`` (default for CUDA builds)

**How it works end-to-end:**

1. **Graph lowering** — the ``fuse_distributed_ops`` pass rewrites each
   ``_c10d_functional.<collective> + wait_tensor`` pair into a single fused custom op:

   * ``tensorrt_fused_nccl_all_reduce_op``
   * ``tensorrt_fused_nccl_all_gather_op``
   * ``tensorrt_fused_nccl_reduce_scatter_op``

2. **TRT compilation** — the ``_TRTInterpreter`` sets
   ``PreviewFeature.MULTIDEVICE_RUNTIME_10_16`` on the builder config and then
   each fused op converter calls ``INetworkDefinition.add_dist_collective()`` to
   insert a native ``DistCollective`` layer (``trt.CollectiveOperation.ALL_REDUCE``,
   ``ALL_GATHER``, or ``REDUCE_SCATTER``) into the TRT network.

3. **Serialization** — the engine is serialized with the ``requires_native_multidevice=True`` flag in the
   Torch-TRT metadata, signalling to the C++ runtime that NCCL communicator binding is
   required at load time.

4. **Runtime — NCCL communicator binding** — on first execution, the C++ runtime calls
   ``TRTEngine::bind_nccl_comm()``:

   * It resolves the process group via the group name stored on the engine (or probes
     the c10d registry for the first group with an NCCL backend).
   * It fetches the ``ncclComm_t`` pointer from PyTorch's ``ProcessGroupNCCL``.
   * It calls ``IExecutionContext::setCommunicator()`` to pass the communicator to TRT.

   This is why ``initialize_nccl_comm()`` must be called before the first inference on a
   **loaded** engine: PyTorch creates the NCCL communicator lazily (on the first
   collective), so without the eager-init call, ``bind_nccl_comm()`` would find a null
   pointer.

.. note::

   The communicator binding happens inside the C++ runtime **after** the process
   launches. ``LD_LIBRARY_PATH`` and ``LD_PRELOAD`` must therefore be set **before**
   the process starts (TRT's internal ``libLoader`` calls ``dlopen("libnccl.so")`` at
   startup). ``torchtrtrun`` handles this automatically.

----

Path 2: TRT-LLM Plugin (TRT < 10.16)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Used automatically when native TRT collectives are not available. Requires:

* The TRT-LLM plugin library (``libnvinfer_plugin_tensorrt_llm.so``)
* Either ``TRTLLM_PLUGINS_PATH=/path/to/lib`` set in the environment, or
  ``USE_TRTLLM_PLUGINS=1`` (downloads the TRT-LLM distribution automatically)

The same ``fuse_distributed_ops`` lowering pass runs, but the converter calls
TRT-LLM's plugin API instead of ``add_dist_collective()``. The runtime behaviour and
``requires_native_multidevice`` flag are identical.

----

Path 3: PyTorch Fallback
^^^^^^^^^^^^^^^^^^^^^^^^^^

If neither backend is available, the ``_c10d_functional`` ops are not registered as TRT
converters and remain outside the TRT subgraph. They execute in PyTorch on every call.
This produces correct results but loses the performance benefit of in-engine collectives.

.. warning::

   Compile with ``use_distributed_mode_trace=True`` regardless of which backend is
   active. Without it, the FX tracer may not see the collective ops at all.

----

Confirming the active backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch_tensorrt
    from torch_tensorrt._features import ENABLED_FEATURES

    if ENABLED_FEATURES.native_trt_collectives:
        print("Native TRT DistCollective (TRT 10.16+)")
    elif ENABLED_FEATURES.trtllm_for_nccl:
        print("TRT-LLM plugin")
    else:
        print("PyTorch fallback — collectives will NOT be inside the TRT engine")

----

Process Group Management and Teardown
---------------------------------------

``torch_tensorrt.distributed.distributed_context(group, module=None)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The **recommended** way to manage the NCCL lifecycle for distributed TRT
engines.  This context manager:

1. **On enter** — sets the active process group and pins it on all TRT engines
   in *module* (both submodule and inlined engine storage patterns).
2. **During the block** — inference uses the specified NCCL communicator.
3. **On exit** — calls ``release_nccl_comm()`` on every tracked multi-device
   engine, detaching the NCCL communicator from TRT's execution context.  This
   makes ``dist.destroy_process_group()`` safe to call afterward.

*module* may be a single compiled module, a list of modules, or omitted:

**Single module** — yields the module as the context value:

.. code-block:: python

    trt_model = torch.compile(model, backend="torch_tensorrt", ...)
    _ = trt_model(inp)  # warmup — triggers engine build
    dist.barrier()

    with torch_tensorrt.distributed.distributed_context(dist.group.WORLD, trt_model) as model:
        output = model(inp)

    dist.destroy_process_group()
    os._exit(0)

**Multiple modules** — pass a list; yields a list in the same order.  Useful
when you have separately compiled programs (e.g. an encoder and a decoder)
that both need NCCL teardown:

.. code-block:: python

    with torch_tensorrt.distributed.distributed_context(
        tp_group, [encoder_trt, decoder_trt]
    ) as (enc, dec):
        output = dec(enc(inp))

    dist.destroy_process_group()
    os._exit(0)

**Non-default TP subgroup** (e.g. tensor-parallel inside a data-parallel job):

.. code-block:: python

    tp_group = dist.new_group(ranks=[0, 1])
    with torch_tensorrt.distributed.distributed_context(tp_group, trt_model) as model:
        output = model(inp)

**Without module** — use at compile time when the model is created inside the
block:

.. code-block:: python

    with torch_tensorrt.distributed.distributed_context(tp_group):
        trt_model = torch.compile(model, backend="torch_tensorrt", ...)
        output = trt_model(inp)

``torch_tensorrt.distributed.set_distributed_mode(group, module)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Permanently pins *group* on all TRT engines in *module* without entering a
context manager.  Use this when the group should remain set across multiple
calls outside any ``with`` block:

.. code-block:: python

    torch_tensorrt.distributed.set_distributed_mode(tp_group, model)
    output1 = model(inp1)  # group already pinned
    output2 = model(inp2)

.. note::

   ``set_distributed_mode`` pins the group but does **not** handle teardown.
   Prefer ``distributed_context(group, module)`` when you need safe cleanup.

Both APIs handle three engine storage patterns:

* **Submodule engines** — ``TorchTensorRTModule`` children produced by
  ``torch.compile`` or ``torch_tensorrt.compile()``.
* **Inlined engines** — ``torch.classes.tensorrt.Engine`` objects stored as
  plain attributes on an ``fx.GraphModule`` after
  ``torch_tensorrt.save()`` / ``torch_tensorrt.load()``.
* **torch.compile engines** — engines inside dynamo's code cache, tracked via
  a thread-local registry populated at engine creation time.

For ``PythonTorchTensorRTModule`` (``use_python_runtime=True``), the group is
read lazily from the active context on the first forward call, so
``distributed_context`` (without the *module* argument) is sufficient — keep the
context manager active for the duration of inference.

.. note::

   **Always end distributed scripts with** ``os._exit(0)`` **after**
   ``dist.destroy_process_group()``.

   TRT engines and CUDA contexts register C++ destructors that can race with
   Python's garbage collector during normal interpreter shutdown, causing
   segfaults or hangs.  ``os._exit(0)`` bypasses the GC and exits immediately
   with a clean exit code, avoiding this entirely.

   .. code-block:: python

       dist.destroy_process_group()
       os._exit(0)  # bypass Python GC — TRT/CUDA destructors race during shutdown

   This applies to all distributed workflows: ``torch.compile``, export/load,
   and multinode.  It is safe because all meaningful work has completed before
   this point.

----

Compilation Settings for Distributed Workloads
-------------------------------------------------

.. list-table::
   :widths: 35 15 50
   :header-rows: 1

   * - Setting
     - Default
     - Description
   * - ``use_distributed_mode_trace``
     - ``False``
     - Use ``aot_autograd`` for tracing instead of the default ``torch._dynamo`` path.
       **Auto-enabled** for ``torch.compile`` when ``dist.is_initialized()`` and
       ``world_size > 1`` — no explicit flag needed. Must be set manually when using
       ``torch_tensorrt.dynamo.compile()`` directly (e.g. AOT export workflows).
   * - ``use_python_runtime``
     - ``None`` (auto)
     - ``False`` (C++ runtime) is recommended for production. The C++ runtime handles
       NCCL via TRT's native ``DistCollective`` layers. The Python runtime uses
       Python-level NCCL wrappers.
   * - ``use_explicit_typing``
     - ``True``
     - Respect dtypes set in model/inputs (recommended). Use ``model.half()`` or
       ``enable_autocast=True`` for lower-precision workloads. ``enabled_precisions``
       is **deprecated** and must not be used alongside ``use_explicit_typing``.
   * - ``assume_dynamic_shape_support``
     - ``False``
     - Set to ``True`` for dynamic sequence lengths in LLM generation loops.
   * - ``use_fp32_acc``
     - ``False``
     - Use FP32 accumulation for FP16 models. Improves numerical accuracy.

----

Launching Distributed Scripts
-------------------------------

**Single node, multiple GPUs** — use ``torchrun`` or ``mpirun``:

.. code-block:: bash

    # torchrun
    torchrun --nproc_per_node=2 tensor_parallel_llama_llm.py

    # mpirun
    mpirun -n 2 python tensor_parallel_llama_llm.py

**Multiple nodes** — use environment variables:

.. code-block:: bash

    # Node 0
    RANK=0 WORLD_SIZE=2 MASTER_ADDR=<ip> MASTER_PORT=29500 \
      python tensor_parallel_llama_multinode.py

    # Node 1
    RANK=1 WORLD_SIZE=2 MASTER_ADDR=<ip> MASTER_PORT=29500 \
      python tensor_parallel_llama_multinode.py

----

Examples
--------

The following complete examples are available in the Torch-TensorRT repository:

.. list-table::
   :widths: 45 55
   :header-rows: 1

   * - Script
     - Description
   * - ``examples/distributed_inference/data_parallel_gpt2.py``
     - Data-parallel GPT-2 inference with Accelerate
   * - ``examples/distributed_inference/data_parallel_stable_diffusion.py``
     - Data-parallel Stable Diffusion with Accelerate
   * - ``examples/distributed_inference/tensor_parallel_simple_example.py``
     - Two-layer MLP with column / row parallel sharding (torch.compile)
   * - ``examples/distributed_inference/tensor_parallel_rotary_embedding.py``
     - Tensor-parallel attention with rotary positional embeddings (RoPE)
   * - ``examples/distributed_inference/test_multinode_nccl.py``
     - Multinode NCCL test (C++ and Python runtime)
   * - ``examples/distributed_inference/test_multinode_export_save_load.py``
     - Export → save → load round-trip test for distributed engines
   * - ``tools/llm/tensor_parallel_llama_multinode.py``
     - Llama TP with torch.compile (multinode, C++ runtime)
   * - ``tools/llm/tensor_parallel_llama_export.py``
     - Llama TP: export + save + load workflow (multinode)
   * - ``tools/llm/tensor_parallel_qwen_multinode.py``
     - Qwen TP with torch.compile (multinode)
