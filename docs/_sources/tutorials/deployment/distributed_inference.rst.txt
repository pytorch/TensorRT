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
weight tensor and participates in collective operations (all-gather, reduce-scatter) to
execute the full model forward pass.

Torch-TensorRT compiles the per-GPU shard of the model. Use
``use_distributed_mode_trace=True`` to switch the export path to ``aot_autograd``, which
handles DTensor inputs correctly:

.. code-block:: python

    import torch
    import torch.distributed as dist
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
        parallelize_module,
    )
    import torch_tensorrt

    # --- Initialize process group ---
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    tp_mesh = dist.device_mesh.init_device_mesh("cuda", (dist.get_world_size(),))

    # --- Shard the model across GPUs ---
    model = MyModel().eval().to(device)
    parallelize_module(
        model,
        tp_mesh,
        {
            "fc1": ColwiseParallel(),
            "fc2": RowwiseParallel(),
        },
    )

    # --- Compile each GPU's shard with Torch-TensorRT ---
    inputs = [torch.randn(8, 512).to(device)]
    trt_model = torch.compile(
        model,
        backend="torch_tensorrt",
        options={
            "use_distributed_mode_trace": True,
            "use_explicit_typing": True,  # enabled_precisions deprecated
            "use_python_runtime": True,
            "min_block_size": 1,
        },
    )

    output = trt_model(*inputs)

``use_distributed_mode_trace=True`` is **required** whenever:

* The model contains ``DTensor`` parameters (from ``parallelize_module``).
* The model uses ``torch.distributed`` collective ops that appear as graph nodes.

Without it, the default export path (``aot_export_joint_simple``) will fail on DTensor
inputs and Torch-TensorRT will emit a warning.

----

NCCL Collective Ops in TRT Graphs
-----------------------------------

For models where collective ops (all-gather, reduce-scatter) appear inside the
TRT-compiled subgraph, Torch-TensorRT can fuse them using TensorRT-LLM plugins.

The ``fuse_distributed_ops`` lowering pass automatically rewrites consecutive
``_c10d_functional.all_gather_into_tensor`` / ``reduce_scatter_tensor`` +
``wait_tensor`` pairs into fused custom ops:

* ``tensorrt_fused_nccl_all_gather_op``
* ``tensorrt_fused_nccl_reduce_scatter_op``

These are then converted by custom converters into TensorRT-LLM AllGather / ReduceScatter
plugin layers (requires ``ENABLED_FEATURES.trtllm_for_nccl``). When the TRT-LLM plugin is
unavailable, the ops fall back to PyTorch execution transparently.

See the `tensor_parallel_rotary_embedding.py <https://github.com/pytorch/TensorRT/blob/main/examples/distributed_inference/tensor_parallel_rotary_embedding.py>`_
example for a Llama-style model with NCCL collective ops compiled end-to-end.

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
     - Use ``aot_autograd`` for tracing instead of the default path. Required when the
       model contains DTensor or other distributed tensors.
   * - ``use_python_runtime``
     - ``None`` (auto)
     - Use the Python runtime. Often set to ``True`` for tensor-parallel models that run
       inside an existing distributed process group.
   * - ``use_explicit_typing``
     - ``True``
     - Respect dtypes set in model/inputs (recommended). Use ``model.half()`` or
       ``enable_autocast=True`` for lower-precision workloads. ``enabled_precisions`` is **deprecated**.

----

Launching Distributed Scripts
-------------------------------

Use ``torchrun`` to launch multi-GPU scripts:

.. code-block:: bash

    # 4-GPU tensor-parallel job
    torchrun --nproc_per_node=4 tensor_parallel_example.py

    # 2-node, 8-GPU data-parallel job
    torchrun --nnodes=2 --nproc_per_node=4 --rdzv_backend=c10d \
             --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT data_parallel_example.py

----

Examples
--------

The following complete examples are available in the Torch-TensorRT repository under
``examples/distributed_inference/``:

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Script
     - Description
   * - ``data_parallel_gpt2.py``
     - Data-parallel GPT-2 inference with Accelerate
   * - ``data_parallel_stable_diffusion.py``
     - Data-parallel Stable Diffusion with Accelerate
   * - ``tensor_parallel_simple_example.py``
     - Two-layer MLP with column / row parallel sharding
   * - ``tensor_parallel_rotary_embedding.py``
     - Llama-3 RoPE module with NCCL collective ops
