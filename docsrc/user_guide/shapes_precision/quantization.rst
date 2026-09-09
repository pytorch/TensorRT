.. _quantization:

Quantization (INT8 / FP8 / FP4)
=================================

Torch-TensorRT supports post-training quantization (PTQ) with **INT8**, **FP8**, and
**FP4** precisions via NVIDIA's
`ModelOpt <https://github.com/NVIDIA/TensorRT-Model-Optimizer>`_ library, and
**FP8 weight-only**, **static FP8**, **INT4 weight-only**, **NVFP4
weight-only**, and **MXFP4** (``MXDynamicActivationMXWeightConfig``) via
`TorchAO <https://github.com/pytorch/ao>`_. Quantizers insert
quantize/dequantize (QDQ) nodes into the model graph; Torch-TensorRT then
converts those nodes into TRT quantization layers and sets the appropriate builder flags.

----

Prerequisites
-------------

Install ModelOpt (requires ``nvidia-modelopt``) and/or TorchAO:

.. code-block:: bash

    pip install nvidia-modelopt
    pip install torchao

Hardware requirements:

* **INT8**: Any NVIDIA GPU with TensorRT support.
* **FP8**: NVIDIA Hopper (H100) or newer.
* **INT4 (TorchAO WOQ)**: TensorRT with INT4 constants (TRT ≥ 10.8); storage + DQ, not low-bit MMA.
* **NVFP4 (TorchAO WOQ)**: TensorRT with FP4 constants (TRT ≥ 10.8); packed FP4 storage + DQ, not native FP4 MMA.
* **MXFP4 (TorchAO)**: TensorRT with FP4 + E8M0 constants (TRT ≥ 10.8); packed FP4 storage + DQ, not native MXFP4 MMA. There is no MXFP4 weight-only config.
* **FP4 (ModelOpt NVFP4)**: NVIDIA Blackwell (B100/B200) or newer; requires TensorRT ≥ 10.8.

----

INT8 / FP8 PTQ Workflow
------------------------

**Step 1 — Calibrate the model with ModelOpt**

ModelOpt's ``mtq.quantize`` replaces eligible layers with QDQ wrappers and calibrates
the quantization scales using a small calibration dataset:

.. code-block:: python

    import torch
    import modelopt.torch.quantization as mtq

    model = MyModel().eval().cuda()

    # Define a calibration loop (no gradient needed)
    def calibration_loop(model):
        for batch in calibration_dataloader:
            model(batch.cuda())

    # INT8 configuration (per-tensor activations, per-channel weights)
    quant_cfg = mtq.INT8_DEFAULT_CFG
    # or FP8: quant_cfg = mtq.FP8_DEFAULT_CFG

    mtq.quantize(model, quant_cfg, forward_loop=calibration_loop)

**Step 2 — Compile with Torch-TensorRT**

Pass the quantized model (with QDQ nodes) to the Torch-TensorRT compiler:

.. code-block:: python

    import torch_tensorrt

    inputs = [torch.randn(1, 3, 224, 224).cuda()]

    # INT8
    trt_model = torch_tensorrt.compile(
        model,
        ir="dynamo",
        arg_inputs=inputs,
        min_block_size=1,
    )

    # FP8
    trt_model = torch_tensorrt.compile(
        model,
        ir="dynamo",
        arg_inputs=inputs,
        min_block_size=1,
    )

    output = trt_model(*inputs)

----

FP4 (NVFP4) Workflow
---------------------

FP4 uses **dynamic block quantization** — weights are quantized offline to a block-scaled
4-bit format; activations are dynamically quantized at runtime. This path requires
TensorRT ≥ 10.8 and a Blackwell GPU.

.. code-block:: python

    import modelopt.torch.quantization as mtq

    # FP4 config (uses block quantization for weights)
    quant_cfg = mtq.NVFP4_DEFAULT_CFG

    mtq.quantize(model, quant_cfg, forward_loop=calibration_loop)

    trt_model = torch_tensorrt.compile(
        model,
        ir="dynamo",
        arg_inputs=inputs,
        min_block_size=1,
    )

----

Using ``ExportedProgram`` (``dynamo.compile``)
-----------------------------------------------

When using the ``torch.export`` → ``dynamo.compile`` path, wrap the export step in
``export_torch_mode`` from ModelOpt so the QDQ custom ops are properly traced:

.. code-block:: python

    from modelopt.torch.quantization.utils import export_torch_mode

    with export_torch_mode():
        exp_program = torch.export.export(model, tuple(inputs))

    trt_gm = torch_tensorrt.dynamo.compile(
        exp_program,
        arg_inputs=inputs,
    )

``MutableTorchTensorRTModule`` handles the ``export_torch_mode`` context automatically
when quantization precisions are detected — no manual wrapping required. See
:ref:`mutable_module`.

----

``torch.compile`` Path
-----------------------

Quantization also works with ``torch.compile``:

.. code-block:: python

    trt_model = torch.compile(
        model,  # already quantized with ModelOpt
        backend="torch_tensorrt",
        options={
            "min_block_size": 1,
        },
    )

    output = trt_model(*inputs)

----

TorchAO FP8 Weight-Only Quantization
------------------------------------

TorchAO ``Float8WeightOnlyConfig`` quantizes Linear weights to FP8 (e4m3) while
leaving activations in BF16/FP16. Unlike ModelOpt PTQ, no calibration dataset is
required.

Default TorchAO ``Float8Tensor.dequantize`` decomposes into primitive ops. The
examples promote weights to a ``Float8TensorNonDecomposed`` subclass so export
emits ``torch.ops.torchao.dequantize_affine``, which Torch-TensorRT converts to
``IDequantizeLayer``.

.. code-block:: python

    from torchao.quantization import Float8WeightOnlyConfig, quantize_

    quantize_(model, Float8WeightOnlyConfig())
    model = pre_process_model_for_export(model)  # emit dequantize_affine

    with exclude_dq_from_constant_folding():
        exp_program = torch.export.export(model, (example_input,), strict=True)

    trt_model = torch_tensorrt.dynamo.compile(
        exp_program,
        inputs=[example_input],
        min_block_size=1,
        use_explicit_typing=True,
        require_full_compilation=True,
    )

The intended engine keeps an FP8 weight constant plus a DQ prologue into GEMM.
On **Blackwell**, Myelin can fuse that prologue into the matmul. On other GPUs
DQ + GEMM may run as two kernels — that is still correct as long as the FP8
weight is not constant-folded into a dense high-precision weight.

See :ref:`quantize_linear_fp8_woq` for a toy Linear model and
:ref:`torch_export_flux_fp8_woq` for FLUX.1-dev.

----

TorchAO INT4 Weight-Only Quantization
-------------------------------------

TorchAO group-wise INT4 quantizes Linear weights to 4-bit integers while
leaving activations in BF16. This is **weight-only storage**: TensorRT keeps an
INT4 constant and dequantizes into a high-precision GEMM. It is not NVFP4
Tensor Core MMA.

TensorRT's ``IDequantizeLayer`` rejects nonzero zero-points, so examples use
TorchAO's **symmetric** ``Int4Tensor.from_hp`` path (the float8-activation
branch, then restore BF16 as the activation dtype). Default
``Int4WeightOnlyConfig`` is asymmetric and will fail conversion.

Parent ``Int4Tensor`` Linear dispatches to fused mslk kernels and never calls
``dequantize()``. Promote weights to ``Int4TensorNonDecomposed`` so export emits
``torch.ops.torchao.dequantize_affine``. The converter re-packs the unpacked
int8 qdata into ``trt.DataType.INT4``. Blocked INT4 DQ currently uses an FP32
DQ output, then a cast back to BF16.

Compile with ``immutable_weights=True``. The converter packs INT4 at build
time; engine refit would push the graph's int8 qdata at an INT4 prototype
(``refit weights data type Int8 must equal to weights prototype Int4``).

.. code-block:: python

    quantize_linear_int4_symmetric(model, group_size=128)
    model = pre_process_model_for_export(model)

    with exclude_dq_from_constant_folding():
        exp_program = torch.export.export(model, (example_input,), strict=True)

    trt_model = torch_tensorrt.dynamo.compile(
        exp_program,
        inputs=[example_input],
        min_block_size=1,
        use_explicit_typing=True,
        require_full_compilation=True,
        immutable_weights=True,
    )

Hub checkpoints such as `pytorch/Qwen3-8B-INT4 <https://huggingface.co/pytorch/Qwen3-8B-INT4>`_
may use HQQ or ``Int4TilePackedTo4dTensor``. Dequantize those weights to dense
BF16, then re-pack with ``quantize_linear_int4_symmetric`` /
``convert_hub_int4_to_symmetric_trt``.

See :ref:`quantize_linear_int4_woq` for a toy Linear model,
:ref:`torch_export_flux_int4_woq` for FLUX.1-dev, and
:ref:`torch_export_qwen3_int4_woq` for Qwen3-8B.

----

TorchAO NVFP4 Weight-Only Quantization
--------------------------------------

TorchAO NVFP4 quantizes Linear weights to packed FP4 E2M1 (two values per
byte) with FP8 E4M3 block scales (block size 16) and an FP32 per-tensor
global scale. Activations stay BF16. This is **weight-only storage**:
TensorRT keeps an FP4 constant and dequantizes into a high-precision GEMM.
It is not native FP4 Tensor Core MMA, and it is a different graph from
ModelOpt ``NVFP4_DEFAULT_CFG`` (``dynamic_block_quantize_op``).

Last two weight dims must be divisible by 16. Parent ``NVFP4Tensor`` Linear
does not emit a DQ op TensorRT can map. Promote weights to
``NVFP4TensorNonDecomposed`` so export emits
``torch.ops.torchao_trt.dequantize_nvfp4``. The converter unswizzles TorchAO
block scales, then builds an FP4 constant, an FP8 block-scale constant, and
two-level ``IDequantizeLayer``.

Torch-TensorRT marks ``dequantize_nvfp4`` impure in constant folding so the
packed FP4 weight is not folded into a dense BF16 constant. Compile with
``immutable_weights=True``.

.. code-block:: python

    quantize_linear_nvfp4(model)
    model = pre_process_model_for_export(model)

    exp_program = torch.export.export(model, (example_input,), strict=True)

    trt_model = torch_tensorrt.dynamo.compile(
        exp_program,
        inputs=[example_input],
        min_block_size=1,
        use_explicit_typing=True,
        require_full_compilation=True,
        immutable_weights=True,
    )

See :ref:`quantize_linear_nvfp4_woq` for a toy Linear model and
:ref:`torch_export_flux_nvfp4_woq` for FLUX.1-dev.

----

TorchAO MXFP4 (Dynamic Activation + MX Weight)
----------------------------------------------

TorchAO MXFP4 stores Linear weights as packed FP4 E2M1 (two values per
byte) with E8M0 block scales (block size 32). The public config is
``MXDynamicActivationMXWeightConfig`` — there is no MXFP4 weight-only
recipe (that is NVFP4 / ``NVFP4WeightOnlyConfig``). Native MXFP4xMXFP4
kernels need B200/B300. This path uses emulated storage and a weight DQ
prologue into a high-precision GEMM. Activations stay BF16.

Last two weight dims must be divisible by 32. Parent ``MXTensor`` Linear
does not emit a DQ op TensorRT can map. Promote weights to
``MXTensorNonDecomposed`` so export emits
``torch.ops.torchao_trt.dequantize_mxfp4``. The converter unswizzles TorchAO
block scales, then builds an FP4 constant, an E8M0 block-scale constant, and
``IDequantizeLayer``. Myelin accepts FP4 + E8M0 at block size 32; FP4 +
float32 scales fail at that block size.

Torch-TensorRT marks ``dequantize_mxfp4`` impure in constant folding so the
packed FP4 weight is not folded into a dense BF16 constant. Compile with
``immutable_weights=True``.

.. code-block:: python

    quantize_linear_mxfp4(model)
    model = pre_process_model_for_export(model)

    exp_program = torch.export.export(model, (example_input,), strict=True)

    trt_model = torch_tensorrt.dynamo.compile(
        exp_program,
        inputs=[example_input],
        min_block_size=1,
        use_explicit_typing=True,
        require_full_compilation=True,
        immutable_weights=True,
    )

See :ref:`quantize_linear_mxfp4` for a toy Linear model and
:ref:`torch_export_flux_mxfp4` for FLUX.1-dev.

----

TorchAO Static FP8 Quantization
--------------------------------

Static FP8 quantizes **activations and weights**. Activation scales are chosen
offline with min/max observers (per-tensor activations, per-channel weights),
then Linear layers are rewritten so export emits
``quantize_affine_float8_non_decomposed`` and
``dequantize_affine_float8_non_decomposed``. Torch-TensorRT maps those ops to
``IQuantizeLayer`` / ``IDequantizeLayer``. After fusion, GEMMs can run in FP8.

.. code-block:: python

    quantize_static_fp8(model, (example_input,), calibration_steps=10)
    exp_program = torch.export.export(model, (example_input,), strict=True)
    trt_model = torch_tensorrt.dynamo.compile(
        exp_program,
        inputs=[example_input],
        enabled_precisions={torch.float8_e4m3fn},
        min_block_size=1,
        require_full_compilation=True,
    )

This needs a calibration loop (unlike weight-only). See
:ref:`quantize_linear_fp8_static`.

----

How QDQ Nodes Are Converted
-----------------------------

When Torch-TensorRT encounters ``torch.ops.tensorrt.quantize_op.default`` nodes in the
graph (inserted by ModelOpt), the
``aten_ops_quantize_op`` converter maps them to TRT ``IQuantizeLayer`` /
``IDequantizeLayer`` pairs. The TRT builder then fuses these with adjacent compute layers
(e.g. Conv, Linear) to produce INT8 or FP8 kernel variants.

For TorchAO weight-only graphs, ``torch.ops.torchao.dequantize_affine.default`` is
mapped to ``IDequantizeLayer`` (weight constant stays FP8 or INT4, activations stay
high precision). Group-wise INT4 qdata is re-packed from unpacked int8 into a
``trt.DataType.INT4`` constant.

For TorchAO static FP8 graphs,
``quantize_affine_float8_non_decomposed`` / ``dequantize_affine_float8_non_decomposed``
map to ``IQuantizeLayer`` / ``IDequantizeLayer`` so both activations and weights
participate in FP8 GEMM fusion.

For TorchAO NVFP4 weight-only graphs, ``torch.ops.torchao_trt.dequantize_nvfp4.default``
is mapped to two-level ``IDequantizeLayer`` (FP8 block scales restored with the
global scale, then packed FP4 dequantized into the GEMM input type).

For TorchAO MXFP4 graphs, ``torch.ops.torchao_trt.dequantize_mxfp4.default``
is mapped to ``IDequantizeLayer`` (packed FP4 + logical E8M0 block scales at
block size 32).

For ModelOpt FP4, ``torch.ops.tensorrt.dynamic_block_quantize_op.default`` nodes
are converted via the dynamic block quantize converter, which uses TRT's
``add_dynamic_quantize`` API (TRT ≥ 10.8).

The ``constant_folding`` lowering pass explicitly marks quantization ops as *impure* to
prevent their scales from being folded away before the TRT conversion step.

----

Verifying Quantized Layers
---------------------------

Use :ref:`dryrun` to check how many ops were partitioned into TRT blocks and whether the
quantized layers were included:

.. code-block:: python

    trt_gm = torch_tensorrt.dynamo.compile(
        exp_program,
        arg_inputs=inputs,
        dryrun=True,
    )

----

Supported Precision / Hardware Matrix
---------------------------------------

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Precision
     - Minimum GPU
     - TRT requirement
   * - INT8
     - Any TRT-capable GPU
     - Any supported TRT version
   * - FP8
     - NVIDIA Hopper (H100+)
     - TRT ≥ 8.6
   * - INT4 (TorchAO WOQ)
     - Any TRT-capable GPU with INT4 support
     - TRT ≥ 10.8
   * - NVFP4 (TorchAO WOQ)
     - GPU with TRT FP4 constants
     - TRT ≥ 10.8
   * - MXFP4 (TorchAO)
     - GPU with TRT FP4 + E8M0 constants
     - TRT ≥ 10.8
   * - FP4 (ModelOpt NVFP4)
     - NVIDIA Blackwell (B100+)
     - TRT ≥ 10.8

----

Troubleshooting
---------------

**"Unable to import quantization op"**
    ModelOpt is not installed or ``torch.ops.tensorrt.quantize_op`` was not registered.
    Run ``pip install nvidia-modelopt`` and ensure the Torch-TensorRT package is imported
    before calling ``mtq.quantize``.

**QDQ nodes fall back to PyTorch (not TRT)**
    Verify ``min_block_size`` is not too large — use ``dryrun=True`` to inspect coverage.

**"TensorRT-RTX does not support int8 activation quantization"**
    INT8 activation quantization (``input_quantizer`` nodes) is not supported by
    **TensorRT-RTX** — the Windows-native RTX inference library. INT8 weight
    quantization still works. Use a weight-only INT8 ModelOpt config, or compile on
    Linux with standard TensorRT instead of TensorRT-RTX.

**FP4 "requires TRT ≥ 10.8" error**
    Upgrade TensorRT. FP4 uses ``add_dynamic_quantize`` which is only available in
    TRT 10.8 and newer.

**TorchAO FP8 weights folded to BF16/FP16**
    Export or Torch-TensorRT constant folding removed ``dequantize_affine``.
    Promote weights with ``pre_process_model_for_export`` and wrap export in
    ``exclude_dq_from_constant_folding`` as in :ref:`quantize_linear_fp8_woq`.
    Install ``torchao`` so the converter and constant-folding exclusion register.

**TorchAO INT4 weights folded to BF16, or engine constants are Int8 not Int4**
    The graph must keep ``dequantize_affine`` (same preprocess / constant-fold
    exclusion as FP8 WOQ). Group-wise INT4 also requires the INT4 pack path in
    the converter; per-channel INT8-style DQ will not emit ``Datatype: Int4``.
    Use symmetric quantization (zero zero-point) and ``immutable_weights=True``.

**"IDequantizeLayer rejects nonzero zero_point" / INT4 compile fails**
    TorchAO's default INT4 config is asymmetric. Use
    ``quantize_linear_int4_symmetric`` (or ``convert_hub_int4_to_symmetric_trt``
    for Hub checkpoints) as in :ref:`quantize_linear_int4_woq`.

**TorchAO NVFP4 weights folded to BF16, or engine has no FP4E2M1 constant**
    The graph must keep ``torchao_trt.dequantize_nvfp4``. Promote weights with
    ``pre_process_model_for_export`` as in :ref:`quantize_linear_nvfp4_woq`.
    Last two Linear dims must be divisible by 16. This path is weight-only DQ
    + BF16 GEMM, not ModelOpt ``dynamic_block_quantize_op``.

**TorchAO MXFP4 weights folded to BF16, or engine has no FP4E2M1 constant**
    The graph must keep ``torchao_trt.dequantize_mxfp4``. Promote weights with
    ``pre_process_model_for_export`` as in :ref:`quantize_linear_mxfp4`.
    Last two Linear dims must be divisible by 32. Pass E8M0 scales (not
    float32) at block size 32. This is a weight DQ prologue, not native
    MXFP4xMXFP4 MMA.
