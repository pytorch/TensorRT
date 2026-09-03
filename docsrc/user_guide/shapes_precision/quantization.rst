.. _quantization:

Quantization (INT8 / FP8 / FP4)
=================================

Torch-TensorRT supports post-training quantization (PTQ) with **INT8**, **FP8**, and
**FP4** precisions via NVIDIA's
`ModelOpt <https://github.com/NVIDIA/TensorRT-Model-Optimizer>`_ library, and
**FP8 weight-only** and **static FP8** quantization via
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
* **FP4 (NVFP4)**: NVIDIA Blackwell (B100/B200) or newer; requires TensorRT ≥ 10.8.

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
mapped to ``IDequantizeLayer`` (weight constant stays FP8/INT, activations stay
high precision).

For TorchAO static FP8 graphs,
``quantize_affine_float8_non_decomposed`` / ``dequantize_affine_float8_non_decomposed``
map to ``IQuantizeLayer`` / ``IDequantizeLayer`` so both activations and weights
participate in FP8 GEMM fusion.

For FP4, ``torch.ops.tensorrt.dynamic_block_quantize_op.default`` nodes are converted
via the dynamic block quantize converter, which uses TRT's
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
   * - FP4 (NVFP4)
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
