"""
TorchAO quantization
====================

Compile models quantized with `TorchAO <https://github.com/pytorch/ao>`_ using
the Torch-TensorRT Dynamo backend.

Weight-only FP8 keeps activations in BF16/FP16. Export emits
``dequantize_affine``, which Torch-TensorRT maps to TensorRT
``IDequantizeLayer`` so the engine can keep an FP8 weight constant.

Static FP8 also quantizes activations. Calibrate observers, then export
``quantize_affine_float8_non_decomposed`` / ``dequantize_affine_float8_non_decomposed``
so TensorRT can fuse Q/DQ into FP8 GEMMs.

Weight-only INT4 is the same DQ pattern with packed 4-bit integer weights.
Use **symmetric** group-wise INT4 (zero zero-point) and compile with
``immutable_weights=True`` so engine constants stay ``Datatype: Int4``.

Weight-only NVFP4 stores packed FP4 E2M1 weights with FP8 block scales.
Export emits ``dequantize_nvfp4``, which Torch-TensorRT maps to two-level
``IDequantizeLayer`` so the engine can keep ``Datatype: FP4E2M1``. This is
storage + DQ, not native FP4 MMA.

MXFP4 uses TorchAO ``MXDynamicActivationMXWeightConfig`` (there is no MXFP4
weight-only config). Export emits ``dequantize_mxfp4``, which Torch-TensorRT
maps to FP4 + E8M0 ``IDequantizeLayer`` (block size 32). Activations stay
BF16 in this path.

.. code-block:: bash

    pip install -r ../requirements.txt
    python quantize_linear_fp8_woq.py
    python quantize_linear_fp8_static.py
    python torch_export_flux_fp8_woq.py
    python quantize_linear_int4_woq.py
    python torch_export_flux_int4_woq.py
    python torch_export_qwen3_int4_woq.py
    python quantize_linear_nvfp4_woq.py
    python torch_export_flux_nvfp4_woq.py
    python quantize_linear_mxfp4.py
    python torch_export_flux_mxfp4.py
"""
