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

.. code-block:: bash

    pip install -r ../requirements.txt
    python quantize_linear_fp8_woq.py
    python quantize_linear_fp8_static.py
    python torch_export_flux_fp8_woq.py
"""
