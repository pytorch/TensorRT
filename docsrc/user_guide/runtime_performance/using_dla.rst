.. _using_dla:

DLA
===

NVIDIA's Deep Learning Accelerator is available on supported embedded platforms.
The legacy TorchScript DLA compiler and ``torchtrtc`` CLI have been removed.
Their compilation examples apply only to older Torch-TensorRT releases; consult
the versioned documentation when maintaining those deployments.

For new DLA deployments, use the TensorRT builder APIs for the TensorRT version
supported by your JetPack installation.
