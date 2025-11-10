.. _compile_groot:

Compiling Vision Language Action Models from Huggingface using Torch-TensorRT
================================================================================
This tutorial walks you through how to compile GR00T N1.5-3B, an open foundation model for generalized humanoid robot reasoning and skills learning, using Torch-TensorRT.
GR00T N1.5-3B is a 3 billion parameter model that combines visual perception with language understanding for robotics applications. 
The model is part of NVIDIA's Isaac-GR00T (General-purpose Robot 00 Technology) initiative, which aims to provide foundation models for humanoid robots and robotic manipulation tasks. It is a Vision-Language Action Model (VLA) that takes multimodal input, including language and images, to perform manipulation tasks in diverse environments.
Developers and researchers can post-train GR00T N1.5 with real or synthetic data for their specific humanoid robot or task.

Model Architecture
------------------
.. image:: /tutorials/images/groot.png

The schematic diagram is shown in the illustration above. Red, Green, Blue (RGB) camera frames are processed through a pre-trained vision transformer (SigLip2). Text is encoded by a pre-trained transformer (Qwen3).
The architecture handles a varying number of views per embodiment by concatenating image token embeddings from all frames into a sequence, followed by language token embeddings. Robot proprioception is encoded using a multi-layer perceptron (MLP) indexed by the embodiment ID. To handle variable-dimension proprio, inputs are padded to a configurable max length before feeding into the MLP. Actions are encoded and velocity predictions decoded by an MLP, one per unique embodiment.
To model proprioception and a sequence of actions conditioned on observations, Isaac GR00T N1.5-3B uses a flow matching transformer. The flow matching transformer interleaves self-attention over proprioception and actions with cross-attention to the vision and language embeddings. During training, the input actions are corrupted by randomly interpolating between the clean action vector and a gaussian noise vector. At inference time, the policy first samples a gaussian noise vector and iteratively reconstructs a continuous-value action using its velocity prediction.
In GR00T-N1.5, the MLP connector between the vision-language features and the diffusion-transformer (DiT) has been modified for improved performance on our sim benchmarks. Also, it was trained jointly with flow matching and world-modeling objectives.

Components of the model architecture include:

* **Vision Transformer (ViT)**
* **Text Transformer (LLM)**
* **Flow Matching Action Head** 

The Flow Matching Action Head includes:

* **VLM backbone processor (includes Self Attention Transformer + Layer Norm)**  
* **State encoder** 
* **Action encoder**
* **Action decoder**
* **Diffusion-Transformer (DiT)**


Inference with Torch-TensorRT on Jetson Thor 
--------------------------------------------

Torch-TensorRT is an inference compiler for PyTorch, designed to target NVIDIA GPUs through NVIDIA’s TensorRT Deep Learning Optimizer and Runtime. It bridges the flexibility of PyTorch with the high-performance execution of TensorRT by compiling models into optimized GPU-specific engines.
Torch-TensorRT supports both just-in-time (JIT) compilation via the torch.compile interface and ahead-of-time (AOT) workflows for deployment scenarios that demand reproducibility and low startup latency. It integrates seamlessly within the PyTorch ecosystem, enabling hybrid execution where optimized TensorRT kernels can run alongside standard PyTorch operations within the same model graph.
By applying a series of graph-level and kernel-level optimizations—including layer fusion, kernel auto-tuning, precision calibration, and dynamic tensor shape handling—Torch-TensorRT produces a specialized TensorRT engine tailored to the target GPU architecture. These optimizations maximize inference throughput and minimize latency, delivering substantial performance gains across both datacenter and edge platforms.
Torch-TensorRT is designed to operate seamlessly across a wide spectrum of NVIDIA hardware, ranging from high-performance datacenter GPUs (e.g., A100, H100, DGX Spark) to resource-constrained edge devices such as Jetson Thor. This versatility allows developers to deploy the same model efficiently across heterogeneous environments without modifying core code.

A key component of this integration is the MutableTorchTensorRTModule (MTTM) — a module provided by Torch-TensorRT. MTTM functions as a transparent and dynamic wrapper around standard PyTorch modules. It automatically intercepts and optimizes the module’s forward() function on-the-fly using TensorRT, while preserving the complete semantics and functionality of the original PyTorch model. This design ensures drop-in compatibility, enabling easy integration of Torch-TensorRT acceleration into complex frameworks, such as multi-stage inference pipelines or Hugging Face Transformers architectures, with minimal code changes.

Within the GR00T N1.5 model, each component is wrapped with MTTM to achieve optimized performance across all compute stages. This modular wrapping approach simplifies benchmarking, and selective optimization, ensuring that each subcomponent (e.g., the vision, language, or action head modules) benefits from TensorRT’s runtime-level acceleration.

To compile and run inference on the GR00T N1.5 model using Torch-TensorRT, follow the steps below:

* Build the deployment environment:
  Refer to the `Jetson deployment instructions <https://github.com/peri044/Isaac-GR00T/tree/a7725787273b62b9cff6c5863f328b1f4a3ead70/deployment_scripts#jetson-deployment>`_ to construct a Docker container that includes GR00T N1.5 and Torch-TensorRT, configured for the Thor platform.

* Compile and optimize the model:
  Follow the `inference setup instructions <https://github.com/peri044/Isaac-GR00T/tree/a7725787273b62b9cff6c5863f328b1f4a3ead70/deployment_scripts#23-inference-with-torch-tensorrt>`_ to prepare the runtime environment and initiate model compilation with Torch-TensorRT.

The primary entry point for model compilation and benchmarking is ``run_groot_torchtrt.py``, which provides an end-to-end workflow — from environment initialization to performance measurement. The script supports configurable arguments for precision modes (FP32, FP16, INT8), explicit type enforcement, and benchmarking strategies.

The ``fn_name`` argument allows users to target specific submodules of the GR00T N1.5 model for optimization, which is particularly useful for profiling and debugging individual components. For example, to benchmark the Vision Transformer module in FP16 precision mode, run:

.. code-block:: bash
  python run_groot_torchtrt.py \
    --precision FP16 \
    --use_fp32_acc \
    --use_explicit_typing \
    --fn_name all \
    --benchmark cuda_event

Preliminary results indicate that Torch-TensorRT achieves performance levels comparable to ONNX-TensorRT on the GR00T N1.5 model. However, certain submodules, particularly the LLM component  still present optimization opportunities to fully match ONNX-TensorRT performance
Support for Torch-TensorRT is currently available in this `PR <https://github.com/NVIDIA/Isaac-GR00T/pull/419>`_ and will be merged.


Requirements
^^^^^^^^^^^^

* Torch-TensorRT 2.9.0
* Transformers v4.51.3