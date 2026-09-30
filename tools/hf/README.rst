Torch-TensorRT Edge-LLM
======================

An out-of-tree package on Torch-TensorRT owning Edge exporters,
``torch.ops.tensorrt_edge_llm.*``, and their ExecuTorch lowering. Bazel builds
compiled artifacts; CMake integrates native sources into applications. The
package can later move into Torch-TensorRT without changing the operator names.

Install alongside a compatible Torch-TensorRT build::

    pip install './tools/hf[executorch,pi05]'

PI0.5 also requires LeRobot. The numerical integration tests use LeRobot 0.6.2
and Transformers 5.5. LeRobot currently restricts PyTorch to <2.12, while this
branch uses a newer Torch-TensorRT/PyTorch build. Install the compatible LeRobot
source separately with ``--no-deps`` and supply its policy dependencies; the
``pi05`` extra deliberately does not resolve or replace your PyTorch installation.
Operator registration itself does not import LeRobot, Transformers or ExecuTorch.

PI0.5 operator contracts
-----------------------

Every embedded operator carries a persistent uint8 TensorRT payload and a
versioned metadata string. Python execution deserializes that payload after
``torch.export.load``; no exporter registry or engine-directory dependency is
needed. ExecuTorch lowers the same bytes through the standalone ``EdgeLLMBackend``
into Torch-TensorRT's existing C++ TensorRT engine backend.

* ``vision_tower(Tensor[] pixels, payload, metadata) -> Tensor[]``: one HWC
  input and one output. PI0.5 emits batch-major ``[B, cameras * tokens, H]``.
* ``fuse_prefix(vision, language, indices) -> prefix``: concatenate and gather
  into the policy prefix. This is an executable Python graph operator. The
  first native application performs the same packing on the host.
* ``llm_prefill(embeddings, attention_mask, position_ids, payload, metadata)``
  returns hidden states and fresh stacked prefix K/V of shape
  ``[layers, batch, kv_heads, prefix_tokens, head_dim]``. The PI0.5 runner uses
  bidirectional prefix attention and has no hidden autoregressive session.
* ``action_expert(x_t, timestep, prefix_k, prefix_v, position_ids,
  attention_mask, payload, metadata) -> velocity`` evaluates one PI0.5
  denoising step. Prefix K/V are read-only and reusable across all steps.

``execute_engine`` and ``scatter_image_tokens`` remain compatibility operators
for other exporters. Directory-backed ``execute_engine`` is not an embedded
native module contract.

Export and run PI0.5
-------------------

Use preprocessed LeRobot observations: normalized NCHW images, one boolean mask
per camera, language tokens and their mask, and float32 caller-provided noise.
Checkpoint processors must perform resize, normalization, tokenization and
output unnormalization. The exported policy returns padded model-space actions,
not physical robot commands.

.. code-block:: python

    from torch_tensorrt_edge_llm.pi05 import (
        export_pi05, prepare_pi05_sample, write_native_inputs,
    )

    # core is a loaded LeRobot PI05Policy.model, on CUDA in float32.
    core.paligemma_with_expert.precision = "float32"
    sample = prepare_pi05_sample(
        core, images, image_masks, tokens, token_mask, noise,
    )
    exported = export_pi05(
        core, sample, engine_dir="pi05/engines",
        cameras=len(images), num_steps=10,
    )
    actions = exported.program.module()(*sample)
    exported.save("pi05")  # runnable full-policy pi05.pt2
    exported.save("pi05", output_format="executorch")  # pi05.pte
    write_native_inputs(sample, "pi05/inputs")

The same path is available as ``EdgeExporter().export_pi05(core, observations,
EdgeConfig(engine_dir="pi05/engines"), num_steps=10)``. ``observations`` uses the
five preprocessed keys described by the checkpoint CLI below. The older
``EdgeExporter.export`` interface remains the component-velocity export path.
Set the core's precision flag as above to disable LeRobot's bfloat16 vision
autocast when converting a bfloat16 checkpoint to the initial float32 path.

The full ExportedProgram contains vision, prefix packing, prefill and the Euler
loop. ExecuTorch save creates three reusable methods: ``vision``, ``prefill`` and
``action_step``. The C++ application performs packing and the Euler loop so
there is one action engine and one delegate handle, regardless of step count.
The first application uses host inputs/outputs and TensorRT's native CUDA
staging. It disables non-CPU memory planning to avoid requiring ExecuTorch CUDA
copy kernels or its libTorch-based AOTI backend. CUDA streams still use the
shared ``extension_cuda`` interface.

Single embedded component programs can also use the existing public
``torch_tensorrt.save(program, path, output_format="executorch",
partitioners=[EdgeLLMPartitioner()])``. For the composed PI0.5 policy use the
multi-method save above; direct lowering of ``fuse_prefix`` and the whole Euler
graph is not implemented in this first stack.

A checkpoint CLI with strict required-weight loading is included::

    python tools/hf/examples/export_pi05.py \
        --checkpoint /path/to/pi05-checkpoint --inputs observations.pt \
        --output /tmp/pi05 --steps 10

``observations.pt`` is a tensor-only dictionary with ``images``, ``image_masks``,
``tokens``, ``token_mask`` and ``noise`` from the checkpoint's processors. The
CLI compares the exported policy with LeRobot using identical noise and writes
both program formats, native inputs and ``reference.pt``. Shapes are fixed.
Base PI0.5 is supported; MEM and RTC variants need separate contracts. The
initial native application contract is float32. Full checkpoint compilation
needs enough free GPU memory for weights and TensorRT builder workspace.

Build compiled artifacts with Bazel
----------------------------------

Run Bazel from this project's root (``tools/hf`` in this checkout)::

    bazel test //cpp:test_blob_header
    bazel build \
        --repo_env=TORCHTRT_NATIVE_ROOT=/path/to/torchtrt-native-sdk \
        --repo_env=EXECUTORCH_ROOT=/path/to/executorch \
        --repo_env=EXECUTORCH_LIBRARY=/path/to/libexecutorch_core.a \
        --repo_env=EXECUTORCH_EXTENSION_CUDA_LIBRARY=/path/to/libextension_cuda.so \
        --repo_env=TENSORRT_ROOT=/path/to/TensorRT \
        --repo_env=CUDA_ROOT=/path/to/cuda \
        //:executorch_backend_archive

``bazel-bin/libexecutorch_edge_llm_backend.a`` contains the standalone delegate
and EL01 parser. Applications also link the compatible Torch-TensorRT native
backend, ExecuTorch, CUDA and TensorRT. No companion Edge-LLM adapter library is
required for these engine-backed PI0.5 operators.

``TORCHTRT_NATIVE_ROOT`` contains ``include/torch_tensorrt/executorch/*.h`` and
``lib/libexecutorch_trt_backend.a`` from a compatible Torch-TensorRT native
backend build. ExecuTorch's root contains its ``runtime`` directory. TensorRT
and CUDA use their standard SDK include/lib layout. The standalone Bazel
repository imports these public headers and libraries lazily, so parser tests
do not need any native SDK. Consuming compiled SDKs also avoids depending on
Torch-TensorRT's source Bazel module assuming it is the repository root.

Application source builds with CMake
-----------------------------------

Add ExecuTorch and Torch-TensorRT's native backend first, then add this package's
``cpp`` directory and link ``tensorrt_edge_llm::executorch_backend``. It preserves
both backend registrations when static archives are linked. A complete source
example builds a generic component runner and a PI0.5 application::

    cmake -S tools/hf/examples/executorch_reference_runner -B build/pi05 \
        -DEXECUTORCH_SOURCE_DIR=/path/to/executorch \
        -DTORCHTRT_EXECUTORCH_SOURCE_DIR=/path/to/Torch-TensorRT/cpp/src/torch_tensorrt/executorch \
        -DTensorRT_ROOT=/path/to/TensorRT -DCUDAToolkit_ROOT=/path/to/cuda \
        -DEXECUTORCH_BUILD_PORTABLE_OPS=OFF
    cmake --build build/pi05 --target pi05_executorch_runner
    build/pi05/pi05_executorch_runner --model_path=/tmp/pi05/pi05.pte \
        --inputs_dir=/tmp/pi05/inputs --num_steps=10 --output_path=actions.bin

``actions.bin`` is contiguous float32 with the shape of the input noise. Keep
``num_steps`` equal to the export/reference step count. The native path links
no libTorch. CPU packing and per-step host staging are correctness-first;
GPU-resident orchestration and optimized precision profiles can follow.
For device-planned generic component programs, enable
``EDGELLM_BUILD_CUDA_AOTI_BACKEND`` and portable operators; that optional
ExecuTorch CUDA backend requires its normal libTorch build dependencies.

Validation
----------

Run ``python -m pytest tests`` from this project. LeRobot tests use real PI0.5
Gemma, AdaRMS and SigLIP layers with reduced dimensions and random weights. The
CUDA test compiles all three components, compares a full action rollout, saves
and reloads the full ExportedProgram in a fresh process, and inspects the three
native methods. Set ``EDGELLM_PI05_RUNNER`` to the built runner to also execute
the saved .pte and compare final C++ actions. Cases include batch size two,
a masked camera, different token masks, and unchanged prefix caches.

Reduced-model Python and native execution have been validated. A full pretrained
checkpoint and its observation/action processors have not yet been validated
with this stack.
