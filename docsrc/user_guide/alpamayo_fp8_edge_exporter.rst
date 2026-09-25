.. _alpamayo_fp8_edge_exporter:

Alpamayo 1.5 ModelOpt FP8 Edge Export
=====================================

This guide walks through the complete workflow for:

1. quantizing ``nvidia/Alpamayo-1.5-10B`` to FP8 with NVIDIA ModelOpt;
2. checking the quantized checkpoint with Alpamayo evaluation;
3. building the TensorRT-Edge-LLM plugin;
4. exporting Alpamayo's vision, language, and action components to TensorRT;
5. checking eager-versus-TensorRT parity and the generated engine artifacts;
6. running the engines with TensorRT-Edge-LLM ``action_inference``.

The Edge exporter produces three engines using the binding names, paged KV
cache layout, sidecars, and tokenizer artifacts consumed by
TensorRT-Edge-LLM:

.. code-block:: text

   camera patches
        │
        ▼
   visual/visual.engine ──► visual token insertion + DeepStack packing
        │
        ▼
   llm.engine ──► paged language K/V cache
        │
        ▼
   action/action.engine ──► ten denoising steps in action_inference
        │
        ▼
   future trajectory [batch, 64, 2]

The action engine implements one Euler denoising step and aliases its per-layer
KV cache outputs to the corresponding inputs. The C++ action runner performs
the ten-step diffusion loop and copies each denoised result into the next
step's input.

.. warning::

   Alpamayo and the PhysicalAI dataset are gated. Request access before
   starting, and use ``hf auth login``. Do not put Hugging Face tokens directly
   in shell commands, documentation, or logs.

Validated component requirements
--------------------------------

Use Python 3.12. The quantization recipe and Edge exporter intentionally use
different environments because their dependency requirements differ.

.. list-table::
   :header-rows: 1
   :widths: 24 36 40

   * - Component
     - Quantization environment
     - Edge export environment
   * - PyTorch
     - ``2.8.0``
     - Version required by the installed Torch-TensorRT wheel
   * - Transformers
     - ``4.57.1``
     - ``>=5.4.0``
   * - ModelOpt
     - ``0.43.0``
     - ``>=0.44.0``
   * - TensorRT
     - Not used to create the checkpoint
     - Must match Torch-TensorRT and the Edge-LLM plugin
   * - CUDA
     - CUDA 12 or 13, matching PyTorch
     - Must match the plugin build

The exporter reads compressed FP8 weights and calibration scales directly from
the checkpoint safetensors. It does not restore ModelOpt's version-specific
``FP8QTensor`` Python wrappers. ``modelopt_state.pth`` remains useful for
official eager evaluation in the recipe environment.

Suggested source layout
-----------------------

The commands below use this layout:

.. code-block:: text

   /workspace/
   ├── alpamayo-recipes/
   │   └── recipes/alpamayo1_5_quant/
   ├── TensorRT-Edge-LLM/
   └── TensorRT-Torch/

Set paths once:

.. code-block:: bash

   export WORKSPACE=/workspace
   export ALPAMAYO_RECIPES="$WORKSPACE/alpamayo-recipes"
   export TORCH_TRT_ROOT="$WORKSPACE/TensorRT-Torch"
   export EDGE_LLM_ROOT="$WORKSPACE/TensorRT-Edge-LLM"
   export QUANT_DIR="$ALPAMAYO_RECIPES/recipes/alpamayo1_5_quant"
   export QUANT_OUTPUT="$QUANT_DIR/outputs"
   export ENGINE_DIR="$WORKSPACE/alpamayo_fp8_edge"
   export HF_HOME="$WORKSPACE/.cache/huggingface"

Use storage with enough space for the base checkpoint, calibration data,
quantized checkpoint, and TensorRT build artifacts. The base model is
approximately 22 GB in BF16; the FP8 parameter payload is approximately 11 GB.
TensorRT compilation also needs temporary host and GPU memory.

0. Obtain the sources and gated assets
--------------------------------------

Clone the required repositories:

.. code-block:: bash

   git clone https://github.com/NVlabs/alpamayo-recipes.git "$ALPAMAYO_RECIPES"
   git clone https://github.com/pytorch/TensorRT.git "$TORCH_TRT_ROOT"
   git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git "$EDGE_LLM_ROOT"

Check out the Torch-TensorRT revision containing Alpamayo Edge exporter support
when it has not yet landed on your default branch.

Request access to:

* `Alpamayo-1.5-10B <https://huggingface.co/nvidia/Alpamayo-1.5-10B>`_
* `PhysicalAI-Autonomous-Vehicles
  <https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles>`_

Authenticate interactively:

.. code-block:: bash

   hf auth login
   hf auth whoami

Configure the PhysicalAI dataset paths required by the Alpamayo recipe:

.. code-block:: bash

   export ALPAMAYO_WORKSPACE="$ALPAMAYO_RECIPES"
   export ALPAMAYO_MODEL_DIR="$WORKSPACE/alpamayo_model_converted_from_hf"
   export ALPAMAYO_PAI_LOCAL_DIR="$WORKSPACE/PAI_mini"
   export ALPAMAYO_LOG_DIR="$WORKSPACE/alpamayo_logs"

Follow the Alpamayo repository's dataset setup instructions to populate
``ALPAMAYO_PAI_LOCAL_DIR`` and obtain the calibration parquet file referenced by
the recipe.

1. Create the ModelOpt quantization environment
-----------------------------------------------

Create the recipe environment with ``uv``:

.. code-block:: bash

   export UV_CACHE_DIR="$WORKSPACE/.cache/uv"
   cd "$QUANT_DIR"

   uv venv am15_quant
   source am15_quant/bin/activate

   # Install torch first, then build flash-attn against it.
   uv sync --active --no-install-package flash-attn
   MAX_JOBS=4 uv sync --active

Verify the important versions:

.. code-block:: bash

   python - <<'PY'
   import modelopt
   import torch
   import transformers

   print("torch", torch.__version__)
   print("transformers", transformers.__version__)
   print("modelopt", modelopt.__version__)
   print("cuda available", torch.cuda.is_available())
   PY

The recipe is defined for Python 3.12, PyTorch 2.8, Transformers 4.57.1, and
ModelOpt 0.43.

2. Run the ModelOpt Alpamayo FP8 example
----------------------------------------

This workflow uses the public
`Alpamayo 1.5 ModelOpt quantization example
<https://github.com/NVlabs/alpamayo-recipes/tree/main/recipes/alpamayo1_5_quant>`_.
Its ``quantize.py``, ``eval.py``, calibration parquet, and pinned ``uv``
environment are the source of truth for checkpoint creation and evaluation.
The quantization procedure is also documented separately in
:ref:`alpamayo_modelopt_fp8`. The steps below use the same full-model
PhysicalAI calibration and compressed FP8 output, then continue into
Torch-TensorRT Edge export.

.. code-block:: bash

   cd "$QUANT_DIR"
   source am15_quant/bin/activate

   uv run --active quantize.py \
     --quant_format=fp8 \
     --num_of_calib_clips=100 \
     --save_model_dir=./outputs

The expected output directory is:

.. code-block:: text

   outputs/alpamayo1.5_fp8_calib100/
   ├── config.json
   ├── modelopt_state.pth
   ├── model*.safetensors
   └── tokenizer and processor assets

The calibration loop exercises the full VLM rollout and diffusion path with
the selected PhysicalAI clips. By default, ``quantize.py`` calls
``mtq.compress(model)`` before saving, so this checkpoint contains real FP8
weights.

Confirm that ModelOpt state was saved:

.. code-block:: bash

   export ALPAMAYO_FP8_CKPT="$QUANT_DIR/outputs/alpamayo1.5_fp8_calib100"

   test -f "$ALPAMAYO_FP8_CKPT/config.json"
   test -f "$ALPAMAYO_FP8_CKPT/modelopt_state.pth"
   ls -lh "$ALPAMAYO_FP8_CKPT"

For a long-running calibration, use the upstream background command:

.. code-block:: bash

   nohup uv run --active quantize.py \
     --quant_format=fp8 \
     --num_of_calib_clips=100 \
     --save_model_dir=./outputs \
     > quantize_fp8.log 2>&1 &

3. Evaluate the quantized checkpoint
------------------------------------

Run the official evaluation command:

.. code-block:: bash

   cd "$QUANT_DIR"
   source am15_quant/bin/activate

   uv run --active eval.py \
     --ckpt ./outputs/alpamayo1.5_fp8_calib100

For a shorter smoke test before the complete evaluation:

.. code-block:: bash

   uv run --active eval.py \
     --ckpt "$ALPAMAYO_FP8_CKPT" \
     --limit 10 \
     --num_traj_samples 6 \
     --seed 42 \
     --print_every 1

The evaluation should:

* restore ``modelopt_state.pth`` without missing-module errors;
* print the quantization summary for a non-base checkpoint;
* report finite per-clip minADE values;
* finish with average minADE and average evaluation time.

Increase ``--limit`` after the smoke test to obtain the customer acceptance
metric.

4. Prepare the Edge export environment
--------------------------------------

Deactivate the quantization environment and activate or create an environment
containing the Torch-TensorRT build used for export:

.. code-block:: bash

   deactivate 2>/dev/null || true
   source /path/to/edge-export/bin/activate

Install the model and exporter dependencies:

.. code-block:: bash

   python -m pip install \
     "transformers>=5.4.0" \
     "nvidia-modelopt[hf]>=0.44.0" \
     physical-ai-av \
     einops hydra-core pillow

   python -m pip install \
     "git+https://github.com/NVlabs/alpamayo1.5.git"

Install a Torch-TensorRT wheel built for this environment, or build/install
Torch-TensorRT from the selected source revision. Confirm that importing
``torch_tensorrt`` resolves to the installed package:

.. code-block:: bash

   cd "$TORCH_TRT_ROOT"

   # tools/hf contains the experimental exporter. Do not add $TORCH_TRT_ROOT/py
   # unless the source tree has been built and generated _version.py.
   export PYTHONPATH="$TORCH_TRT_ROOT/tools/hf:$EDGE_LLM_ROOT"

   python - <<'PY'
   import modelopt
   import torch
   import torch_tensorrt
   import transformers

   print("torch", torch.__version__)
   print("torch_tensorrt", torch_tensorrt.__version__)
   print("torch_tensorrt path", torch_tensorrt.__file__)
   print("transformers", transformers.__version__)
   print("modelopt", modelopt.__version__)
   PY

``torch_tensorrt.__file__`` should point to ``site-packages`` unless you built
the source tree in place.

5. Build the TensorRT-Edge-LLM plugin
---------------------------------------------

The language engine contains TensorRT-Edge-LLM attention plugins. Build the
plugin and ``action_inference`` against the same CUDA and TensorRT major
versions used by Torch-TensorRT. Build engines on the GPU architecture where
they will run; TensorRT plans are not portable between RTX 5090 (SM120) and
DRIVE AGX Thor (SM110).

Initialize dependencies:

.. code-block:: bash

   cd "$EDGE_LLM_ROOT"
   git submodule update --init --recursive

Install the CuTe DSL build dependencies required by the language prefill
kernels:

.. code-block:: bash

   python -m pip install \
     "nvidia-cutlass-dsl[cu13]==4.7.0" \
     "cupy-cuda13x==13.6.0" \
     cuda-python

Generate the FMHA artifact for the current GPU. On Thor this auto-detects
SM110:

.. code-block:: bash

   python kernelSrcs/build_cutedsl.py \
     --kernels fmha \
     --clean

Configure the CUDA 13 / TensorRT 11 build:

.. code-block:: bash

   export CUDA_HOME=/usr/local/cuda-13.0
   export PATH="$CUDA_HOME/bin:$PATH"

   cmake -S . -B build-alpamayo-export \
     -DCMAKE_BUILD_TYPE=Release \
     -DTRT_PACKAGE_DIR=/usr \
     -DCUDA_CTK_VERSION=13.0 \
     -DENABLE_CUTE_DSL=fmha \
     -DBUILD_UNIT_TESTS=OFF \
     -DBUILD_PYTHON_BINDINGS=OFF

   cmake --build build-alpamayo-export \
     --target NvInfer_edgellm_plugin action_inference \
     -j"$(nproc)"

If TensorRT headers and libraries are installed under a separate SDK root, use
that path for ``TRT_PACKAGE_DIR``. It must contain ``include/NvInfer.h`` and a
``lib`` or ``lib64`` directory containing ``libnvinfer.so``.

Set and verify the plugin:

.. code-block:: bash

   export EDGE_LLM_PLUGIN_SO="$EDGE_LLM_ROOT/build-alpamayo-export/libNvInfer_edgellm_plugin.so.1.0"
   export LD_LIBRARY_PATH="$(dirname "$EDGE_LLM_PLUGIN_SO"):${LD_LIBRARY_PATH:-}"

   test -f "$EDGE_LLM_PLUGIN_SO"

   python - <<'PY'
   import ctypes
   import os

   path = os.environ["EDGE_LLM_PLUGIN_SO"]
   ctypes.CDLL(path)
   print("loaded", path)
   PY

Use the CUDA and TensorRT versions available on your target system rather than
copying the example values blindly.

.. warning::

   Do not build the current plugin with ``ENABLE_CUTE_DSL=OFF`` when using
   ``AttentionPlugin`` language prefill. Such a build may contain decode-only
   XQA support and fail at runtime with ``selected prefill kernel is
   unavailable``. The required FMHA variant must support both the model's
   ``head_dim`` (128 for Alpamayo) and the target SM.

6. Choose an export sample
--------------------------

The exporter uses one PhysicalAI clip to construct realistic image, language,
trajectory-history, mRoPE, DeepStack, and action inputs. Select a clip from the
calibration parquet:

.. code-block:: bash

   cd "$QUANT_DIR"
   source am15_quant/bin/activate

   export ALPAMAYO_CLIP_ID="$(
     python - <<'PY'
   from alpamayo1_5_quant.utils import read_clip_ids_from_parquet

   clips = read_clip_ids_from_parquet(
       "0417_5k_train_set_for_calibration_25.10.parquet"
   )
   print(clips[0])
   PY
   )"

   echo "$ALPAMAYO_CLIP_ID"

Return to the Edge export environment before running the exporter.

7. Export Alpamayo to TensorRT
------------------------------

Load the plugin and run the unified exporter:

.. code-block:: bash

   cd "$TORCH_TRT_ROOT"
   source /path/to/edge-export/bin/activate

   export PYTHONPATH="$TORCH_TRT_ROOT/tools/hf:$EDGE_LLM_ROOT"
   export EDGE_LLM_PLUGIN_SO="$EDGE_LLM_ROOT/build-alpamayo-export/libNvInfer_edgellm_plugin.so.1.0"
   export LD_LIBRARY_PATH="$(dirname "$EDGE_LLM_PLUGIN_SO"):${LD_LIBRARY_PATH:-}"

   rm -rf "$ENGINE_DIR"

   python tools/hf/run_export.py alpamayo \
     --checkpoint "$ALPAMAYO_FP8_CKPT" \
     --clip-id "$ALPAMAYO_CLIP_ID" \
     --t0-us 5100000 \
     --engine-dir "$ENGINE_DIR" \
     --max-seq-len 4096 \
     --dtype float16 \
     --device cuda:0

During a successful run, the exporter:

1. loads 625 compressed FP8 linears and their ModelOpt calibration scales
   directly from safetensors;
2. loads the Alpamayo sample clip;
3. captures eager vision embeddings, language logits, and one Edge-compatible
   denoising step;
4. temporarily patches the original Qwen3-VL and Alpamayo ``forward`` methods;
5. compiles engines using the exact Edge runtime bindings;
6. restores the original class methods;
7. executes each serialized engine and prints eager-versus-TensorRT parity;
8. writes the tokenizer, processed chat template, embedding table, visual
   processor, and runtime configuration sidecars.

TensorRT autotuning can remain silent for several minutes. High CPU or GPU
utilization during this period usually means the build is still progressing.

8. Validate the result
----------------------

The runner prints:

* the engine mapping;
* eager-versus-TensorRT parity for each component;
* eager and TensorRT timings;
* the one-step denoised action shape and statistics.

The final tensor should have shape:

.. code-block:: text

   denoised_trajectory (1, 64, 2)

Inspect the output tree:

.. code-block:: bash

   ls -lh "$ENGINE_DIR"
   ls -lh "$ENGINE_DIR"/visual
   ls -lh "$ENGINE_DIR"/action

Expected layout:

.. code-block:: text

   $ENGINE_DIR/
   ├── llm.engine
   ├── config.json
   ├── embedding.safetensors
   ├── tokenizer.json
   ├── tokenizer_config.json
   ├── processed_chat_template.json
   ├── visual/
   │   ├── visual.engine
   │   ├── config.json
   │   └── preprocessor_config.json
   └── action/
       ├── action.engine
       └── config.json

The root ``config.json`` follows ``LLMEngineConfig`` and records paged KV,
DeepStack, RoPE, and optimization-profile metadata. The visual and action
sidecars follow the corresponding C++ runner contracts.

Parity interpretation
^^^^^^^^^^^^^^^^^^^^^

Review component parity independently:

* vision compares merged Qwen3-VL image features;
* language compares final logits from multimodal prefill;
* action compares one denoised trajectory step with identical noise, timestep
  interval, position IDs, RoPE, and KV caches.

Do not accept NaN/Inf values. Investigate large errors before measuring
performance. FP8 tolerances are necessarily looser than FP16, but a result that
is effectively uncorrelated with eager output indicates a packing, mRoPE,
DeepStack, mask, or checkpoint-restore mismatch.

9. Run with action_inference
----------------------------

Set the plugin loaded by the C++ runtime:

.. code-block:: bash

   cd "$EDGE_LLM_ROOT"

   export EDGELLM_PLUGIN_PATH="$EDGE_LLM_ROOT/build-alpamayo-export/libNvInfer_edgellm_plugin.so.1.0"
   export LD_LIBRARY_PATH="$(dirname "$EDGELLM_PLUGIN_PATH"):${LD_LIBRARY_PATH:-}"

Prepare an input JSON containing:

* the same number of camera images supported by the visual engine profile;
* a trajectory history;
* the Alpamayo system and user prompts;
* generation settings such as ``max_generate_length`` and ``temperature``.

Set its path and run:

.. code-block:: bash

   export INPUT_JSON=/path/to/input_action.json
   export OUTPUT_JSON="$ENGINE_DIR/output.json"

   ./build-alpamayo-export/examples/multimodal/action_inference \
     --engineDir="$ENGINE_DIR" \
     --multimodalEngineDir="$ENGINE_DIR" \
     --checkpointDir="$ALPAMAYO_FP8_CKPT" \
     --inputFile="$INPUT_JSON" \
     --outputFile="$OUTPUT_JSON" \
     --maxGenerateLength=128 \
     --noiseSeed=42 \
     --dumpOutput

The runtime:

1. loads ``llm.engine`` and the paged KV configuration;
2. preprocesses images and executes ``visual/visual.engine``;
3. runs language prefill and autoregressive decode;
4. gathers the language paged KV cache into the action runner's head-major
   cache layout;
5. executes ``action/action.engine`` for ten denoising steps;
6. writes the future trajectory to ``OUTPUT_JSON``.

Use the same clip, generation parameters, trajectory sample count, and random
seed as eager evaluation when comparing accuracy. Compare final trajectories
or minADE rather than requiring bitwise equality.

10. Run focused tests
---------------------

From the Torch-TensorRT checkout:

.. code-block:: bash

   cd "$TORCH_TRT_ROOT"
   export PYTHONPATH="$TORCH_TRT_ROOT/tools/hf"

   python -m pytest \
     -o addopts='' \
     tools/hf/exporters/tests/test_edge_exporter.py \
     -k alpamayo \
     -q

These tests cover registration, Edge action and KV-cache patch wiring,
visual-token packing, ModelOpt checkpoint loading, and multimodal
generation-position extension. They do not replace the GPU export, parity run,
or target ``action_inference`` test.

End-to-end checklist
--------------------

1. [ ] Access granted for the Alpamayo model and PhysicalAI dataset.
2. [ ] Quantization environment uses Python 3.12 and the recipe-pinned packages.
3. [ ] FP8 checkpoint contains ``config.json`` and ``modelopt_state.pth``.
4. [ ] Quantized checkpoint produces finite minADE on a small evaluation subset.
5. [ ] Edge environment imports the intended Torch-TensorRT installation.
6. [ ] Edge environment uses Transformers 5.4 or newer and ModelOpt 0.44 or newer.
7. [ ] Edge-LLM plugin is built against matching CUDA/TensorRT versions.
8. [ ] ``ctypes.CDLL(EDGE_LLM_PLUGIN_SO)`` succeeds.
9. [ ] ``PYTHONPATH`` contains ``tools/hf`` but does not accidentally shadow
   the installed Torch-TensorRT package with an unbuilt ``py/`` directory.
10. [ ] A valid PhysicalAI clip ID and ``t0_us`` are selected.
11. [ ] Vision, language, and action engines are written.
12. [ ] Component parity is finite and within the acceptance threshold.
13. [ ] ``action_inference`` initializes the language, visual, and action
    runners.
14. [ ] Final denoised trajectory output has shape ``[1, 64, 2]``.
15. [ ] Target minADE is within the customer acceptance threshold.

Troubleshooting
---------------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Symptom
     - Resolution
   * - ``No module named torch_tensorrt._version``
     - An unbuilt source ``py/`` directory is shadowing the installed package.
       Use the installed package with
       ``PYTHONPATH=$TORCH_TRT_ROOT/tools/hf``, or build/install the source tree
       before adding ``$TORCH_TRT_ROOT/py``.
   * - ``Set EDGE_LLM_PLUGIN_SO ...``
     - Build the Edge-LLM plugin, set the environment variable to the real
       ``.so`` file, and add its directory to ``LD_LIBRARY_PATH``.
   * - ``Plugin missing``
     - A documentation placeholder was copied literally. Use the actual build
       artifact path and check it with ``test -f``.
   * - ``undefined symbol`` while loading the plugin
     - The plugin and Python environment use different CUDA or TensorRT
       versions. Rebuild the plugin against the same TensorRT SDK.
   * - ``No module named transformers.exporters``
     - Upgrade the Edge environment to Transformers 5.4 or newer. Do not use
       the recipe's Transformers 4.57 environment for Edge export.
   * - Compressed checkpoint reports no FP8 linears
     - Confirm the safetensors contain ``torch.float8_e4m3fn`` weights,
       ``weight_quantizer._scale``, and ``input_quantizer._amax`` tensors. The
       Edge loader reads these tensors directly.
   * - Real FP8 checkpoint fails to restore or export
     - Retry with the ``--fake_quant`` checkpoint first. Real compressed
       checkpoint restore is experimental.
   * - Dataset access, 401, or missing clip
     - Confirm gated access with ``hf auth whoami`` and verify
       ``ALPAMAYO_PAI_LOCAL_DIR`` plus the requested clip ID.
   * - Image token count does not match visual features
     - Use the same processor, image set, ``image_grid_thw``, and checkpoint
       tokenizer. Do not mix processor assets from another Alpamayo run.
   * - Language parity is poor while vision is good
     - Check multimodal position IDs, RoPE delta extension, image-token masks,
       and dense DeepStack insertion.
   * - Action parity is poor
     - Check the per-layer K/V aliases, non-causal length mask, FP32
       ``time_steps_t0/t1``, RoPE positions, and action dimensions
       ``[64, 2]``.
   * - ``selected prefill kernel is unavailable``
     - The loaded Edge plugin has decode support but no FMHA prefill kernel for
       the model ``head_dim`` and target SM. Rebuild with the matching CuTe DSL
       FMHA artifact, use a supported prebuilt plugin, or select a supported
       TensorRT-native attention path. An ``ENABLE_CUTE_DSL=OFF`` build is not
       sufficient for this language engine.
   * - Runtime initializes but rejects an optimization profile
     - Rebuild the language engine with both prefill and decode profiles.
       Initial prefill requires the zero-length
       ``kvcache_start_index`` sentinel; decode uses shape ``[batch]``.
   * - CUDA out of memory during compile
     - Stop unrelated GPU workloads, use a larger-memory GPU, and keep
       checkpoint/cache/build directories off the root filesystem.
   * - TensorRT build appears hung
     - TensorRT may be autotuning without console output. Check CPU/GPU
       utilization before interrupting it.

Security and cleanup
--------------------

Do not export tokens directly in commands that may be captured in shell history.
If a token appears in logs, revoke it immediately and create a replacement.

Engine files are specific to the TensorRT version, plugin build, GPU
architecture, shapes, and compilation settings used to produce them. Rebuild
the engines after changing any of those inputs.
