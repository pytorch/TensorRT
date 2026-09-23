.. _alpamayo_modelopt_fp8:

Alpamayo 1.5 ModelOpt FP8 Quantization
======================================

This guide follows the public
`Alpamayo 1.5 quantization recipe
<https://github.com/NVlabs/alpamayo-recipes/tree/main/recipes/alpamayo1_5_quant>`_
to post-training quantize the complete Alpamayo model with NVIDIA ModelOpt and
evaluate the resulting checkpoint with minADE.

Unlike an isolated expert smoke test, this workflow:

* loads the full Alpamayo vision-language-action model;
* calibrates with PhysicalAI driving clips;
* exercises both VLM rollout and diffusion paths;
* compresses fake-quant weights to real FP8 by default;
* saves ``modelopt_state.pth`` with the Hugging Face checkpoint;
* reloads the checkpoint and evaluates trajectory accuracy.

The commands below intentionally match the upstream recipe. After completing
this guide, continue with :ref:`alpamayo_fp8_edge_exporter` to build TensorRT
engines.

Prerequisites
-------------

The upstream recipe is tested with:

* NVIDIA RTX 5090 with CUDA 12;
* NVIDIA B300 with CUDA 13;
* Python 3.12;
* PyTorch 2.8.0;
* torchvision 0.23.0;
* NVIDIA ModelOpt 0.43.0.

Request access to:

* `Alpamayo-1.5-10B <https://huggingface.co/nvidia/Alpamayo-1.5-10B>`_
* `PhysicalAI-Autonomous-Vehicles
  <https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles>`_

Clone the official recipe:

.. code-block:: bash

   export YOUR_HOME=/path/to/your/workspace
   git clone https://github.com/NVlabs/alpamayo-recipes.git \
     "$YOUR_HOME/alpamayo-recipes"

1. Create the recipe environment
--------------------------------

.. code-block:: bash

   export UV_CACHE_DIR="$YOUR_HOME/.cache/uv"

   cd "$YOUR_HOME/alpamayo-recipes/recipes/alpamayo1_5_quant"
   uv venv am15_quant
   source am15_quant/bin/activate

   # Install all dependencies except flash-attn first so torch is available.
   uv sync --active --no-install-package flash-attn

   # Build flash-attn against the installed torch.
   MAX_JOBS=4 uv sync --active

Verify the pinned environment:

.. code-block:: bash

   python - <<'PY'
   import modelopt
   import torch
   import torchvision
   import transformers

   print("torch", torch.__version__)
   print("torchvision", torchvision.__version__)
   print("transformers", transformers.__version__)
   print("modelopt", modelopt.__version__)
   print("cuda available", torch.cuda.is_available())
   PY

2. Configure model, dataset, and cache paths
-----------------------------------------------

.. code-block:: bash

   export ALPAMAYO_WORKSPACE="$YOUR_HOME/alpamayo-recipes"
   export ALPAMAYO_MODEL_DIR="$YOUR_HOME/alpamayo_model_converted_from_hf"
   export ALPAMAYO_PAI_LOCAL_DIR="$YOUR_HOME/PAI_mini"
   export ALPAMAYO_LOG_DIR="$YOUR_HOME/alpamayo_logs"
   export HF_HOME="$YOUR_HOME/.cache/huggingface"

``ALPAMAYO_PAI_LOCAL_DIR`` must contain the PhysicalAI data expected by
``load_physical_aiavdataset``. The recipe directory already contains:

* ``0417_5k_train_set_for_calibration_25.10.parquet`` for calibration;
* ``1005_7cam_gold_eval_metadb_public.parquet`` for evaluation.

For offline execution after the assets are cached:

.. code-block:: bash

   export HF_HUB_OFFLINE=1
   export TRANSFORMERS_OFFLINE=1

3. Authenticate with Hugging Face
-------------------------------------

Authenticate interactively so the token is not written into shell history:

.. code-block:: bash

   hf auth login
   hf auth whoami

4. Quantize the full model to FP8
-------------------------------------

Run the upstream FP8 command:

.. code-block:: bash

   cd "$YOUR_HOME/alpamayo-recipes/recipes/alpamayo1_5_quant"
   source am15_quant/bin/activate

   uv run --active quantize.py \
     --quant_format=fp8 \
     --num_of_calib_clips=100 \
     --save_model_dir=./outputs

The calibration loop loads the requested PhysicalAI clips and calls
``sample_trajectories_from_data_with_vlm_rollout``. This observes activation
ranges through the VLM, expert, action projections, and diffusion-related
paths.

By default, ``quantize.py`` calls ``mtq.compress(model)`` before saving. The
checkpoint therefore contains real FP8 weights and is expected at:

.. code-block:: text

   outputs/alpamayo1.5_fp8_calib100/
   ├── config.json
   ├── modelopt_state.pth
   ├── model*.safetensors
   └── tokenizer and processor assets

Capture logs for a long-running calibration:

.. code-block:: bash

   nohup uv run --active quantize.py \
     --quant_format=fp8 \
     --num_of_calib_clips=100 \
     --save_model_dir=./outputs \
     > quantize_fp8.log 2>&1 &

   tail -f quantize_fp8.log

Check the result:

.. code-block:: bash

   export ALPAMAYO_FP8_CKPT="$PWD/outputs/alpamayo1.5_fp8_calib100"

   test -f "$ALPAMAYO_FP8_CKPT/config.json"
   test -f "$ALPAMAYO_FP8_CKPT/modelopt_state.pth"
   ls -lh "$ALPAMAYO_FP8_CKPT"

Fake-quant checkpoint option
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The upstream recipe also supports ``--fake_quant`` for downstream SDK
debugging. It preserves the original weights while saving Q/DQ state:

.. code-block:: bash

   uv run --active quantize.py \
     --quant_format=fp8 \
     --num_of_calib_clips=100 \
     --save_model_dir=./outputs \
     --fake_quant

This writes ``alpamayo1.5_fp8_calib100_fakequant``. It is not the default
workflow and does not provide the real-weight memory reduction of the
compressed checkpoint.

5. Evaluate the FP8 checkpoint
------------------------------

Run the official evaluation command:

.. code-block:: bash

   uv run --active eval.py \
     --ckpt ./outputs/alpamayo1.5_fp8_calib100

For a short smoke test before the complete evaluation:

.. code-block:: bash

   uv run --active eval.py \
     --ckpt ./outputs/alpamayo1.5_fp8_calib100 \
     --limit 10 \
     --num_traj_samples 6 \
     --seed 42 \
     --print_every 1

Capture a full evaluation in the background:

.. code-block:: bash

   nohup uv run --active eval.py \
     --ckpt ./outputs/alpamayo1.5_fp8_calib100 \
     > eval_fp8.log 2>&1 &

   tail -f eval_fp8.log

A correct run reports:

* the number of clip IDs loaded from the evaluation parquet;
* per-clip minADE and inference time;
* failed clips, if any;
* average minADE over successful clips;
* average evaluation time per clip.

Compare the FP8 result with a base-model run using the same settings:

.. code-block:: bash

   uv run --active eval.py \
     --ckpt nvidia/Alpamayo-1.5-10B \
     --limit 10 \
     --num_traj_samples 6 \
     --seed 42 \
     --print_every 1

Keep ``--limit``, ``--num_traj_samples``, and ``--seed`` identical when
comparing checkpoints.

6. Continue to TensorRT Edge export
-----------------------------------

The checkpoint consumed by the Edge exporter is:

.. code-block:: bash

   export ALPAMAYO_FP8_CKPT="$YOUR_HOME/alpamayo-recipes/recipes/alpamayo1_5_quant/outputs/alpamayo1.5_fp8_calib100"

Continue with :ref:`alpamayo_fp8_edge_exporter`. That guide creates a separate
export environment, restores ``modelopt_state.pth``, builds the Edge-LLM
plugin, compiles the vision/language/action engines, and checks component
parity.

Troubleshooting
---------------

``401`` or gated-repository error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Confirm access and run ``hf auth login`` again. Do not paste tokens into
commands or logs.

``No space left on device``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Move ``HF_HOME``, ``UV_CACHE_DIR``, and ``outputs`` to storage with enough
capacity for the base and quantized checkpoints.

FlashAttention build failure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install all other dependencies first, confirm PyTorch imports, then rerun
``MAX_JOBS=4 uv sync --active``.

ModelOpt state is not restored
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Confirm ``modelopt_state.pth`` is in the checkpoint directory and that the
loading process calls ``mto.enable_huggingface_checkpointing()`` before
``Alpamayo1_5.from_pretrained``.

Evaluation produces no successful clips
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Verify ``ALPAMAYO_PAI_LOCAL_DIR``, the parquet path, dataset access, and
``t0_us``. Test one known clip directly before starting the full evaluation.
