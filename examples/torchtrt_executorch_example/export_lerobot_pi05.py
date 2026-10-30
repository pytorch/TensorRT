# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

r"""
.. _executorch_lerobot_pi05:

Deploying a LeRobot Policy (pi0.5) with ExecuTorch and TensorRT
===============================================================

`LeRobot <https://github.com/huggingface/lerobot>`_'s pi0.5 is a vision-language-action policy: from a
robot's camera images and a task in words, it predicts the robot's next 50 actions. This example
compiles ``lerobot/pi05_base`` (3.6 billion parameters) with Torch-TensorRT and saves it as one
ExecuTorch program, ``pi05.pte``. The program runs a whole chunk of actions in one TensorRT engine: the
cameras, the prompt, and the 10 denoising steps. The script then loads the program with ExecuTorch's
runtime, checks it against the PyTorch policy, and times both.

LeRobot's own processors stay in Python around the program: they normalize the observation and
tokenize the task before it, and unnormalize the actions after it. The program takes inputs of one
shape: a batch of 1, camera frames of 224x224 (resize yours first), and the prompt padded to 200
tokens, as LeRobot's preprocessor pads it.

Prerequisites
-------------
Python 3.12 or newer, which LeRobot requires. On one A40 the script takes about 8 minutes, 3 of them
compiling, and peaks at 24 GB of host memory and about 24 GB of GPU memory. It downloads the 14.5 GB
checkpoint and writes a 7.1 GB program. A chunk of 50 actions then takes 104 ms, against about 360 ms
in PyTorch.

Install Torch-TensorRT and its ExecuTorch delegate, ``torch_tensorrt_executorch_runtime``. The delegate
brings the exact ExecuTorch build it was compiled against, a nightly, hence the nightly index. Then
install LeRobot without its dependencies, because it pins an older PyTorch, and add the ones pi0.5
needs. Keep transformers in LeRobot's range: outside it, LeRobot can fail to load the weights and only
warn, leaving the policy randomly initialized.

.. code-block:: bash

    pip install "torch_tensorrt==2.15.0" "torch_tensorrt_executorch_runtime==0.2.0" "torchvision==0.30.0" \
        --extra-index-url https://download.pytorch.org/whl/cu132 \
        --extra-index-url https://download.pytorch.org/whl/nightly/cu132
    pip install --no-deps "lerobot==0.6.1"
    pip install "transformers>=5.4,<5.6" "draccus>=0.11.6,<0.12" "einops>=0.8,<0.9" "huggingface-hub>=1.6,<2" \
        "opencv-python-headless>=4.9,<4.14" "gymnasium>=1.1.1,<2" "scipy>=1.14,<2"

pi0.5 tokenizes the task with PaliGemma's tokenizer, which is gated on Hugging Face: accept the license
of `google/paligemma-3b-pt-224 <https://huggingface.co/google/paligemma-3b-pt-224>`_, then run
``hf auth login``.

Run
---

.. code-block:: bash

    python export_lerobot_pi05.py
"""

import argparse
import statistics
import sys
import time

import torch
import torch_tensorrt
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


class ActionChunk(torch.nn.Module):
    """The policy's action-chunk prediction, with tensors in and out.

    Inputs: the camera images stacked as (cameras, batch, 3, height, width) in [0, 1], the prompt's
    token ids and attention mask from LeRobot's preprocessor, and the starting noise of the denoising.
    Output: the normalized actions, (batch, 50, action dimension).
    """

    def __init__(self, policy: PI05Policy) -> None:
        super().__init__()
        self.policy = policy
        self.image_keys = list(policy.config.image_features)

    def forward(self, images, tokens, attention_mask, noise):
        batch = dict(zip(self.image_keys, images.unbind(0), strict=True))
        batch[OBS_LANGUAGE_TOKENS] = tokens
        batch[OBS_LANGUAGE_ATTENTION_MASK] = attention_mask
        return self.policy.predict_action_chunk(batch, noise=noise)


def median_ms(run, repeats: int = 20) -> float:
    run()  # warm up
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        run()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1e3)
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy", default="lerobot/pi05_base", help="LeRobot pi0.5 checkpoint"
    )
    parser.add_argument(
        "--task", default="pick up the red block", help="the task, in words"
    )
    parser.add_argument(
        "--model_path", default="pi05.pte", help="path to save the .pte"
    )
    args = parser.parse_args()

    # The policy in bfloat16, on the GPU, with LeRobot's processors.
    config = PreTrainedConfig.from_pretrained(args.policy)
    config.dtype = "bfloat16"
    config.device = "cuda"
    policy = PI05Policy.from_pretrained(args.policy, config=config).eval()
    preprocess, postprocess = make_pre_post_processors(config, args.policy)

    # A robot's observation: camera frames in [0, 1], the joint state, and the task. Random here.
    torch.manual_seed(0)
    observation = {
        key: torch.rand(3, *config.image_resolution) for key in config.image_features
    }
    observation[OBS_STATE] = torch.rand(config.input_features[OBS_STATE].shape[0])
    observation["task"] = args.task
    batch = preprocess(observation)

    images = torch.stack([batch[key] for key in config.image_features]).cuda()
    noise = torch.randn(1, config.chunk_size, config.max_action_dim, device="cuda")
    inputs = (
        images,
        batch[OBS_LANGUAGE_TOKENS].cuda(),
        batch[OBS_LANGUAGE_ATTENTION_MASK].cuda(),
        noise,
    )

    model = ActionChunk(policy)
    with torch.no_grad():
        expected = model(*inputs)

        # Export the whole chunk, the 10 denoising steps unrolled, compile it, and save the program.
        exported = torch.export.export(model, inputs)
        start = time.perf_counter()
        trt_model = torch_tensorrt.dynamo.compile(
            exported, arg_inputs=inputs, min_block_size=1
        )
        print(f"compiled in {time.perf_counter() - start:.0f} s")
        torch_tensorrt.save(trt_model, args.model_path, output_format="executorch")
        print(f"saved {args.model_path}")

    # Run the saved program. Importing torch_tensorrt_executorch_runtime registers the TensorRT
    # delegate with ExecuTorch; nothing else in this part needs Torch-TensorRT. The program takes CPU
    # tensors and copies them to the GPU itself (export_device_resident.py shows how to skip that).
    import torch_tensorrt_executorch_runtime  # noqa: F401
    from executorch.runtime import Runtime

    program = Runtime.get().load_program(args.model_path)
    method = program.load_method("forward")
    cpu_inputs = [t.cpu() for t in inputs]
    actions = method.execute(cpu_inputs)[0]

    error = (actions.float() - expected.float().cpu()).abs().max().item()
    print(f"max difference from PyTorch: {error:.4f}")
    if error > 0.1:
        sys.exit("the program's actions differ from the PyTorch policy's")
    robot_actions = postprocess(actions)
    print(f"actions for the robot: {tuple(robot_actions.shape)}")

    with torch.no_grad():
        pytorch_ms = median_ms(lambda: model(*inputs))
    executorch_ms = median_ms(lambda: method.execute(cpu_inputs))
    print(
        f"one chunk: PyTorch {pytorch_ms:.1f} ms, ExecuTorch + TensorRT {executorch_ms:.1f} ms"
    )


if __name__ == "__main__":
    main()
