# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
.. _run_agentic:

Running an Auto-Tuner session on any workload
=============================================

Runs a ``torch_tensorrt.autotune`` tuning session on a workload factory,
``path/to/file.py:create_case`` called with JSON keyword arguments. It serves
every workload in this directory (decode, prefill, encoder, Qwen3.5 and
Nemotron); ``run_agentic_decode.py`` keeps the decode workload's own options.
Rerunning the command resumes the session. ``agentic_bench/README.md`` gives
the commands of the published benchmark sessions.

.. code-block:: bash

    python tools/llm/run_agentic.py --session /tmp/gemma3-prefill-2k \
        --workload tools/llm/agentic_prefill.py:create_case \
        --args '{"model": "unsloth/gemma-3-1b-it", "length": 2048}' \
        --accuracy top1_agreement '>=' 0.97 --max-trials 10

Accuracy limits not given with ``--accuracy`` use the workload's defaults.
``--planner llm`` asks the logged-in ``claude`` (or ``--provider codex``) CLI
to choose the experiments, with ``--guidance FILE`` as its guidance. It is
``python -m torch_tensorrt.autotune run`` with a budget of 10 trials by
default; every option is that command's.
"""

import sys

from torch_tensorrt.autotune.__main__ import main

if __name__ == "__main__":
    sys.exit(main(["run", "--max-trials", "10", *sys.argv[1:]]))
