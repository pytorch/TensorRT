# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import io
import sys
from pathlib import Path

import torch
from test_000_runtime_cache import ConvModel, _compile
from torch_tensorrt.runtime import runtime_cache


def main():
    path, payload = sys.argv[1:]
    state = torch.load(payload, weights_only=True)
    model = ConvModel().eval().cuda()
    model.load_state_dict(state["state"])
    inputs = [state["input"].cuda()]
    compiled = _compile(model, inputs)
    expected_bytes = Path(path).read_bytes()
    with torch.no_grad():
        with runtime_cache(compiled, path) as cache:
            before = io.BytesIO()
            cache.save_to_stream(before)
            assert before.getvalue() == expected_bytes, "The cache did not load."
            torch.testing.assert_close(
                compiled(*inputs), model(*inputs), rtol=1e-3, atol=1e-3
            )
            assert cache.has_cache(), "The engine did not attach the cache."


if __name__ == "__main__":
    main()
