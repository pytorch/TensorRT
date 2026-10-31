# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Load real TensorRT programs in ways that have to fail, and check each fails with a clear error."""

import importlib.util
import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("executorch.exir")

import torch  # noqa: E402


@pytest.fixture(autouse=True)
def requires_runtime():
    # Decided at run time, not collection time, for the reason test_cuda_partitioner_composition.py
    # gives: remote GPU runners collect off the GPU host.
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device to build a TensorRT engine")
    if importlib.util.find_spec("torch_tensorrt_executorch_runtime") is None:
        pytest.skip(
            "needs the torch-tensorrt-executorch-runtime package to load the program"
        )


def _save(path, **kwargs):
    import torch_tensorrt

    torch.manual_seed(0)
    inputs = (torch.randn(4, 16, device="cuda"),)
    model = torch.nn.Linear(16, 16).eval().cuda()
    trt_gm = torch_tensorrt.dynamo.compile(
        torch.export.export(model, inputs), inputs=list(inputs), min_block_size=1
    )
    torch_tensorrt.save(
        trt_gm,
        str(path),
        output_format="executorch",
        retrace=False,
        arg_inputs=list(inputs),
        **kwargs,
    )


def test_loading_with_no_visible_gpu_fails_with_a_clear_error(tmp_path):
    from executorch.exir import ExecutorchBackendConfig

    pte = tmp_path / "model.pte"
    # Planned in host memory, so loading reaches the delegate. A device-planned buffer would stop
    # the load earlier, in ExecuTorch's CUDA allocator, which reports a missing GPU on its own.
    _save(
        pte,
        backend_config=ExecutorchBackendConfig(enable_non_cpu_memory_planning=False),
    )
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "import torch_tensorrt_executorch_runtime\n"
        "from executorch.runtime import Runtime\n"
        "program = Runtime.get().load_program(Path(sys.argv[1]))\n"
        "try:\n"
        "    program.load_method('forward')\n"
        "except RuntimeError as error:\n"
        "    print(error)\n"
        "    sys.exit(3)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(pte)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 3, (result.returncode, result.stderr[-2000:])
    output = result.stdout + result.stderr
    assert "CUDA is unavailable" in output
    # The backend reports itself unavailable, which ExecuTorch turns into NotFound (0x20).
    assert "0x20" in output or "0x:20" in output, output[-2000:]


def test_an_engine_tensorrt_cannot_read_names_both_tensorrt_versions(tmp_path, capfd):
    """A real version mismatch needs a second TensorRT install, so this corrupts the stored engine.

    TensorRT then refuses it on the same path it takes for an engine from another version, which
    is the path that has to name both versions.
    """
    import tensorrt
    import torch_tensorrt_executorch_runtime  # noqa: F401
    from executorch.exir._serialize._program import deserialize_pte_binary
    from executorch.runtime import Runtime

    pte = tmp_path / "model.pte"
    _save(pte)
    data = bytearray(pte.read_bytes())
    named = deserialize_pte_binary(bytes(data)).named_data
    (engine,) = [bytes(buffer) for buffer in named.buffers]
    start = data.find(engine)
    assert start > 0 and data.find(engine, start + 1) == -1
    data[start : start + 64] = bytes(64)
    # A stand-in export version of the same length, so the log can only have read it from the
    # program and not from the TensorRT that is loading it.
    exported = tensorrt.__version__.encode()
    stand_in = b".".join(b"9" * len(part) for part in exported.split(b"."))
    assert data.count(exported) == 1
    data[data.find(exported) : data.find(exported) + len(exported)] = stand_in
    pte.write_bytes(data)

    program = Runtime.get().load_program(pte)
    with pytest.raises(RuntimeError, match="0x:?23") as raised:
        program.load_method("forward")
    log = str(raised.value) + capfd.readouterr().err
    assert f"exported with TensorRT {stand_in.decode()}" in log, log
    runtime_version = ".".join(tensorrt.__version__.split(".")[:4])
    assert f"this runtime is TensorRT {runtime_version}" in log, log


def test_an_engine_with_a_pytorch_plugin_is_refused_at_save_and_the_advice_loads(
    tmp_path,
):
    """ScatterAdd runs PyTorch, which the runtime does not load, so the program could never load."""
    import tensorrt
    import torch_tensorrt

    if (
        tensorrt.get_plugin_registry().get_creator("ScatterAdd", "1", "torch_tensorrt")
        is None
    ):
        pytest.skip("needs Torch-TensorRT's ScatterAdd plugin")

    class AccumulateRows(torch.nn.Module):
        def forward(self, x, index, values, seen):
            # A float accumulate uses TensorRT's ScatterElements plugin, a bool one ScatterAdd.
            x = x.index_put((index,), values * 2, accumulate=True)
            seen = seen.index_put((index,), values > 0, accumulate=True)
            return x.index_put((index[:1],), values[:1]) + 1, seen

    inputs = (
        torch.zeros(16, 8, device="cuda"),
        torch.tensor([1, 3, 3, 7], device="cuda"),
        torch.ones(4, 8, device="cuda"),
        torch.zeros(16, 8, dtype=torch.bool, device="cuda"),
    )
    exported = torch.export.export(AccumulateRows(), inputs)
    pte = tmp_path / "model.pte"

    def compile_and_save(**settings):
        compiled = torch_tensorrt.dynamo.compile(
            exported, arg_inputs=inputs, min_block_size=1, **settings
        )
        torch_tensorrt.save(
            compiled,
            str(pte),
            output_format="executorch",
            retrace=False,
            arg_inputs=inputs,
        )
        return compiled

    with pytest.raises(RuntimeError, match="run PyTorch"):
        compile_and_save()
    assert not pte.exists()
    from torch_tensorrt.dynamo._engine_cache import DiskEngineCache

    cache = DiskEngineCache(str(tmp_path / "cache"), 1 << 30)
    # The second compile takes the engine from the cache.
    for _ in range(2):
        with pytest.raises(RuntimeError, match="run PyTorch"):
            compile_and_save(
                cache_built_engines=True,
                reuse_cached_engines=True,
                immutable_weights=False,
                custom_engine_cache=cache,
            )

    compiled = compile_and_save(fallback_pytorch_plugin_ops=True)
    # Only the bool accumulate leaves TensorRT.
    left_to_pytorch = [
        node.args[0].meta["val"].dtype
        for module in compiled.modules()
        if isinstance(module, torch.fx.GraphModule)
        for node in module.graph.nodes
        if node.target is torch.ops.aten.index_put.default
    ]
    assert left_to_pytorch == [torch.bool]
    script = (
        "import json\n"
        "import sys\n"
        "import numpy as np\n"
        "import torch_tensorrt_executorch_runtime\n"
        "from executorch.runtime import Runtime\n"
        "method = Runtime.get().load_program(sys.argv[1]).load_method('forward')\n"
        "inputs = [np.zeros((16, 8), np.float32), np.array([1, 3, 3, 7]),"
        " np.ones((4, 8), np.float32), np.zeros((16, 8), bool)]\n"
        "assert 'torch' not in sys.modules\n"
        "print(json.dumps([np.asarray(out).tolist() for out in method.execute(inputs)]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(pte)],
        env={**os.environ, "EXECUTORCH_PYBINDINGS_TENSOR_OUTPUT": "executorch"},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    expected = [out.cpu().tolist() for out in AccumulateRows()(*inputs)]
    assert json.loads(result.stdout.strip().splitlines()[-1]) == expected
