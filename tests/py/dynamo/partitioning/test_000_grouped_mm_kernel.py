# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import subprocess
import sys
import tempfile
import textwrap
from unittest import mock

import torch
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo.conversion import aten_ops_converters, impl


class Experts(torch.nn.Module):
    """Two grouped matmuls around an activation, as in a mixture-of-experts layer."""

    def __init__(self, experts, k, n, dtype):
        super().__init__()
        self.up = torch.nn.Parameter(torch.randn(experts, k, n, dtype=dtype) * 0.05)
        self.down = torch.nn.Parameter(torch.randn(experts, n, k, dtype=dtype) * 0.05)

    def forward(self, x, offs):
        h = torch.ops.aten._grouped_mm.default(torch.relu(x), self.up, offs)
        return torch.ops.aten._grouped_mm.default(torch.relu(h), self.down, offs)


class GroupedMM(torch.nn.Module):
    def __init__(self, groups, k, n):
        super().__init__()
        self.w = torch.nn.Parameter(
            torch.randn(groups, k, n, dtype=torch.bfloat16) * 0.05
        )

    def forward(self, x, offs):
        return torch.ops.aten._grouped_mm.default(x, self.w, offs)


def _offsets(counts):
    return torch.cumsum(torch.tensor(counts), 0).to(torch.int32).cuda()


def _kernel_spy():
    """Counts the ops the grouped GEMM kernel converter takes."""
    return mock.patch.object(
        impl.moe, "grouped_mm_kernel", wraps=impl.moe.grouped_mm_kernel
    )


def _in_pytorch(compiled):
    return [
        node.target
        for name, sub in compiled.named_children()
        if name.startswith("_run_on_gpu")
        for node in sub.graph.nodes
        if node.op == "call_function"
    ]


# Runs the engine in a fresh process that loads only TensorRT and PyTorch, as a deployed
# engine would, so an engine that needed Triton or the plugin's Python registration fails.
_EXECUTE = textwrap.dedent("""
    import sys

    import tensorrt as trt
    import torch

    engine_path, io_path, out_path = sys.argv[1:]
    io = torch.load(io_path)
    guard, x_all, out_all = io["guard"], io["x_all"].cuda(), io["out_all"].cuda()
    x, out = (t[guard : t.shape[0] - guard] for t in (x_all, out_all))
    offs = io["offs"].cuda()
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        context = runtime.deserialize_cuda_engine(f.read()).create_execution_context()
    context.set_input_shape("x", tuple(x.shape))
    for name, tensor in (("x", x), ("offs", offs), ("out", out)):
        context.set_tensor_address(name, tensor.data_ptr())
    stream = torch.cuda.current_stream()
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    torch.save(out_all.cpu(), out_path)
    assert "triton" not in sys.modules and "torch_tensorrt" not in sys.modules
    """)


def _run_engine(model, x_all, offs, out_all, guard=0, max_rows=None):
    """Builds an engine-only export of model and runs it with _EXECUTE.

    x and the output are the rows of x_all and out_all between guard rows at either end.
    With max_rows, the engine takes 1 to max_rows rows and is built from 16 rows of x.
    Returns how many ops the grouped GEMM kernel converter took, and out_all after the run.
    """
    x = x_all[guard : x_all.shape[0] - guard]
    example, inputs, dynamic_shapes = x, [x, offs], None
    if max_rows is not None:
        example = x[:16]
        k = x.shape[1]
        inputs = [
            torch_tensorrt.Input(
                min_shape=(1, k),
                opt_shape=(16, k),
                max_shape=(max_rows, k),
                dtype=x.dtype,
            ),
            offs,
        ]
        dynamic_shapes = ({0: torch.export.Dim("rows", min=1, max=max_rows)}, None)
    with _kernel_spy() as spy:
        serialized = (
            torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
                torch.export.export(
                    model, (example, offs), dynamic_shapes=dynamic_shapes
                ),
                arg_inputs=inputs,
                min_block_size=1,
                grouped_mm_kernel=True,
                arg_input_binding_names=("x", "offs"),
                output_binding_names="out",
            )
        )
    with tempfile.TemporaryDirectory() as tmp:
        engine, io, out = (os.path.join(tmp, f) for f in ("engine", "io.pt", "out.pt"))
        with open(engine, "wb") as f:
            f.write(serialized)
        torch.save(
            {
                "guard": guard,
                "x_all": x_all.cpu(),
                "offs": offs.cpu(),
                "out_all": out_all.cpu(),
            },
            io,
        )
        subprocess.run([sys.executable, "-c", _EXECUTE, engine, io, out], check=True)
        return spy.call_count, torch.load(out).cuda()


def _kernel_unavailable():
    if not torch.cuda.is_available():
        return "it needs a CUDA GPU"
    return aten_ops_converters._grouped_mm_kernel_unavailable()


class TestGroupedMMKernel(TestCase):
    def setUp(self):
        # Decided at run time, not at collection, as tests/py/dynamo/executorch/test_export.py
        # explains: remote GPU runners can collect off the GPU host.
        reason = _kernel_unavailable()
        if reason is not None:
            self.skipTest(f"the grouped GEMM kernel is unavailable: {reason}")

    @parameterized.expand(
        [
            # From sm_90 on, the kernel gets the code that ran on A100.
            ("hopper", 90),
            ("blackwell", 100),
        ]
    )
    def test_code_for_newer_gpus_fits_a_plugin_launch(self, _, arch):
        # Compiles for each target on whatever GPU runs the test.
        import triton
        from torch_tensorrt.dynamo.conversion.impl import _grouped_mm_kernel
        from torch_tensorrt.kernels import _common, _triton
        from triton.backends.compiler import GPUTarget

        # PTX ISA 8.6 (CUDA 12.8) is the first that can target sm_100.
        driver_isa = _triton._driver_max_ptx_version()
        if arch >= 100 and driver_isa is not None and driver_isa < 86:
            self.skipTest("the CUDA driver cannot load PTX for sm_100")

        def build(target):
            # The PTX ISA cap reads the device, so it is mocked as well: the ptxas for
            # sm_80 cannot assemble the newer ISA of the ptxas for sm_90 on.
            with (
                mock.patch.object(
                    triton.runtime.driver.active,
                    "get_current_target",
                    return_value=GPUTarget("cuda", target, 32),
                ),
                mock.patch.object(_triton, "_device_arch", return_value=target),
            ):
                artifact = _grouped_mm_kernel._build("*bf16", 16, 2048, 768, target)
            self.assertIn(f".target sm_{target}".encode(), artifact.ptx)
            self.assertLessEqual(
                artifact.shared_mem, _common.AOT_MAX_DYNAMIC_SHARED_MEM
            )
            # Without the header, which names the PTX ISA and the GPU.
            return artifact.ptx.split(b"\n.target")[1].split(b"\n", 1)[1]

        self.assertEqual(build(arch), build(80))

    def test_kernel_drops_a_pipeline_stage_to_fit(self):
        import triton
        from torch_tensorrt.dynamo.conversion.impl import _grouped_mm_kernel
        from torch_tensorrt.kernels import _common, _triton

        capability = triton.runtime.driver.active.get_current_target().arch
        with (
            # 6 stages of these tiles need more than 48 KiB of shared memory.
            mock.patch.object(_grouped_mm_kernel, "_NUM_STAGES", (6, 3)),
            mock.patch.object(
                _triton, "compile_triton_to_ptx", wraps=_triton.compile_triton_to_ptx
            ) as compile_,
        ):
            artifact = _grouped_mm_kernel._build("*bf16", 16, 2048, 768, capability)
        self.assertEqual(
            [call.kwargs["num_stages"] for call in compile_.call_args_list], [6, 3]
        )
        self.assertLessEqual(artifact.shared_mem, _common.AOT_MAX_DYNAMIC_SHARED_MEM)

    def test_kernel_too_big_for_a_plugin_launch_does_not_build(self):
        import triton
        from torch_tensorrt.dynamo.conversion.impl import _grouped_mm_kernel

        capability = triton.runtime.driver.active.get_current_target().arch
        with (
            mock.patch.object(_grouped_mm_kernel, "_BLOCK_K", 256),
            self.assertRaisesRegex(RuntimeError, "shared memory"),
        ):
            _grouped_mm_kernel._build("*bf16", 16, 2048, 768, capability)

    def test_dynamic_rows_stay_in_one_engine(self):
        dtype = torch.bfloat16
        experts, k, n = 16, 128, 256
        model = Experts(experts, k, n, dtype).eval().cuda()
        offs = _offsets([50] * 14 + [0, 0])
        x = torch.randn(700, k, dtype=dtype).cuda()
        rows = torch.export.Dim("rows", min=1, max=4096)
        ep = torch.export.export(model, (x, offs), dynamic_shapes=({0: rows}, None))
        compiled = torch_tensorrt.dynamo.compile(
            ep,
            arg_inputs=[
                torch_tensorrt.Input(
                    min_shape=(1, k),
                    opt_shape=(700, k),
                    max_shape=(4096, k),
                    dtype=dtype,
                ),
                torch_tensorrt.Input(shape=(experts,), dtype=torch.int32),
            ],
            min_block_size=1,
            grouped_mm_kernel=True,
        )
        submodules = [name for name, _ in compiled.named_children()]
        self.assertEqual(submodules, ["_run_on_acc_0"])

        # Other row counts and routings, with empty groups first, last and in between.
        for counts in (
            [0, 3, 0, 1000, 0, 1, 129, 0, 2, 0, 0, 7, 0, 0, 900, 0],
            [1] + [0] * 15,
        ):
            offs = _offsets(counts)
            x = torch.randn(int(offs[-1]), k, dtype=dtype).cuda()
            with torch.no_grad():
                torch.testing.assert_close(
                    compiled(x, offs), model(x, offs), rtol=1e-2, atol=1e-2
                )

    def test_dynamic_rows_engine_runs_without_triton(self):
        # The plugin's output shape, launch grid and row count argument come from the
        # saved engine, here at a row count other than the one it was built from.
        groups, k, n, rows = 4, 64, 128, 300
        model = GroupedMM(groups, k, n).eval().cuda()
        x = torch.randn(rows, k, dtype=torch.bfloat16, device="cuda")
        out = torch.full((rows, n), 7.0, dtype=torch.bfloat16, device="cuda")
        offs = _offsets([70, 0, 130, rows - 200])
        kernel_ops, out = _run_engine(model, x, offs, out, max_rows=1024)
        self.assertEqual(kernel_ops, 1)
        with torch.no_grad():
            torch.testing.assert_close(out, model(x, offs), rtol=1e-2, atol=1e-2)

    @parameterized.expand([("static_rows", False), ("dynamic_rows", True)])
    def test_refit(self, _, dynamic):
        # Refit converts the new weights module again, so it builds the kernel too.
        groups, k, n, rows = 4, 64, 128, 300
        x = torch.randn(rows, k, dtype=torch.bfloat16, device="cuda")
        offs = _offsets([70, 0, 130, rows - 200])
        dynamic_shapes, inputs = None, [x, offs]
        if dynamic:
            dynamic_shapes = ({0: torch.export.Dim("rows", min=1, max=1024)}, None)
            inputs = [
                torch_tensorrt.Input(
                    min_shape=(1, k),
                    opt_shape=(rows, k),
                    max_shape=(1024, k),
                    dtype=torch.bfloat16,
                ),
                offs,
            ]

        def export(model):
            return torch.export.export(
                model.eval().cuda(), (x, offs), dynamic_shapes=dynamic_shapes
            )

        with _kernel_spy() as spy:
            compiled = torch_tensorrt.dynamo.compile(
                export(GroupedMM(groups, k, n)),
                arg_inputs=inputs,
                min_block_size=1,
                grouped_mm_kernel=True,
                immutable_weights=False,
            )
            new_model = GroupedMM(groups, k, n)
            torch_tensorrt.dynamo.refit_module_weights(
                compiled, export(new_model), in_place=True
            )
        if dynamic:
            offs = _offsets([50, 0, 100, 50])
            x = x[:200]
        with torch.no_grad():
            torch.testing.assert_close(
                compiled(x, offs), new_model(x, offs), rtol=1e-2, atol=1e-2
            )
        self.assertEqual(spy.call_count, 2)

    @parameterized.expand(
        [
            # The last offset counts 300 rows; mat1 has 200.
            (
                "past_the_rows",
                200,
                [60, 60, 160, 300],
                ((0, 60, 0), (60, 160, 2), (160, 200, 3)),
            ),
            # Group 1 starts 300 rows before mat1, and rows 100 on are past the last offset.
            ("negative", 200, [-300, 100, 100, 100], ((0, 100, 1),)),
            # Group 1 ends 400 rows, more than 2 row tiles, below where it starts.
            ("going_down", 600, [500, 100, 600, 600], ((0, 500, 0), (500, 600, 2))),
        ]
    )
    def test_invalid_offsets_stay_in_bounds(self, _, rows, offsets, row_groups):
        # Such offsets are invalid, but they must not make the engine's kernel write
        # outside the output, which the guard rows before and after it would show. A
        # read outside mat1 only feeds rows that are never stored, and here it lands in
        # the guard rows of a_all, so this test cannot catch it.
        # Rows past the largest offset belong to no group, and the kernel zeroes them
        # instead of leaving stale memory; the output starts as 7.0 to show either.
        groups, k, n, guard = 4, 64, 128, 128

        model = GroupedMM(groups, k, n).eval().cuda()
        a_all = torch.randn(
            guard + rows + guard, k, dtype=torch.bfloat16, device="cuda"
        )
        c_all = torch.full(
            (guard + rows + guard, n), 7.0, dtype=torch.bfloat16, device="cuda"
        )
        offs = torch.tensor(offsets, dtype=torch.int32, device="cuda")
        kernel_ops, c_all = _run_engine(model, a_all, offs, c_all, guard)
        self.assertEqual(kernel_ops, 1)
        self.assertTrue(torch.all(c_all[:guard] == 7.0))
        self.assertTrue(torch.all(c_all[guard + rows :] == 7.0))
        a, c = a_all[guard : guard + rows], c_all[guard : guard + rows]
        self.assertEqual(
            torch.count_nonzero(c[min(max(*offsets, 0), rows) :]).item(), 0
        )
        for start, end, group in row_groups:
            torch.testing.assert_close(
                c[start:end],
                a[start:end] @ model.w[group].detach(),
                rtol=1e-2,
                atol=1e-2,
            )

    def test_compile_failure_falls_back(self):
        # A kernel that does not build, for example under a new Triton or GPU,
        # declines the op, so the dense lowering keeps it in TensorRT. Each compile
        # tries the kernel once, in case the failure was temporary, and warns with its
        # cause.
        from torch_tensorrt.dynamo.conversion.impl import _grouped_mm_kernel

        model = Experts(4, 64, 64, torch.float16).eval().cuda()
        offs = _offsets([10, 0, 90, 28])
        x = torch.randn(128, 64, dtype=torch.float16).cuda()
        _grouped_mm_kernel._compile_for.cache_clear()
        self.addCleanup(_grouped_mm_kernel._compile_for.cache_clear)
        with (
            mock.patch.object(
                _grouped_mm_kernel, "_build", side_effect=RuntimeError("no kernel")
            ) as build,
            _kernel_spy() as spy,
            self.assertLogs(aten_ops_converters.__name__, level="WARNING") as logs,
        ):
            for _ in range(2):
                compiled = torch_tensorrt.compile(
                    model,
                    ir="dynamo",
                    inputs=[x, offs],
                    min_block_size=1,
                    grouped_mm_kernel=True,
                    pass_through_build_failures=True,
                )
        spy.assert_not_called()
        self.assertEqual(build.call_count, 2)
        warnings = [line for line in logs.output if "not used" in line]
        self.assertEqual(len(warnings), 2)
        self.assertIn("no kernel", warnings[0])
        self.assertNotIn(torch.ops.aten._grouped_mm.default, _in_pytorch(compiled))
        with torch.no_grad():
            torch.testing.assert_close(
                compiled(x, offs), model(x, offs), rtol=1e-2, atol=1e-2
            )

    def test_failed_plugin_registration_is_retried(self):
        # In a fresh process, since the plugin registers once per process.
        child = textwrap.dedent("""
            from unittest import mock

            import tensorrt.plugin as trtp
            from torch_tensorrt.dynamo.conversion.impl import _grouped_mm_kernel

            with mock.patch.object(trtp, "aot_impl", side_effect=RuntimeError("once")):
                try:
                    _grouped_mm_kernel.plugin()
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("the registration did not fail")
            _grouped_mm_kernel.plugin()
            """)
        subprocess.run([sys.executable, "-c", child], check=True)


if __name__ == "__main__":
    run_tests()
