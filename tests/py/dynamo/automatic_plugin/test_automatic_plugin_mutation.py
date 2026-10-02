# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import io
import platform
import unittest

import torch
import torch_tensorrt
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

import tensorrt as trt


@torch.library.custom_op("torchtrt_mutation::add_one", mutates_args=("x",))
def add_one(x: torch.Tensor) -> None:
    x.add_(1)


@add_one.register_fake
def _(x):
    return None


@torch.library.custom_op("torchtrt_mutation::two_buffers", mutates_args=("x", "y"))
def two_buffers(x: torch.Tensor, y: torch.Tensor) -> None:
    x.add_(1)
    y.mul_(2)


@two_buffers.register_fake
def _(x, y):
    return None


if torch_tensorrt.ENABLED_FEATURES.qdp_plugin:
    for op in ("torchtrt_mutation::add_one", "torchtrt_mutation::two_buffers"):
        torch_tensorrt.dynamo.conversion.plugins.custom_op(
            op, supports_dynamic_shapes=True
        )


@torch.library.custom_op("torchtrt_mutation::aot_add_one", mutates_args=("x",))
def aot_add_one(x: torch.Tensor) -> None:
    x.add_(1)


@aot_add_one.register_fake
def _(x):
    return None


if torch_tensorrt.ENABLED_FEATURES.qdp_plugin and platform.system() != "Windows":
    import tensorrt.plugin as trtp

    @trtp.register("torchtrt_mutation::aot_add_one")
    def aot_desc(x: trtp.TensorDesc) -> tuple[trtp.TensorDesc]:
        return (x.aliased(),)

    @trtp.aot_impl("torchtrt_mutation::aot_add_one")
    def aot_impl(
        x: trtp.TensorDesc, outputs: tuple[trtp.TensorDesc], tactic: int
    ) -> tuple[str | bytes, str | bytes, trtp.KernelLaunchParams, trtp.SymExprs]:
        # Portable PTX keeps this test independent of Triton / cuda-python.
        ptx = """
.version 8.0
.target sm_80
.address_size 64
.visible .entry trt_mutation_add_one(
    .param .u64 input, .param .u32 count, .param .u64 output
) {
    .reg .pred p;
    .reg .b32 r<5>;
    .reg .b64 rd<5>;
    .reg .f32 f<3>;
    ld.param.u64 rd1, [input];
    ld.param.u32 r1, [count];
    ld.param.u64 rd2, [output];
    mov.u32 r2, %tid.x;
    mov.u32 r3, %ctaid.x;
    mov.u32 r4, %ntid.x;
    mad.lo.u32 r2, r3, r4, r2;
    setp.ge.u32 p, r2, r1;
    @p bra DONE;
    mul.wide.u32 rd3, r2, 4;
    add.s64 rd4, rd1, rd3;
    ld.global.f32 f1, [rd4];
    add.f32 f2, f1, 0f3f800000;
    add.s64 rd4, rd2, rd3;
    st.global.f32 [rd4], f2;
DONE:
    ret;
}
"""
        n = x.shape_expr.numel()
        params = trtp.KernelLaunchParams()
        params.grid_x, params.block_x, params.shared_mem = trtp.cdiv(n, 256), 256, 0
        extra = trtp.SymIntExprs(1)
        extra[0] = trtp.SymInt32(n)
        return "trt_mutation_add_one", ptx, params, extra

    torch_tensorrt.dynamo.conversion.plugins.generate_plugin_converter(
        "torchtrt_mutation::aot_add_one", supports_dynamic_shapes=True
    )


@unittest.skipIf(platform.system() == "Windows", "QDP plugins require Linux")
@unittest.skipIf(
    not torch_tensorrt.ENABLED_FEATURES.qdp_plugin, "QDP plugin is unavailable"
)
class TestPluginMutation(unittest.TestCase):
    def compile(self, model, inputs, **settings):
        compiled = torch_tensorrt.compile(
            model,
            ir="dynamo",
            inputs=inputs,
            min_block_size=1,
            immutable_weights=True,
            **settings,
        )
        engines = [m for m in compiled.modules() if isinstance(m, TorchTensorRTModule)]
        self.assertEqual(len(engines), 1, str(compiled.graph))
        return compiled, engines[0]

    def test_none_return_and_downstream_consumer(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                add_one(x)
                return x * 2

        compiled, engine = self.compile(Model(), [torch.randn(8, 32, device="cuda")])
        self.assertEqual(len(engine.aliased_io), 1)
        self.assertEqual(engine.num_user_outputs, 1)
        self.assertEqual(len(engine.output_binding_names), 2)
        for _ in range(2):
            x = torch.randn(8, 32, device="cuda")
            expected = x + 1
            torch.testing.assert_close(compiled(x), expected * 2)
            torch.testing.assert_close(x, expected)

    def test_visible_aliased_return_is_preserved(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                add_one(x)
                return x * 2, x

        compiled, engine = self.compile(Model(), [torch.randn(8, 32, device="cuda")])
        self.assertEqual(engine.num_user_outputs, 2)
        x = torch.randn(8, 32, device="cuda")
        expected = x + 1
        fresh, aliased = compiled(x)
        torch.testing.assert_close(fresh, expected * 2)
        torch.testing.assert_close(aliased, expected)
        torch.testing.assert_close(x, expected)

    def test_successive_mutations_write_back_final_state(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                add_one(x)
                add_one(x)
                return x * 2

        compiled, engine = self.compile(Model(), [torch.randn(8, 32, device="cuda")])
        self.assertEqual(len(engine.aliased_io), 1)
        x = torch.randn(8, 32, device="cuda")
        for _ in range(2):
            expected = x + 2
            torch.testing.assert_close(compiled(x), expected * 2)
            torch.testing.assert_close(x, expected)

    def test_multiple_mutated_buffers(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                two_buffers(x, y)
                return x, y

        inputs = [torch.randn(8, 32, device="cuda") for _ in range(2)]
        compiled, engine = self.compile(Model(), [t.clone() for t in inputs])
        self.assertEqual(len(engine.aliased_io), 2)
        expected = inputs[0] + 1, inputs[1] * 2
        actual = compiled(*inputs)
        for out, inp, ref in zip(actual, inputs, expected):
            torch.testing.assert_close(out, ref)
            torch.testing.assert_close(inp, ref)

    def test_dynamic_none_mutation(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                add_one(x)
                return x

        compiled, engine = self.compile(
            Model(),
            [
                torch_tensorrt.Input(
                    min_shape=(1, 32),
                    opt_shape=(8, 32),
                    max_shape=(16, 32),
                    dtype=torch.float32,
                )
            ],
        )
        self.assertEqual(len(engine.aliased_io), 1)
        for batch in (1, 8, 16):
            x = torch.randn(batch, 32, device="cuda")
            expected = x + 1
            torch.testing.assert_close(compiled(x), expected)
            torch.testing.assert_close(x, expected)

    def test_consumed_multi_buffer_mutation_falls_back_correctly(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                two_buffers(x, y)
                return x * 3 + y * 4

        x, y = [torch.randn(8, 32, device="cuda") for _ in range(2)]
        expected_x, expected_y = x + 1, y * 2
        compiled, engine = self.compile(Model(), [x.clone(), y.clone()])
        self.assertFalse(engine.aliased_io)
        torch.testing.assert_close(compiled(x, y), expected_x * 3 + expected_y * 4)
        torch.testing.assert_close(x, expected_x)
        torch.testing.assert_close(y, expected_y)

    def test_mutated_view_falls_back_correctly(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                add_one(x[:, 8:24])
                return x * 2

        x = torch.randn(8, 32, device="cuda")
        expected = x.clone()
        expected[:, 8:24] += 1
        compiled, engine = self.compile(Model(), [x.clone()])
        self.assertFalse(engine.aliased_io)
        torch.testing.assert_close(compiled(x), expected * 2)
        torch.testing.assert_close(x, expected)

    def test_state_dict_preserves_hidden_output_count(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                add_one(x)
                return x * 2, x

        _, engine = self.compile(Model(), [torch.randn(8, 32, device="cuda")])
        buffer = io.BytesIO()
        torch.save(engine.state_dict(), buffer)
        buffer.seek(0)
        restored = TorchTensorRTModule()
        restored.load_state_dict(torch.load(buffer, weights_only=False))
        self.assertEqual(restored.num_user_outputs, engine.num_user_outputs)
        self.assertEqual(restored.aliased_io, engine.aliased_io)
        x = torch.randn(8, 32, device="cuda")
        expected = x + 1
        fresh, aliased = restored(x)
        torch.testing.assert_close(fresh, expected * 2)
        torch.testing.assert_close(aliased, expected)

    def test_aot_standalone_engine_with_hidden_mutation_output(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                aot_add_one(x)
                return x * 2

        x = torch.randn(8, 32, device="cuda")
        exported = torch.export.export(Model(), (x.clone(),), strict=False)
        engine_bytes = (
            torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
                exported,
                inputs=[x.clone()],
                immutable_weights=True,
                output_binding_names="result",
                require_full_compilation=True,
            )
        )
        runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        self.assertIsNotNone(engine)
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        input_name = next(
            n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
        )
        output_names = [
            n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT
        ]
        self.assertEqual(len(output_names), 2)
        mutation_name = next(n for n in output_names if n != "result")
        # TensorRT can insert copies around an aliased plugin and does not
        # necessarily expose it through get_aliased_input_tensor. Standalone
        # callers must bind the mutation output to the corresponding input.
        context = engine.create_execution_context()
        expected = x + 1
        result = torch.empty_like(x)
        context.set_tensor_address(input_name, x.data_ptr())
        context.set_tensor_address(mutation_name, x.data_ptr())
        context.set_tensor_address("result", result.data_ptr())
        self.assertTrue(
            context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(x, expected)
        torch.testing.assert_close(result, expected * 2)

    def test_model_returning_none(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return add_one(x)

        compiled, _ = self.compile(Model(), [torch.randn(8, 32, device="cuda")])
        x = torch.randn(8, 32, device="cuda")
        expected = x + 1
        self.assertIsNone(compiled(x))
        torch.testing.assert_close(x, expected)
