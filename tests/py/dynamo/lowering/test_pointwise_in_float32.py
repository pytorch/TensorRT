# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import unittest

import torch
import torch_tensorrt
from parameterized import parameterized
from torch_tensorrt.dynamo import CompilationSettings, refit_module_weights
from torch_tensorrt.dynamo.lowering import get_decompositions, post_lowering
from torch_tensorrt.dynamo.lowering.passes.pointwise_in_float32 import (
    pointwise_in_float32,
)

ON = CompilationSettings(pointwise_in_float32=True)
TO_COPY = torch.ops.aten._to_copy.default
LINEAR = torch.ops.aten.linear.default
EXPAND = torch.ops.aten.expand.default
_HAS_BF16 = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0)


class Block(torch.nn.Module):
    """A residual add, an RMSNorm and a GELU, optionally with a linear layer between them."""

    def __init__(self, linear: bool) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(256) * 0.1)
        self.linear = torch.nn.Linear(256, 256) if linear else torch.nn.Identity()

    def forward(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        h = (x.float() + r).to(x.dtype)
        v = h.float().pow(2).mean(-1, keepdim=True)
        n = (h.float() * torch.rsqrt(v + 1e-6)).to(h.dtype) * (1.0 + self.weight)
        return torch.nn.functional.gelu(self.linear(n) * 3.0, approximate="tanh") * h


class Fn(torch.nn.Module):
    def __init__(self, fn) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return self.fn(x, r)


WEIGHT = torch.randn(8, 4).bfloat16()


def _split_by_cumsum(x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    a = x * 2.0 + r
    return (torch.cumsum(a, -1) + a).sigmoid() * 3.0


def _inputs(device: str, dtype: torch.dtype = torch.bfloat16, shape=(2, 64, 256)):
    return tuple(torch.randn(*shape, device=device, dtype=dtype) for _ in range(2))


def _graph(module: torch.nn.Module, inputs) -> torch.fx.GraphModule:
    program = torch.export.export(module, inputs)
    return program.run_decompositions(get_decompositions()).module()


def _rewritten(module: torch.nn.Module, inputs) -> torch.fx.GraphModule:
    return pointwise_in_float32(post_lowering(_graph(module, inputs), ON), ON)


def _dtype(node: torch.fx.Node):
    return node.meta["val"].dtype


def _pytorch_parts(module: torch.nn.Module):
    """Each part of a compiled module, with the ops and dtypes of the parts that run
    in PyTorch."""
    return [
        (
            name,
            (
                [
                    (str(n.target), n.meta["val"].dtype)
                    for n in part.graph.nodes
                    if n.op == "call_function"
                    and isinstance(n.meta.get("val"), torch.Tensor)
                ]
                if isinstance(part, torch.fx.GraphModule)
                else None
            ),
        )
        for name, part in [("", module), *module.named_children()]
    ]


class TestPointwiseInFloat32(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    @parameterized.expand(
        [
            ("setting_off", Block(linear=True), CompilationSettings()),
            (
                "casts_only",
                Fn(lambda x, r: (x.float() + r.float()).to(torch.bfloat16)),
                ON,
            ),
            ("own_output_dtype", Fn(lambda x, r: x.sum(-1, dtype=torch.bfloat16)), ON),
            ("exact_ops_only", Fn(lambda x, r: torch.relu(x).maximum(r)), ON),
            # No 16-bit value enters this float32 region.
            (
                "float32_round_trip",
                Fn(lambda x, r: (x * 3.0).bfloat16().float() * 2.0),
                ON,
                torch.float32,
            ),
            # The cumsum ends the region, so its bfloat16 result enters it.
            (
                "round_trip_of_an_outside_value",
                Fn(lambda x, r: (x - x.bfloat16().float()) * r.bfloat16().cumsum(-1)),
                ON,
                torch.float32,
            ),
        ]
    )
    def test_leaves_graph_alone(
        self, _, module, settings, dtype=torch.bfloat16
    ) -> None:
        gm = post_lowering(_graph(module.eval().to(dtype), _inputs("cpu", dtype)), ON)
        code = gm.code
        pointwise_in_float32(gm, settings)
        self.assertEqual(gm.code, code)

    def test_casts_at_region_boundaries(self) -> None:
        gm = _rewritten(Block(linear=True).eval().bfloat16(), _inputs("cpu"))
        nodes = [n for n in gm.graph.nodes if n.op == "call_function"]

        # Only the linear layer and the casts back into it and to the output stay in bfloat16.
        self.assertEqual(
            [n.target for n in nodes if _dtype(n) == torch.bfloat16],
            [TO_COPY, LINEAR, TO_COPY],
        )
        # Each bfloat16 value that enters a region is cast to float32 once.
        sources = [
            n.args[0]
            for n in nodes
            if n.target == TO_COPY
            and _dtype(n) == torch.float32
            and _dtype(n.args[0]) == torch.bfloat16
        ]
        self.assertEqual(
            sorted(s.op if s.op != "call_function" else str(s.target) for s in sources),
            ["aten.linear.default", "get_attr", "placeholder", "placeholder"],
        )
        # x.float() reads that cast instead of casting it again.
        self.assertFalse(
            [
                n
                for n in nodes
                if n.target == TO_COPY
                and n.args[0].target == TO_COPY
                and n.args[0].args[0].op == "placeholder"
            ]
        )

    def test_views_of_outside_values_stay(self) -> None:
        module = Fn(lambda x, r: torch.cat([x, r], -1)[..., ::2] * 3.0)
        gm = _rewritten(module.eval(), _inputs("cpu"))
        nodes = [n for n in gm.graph.nodes if n.op == "call_function"]

        # The cast to float32 comes after the slice, so the concatenation stays in bfloat16.
        self.assertEqual(
            [(str(n.target), _dtype(n)) for n in nodes],
            [
                ("aten.cat.default", torch.bfloat16),
                ("aten.slice.Tensor", torch.bfloat16),
                ("aten._to_copy.default", torch.float32),
                ("aten.mul.Tensor", torch.float32),
                ("aten._to_copy.default", torch.bfloat16),
            ],
        )

    @parameterized.expand(
        [
            ("into_math", lambda h, w: (h + 1.0) * 2.0, torch.float32),
            ("through_a_view", lambda h, w: h.reshape(2, -1) * 2.0, torch.float32),
            ("into_math_and_matmul", lambda h, w: (h @ w) * h, torch.bfloat16),
            ("through_a_cast", lambda h, w: h.float() * 2.0, torch.float32),
            ("through_an_integer_cast", lambda h, w: h.long() + 1, torch.bfloat16),
            (
                "through_a_cast_into_matmul",
                lambda h, w: h.half() @ w.half(),
                torch.bfloat16,
            ),
        ]
    )
    def test_concatenation_with_an_outside_value(self, _, tail, dtype) -> None:
        weight = torch.randn(256, 256, dtype=torch.bfloat16)
        module = Fn(lambda x, r: tail(torch.cat([x, r * 3.0], 1), weight))
        gm = _rewritten(module.eval(), _inputs("cpu"))
        cat = next(n for n in gm.graph.nodes if n.target == torch.ops.aten.cat.default)

        # Into math, r * 3.0 is not rounded before the concatenation. Into a matrix
        # multiply, the outside x is not cast in full.
        self.assertEqual(_dtype(cat), dtype)

    @parameterized.expand(
        [
            # x + r is rounded to bfloat16 before .float() widens it for the matrix
            # multiply, as without the setting.
            (
                "widening_cast",
                Fn(lambda x, r: (x + r).float() @ WEIGHT.float()),
                torch.bfloat16,
                ["aten._to_copy.default"],
            ),
            # A cast of the float32 cache r does not pull the concatenation it extends
            # into float32.
            (
                "cast_into_a_concatenation",
                Fn(lambda x, r: torch.cat([r.bfloat16(), x * 2.0]) @ WEIGHT),
                torch.float32,
                [
                    "aten._to_copy.default",
                    "aten._to_copy.default",
                    "aten.cat.default",
                    "aten.matmul.default",
                ],
            ),
        ]
    )
    def test_cast_into_a_matrix_multiply_stays(self, _, module, r_dtype, ops) -> None:
        inputs = (torch.randn(4, 8).bfloat16(), torch.randn(4, 8, dtype=r_dtype))
        gm = _rewritten(module.eval(), inputs)
        nodes = [n for n in gm.graph.nodes if n.op == "call_function"]

        self.assertEqual(
            [str(n.target) for n in nodes if _dtype(n) == torch.bfloat16], ops
        )

    @parameterized.expand(
        [
            # x * r is cast back to bfloat16 before the expand.
            ("into_matmul", lambda x, r: (x * r)[:, None].expand(4, 2, 8) @ WEIGHT),
            # x is cast to float32 before the expand.
            ("from_outside", lambda x, r: x[:, None].expand(4, 2, 8) * 3.0),
        ]
    )
    def test_casts_land_before_an_expand(self, _, fn) -> None:
        inputs = (torch.randn(4, 8).bfloat16(), torch.randn(4, 8).bfloat16())
        gm = _rewritten(Fn(fn).eval(), inputs)
        expand = next(n for n in gm.graph.nodes if n.target == EXPAND)

        self.assertEqual(expand.args[0].meta["val"].shape, (4, 1, 8))
        self.assertEqual(expand.args[0].target, TO_COPY)

    @parameterized.expand(
        [
            ("view", Fn(lambda x, r: (x + r).reshape(2, -1) * 3.0), (2, 64, 256)),
            (
                "type_as",
                Fn(lambda x, r: ((x + r).float() * 2.0).type_as(x) * 3.0),
                (2, 64, 256),
            ),
            (
                "channels_last_cast",
                Fn(
                    lambda x, r: ((x + r).float() * 2.0).to(
                        torch.bfloat16, memory_format=torch.channels_last
                    )
                    * 3.0
                ),
                (2, 4, 8, 8),
            ),
            (
                "round_trip_cast",
                Fn(
                    lambda x, r: ((x.float() * 3.0).bfloat16().float() * 2.0).bfloat16()
                ),
                (2, 64, 256),
            ),
        ]
    )
    def test_views_and_casts_inside_a_region(self, _, module, shape) -> None:
        inputs = _inputs("cpu", shape=shape)
        gm = _rewritten(module.eval(), inputs)
        nodes = [n for n in gm.graph.nodes if n.op == "call_function"]

        self.assertEqual(
            [n.target for n in nodes if _dtype(n) == torch.bfloat16], [TO_COPY]
        )
        with torch.no_grad():
            output, expected = gm(*inputs), module(*inputs)
        self.assertEqual(output.stride(), expected.stride())
        torch.testing.assert_close(output, expected)

    @parameterized.expand(
        [
            ("view", lambda v: torch.as_strided(v[1:], (2,), (1,), 1)),
            ("split", lambda v: torch.as_strided(v[1:].split(2)[1], (2,), (1,), 3)),
        ]
    )
    def test_view_whose_memory_is_read_ends_the_region(self, _, read) -> None:
        module = Fn(lambda x, r: read(x + r))
        inputs = (torch.arange(8, dtype=torch.bfloat16), torch.ones(8).bfloat16())
        gm = _rewritten(module.eval(), inputs)

        with torch.no_grad():
            torch.testing.assert_close(gm(*inputs), module(*inputs))

    @parameterized.expand(
        [
            ("mean", Fn(lambda x, r: x * torch.rsqrt((x * x).mean()))),
            ("sum", Fn(lambda x, r: x * torch.rsqrt((x * x).sum() / x.numel()))),
        ]
    )
    def test_whole_tensor_reductions_join(self, _, module) -> None:
        inputs = (torch.full((64, 256), 300.0).half(), torch.zeros(64, 256).half())
        gm = _rewritten(module.eval(), inputs)

        with torch.no_grad():
            torch.testing.assert_close(gm(*inputs), torch.ones_like(inputs[0]))
        # x enters the region through 3 uses, and is cast once.
        x = next(n for n in gm.graph.nodes if n.op == "placeholder")
        self.assertEqual(len([u for u in x.users if u.target == TO_COPY]), 1)

    def test_cond_branches_join(self) -> None:
        module = Fn(
            lambda x, r: torch.cond(
                r.sum() >= 0, lambda x: x * torch.rsqrt((x * x).mean()), torch.neg, (x,)
            )
        )
        inputs = (torch.full((64, 256), 300.0).half(), torch.zeros(64, 256).half())
        gm = _rewritten(module.eval(), inputs)

        with torch.no_grad():
            torch.testing.assert_close(gm(*inputs), torch.ones_like(inputs[0]))

    def test_node_without_meta_stays(self) -> None:
        module = Fn(lambda x, r: (x * 2.0).float() + r * 3.0)
        gm = post_lowering(_graph(module.eval(), _inputs("cpu")), ON)
        first, _, scaled = [n for n in gm.graph.nodes if n.op == "call_function"][:3]
        del first.meta["val"]
        pointwise_in_float32(gm, ON)

        self.assertEqual(_dtype(scaled), torch.float32)

    @parameterized.expand([("bfloat16", torch.bfloat16), ("float16", torch.float16)])
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_rounds_less(self, _, dtype) -> None:
        if dtype == torch.bfloat16 and not _HAS_BF16:
            self.skipTest("BF16 needs an Ampere or newer GPU")
        model = Block(linear=False).eval().cuda()
        model_16 = copy.deepcopy(model).to(dtype)
        inputs = _inputs("cuda", dtype)
        with torch.no_grad():
            reference = model(*(x.float() for x in inputs))
            errors, engines = [], []
            for enabled in (False, True):
                trt_module = torch_tensorrt.dynamo.compile(
                    torch.export.export(model_16, inputs),
                    inputs,
                    pointwise_in_float32=enabled,
                )
                output = trt_module(*inputs)
                self.assertEqual(output.dtype, dtype)
                errors.append((output.float() - reference).abs().mean())
                engines.append(
                    [n for n, _ in trt_module.named_children() if "_run_on_acc" in n]
                )
        self.assertTrue(engines[0])
        self.assertEqual(engines[0], engines[1])
        self.assertLess(errors[1], errors[0] / 2)

    @unittest.skipUnless(_HAS_BF16, "BF16 needs an Ampere or newer GPU")
    def test_code_left_to_pytorch_does_not_change(self) -> None:
        module = Fn(_split_by_cumsum).eval().cuda().bfloat16()
        inputs = _inputs("cuda")
        parts = []
        with torch.no_grad():
            for enabled in (False, True):
                trt_module = torch_tensorrt.dynamo.compile(
                    torch.export.export(module, inputs),
                    inputs,
                    pointwise_in_float32=enabled,
                    torch_executed_ops={"torch.ops.aten.cumsum.default"},
                    use_fast_partitioner=False,
                    min_block_size=1,
                )
                parts.append(_pytorch_parts(trt_module))
        # x * 2.0 + r feeds both the cumsum, which runs in PyTorch, and the second engine.
        engines = [name for name, _ in parts[1] if name.startswith("_run_on_acc")]
        self.assertEqual(len(engines), 2)
        self.assertEqual(parts[0], parts[1])

    @unittest.skipUnless(_HAS_BF16, "BF16 needs an Ampere or newer GPU")
    @unittest.skipIf(
        not torch_tensorrt.ENABLED_FEATURES.refit, "Refit feature required"
    )
    def test_refit(self) -> None:
        inputs = _inputs("cuda")
        programs = [
            torch.export.export(Block(linear=True).eval().cuda().bfloat16(), inputs)
            for _ in range(2)
        ]
        with torch.no_grad():
            trt_module = torch_tensorrt.dynamo.compile(
                programs[0],
                inputs,
                pointwise_in_float32=True,
                immutable_weights=False,
                min_block_size=1,
            )
            refitted = refit_module_weights(trt_module, programs[1], inputs)
            expected = torch_tensorrt.dynamo.compile(
                programs[1], inputs, pointwise_in_float32=True, min_block_size=1
            )
            output, reference = refitted(*inputs).float(), expected(*inputs).float()
        # The two engines can pick different kernels, so compare the mean error.
        error = (output - reference).abs().mean() / reference.abs().mean()
        self.assertLess(error, 0.01)


if __name__ == "__main__":
    unittest.main()
