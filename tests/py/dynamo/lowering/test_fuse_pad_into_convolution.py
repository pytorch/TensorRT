import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from parameterized import parameterized

from ..conversion.harness import DispatchTestCase


def _node_targets(gm: torch.fx.GraphModule) -> list:
    return [n.target for n in gm.graph.nodes if n.op == "call_function"]


def _build_pad_conv_graph(
    *,
    pad: list[int],
    pad_value: float = 0.0,
    stride: list[int],
    conv_padding: list[int],
    dilation: list[int],
    groups: int = 1,
    bias: bool = True,
    pad_num_users: int = 1,
) -> torch.fx.GraphModule:
    """Build ``placeholder -> constant_pad_nd -> convolution -> output``."""
    g = torch.fx.Graph()
    x = g.placeholder("x")
    weight = g.placeholder("weight")
    bias_node = g.placeholder("bias") if bias else None
    pad_node = g.call_function(
        torch.ops.aten.constant_pad_nd.default,
        args=(x, pad, pad_value),
    )
    conv = g.call_function(
        torch.ops.aten.convolution.default,
        args=(
            pad_node,
            weight,
            bias_node,
            stride,
            conv_padding,
            dilation,
            False,
            [0] * len(stride),
            groups,
        ),
    )
    if pad_num_users > 1:
        # A second consumer keeps the pad from being fused.
        add = g.call_function(torch.ops.aten.add.Tensor, args=(pad_node, 0.0))
        g.output((conv, add))
    else:
        g.output(conv)
    return torch.fx.GraphModule({}, g)


class TestFusePadIntoConvolutionPass(unittest.TestCase):
    """Unit tests for the fuse_pad_into_convolution lowering pass."""

    def _settings(self):
        from torch_tensorrt.dynamo._settings import CompilationSettings

        return CompilationSettings()

    def _run_pass(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        from torch_tensorrt.dynamo.lowering.passes.fuse_pad_into_convolution import (
            fuse_pad_into_convolution,
        )

        return fuse_pad_into_convolution(gm, self._settings())

    def test_fuses_causal_3d_pad(self) -> None:
        """constant_pad_nd + convolution → tensorrt::conv_asym_pad."""
        from torch_tensorrt.dynamo.lowering.passes.fuse_pad_into_convolution import (
            tensorrt_conv_asym_pad_op,
        )

        gm = _build_pad_conv_graph(
            pad=[1, 1, 1, 1, 2, 0],
            stride=[1, 1, 1],
            conv_padding=[0, 0, 0],
            dilation=[1, 1, 1],
        )
        gm = self._run_pass(gm)
        targets = _node_targets(gm)
        self.assertNotIn(torch.ops.aten.constant_pad_nd.default, targets)
        self.assertNotIn(torch.ops.aten.convolution.default, targets)
        self.assertIn(tensorrt_conv_asym_pad_op, targets)

    def test_fused_args_carry_pre_post_padding(self) -> None:
        """Fused node stores independent pre/post padding per spatial dim."""
        from torch_tensorrt.dynamo.lowering.passes.fuse_pad_into_convolution import (
            tensorrt_conv_asym_pad_op,
        )

        gm = _build_pad_conv_graph(
            pad=[1, 1, 1, 1, 2, 0],
            stride=[1, 1, 1],
            conv_padding=[0, 0, 0],
            dilation=[1, 1, 1],
        )
        gm = self._run_pass(gm)
        fused = next(n for n in gm.graph.nodes if n.target == tensorrt_conv_asym_pad_op)
        # args: (x, weight, bias, stride, pre, post, dilation, groups)
        self.assertEqual(fused.args[3], [1, 1, 1])
        self.assertEqual(fused.args[4], [2, 1, 1])  # pre: (t, h, w)
        self.assertEqual(fused.args[5], [0, 1, 1])  # post: (t, h, w)
        self.assertEqual(fused.args[6], [1, 1, 1])
        self.assertEqual(fused.args[7], 1)

    def test_combines_explicit_pad_with_conv_padding(self) -> None:
        """Symmetric conv padding is added into both pre and post."""
        from torch_tensorrt.dynamo.lowering.passes.fuse_pad_into_convolution import (
            tensorrt_conv_asym_pad_op,
        )

        gm = _build_pad_conv_graph(
            pad=[0, 0, 0, 0, 1, 0],
            stride=[1, 1, 1],
            conv_padding=[1, 1, 1],
            dilation=[1, 1, 1],
        )
        gm = self._run_pass(gm)
        fused = next(n for n in gm.graph.nodes if n.target == tensorrt_conv_asym_pad_op)
        self.assertEqual(fused.args[4], [2, 1, 1])
        self.assertEqual(fused.args[5], [1, 1, 1])

    def test_fuses_2d_asymmetric_pad(self) -> None:
        from torch_tensorrt.dynamo.lowering.passes.fuse_pad_into_convolution import (
            tensorrt_conv_asym_pad_op,
        )

        gm = _build_pad_conv_graph(
            pad=[0, 1, 0, 1],
            stride=[1, 1],
            conv_padding=[0, 0],
            dilation=[1, 1],
        )
        gm = self._run_pass(gm)
        fused = next(n for n in gm.graph.nodes if n.target == tensorrt_conv_asym_pad_op)
        self.assertEqual(fused.args[4], [0, 0])
        self.assertEqual(fused.args[5], [1, 1])

    def test_skips_multi_user_pad(self) -> None:
        """Pad with more than one consumer must not be folded."""
        from torch_tensorrt.dynamo.lowering.passes.fuse_pad_into_convolution import (
            tensorrt_conv_asym_pad_op,
        )

        gm = _build_pad_conv_graph(
            pad=[1, 1, 1, 1, 2, 0],
            stride=[1, 1, 1],
            conv_padding=[0, 0, 0],
            dilation=[1, 1, 1],
            pad_num_users=2,
        )
        gm = self._run_pass(gm)
        targets = _node_targets(gm)
        self.assertIn(torch.ops.aten.constant_pad_nd.default, targets)
        self.assertIn(torch.ops.aten.convolution.default, targets)
        self.assertNotIn(tensorrt_conv_asym_pad_op, targets)

    def test_skips_nonzero_pad_value(self) -> None:
        """Non-zero fill cannot be expressed as convolution padding."""
        from torch_tensorrt.dynamo.lowering.passes.fuse_pad_into_convolution import (
            tensorrt_conv_asym_pad_op,
        )

        gm = _build_pad_conv_graph(
            pad=[1, 1, 1, 1, 2, 0],
            pad_value=1.0,
            stride=[1, 1, 1],
            conv_padding=[0, 0, 0],
            dilation=[1, 1, 1],
        )
        gm = self._run_pass(gm)
        targets = _node_targets(gm)
        self.assertIn(torch.ops.aten.constant_pad_nd.default, targets)
        self.assertNotIn(tensorrt_conv_asym_pad_op, targets)

    def test_idempotent(self) -> None:
        """Running the pass twice is a no-op after the first fusion."""
        from torch_tensorrt.dynamo.lowering.passes.fuse_pad_into_convolution import (
            tensorrt_conv_asym_pad_op,
        )

        gm = _build_pad_conv_graph(
            pad=[1, 1, 1, 1, 2, 0],
            stride=[1, 1, 1],
            conv_padding=[0, 0, 0],
            dilation=[1, 1, 1],
        )
        gm = self._run_pass(gm)
        first = _node_targets(gm)
        gm = self._run_pass(gm)
        self.assertEqual(first, _node_targets(gm))
        self.assertEqual(first.count(tensorrt_conv_asym_pad_op), 1)

    def test_eager_custom_op_matches_pad_then_conv(self) -> None:
        """Eager tensorrt::conv_asym_pad matches F.pad + F.conv3d."""
        from torch_tensorrt.dynamo.lowering.passes.fuse_pad_into_convolution import (
            tensorrt_conv_asym_pad_op,
        )

        x = torch.randn(1, 4, 5, 8, 8)
        weight = torch.randn(4, 4, 3, 3, 3)
        bias = torch.randn(4)
        pre, post = [2, 1, 1], [0, 1, 1]
        expected = F.conv3d(F.pad(x, [1, 1, 1, 1, 2, 0]), weight, bias, 1, 0, 1, 1)
        got = tensorrt_conv_asym_pad_op(x, weight, bias, [1, 1, 1], pre, post, [1, 1, 1], 1)
        torch.testing.assert_close(got, expected)


class TestFusePadIntoConvolutionConverter(DispatchTestCase):
    """End-to-end TRT numerical tests exercising the pad→conv fusion."""

    @parameterized.expand(
        [
            ("causal_t_2_0", (1, 1, 1, 1, 2, 0), 1, 1, 1, True),
            ("causal_t_1_0", (1, 1, 1, 1, 1, 0), 1, 1, 1, True),
            ("zero_pad", (0, 0, 0, 0, 0, 0), 1, 1, 1, True),
            ("no_bias", (1, 1, 1, 1, 2, 0), 1, 1, 1, False),
            ("stride_2", (1, 1, 1, 1, 2, 0), 2, 1, 1, True),
            ("dilation_2", (2, 2, 2, 2, 4, 0), 1, 2, 1, True),
            ("groups_2", (1, 1, 1, 1, 2, 0), 1, 1, 2, True),
            ("hw_only", (1, 1, 1, 1), 1, 1, 1, True),
        ]
    )
    def test_padded_conv3d(self, _, pad, stride, dilation, groups, bias):
        class PaddedConv3d(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.pad = pad
                self.conv = nn.Conv3d(
                    8,
                    8,
                    3,
                    stride=stride,
                    padding=0,
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.conv(F.pad(x, self.pad))

        # Temporal extent must leave room for causal pad + dilation/stride.
        t = 8 if dilation > 1 or stride > 1 else 5
        inputs = [torch.randn(1, 8, t, 16, 16)]
        self.run_test(
            PaddedConv3d(),
            inputs,
            use_dynamo_tracer=True,
            enable_passes=True,
        )

    @parameterized.expand(
        [
            ("asym_01_01", (0, 1, 0, 1), 1),
            ("asym_stride_2", (0, 1, 0, 1), 2),
        ]
    )
    def test_padded_conv2d(self, _, pad, stride):
        class PaddedConv2d(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.pad = pad
                self.conv = nn.Conv2d(8, 4, 3, stride=stride, padding=0)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.conv(F.pad(x, self.pad))

        inputs = [torch.randn(1, 8, 16, 16)]
        self.run_test(
            PaddedConv2d(),
            inputs,
            use_dynamo_tracer=True,
            enable_passes=True,
        )

    def test_graph_contains_fused_op_after_lowering(self):
        """post_lowering should rewrite the pad+conv pair before conversion."""
        from torch_tensorrt.dynamo._settings import CompilationSettings
        from torch_tensorrt.dynamo.lowering import get_decompositions, post_lowering
        from torch_tensorrt.dynamo.lowering.passes.fuse_pad_into_convolution import (
            tensorrt_conv_asym_pad_op,
        )

        class PaddedConv3d(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv3d(4, 4, 3, padding=0)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.conv(F.pad(x, (1, 1, 1, 1, 2, 0)))

        model = PaddedConv3d().eval()
        x = torch.randn(1, 4, 5, 8, 8)
        exported = torch.export.export(model, (x,))
        exported = exported.run_decompositions(get_decompositions(False))
        gm = post_lowering(exported.module(), CompilationSettings())
        targets = _node_targets(gm)
        self.assertIn(tensorrt_conv_asym_pad_op, targets)
        self.assertNotIn(torch.ops.aten.constant_pad_nd.default, targets)


if __name__ == "__main__":
    unittest.main()
