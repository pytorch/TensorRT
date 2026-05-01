import unittest

import numpy as np
import tensorrt as trt
import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext

from .harness import DispatchTestCase


class TestExpandConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d_dim", (2, 1), (2, 3)),
            ("3d_dim", (2, 1, 1), (2, 3, 4)),
            ("4d_dim", (2, 1, 1, 1), (2, 3, 4, 5)),
            ("keep_dim", (2, 1, 5, 5), (2, 3, -1, -1)),
            ("different_ranks", (1, 5, 7), (2, 3, -1, -1)),
        ]
    )
    def test_expand(self, _, input_shape, expanded_shape):
        class Expand(nn.Module):
            def forward(self, x):
                return torch.ops.aten.expand.default(x, expanded_shape)

        inputs = [torch.randn(*input_shape)]
        self.run_test(
            Expand(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2d_dim", (2, 1), (4, 1), (6, 1), (-1, 3)),
            ("3d_dim", (2, 1, 1), (4, 1, 1), (6, 1, 1), (-1, 3, 4)),
            ("4d_dim", (1, 1, 1, 1), (3, 1, 1, 1), (5, 1, 1, 1), (-1, 2, 3, 6)),
            ("keep_dim", (2, 1, 5, 5), (4, 1, 5, 5), (6, 1, 5, 5), (-1, 3, -1, -1)),
            ("different_ranks", (1, 2, 1), (1, 2, 1), (2, 2, 1), (2, -1, -1, -1)),
        ]
    )
    def test_expand_dynamic_input(
        self, _, min_shape, opt_shape, max_shape, expanded_shape
    ):
        class ExpandInputDynamic(nn.Module):
            def forward(self, x):
                return torch.ops.aten.expand.default(x, expanded_shape)

        input_specs = [
            Input(
                dtype=torch.float32,
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
            ),
        ]
        self.run_test_with_dynamic_shape(
            ExpandInputDynamic(),
            input_specs,
        )

    @parameterized.expand(
        [
            ("3d_dim", (4, 1, 768), (1, 1, 768)),
        ]
    )
    def test_expand_dynamic_target_shape(self, _, input_shape, weight_shape):
        class ExpandTargetDynamic(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.cls_token = torch.nn.Parameter(torch.randn(weight_shape).cuda())

            def forward(self, x):
                batch_size = x.shape[0]
                cls_tokens = self.cls_token.expand(batch_size, -1, -1)
                embeddings = torch.cat((cls_tokens, x), dim=0)
                return embeddings

        input_specs = [
            Input(dtype=torch.float32, shape=input_shape),
        ]
        self.run_test_with_dynamic_shape(
            ExpandTargetDynamic(), input_specs, use_dynamo_tracer=True
        )


class TestExpandFromDynamicArange(DispatchTestCase):
    """Regression coverage for ``slice/ops.py::expand`` after the
    unknown-rank input fix.

    Background
    ----------
    The fix wraps ``tuple(input_t.shape)`` / ``len(input_t.shape)``
    inside ``expand`` in ``try/except``, because ``trt.Dims.__len__``
    raises ``ValueError`` when ``Dims.nbDims == -1`` (TRT's marker for
    "rank statically unknown at build time").

    The original failure was an emergent property of a specific
    multi-converter composition inside an ASR/Whisper-style workload
    We do not have a minimal PyTorch-level repro and any white-box construction we
    tried either gets constant-folded back to known rank by TRT or
    produces a structurally invalid graph that fails downstream for
    unrelated reasons.

    What this test does
    -------------------
    Acts as a regression check for the *static fast path*. It
    constructs an ASR-style ``arange + expand`` model where the
    intermediate tensor has dynamic dim sizes but known rank, and
    confirms the converter still produces a working engine.
    """

    def test_arange_then_expand_dynamic_seqlen(self):
        """ASR-style: arange driven by a dynamic seq length, expanded
        across a dynamic batch."""

        class ArangeExpand(nn.Module):
            def forward(self, x):
                T = x.shape[1]
                positions = torch.arange(T, device=x.device)
                return positions.expand(x.shape[0], -1)

        input_specs = [
            Input(
                dtype=torch.float32,
                min_shape=(1, 1, 4),
                opt_shape=(2, 5, 4),
                max_shape=(4, 16, 4),
            ),
        ]
        self.run_test_with_dynamic_shape(
            ArangeExpand(),
            input_specs,
            use_dynamo_tracer=True,
        )


class TestExpandUnknownRankInputDiagnostic(unittest.TestCase):
    """Developer-facing diagnostic that exercises the expand fix path
    on a volume-VALID synthetic unknown-rank construction.

    Pattern: take a volume-1 input (e.g. shape ``(1,)``), feed a
    runtime-sized (length=1) shape tensor with value ``[1]`` into a
    shuffle. The reshape ``(1,) → [1]`` is identity (volume preserved),
    so the network is valid. TRT still marks the shuffle's output as
    ``nbDims = -1`` because the reshape_dims operand's length is
    statically unknown.

    Note: the build_unknown_rank_tensor produces nbdims=-1 but is not a shape compatible example for _buggy* example
    """

    def _build_unknown_rank_tensor(self, net, inp, runtime_len):
        """Volume-valid construction.

        ``runtime_len`` is the *number of axes* the resulting shuffle
        output should have. For a volume-preserving identity reshape
        it must equal ``len(inp.shape)`` (i.e. ``inp``'s rank), so
        that the slice samples exactly all of ``add_shape(inp)``'s
        entries and the shuffle's reshape_dims = inp's actual shape.
        """
        shape_layer = net.add_shape(inp)
        shape_t = shape_layer.get_output(0)

        slice_layer = net.add_slice(shape_t, (0,), (1,), (1,))
        runtime_size = net.add_constant(
            (1,), trt.Weights(np.array([runtime_len], dtype=np.int32))
        ).get_output(0)
        slice_layer.set_input(2, runtime_size)
        runtime_shape_t = slice_layer.get_output(0)

        shuffle = net.add_shuffle(inp)
        shuffle.set_input(1, runtime_shape_t)
        return shuffle.get_output(0)

    def _new_ctx(self):
        builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
        net = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        return ConversionContext(net=net), net

    @parameterized.expand(
        [
            # (name, input_shape, runtime_len, expanded_shape)
            #
            # runtime_len == len(input_shape) gives an identity reshape
            # (volume-preserving, so the synthetic graph is valid).
            #
            # Group A: rank-1 input (volume 1) → shuffle output rank 1.
            ("rank1_static_target_2d", (1,), 1, (2, 4)),
            ("rank1_static_target_3d", (1,), 1, (3, 2, 4)),
            ("rank1_dyn_keep_last", (1,), 1, (3, -1)),
            ("rank1_dyn_all", (1,), 1, (-1, -1)),
            ("rank1_dyn_3d", (1,), 1, (2, -1, 4)),
            # Group B: rank-2 input (volume > 1) → shuffle output rank 2.
            ("rank2_vol4_static_target", (1, 4), 2, (3, 4)),
            ("rank2_vol4_dyn_last", (1, 4), 2, (3, -1)),
            ("rank2_vol4_dyn_all", (1, 4), 2, (-1, -1)),
            ("rank2_vol4_rank_up", (1, 4), 2, (5, 1, 4)),
            # Group C (buggy-on-purpose): rank-2 input + runtime_len=1.
            # Shuffle reshape (1, 4) → [<runtime-1>] is volume-mismatched
            # (4 != 1). TRT refuses to fold the chain and stamps the
            # shuffle output as nbDims = -1, which more reliably forces
            # the fix's fallback path. The dynamic-target cases may
            # surface the invalid graph downstream in cat() — those are
            # skipped, not failed.
            ("rank2_buggy_static_target", (1, 4), 1, (3, 4)),
            ("rank2_buggy_dyn_last", (1, 4), 1, (3, -1)),
            ("rank2_buggy_rank_up", (1, 4), 1, (5, 1, 4)),
        ]
    )
    def test_expand_on_unknown_rank_input(
        self, _name, input_shape, runtime_len, expanded_shape
    ):
        ctx, net = self._new_ctx()
        inp = net.add_input("x", trt.float32, tuple(input_shape))
        unk = self._build_unknown_rank_tensor(net, inp, runtime_len)

        out = impl.slice.expand(
            ctx,
            target="test_expand_unknown_rank",
            source_ir=None,
            name=f"test_expand_unknown_rank_{_name}",
            input_t=unk,
            shape=tuple(expanded_shape),
        )
        self.assertIsInstance(out, trt.ITensor)


if __name__ == "__main__":
    run_tests()
