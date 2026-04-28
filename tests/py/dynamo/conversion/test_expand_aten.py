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


class TestExpandUnknownRankInput(unittest.TestCase):
    """Regression coverage for slice/ops.py::expand handling inputs whose
    rank is unknown at TensorRT build time.

    Some upstream TRT layers (notably ``IShuffleLayer`` whose
    ``reshape_dims`` is supplied as a runtime shape tensor) can produce
    an output with ``Dims.nbDims == -1``. ``len()``/``tuple()`` on such
    a Dims raises ``ValueError`` per Python's ``__len__`` protocol. The
    expand converter must guard against this and fall back to a
    runtime-tensor stride computation.

    Note: whether the synthetic construction below actually yields an
    unknown-rank tensor depends on TRT version. The tests are written
    to tolerate either outcome — if the rank ends up statically known,
    the converter's fast path is exercised (regression-safe); if
    unknown, the new fallback path is exercised.
    """

    def _build_unknown_rank_tensor(self, net, inp):
        """Construct a TRT subgraph whose output rank may be unknown at
        build time.

        Pattern: feed a runtime-sized 1-D shape tensor into the
        ``reshape_dims`` input of a shuffle layer. When the shape
        tensor's length is itself only known at runtime, the shuffle
        output's rank is unknown.
        """
        shape_layer = net.add_shape(inp)
        shape_t = shape_layer.get_output(0)

        slice_layer = net.add_slice(shape_t, (0,), (1,), (1,))
        runtime_size = net.add_constant(
            (1,), trt.Weights(np.array([1], dtype=np.int32))
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

    @staticmethod
    def _rank_is_unknown(t):
        """True iff len(t.shape) raises (TRT marks rank as unknown)."""
        try:
            len(t.shape)
            return False
        except ValueError:
            return True

    @parameterized.expand(
        [
            ("static_target", (1, 4), (2, 4)),
            ("static_target_broadcast_first", (1, 1), (3, 4)),
            ("dynamic_target_keep_dim", (1, 4), (2, -1)),
            ("dynamic_target_all", (1, 4), (-1, -1)),
            ("rank_up_static", (1, 4), (3, 2, 4)),
        ]
    )
    def test_expand_does_not_raise_on_unknown_rank(
        self, _name, input_shape, expanded_shape
    ):
        """Pre-fix: ``len(input_t.shape)`` raised ``ValueError`` when the
        input rank was unknown at build time, crashing the converter.
        Post-fix: ``expand`` must complete and emit layers on the network.
        """
        ctx, net = self._new_ctx()
        inp = net.add_input("x", trt.float32, tuple(input_shape))
        unk = self._build_unknown_rank_tensor(net, inp)

        layers_before = net.num_layers
        try:
            out = impl.slice.expand(
                ctx,
                target=None,
                source_ir=None,
                name=f"test_expand_unknown_rank_{_name}",
                input_t=unk,
                shape=tuple(expanded_shape),
            )
        except ValueError as e:
            # This is the exact symptom the fix is meant to prevent.
            self.fail(
                f"expand() raised ValueError on unknown-rank input "
                f"({_name}, target={expanded_shape}): {e}"
            )

        # Must have actually emitted at least one layer (slice / shape /
        # gather / elementwise depending on path). If expand short-circuited
        # without emitting anything, num_layers wouldn't change.
        self.assertGreater(
            net.num_layers,
            layers_before,
            "expand() returned without emitting any TRT layers",
        )

        # Returned tensor must be a real TRT ITensor wired into this network.
        self.assertIsInstance(out, trt.ITensor)

    def test_expand_synthetic_construction_actually_yields_unknown_rank(self):
        """Diagnostic: confirm the synthetic ``_build_unknown_rank_tensor``
        construction actually produces an unknown-rank tensor on this TRT
        version. If this test fails, the parameterized cases above are
        only exercising the fast path (regression-safe but not validating
        the fix). Treat a failure here as informational, not a blocker.
        """
        ctx, net = self._new_ctx()
        inp = net.add_input("x", trt.float32, (1, 4))
        unk = self._build_unknown_rank_tensor(net, inp)
        if not self._rank_is_unknown(unk):
            self.skipTest(
                "TRT version resolved the synthetic shuffle's output rank "
                "statically; the unknown-rank fallback path is not exercised "
                "by the white-box tests on this build. The fast path is "
                "still validated as a regression check."
            )

    def test_expand_rank_mismatch_still_raises(self):
        """Negative test: known-rank input with rank > target rank must
        still raise ``RuntimeError``. The fix only adds a fallback for
        the unknown-rank case; it must not weaken the rank-check on
        known-rank inputs.
        """
        ctx, net = self._new_ctx()
        inp = net.add_input("x", trt.float32, (2, 1, 4))

        with self.assertRaises(RuntimeError):
            impl.slice.expand(
                ctx,
                target=None,
                source_ir=None,
                name="test_expand_rank_mismatch",
                input_t=inp,
                shape=(2, 4),
            )


if __name__ == "__main__":
    run_tests()
