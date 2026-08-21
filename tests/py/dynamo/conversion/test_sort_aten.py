import unittest

import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt import Input
from torch_tensorrt.dynamo.conversion.aten_ops_converters import sort_validator

from .harness import DispatchTestCase


class TestSortConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((3, 2, 4), 0, True),
            ((2, 3, 4, 5), 1, True),
            ((2, 3, 4, 5), 2, False),
            ((6, 7, 5, 4, 5), 4, False),
            ((1, 5, 2, 1), -1, True),
            ((1, 2, 5, 3), -2, False),
            ((6, 2, 1, 3), -4, True),
        ]
    )
    def test_sort(self, input_shape, dim, descending):
        class Sort(nn.Module):
            def forward(self, x):
                return torch.ops.aten.sort.default(x, dim, descending)

        inputs = [torch.randn(*input_shape)]
        self.run_test(
            Sort(),
            inputs,
            enable_passes=True,
            use_dynamo_tracer=True,
        )


class TestSortConverterDynamic(DispatchTestCase):
    @parameterized.expand(
        [
            (
                "3d_dynamic_descending",
                (2, 1, 4),
                (3, 2, 4),
                (3, 3, 4),
                2,
                True,
            ),
            (
                "4d_dynamic_ascending",
                (2, 2, 1, 4),
                (2, 2, 2, 4),
                (3, 3, 2, 4),
                3,
                False,
            ),
            (
                "4d_dynamic_descending_neg_dim",
                (1, 3, 1, 1),
                (2, 3, 2, 2),
                (3, 3, 2, 4),
                -3,
                True,
            ),
        ]
    )
    def test_sort_dynamic(self, _, min_shape, opt_shape, max_shape, dim, descending):
        class Sort(nn.Module):
            def forward(self, x):
                return torch.ops.aten.sort.default(x, dim, descending)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=torch.float,
            ),
        ]
        self.run_test_with_dynamic_shape(
            Sort(),
            input_specs,
            output_dtypes=[torch.float, torch.int64],
            use_dynamo_tracer=True,
        )


@unittest.skipIf(not torch.cuda.is_available(), "Skip because CUDA is not available")
class TestSortValidatorDefaultDim(TestCase):
    """Regression test for https://github.com/pytorch/TensorRT/issues/3777.

    sort_validator is only consulted by the partitioner, which
    DispatchTestCase bypasses (it interprets the traced graph directly), so
    the cases above don't exercise it. torch.sort(x) with no explicit dim
    omits dim from the FX node's args entirely (relying on the schema
    default of -1), so this goes through the full compile() pipeline, where
    sort_validator used to crash with `IndexError: tuple index out of
    range` from indexing node.args[1] unconditionally.
    """

    def test_sort_default_dim_compiles(self):
        class Sort(nn.Module):
            def forward(self, x):
                values, _ = torch.sort(x)
                return values

        mod = Sort().eval().cuda()
        inputs = [torch.randn(4, 8).cuda()]
        trt_mod = torch_tensorrt.compile(
            mod, ir="dynamo", inputs=inputs, min_block_size=1
        )
        acc_count = sum(
            1 for name, _ in trt_mod.named_children() if "_run_on_acc" in name
        )
        self.assertEqual(acc_count, 1)
        torch.testing.assert_close(trt_mod(*inputs), mod(*inputs))

    def test_sort_scalar_input_falls_back(self):
        """A 0-D (scalar) input has no dim to normalize: get_positive_dim
        computes dim % rank, and rank=0 for a scalar raises ZeroDivisionError.
        sort_validator must reject scalar inputs before reaching that call so
        the node falls back to PyTorch instead of crashing the partitioner."""

        class Sort(nn.Module):
            def forward(self, x):
                values, _ = torch.sort(x)
                return values

        mod = Sort().eval().cuda()
        inputs = [torch.tensor(5.0).cuda()]
        trt_mod = torch_tensorrt.compile(
            mod, ir="dynamo", inputs=inputs, min_block_size=1
        )
        acc_count = sum(
            1 for name, _ in trt_mod.named_children() if "_run_on_acc" in name
        )
        self.assertEqual(acc_count, 0)
        torch.testing.assert_close(trt_mod(*inputs), mod(*inputs))


class TestSortValidator(unittest.TestCase):
    def _sort_node(self, fn):
        class Mod(nn.Module):
            def forward(self, x):
                return fn(x)

        gm = torch.export.export(Mod(), (torch.randn(4, 8),)).module()
        return next(
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch.ops.aten.sort.default
        )

    def test_defaulted_dim_does_not_raise(self):
        # export drops dim=-1, so args is (x,) only; indexing args[1] used to IndexError
        node = self._sort_node(lambda x: torch.sort(x))
        self.assertEqual(len(node.args), 1)
        self.assertTrue(sort_validator(node))

    def test_explicit_dim_still_works(self):
        node = self._sort_node(lambda x: torch.sort(x, 0))
        self.assertGreaterEqual(len(node.args), 2)
        self.assertTrue(sort_validator(node))


if __name__ == "__main__":
    run_tests()
