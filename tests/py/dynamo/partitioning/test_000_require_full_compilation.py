import torch
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests


class TestRequireFullCompilation(TestCase):
    """The early return must reject unsupported operators with full compilation required.

    Fully supported graphs below min_block_size still return without building an engine.
    These controls check that they are not rejected, not that they run in TensorRT.
    """

    @staticmethod
    def _six_linear_layers():
        return torch.nn.ModuleList([torch.nn.Linear(64, 64) for _ in range(6)])

    @classmethod
    def _no_converter_module(cls):
        """Small, and its only non-trivial operator has no converter."""

        class NoConverter(torch.nn.Module):
            def forward(self, x):
                return x + torch.normal(
                    0.0, 1.0, size=x.shape, device=x.device, dtype=x.dtype
                )

        return NoConverter().eval().cuda()

    @classmethod
    def _small_fully_convertible_module(cls):
        """Every operator converts, and there are fewer than min_block_size of them."""

        class SmallFullyConvertible(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x + 1)

        return SmallFullyConvertible().eval().cuda()

    @classmethod
    def _large_fully_convertible_module(cls):
        class LargeFullyConvertible(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = cls._six_linear_layers()

            def forward(self, x):
                out = x
                for layer in self.layers:
                    out = torch.relu(layer(out))
                return out

        return LargeFullyConvertible().eval().cuda()

    @staticmethod
    def _compile(module, inputs, **kwargs):
        return torch_tensorrt.dynamo.compile(
            torch.export.export(module, tuple(inputs)),
            inputs=list(inputs),
            require_full_compilation=True,
            enabled_precisions={torch.float32},
            truncate_double=True,
            **kwargs,
        )

    @staticmethod
    def _segments(compiled):
        return [name for name, _ in compiled.named_children() if "_run_on" in name]

    def test_operator_without_converter_raises(self):
        """The case the flag exists for: something in the graph cannot be converted."""
        inputs = [torch.randn(8, 64, device="cuda")]
        with self.assertRaisesRegex(AssertionError, "have no TensorRT converter"):
            self._compile(self._no_converter_module(), inputs)

    def test_message_counts_the_unconvertible_operators(self):
        """The count has to name the operators that cannot be converted. Counting the
        convertible ones instead produced "only 2 of 2 operations are convertible", which
        contradicts itself."""
        inputs = [torch.randn(8, 64, device="cuda")]
        with self.assertRaises(AssertionError) as raised:
            self._compile(self._no_converter_module(), inputs)
        message = str(raised.exception)
        self.assertIn("require_full_compilation=True", message)
        self.assertIn("1 of 2 operations", message)
        self.assertIn("have no TensorRT converter", message)

    def test_small_fully_convertible_module_is_not_rejected(self):
        """The existing early return for a supported graph is still allowed."""
        inputs = [torch.randn(8, 64, device="cuda")]
        compiled = self._compile(self._small_fully_convertible_module(), inputs)
        torch.testing.assert_close(
            compiled(*inputs),
            self._small_fully_convertible_module()(*inputs),
            rtol=5e-3,
            atol=5e-3,
        )

    def test_empty_graph_is_not_rejected(self):
        """A module that returns its input has nothing to convert and nothing to refuse."""

        class Identity(torch.nn.Module):
            def forward(self, x):
                return x

        inputs = [torch.randn(8, 64, device="cuda")]
        compiled = self._compile(Identity().eval().cuda(), inputs)
        torch.testing.assert_close(compiled(*inputs), inputs[0])

    def test_dryrun_reports_instead_of_raising(self):
        """dryrun is documented as the way to inspect what would fall back, so it must
        not become fatal."""
        inputs = [torch.randn(8, 64, device="cuda")]
        compiled = self._compile(self._no_converter_module(), inputs, dryrun=True)
        self.assertIsNotNone(compiled)

    @parameterized.expand(
        [("default_min_block_size", {}), ("min_block_size_1", {"min_block_size": 1})]
    )
    def test_large_fully_convertible_module_builds_one_engine(self, _, kwargs):
        """Guards against over correction. Also asserts an engine was really built, since
        a check for the absence of a PyTorch segment passes on an empty segment list."""
        inputs = [torch.randn(8, 64, device="cuda")]
        module = self._large_fully_convertible_module()
        compiled = self._compile(module, inputs, **kwargs)
        segments = self._segments(compiled)
        self.assertTrue(
            any("_run_on_acc" in segment for segment in segments),
            f"expected a TensorRT engine, got {segments}",
        )
        self.assertFalse(
            any("_run_on_gpu" in segment for segment in segments),
            f"expected no PyTorch segment, got {segments}",
        )
        torch.testing.assert_close(
            compiled(*inputs), module(*inputs), rtol=5e-3, atol=5e-3
        )


if __name__ == "__main__":
    run_tests()
