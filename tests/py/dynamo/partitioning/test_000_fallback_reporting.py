import logging

import torch
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests

SUMMARY_MARKER = "operator(s)"


class _SummaryCollector(logging.Filter):

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []
        self.levels: list[int] = []

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if SUMMARY_MARKER in message:
            self.messages.append(message)
            self.levels.append(record.levelno)
        return True


class TestFallbackIsReported(TestCase):
    """Report fallback at INFO without warning about expected behavior."""

    @staticmethod
    def _six_linear_layers() -> torch.nn.ModuleList:
        return torch.nn.ModuleList([torch.nn.Linear(64, 64) for _ in range(6)])

    @classmethod
    def _impure_fallback_module(cls) -> torch.nn.Module:
        """Refused on the last path in the support test, and impure, so the older counter
        never recorded it."""

        class ImpureFallback(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = cls._six_linear_layers()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = x
                for layer in self.layers:
                    out = torch.relu(layer(out))
                return out + torch.rand_like(out)

        return ImpureFallback().eval().cuda()

    @classmethod
    def _complex_fallback_module(cls) -> torch.nn.Module:
        """Refused by the complex dtype check, which is one of the earlier returns."""

        class ComplexFallback(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = cls._six_linear_layers()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = x
                for layer in self.layers:
                    out = torch.relu(layer(out))
                return torch.real(torch.fft.fft(out)) + out

        return ComplexFallback().eval().cuda()

    @classmethod
    def _fully_supported_module(cls) -> torch.nn.Module:
        class FullySupported(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = cls._six_linear_layers()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = x
                for layer in self.layers:
                    out = torch.relu(layer(out))
                return out

        return FullySupported().eval().cuda()

    def _compile(self, module, inputs, log_level=logging.INFO, **kwargs):
        collector = _SummaryCollector()
        logger = logging.getLogger("torch_tensorrt.dynamo._compiler")
        previous_level = logger.level
        logger.setLevel(log_level)
        logger.addFilter(collector)
        try:
            compiled = torch_tensorrt.dynamo.compile(
                torch.export.export(module, tuple(inputs)),
                inputs=list(inputs),
                min_block_size=1,
                enabled_precisions={torch.float32},
                truncate_double=True,
                **kwargs,
            )
        finally:
            logger.removeFilter(collector)
            logger.setLevel(previous_level)
        self.assertTrue(all(level == logging.INFO for level in collector.levels))
        segments = [name for name, _ in compiled.named_children()]
        return segments, collector.messages

    @parameterized.expand(
        [
            (
                "impure_refusal",
                "_impure_fallback_module",
                "rand_like",
                "no validated TensorRT converter",
            ),
            (
                "complex_dtype_refusal",
                "_complex_fallback_module",
                "fft",
                "complex tensor dtype",
            ),
        ]
    )
    def test_fallback_is_reported(self, _, factory, expected_operator, expected_reason):
        """Both of these split the graph, and a refusal on any path has to be reported, not
        only one of them."""
        inputs = [torch.randn(8, 64, device="cuda")]
        segments, messages = self._compile(getattr(self, factory)(), inputs)
        self.assertTrue(
            any("_run_on_gpu" in segment for segment in segments),
            f"expected a PyTorch segment, got {segments}",
        )
        self.assertEqual(
            len(messages), 1, f"expected exactly one report, got {messages}"
        )
        self.assertIn(expected_operator, messages[0])
        self.assertIn(expected_reason, messages[0])

    def test_fully_supported_module_is_silent(self):
        """Nothing fell back, so there is nothing to report."""
        inputs = [torch.randn(8, 64, device="cuda")]
        segments, messages = self._compile(self._fully_supported_module(), inputs)
        self.assertFalse(
            any("_run_on_gpu" in segment for segment in segments),
            f"expected no PyTorch segment, got {segments}",
        )
        self.assertEqual(messages, [])

    @parameterized.expand([("fast", True), ("global", False)])
    def test_requested_fallback_is_reported(self, _, use_fast_partitioner):
        inputs = [torch.randn(8, 64, device="cuda")]
        segments, messages = self._compile(
            self._fully_supported_module(),
            inputs,
            torch_executed_ops={"torch.ops.aten.relu.default"},
            use_fast_partitioner=use_fast_partitioner,
        )
        self.assertEqual(len(messages), 1)
        self.assertIn("torch.ops.aten.relu.default + Operator Count: 6", messages[0])
        self.assertIn("excluded by torch_executed_ops", messages[0])

    def test_global_partitioner_reports_too(self):
        """The global partitioner is the automatic fallback when the fast one raises, so a
        user reaches it exactly when they most need to be told something happened."""
        inputs = [torch.randn(8, 64, device="cuda")]
        segments, messages = self._compile(
            self._impure_fallback_module(), inputs, use_fast_partitioner=False
        )
        self.assertEqual(
            len(messages), 1, f"expected exactly one report, got {messages}"
        )

    def test_global_partitioner_silent_on_fully_supported(self):
        """The global partitioner asks about placeholder and output nodes as well as
        operators. Recording those made a fully supported graph report its own inputs and
        outputs as fallbacks, so this warns falsely without the callable-node guard."""
        inputs = [torch.randn(8, 64, device="cuda")]
        segments, messages = self._compile(
            self._fully_supported_module(), inputs, use_fast_partitioner=False
        )
        self.assertFalse(
            any("_run_on_gpu" in segment for segment in segments),
            f"expected no PyTorch segment, got {segments}",
        )
        self.assertEqual(messages, [])

    @parameterized.expand([("fast", True), ("global", False)])
    def test_requested_fallback_by_target_is_reported(self, _, use_fast_partitioner):
        inputs = [torch.randn(8, 64, device="cuda")]
        segments, messages = self._compile(
            self._fully_supported_module(),
            inputs,
            torch_executed_ops={torch.ops.aten.relu.default},
            use_fast_partitioner=use_fast_partitioner,
        )
        self.assertEqual(len(messages), 1)
        self.assertIn("torch.ops.aten.relu.default + Operator Count: 6", messages[0])
        self.assertIn("excluded by torch_executed_ops", messages[0])

    @parameterized.expand([("fast", True), ("global", False)])
    def test_mixed_fallback_reasons(self, _, use_fast_partitioner):
        inputs = [torch.randn(8, 64, device="cuda")]
        _, messages = self._compile(
            self._impure_fallback_module(),
            inputs,
            torch_executed_ops={torch.ops.aten.relu.default},
            use_fast_partitioner=use_fast_partitioner,
        )
        self.assertEqual(len(messages), 1)
        self.assertIn(
            "torch.ops.aten.rand_like.default + Operator Count: 1 "
            "(Reasons: no validated TensorRT converter)",
            messages[0],
        )
        self.assertIn(
            "torch.ops.aten.relu.default + Operator Count: 6 "
            "(Reasons: excluded by torch_executed_ops)",
            messages[0],
        )
        self.assertLess(messages[0].index("rand_like"), messages[0].index("relu"))

    def test_warning_level_suppresses_summary(self):
        inputs = [torch.randn(8, 64, device="cuda")]
        _, messages = self._compile(
            self._impure_fallback_module(), inputs, log_level=logging.WARNING
        )
        self.assertEqual(messages, [])


if __name__ == "__main__":
    run_tests()
