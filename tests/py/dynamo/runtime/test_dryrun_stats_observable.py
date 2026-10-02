# type: ignore

import unittest

import torch
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo._DryRunTracker import DryRunTracker, dryrun_stats_display
from torch_tensorrt.dynamo.observer import ObserveContext


class Add(torch.nn.Module):
    def forward(self, x):
        return x + x


def capture_trackers(trackers):
    def capture(ctx: ObserveContext) -> None:
        trackers.append(ctx.args[0])

    return capture


def exported_add():
    x = torch.randn(2, 3, device="cuda")
    return torch.export.export(Add().cuda().eval(), (x,)), x


@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestDryRunStatsObservable(TestCase):
    def test_observer_captures_tracker(self):
        exp_program, x = exported_add()
        trackers = []

        with dryrun_stats_display.observers.pre.add(capture_trackers(trackers)):
            torch_tensorrt.dynamo.compile(
                exp_program,
                arg_inputs=(x,),
                dryrun=True,
                min_block_size=1,
            )

        self.assertEqual(len(trackers), 1)
        tracker = trackers[0]
        self.assertIsInstance(tracker, DryRunTracker)
        self.assertGreater(tracker.total_ops_in_graph, 0)
        self.assertGreaterEqual(tracker.supported_ops_in_graph, 0)
        self.assertLessEqual(tracker.supported_ops_in_graph, tracker.total_ops_in_graph)
        self.assertGreaterEqual(tracker.tensorrt_graph_count, 1)
        self.assertEqual(len(tracker.per_subgraph_data), tracker.tensorrt_graph_count)

    def test_observer_fires_with_no_supported_ops(self):
        """compile_module returns before partitioning, but still emits a tracker"""
        exp_program, x = exported_add()
        trackers = []

        with dryrun_stats_display.observers.pre.add(capture_trackers(trackers)):
            torch_tensorrt.dynamo.compile(
                exp_program,
                arg_inputs=(x,),
                dryrun=True,
                min_block_size=1,
                torch_executed_ops={"torch.ops.aten.add.Tensor"},
            )

        self.assertEqual(len(trackers), 1)
        tracker = trackers[0]
        self.assertIsInstance(tracker, DryRunTracker)
        self.assertGreater(tracker.total_ops_in_graph, 0)
        self.assertEqual(tracker.supported_ops_in_graph, 0)
        self.assertEqual(tracker.tensorrt_graph_count, 0)
        self.assertEqual(tracker.per_subgraph_data, [])
        self.assertEqual(tracker.unsupported_ops, {"torch.ops.aten.add.Tensor": 1})
        self.assertTrue(
            any("add" in node for node in tracker.to_run_in_torch),
            msg=f"Expected the add node in to_run_in_torch, got {tracker.to_run_in_torch}",
        )

    def test_post_observer_receives_tracker(self):
        exp_program, x = exported_add()
        trackers = []

        with dryrun_stats_display.observers.post.add(capture_trackers(trackers)):
            torch_tensorrt.dynamo.compile(
                exp_program,
                arg_inputs=(x,),
                dryrun=True,
                min_block_size=1,
            )

        self.assertEqual(len(trackers), 1)
        self.assertIsInstance(trackers[0], DryRunTracker)

    def test_observer_deregistered_after_context(self):
        exp_program, x = exported_add()
        trackers = []

        with dryrun_stats_display.observers.pre.add(capture_trackers(trackers)):
            pass

        torch_tensorrt.dynamo.compile(
            exp_program,
            arg_inputs=(x,),
            dryrun=True,
            min_block_size=1,
        )

        self.assertEqual(trackers, [])

    def _dryrun_unsupported_ops(self, module, x, **kwargs):
        trackers = []
        with dryrun_stats_display.observers.pre.add(capture_trackers(trackers)):
            torch_tensorrt.dynamo.compile(
                torch.export.export(module, (x,)),
                arg_inputs=(x,),
                dryrun=True,
                min_block_size=1,
                **kwargs,
            )
        self.assertEqual(len(trackers), 1)
        return trackers[0].unsupported_ops

    def test_report_lists_operator_with_side_effects(self):
        """rand_like has side effects, so the support counters used to decide full support
        leave it out. The report must still list it, alone and next to a supported op.
        """

        class RandOnly(torch.nn.Module):
            def forward(self, x):
                return torch.rand_like(x)

        class RandBesideSupported(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x) + torch.rand_like(x)

        x = torch.randn(2, 3, device="cuda")
        for module in (RandOnly(), RandBesideSupported()):
            for use_fast_partitioner in (True, False):
                with self.subTest(
                    module=type(module).__name__, fast=use_fast_partitioner
                ):
                    unsupported = self._dryrun_unsupported_ops(
                        module.cuda().eval(),
                        x,
                        use_fast_partitioner=use_fast_partitioner,
                    )
                    self.assertEqual(
                        unsupported.get("torch.ops.aten.rand_like.default"), 1
                    )

    def test_report_lists_operators_not_buffers(self):
        """A buffer TensorRT cannot hold is not an operator, so it must not be listed as
        one. The global partitioner refuses the buffer node itself."""

        class HighRankBuffer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("high", torch.ones((1,) * 9))

            def forward(self, x):
                return torch.relu(x), x + self.high

        x = torch.ones(2, device="cuda")
        for use_fast_partitioner in (True, False):
            with self.subTest(fast=use_fast_partitioner):
                unsupported = self._dryrun_unsupported_ops(
                    HighRankBuffer().cuda().eval(),
                    x,
                    use_fast_partitioner=use_fast_partitioner,
                )
                self.assertEqual(unsupported, {"torch.ops.aten.add.Tensor": 1})

    def test_display_handles_graph_with_no_operators(self):
        """No computational nodes reports 0% rather than dividing by zero"""
        dryrun_stats_display(DryRunTracker(), False)


if __name__ == "__main__":
    run_tests()
