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

    def test_display_handles_graph_with_no_operators(self):
        """No computational nodes reports 0% rather than dividing by zero"""
        dryrun_stats_display(DryRunTracker(), False)


if __name__ == "__main__":
    run_tests()
