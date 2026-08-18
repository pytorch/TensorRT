# type: ignore

import unittest

import torch
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo._DryRunTracker import DryRunTracker, dryrun_stats_display
from torch_tensorrt.dynamo.observer import ObserveContext


@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestDryRunStatsObservable(TestCase):
    def test_observer_captures_tracker(self):
        class Add(torch.nn.Module):
            def forward(self, x):
                return x + x

        model = Add().cuda().eval()
        x = torch.randn(2, 3, device="cuda")
        trackers = []

        def capture(ctx: ObserveContext) -> None:
            trackers.append(ctx.args[0])

        with dryrun_stats_display.observers.pre.add(capture):
            torch_tensorrt.dynamo.compile(
                model,
                [x],
                dryrun=True,
                min_block_size=1,
                enabled_precisions={torch.float32},
            )

        self.assertEqual(len(trackers), 1)
        tracker = trackers[0]
        self.assertIsInstance(tracker, DryRunTracker)
        self.assertGreater(tracker.total_ops_in_graph, 0)
        self.assertGreaterEqual(tracker.supported_ops_in_graph, 0)
        self.assertLessEqual(
            tracker.supported_ops_in_graph, tracker.total_ops_in_graph
        )
        self.assertGreaterEqual(tracker.tensorrt_graph_count, 1)
        self.assertEqual(len(tracker.per_subgraph_data), tracker.tensorrt_graph_count)


if __name__ == "__main__":
    run_tests()
