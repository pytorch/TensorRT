# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Compile-time parent mapping and runtime contract checks; no NCCL/GPU needed."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch.distributed as dist


class TestNativeCollectiveParent(unittest.TestCase):
    def translate(self, parent, ranks, *, supported=True, ctx=None):
        from torch_tensorrt import _features
        from torch_tensorrt.dynamo.conversion.impl import nccl_ops

        ctx = ctx or SimpleNamespace(native_collective_parent=None)
        features = _features.ENABLED_FEATURES._replace(
            native_trt_collective_subgroups=supported
        )
        with (
            patch(
                "torch_tensorrt.distributed._distributed._active_native_parent",
                return_value=parent,
            ),
            patch.object(_features, "ENABLED_FEATURES", features),
            patch.object(
                nccl_ops,
                "_collective_group_ranks",
                return_value=np.array(ranks, dtype=np.int64),
            ),
        ):
            return nccl_ops._native_collective_ranks(ctx, "op_group", 8).tolist(), ctx

    def test_nonzero_parent_uses_local_ids_without_subset_support(self):
        ranks, ctx = self.translate("4,5,6,7", [4, 5, 6, 7], supported=False)
        self.assertEqual(ranks, [0, 1, 2, 3])
        self.assertEqual(ctx.native_collective_parent, "4,5,6,7")

    def test_preserve_both_parent_and_collective_order(self):
        ranks, _ = self.translate("5,2,7,3", [3, 5, 2])
        self.assertEqual(ranks, [3, 0, 1])

    def test_world_keeps_global_ids(self):
        ranks, ctx = self.translate("", [5, 2])
        self.assertEqual(ranks, [5, 2])
        self.assertEqual(ctx.native_collective_parent, "")

    def test_multiple_contained_groups_share_one_parent(self):
        first, ctx = self.translate("2,3,4,5", [2, 3])
        second, _ = self.translate("2,3,4,5", [2, 4], ctx=ctx)
        self.assertEqual((first, second), ([0, 1], [0, 2]))

    def test_outside_parent_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "outside parent"):
            self.translate("2,3", [0, 2])

    def test_world_operation_is_not_silently_narrowed(self):
        from torch_tensorrt.dynamo.conversion.impl import nccl_ops

        ctx = SimpleNamespace(native_collective_parent=None)
        with patch(
            "torch_tensorrt.distributed._distributed._active_native_parent",
            return_value="2,3",
        ):
            with self.assertRaisesRegex(RuntimeError, "outside parent"):
                nccl_ops._native_collective_ranks(ctx, None, 4)

    def test_subset_and_reordering_need_routing_support(self):
        for ranks in ([2], [3, 2]):
            with (
                self.subTest(ranks=ranks),
                self.assertRaisesRegex(RuntimeError, "TensorRT >= 11.4"),
            ):
                self.translate("2,3", ranks, supported=False)

    def test_parent_cannot_change_during_conversion(self):
        _, ctx = self.translate("2,3", [2, 3])
        with self.assertRaisesRegex(RuntimeError, "parent communicator changed"):
            self.translate("", [2, 3], ctx=ctx)

    def test_global_root_translates_to_parent_rank(self):
        from torch_tensorrt.dynamo.conversion.impl import nccl_ops

        ctx = SimpleNamespace(native_collective_parent="5,2,7,3")
        self.assertEqual(nccl_ops._native_collective_root(ctx, 7), 2)
        with self.assertRaisesRegex(RuntimeError, "outside parent"):
            nccl_ops._native_collective_root(ctx, 0)

    def test_matching_membership_is_accepted_after_group_recreation(self):
        from torch_tensorrt.distributed._distributed import _require_native_parent

        with (
            patch.object(dist, "is_available", return_value=True),
            patch.object(dist, "is_initialized", return_value=True),
            patch.object(dist, "get_rank", return_value=2),
            patch.object(dist, "get_process_group_ranks", return_value=[3, 2]),
        ):
            _require_native_parent(object(), "3,2")
            with self.assertRaisesRegex(RuntimeError, "compiled for parent"):
                _require_native_parent(object(), "2,3")
            with self.assertRaisesRegex(RuntimeError, "compiled for parent"):
                _require_native_parent(object(), "0,1")

    def test_nonmember_rejected_without_querying_handle(self):
        from torch_tensorrt.distributed._distributed import _require_native_parent

        with (
            patch.object(dist, "is_available", return_value=True),
            patch.object(dist, "is_initialized", return_value=True),
            patch.object(dist, "get_process_group_ranks") as ranks,
        ):
            with self.assertRaisesRegex(RuntimeError, "not a member"):
                _require_native_parent(dist.GroupMember.NON_GROUP_MEMBER, "2,3")
            ranks.assert_not_called()


if __name__ == "__main__":
    unittest.main()
