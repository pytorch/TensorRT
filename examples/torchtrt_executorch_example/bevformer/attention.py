# SPDX-License-Identifier: Apache-2.0
# Adapted from BEVFormer and MMCV. Copyright (c) OpenMMLab. All rights reserved.
# BEVFormer modifications by Zhiqi Li. Original licenses are under upstream/.
# This adaptation uses fixed feature shapes and dense, masked camera queries
# to remove custom extensions and data-dependent rebatching from inference.
"""Exportable BEVFormer-Tiny spatial attention, NOT a complete detector.

Only FP32 evaluation, one feature level, four height anchors and batch one
are initially validated. Dense camera evaluation trades extra work/memory
for fixed shapes. No performance improvement is claimed.
"""

import torch
from torch import nn
from torch.nn import functional as F


class DeformableAttention3D(nn.Module):
    def __init__(self, feature_hw=(15, 25)):
        super().__init__()
        self.feature_hw = tuple(feature_hw)
        self.embed_dims = 256
        self.num_heads = 8
        self.num_points = 8
        self.num_levels = 1
        self.batch_first = True
        self.sampling_offsets = nn.Linear(256, 8 * 8 * 2)
        self.attention_weights = nn.Linear(256, 8 * 8)
        self.value_proj = nn.Linear(256, 256)

    def forward(self, query, value, reference_points, **_unused):
        batch, queries, _ = query.shape
        height, width = self.feature_hw
        values = self.value_proj(value).reshape(batch, height * width, 8, 32)
        offsets = self.sampling_offsets(query).reshape(batch, queries, 8, 1, 8, 2)
        normalizer = offsets.new_tensor([width, height])
        # Eight sampling points: two offsets per each of four height anchors.
        offsets = (offsets / normalizer).reshape(batch, queries, 8, 1, 2, 4, 2)
        locations = reference_points[:, :, None, None, None, :, :] + offsets
        locations = locations.reshape(batch, queries, 8, 8, 2)
        grids = (
            (locations * 2 - 1).permute(0, 2, 1, 3, 4).reshape(batch * 8, queries, 8, 2)
        )
        images = values.flatten(2).transpose(1, 2).reshape(batch * 8, 32, height, width)
        sampled = F.grid_sample(
            images, grids, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        weights = self.attention_weights(query).reshape(batch, queries, 8, 8)
        weights = weights.softmax(-1).transpose(1, 2).reshape(batch * 8, 1, queries, 8)
        output = (sampled * weights).sum(-1).reshape(batch, 256, queries)
        return output.transpose(1, 2).contiguous()


class SpatialCrossAttention(nn.Module):
    def __init__(self, feature_hw=(15, 25)):
        super().__init__()
        self.embed_dims = 256
        self.num_cams = 6
        self.deformable_attention = DeformableAttention3D(feature_hw)
        self.output_proj = nn.Linear(256, 256)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, value, reference_points_cam, visibility):
        batch, queries, channels = query.shape
        # Deliberately evaluate all queries for each camera, then mask results.
        # Each query's attention is independent, so omitted invisible queries
        # can be evaluated without affecting visible queries. This needs more
        # temporary memory than upstream's sparse rebatching.
        camera_queries = query[:, None].expand(batch, 6, queries, channels)
        camera_values = value.permute(2, 0, 1, 3).reshape(batch * 6, -1, channels)
        camera_refs = reference_points_cam.permute(1, 0, 2, 3, 4)
        sampled = self.deformable_attention(
            camera_queries.reshape(batch * 6, queries, channels),
            camera_values,
            camera_refs.reshape(batch * 6, queries, 4, 2),
        ).reshape(batch, 6, queries, channels)
        # Visibility is float32 at this test interface, for the existing native
        # runner's float-only inputs; upstream's mask is boolean.
        visible = (visibility > 0).any(-1).permute(1, 0, 2)
        sampled = torch.where(visible[..., None], sampled, 0.0)
        count = visible.sum(1).clamp(min=1).to(sampled.dtype)
        output = sampled.sum(1) / count[..., None]
        return query + self.dropout(self.output_proj(output))
