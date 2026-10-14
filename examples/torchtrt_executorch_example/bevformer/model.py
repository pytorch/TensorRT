# SPDX-License-Identifier: Apache-2.0
# Inference adaptation of BEVFormer and OpenMMLab ResNet/FPN/transformers.
# Copyright (c) OpenMMLab. All rights reserved. BEVFormer by Zhiqi Li.
"""Fixed-shape BEVFormer-Tiny tensor core; experimental, not an accuracy result.

Inputs are PREPROCESSED images, resized lidar-to-image matrices, CAN-bus
metadata, previous BEV features and a history-valid flag. Dataset parsing,
scene/pose bookkeeping and detection postprocessing belong to the caller.
All learned parameter names match the official Tiny checkpoint.
"""

import hashlib
import math
from pathlib import Path

import torch
from attention import SpatialCrossAttention
from torch import nn
from torch.nn import functional as F


class Bottleneck(nn.Module):
    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        self.relu = nn.ReLU()
        self.downsample = None
        if stride != 1 or in_channels != channels * 4:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, channels * 4, 1, stride, bias=False),
                nn.BatchNorm2d(channels * 4),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return self.relu(self.bn3(self.conv3(out)) + identity)


class ResNetC5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        in_channels = 64
        for index, (channels, count) in enumerate(
            zip((64, 128, 256, 512), (3, 4, 6, 3))
        ):
            blocks = [Bottleneck(in_channels, channels, 1 if index == 0 else 2)]
            blocks.extend(Bottleneck(channels * 4, channels) for _ in range(count - 1))
            setattr(self, f"layer{index + 1}", nn.Sequential(*blocks))
            in_channels = channels * 4

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return (self.layer4(x),)


class ConvModule(nn.Module):
    def __init__(self, in_channels, kernel):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 256, kernel, padding=kernel // 2)

    def forward(self, x):
        return self.conv(x)


class SingleLevelFPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lateral_convs = nn.ModuleList([ConvModule(2048, 1)])
        self.fpn_convs = nn.ModuleList([ConvModule(256, 3)])

    def forward(self, inputs):
        return (self.fpn_convs[0](self.lateral_convs[0](inputs[0])),)


class LearnedPosition(nn.Module):
    def __init__(self):
        super().__init__()
        self.row_embed = nn.Embedding(50, 128)
        self.col_embed = nn.Embedding(50, 128)

    def forward(self):
        x = self.col_embed.weight[None].expand(50, 50, 128)
        y = self.row_embed.weight[:, None].expand(50, 50, 128)
        return torch.cat((x, y), -1).reshape(1, 2500, 256)


def sample_bev(values, locations, weights):
    """One 50x50 level, eight heads; equivalent to MMCV's PyTorch sampler."""
    batch, queries, heads, points, _ = locations.shape
    images = values.reshape(batch, 2500, 256).transpose(1, 2)
    images = images.reshape(batch * heads, 32, 50, 50)
    grids = (locations * 2 - 1).permute(0, 2, 1, 3, 4)
    grids = grids.reshape(batch * heads, queries, points, 2)
    sampled = F.grid_sample(images, grids, align_corners=False)
    weights = weights.transpose(1, 2).reshape(batch * heads, 1, queries, points)
    return (sampled * weights).sum(-1).reshape(batch, 256, queries).transpose(1, 2)


class TemporalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dims, self.num_heads, self.num_levels = 256, 8, 1
        self.num_points, self.num_bev_queue = 4, 2
        self.batch_first, self.im2col_step = True, 64
        self.sampling_offsets = nn.Linear(512, 128)
        self.attention_weights = nn.Linear(512, 64)
        self.value_proj = nn.Linear(256, 256)
        self.output_proj = nn.Linear(256, 256)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, value, query_pos, reference_points):
        # Batch one: value is [previous, current], with two independent queues.
        combined = torch.cat((value[:1], query + query_pos), -1)
        offsets = self.sampling_offsets(combined).reshape(1, 2500, 8, 2, 4, 2)
        offsets = offsets.permute(0, 3, 1, 2, 4, 5).reshape(2, 2500, 8, 4, 2)
        weights = self.attention_weights(combined).reshape(1, 2500, 8, 2, 4)
        weights = weights.softmax(-1).permute(0, 3, 1, 2, 4).reshape(2, 2500, 8, 4)
        locations = reference_points[:, :, :, None, :] + offsets / 50.0
        sampled = sample_bev(self.value_proj(value), locations, weights).mean(
            0, keepdim=True
        )
        return query + self.dropout(self.output_proj(sampled))


class DecoderCrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dims, self.num_heads, self.num_levels, self.num_points = 256, 8, 1, 4
        self.batch_first, self.im2col_step = True, 64
        self.sampling_offsets = nn.Linear(256, 64)
        self.attention_weights = nn.Linear(256, 32)
        self.value_proj = nn.Linear(256, 256)
        self.output_proj = nn.Linear(256, 256)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, value, query_pos, reference_points):
        positioned = query + query_pos
        offsets = self.sampling_offsets(positioned).reshape(1, 900, 8, 4, 2)
        weights = self.attention_weights(positioned).reshape(1, 900, 8, 4).softmax(-1)
        locations = reference_points[:, :, :, None, :] + offsets / 50.0
        sampled = sample_bev(self.value_proj(value), locations, weights)
        return query + self.dropout(self.output_proj(sampled))


class QuerySelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(256, 8, dropout=0.1)

    def forward(self, query, query_pos):
        positioned = (query + query_pos).transpose(0, 1)
        output = self.attn(positioned, positioned, query.transpose(0, 1))[0]
        return query + output.transpose(0, 1)


class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Dropout(0.1)),
            nn.Linear(512, 256),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return x + self.layers(x)


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attentions = nn.ModuleList(
            [TemporalSelfAttention(), SpatialCrossAttention()]
        )
        self.ffns = nn.ModuleList([FFN()])
        self.norms = nn.ModuleList(nn.LayerNorm(256) for _ in range(3))

    def forward(
        self, query, position, temporal_value, temporal_refs, value, refs, mask
    ):
        query = self.norms[0](
            self.attentions[0](query, temporal_value, position, temporal_refs)
        )
        query = self.norms[1](self.attentions[1](query, value, refs, mask))
        return self.norms[2](self.ffns[0](query))


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attentions = nn.ModuleList([QuerySelfAttention(), DecoderCrossAttention()])
        self.ffns = nn.ModuleList([FFN()])
        self.norms = nn.ModuleList(nn.LayerNorm(256) for _ in range(3))

    def forward(self, query, position, bev, reference):
        query = self.norms[0](self.attentions[0](query, position))
        query = self.norms[1](
            self.attentions[1](query, bev, position, reference[:, :, None, :2])
        )
        return self.norms[2](self.ffns[0](query))


class LayerSequence(nn.Module):
    def __init__(self, kind, count):
        super().__init__()
        self.layers = nn.ModuleList(kind() for _ in range(count))


class PerceptionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # The checkpoint retains four level embeddings although Tiny uses one.
        self.level_embeds = nn.Parameter(torch.empty(4, 256))
        self.cams_embeds = nn.Parameter(torch.empty(6, 256))
        self.encoder = LayerSequence(EncoderLayer, 3)
        self.decoder = LayerSequence(DecoderLayer, 6)
        self.reference_points = nn.Linear(256, 3)
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU()
        )
        self.can_bus_mlp.add_module("norm", nn.LayerNorm(256))


class BEVFormerHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.code_weights = nn.Parameter(torch.ones(10), requires_grad=False)
        self.positional_encoding = LearnedPosition()
        self.bev_embedding = nn.Embedding(2500, 256)
        self.query_embedding = nn.Embedding(900, 512)
        self.transformer = PerceptionTransformer()
        self.cls_branches = nn.ModuleList(
            nn.Sequential(
                nn.Linear(256, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, 10),
            )
            for _ in range(6)
        )
        self.reg_branches = nn.ModuleList(
            nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 10),
            )
            for _ in range(6)
        )


def inverse_sigmoid(x):
    x = x.clamp(0, 1)
    return torch.log(x.clamp(min=1e-5) / (1 - x).clamp(min=1e-5))


class Geometry(nn.Module):
    def __init__(self):
        super().__init__()
        centers = torch.arange(50, dtype=torch.float32) + 0.5
        y, x = torch.meshgrid(centers, centers, indexing="ij")
        rotation_grid = torch.stack((x - 25, y - 25, torch.ones_like(x)), -1)
        self.register_buffer("xy_centers", torch.stack((x, y), -1), persistent=False)
        self.register_buffer("rotation_grid", rotation_grid, persistent=False)

    def forward(self, lidar2img, can_bus, previous):
        refs, visibility = self.project(lidar2img)
        return (
            refs,
            visibility,
            self.rotate(previous, can_bus[:, -1]),
            self.reference2d(),
            self.shift(can_bus),
        )

    def reference2d(self):
        return (self.xy_centers / 50).reshape(1, 2500, 1, 2)

    def project(self, lidar2img):
        # Build normalized points on the execution device, as upstream does.
        # Computing world coordinates on CPU and copying them changes rounding
        # at strict image-boundary visibility tests (0 < coordinate < 1).
        zs = torch.linspace(0.5, 7.5, 4, device=lidar2img.device) / 8
        xy = self.xy_centers.reshape(1, 2500, 2).expand(4, 2500, 2) / 50
        z = zs[:, None, None].expand(4, 2500, 1)
        points = torch.cat((xy * 102.4 - 51.2, z * 8 - 5, torch.ones_like(z)), -1)
        points = points.reshape(4, 1, 1, 2500, 4).repeat(1, 1, 6, 1, 1).unsqueeze(-1)
        projected = torch.matmul(
            lidar2img.reshape(1, 1, 6, 1, 4, 4).repeat(4, 1, 1, 2500, 1, 1), points
        ).squeeze(-1)
        depth = projected[..., 2:3]
        xy = projected[..., :2] / depth.clamp(min=1e-5)
        xy = xy / xy.new_tensor([800, 480])
        visible = (depth[..., 0] > 1e-5) & (xy > 0).all(-1) & (xy < 1).all(-1)
        return xy.permute(2, 1, 3, 0, 4), visible.permute(2, 1, 3, 0).float()

    def rotate(self, prev_bev, degrees):
        # Preserve upstream rotate_center=[100,100], including for Tiny's 50x50
        # grid. Changing it to [25,25] would change the pretrained model semantics.
        angle = degrees * (math.pi / 180)
        c, s = angle.cos(), angle.sin()
        row0 = torch.stack((c, -s, 75 - 75 * c + 75 * s), -1)
        row1 = torch.stack((s, c, 75 - 75 * s - 75 * c), -1)
        theta = torch.stack((row0, row1), 1)
        grid = (
            torch.matmul(self.rotation_grid.reshape(1, 2500, 3), theta.transpose(1, 2))
            / 25
        )
        image = prev_bev.transpose(1, 2).reshape(1, 256, 50, 50)
        rotated = F.grid_sample(
            image, grid.reshape(1, 50, 50, 2), mode="nearest", align_corners=False
        )
        return rotated.flatten(2).transpose(1, 2)

    def shift(self, can_bus):
        dx, dy, yaw = can_bus[:, 0], can_bus[:, 1], can_bus[:, -2]
        # Algebraically identical to upstream's length/atan2/cos/sin formula.
        return (
            torch.stack(
                (dx * yaw.sin() - dy * yaw.cos(), dx * yaw.cos() + dy * yaw.sin()), -1
            )
            / 102.4
        )


class BEVFormerTiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_backbone = ResNetC5()
        self.img_neck = SingleLevelFPN()
        self.pts_bbox_head = BEVFormerHead()
        self.geometry = Geometry()

    def forward(self, images, lidar2img, can_bus, prev_bev, has_prev):
        assert images.shape == (1, 6, 3, 480, 800)
        assert lidar2img.shape == (1, 6, 4, 4)
        assert can_bus.shape == (1, 18)
        assert prev_bev.shape == (1, 2500, 256)
        assert has_prev.shape == (1, 1, 1)
        features = self.img_neck(self.img_backbone(images.flatten(0, 1)))[0]
        value = features.flatten(2).transpose(1, 2).unsqueeze(2)
        head = self.pts_bbox_head
        transformer = head.transformer
        value = (
            value + transformer.cams_embeds[:, None, None] + transformer.level_embeds[0]
        )
        initial = (
            head.bev_embedding.weight[None] + transformer.can_bus_mlp(can_bus)[:, None]
        )
        position = head.positional_encoding()
        camera_refs, visibility, previous, base_refs, shift = self.geometry(
            lidar2img, can_bus, prev_bev
        )
        shift = shift[:, None, None]
        shifted = torch.where((has_prev > 0)[..., None], base_refs + shift, base_refs)
        temporal_refs = torch.cat((shifted, base_refs), 0)
        query = initial
        for layer in transformer.encoder.layers:
            # Upstream captures [previous, INITIAL query] once when history is
            # present, but constructs [CURRENT query, CURRENT query] per layer
            # when prev_bev is None. A tensor flag keeps both paths exportable.
            temporal_value = torch.cat(
                (
                    torch.where(has_prev > 0, previous, query),
                    torch.where(has_prev > 0, initial, query),
                ),
                0,
            )
            query = layer(
                query,
                position,
                temporal_value,
                temporal_refs,
                value,
                camera_refs,
                visibility,
            )
        bev = query
        query_pos, query = head.query_embedding.weight.split(256, dim=-1)
        query_pos, query = query_pos[None], query[None]
        reference = transformer.reference_points(query_pos).sigmoid()
        for index, layer in enumerate(transformer.decoder.layers):
            query = layer(query, query_pos, bev, reference)
            raw = head.reg_branches[index](query)
            inverse_ref = inverse_sigmoid(reference)
            xy = (raw[..., :2] + inverse_ref[..., :2]).sigmoid()
            z = (raw[..., 4:5] + inverse_ref[..., 2:3]).sigmoid()
            reference = torch.cat((xy, z), -1).detach()
        boxes = torch.cat(
            (xy * 102.4 - 51.2, raw[..., 2:4], z * 8 - 5, raw[..., 5:]), -1
        )
        return head.cls_branches[-1](query), boxes, bev


def load_model(checkpoint: Path):
    expected = "7305046dbaa4fe8b1fa6d6acb9e0e3d605a70a3c473f763e936103428d2b2f12"
    with checkpoint.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
            raise ValueError(
                "Expected the official BEVFormer-Tiny checkpoint; hash mismatch"
            )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    model = BEVFormerTiny().eval()
    model.load_state_dict(state, strict=True)
    return model


def example_inputs(seed=7, history=False):
    """Synthetic tensors for correctness/export tests, NOT dataset validation."""
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(1, 6, 3, 480, 800, generator=generator)
    # Synthetic six-camera rig with positive-forward camera depth and diverse
    # visibility. Matrices already include the resized focal lengths/centers.
    matrices = []
    for index in range(6):
        angle = index * math.pi / 3
        c, s = math.cos(angle), math.sin(angle)
        extrinsic = torch.tensor(
            [[-s, c, 0, 0], [0, 0, -1, 1.5], [c, s, 0, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
        )
        intrinsic = torch.tensor(
            [[400, 0, 400, 0], [0, 400, 225, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
        )
        matrices.append(intrinsic @ extrinsic)
    can_bus = torch.zeros(1, 18)
    if history:
        can_bus[0, 0], can_bus[0, 1] = 0.4, -0.1
        can_bus[0, -2], can_bus[0, -1] = 0.3, 2.0
    previous = (
        torch.randn(1, 2500, 256, generator=generator)
        if history
        else torch.zeros(1, 2500, 256)
    )
    return (
        images,
        torch.stack(matrices)[None],
        can_bus,
        previous,
        torch.full((1, 1, 1), float(history)),
    )
