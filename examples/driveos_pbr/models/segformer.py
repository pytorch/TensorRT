# SPDX-License-Identifier: BSD-3-Clause
"""The existing Transformers SegFormer-B0 model, with raw logits as output."""

import hashlib
import torch
from transformers import SegformerConfig, SegformerForSemanticSegmentation


class SegmentationLogits(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return (self.model(pixel_values=pixel_values, return_dict=False)[0],)


def load_model(checkpoint):
    weights = checkpoint / "pytorch_model.bin"
    with weights.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    if digest != "027ed78d8ff9c535df5a66361c92b9673cca3cb923cbeb8c5802385b6b93194c":
        raise ValueError("Unexpected SegFormer checkpoint")
    config = SegformerConfig.from_json_file(checkpoint / "config.json")
    if config.hidden_sizes != [32, 64, 160, 256] or config.num_labels != 19:
        raise ValueError("Expected Cityscapes SegFormer-B0")
    model = SegformerForSemanticSegmentation(config)
    model.load_state_dict(
        torch.load(weights, map_location="cpu", weights_only=True), strict=True
    )
    return SegmentationLogits(model).eval()
