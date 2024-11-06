import torch


class ImageEncoder(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]

    def forward(self, torch_img: torch.Tensor):
        backbone_out = self.module.forward_image(torch_img)
        _, vision_feats, _, _ = self.module._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.module.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.module.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        image_features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        return image_features
