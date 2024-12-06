import torch


class SAM2FullModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.image_encoder = model.forward_image
        self._prepare_backbone_features = model._prepare_backbone_features
        self.directly_add_no_mem_embed = model.directly_add_no_mem_embed
        self.no_mem_embed = model.no_mem_embed
        self._features = None

        self.prompt_encoder = model.sam_prompt_encoder
        self.mask_decoder = model.sam_mask_decoder

        self._bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]

    def forward(self, image, point_coords, point_labels):
        backbone_out = self.image_encoder(image)
        _, vision_feats, _, _ = self._prepare_backbone_features(backbone_out)

        if self.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

        high_res_features = [
            feat_level[-1].unsqueeze(0) for feat_level in features["high_res_feats"]
        ]

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=(point_coords, point_labels), boxes=None, masks=None
        )

        low_res_masks, iou_predictions, _, _ = self.mask_decoder(
            image_embeddings=features["image_embed"][-1].unsqueeze(0),
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=point_coords.shape[0] > 1,
            high_res_features=high_res_features,
        )

        out = {"low_res_masks": low_res_masks, "iou_predictions": iou_predictions}
        return out
