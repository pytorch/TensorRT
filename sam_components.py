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


class HeadModule(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module


    def forward(self, image_embedding, point_coords, point_labels, mask_input, high_res_features):
        sparse_embeddings, dense_embeddings = self.module.sam_prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None, 
            masks=mask_input, 
        )

        batched_mode = point_coords.shape[0] > 1

        low_res_masks, iou_predictions, _, _ = self.module.sam_mask_decoder(
            image_embeddings=image_embedding, 
            image_pe=self.module.sam_prompt_encoder.get_dense_pe(), 
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )

        out = {"low_res_masks": low_res_masks, "iou_predictions": iou_predictions}
        return out
    

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

    def forward(self, image, point_coords, point_labels, mask_input):
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
            feat_level[-1].unsqueeze(0)
            for feat_level in features["high_res_feats"]
        ]

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=(point_coords, point_labels), boxes=None, masks=mask_input
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