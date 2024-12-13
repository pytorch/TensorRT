"""
.. _torch_export_sam2:

Compiling SAM2 using the dynamo backend
==========================================================

This script illustrates Torch-TensorRT workflow with dynamo backend on popular SAM2 model."""

# %%
# Imports and Model Definition
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch_tensorrt
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam_components import SAM2FullModel

matplotlib.use("Agg")

# %%
# Segment Anything Model 2 is a foundation model towards solving promptable visual segmentation in images and videos.
# Load the facebook/sam2-hiera-large pretrained model using SAM2ImagePredictor class.
# SAM2ImagePredictor class provides utilities to preprocess images, store image features (via set_image function)
# and predict the masks (via predict function)

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

# %%
# To ensure we export the entire model (image encoder and mask predictor) components successfully, we create a
# standalone module ``SAM2FullModel`` which uses these utilities from ``SAM2ImagePredictor`` class.
# SAM2Full model performs feature extraction and mask prediction in a single step instead of two step process of
# SAM2ImagePredictor (SAM2ImagePredictor.set_image and SAM2ImagePredictor.predict)


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


# %%
# Initialize the SAM2FullModel with the pretrained weights. Since we already initialized
# SAM2ImagePredictor, we can directly use the model from it (predictor.model). We cast the model
# to FP16 precision for faster performance.
encoder = predictor.model.eval().cuda()
sam_model = SAM2FullModel(encoder.half()).eval().cuda()

# %%
# Load a sample image provided in the repository.
input_image = Image.open("./truck.jpg").convert("RGB")

# %%
# Define the preprocessing components which include applying transformations on the input image
# and transforming given point coordinates. We use the SAM2Transforms available via the SAM2ImagePredictor class.
# To read more about the transforms, refer to https://github.com/facebookresearch/sam2/blob/main/sam2/utils/transforms.py


def preprocess_inputs(image, predictor):
    w, h = image.size
    orig_hw = [(h, w)]
    input_image = predictor._transforms(np.array(image))[None, ...].to("cuda:0")

    point_coords = torch.tensor([[500, 375]], dtype=torch.float).to("cuda:0")
    point_labels = torch.tensor([1], dtype=torch.int).to("cuda:0")

    point_coords = torch.as_tensor(
        point_coords, dtype=torch.float, device=predictor.device
    )
    unnorm_coords = predictor._transforms.transform_coords(
        point_coords, normalize=True, orig_hw=orig_hw[0]  # predictor._orig_hw[img_idx]
    )
    labels = torch.as_tensor(point_labels, dtype=torch.int, device=predictor.device)
    if len(unnorm_coords.shape) == 2:
        unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]

    input_image = input_image.half()
    unnorm_coords = unnorm_coords.half()

    return (input_image, unnorm_coords, labels)


# %%
# Define the postprocessing components which include plotting and visualizing masks and points.
# We use the SAM2Transforms to post process these masks and sort them via confidence score.


def postprocess_masks(out, predictor, image):
    """Postprocess low-resolution masks and convert them for visualization."""
    orig_hw = (image.size[1], image.size[0])  # (height, width)
    masks = predictor._transforms.postprocess_masks(out["low_res_masks"], orig_hw)
    masks = (masks > 0.0).squeeze(0).cpu().numpy()
    scores = out["iou_predictions"].squeeze(0).cpu().numpy()
    sorted_indices = np.argsort(scores)[::-1]
    return masks[sorted_indices], scores[sorted_indices]


def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [
            cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours
        ]
        mask_image = cv2.drawContours(
            mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2
        )
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def visualize_masks(
    image, masks, scores, point_coords, point_labels, title_prefix="", save=True
):
    """Visualize and save masks overlaid on the original image."""
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(point_coords, point_labels, plt.gca())
        plt.title(f"{title_prefix} Mask {i + 1}, Score: {score:.3f}", fontsize=18)
        plt.axis("off")
        plt.savefig(f"{title_prefix}_output_mask_{i + 1}.png")
        plt.close()


# %%
# Preprocess the inputs
# torchtrt_inputs contain (input_image, unnormalized_coordinates and labels)
torchtrt_inputs = preprocess_inputs(input_image, predictor)

# %%
# Export the model in non-strict mode and perform Torch-TensorRT compilation in FP16 precision.
# We enable FP32 matmul accumulation using use_fp32_acc=True to preserve accuracy with the original Pytorch model.
exp_program = torch.export.export(sam_model, torchtrt_inputs, strict=False)
trt_model = torch_tensorrt.dynamo.compile(
    exp_program,
    inputs=torchtrt_inputs,
    min_block_size=1,
    enabled_precisions={torch.float16},
    use_fp32_acc=True,
)
trt_out = trt_model(*torchtrt_inputs)

# %%
# Post process the outputs of Torch-TensorRT and visualize the masks using the post processing
# components provided above. The outputs should be stored in your current directory.
# You can also view them at https://github.com/pytorch/TensorRT/tree/sam/examples/dynamo/sam#sam2-inference-results-with-torch-tensorrt
trt_masks, trt_scores = postprocess_masks(trt_out, predictor, input_image)
visualize_masks(
    input_image,
    trt_masks,
    trt_scores,
    torch.tensor([[500, 375]]),
    torch.tensor([1]),
    title_prefix="Torch-TRT",
)
