import argparse
import timeit

# Set 'Agg' backend to avoid displaying windows
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


def load_image(file_path):
    """Load and preprocess an image."""
    image = Image.open(file_path).convert("RGB")
    return image


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


def build_model(args):
    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
    model = predictor.model.eval().cuda()

    image = load_image(args.img_path)
    predictor.set_image(image)

    return SAM2FullModel(model.half()).eval().cuda(), predictor


def build_input(args, predictor):
    # Raw input
    image = load_image(args.img_path)
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


def compile_with_torchtrt(model, inputs, args):

    ep = torch.export.export(model, inputs, strict=False)
    trt_gm = torch_tensorrt.dynamo.compile(
        ep,
        inputs=inputs,
        min_block_size=1,
        enabled_precisions={
            torch.float16 if args.precision == "fp16" else torch.float32
        },
        use_fp32_acc=False if args.no_fp32_acc else True,
    )

    return trt_gm


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Run inference on SAM")
    arg_parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        help="Precision of the model to compile for TensorRT",
    )
    arg_parser.add_argument(
        "--mode",
        type=str,
        choices=["encoder", "head", "all"],
        default="all",
        help="Supported options include encoder | prediction_head",
    )
    arg_parser.add_argument(
        "--img_path", type=str, default="./truck.jpg", help="input image path"
    )

    arg_parser.add_argument(
        "--save_visualization",
        action="store_true",
        default=True,
        help="Flag to save visualizations",
    )
    arg_parser.add_argument(
        "--no_fp32_acc", action="store_true", help="Flag to save visualizations"
    )

    args = arg_parser.parse_args()

    pyt_model, predictor = build_model(args)
    inputs = build_input(args, predictor)

    # Torch-TensorRT
    trt_model = compile_with_torchtrt(pyt_model, inputs, args)
    trt_out = trt_model(*inputs)

    # Mask Postprocessing and Visualization
    if args.save_visualization:
        raw_image = load_image(args.img_path)
        trt_masks, trt_scores = postprocess_masks(trt_out, predictor, raw_image)
        visualize_masks(
            raw_image,
            trt_masks,
            trt_scores,
            torch.tensor([[500, 375]]),
            torch.tensor([1]),
            title_prefix="Torch-TRT",
        )
