import argparse
import timeit
import numpy as np
import pandas as pd
import torch
import torch_tensorrt
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor

WARMUP_ITER = 10

def recordStats(backend, timings, precision, batch_size=1, compile_time_s=None):
    results = []
    times = np.array(timings)
    speeds = batch_size / times
    time_med = np.median(times)
    speed_med = np.median(speeds)
    stats = {
        "Backend": backend,
        "Precision": precision,
        "Median(FPS)": speed_med,
        "Median-Latency(ms)": time_med * 1000,
    }
    results.append(stats)
    return results

def record_perf(model, backend, input_tensors, precision, iterations, batch_size):
    with torch.no_grad():
        for _ in range(WARMUP_ITER):
            model(*input_tensors)
    torch.cuda.synchronize()
    timings = []
    with torch.no_grad():
        for i in range(iterations):
            start_time = timeit.default_timer()
            _ = model(*input_tensors)
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            timings.append(end_time - start_time)
    return recordStats(backend, timings, precision, batch_size)

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

        # high_res_feats = [
        #     feat.permute(1, 2, 0).view(1, -1, *feat_size)
        #     for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        # ][::-1]

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
        return low_res_masks, iou_predictions
    
        # feats = [
        #     feat.permute(1, 2, 0).view(1, -1, *feat_size)
        #     for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        # ][::-1]
        # image_features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

        # high_res_feats = [
        #     feat.permute(1, 2, 0).view(1, -1, *feat_size)
        #     for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        # ][::-1]
        
        # sparse_embeddings, dense_embeddings = self.prompt_encoder(
        #     points=(point_coords, point_labels), boxes=None, masks=mask_input
        # )
        
        # low_res_masks, iou_predictions, _, _ = self.mask_decoder(
        #     image_embeddings=high_res_feats[-1],
        #     image_pe=self.prompt_encoder.get_dense_pe(),
        #     sparse_prompt_embeddings=sparse_embeddings.half(),
        #     dense_prompt_embeddings=dense_embeddings.half(),
        #     multimask_output=True,
        #     repeat_image=point_coords.shape[0] > 1,
        #     high_res_features=high_res_feats[:-1],
        # )
        # return low_res_masks, iou_predictions

def build_full_model(args):
    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-small")
    predictor.model = predictor.model.eval().cuda()
    if args.precision == "fp16":
        predictor.model = predictor.model.half()
    full_model = SAM2FullModel(predictor.model).eval().cuda()
    if args.precision == "fp16":
        full_model = full_model.half()
    return full_model, predictor

def build_input(args, file_path, predictor):
    image = Image.open(file_path).convert("RGB")
    image = np.array(image)
    input_image = predictor._transforms(image)[None, ...].to("cuda:0")
    point_coords = torch.tensor([[500, 375]], dtype=torch.float).unsqueeze(0).to("cuda:0").half()
    point_labels = torch.tensor([1], dtype=torch.int).unsqueeze(0).to("cuda:0")
    mask_input = torch.zeros(1, 1, 256, 256).to("cuda:0").half()
    if args.precision == "fp16":
        input_image = input_image.half()
    return input_image, point_coords, point_labels, mask_input

def compile_with_torchtrt(model, inputs, precision):
    # ep = torch.export._trace._export(
    #     model,
    #     inputs=inputs,
    #     strict=False,
    #     allow_complex_guards_as_runtime_asserts=True,
    # )
    # ep = torch.export.export(model,inputs)
    # ep = torch.export._trace._export(
    with torch.no_grad():
        ep = torch.export.export(
            model,
            inputs,
            strict=False, # 이게 없으면 왜 에러가 나?
        )
    
    with torch_tensorrt.logging.debug():
        trt_model = torch_tensorrt.dynamo.compile(
            ep,
            inputs=inputs,
            min_block_size=1,
            enabled_precisions={torch.float16 if precision == "fp16" else torch.float32},
        )
    return trt_model

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Run inference on SAM2 Full Model")
    arg_parser.add_argument(
        "--precision", type=str, default="fp16", help="Model precision for TensorRT"
    )
    args = arg_parser.parse_args()

    full_model, predictor = build_full_model(args)
    input_image, point_coords, point_labels, mask_input = build_input(args, "./images/truck.jpg", predictor)

    pyt_results = record_perf(full_model, "Torch", [input_image, point_coords, point_labels, mask_input], args.precision, 3, 1)

    trt_model = compile_with_torchtrt(full_model, (input_image, point_coords, point_labels, mask_input), args.precision)
    
    pyt_results = record_perf(full_model, "Torch", [input_image, point_coords, point_labels, mask_input], args.precision, 10, 1)
    trt_results = record_perf(trt_model, "TensorRT", [input_image, point_coords, point_labels, mask_input], args.precision, 10, 1)

    print("==================================")
    print("PyTorch Results:")
    print(pd.DataFrame(pyt_results))

    print("==================================")
    print("TensorRT Results:")
    print(pd.DataFrame(trt_results))
