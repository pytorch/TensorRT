import torch

COSINE_THRESHOLD = 0.99


def cosine_similarity(gt_tensor, pred_tensor):
    gt_tensor = gt_tensor.flatten().to(torch.float32)
    pred_tensor = pred_tensor.flatten().to(torch.float32)
    if torch.sum(gt_tensor) == 0.0 or torch.sum(pred_tensor) == 0.0:
        if torch.allclose(gt_tensor, pred_tensor, atol=1e-4, rtol=1e-4, equal_nan=True):
            return 1.0
    res = torch.nn.functional.cosine_similarity(gt_tensor, pred_tensor, dim=0, eps=1e-6)
    res = res.cpu().detach().item()

    return res


def same_output_format(trt_output, torch_output):
    # For each encountered collection type, ensure the torch and trt outputs agree
    # on type and size, checking recursively through all member elements.
    if isinstance(trt_output, tuple):
        return (
            isinstance(torch_output, tuple)
            and (len(trt_output) == len(torch_output))
            and all(
                same_output_format(trt_entry, torch_entry)
                for trt_entry, torch_entry in zip(trt_output, torch_output)
            )
        )
    elif isinstance(trt_output, list):
        return (
            isinstance(torch_output, list)
            and (len(trt_output) == len(torch_output))
            and all(
                same_output_format(trt_entry, torch_entry)
                for trt_entry, torch_entry in zip(trt_output, torch_output)
            )
        )
    elif isinstance(trt_output, dict):
        return (
            isinstance(torch_output, dict)
            and (len(trt_output) == len(torch_output))
            and (trt_output.keys() == torch_output.keys())
            and all(
                same_output_format(trt_output[key], torch_output[key])
                for key in trt_output.keys()
            )
        )
    elif isinstance(trt_output, set) or isinstance(trt_output, frozenset):
        raise AssertionError(
            "Unsupported output type 'set' encountered in output format check."
        )
    else:
        return type(trt_output) is type(torch_output)
