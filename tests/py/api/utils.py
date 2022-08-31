import torch

COSINE_THRESHOLD=0.99

def cosine_similarity(gt_tensor, pred_tensor):
    res = torch.nn.functional.cosine_similarity(gt_tensor.flatten().to(torch.float32), pred_tensor.flatten().to(torch.float32), dim=0, eps=1e-6)
    res = res.cpu().detach().item()

    return res
