import torch


def complex_to_ri_stacked_tensor(t: torch.Tensor) -> torch.Tensor:
    # Converts complex tensor to real/imag stack
    if torch.is_complex(t):
        return torch.stack([t.real, t.imag], dim=-1)
    return t
