import torch
from torch.utils.data import DataLoader, TensorDataset
from torch_tensorrt.verification import InputSample


class Add(torch.nn.Module):  # type: ignore[misc]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + x


def model() -> Add:
    add_model = Add()
    add_model.cuda().eval()
    return add_model


def inputs(seed: int, batch_size: int = 1) -> list[InputSample]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    tensors = torch.randn(
        batch_size * 4,
        3,
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    loader = DataLoader(
        TensorDataset(tensors),
        batch_size=batch_size,
        shuffle=False,
    )
    return [
        InputSample(
            args=batch,
            profile="static",
            sample_id=f"batch-{index}",
        )
        for index, batch in enumerate(loader)
    ]
