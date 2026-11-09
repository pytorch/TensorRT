import torch
from torch_tensorrt.verification import InputSample
from transformers import Qwen3_5ForConditionalGeneration

MODEL_ID = "Qwen/Qwen3.5-0.8B"
VOCAB_SIZE = 248_320


class Qwen35Prefill(torch.nn.Module):  # type: ignore[misc]
    """Text-only prefill graph that returns logits for the final input token."""

    def __init__(self) -> None:
        super().__init__()
        full_model = Qwen3_5ForConditionalGeneration.from_pretrained(
            MODEL_ID,
            dtype=torch.float16,
            attn_implementation="sdpa",
        )
        self.model = full_model.model.language_model
        self.lm_head = full_model.lm_head
        del full_model
        self.cuda().eval()

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=False,
            return_dict=False,
        )[0]
        return self.lm_head(hidden_states[:, -1:, :])


def qwen35_model() -> Qwen35Prefill:
    return Qwen35Prefill()


def qwen35_inputs(
    seed: int, batch_size: int = 1, sequence_length: int = 16
) -> InputSample:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    input_ids = torch.randint(
        0,
        VOCAB_SIZE,
        (batch_size, sequence_length),
        device="cuda",
        dtype=torch.int64,
        generator=generator,
    )
    position_ids = (
        torch.arange(sequence_length, device="cuda", dtype=torch.int64)
        .unsqueeze(0)
        .repeat(batch_size, 1)
    )
    return InputSample(
        args=(input_ids, position_ids),
        profile="static",
        sample_id=f"batch-{batch_size}-sequence-{sequence_length}",
        label="Qwen3.5 text prefill",
    )
