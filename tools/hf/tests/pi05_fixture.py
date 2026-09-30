from types import SimpleNamespace

from lerobot.policies.pi05.modeling_pi05 import PaliGemmaWithExpertModel, PI05Pytorch
from lerobot.policies.pi_gemma import (
    PaliGemmaForConditionalGenerationWithPiGemma,
    PiGemmaForCausalLM,
)
from torch import nn
from transformers import GemmaConfig, PaliGemmaConfig, SiglipVisionConfig


def tiny_core(device="cpu"):
    common = dict(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        vocab_size=64,
        hidden_activation="gelu_pytorch_tanh",
        output_attentions=False,
        output_hidden_states=False,
    )
    language = GemmaConfig(hidden_size=32, intermediate_size=64, **common)
    language.use_adarms = False
    language.adarms_cond_dim = None
    language._attn_implementation = "eager"
    expert = GemmaConfig(hidden_size=16, intermediate_size=32, **common)
    expert.use_adarms = True
    expert.adarms_cond_dim = 16
    expert._attn_implementation = "eager"
    vision = SiglipVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=16,
        patch_size=8,
    )
    vision._attn_implementation = "eager"
    config = PaliGemmaConfig(
        text_config=language,
        vision_config=vision,
        projection_dim=32,
        image_token_index=63,
    )
    core = PI05Pytorch.__new__(PI05Pytorch)
    nn.Module.__init__(core)
    core.config = SimpleNamespace(
        chunk_size=3,
        max_action_dim=4,
        num_inference_steps=3,
        min_period=4e-3,
        max_period=4.0,
        rtc_config=None,
        use_proprioceptive_memory=False,
    )
    core.gradient_checkpointing_enabled = False
    core.rtc_processor = None
    pair = PaliGemmaWithExpertModel.__new__(PaliGemmaWithExpertModel)
    nn.Module.__init__(pair)
    pair.precision = "float32"
    pair.freeze_vision_encoder = False
    pair.train_expert_only = False
    pair.paligemma = PaliGemmaForConditionalGenerationWithPiGemma(config)
    pair.gemma_expert = PiGemmaForCausalLM(expert)
    pair.gemma_expert.model.embed_tokens = None
    core.paligemma_with_expert = pair
    core.action_in_proj = nn.Linear(4, 16)
    core.action_out_proj = nn.Linear(16, 4)
    core.time_mlp_in = nn.Linear(16, 16)
    core.time_mlp_out = nn.Linear(16, 16)
    core.proprio_history_proj = None
    return core.to(device).eval()
