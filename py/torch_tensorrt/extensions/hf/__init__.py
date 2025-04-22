from torch_tensorrt.extensions.hf import flashinfer_attention

from .attention_interface import *


def initialize_kvcache(max_seq_len, device):
    # Initialize kvcaching
    # construct a cache info object
    sequence_info = SequenceInfo(
        max_seq_len=max_seq_len,
        max_batch_size=1,
        page_size=max_seq_len,
    )
    return CachedSequenceInterface(
        AttentionRegistry.get("FlashInfer"), sequence_info=sequence_info, device=device
    )


kv_cache_manager = initialize_kvcache(2176, torch.device("cuda:0"))

from .insert_flashinfer_ops import *
