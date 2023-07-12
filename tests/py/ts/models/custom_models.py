import torch
from transformers import BertModel, BertTokenizer, BertConfig


def BertModule():
    model_name = "bert-base-uncased"
    enc = BertTokenizer.from_pretrained(model_name)
    text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
    tokenized_text = enc.tokenize(text)
    masked_index = 8
    tokenized_text[masked_index] = "[MASK]"
    indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    config = BertConfig(
        vocab_size_or_config_json_file=32000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        torchscript=True,
    )
    model = BertModel(config)
    model.eval()
    model = BertModel.from_pretrained(model_name, torchscript=True)
    traced_model = torch.jit.trace(model, [tokens_tensor, segments_tensors])
    return traced_model
