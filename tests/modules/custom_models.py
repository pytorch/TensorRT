import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig
import torch.nn.functional as F

# Sample Pool Model (for testing plugin serialization)
class Pool(nn.Module):

    def __init__(self):
        super(Pool, self).__init__()

    def forward(self, x):
        return F.adaptive_avg_pool2d(x, (5, 5))


# Sample Nested Module (for module-level fallback testing)
class ModuleFallbackSub(nn.Module):

    def __init__(self):
        super(ModuleFallbackSub, self).__init__()
        self.conv = nn.Conv2d(1, 3, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class ModuleFallbackMain(nn.Module):

    def __init__(self):
        super(ModuleFallbackMain, self).__init__()
        self.layer1 = ModuleFallbackSub()
        self.conv = nn.Conv2d(3, 6, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(self.layer1(x)))


# Sample Looping Modules (for loop fallback testing)
class LoopFallbackEval(nn.Module):

    def __init__(self):
        super(LoopFallbackEval, self).__init__()

    def forward(self, x):
        add_list = torch.empty(0).to(x.device)
        for i in range(x.shape[1]):
            add_list = torch.cat((add_list, torch.tensor([x.shape[1]]).to(x.device)), 0)
        return x + add_list


class LoopFallbackNoEval(nn.Module):

    def __init__(self):
        super(LoopFallbackNoEval, self).__init__()

    def forward(self, x):
        for _ in range(x.shape[1]):
            x = x + torch.ones_like(x)
        return x


# Sample Conditional Model (for testing partitioning and fallback in conditionals)
class FallbackIf(torch.nn.Module):

    def __init__(self):
        super(FallbackIf, self).__init__()
        self.relu1 = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1)
        self.log_sig = torch.nn.LogSigmoid()
        self.conv2 = torch.nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(32, 3, 3, 1, 1)

    def forward(self, x):
        x = self.relu1(x)
        x_first = x[0][0][0][0].item()
        if x_first > 0:
            x = self.conv1(x)
            x1 = self.log_sig(x)
            x2 = self.conv2(x)
            x = self.conv3(x1 + x2)
        else:
            x = self.log_sig(x)
        x = self.conv1(x)
        return x


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

