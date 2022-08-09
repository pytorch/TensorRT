import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig
import torch.nn.functional as F
from typing import Tuple, List, Dict


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


# Sample Inplace OP in Conditional Block Model
class FallbackInplaceOPIf(nn.Module):
    def __init__(self):
        super(FallbackInplaceOPIf, self).__init__()

    def forward(self, x, y):
        mod_list = [x]
        if x.sum() > y.sum():
            mod_list.append(y)
        z = torch.cat(mod_list)
        return z


# Collection input/output models
class StandardTensorInput(nn.Module):
    def __init__(self):
        super(StandardTensorInput, self).__init__()

    def forward(self, x, y):
        r = x + y
        return r


class TupleInput(nn.Module):
    def __init__(self):
        super(TupleInput, self).__init__()

    def forward(self, z: Tuple[torch.Tensor, torch.Tensor]):
        r = z[0] + z[1]
        return r


class ListInput(nn.Module):
    def __init__(self):
        super(ListInput, self).__init__()

    def forward(self, z: List[torch.Tensor]):
        r = z[0] + z[1]
        return r


class TupleInputOutput(nn.Module):
    def __init__(self):
        super(TupleInputOutput, self).__init__()

    def forward(self, z: Tuple[torch.Tensor, torch.Tensor]):
        r1 = z[0] + z[1]
        r2 = z[0] - z[1]
        r1 = r1 * 10
        r = (r1, r2)
        return r


class ListInputOutput(nn.Module):
    def __init__(self):
        super(ListInputOutput, self).__init__()

    def forward(self, z: List[torch.Tensor]):
        r1 = z[0] + z[1]
        r2 = z[0] - z[1]
        r = [r1, r2]
        return r


class ListInputTupleOutput(nn.Module):
    def __init__(self):
        super(ListInputTupleOutput, self).__init__()
        self.list_model = ListInputOutput()
        self.tuple_model = TupleInputOutput()

    def forward(self, z: List[torch.Tensor]):
        r1 = z[0] + z[1]
        r2 = z[0] - z[1]
        r3 = (r1, r2)
        r4 = [r2, r1]
        tuple_out = self.tuple_model(r3)
        list_out = self.list_model(r4)
        r = (tuple_out[1], list_out[0])
        return r


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
