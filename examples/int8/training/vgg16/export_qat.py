import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from vgg16 import vgg16

from pytorch_quantization import quant_modules
from pytorch_quantization import nn as quant_nn


def test(model, dataloader, crit):
    global writer
    global classes
    total = 0
    correct = 0
    loss = 0.0
    class_probs = []
    class_preds = []

    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.cuda(), labels.cuda(non_blocking=True)
            out = model(data)
            loss += crit(out, labels)
            preds = torch.max(out, 1)[1]
            class_probs.append([F.softmax(i, dim=0) for i in out])
            class_preds.append(preds)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    test_preds = torch.cat(class_preds)
    return loss / total, correct / total


PARSER = argparse.ArgumentParser(description="Export trained VGG")
PARSER.add_argument("ckpt", type=str, help="Path to saved checkpoint")

args = PARSER.parse_args()

quant_modules.initialize()
model = vgg16(num_classes=10, init_weights=False)
model = model.cuda()

ckpt = torch.load(args.ckpt)
weights = {}
for key, val in ckpt["model_state_dict"].items():
    # Remove 'module.' from the key names
    if key.startswith("module"):
        weights[key[7:]] = val
    else:
        weights[key] = val

model.load_state_dict(weights)
model.eval()

testing_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    ),
)

testing_dataloader = torch.utils.data.DataLoader(
    testing_dataset, batch_size=32, shuffle=False, num_workers=2
)

crit = torch.nn.CrossEntropyLoss()

quant_nn.TensorQuantizer.use_fb_fake_quant = True
with torch.no_grad():
    data = iter(testing_dataloader)
    images, _ = data.next()
    jit_model = torch.jit.trace(model, images.to("cuda"))
    torch.jit.save(jit_model, "trained_vgg16_qat.jit.pt")

test_loss, test_acc = test(jit_model, testing_dataloader, crit)
print("[JIT] Test Loss: {:.5f} Test Acc: {:.2f}%".format(test_loss, 100 * test_acc))
