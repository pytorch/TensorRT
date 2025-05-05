"""
.. _vgg16_ptq:

Deploy Quantized Models using Torch-TensorRT
======================================================

Here we demonstrate how to deploy a model quantized to INT8 or FP8 using the Dynamo frontend of Torch-TensorRT
"""

# %%
# Imports and Model Definition
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import argparse

import modelopt.torch.quantization as mtq
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_tensorrt as torchtrt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from modelopt.torch.quantization.utils import export_torch_mode


class VGG(nn.Module):
    def __init__(self, layer_spec, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()

        layers = []
        in_channels = 3
        for l in layer_spec:
            if l == "pool":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers += [
                    nn.Conv2d(in_channels, l, kernel_size=3, padding=1),
                    nn.BatchNorm2d(l),
                    nn.ReLU(),
                ]
                in_channels = l

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def vgg16(num_classes=1000, init_weights=False):
    vgg16_cfg = [
        64,
        64,
        "pool",
        128,
        128,
        "pool",
        256,
        256,
        256,
        "pool",
        512,
        512,
        512,
        "pool",
        512,
        512,
        512,
        "pool",
    ]
    return VGG(vgg16_cfg, num_classes, init_weights)


PARSER = argparse.ArgumentParser(
    description="Load pre-trained VGG model and then tune with FP8 and PTQ. For having a pre-trained VGG model, please refer to https://github.com/pytorch/TensorRT/tree/main/examples/int8/training/vgg16"
)
PARSER.add_argument(
    "--ckpt", type=str, required=True, help="Path to the pre-trained checkpoint"
)
PARSER.add_argument(
    "--batch-size",
    default=128,
    type=int,
    help="Batch size for tuning the model with PTQ and FP8",
)
PARSER.add_argument(
    "--quantize-type",
    default="int8",
    type=str,
    help="quantization type, currently supported int8 or fp8 for PTQ",
)
args = PARSER.parse_args()

model = vgg16(num_classes=10, init_weights=False)
model = model.cuda()

# %%
# Load the pre-trained model weights
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ckpt = torch.load(args.ckpt)
weights = ckpt["model_state_dict"]

if torch.cuda.device_count() > 1:
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in weights.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    weights = new_state_dict

model.load_state_dict(weights)
# Don't forget to set the model to evaluation mode!
model.eval()

# %%
# Load training dataset and define loss function for PTQ
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

training_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    ),
)
training_dataloader = torch.utils.data.DataLoader(
    training_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=2,
    drop_last=True,
)

data = iter(training_dataloader)
images, _ = next(data)

crit = nn.CrossEntropyLoss()

# %%
# Define Calibration Loop for quantization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


def calibrate_loop(model):
    # calibrate over the training dataset
    total = 0
    correct = 0
    loss = 0.0
    for data, labels in training_dataloader:
        data, labels = data.cuda(), labels.cuda(non_blocking=True)
        out = model(data)
        loss += crit(out, labels)
        preds = torch.max(out, 1)[1]
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    print("PTQ Loss: {:.5f} Acc: {:.2f}%".format(loss / total, 100 * correct / total))


# %%
# Tune the pre-trained model with FP8 and PTQ
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
if args.quantize_type == "int8":
    quant_cfg = mtq.INT8_DEFAULT_CFG
elif args.quantize_type == "fp8":
    quant_cfg = mtq.FP8_DEFAULT_CFG
# PTQ with in-place replacement to quantized modules
mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
# model has FP8 qdq nodes at this point

# %%
# Inference
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Load the testing dataset
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
    testing_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=2,
    drop_last=True,
)  # set drop_last=True to drop the last incomplete batch for static shape `torchtrt.dynamo.compile()`

with torch.no_grad():
    with export_torch_mode():
        # Compile the model with Torch-TensorRT Dynamo backend
        input_tensor = images.cuda()

        exp_program = torch.export.export(model, (input_tensor,), strict=False)
        if args.quantize_type == "int8":
            enabled_precisions = {torch.int8}
        elif args.quantize_type == "fp8":
            enabled_precisions = {torch.float8_e4m3fn}
        trt_model = torchtrt.dynamo.compile(
            exp_program,
            inputs=[input_tensor],
            enabled_precisions=enabled_precisions,
            min_block_size=1,
            debug=False,
        )
        # You can also use torch compile path to compile the model with Torch-TensorRT:
        # trt_model = torch.compile(model, backend="tensorrt")

        # Inference compiled Torch-TensorRT model over the testing dataset
        total = 0
        correct = 0
        loss = 0.0
        class_probs = []
        class_preds = []
        for data, labels in testing_dataloader:
            data, labels = data.cuda(), labels.cuda(non_blocking=True)
            out = trt_model(data)
            loss += crit(out, labels)
            preds = torch.max(out, 1)[1]
            class_probs.append([F.softmax(i, dim=0) for i in out])
            class_preds.append(preds)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
        test_preds = torch.cat(class_preds)
        test_loss = loss / total
        test_acc = correct / total
        print("Test Loss: {:.5f} Test Acc: {:.2f}%".format(test_loss, 100 * test_acc))
