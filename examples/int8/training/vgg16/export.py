import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
from vgg16 import vgg16


def test(model, dataloader, crit):
    """
    Run the model on a dataset and measure accuracy/loss
    """
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

    return loss / total, correct / total


def evaluate(model):
    """
    Evaluate pre-trained model on CIFAR 10 dataset
    """
    testing_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
    )

    testing_dataloader = torch.utils.data.DataLoader(
        testing_dataset, batch_size=32, shuffle=False, num_workers=2
    )

    crit = torch.nn.CrossEntropyLoss()

    test_loss, test_acc = test(model, testing_dataloader, crit)
    print("Test Loss: {:.5f} Test Acc: {:.2f}%".format(test_loss, 100 * test_acc))


def export_model(args):
    """
    Evaluate and export the model to Torchscript or exported program
    """
    # Define the VGG model
    # model = vgg16(num_classes=10, init_weights=False)
    model = models.vgg16(weights=None).eval().cuda()
    # Load the checkpoint
    ckpt = torch.load(args.ckpt)
    weights = ckpt["model_state_dict"]
    model.load_state_dict(weights)
    # Setting eval here causes both JIT and TRT accuracy to tank in LibTorch will follow up with PyTorch Team
    # model.eval()
    random_inputs = [torch.rand([32, 3, 32, 32]).to("cuda")]
    if args.ir == "torchscript":
        jit_model = torch.jit.trace(model, random_inputs)
        jit_model.eval()
        # Evaluating JIT model
        evaluate(jit_model)
        torch.jit.save(jit_model, args.output)
    elif args.ir == "exported_program":
        dim_x = torch.export.Dim("dim_x", min=1, max=32)
        exp_program = torch.export.export(
            model, tuple(random_inputs), dynamic_shapes={"x": {0: dim_x}}
        )
        evaluate(exp_program)
        torch.export.save(exp_program, args.output)
    else:
        raise ValueError(
            f"Invalid IR {args.ir} provided to export the VGG model. Select among torchscript | exported_program"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export trained VGG")
    parser.add_argument("--ckpt", type=str, help="Path to saved checkpoint")
    parser.add_argument(
        "--ir",
        type=str,
        default="torchscript",
        help="IR to determine the output type of exported graph",
    )
    parser.add_argument(
        "--output", type=str, default="vgg.ts", help="Path to saved checkpoint"
    )
    parser.add_argument(
        "--qat",
        action="store_true",
        help="Perform QAT using pytorch-quantization toolkit",
    )
    args = parser.parse_args()
    export_model(args)
