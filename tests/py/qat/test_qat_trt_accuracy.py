import unittest
import torch_tensorrt as torchtrt
from torch_tensorrt.logging import *
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
import os
import sys


def find_repo_root(max_depth=10):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    for i in range(max_depth):
        files = os.listdir(dir_path)
        if "WORKSPACE" in files:
            return dir_path
        else:
            dir_path = os.path.dirname(dir_path)

    raise RuntimeError("Could not find repo root")


MODULE_DIR = find_repo_root() + "/tests/modules"

set_reportable_log_level(Level.Graph)


def compute_accuracy(testing_dataloader, model):
    total = 0
    correct = 0
    loss = 0.0
    class_probs = []
    class_preds = []
    device = torch.device("cuda:0")
    with torch.no_grad():
        idx = 0
        for data, labels in testing_dataloader:
            data, labels = data.to(device), labels.to(device)
            out = model(data)
            preds = torch.max(out, 1)[1]
            class_probs.append([F.softmax(i, dim=0) for i in out])
            class_preds.append(preds)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            idx += 1

    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    test_preds = torch.cat(class_preds)
    return correct / total


class TestAccuracy(unittest.TestCase):
    def test_compile_script(self):
        self.model = (
            torch.jit.load(MODULE_DIR + "/trained_vgg16_qat.jit.pt").eval().to("cuda")
        )
        self.testing_dataset = torchvision.datasets.CIFAR10(
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

        self.testing_dataloader = torch.utils.data.DataLoader(
            self.testing_dataset, batch_size=16, shuffle=False, num_workers=1
        )

        fp32_test_acc = compute_accuracy(self.testing_dataloader, self.model)
        log(Level.Info, "[Pyt FP32] Test Acc: {:.2f}%".format(100 * fp32_test_acc))

        compile_spec = {
            "inputs": [torchtrt.Input([16, 3, 32, 32])],
            "enabled_precisions": {torch.int8},
            # "enabled_precision": {torch.float32, torch.int8},
        }

        trt_mod = torchtrt.ts.compile(self.model, **compile_spec)
        int8_test_acc = compute_accuracy(self.testing_dataloader, trt_mod)
        log(Level.Info, "[TRT QAT INT8] Test Acc: {:.2f}%".format(100 * int8_test_acc))
        acc_diff = fp32_test_acc - int8_test_acc
        self.assertTrue(abs(acc_diff) < 3)


if __name__ == "__main__":
    unittest.main()
