import os
import unittest

import torch
import torch.nn as nn
import torch_tensorrt as torchtrt
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn import functional as F
from torch_tensorrt.dynamo import CalibrationAlgo, DataLoaderCalibrator
from torch_tensorrt.logging import *
from vgg16 import vgg16


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
        # self.model = models.vgg16(weights=None).eval()
        self.model = vgg16(num_classes=10, init_weights=False).eval().cuda()
        # ckpt = torch.load("./ckpt_epoch25.pth")
        # weights = ckpt["model_state_dict"]
        # self.model.load_state_dict(weights)
        self.model = self.model.cuda()
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
            self.testing_dataset, batch_size=100, shuffle=False, num_workers=1
        )
        self.calibrator = DataLoaderCalibrator(
            self.testing_dataloader,
            cache_file="./calibration.cache",
            use_cache=False,
            algo_type=CalibrationAlgo.ENTROPY_CALIBRATION_2,
            device=torch.device("cuda:0"),
        )

        compile_spec = {
            "inputs": [torchtrt.Input([100, 3, 32, 32])],
            "enabled_precisions": {torch.int8},
            "calibrator": self.calibrator,
            "truncate_long_and_double": True,
            "debug": True,
            "require_full_compilation": True,
            "enable_experimental_decompositions": True,
            "min_block_size": 1,
        }
        trt_mod = torch.compile(
            self.model, backend="torch_tensorrt", dynamic=False, options=compile_spec
        )

        fp32_test_acc = compute_accuracy(self.testing_dataloader, self.model)
        log(Level.Info, "[Pyt FP32] Test Acc: {:.2f}%".format(100 * fp32_test_acc))

        int8_test_acc = compute_accuracy(self.testing_dataloader, trt_mod)
        log(Level.Info, "[TRT INT8] Test Acc: {:.2f}%".format(100 * int8_test_acc))
        acc_diff = fp32_test_acc - int8_test_acc
        self.assertTrue(abs(acc_diff) < 3)


if __name__ == "__main__":
    unittest.main()
