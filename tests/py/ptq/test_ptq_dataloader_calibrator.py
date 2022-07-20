import unittest
import torch_tensorrt as torchtrt
from torch_tensorrt.logging import *
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
from model_test_case import ModelTestCase


class TestAccuracy(ModelTestCase):

    def setUp(self):
        self.input = torch.randn((1, 3, 32, 32)).to("cuda")
        self.testing_dataset = torchvision.datasets.CIFAR10(root='./data',
                                                            train=False,
                                                            download=True,
                                                            transform=transforms.Compose([
                                                                transforms.ToTensor(),
                                                                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                     (0.2023, 0.1994, 0.2010))
                                                            ]))

        self.testing_dataloader = torch.utils.data.DataLoader(self.testing_dataset,
                                                              batch_size=1,
                                                              shuffle=False,
                                                              num_workers=1)
        self.calibrator = torchtrt.ptq.DataLoaderCalibrator(
            self.testing_dataloader,
            cache_file='./calibration.cache',
            use_cache=False,
            algo_type=torchtrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
            device=torch.device('cuda:0'))

    def compute_accuracy(self, testing_dataloader, model):
        total = 0
        correct = 0
        loss = 0.0
        class_probs = []
        class_preds = []
        device = torch.device('cuda:0')
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

    def test_compile_script(self):

        fp32_test_acc = self.compute_accuracy(self.testing_dataloader, self.model)
        log(Level.Info, "[Pyt FP32] Test Acc: {:.2f}%".format(100 * fp32_test_acc))

        compile_spec = {
            "inputs": [torchtrt.Input([1, 3, 32, 32])],
            "enabled_precisions": {torch.float, torch.int8},
            "calibrator": self.calibrator,
            "truncate_long_and_double": True,
            "device": {
                "device_type": torchtrt.DeviceType.GPU,
                "gpu_id": 0,
                "dla_core": 0,
                "allow_gpu_fallback": False,
            }
        }

        trt_mod = torchtrt.ts.compile(self.model, **compile_spec)
        int8_test_acc = self.compute_accuracy(self.testing_dataloader, trt_mod)
        log(Level.Info, "[TRT INT8] Test Acc: {:.2f}%".format(100 * int8_test_acc))
        acc_diff = fp32_test_acc - int8_test_acc
        self.assertTrue(abs(acc_diff) < 3)


def test_suite():
    suite = unittest.TestSuite()
    # You need a pre-trained VGG cifar10 model to run this test. Please follow instructions at
    # https://github.com/NVIDIA/torchtrt/tree/master/cpp/ptq/training/vgg16 to export this model.
    suite.addTest(TestAccuracy.parametrize(TestAccuracy, model=torch.jit.load('./trained_vgg16.jit.pt')))

    return suite


suite = test_suite()

runner = unittest.TextTestRunner()
result = runner.run(suite)

exit(int(not result.wasSuccessful()))
