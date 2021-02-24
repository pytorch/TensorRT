import unittest
import trtorch
import torch
import torchvision.models as models


class MultiGpuTestCase(unittest.TestCase):

    def __init__(self, methodName='runTest', model=None):
        super(MultiGpuTestCase, self).__init__(methodName)
        self.model = model
        self.model.eval().to("cuda")

    @staticmethod
    def parametrize(testcase_class, model=None):
        testloader = unittest.TestLoader()
        testnames = testloader.getTestCaseNames(testcase_class)
        suite = unittest.TestSuite()
        for name in testnames:
            suite.addTest(testcase_class(name, model=model))
        return suite
