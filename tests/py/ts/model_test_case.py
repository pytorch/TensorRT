# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import unittest

import torch

REPO_ROOT = os.path.abspath(os.getcwd()) + "/../../"


class ModelTestCase(unittest.TestCase):
    def __init__(self, methodName="runTest", model=None):
        super(ModelTestCase, self).__init__(methodName)
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
