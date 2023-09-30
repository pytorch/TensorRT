import numpy as np
import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo.conversion.converter_utils import enforce_tensor_types
from torch_tensorrt.fx.types import TRTTensor

from ..testing_utilities import DECIMALS_OF_AGREEMENT, lower_graph_testing


class TestTensorTypeEnforcement(TestCase):
    def test_valid_type_no_promotion(self):
        @enforce_tensor_types({0: (np.ndarray, torch.Tensor)}, promote=False)
        def fake_converter(network, target, args, kwargs, name):
            self.assertIsInstance(args[0], np.ndarray)
            return

        fake_converter(None, None, (np.ones((4, 4)),), {}, "fake")

    def test_different_type_no_promotion(self):
        @enforce_tensor_types({0: (TRTTensor,)}, promote=False)
        def fake_converter(network, target, args, kwargs, name):
            return

        with self.assertRaises(AssertionError):
            fake_converter(None, None, (np.ones((4, 4)),), {}, "fake")

    def test_different_type_with_promotion(self):
        @enforce_tensor_types({"sample": (np.ndarray,)}, promote=True)
        def fake_converter(network, target, args, kwargs, name):
            self.assertIsInstance(kwargs["sample"], np.ndarray)
            return

        fake_converter(None, None, tuple(), {"sample": torch.ones((4, 4))}, "fake")

    def test_invalid_invocation_type(self):
        with self.assertRaises(AssertionError):
            enforce_tensor_types({0: (int, bool)})


if __name__ == "__main__":
    run_tests()
