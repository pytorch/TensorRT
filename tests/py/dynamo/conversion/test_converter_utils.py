import numpy as np
import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo.conversion.converter_utils import (
    enforce_tensor_types,
    flatten_dims,
)
from torch_tensorrt.dynamo.types import TRTTensor

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


class TestFlattenDimsEnforcement(TestCase):
    @parameterized.expand(
        [
            ((1, 2), 0, 0, (1, 2)),
            ((1, 2), 0, 1, (2,)),
            ((2, 3, 4), 1, 2, (2, 12)),
            ((2, 3, 4), 0, 1, (6, 4)),
            ((2, 3, 4), -3, 2, (24,)),
            ((2, 3, 4, 5), 0, -2, (24, 5)),
            ((2, 3, 4, 5), -4, -1, (120,)),
        ]
    )
    def test_numpy_array(self, input_shape, start_dim, end_dim, true_shape):
        inputs = np.random.randn(*input_shape)
        new_shape = flatten_dims(inputs, start_dim, end_dim)
        self.assertEqual(new_shape, true_shape)

    @parameterized.expand(
        [
            ((1, 2), 0, 0, (1, 2)),
            ((1, 2), 0, 1, (2,)),
            ((2, 3, 4), 1, 2, (2, 12)),
            ((2, 3, 4), 0, 1, (6, 4)),
            ((2, 3, 4), -3, 2, (24,)),
            ((2, 3, 4, 5), 0, -2, (24, 5)),
            ((2, 3, 4, 5), -4, -1, (120,)),
        ]
    )
    def test_torch_tensor(self, input_shape, start_dim, end_dim, true_shape):
        inputs = torch.randn(input_shape)
        new_shape = flatten_dims(inputs, start_dim, end_dim)
        self.assertEqual(new_shape, true_shape)


if __name__ == "__main__":
    run_tests()
