import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestResizeConverter(DispatchTestCase):
    def compare_resized_tensors(self, tensor1, tensor2, input_shape, target_shape):
        # Check if the sizes match
        if tensor1.size() != tensor2.size():
            return False

        # Flatten the tensors to ensure we are comparing the valid elements
        flat_tensor1 = tensor1.flatten()
        flat_tensor2 = tensor2.flatten()

        # Calculate the number of valid elements to compare
        input_numel = torch.Size(input_shape).numel()
        target_numel = torch.Size(target_shape).numel()
        min_size = min(input_numel, target_numel)

        # Compare only the valid elements
        return torch.equal(flat_tensor1[:min_size], flat_tensor2[:min_size])

    @parameterized.expand(
        [
            ((3,),),
            ((5,),),
            ((10,),),
            ((2, 2),),
            ((3, 5),),
            ((8, 3),),
            ((7, 7),),
            ((5, 5, 5),),
            ((3, 3, 10),),
            ((10, 15, 10),),
        ]
    )
    def test_resize_1d_input_float(self, target_shape):
        class Resize(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.resize_.default(x, target_shape)

        input_shape = (5,)
        inputs = [torch.randn(input_shape)]

        comparators = [(self.compare_resized_tensors, [input_shape, target_shape])]

        self.run_test_compare_tensor_attributes_only(
            Resize(),
            inputs,
            expected_ops=[],
            comparators=comparators,
        )

    @parameterized.expand(
        [
            ((3,),),
            ((5,),),
            ((10,),),
            ((3, 5),),
            ((8, 3),),
            ((7, 7),),
            ((5, 5, 5),),
            ((3, 3, 5),),
            ((15, 10, 3),),
            ((15, 10, 12),),
        ]
    )
    def test_resize_1d_input_int(self, target_shape):
        class Resize(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.resize_.default(x, target_shape)

        input_shape = (5,)
        inputs = [torch.randint(1, 5, input_shape)]

        comparators = [(self.compare_resized_tensors, [input_shape, target_shape])]

        self.run_test_compare_tensor_attributes_only(
            Resize(),
            inputs,
            expected_ops=[],
            comparators=comparators,
        )

    @parameterized.expand(
        [
            ((3,),),
            ((5,),),
            ((10,),),
            ((4, 4),),
            ((3, 5),),
            ((8, 3),),
            ((7, 7),),
            ((20, 12, 13),),
            ((3, 3, 5),),
            ((3, 10, 15),),
        ]
    )
    def test_resize_2d_input_float(self, target_shape):
        class Resize(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.resize_.default(x, target_shape)

        input_shape = (4, 4)
        inputs = [torch.randint(1, 10, input_shape)]

        comparators = [(self.compare_resized_tensors, [input_shape, target_shape])]

        self.run_test_compare_tensor_attributes_only(
            Resize(),
            inputs,
            expected_ops=[],
            comparators=comparators,
        )

    @parameterized.expand(
        [
            ((3,),),
            ((5,),),
            ((20,),),
            ((4, 4),),
            ((3, 12),),
            ((12, 3),),
            ((15, 15),),
            ((20, 20, 20),),
            ((3, 3, 10),),
        ]
    )
    def test_resize_2d_input_int(self, target_shape):
        class Resize(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.resize_.default(x, target_shape)

        input_shape = (4, 4)
        inputs = [torch.randint(1, 10, input_shape)]

        comparators = [(self.compare_resized_tensors, [input_shape, target_shape])]

        self.run_test_compare_tensor_attributes_only(
            Resize(),
            inputs,
            expected_ops=[],
            comparators=comparators,
        )


if __name__ == "__main__":
    run_tests()
