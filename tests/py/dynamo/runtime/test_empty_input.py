import pytest
import torch
import torch.nn as nn
import torch_tensorrt as torchtrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests

DECIMALS_OF_AGREEMENT = 5  # for output comparison


# We provide non null address to TRT
class ConcatEmptyModel(nn.Module):
    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim

    def forward(self, x, y):
        return torch.cat([x, y], dim=self.dim)


# TRT will handle
class ConcatEmptyModelEmptyConstant(nn.Module):
    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        y = torch.empty((0, 4), dtype=torch.float).cuda()
        return torch.cat([x, y], dim=self.dim)


# makes use of validator
class ConcatEmptyModelEmptyConstantMisMatchDim(nn.Module):
    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        y = torch.tensor([], device="cuda")
        return torch.cat([x, y], dim=self.dim)


class TestConcatEmptyTensor(TestCase):

    @parameterized.expand(
        [
            (
                "python_runtime_model_one_empty_0",
                True,
                ConcatEmptyModel,
                "two_inputs",
                (0,),
            ),
            (
                "cpp_runtime_model_one_empty_0",
                False,
                ConcatEmptyModel,
                "two_inputs",
                (0,),
            ),
            (
                "python_runtime_model_one_empty_0_4",
                True,
                ConcatEmptyModel,
                "two_inputs",
                (0, 4),
            ),
            (
                "cpp_runtime_model_one_empty_0_4",
                False,
                ConcatEmptyModel,
                "two_inputs",
                (0, 4),
            ),
            (
                "python_runtime_model_two_empty_0_4",
                True,
                ConcatEmptyModelEmptyConstant,
                "one_input",
                (0, 4),
            ),
            (
                "cpp_runtime_model_two_empty_0_4",
                False,
                ConcatEmptyModelEmptyConstant,
                "one_input",
                (0, 4),
            ),
            (
                "python_runtime_model_three_empty_0",
                True,
                ConcatEmptyModelEmptyConstantMisMatchDim,
                "one_input",
                (0,),
            ),
            (
                "cpp_runtime_model_three_empty_0",
                False,
                ConcatEmptyModelEmptyConstantMisMatchDim,
                "one_input",
                (0,),
            ),
        ]
    )
    def test_concat_empty_with_nonempty(
        self, _, use_python_runtime, model_class, input_type, empty_shape
    ):
        """
        Test concatenation of empty tensor with non-empty tensor
        along a specific dimension using Torch-TensorRT compiled model.
        """
        # Create model
        model = model_class(dim=0).eval().cuda()

        # Inputs: prepare based on model requirements
        empty_input = torch.empty(empty_shape, dtype=torch.float).cuda()
        non_empty_input = torch.randn((3, 4), dtype=torch.float).cuda()

        if input_type == "two_inputs":
            inputs = [empty_input, non_empty_input]
        else:  # one_input
            inputs = [non_empty_input]

        # Compile with Torch-TensorRT
        compiled_model = torchtrt.compile(
            model,
            "dynamo",
            inputs,
            min_block_size=5,
            use_python_runtime=use_python_runtime,
        )

        # Run reference model
        ref_out = model(*inputs)
        # Run compiled model
        trt_out = compiled_model(*inputs)

        # Assertions
        self.assertEqual(ref_out.shape, trt_out.shape)
        self.assertAlmostEqual(
            float(torch.max(torch.abs(ref_out - trt_out))),
            0,
            DECIMALS_OF_AGREEMENT,
            msg="Concat with empty tensor output mismatch",
        )

    @parameterized.expand(
        [
            ("python_runtime_empty_0", True, (0,)),
            ("cpp_runtime_empty_0", False, (0,)),
            ("python_runtime_empty_0_4", True, (0, 4)),
            ("cpp_runtime_empty_0_4", False, (0, 4)),
        ]
    )
    def test_concat_nonempty_with_empty(self, _, use_python_runtime, empty_shape):
        """
        Concatenate non-empty tensor with empty tensor (opposite order)
        """
        model = ConcatEmptyModel(dim=0).eval().cuda()

        non_empty_input = torch.randn((3, 4), dtype=torch.float).cuda()
        empty_input = torch.empty(empty_shape, dtype=torch.float).cuda()
        inputs = [non_empty_input, empty_input]

        compiled_model = torchtrt.compile(
            model,
            "dynamo",
            inputs,
            min_block_size=5,
            use_python_runtime=use_python_runtime,
        )

        ref_out = model(*inputs)
        trt_out = compiled_model(*inputs)

        self.assertEqual(ref_out.shape, trt_out.shape)
        self.assertAlmostEqual(
            float(torch.max(torch.abs(ref_out - trt_out))),
            0,
            DECIMALS_OF_AGREEMENT,
            msg="Concat with empty tensor (opposite order) output mismatch",
        )


if __name__ == "__main__":
    run_tests()
