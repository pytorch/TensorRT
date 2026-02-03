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
            min_block_size=1,
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
            min_block_size=1,
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


class TestEmptyTensorMemoryLeak(TestCase):
    """
    Tests to verify that repeated inferences with empty tensors
    do not cause memory leaks and produce correct results.
    """

    @parameterized.expand(
        [
            ("cpp_runtime", False),
            ("python_runtime", True),
        ]
    )
    def test_repeated_empty_tensor_no_leak_and_correct(self, _, use_python_runtime):
        """
        Run many inferences with empty tensor input to verify:
        1. Memory doesn't grow (placeholder is reused, not reallocated)
        2. Outputs are correct (placeholder doesn't corrupt results)
        """
        model = ConcatEmptyModel(dim=0).eval().cuda()

        empty_input = torch.empty((0, 4), dtype=torch.float).cuda()
        non_empty_input = torch.randn((3, 4), dtype=torch.float).cuda()
        inputs = [empty_input, non_empty_input]

        compiled_model = torchtrt.compile(
            model,
            "dynamo",
            inputs,
            min_block_size=1,
            use_python_runtime=use_python_runtime,
        )

        # Record initial GPU memory
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()

        # Run many inferences with empty tensor
        num_iterations = 1000
        for i in range(num_iterations):
            # Use different non_empty data each iteration to test correctness
            non_empty_input = torch.randn((3, 4), dtype=torch.float).cuda()
            inputs = [empty_input, non_empty_input]

            ref_out = model(*inputs)
            trt_out = compiled_model(*inputs)

            # Verify correctness every 100 iterations (to keep test fast)
            if i % 100 == 0:
                self.assertEqual(ref_out.shape, trt_out.shape)
                self.assertAlmostEqual(
                    float(torch.max(torch.abs(ref_out - trt_out))),
                    0,
                    DECIMALS_OF_AGREEMENT,
                    msg=f"Output mismatch at iteration {i}",
                )

        torch.cuda.synchronize()
        final_memory = torch.cuda.memory_allocated()

        # Memory growth should be minimal (not proportional to num_iterations)
        memory_growth = final_memory - initial_memory
        max_allowed_growth = 1024 * 1024  # 1 MB max threshold

        print(f"Memory growth: {memory_growth} bytes")

        self.assertLess(
            memory_growth,
            max_allowed_growth,
            msg=f"Memory grew by {memory_growth} bytes after {num_iterations} iterations. "
            f"Possible memory leak with empty tensor handling.",
        )


if __name__ == "__main__":
    run_tests()
