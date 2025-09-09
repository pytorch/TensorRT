import unittest

import pytest
import torch
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo.utils import is_tegra_platform, is_thor

from ..testing_utilities import DECIMALS_OF_AGREEMENT

INPUT_SIZE = (3, 16, 16)
TRIALS = 5


class StaticModel(torch.nn.Module):
    def forward(self, input):
        return torch.ops.aten.abs.default(input)


class DDSModel(torch.nn.Module):
    def forward(self, input):
        return torch.ops.aten.nonzero.default(input)


class DDSOpWithReductionOpModel(torch.nn.Module):
    """
    DDSOpWithReductionOpModel is a model that contains DDS op + reduction op.
    Since nonzero requires output allocator, this model will use output allocator by default.
    """

    def forward(self, inputs):
        out = torch.ops.aten.nonzero.default(inputs)
        out = torch.ops.aten.sum.dim_IntList(out, 0)
        return out


class DDSModel2(torch.nn.Module):
    def forward(self, input):
        # combination of multiple non-zero and other ops
        out = torch.ops.aten.nonzero.default(input)
        out = torch.ops.aten.abs.default(out)
        out = torch.ops.aten.nonzero.default(out)
        return out


@unittest.skipIf(
    torch_tensorrt.ENABLED_FEATURES.tensorrt_rtx or is_thor() or is_tegra_platform(),
    "TensorRT RTX does not support nonzero which are required for this test",
)
class TestOutputAllocatorStaticModel(TestCase):
    @parameterized.expand(
        [
            ("python_runtime", True),
            ("cpp_runtime", False),
        ]
    )
    def test_cudagraphs_and_output_allocator(self, _, use_python_runtime):
        model = StaticModel().eval().cuda()
        inputs = [torch.randn((2, 3), dtype=torch.float).cuda()]
        compiled_model = torch_tensorrt.compile(
            model,
            "dynamo",
            inputs,
            min_block_size=1,
            use_python_runtime=use_python_runtime,
        )

        ref_out = model(*inputs)

        with torch_tensorrt.runtime.enable_cudagraphs(
            compiled_model
        ) as cudagraphs_module:
            cg_out = cudagraphs_module(*inputs)

        self.assertAlmostEqual(
            float(torch.max(torch.abs(ref_out - cg_out))),
            0,
            DECIMALS_OF_AGREEMENT,
            msg="CUDA Graphs runtime outputs don't match with the original model.",
        )

        with torch_tensorrt.runtime.enable_output_allocator(compiled_model):
            oa_out = compiled_model(*inputs)

        self.assertAlmostEqual(
            float(torch.max(torch.abs(ref_out - oa_out))),
            0,
            DECIMALS_OF_AGREEMENT,
            msg="Output Allocator runtime outputs don't match with the original model.",
        )

    @parameterized.expand(
        [
            ("python_runtime", True),
            ("cpp_runtime", False),
        ]
    )
    def test_default(self, _, use_python_runtime):
        """
        Static models use standard execution with cudagraphs=False by default.
        """
        model = StaticModel().eval().cuda()
        inputs = [torch.randn((2, 3), dtype=torch.float).cuda()]
        compiled_model = torch_tensorrt.compile(
            model,
            "dynamo",
            inputs,
            min_block_size=1,
            use_python_runtime=use_python_runtime,
        )
        standard_out = compiled_model(*inputs)
        ref_out = model(*inputs)

        self.assertAlmostEqual(
            float(torch.max(torch.abs(ref_out - standard_out))),
            0,
            DECIMALS_OF_AGREEMENT,
            msg="Default standard execution (cudagraphs=False) outputs don't match with the original model.",
        )

    @parameterized.expand(
        [
            ("python_runtime", True),
            ("cpp_runtime", False),
        ]
    )
    def test_combination_of_cg_and_oa(self, _, use_python_runtime):
        model = StaticModel().eval().cuda()
        inputs = [torch.randn((2, 3), dtype=torch.float).cuda()]
        compiled_model = torch_tensorrt.compile(
            model,
            "dynamo",
            inputs,
            min_block_size=1,
            use_python_runtime=use_python_runtime,
        )

        with pytest.raises(
            RuntimeError,
            match="Both CUDA Graphs and dynamic output allocation are enabled, which are incompatible runtime modes. Please disable one of the two.",
        ):
            with torch_tensorrt.runtime.enable_cudagraphs(
                compiled_model
            ) as cudagraphs_module:
                with torch_tensorrt.runtime.enable_output_allocator(cudagraphs_module):
                    out = cudagraphs_module(*inputs)

        with pytest.raises(
            RuntimeError,
            match="Both CUDA Graphs and dynamic output allocation are enabled, which are incompatible runtime modes. Please disable one of the two.",
        ):
            with torch_tensorrt.runtime.enable_output_allocator(compiled_model):
                with torch_tensorrt.runtime.enable_cudagraphs(
                    compiled_model
                ) as cudagraphs_module:
                    out = cudagraphs_module(*inputs)


@unittest.skipIf(
    torch_tensorrt.ENABLED_FEATURES.tensorrt_rtx or is_thor() or is_tegra_platform(),
    "TensorRT RTX does not support nonzero which are required for this test",
)
class TestOutputAllocatorDDSModel(TestCase):
    @parameterized.expand(
        [
            ("python_runtime", True),
            ("cpp_runtime", False),
        ]
    )
    def test_cudagraphs_and_output_allocator(self, _, use_python_runtime):
        model = DDSModel().eval().cuda()
        inputs = (torch.randint(low=0, high=3, size=(10,), dtype=torch.int).to("cuda"),)
        compiled_model = torch_tensorrt.compile(
            model,
            "dynamo",
            inputs,
            min_block_size=1,
            use_python_runtime=use_python_runtime,
        )

        with pytest.raises(
            RuntimeError,
            match="The model contains submodules that require a dynamic output allocator at runtime, which is incompatible with CUDA Graphs. Please disable CUDA Graphs.",
        ):
            with torch_tensorrt.runtime.enable_cudagraphs(
                compiled_model
            ) as cudagraphs_module:
                cg_out = cudagraphs_module(*inputs)

        with torch_tensorrt.runtime.enable_output_allocator(compiled_model):
            oa_out = compiled_model(*inputs)

        ref_out = model(*inputs)

        self.assertAlmostEqual(
            float(torch.max(torch.abs(ref_out - oa_out))),
            0,
            DECIMALS_OF_AGREEMENT,
            msg="Output Allocator runtime outputs don't match with the original model.",
        )

    @parameterized.expand(
        [
            ("python_runtime", True),
            ("cpp_runtime", False),
        ]
    )
    def test_default(self, _, use_python_runtime):
        """
        DDS models use OutputAllocator by default.
        """
        model = DDSModel().eval().cuda()
        inputs = (torch.randint(low=0, high=3, size=(10,), dtype=torch.int).to("cuda"),)
        compiled_model = torch_tensorrt.compile(
            model,
            "dynamo",
            inputs,
            min_block_size=1,
            use_python_runtime=use_python_runtime,
        )
        oa_out = compiled_model(*inputs)
        ref_out = model(*inputs)

        self.assertAlmostEqual(
            float(torch.max(torch.abs(ref_out - oa_out))),
            0,
            DECIMALS_OF_AGREEMENT,
            msg="Default Output Allocator runtime outputs don't match with the original model.",
        )

    @parameterized.expand(
        [
            ("python_runtime", True),
            ("cpp_runtime", False),
        ]
    )
    def test_combination_of_cg_and_oa(self, _, use_python_runtime):
        model = DDSModel().eval().cuda()
        inputs = (torch.randint(low=0, high=3, size=(10,), dtype=torch.int).to("cuda"),)
        compiled_model = torch_tensorrt.compile(
            model,
            "dynamo",
            inputs,
            min_block_size=1,
            use_python_runtime=use_python_runtime,
        )

        with pytest.raises(
            RuntimeError,
            match="The model contains submodules that require a dynamic output allocator at runtime, which is incompatible with CUDA Graphs. Please disable CUDA Graphs.",
        ):
            with torch_tensorrt.runtime.enable_cudagraphs(
                compiled_model
            ) as cudagraphs_module:
                with torch_tensorrt.runtime.enable_output_allocator(cudagraphs_module):
                    out = cudagraphs_module(*inputs)

        with pytest.raises(
            RuntimeError,
            match="The model contains submodules that require a dynamic output allocator at runtime, which is incompatible with CUDA Graphs. Please disable CUDA Graphs.",
        ):
            with torch_tensorrt.runtime.enable_output_allocator(compiled_model):
                with torch_tensorrt.runtime.enable_cudagraphs(
                    compiled_model
                ) as cudagraphs_module:
                    out = cudagraphs_module(*inputs)


@unittest.skipIf(
    torch_tensorrt.ENABLED_FEATURES.tensorrt_rtx or is_thor() or is_tegra_platform(),
    "TensorRT RTX does not support nonzero which are required for this test",
)
class TestOutputAllocatorDDSOpWithReductionOpModel(TestCase):
    """
    The DDSOpWithReductionOpModel is a model that contains DDS op + reduction op.
    """

    @parameterized.expand(
        [
            ("python_runtime", True),
            ("cpp_runtime", False),
        ]
    )
    def test_cudagraphs_and_output_allocator(self, _, use_python_runtime):
        model = DDSOpWithReductionOpModel().eval().cuda()
        inputs = (torch.randint(low=0, high=3, size=(10,), dtype=torch.int).to("cuda"),)
        compiled_model = torch_tensorrt.compile(
            model,
            "dynamo",
            inputs,
            min_block_size=1,
            use_python_runtime=use_python_runtime,
        )

        with pytest.raises(
            RuntimeError,
            match="The model contains submodules that require a dynamic output allocator at runtime, which is incompatible with CUDA Graphs. Please disable CUDA Graphs.",
        ):
            with torch_tensorrt.runtime.enable_cudagraphs(
                compiled_model
            ) as cudagraphs_module:
                cg_out = cudagraphs_module(*inputs)

        with torch_tensorrt.runtime.enable_output_allocator(compiled_model):
            oa_out = compiled_model(*inputs)

        ref_out = model(*inputs)

        self.assertAlmostEqual(
            float(torch.max(torch.abs(ref_out - oa_out))),
            0,
            DECIMALS_OF_AGREEMENT,
            msg="Output Allocator runtime outputs don't match with the original model.",
        )

    @parameterized.expand(
        [
            ("python_runtime", True),
            ("cpp_runtime", False),
        ]
    )
    def test_default(self, _, use_python_runtime):
        """
        The DDSOpWithReductionOpModel is a model that contains nonzero op + reduction op, in which nonzero op requires output allocator.
        """
        model = DDSOpWithReductionOpModel().eval().cuda()
        inputs = (torch.randint(low=0, high=3, size=(10,), dtype=torch.int).to("cuda"),)
        compiled_model = torch_tensorrt.compile(
            model,
            "dynamo",
            inputs,
            min_block_size=1,
            use_python_runtime=use_python_runtime,
        )
        oa_out = compiled_model(*inputs)
        ref_out = model(*inputs)

        self.assertAlmostEqual(
            float(torch.max(torch.abs(ref_out - oa_out))),
            0,
            DECIMALS_OF_AGREEMENT,
            msg="Default Output Allocator runtime outputs don't match with the original model.",
        )

    @parameterized.expand(
        [
            ("python_runtime", True),
            ("cpp_runtime", False),
        ]
    )
    def test_combination_of_cg_and_oa(self, _, use_python_runtime):
        model = DDSOpWithReductionOpModel().eval().cuda()
        inputs = (torch.randint(low=0, high=3, size=(10,), dtype=torch.int).to("cuda"),)
        compiled_model = torch_tensorrt.compile(
            model,
            "dynamo",
            inputs,
            min_block_size=1,
            use_python_runtime=use_python_runtime,
        )

        with pytest.raises(
            RuntimeError,
            match="The model contains submodules that require a dynamic output allocator at runtime, which is incompatible with CUDA Graphs. Please disable CUDA Graphs.",
        ):
            with torch_tensorrt.runtime.enable_cudagraphs(
                compiled_model
            ) as cudagraphs_module:
                with torch_tensorrt.runtime.enable_output_allocator(cudagraphs_module):
                    out = cudagraphs_module(*inputs)

        with pytest.raises(
            RuntimeError,
            match="The model contains submodules that require a dynamic output allocator at runtime, which is incompatible with CUDA Graphs. Please disable CUDA Graphs.",
        ):
            with torch_tensorrt.runtime.enable_output_allocator(compiled_model):
                with torch_tensorrt.runtime.enable_cudagraphs(
                    compiled_model
                ) as cudagraphs_module:
                    out = cudagraphs_module(*inputs)


@unittest.skipIf(
    torch_tensorrt.ENABLED_FEATURES.tensorrt_rtx or is_thor() or is_tegra_platform(),
    "TensorRT RTX does not support nonzero which are required for this test",
)
class TestOutputAllocatorDDSModelWithGraphBreak(TestCase):
    @parameterized.expand(
        [
            ("python_runtime", True),
            ("cpp_runtime", False),
        ]
    )
    def test_cudagraphs_and_output_allocator(self, _, use_python_runtime):
        model = DDSModel2().eval().cuda()
        inputs = (torch.randint(low=0, high=3, size=(10,), dtype=torch.int).to("cuda"),)
        compiled_model = torch_tensorrt.compile(
            model,
            "dynamo",
            inputs,
            min_block_size=1,
            use_python_runtime=use_python_runtime,
            torch_executed_ops={"torch.ops.aten.abs.default"},
        )

        with pytest.raises(
            RuntimeError,
            match="The model contains submodules that require a dynamic output allocator at runtime, which is incompatible with CUDA Graphs. Please disable CUDA Graphs.",
        ):
            with torch_tensorrt.runtime.enable_cudagraphs(
                compiled_model
            ) as cudagraphs_module:
                cg_out = cudagraphs_module(*inputs)

        with torch_tensorrt.runtime.enable_output_allocator(compiled_model):
            oa_out = compiled_model(*inputs)

        ref_out = model(*inputs)

        self.assertAlmostEqual(
            float(torch.max(torch.abs(ref_out - oa_out))),
            0,
            DECIMALS_OF_AGREEMENT,
            msg="Output Allocator runtime outputs don't match with the original model.",
        )

    @parameterized.expand(
        [
            ("python_runtime", True),
            ("cpp_runtime", False),
        ]
    )
    def test_default(self, _, use_python_runtime):
        """
        Use Output Allocator by default.
        """
        model = DDSModel2().eval().cuda()
        inputs = (torch.randint(low=0, high=3, size=(10,), dtype=torch.int).to("cuda"),)
        compiled_model = torch_tensorrt.compile(
            model,
            "dynamo",
            inputs,
            min_block_size=1,
            use_python_runtime=use_python_runtime,
            torch_executed_ops={"torch.ops.aten.abs.default"},
        )
        oa_out = compiled_model(*inputs)
        ref_out = model(*inputs)

        self.assertAlmostEqual(
            float(torch.max(torch.abs(ref_out - oa_out))),
            0,
            DECIMALS_OF_AGREEMENT,
            msg="Default Output Allocator runtime outputs don't match with the original model.",
        )

    @parameterized.expand(
        [
            ("python_runtime", True),
            ("cpp_runtime", False),
        ]
    )
    def test_combination_of_cg_and_oa(self, _, use_python_runtime):
        model = DDSModel2().eval().cuda()
        inputs = (torch.randint(low=0, high=3, size=(10,), dtype=torch.int).to("cuda"),)
        compiled_model = torch_tensorrt.compile(
            model,
            "dynamo",
            inputs,
            min_block_size=1,
            use_python_runtime=use_python_runtime,
            torch_executed_ops={"torch.ops.aten.abs.default"},
        )

        with pytest.raises(
            RuntimeError,
            match="The model contains submodules that require a dynamic output allocator at runtime, which is incompatible with CUDA Graphs. Please disable CUDA Graphs.",
        ):
            with torch_tensorrt.runtime.enable_cudagraphs(
                compiled_model
            ) as cudagraphs_module:
                with torch_tensorrt.runtime.enable_output_allocator(cudagraphs_module):
                    out = cudagraphs_module(*inputs)

        with pytest.raises(
            RuntimeError,
            match="The model contains submodules that require a dynamic output allocator at runtime, which is incompatible with CUDA Graphs. Please disable CUDA Graphs.",
        ):
            with torch_tensorrt.runtime.enable_output_allocator(compiled_model):
                with torch_tensorrt.runtime.enable_cudagraphs(
                    compiled_model
                ) as cudagraphs_module:
                    out = cudagraphs_module(*inputs)


if __name__ == "__main__":
    run_tests()
