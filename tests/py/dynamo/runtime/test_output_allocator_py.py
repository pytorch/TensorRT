import pytest
import torch
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase, run_tests

from ..testing_utilities import DECIMALS_OF_AGREEMENT

INPUT_SIZE = (3, 16, 16)
TRIALS = 5


class StaticModel(torch.nn.Module):
    def forward(self, input):
        return torch.ops.aten.abs.default(input)


class DDSModel(torch.nn.Module):
    def forward(self, input):
        return torch.ops.aten.nonzero.default(input)


class NonDDSModel(torch.nn.Module):
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


class TestOutputAllocatorStaticModelPython(TestCase):
    def test_cudagraphs(self):
        model = StaticModel().eval().cuda()
        inputs = [torch.randn((2, 3), dtype=torch.float).cuda()]
        compiled_model = torch_tensorrt.compile(
            model,
            "dynamo",
            inputs,
            min_block_size=1,
            use_python_runtime=True,
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
            msg="CUDA Graphs Python TRT outputs don't match with the original model.",
        )

    def test_output_allocator(self):
        model = StaticModel().eval().cuda()
        inputs = [torch.randn((2, 3), dtype=torch.float).cuda()]
        compiled_model = torch_tensorrt.compile(
            model,
            "dynamo",
            inputs,
            min_block_size=1,
            use_python_runtime=True,
        )

        ref_out = model(*inputs)

        with torch_tensorrt.runtime.enable_output_allocator(compiled_model):
            oa_out = compiled_model(*inputs)

        self.assertAlmostEqual(
            float(torch.max(torch.abs(ref_out - oa_out))),
            0,
            DECIMALS_OF_AGREEMENT,
            msg="Output Allocator Python TRT outputs don't match with the original model.",
        )

    def test_default(self):
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
            use_python_runtime=True,
        )
        standard_out = compiled_model(*inputs)
        ref_out = model(*inputs)

        self.assertAlmostEqual(
            float(torch.max(torch.abs(ref_out - standard_out))),
            0,
            DECIMALS_OF_AGREEMENT,
            msg="Default standard execution outputs don't match with the original model.",
        )

    @pytest.mark.xfail(
        strict=True,
        raises=RuntimeError,
        reason="Both CUDA Graphs and OutputAllocator are enabled. Please disable either one.",
    )
    def test_cudagraphs_and_output_allocator(self):
        model = StaticModel().eval().cuda()
        inputs = [torch.randn((2, 3), dtype=torch.float).cuda()]
        compiled_model = torch_tensorrt.compile(
            model,
            "dynamo",
            inputs,
            min_block_size=1,
            use_python_runtime=True,
        )

        with torch_tensorrt.runtime.enable_cudagraphs(
            compiled_model
        ) as cudagraphs_module:
            with torch_tensorrt.runtime.enable_output_allocator(cudagraphs_module):
                out = cudagraphs_module(*inputs)

    @pytest.mark.xfail(
        strict=True,
        raises=RuntimeError,
        reason="Both CUDA Graphs and OutputAllocator are enabled. Please disable either one.",
    )
    def test_output_allocator_and_cudagraphs(self):
        model = StaticModel().eval().cuda()
        inputs = [torch.randn((2, 3), dtype=torch.float).cuda()]
        compiled_model = torch_tensorrt.compile(
            model,
            "dynamo",
            inputs,
            min_block_size=1,
            use_python_runtime=True,
        )
        with torch_tensorrt.runtime.enable_output_allocator(compiled_model):
            with torch_tensorrt.runtime.enable_cudagraphs(
                compiled_model
            ) as cudagraphs_module:
                out = cudagraphs_module(*inputs)


class TestOutputAllocatorDDSModelPython(TestCase):
    @pytest.mark.xfail(
        strict=True,
        raises=RuntimeError,
        reason="This module requires OutputAllocator which is not compatible with CUDA Graphs. Please disable CUDA Graphs.",
    )
    def test_cudagraphs(self):
        model = DDSModel().eval().cuda()
        inputs = (torch.randint(low=0, high=3, size=(10,), dtype=torch.int).to("cuda"),)
        compiled_model = torch_tensorrt.compile(
            model,
            "dynamo",
            inputs,
            min_block_size=1,
            use_python_runtime=True,
        )

        with torch_tensorrt.runtime.enable_cudagraphs(
            compiled_model
        ) as cudagraphs_module:
            cg_out = cudagraphs_module(*inputs)

    def test_output_allocator(self):
        model = DDSModel().eval().cuda()
        inputs = (torch.randint(low=0, high=3, size=(10,), dtype=torch.int).to("cuda"),)
        compiled_model = torch_tensorrt.compile(
            model,
            "dynamo",
            inputs,
            min_block_size=1,
            use_python_runtime=True,
        )

        ref_out = model(*inputs)

        with torch_tensorrt.runtime.enable_output_allocator(compiled_model):
            oa_out = compiled_model(*inputs)

        self.assertAlmostEqual(
            float(torch.max(torch.abs(ref_out - oa_out))),
            0,
            DECIMALS_OF_AGREEMENT,
            msg="Output Allocator Python TRT outputs don't match with the original model.",
        )

    def test_default(self):
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
            use_python_runtime=True,
        )
        oa_out = compiled_model(*inputs)
        ref_out = model(*inputs)

        self.assertAlmostEqual(
            float(torch.max(torch.abs(ref_out - oa_out))),
            0,
            DECIMALS_OF_AGREEMENT,
            msg="Default Output Allocator Python TRT outputs don't match with the original model.",
        )

    @pytest.mark.xfail(
        strict=True,
        raises=RuntimeError,
        reason="Both CUDA Graphs and OutputAllocator are enabled. Please disable either one.",
    )
    def test_cudagraphs_and_output_allocator(self):
        model = DDSModel().eval().cuda()
        inputs = (torch.randint(low=0, high=3, size=(10,), dtype=torch.int).to("cuda"),)
        compiled_model = torch_tensorrt.compile(
            model,
            "dynamo",
            inputs,
            min_block_size=1,
            use_python_runtime=True,
        )

        with torch_tensorrt.runtime.enable_cudagraphs(
            compiled_model
        ) as cudagraphs_module:
            with torch_tensorrt.runtime.enable_output_allocator(cudagraphs_module):
                out = cudagraphs_module(*inputs)

    @pytest.mark.xfail(
        strict=True,
        raises=RuntimeError,
        reason="Both CUDA Graphs and OutputAllocator are enabled. Please disable either one.",
    )
    def test_output_allocator_and_cudagraphs(self):
        model = DDSModel().eval().cuda()
        inputs = (torch.randint(low=0, high=3, size=(10,), dtype=torch.int).to("cuda"),)
        compiled_model = torch_tensorrt.compile(
            model,
            "dynamo",
            inputs,
            min_block_size=1,
            use_python_runtime=True,
        )
        with torch_tensorrt.runtime.enable_output_allocator(compiled_model):
            with torch_tensorrt.runtime.enable_cudagraphs(
                compiled_model
            ) as cudagraphs_module:
                out = cudagraphs_module(*inputs)


class TestOutputAllocatorNonDDSModelPython(TestCase):
    @pytest.mark.xfail(
        strict=True,
        raises=RuntimeError,
        reason="This module requires OutputAllocator which is not compatible with CUDA Graphs. Please disable CUDA Graphs.",
    )
    def test_cudagraphs(self):
        model = NonDDSModel().eval().cuda()
        inputs = (torch.randint(low=0, high=3, size=(10,), dtype=torch.int).to("cuda"),)
        compiled_model = torch_tensorrt.compile(
            model,
            "dynamo",
            inputs,
            min_block_size=1,
            use_python_runtime=True,
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
            msg="CUDA Graphs Python TRT outputs don't match with the original model.",
        )

    def test_output_allocator(self):
        model = NonDDSModel().eval().cuda()
        inputs = (torch.randint(low=0, high=3, size=(10,), dtype=torch.int).to("cuda"),)
        compiled_model = torch_tensorrt.compile(
            model,
            "dynamo",
            inputs,
            min_block_size=1,
            use_python_runtime=True,
        )

        ref_out = model(*inputs)

        with torch_tensorrt.runtime.enable_output_allocator(compiled_model):
            oa_out = compiled_model(*inputs)

        self.assertAlmostEqual(
            float(torch.max(torch.abs(ref_out - oa_out))),
            0,
            DECIMALS_OF_AGREEMENT,
            msg="Output Allocator Python TRT outputs don't match with the original model.",
        )

    def test_default(self):
        """
        NonDDS models use standard execution with cudagraphs=False by default.
        """
        model = DDSModel().eval().cuda()
        inputs = (torch.randint(low=0, high=3, size=(10,), dtype=torch.int).to("cuda"),)
        compiled_model = torch_tensorrt.compile(
            model,
            "dynamo",
            inputs,
            min_block_size=1,
            use_python_runtime=True,
        )
        standard_out = compiled_model(*inputs)
        ref_out = model(*inputs)

        self.assertAlmostEqual(
            float(torch.max(torch.abs(ref_out - standard_out))),
            0,
            DECIMALS_OF_AGREEMENT,
            msg="Default standard execution outputs don't match with the original model.",
        )

    @pytest.mark.xfail(
        strict=True,
        raises=RuntimeError,
        reason="Both CUDA Graphs and OutputAllocator are enabled. Please disable either one.",
    )
    def test_cudagraphs_and_output_allocator(self):
        model = NonDDSModel().eval().cuda()
        inputs = (torch.randint(low=0, high=3, size=(10,), dtype=torch.int).to("cuda"),)
        compiled_model = torch_tensorrt.compile(
            model,
            "dynamo",
            inputs,
            min_block_size=1,
            use_python_runtime=True,
        )

        with torch_tensorrt.runtime.enable_cudagraphs(
            compiled_model
        ) as cudagraphs_module:
            with torch_tensorrt.runtime.enable_output_allocator(cudagraphs_module):
                out = cudagraphs_module(*inputs)

    @pytest.mark.xfail(
        strict=True,
        raises=RuntimeError,
        reason="Both CUDA Graphs and OutputAllocator are enabled. Please disable either one.",
    )
    def test_output_allocator_and_cudagraphs(self):
        model = NonDDSModel().eval().cuda()
        inputs = (torch.randint(low=0, high=3, size=(10,), dtype=torch.int).to("cuda"),)
        compiled_model = torch_tensorrt.compile(
            model,
            "dynamo",
            inputs,
            min_block_size=1,
            use_python_runtime=True,
        )
        with torch_tensorrt.runtime.enable_output_allocator(compiled_model):
            with torch_tensorrt.runtime.enable_cudagraphs(
                compiled_model
            ) as cudagraphs_module:
                out = cudagraphs_module(*inputs)


class TestOutputAllocatorDDSModelWithGraphBreakPython(TestCase):
    @pytest.mark.xfail(
        strict=True,
        raises=RuntimeError,
        reason="This module requires OutputAllocator which is not compatible with CUDA Graphs. Please disable CUDA Graphs.",
    )
    def test_cudagraphs(self):
        model = DDSModel2().eval().cuda()
        inputs = (torch.randint(low=0, high=3, size=(10,), dtype=torch.int).to("cuda"),)
        compiled_model = torch_tensorrt.compile(
            model,
            "dynamo",
            inputs,
            min_block_size=1,
            use_python_runtime=True,
            torch_executed_ops={"torch.ops.aten.abs.default"},
        )

        with torch_tensorrt.runtime.enable_cudagraphs(
            compiled_model
        ) as cudagraphs_module:
            cg_out = cudagraphs_module(*inputs)

    def test_output_allocator(self):
        model = DDSModel2().eval().cuda()
        inputs = (torch.randint(low=0, high=3, size=(10,), dtype=torch.int).to("cuda"),)
        compiled_model = torch_tensorrt.compile(
            model,
            "dynamo",
            inputs,
            min_block_size=1,
            use_python_runtime=True,
            torch_executed_ops={"torch.ops.aten.abs.default"},
        )

        ref_out = model(*inputs)

        with torch_tensorrt.runtime.enable_output_allocator(compiled_model):
            oa_out = compiled_model(*inputs)

        self.assertAlmostEqual(
            float(torch.max(torch.abs(ref_out - oa_out))),
            0,
            DECIMALS_OF_AGREEMENT,
            msg="Output Allocator Python TRT outputs don't match with the original model.",
        )

    def test_default(self):
        """
        Use OutputAllocator by default.
        """
        model = DDSModel2().eval().cuda()
        inputs = (torch.randint(low=0, high=3, size=(10,), dtype=torch.int).to("cuda"),)
        compiled_model = torch_tensorrt.compile(
            model,
            "dynamo",
            inputs,
            min_block_size=1,
            use_python_runtime=True,
            torch_executed_ops={"torch.ops.aten.abs.default"},
        )
        oa_out = compiled_model(*inputs)
        ref_out = model(*inputs)

        self.assertAlmostEqual(
            float(torch.max(torch.abs(ref_out - oa_out))),
            0,
            DECIMALS_OF_AGREEMENT,
            msg="Default Output Allocator Python TRT outputs don't match with the original model.",
        )

    @pytest.mark.xfail(
        strict=True,
        raises=RuntimeError,
        reason="Both CUDA Graphs and OutputAllocator are enabled. Please disable either one.",
    )
    def test_cudagraphs_and_output_allocator(self):
        model = DDSModel2().eval().cuda()
        inputs = (torch.randint(low=0, high=3, size=(10,), dtype=torch.int).to("cuda"),)
        compiled_model = torch_tensorrt.compile(
            model,
            "dynamo",
            inputs,
            min_block_size=1,
            use_python_runtime=True,
            torch_executed_ops={"torch.ops.aten.abs.default"},
        )

        with torch_tensorrt.runtime.enable_cudagraphs(
            compiled_model
        ) as cudagraphs_module:
            with torch_tensorrt.runtime.enable_output_allocator(cudagraphs_module):
                out = cudagraphs_module(*inputs)

    @pytest.mark.xfail(
        strict=True,
        raises=RuntimeError,
        reason="Both CUDA Graphs and OutputAllocator are enabled. Please disable either one.",
    )
    def test_output_allocator_and_cudagraphs(self):
        model = DDSModel2().eval().cuda()
        inputs = (torch.randint(low=0, high=3, size=(10,), dtype=torch.int).to("cuda"),)
        compiled_model = torch_tensorrt.compile(
            model,
            "dynamo",
            inputs,
            min_block_size=1,
            use_python_runtime=True,
            torch_executed_ops={"torch.ops.aten.abs.default"},
        )
        with torch_tensorrt.runtime.enable_output_allocator(compiled_model):
            with torch_tensorrt.runtime.enable_cudagraphs(
                compiled_model
            ) as cudagraphs_module:
                out = cudagraphs_module(*inputs)


if __name__ == "__main__":
    run_tests()
