import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestCatConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("pos", 1),
            ("neg", -2),
        ]
    )
    def test_cat(self, _, dim):
        class Cat(nn.Module):
            def forward(self, x, y, z):
                return torch.ops.aten.cat.default((x, y, z), dim)

        inputs = [torch.randn(1, 2, 3), torch.randn(1, 1, 3), torch.randn(1, 3, 3)]
        self.run_test(
            Cat(),
            inputs,
        )

    @parameterized.expand(
        [
            ("pos", 1),
            ("neg", -2),
        ]
    )
    def test_cat_dim_in_kwargs(self, _, dim):
        class Cat(nn.Module):
            def forward(self, x, y, z):
                return torch.ops.aten.cat.default((x, y, z), dim=dim)

        inputs = [torch.randn(1, 2, 3), torch.randn(1, 1, 3), torch.randn(1, 3, 3)]
        self.run_test(
            Cat(),
            inputs,
        )

    @parameterized.expand(
        [
            ("pos", 1),
            ("neg", -2),
        ]
    )
    def test_cat_dynamic_shape(self, _, dim):
        class Cat(nn.Module):
            def forward(self, x, y):
                return torch.ops.aten.cat.default((x, y), dim)

        input_specs = [
            Input(
                dtype=torch.float32,
                min_shape=(16, 2, 3),
                opt_shape=(16, 3, 3),
                max_shape=(16, 32, 3),
            ),
            Input(
                dtype=torch.float32,
                min_shape=(16, 2, 3),
                opt_shape=(16, 16, 3),
                max_shape=(16, 32, 3),
            ),
        ]
        self.run_test_with_dynamic_shape(
            Cat(),
            input_specs,
        )

    def test_cat_no_dim(self):
        class Cat(nn.Module):
            def forward(self, x, y, z):
                return torch.ops.aten.cat.default((x, y, z))

        inputs = [torch.randn(2, 1, 3), torch.randn(1, 1, 3), torch.randn(3, 1, 3)]
        self.run_test(
            Cat(),
            inputs,
        )

    def test_cat_dynamic_shape_no_dim(self):
        class Cat(nn.Module):
            def forward(self, x, y):
                return torch.ops.aten.cat.default((x, y))

        input_specs = [
            Input(
                dtype=torch.float32,
                min_shape=(2, 16, 3),
                opt_shape=(3, 16, 3),
                max_shape=(32, 16, 3),
            ),
            Input(
                dtype=torch.float32,
                min_shape=(2, 16, 3),
                opt_shape=(3, 16, 3),
                max_shape=(32, 16, 3),
            ),
        ]
        self.run_test_with_dynamic_shape(
            Cat(),
            input_specs,
        )

    @parameterized.expand(
        [
            ("pos", 1),
            ("neg", -2),
        ]
    )
    def test_cat_dynamic_shape_dim(self, _, dim):
        class Cat(nn.Module):
            def forward(self, x, y):
                return torch.ops.aten.cat.default((x, y), dim)

        input_specs = [
            Input(
                dtype=torch.float32,
                min_shape=(2, 1, 1),
                opt_shape=(3, 1, 2),
                max_shape=(4, 1, 3),
            ),
            Input(
                dtype=torch.float32,
                min_shape=(2, 2, 1),
                opt_shape=(3, 3, 2),
                max_shape=(4, 4, 3),
            ),
        ]
        self.run_test_with_dynamic_shape(
            Cat(),
            input_specs,
        )

    def test_cat_mixed_dtype_fp32_fp16(self):
        """Test cat with mixed float32 and float16 tensors - should promote to float32"""

        class MixedDtypeCat(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "const_fp16", torch.ones(2, 3, dtype=torch.float16)
                )

            def forward(self, x):
                # x is float32, const_fp16 is float16, result should be float32
                return torch.ops.aten.cat.default((self.const_fp16, x), 0)

        inputs = [torch.randn(2, 3, device="cuda", dtype=torch.float32)]
        self.run_test(
            MixedDtypeCat(),
            inputs,
        )

    def test_cat_mixed_dtype_int32_int64(self):
        """Test cat with mixed int32 and int64 tensors - should promote to int64"""

        class MixedIntCat(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("const_int32", torch.ones(2, 3, dtype=torch.int32))

            def forward(self, x):
                # x is int64, const_int32 is int32, result should be int64
                return torch.ops.aten.cat.default((self.const_int32, x), 0)

        inputs = [torch.ones(2, 3, device="cuda", dtype=torch.int64)]
        self.run_test(
            MixedIntCat(),
            inputs,
        )

    def test_cat_three_different_dtypes(self):
        """Test cat with three different dtypes - bfloat16, float16, float32"""

        class ThreeDtypeCat(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "const_bf16", torch.ones(2, 3, dtype=torch.bfloat16)
                )
                self.register_buffer(
                    "const_fp16", torch.ones(2, 3, dtype=torch.float16)
                )

            def forward(self, x):
                # bf16, fp16, fp32 -> should promote to fp32
                return torch.ops.aten.cat.default(
                    (self.const_bf16, self.const_fp16, x), 0
                )

        inputs = [torch.randn(2, 3, device="cuda", dtype=torch.float32)]
        self.run_test(
            ThreeDtypeCat(),
            inputs,
        )

    def test_cat_many_tensors(self):
        """Test cat with many tensors (10+)"""

        class ManyCat(nn.Module):
            def forward(self, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9):
                return torch.ops.aten.cat.default(
                    (t0, t1, t2, t3, t4, t5, t6, t7, t8, t9), 0
                )

        # Create 10 small tensors
        inputs = [torch.randn(1, 3, device="cuda") for _ in range(10)]
        self.run_test(
            ManyCat(),
            inputs,
        )

    def test_cat_single_tensor(self):
        """Test cat with a single tensor (edge case)"""

        class SingleCat(nn.Module):
            def forward(self, x):
                return torch.ops.aten.cat.default((x,), 0)

        inputs = [torch.randn(2, 3, device="cuda")]
        self.run_test(
            SingleCat(),
            inputs,
        )

    def test_cat_4d_tensors(self):
        """Test cat with 4D tensors (batch, channels, height, width)"""

        class Cat4D(nn.Module):
            def forward(self, x, y, z):
                return torch.ops.aten.cat.default((x, y, z), 1)  # concat on channels

        inputs = [
            torch.randn(2, 3, 8, 8, device="cuda"),
            torch.randn(2, 5, 8, 8, device="cuda"),
            torch.randn(2, 7, 8, 8, device="cuda"),
        ]
        self.run_test(
            Cat4D(),
            inputs,
        )

    def test_cat_5d_tensors(self):
        """Test cat with 5D tensors"""

        class Cat5D(nn.Module):
            def forward(self, x, y):
                return torch.ops.aten.cat.default((x, y), 2)

        inputs = [
            torch.randn(2, 3, 4, 5, 6, device="cuda"),
            torch.randn(2, 3, 7, 5, 6, device="cuda"),
        ]
        self.run_test(
            Cat5D(),
            inputs,
        )

    def test_cat_1d_tensors(self):
        """Test cat with 1D tensors"""

        class Cat1D(nn.Module):
            def forward(self, x, y, z):
                return torch.ops.aten.cat.default((x, y, z), 0)

        inputs = [
            torch.randn(10, device="cuda"),
            torch.randn(20, device="cuda"),
            torch.randn(15, device="cuda"),
        ]
        self.run_test(
            Cat1D(),
            inputs,
        )

    def test_cat_large_tensors(self):
        """Test cat with larger tensors"""

        class CatLarge(nn.Module):
            def forward(self, x, y):
                return torch.ops.aten.cat.default((x, y), 1)

        inputs = [
            torch.randn(32, 512, 14, 14, device="cuda"),
            torch.randn(32, 256, 14, 14, device="cuda"),
        ]
        self.run_test(
            CatLarge(),
            inputs,
        )

    @parameterized.expand(
        [
            ("dim0", 0),
            ("dim1", 1),
            ("dim2", 2),
        ]
    )
    def test_cat_different_concat_dims(self, _, dim):
        """Test cat along different dimensions with same-sized inputs"""

        class CatDifferentDims(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim

            def forward(self, x, y):
                return torch.ops.aten.cat.default((x, y), self.dim)

        inputs = [
            torch.randn(4, 5, 6, device="cuda"),
            torch.randn(4, 5, 6, device="cuda"),
        ]
        self.run_test(
            CatDifferentDims(dim),
            inputs,
        )

    # Note: int8 test removed - TensorRT requires dynamic range/calibration for int8
    # which is not supported in this test framework

    def test_cat_mixed_int_float(self):
        """Test cat with mixed int32 and float32 - should promote to float32"""

        class MixedIntFloatCat(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("const_int", torch.ones(2, 3, dtype=torch.int32))

            def forward(self, x):
                return torch.ops.aten.cat.default((self.const_int, x), 0)

        inputs = [torch.randn(2, 3, device="cuda", dtype=torch.float32)]
        self.run_test(
            MixedIntFloatCat(),
            inputs,
        )

    def test_cat_bf16_dtype_preservation(self):
        """Test that bfloat16 dtype is preserved in constant layers (not converted to fp32)"""

        class CatBF16Constants(nn.Module):
            def __init__(self):
                super().__init__()
                # Register multiple bf16 constant buffers
                self.register_buffer(
                    "bf16_const1", torch.ones(2, 3, dtype=torch.bfloat16)
                )
                self.register_buffer(
                    "bf16_const2", torch.full((2, 3), 2.0, dtype=torch.bfloat16)
                )

            def forward(self, x):
                # Cat bf16 input with bf16 constants - output should be bf16
                return torch.ops.aten.cat.default(
                    (self.bf16_const1, x, self.bf16_const2), 0
                )

        inputs = [torch.randn(2, 3, device="cuda", dtype=torch.bfloat16)]
        self.run_test(
            CatBF16Constants(),
            inputs,
        )


if __name__ == "__main__":
    run_tests()
