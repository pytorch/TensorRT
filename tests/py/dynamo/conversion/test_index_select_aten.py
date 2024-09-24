import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import param, parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input
from torch_tensorrt.dynamo.utils import ATOL, RTOL

from .harness import DispatchTestCase


class TestIndexSelectConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("1d_input", (10,), 0, (1,)),
            ("2d_input_dim_0", (10, 3), 0, (0, 2)),
            ("2d_input_dim_1", (5, 10), 1, (1, 2, 3)),
            ("2d_input_dim_-2", (5, 10), -2, (1, 2, 3)),
            ("3d_input_dim_0", (10, 5, 10), 0, (0, 5)),
            ("3d_input_dim_2", (10, 5, 10), 2, (3, 3, 4)),
            ("3d_input_dim_-1", (10, 5, 10), -1, (3, 3, 4)),
            ("3d_input_dim_-3", (10, 5, 10), -3, (5, 3, 4)),
        ]
    )
    def test_index_select(self, _, source_shape, dim, indices_val):
        class TestIndexSelect(torch.nn.Module):
            def forward(self, source_tensor, indices_tensor):
                return torch.ops.aten.index_select.default(
                    source_tensor, dim, indices_tensor
                )

        input = [
            torch.randn(*source_shape, dtype=torch.float32),
            torch.tensor([*indices_val], dtype=torch.int32),
        ]

        self.run_test(
            TestIndexSelect(),
            input,
        )

    @parameterized.expand(
        [
            param(
                # 1d_source_tensor
                # source_tensor is for compile
                source_tensor=torch.randn((3,), dtype=torch.float32),
                # source_tensor_1 is for inference
                source_tensor_1=torch.randn((5,), dtype=torch.float32),
                dynamic_shapes={
                    "source_tensor": {0: torch.export.Dim("dyn_dim", min=3, max=6)},
                    "indice_tensor": {},
                },
                dim=0,
                indice_tensor=torch.tensor(
                    [
                        1,
                    ],
                    dtype=torch.int32,
                ),
            ),
            param(
                # 2d_source_tensor
                # source_tensor is for compile
                source_tensor=torch.randn((3, 3), dtype=torch.float32),
                # source_tensor_1 is for inference
                source_tensor_1=torch.randn((4, 6), dtype=torch.float32),
                dynamic_shapes={
                    "source_tensor": {
                        0: torch.export.Dim("dyn_dim1", min=3, max=6),
                        1: torch.export.Dim("dyn_dim2", min=2, max=7),
                    },
                    "indice_tensor": {},
                },
                dim=-1,
                indice_tensor=torch.tensor([0, 2], dtype=torch.int32),
            ),
            param(
                # 3d_source_tensor
                # source_tensor is for compile
                source_tensor=torch.randn((3, 4, 2), dtype=torch.float32),
                # source_tensor_1 is for inference
                source_tensor_1=torch.randn((6, 7, 2), dtype=torch.float32),
                dynamic_shapes={
                    "source_tensor": {
                        0: torch.export.Dim("dyn_dim1", min=3, max=6),
                        1: torch.export.Dim("dyn_dim2", min=2, max=7),
                    },
                    "indice_tensor": {},
                },
                dim=-2,
                indice_tensor=torch.tensor([0, 0, 2], dtype=torch.int32),
            ),
        ]
    )
    def test_index_select_dynamic_shape(
        self, source_tensor, source_tensor_1, dynamic_shapes, dim, indice_tensor
    ):
        class IndexSelect(torch.nn.Module):
            def forward(self, source_tensor, indice_tensor):
                return torch.ops.aten.index_select.default(
                    source_tensor,
                    dim,
                    indice_tensor,
                )

        inputs = (source_tensor, indice_tensor)
        mod = IndexSelect()

        fx_mod = torch.export.export(mod, inputs, dynamic_shapes=dynamic_shapes)
        trt_mod = torch_tensorrt.dynamo.compile(
            fx_mod,
            inputs=inputs,
            enable_precisions=torch.float32,
            min_block_size=1,
            cache_built_engines=False,
            reuse_cached_engines=False,
        )
        # use different shape of inputs for inference:
        inputs = (source_tensor_1, indice_tensor)
        with torch.no_grad():
            cuda_inputs = []
            for i in inputs:
                cuda_inputs.append(i.cuda())
            ref_outputs = mod(*cuda_inputs)
            outputs = trt_mod(*cuda_inputs)
            for out, ref in zip(outputs, ref_outputs):
                torch.testing.assert_close(
                    out,
                    ref,
                    rtol=RTOL,
                    atol=ATOL,
                    equal_nan=True,
                    check_dtype=True,
                )


if __name__ == "__main__":
    run_tests()
