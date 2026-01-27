import torch
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from ..conversion.harness import DispatchTestCase


class TestFusedRMSNormLoweringPass(DispatchTestCase):
    """
    Tests for the aten._fused_rms_norm.default lowering pass.
    RMS Normalization formula: output = input / sqrt(mean(input^2) + eps) * weight
    The operation signature is: _fused_rms_norm(input, normalized_shape, weight, eps)
    Returns: (output, rstd) - where rstd is the reciprocal standard deviation
    """

    @parameterized.expand(
        [
            # Test normalizing over last dimension
            ("1d_last_dim", (2, 4, 8), [8]),
            # Test normalizing over last 2 dimensions
            ("2d_last_two_dims", (2, 4, 8), [4, 8]),
            # Test normalizing over all dimensions
            ("3d_all_dims", (2, 4, 8), [2, 4, 8]),
            # Test with 4D tensor, last dimension
            ("4d_last_dim", (2, 3, 4, 8), [8]),
            # Test with 4D tensor, last 2 dimensions
            ("4d_last_two_dims", (2, 3, 4, 8), [4, 8]),
            # Test with 4D tensor, last 3 dimensions
            ("4d_last_three_dims", (2, 3, 4, 8), [3, 4, 8]),
        ]
    )
    def test_rms_norm_with_weight(self, name, input_shape, normalized_shape):
        """
        Test RMS norm with weight parameter across various tensor shapes.
        This tests:
        - Correct dimension calculation for normalization
        - Weight broadcasting/expansion to match input shape
        - Output correctness vs PyTorch reference
        """

        class RMSNorm(torch.nn.Module):
            def forward(self, x, weight):
                return torch.ops.aten._fused_rms_norm.default(
                    x, normalized_shape, weight, 1e-5
                )[
                    0
                ]  # Return only the normalized output, not rstd

        inputs = [
            torch.randn(input_shape),
            torch.randn(normalized_shape),
        ]
        self.run_test(
            RMSNorm(),
            inputs,
            use_dynamo_tracer=True,
            enable_passes=True,
        )

    @parameterized.expand(
        [
            # Test without weight (None)
            ("1d_no_weight", (2, 4, 8), [8]),
            ("2d_no_weight", (2, 4, 8), [4, 8]),
            ("4d_no_weight", (2, 3, 4, 8), [8]),
        ]
    )
    def test_rms_norm_without_weight(self, name, input_shape, normalized_shape):
        """
        Test RMS norm without weight parameter (weight=None).
        This ensures the lowering pass handles optional weight correctly.
        """

        class RMSNorm(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten._fused_rms_norm.default(
                    x, normalized_shape, None, 1e-5
                )[0]

        inputs = [torch.randn(input_shape)]
        self.run_test(
            RMSNorm(),
            inputs,
            use_dynamo_tracer=True,
            enable_passes=True,
        )

    @parameterized.expand(
        [
            # Test different epsilon values
            ("eps_1e5", (2, 4, 8), [8], 1e-5),
            ("eps_1e6", (2, 4, 8), [8], 1e-6),
            ("eps_1e4", (2, 4, 8), [8], 1e-4),
        ]
    )
    def test_rms_norm_different_eps(self, name, input_shape, normalized_shape, eps):
        """
        Test RMS norm with different epsilon values.
        Epsilon is critical for numerical stability, especially with small values.
        """

        class RMSNorm(torch.nn.Module):
            def forward(self, x, weight):
                return torch.ops.aten._fused_rms_norm.default(
                    x, normalized_shape, weight, eps
                )[0]

        inputs = [
            torch.randn(input_shape),
            torch.randn(normalized_shape),
        ]
        self.run_test(
            RMSNorm(),
            inputs,
            use_dynamo_tracer=True,
            enable_passes=True,
        )

    def test_rms_norm_with_dynamic_shape_batch(self):
        """
        Test RMS norm with dynamic batch dimension.
        This is common in inference scenarios where batch size varies.
        """

        class RMSNorm(torch.nn.Module):
            def forward(self, x, weight):
                return torch.ops.aten._fused_rms_norm.default(x, [128], weight, 1e-6)[0]

        input_specs = [
            Input(
                shape=(-1, 128),
                dtype=torch.float32,
                shape_ranges=[((1, 128), (4, 128), (8, 128))],
            ),
            Input(
                shape=(128,),
                dtype=torch.float32,
            ),
        ]

        self.run_test_with_dynamic_shape(
            RMSNorm(),
            input_specs,
            use_dynamo_tracer=True,
            enable_passes=True,
        )

    def test_rms_norm_with_dynamic_shape_sequence(self):
        """
        Test RMS norm with dynamic sequence length.
        This is critical for transformer models with variable sequence lengths.
        """

        class RMSNorm(torch.nn.Module):
            def forward(self, x, weight):
                return torch.ops.aten._fused_rms_norm.default(x, [256], weight, 1e-5)[0]

        input_specs = [
            Input(
                shape=(2, -1, 256),
                dtype=torch.float32,
                shape_ranges=[((2, 16, 256), (2, 64, 256), (2, 128, 256))],
            ),
            Input(
                shape=(256,),
                dtype=torch.float32,
            ),
        ]

        self.run_test_with_dynamic_shape(
            RMSNorm(),
            input_specs,
            use_dynamo_tracer=True,
            enable_passes=True,
        )

    def test_rms_norm_with_dynamic_shape_multi_dim(self):
        """
        Test RMS norm with multiple dynamic dimensions.
        Tests both batch and sequence length being dynamic simultaneously.
        """

        class RMSNorm(torch.nn.Module):
            def forward(self, x, weight):
                return torch.ops.aten._fused_rms_norm.default(x, [64], weight, 1e-6)[0]

        input_specs = [
            Input(
                shape=(-1, -1, 64),
                dtype=torch.float32,
                shape_ranges=[((1, 8, 64), (4, 16, 64), (8, 32, 64))],
            ),
            Input(
                shape=(64,),
                dtype=torch.float32,
            ),
        ]

        self.run_test_with_dynamic_shape(
            RMSNorm(),
            input_specs,
            use_dynamo_tracer=True,
            enable_passes=True,
        )

    def test_rms_norm_2d_input(self):
        """
        Test RMS norm with 2D input (batch, features).
        Common in MLP layers or simple feedforward networks.
        """

        class RMSNorm(torch.nn.Module):
            def forward(self, x, weight):
                return torch.ops.aten._fused_rms_norm.default(x, [512], weight, 1e-5)[0]

        inputs = [
            torch.randn(32, 512),
            torch.randn(512),
        ]
        self.run_test(
            RMSNorm(),
            inputs,
            use_dynamo_tracer=True,
            enable_passes=True,
        )

    def test_rms_norm_large_hidden_dim(self):
        """
        Test RMS norm with larger hidden dimensions typical in modern LLMs.
        Tests numerical stability and performance with realistic model sizes.
        """

        class RMSNorm(torch.nn.Module):
            def forward(self, x, weight):
                return torch.ops.aten._fused_rms_norm.default(x, [4096], weight, 1e-6)[
                    0
                ]

        inputs = [
            torch.randn(2, 8, 4096),
            torch.randn(4096),
        ]
        self.run_test(
            RMSNorm(),
            inputs,
            use_dynamo_tracer=True,
            enable_passes=True,
        )

    def test_rms_norm_flux_pattern(self):
        """
        Test RMS norm with pattern similar to FLUX and modern diffusion models.
        This tests the actual use case that motivated the lowering pass implementation.
        """

        class RMSNorm(torch.nn.Module):
            def forward(self, x, weight):
                # FLUX-style: normalize over last dimension with small epsilon
                normalized_shape = [x.shape[-1]]
                return torch.ops.aten._fused_rms_norm.default(
                    x, normalized_shape, weight, 1e-6
                )[0]

        inputs = [
            torch.randn(1, 16, 3072),  # Typical FLUX dimensions
            torch.randn(3072),
        ]
        self.run_test(
            RMSNorm(),
            inputs,
            use_dynamo_tracer=True,
            enable_passes=True,
        )

    def test_rms_norm_with_dynamic_shape_and_graph_break(self):
        """
        Test RMS norm with dynamic batch dimension and graph break.
        This tests the lowering logic to handle graph breaks correctly.
        """

        class RMSNorm(torch.nn.Module):
            def forward(self, x, weight):
                return torch.ops.aten._fused_rms_norm.default(x, [128], weight, 1e-6)

        inputs = (
            torch.randn(1, 128).cuda(),
            torch.randn(128).cuda(),
        )

        ep = torch.export.export(
            RMSNorm().cuda(),
            args=inputs,
        )
        trt_gm = torch_tensorrt.dynamo.compile(
            ep,
            inputs=inputs,
            torch_executed_ops={torch.ops.aten.rsqrt.default},
            truncate_double=True,
            dryrun=False,
            min_block_size=1,
        )

        self.assertTrue(torch.allclose(trt_gm(*inputs)[0], RMSNorm()(*inputs)[0]))
        self.assertTrue(torch.allclose(trt_gm(*inputs)[1], RMSNorm()(*inputs)[1]))


if __name__ == "__main__":
    run_tests()
