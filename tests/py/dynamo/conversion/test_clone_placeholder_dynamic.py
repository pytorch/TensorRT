import torch
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests


class TestClonePlaceholderDynamicShape(TestCase):
    """clone and _to_copy each have two registrations, a general one and one for the case
    where the node takes a graph input and its result is a graph output. Both call the same
    dtype cast, which reads no shapes, so both must accept a dynamic input.

    The cases the cast cannot serve have to keep falling back, since they run correctly in
    PyTorch: a memory_format it cannot preserve, and an input dtype the engine cannot bind.
    """

    @staticmethod
    def _compile(module, inputs, dim_max=6, **kwargs):
        batch = torch.export.Dim("batch", min=1, max=dim_max)
        exported = torch.export.export(module, inputs, dynamic_shapes=({0: batch},))
        return torch_tensorrt.dynamo.compile(
            exported, arg_inputs=inputs, min_block_size=1, **kwargs
        )

    @staticmethod
    def _engines(compiled):
        return sum(1 for name, _ in compiled.named_children() if "_run_on_acc" in name)

    @parameterized.expand(
        [
            ("clone", lambda x: torch.ops.aten.clone.default(x)),
            (
                "to_copy",
                lambda x: torch.ops.aten._to_copy.default(x, dtype=torch.float16),
            ),
        ]
    )
    def test_placeholder_copy_with_dynamic_dim(self, _, operation):
        """Both registrations are fixed, so both need a case. Reverting either flag alone
        left the other one's test green."""

        class OnlyCopy(torch.nn.Module):
            def forward(self, x):
                return operation(x)

        module = OnlyCopy().eval().cuda()
        inputs = (torch.randn(3, 5, device="cuda"),)
        compiled = self._compile(module, inputs)

        self.assertEqual(
            self._engines(compiled),
            1,
            f"expected one engine, got {[n for n, _ in compiled.named_children()]}",
        )
        # The declared range is 1 to 6, so run both ends of it as well as the middle.
        for size in (1, 3, 6):
            sized = (torch.randn(size, 5, device="cuda"),)
            torch.testing.assert_close(compiled(*sized), module(*sized))

    def test_channels_last_falls_back(self):
        """The cast produces standard contiguous strides, so a channels-last copy would
        return the right values with the wrong layout. Comparing values alone cannot see
        that, which is why this asserts on the strides."""

        class CloneChannelsLast(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.clone.default(
                    x, memory_format=torch.channels_last
                )

        module = CloneChannelsLast().eval().cuda()
        inputs = (
            torch.randn(3, 4, 5, 3, device="cuda").to(
                memory_format=torch.channels_last
            ),
        )
        compiled = self._compile(module, inputs)
        result = compiled(*inputs)

        self.assertTrue(
            result.is_contiguous(memory_format=torch.channels_last),
            f"expected channels-last strides, got {result.stride()}",
        )
        torch.testing.assert_close(result, module(*inputs))

    def test_default_clone_of_channels_last_falls_back(self):
        """A default clone with no memory_format still preserves the input strides, so a
        channels-last input keeps its layout in eager. Export represents this with empty
        kwargs, so it is not caught by the explicit-memory_format check; the copy the layer
        builds is contiguous, which is the same silent layout change one level down."""

        class CloneDefault(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.clone.default(x)

        module = CloneDefault().eval().cuda()
        inputs = (
            torch.randn(3, 4, 5, 3, device="cuda").to(
                memory_format=torch.channels_last
            ),
        )
        compiled = self._compile(module, inputs)
        result = compiled(*inputs)

        self.assertTrue(
            result.is_contiguous(memory_format=torch.channels_last),
            f"expected channels-last strides, got {result.stride()}",
        )
        torch.testing.assert_close(result, module(*inputs))

    @parameterized.expand([("float64", torch.float64), ("uint8", torch.uint8)])
    def test_unbindable_input_dtype_falls_back(self, _, dtype):
        """float64 needs truncate_double, and without it the binding expects float32 and
        rejects the caller's tensor. uint8 aborts the build. Both run in PyTorch."""

        class OnlyClone(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.clone.default(x)

        module = OnlyClone().eval().cuda()
        if dtype == torch.uint8:
            inputs = (torch.randint(0, 255, (3, 5), dtype=dtype, device="cuda"),)
        else:
            inputs = (torch.randn(3, 5, dtype=dtype, device="cuda"),)

        compiled = self._compile(module, inputs)
        result = compiled(*inputs)

        self.assertEqual(self._engines(compiled), 0)
        self.assertEqual(result.dtype, dtype)
        torch.testing.assert_close(result, module(*inputs))

    def test_float64_converts_when_truncation_is_allowed(self):
        """With truncate_double the engine can bind it, so it should not fall back."""

        class OnlyClone(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.clone.default(x)

        module = OnlyClone().eval().cuda()
        inputs = (torch.randn(3, 5, dtype=torch.float64, device="cuda"),)
        compiled = self._compile(module, inputs, truncate_double=True)

        self.assertEqual(self._engines(compiled), 1)
        torch.testing.assert_close(
            compiled(*inputs).double(), module(*inputs), rtol=1e-3, atol=1e-3
        )


if __name__ == "__main__":
    run_tests()
