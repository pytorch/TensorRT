import torch
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase, run_tests


class TestRandomOperatorsFallBack(TestCase):
    """rand, randn and randperm used to have converters that called numpy during engine
    construction and froze one sample into the engine, so a compiled model returned the
    same values on every call. They now fall back to PyTorch like the rest of the family,
    which is the correct behaviour for an operator TensorRT cannot express.
    """

    @staticmethod
    def _module_with(random_call):
        class WithRandom(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [torch.nn.Linear(64, 64) for _ in range(6)]
                )

            def forward(self, x):
                out = x
                for layer in self.layers:
                    out = torch.relu(layer(out))
                return out + random_call(out)

        return WithRandom().eval().cuda()

    def _assert_not_frozen(self, module, inputs):
        compiled = torch_tensorrt.dynamo.compile(
            torch.export.export(module, tuple(inputs)),
            inputs=list(inputs),
            enabled_precisions={torch.float32},
            truncate_double=True,
        )
        first = compiled(*inputs).clone()
        second = compiled(*inputs).clone()
        self.assertFalse(
            torch.equal(first, second),
            "a compiled model containing a random operator returned identical values "
            "on two calls, so the values were frozen into the engine",
        )

    def test_randn_is_not_frozen(self):
        inputs = [torch.randn(8, 64, device="cuda")]
        module = self._module_with(
            lambda out: torch.randn(out.shape, device=out.device, dtype=out.dtype)
        )
        self._assert_not_frozen(module, inputs)

    def test_rand_is_not_frozen(self):
        inputs = [torch.randn(8, 64, device="cuda")]
        module = self._module_with(
            lambda out: torch.rand(out.shape, device=out.device, dtype=out.dtype)
        )
        self._assert_not_frozen(module, inputs)

    def test_manual_seed_is_honored(self):
        """The frozen converters ignored torch.manual_seed, because the values came from
        numpy's global state at build time."""
        inputs = [torch.randn(8, 64, device="cuda")]
        module = self._module_with(
            lambda out: torch.randn(out.shape, device=out.device, dtype=out.dtype)
        )
        compiled = torch_tensorrt.dynamo.compile(
            torch.export.export(module, tuple(inputs)),
            inputs=list(inputs),
            enabled_precisions={torch.float32},
            truncate_double=True,
        )
        torch.manual_seed(0)
        first = compiled(*inputs).clone()
        torch.manual_seed(0)
        second = compiled(*inputs).clone()
        torch.testing.assert_close(first, second)

    def test_randperm_is_a_permutation(self):
        """randperm's correctness is exact, so this needs no tolerance: the output has to be
        a rearrangement of 0..n-1. The converter that was removed here returned int32 where
        PyTorch returns int64, which is why the test it replaces compared shape only."""

        class WithRandperm(torch.nn.Module):
            def forward(self, x):
                return torch.randperm(16, device=x.device) + (x.sum() * 0).long()

        inputs = [torch.randn(8, 64, device="cuda")]
        module = WithRandperm().eval().cuda()
        compiled = torch_tensorrt.dynamo.compile(
            torch.export.export(module, tuple(inputs)),
            inputs=list(inputs),
            min_block_size=1,
            enabled_precisions={torch.float32},
            truncate_double=True,
        )
        out = compiled(*inputs)
        self.assertEqual(out.dtype, module(*inputs).dtype)
        torch.testing.assert_close(
            out.sort().values, torch.arange(16, device=out.device, dtype=out.dtype)
        )

    def test_random_operators_are_not_registered(self):
        """The direct check, and the only one that fails without this change.

        Whether a graph ends up with a frozen value depends on the partitioner, which on
        some stacks declines the surrounding block for unrelated reasons, so an end to end
        test can pass either way. What this change actually does is take these three
        operators out of the registry, so assert that.
        """
        from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
            DYNAMO_CONVERTERS,
        )

        for target in (
            torch.ops.aten.rand.default,
            torch.ops.aten.randn.default,
            torch.ops.aten.randperm.default,
        ):
            self.assertNotIn(
                target,
                DYNAMO_CONVERTERS,
                f"{target} is still registered, so it can be folded into an engine at "
                "build time instead of running fresh on every call",
            )


if __name__ == "__main__":
    run_tests()
