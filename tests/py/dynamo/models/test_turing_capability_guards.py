import unittest
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_tensorrt as torchtrt
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion._TRTInterpreter import (
    TRTInterpreter,
    set_rtx_compute_capabilities,
)

TURING = (7, 5)


def _is_turing() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == TURING


def _trt_submodule_count(compiled) -> int:
    """Number of TensorRT engine submodules in a compiled module.

    Matches on class name so the count is independent of which runtime variant
    (C++, CUDA-graphs, ...) was selected.
    """
    return sum(
        1 for _, m in compiled.named_modules() if "TensorRTModule" in type(m).__name__
    )


class MatMul(nn.Module):
    def forward(self, a, b):
        return torch.matmul(a, b)


class Addmm(nn.Module):
    """aten.addmm.default: the fused bias-add form of a GEMM, guarded on the same target."""

    def forward(self, inp, mat1, mat2):
        return torch.ops.aten.addmm.default(inp, mat1, mat2)


class Conv3d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(4, 8, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


class ConvTranspose3d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.ConvTranspose3d(4, 8, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


class Conv2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 8, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


class PaddedConv3d(nn.Module):
    """A conv3d the 3D-conv guard never sees as a convolution.

    fuse_pad_into_convolution folds the pad into the conv and rewrites the pair as
    tensorrt::conv_asym_pad, erasing the aten.convolution node. It fires for any
    zero-fill, non-negative pad -- asymmetry is not required.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(4, 8, 3, padding=0)

    def forward(self, x):
        return self.conv(F.pad(x, (1, 1, 1, 1, 2, 0)))


class PaddedConv2d(nn.Module):
    """Same fusion, 2D -- which Turing does support."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 8, 3, padding=0)

    def forward(self, x):
        return self.conv(F.pad(x, (0, 1, 0, 1)))


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "Turing capability guards only apply to TensorRT-RTX",
)
class TestTuringCapabilityGuards(TestCase):
    """TensorRT-RTX cannot serve some ops on Turing (SM 7.5); they must fall back.

    Per the TensorRT-RTX support matrix, Turing does not support FP32 GEMMs or 3D
    convolutions, and it has no bfloat16 hardware. Without a fallback these produce a
    null execution context, a segfault, or -- for dynamic-shape FP32 GEMMs -- a
    silently all-zero result.
    """

    def _compile(self, mod, inputs, **kwargs):
        return torchtrt.compile(
            mod.eval().cuda(),
            ir="dynamo",
            inputs=list(inputs),
            min_block_size=1,
            cache_built_engines=False,
            reuse_cached_engines=False,
            use_python_runtime=True,
            **kwargs,
        )

    def _assert_matches_eager(self, mod, compiled, inputs):
        with torch.no_grad():
            ref = mod.eval().cuda()(*inputs)
            out = compiled(*inputs)
        self.assertFalse(
            bool(torch.all(out == 0).item()) and not bool(torch.all(ref == 0).item()),
            "output is all zeros while eager is not -- silent corruption",
        )
        cos = F.cosine_similarity(
            ref.flatten().unsqueeze(0).float(), out.flatten().unsqueeze(0).float()
        )
        self.assertGreater(cos.item(), 0.99)

    # -- declared-target tests: these run on ANY GPU ------------------------------
    # Declaring Turing as a build target must produce Turing's partitioning even on
    # non-Turing hardware. This is what makes the guards testable without a Turing GPU.

    def test_declared_turing_target_falls_back_fp32_gemm(self):
        mod = MatMul()
        inputs = (torch.randn(4, 8).cuda(), torch.randn(8, 16).cuda())
        compiled = self._compile(mod, inputs, target_compute_capabilities=[TURING])
        self.assertEqual(_trt_submodule_count(compiled), 0)
        self._assert_matches_eager(mod, compiled, inputs)

    def test_declared_turing_target_falls_back_fp32_addmm(self):
        # addmm survives lowering as its own node, so the GEMM guard sees it directly.
        mod = Addmm()
        inputs = (
            torch.randn(4, 6).cuda(),
            torch.randn(4, 5).cuda(),
            torch.randn(5, 6).cuda(),
        )
        compiled = self._compile(mod, inputs, target_compute_capabilities=[TURING])
        self.assertEqual(_trt_submodule_count(compiled), 0)
        self._assert_matches_eager(mod, compiled, inputs)

    def test_declared_turing_target_falls_back_conv3d(self):
        mod = Conv3d()
        inputs = (torch.randn(1, 4, 8, 8, 8).cuda(),)
        compiled = self._compile(mod, inputs, target_compute_capabilities=[TURING])
        self.assertEqual(_trt_submodule_count(compiled), 0)
        self._assert_matches_eager(mod, compiled, inputs)

    def test_declared_turing_target_keeps_fp16_gemm_on_trt(self):
        mod = MatMul()
        inputs = (
            torch.randn(4, 8, dtype=torch.half).cuda(),
            torch.randn(8, 16, dtype=torch.half).cuda(),
        )
        compiled = self._compile(mod, inputs, target_compute_capabilities=[TURING])
        self.assertGreater(_trt_submodule_count(compiled), 0)
        self._assert_matches_eager(mod, compiled, inputs)

    def test_declared_turing_target_keeps_conv2d_on_trt(self):
        mod = Conv2d()
        inputs = (torch.randn(1, 4, 16, 16).cuda(),)
        compiled = self._compile(mod, inputs, target_compute_capabilities=[TURING])
        self.assertGreater(_trt_submodule_count(compiled), 0)

    def test_declared_turing_target_falls_back_padded_conv3d(self):
        # The fused op needs its own guard: by partitioning time the aten.convolution
        # node the conv guard keys on no longer exists.
        mod = PaddedConv3d()
        inputs = (torch.randn(1, 4, 8, 8, 8).cuda(),)
        compiled = self._compile(mod, inputs, target_compute_capabilities=[TURING])
        self.assertEqual(_trt_submodule_count(compiled), 0)
        self._assert_matches_eager(mod, compiled, inputs)

    def test_declared_turing_target_keeps_padded_conv2d_on_trt(self):
        # Only rank 3 is rejected; the fused op must keep serving 2D.
        mod = PaddedConv2d()
        inputs = (torch.randn(1, 4, 16, 16).cuda(),)
        compiled = self._compile(mod, inputs, target_compute_capabilities=[TURING])
        self.assertGreater(_trt_submodule_count(compiled), 0)
        self._assert_matches_eager(mod, compiled, inputs)

    def test_declared_turing_target_keeps_transposed_conv3d_on_trt(self):
        # Transposed 3D convolution is a distinct layer and does work on Turing.
        mod = ConvTranspose3d()
        inputs = (torch.randn(1, 4, 8, 8, 8).cuda(),)
        compiled = self._compile(mod, inputs, target_compute_capabilities=[TURING])
        self.assertGreater(_trt_submodule_count(compiled), 0)

    @unittest.skipIf(
        _is_turing(), "on Turing the native path is already guarded; see the SM75 tests"
    )
    def test_non_turing_default_keeps_fp32_gemm_on_trt(self):
        # Guards must not fire when Turing is not targeted.
        mod = MatMul()
        inputs = (torch.randn(4, 8).cuda(), torch.randn(8, 16).cuda())
        compiled = self._compile(mod, inputs)
        self.assertGreater(_trt_submodule_count(compiled), 0)

    def test_target_compute_capabilities_is_engine_invariant(self):
        # A cached engine built for one target set must never be reused for another.
        from torch_tensorrt.dynamo._settings import _SETTINGS_TO_BE_ENGINE_INVARIANT

        self.assertIn("target_compute_capabilities", _SETTINGS_TO_BE_ENGINE_INVARIANT)

    # -- engine targeting resolves the same setting partitioning does --------------
    # Asserting on the builder config rather than on compile success is what makes these
    # runnable off Turing, where an unset capability drifts silently instead of failing.

    def _compute_capability_counts(self, mod, inputs, **kwargs):
        """num_compute_capabilities of every builder config this compile produced."""
        counts = []
        original = TRTInterpreter._populate_trt_builder_config

        def spy(interpreter, *args, **kwargs_):
            builder_config = original(interpreter, *args, **kwargs_)
            counts.append(builder_config.num_compute_capabilities)
            return builder_config

        with mock.patch.object(TRTInterpreter, "_populate_trt_builder_config", spy):
            self._compile(mod, inputs, **kwargs)
        self.assertTrue(counts, "no engine was built, so nothing was asserted")
        return counts

    def test_refittable_build_targets_a_compute_capability_by_default(self):
        # Refittable builds are the ones that fail when the capability is left unset.
        counts = self._compute_capability_counts(
            Conv2d(), (torch.randn(1, 4, 8, 8).cuda(),), immutable_weights=False
        )
        self.assertTrue(all(count == 1 for count in counts), counts)

    def test_declared_target_sets_a_capability_per_declared_target(self):
        targets = [TURING]
        counts = self._compute_capability_counts(
            Conv2d(),
            (torch.randn(1, 4, 8, 8).cuda(),),
            immutable_weights=False,
            target_compute_capabilities=targets,
        )
        self.assertTrue(all(count == len(targets) for count in counts), counts)


class _FakeBuilderConfig:
    """The two members set_rtx_compute_capabilities touches, and nothing else."""

    def __init__(self, accept=True):
        self.num_compute_capabilities = 0
        self.set = []
        self._accept = accept

    def set_compute_capability(self, compute_capability, idx):
        self.set.append((compute_capability, idx))
        return self._accept


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "set_rtx_compute_capabilities is TensorRT-RTX only",
)
class TestSetRtxComputeCapabilities(TestCase):
    """Unit tests for the builder-config helper -- no GPU, no engine build."""

    def setUp(self):
        # Imported here, not at module scope: on TensorRT-RTX the ``tensorrt`` alias is
        # only registered once torch_tensorrt has been imported.
        import tensorrt

        self.trt = tensorrt

    def test_undeclared_targets_the_current_device(self):
        config = _FakeBuilderConfig()
        set_rtx_compute_capabilities(config, None)
        self.assertEqual(config.num_compute_capabilities, 1)
        self.assertEqual(config.set, [(self.trt.ComputeCapability.CURRENT, 0)])

    def test_empty_list_is_treated_as_undeclared(self):
        config = _FakeBuilderConfig()
        set_rtx_compute_capabilities(config, [])
        self.assertEqual(config.set, [(self.trt.ComputeCapability.CURRENT, 0)])

    def test_declared_targets_are_set_by_name_in_order(self):
        config = _FakeBuilderConfig()
        set_rtx_compute_capabilities(config, [TURING, (8, 9)])
        self.assertEqual(config.num_compute_capabilities, 2)
        self.assertEqual(
            config.set,
            [
                (self.trt.ComputeCapability.SM75, 0),
                (self.trt.ComputeCapability.SM89, 1),
            ],
        )

    def test_unknown_target_raises(self):
        with self.assertRaisesRegex(ValueError, "SM99"):
            set_rtx_compute_capabilities(_FakeBuilderConfig(), [(9, 9)])

    def test_a_refused_capability_raises(self):
        with self.assertRaisesRegex(RuntimeError, "SM75"):
            set_rtx_compute_capabilities(_FakeBuilderConfig(accept=False), [TURING])


@unittest.skipIf(
    ENABLED_FEATURES.tensorrt_rtx,
    "the rejection only applies to standard TensorRT",
)
class TestTargetComputeCapabilitiesRejectedOffRtx(TestCase):
    def test_standard_tensorrt_rejects_declared_targets(self):
        with self.assertRaisesRegex(ValueError, "only supported on TensorRT-RTX"):
            CompilationSettings(target_compute_capabilities=[TURING])

    def test_standard_tensorrt_accepts_the_default(self):
        self.assertIsNone(CompilationSettings().target_compute_capabilities)


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx or not _is_turing(),
    "requires TensorRT-RTX on a Turing (SM 7.5) device",
)
class TestTuringNativeFallback(TestCase):
    """Same guards, exercised natively on Turing hardware."""

    def _compile(self, mod, inputs, **kwargs):
        return torchtrt.compile(
            mod.eval().cuda(),
            ir="dynamo",
            inputs=list(inputs),
            min_block_size=1,
            cache_built_engines=False,
            reuse_cached_engines=False,
            use_python_runtime=True,
            **kwargs,
        )

    def test_fp32_gemm_static_falls_back(self):
        mod = MatMul()
        inputs = (torch.randn(4, 8).cuda(), torch.randn(8, 16).cuda())
        compiled = self._compile(mod, inputs)
        self.assertEqual(_trt_submodule_count(compiled), 0)

    def test_fp32_gemm_dynamic_is_not_silently_zero(self):
        # Unguarded, this returned an all-zero tensor of the right shape and dtype.
        mod = MatMul().eval().cuda()
        a = torch.randn(4, 8).cuda()
        b = torch.randn(8, 16).cuda()
        with torch.no_grad():
            ref = mod(a, b)
        dim = torch.export.Dim("m", min=1, max=64)
        ep = torch.export.export(mod, (a, b), dynamic_shapes=({0: dim}, {}))
        compiled = torchtrt.compile(
            ep,
            arg_inputs=[a, b],
            ir="dynamo",
            min_block_size=1,
            cache_built_engines=False,
            reuse_cached_engines=False,
            use_python_runtime=True,
        )
        with torch.no_grad():
            out = compiled(a, b)
        self.assertFalse(bool(torch.all(out == 0).item()))
        cos = F.cosine_similarity(
            ref.flatten().unsqueeze(0), out.flatten().unsqueeze(0)
        )
        self.assertGreater(cos.item(), 0.99)

    def test_conv3d_falls_back(self):
        mod = Conv3d()
        inputs = (torch.randn(1, 4, 8, 8, 8).cuda(),)
        compiled = self._compile(mod, inputs)
        self.assertEqual(_trt_submodule_count(compiled), 0)

    def test_bfloat16_falls_back_without_crashing(self):
        # Unguarded, compiling bfloat16 for SM 7.5 segfaulted the process.
        class Add(nn.Module):
            def forward(self, x):
                return x + 1.0

        mod = Add()
        inputs = (torch.randn(4, 8, dtype=torch.bfloat16).cuda(),)
        compiled = self._compile(mod, inputs, enabled_precisions={torch.bfloat16})
        self.assertEqual(_trt_submodule_count(compiled), 0)
        with torch.no_grad():
            out = compiled(*inputs)
        self.assertEqual(out.dtype, torch.bfloat16)


if __name__ == "__main__":
    run_tests()
