import copy
import unittest

import torch
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import TorchTensorRTModule
from torch_tensorrt.dynamo.runtime._TRTEngine import TRTEngine
from torch_tensorrt.runtime import optimization_profile

# Profiles are an ordered list; the list index is the optimization-profile
# index selected at runtime. Order is meaningful for lazy auto-selection: the
# decode profile ([1, 1]) and prefill profile ([1, 64]) overlap at seq=1, so we
# declare decode FIRST (index 0) to make it win the overlap (first-working).
DECODE_IDX = 0
PREFILL_IDX = 1


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(16, 32)
        self.l2 = torch.nn.Linear(32, 16)

    def forward(self, x):
        return self.l2(torch.relu(self.l1(x)))


def _make_profiles_input():
    return Input(
        name="x",
        dtype=torch.float16,
        profiles=[
            {"min": (4, 1, 16), "opt": (4, 1, 16), "max": (4, 1, 16)},  # decode
            {"min": (4, 1, 16), "opt": (4, 48, 16), "max": (4, 64, 16)},  # prefill
        ],
    )


def _compile_mlp(model, **kwargs):
    inp = _make_profiles_input()
    example = torch.randn(4, 48, 16, dtype=torch.float16, device="cuda")
    ep = torch.export.export(
        model,
        (example,),
        dynamic_shapes=({1: torch.export.Dim("seq", min=1, max=64)},),
    )
    return torch_tensorrt.dynamo.compile(
        ep,
        arg_inputs=[inp],
        min_block_size=1,
        enabled_precisions={torch.float16},
        **kwargs,
    )


class TestInputProfilesValidation(TestCase):
    def test_union_envelope(self):
        inp = _make_profiles_input()
        self.assertEqual(inp.shape["min_shape"], (4, 1, 16))
        self.assertEqual(inp.shape["max_shape"], (4, 64, 16))
        # opt is taken from the first declared profile (decode at index 0)
        self.assertEqual(inp.shape["opt_shape"], (4, 1, 16))
        self.assertEqual(len(inp.profiles), 2)
        self.assertEqual(inp.profiles[DECODE_IDX]["max"], (4, 1, 16))
        self.assertEqual(inp.profiles[PREFILL_IDX]["opt"], (4, 48, 16))

    def test_min_zero_rejected(self):
        with self.assertRaises(ValueError):
            Input(profiles=[{"min": (0, 1), "opt": (1, 1), "max": (2, 1)}])

    def test_min_opt_max_ordering(self):
        with self.assertRaises(ValueError):
            Input(profiles=[{"min": (4, 1), "opt": (4, 8), "max": (4, 2)}])

    def test_empty_profiles_rejected(self):
        with self.assertRaises(ValueError):
            Input(profiles=[])

    def test_mutual_exclusion(self):
        with self.assertRaises(ValueError):
            Input(
                shape=(1, 2),
                profiles=[{"min": (1,), "opt": (1,), "max": (1,)}],
            )

    def test_str_includes_profiles(self):
        inp = _make_profiles_input()
        self.assertIn("profiles=", str(inp))

    def test_profiles_with_shared_dims(self):
        # ``profiles`` and ``shared_dims`` compose: profiles set the per-profile
        # ranges while shared_dims names the dynamic axis (validated against the
        # union envelope) for cross-input symbol sharing.
        inp = Input(
            name="input_ids",
            profiles=[
                {"min": (1, 1), "opt": (1, 1), "max": (1, 1)},  # decode
                {"min": (1, 1), "opt": (1, 128), "max": (1, 512)},  # prefill
            ],
            shared_dims={1: "seq"},
        )
        self.assertEqual(len(inp.profiles), 2)
        self.assertEqual(inp.shared_dims, {1: "seq"})
        # union envelope marks the shared axis dynamic (1..512)
        self.assertEqual(inp.shape["min_shape"], (1, 1))
        self.assertEqual(inp.shape["max_shape"], (1, 512))

    def test_shared_dims_on_static_union_axis_rejected(self):
        # Axis 0 has min == max across every profile, so it is static in the
        # union; naming it for sharing is a user error.
        with self.assertRaises(ValueError):
            Input(
                profiles=[
                    {"min": (1, 1), "opt": (1, 8), "max": (1, 16)},
                    {"min": (1, 1), "opt": (1, 4), "max": (1, 32)},
                ],
                shared_dims={0: "batch"},
            )


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestMultiProfileRuntime(TestCase):
    """Runtime behavior of multi-profile engines.

    The runtime (C++ or Python ``TRTEngine``) is selected automatically by the
    Torch-TensorRT build, so these tests drive the runtime-agnostic engine API
    (``num_optimization_profiles`` / ``set_active_profile`` /
    ``_active_profile_index`` / ``_auto_select_profiles``) exposed identically by
    both runtimes via the ``optimization_profile`` context manager.
    """

    def setUp(self):
        self.model = MLP().eval().cuda().half()
        self.trt_gm = _compile_mlp(self.model)
        self.decode_in = torch.randn(4, 1, 16, dtype=torch.float16, device="cuda")
        self.prefill_in = torch.randn(4, 32, 16, dtype=torch.float16, device="cuda")

    def _trt_engines(self, module):
        return [
            m.engine for m in module.modules() if isinstance(m, TorchTensorRTModule)
        ]

    def test_two_profiles_built(self):
        engines = self._trt_engines(self.trt_gm)
        self.assertGreaterEqual(len(engines), 1)
        for e in engines:
            self.assertEqual(e.num_optimization_profiles, 2)

    def test_manual_pin_by_decode_index(self):
        ref = self.model(self.decode_in)
        with optimization_profile(self.trt_gm, DECODE_IDX):
            out = self.trt_gm(self.decode_in)
        self.assertEqual(tuple(out.shape), (4, 1, 16))
        self.assertTrue(torch.allclose(out, ref, atol=1e-2))

    def test_manual_pin_by_prefill_index(self):
        with optimization_profile(self.trt_gm, PREFILL_IDX):
            out = self.trt_gm(self.prefill_in)
        self.assertEqual(tuple(out.shape), (4, 32, 16))

    def test_auto_selection_decode_and_prefill(self):
        ref_d = self.model(self.decode_in)
        ref_p = self.model(self.prefill_in)
        with optimization_profile(self.trt_gm, "auto"):
            out_d = self.trt_gm(self.decode_in)
            out_p = self.trt_gm(self.prefill_in)
        self.assertTrue(torch.allclose(out_d, ref_d, atol=1e-2))
        self.assertTrue(torch.allclose(out_p, ref_p, atol=1e-2))

    def test_auto_selection_is_lazy_first_working(self):
        # decode (idx 0) and prefill (idx 1) both accept seq=1; lazy selection
        # must pick the first that fits (decode). seq=32 only fits prefill.
        engines = self._trt_engines(self.trt_gm)
        with optimization_profile(self.trt_gm, "auto"):
            self.trt_gm(self.decode_in)
            for e in engines:
                self.assertEqual(e._active_profile_index, DECODE_IDX)
            self.trt_gm(self.prefill_in)
            for e in engines:
                self.assertEqual(e._active_profile_index, PREFILL_IDX)

    def test_manual_pin_persists_across_calls(self):
        # A manual pin (no "auto") must stick: the engine keeps the pinned
        # profile across invocations instead of re-selecting per call.
        engines = self._trt_engines(self.trt_gm)
        with optimization_profile(self.trt_gm, PREFILL_IDX):
            self.trt_gm(self.prefill_in)
            for e in engines:
                self.assertEqual(e._active_profile_index, PREFILL_IDX)
                self.assertFalse(e._auto_select_profiles)
            self.trt_gm(self.prefill_in)
            for e in engines:
                self.assertEqual(e._active_profile_index, PREFILL_IDX)

    def test_profile_state_restored_after_context(self):
        # Leaving the context manager restores the engine's prior profile state.
        engines = self._trt_engines(self.trt_gm)
        before = [(e._auto_select_profiles, e._active_profile_index) for e in engines]
        with optimization_profile(self.trt_gm, PREFILL_IDX):
            self.trt_gm(self.prefill_in)
        after = [(e._auto_select_profiles, e._active_profile_index) for e in engines]
        self.assertEqual(before, after)

    def test_out_of_range_index_raises(self):
        with self.assertRaises(ValueError):
            with optimization_profile(self.trt_gm, 99):
                self.trt_gm(self.decode_in)

    def test_non_int_profile_raises(self):
        with self.assertRaises(TypeError):
            with optimization_profile(self.trt_gm, "decode"):
                self.trt_gm(self.decode_in)

    def test_serialization_round_trip_preserves_profiles(self):
        trt_gm2 = copy.deepcopy(self.trt_gm)
        for e in self._trt_engines(trt_gm2):
            self.assertEqual(e.num_optimization_profiles, 2)
        with optimization_profile(trt_gm2, DECODE_IDX):
            out = trt_gm2(self.decode_in)
        self.assertEqual(tuple(out.shape), (4, 1, 16))

    def _check_reconstructed_profile_state(self, engine):
        # A Python ``TRTEngine`` reconstructed with NO optimization-profile
        # metadata must rebuild its profile count and per-profile [min, max] dim
        # ranges purely from the TensorRT API (getNbOptimizationProfiles /
        # getProfileShape). This white-box check reads the Python runtime's
        # internal ``_profile_dynamic_dims`` / ``_auto_select_profile`` (not part
        # of the user-facing API; the C++ runtime is covered behaviorally below).
        self.assertEqual(engine.num_optimization_profiles, 2)

        dynamic_dims = engine._profile_dynamic_dims
        # Cache is indexed by input binding position; the model has a single
        # input ``x`` at position 0. Only the dynamic seq axis (dim 1) is cached;
        # the static batch/hidden dims are omitted. ``dynamic_dims[0]`` is a list
        # of (dim_index, per-profile ranges); index by dim to get the seq ranges.
        # tuple() normalizes the Python runtime (tuples) and C++ runtime (lists).
        seq_ranges = dict(dynamic_dims[0])[1]
        self.assertEqual(tuple(seq_ranges[DECODE_IDX]), (1, 1))
        self.assertEqual(tuple(seq_ranges[PREFILL_IDX]), (1, 64))

        # seq=1 -> decode (first profile that fits), seq=32 -> prefill.
        self.assertEqual(engine._auto_select_profile([self.decode_in]), DECODE_IDX)
        self.assertEqual(engine._auto_select_profile([self.prefill_in]), PREFILL_IDX)

    def test_profiles_restored_from_trt_api_without_metadata(self):
        # Python runtime: rebuild a fresh ``TRTEngine`` straight from the
        # serialized layout (engine bytes + binding names + device only),
        # simulating loading an engine that carries no profile metadata.
        src = self._trt_engines(self.trt_gm)[0]
        if not isinstance(src, TRTEngine):
            self.skipTest("Python TRTEngine-specific construction path")
        fresh = TRTEngine(list(src.serialized_info))
        self._check_reconstructed_profile_state(fresh)

    def test_profiles_restored_from_trt_api_without_metadata_cpp(self):
        # C++ runtime: a deep-copied module round-trips the engine through
        # serialize/deserialize, which is the "load with no profile metadata"
        # path; ``setup_optimization_profiles()`` (called from the C++ ctor)
        # rebuilds everything from the TRT API. Verified behaviorally through the
        # user-facing API only (profile count + shape-based auto-selection); the
        # internal dim-range introspection is intentionally not exposed to users.
        src = self._trt_engines(self.trt_gm)[0]
        if isinstance(src, TRTEngine):
            self.skipTest("C++ runtime-specific test")
        trt_gm2 = copy.deepcopy(self.trt_gm)
        engines = self._trt_engines(trt_gm2)
        for e in engines:
            self.assertEqual(e.num_optimization_profiles, 2)
        # Auto-selection relies on the dim ranges rebuilt from the TRT API: seq=1
        # must resolve to decode, seq>1 to prefill.
        with optimization_profile(trt_gm2, "auto"):
            trt_gm2(self.decode_in)
            for e in engines:
                self.assertEqual(e._active_profile_index, DECODE_IDX)
            trt_gm2(self.prefill_in)
            for e in engines:
                self.assertEqual(e._active_profile_index, PREFILL_IDX)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestMultiProfileSharedDims(TestCase):
    """``profiles`` combined with ``shared_dims``.

    Two inputs share a dynamic ``seq`` axis (one exported ``Dim``) while each
    declares the same prefill/decode profiles. The shared dim feeds the export
    envelope; the profiles feed the per-profile TRT ranges.
    """

    class TwoInput(torch.nn.Module):
        def forward(self, a, b):
            return torch.relu(a * b + a)

    def _shared_inputs(self):
        # prefill at index 0 so the export opt (taken from profile[0]) is > 1 and
        # the seq axis is not specialized to a constant during tracing.
        prof = [
            {"min": (1, 1), "opt": (1, 128), "max": (1, 512)},  # prefill
            {"min": (1, 1), "opt": (1, 1), "max": (1, 16)},  # decode
        ]
        a = Input(profiles=prof, shared_dims={1: "seq"}, dtype=torch.float32, name="a")
        b = Input(profiles=prof, shared_dims={1: "seq"}, dtype=torch.float32, name="b")
        return a, b

    def test_shared_axis_is_single_export_symbol(self):
        # Both inputs' dynamic seq axis must trace to the *same* export symbol.
        model = self.TwoInput().eval().cuda()
        a, b = self._shared_inputs()
        ep = torch_tensorrt.dynamo.trace(model, arg_inputs=[a, b])
        placeholders = [n for n in ep.graph.nodes if n.op == "placeholder"]
        symbols = []
        for ph in placeholders[:2]:
            val = ph.meta["val"]
            dim = val.shape[1]
            self.assertIsInstance(dim, torch.SymInt)
            symbols.append(dim.node.expr.name)
        self.assertEqual(symbols[0], symbols[1])

    def _trt_engines(self, module):
        return [
            m.engine for m in module.modules() if isinstance(m, TorchTensorRTModule)
        ]

    def test_compile_and_run_across_profiles(self):
        model = self.TwoInput().eval().cuda()
        a, b = self._shared_inputs()
        ep = torch_tensorrt.dynamo.trace(model, arg_inputs=[a, b])
        trt_gm = torch_tensorrt.dynamo.compile(
            ep,
            arg_inputs=[a, b],
            min_block_size=1,
            enabled_precisions={torch.float32},
        )
        for e in self._trt_engines(trt_gm):
            self.assertEqual(e.num_optimization_profiles, 2)

        # seq=1 fits the decode profile, seq=256 fits prefill.
        for seq in (1, 8, 256):
            x = torch.randn(1, seq, device="cuda")
            y = torch.randn(1, seq, device="cuda")
            with optimization_profile(trt_gm, "auto"):
                out = trt_gm(x, y)
            self.assertTrue(torch.allclose(out, model(x, y), atol=1e-2))


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestMultiProfileGraphBreak(TestCase):
    """Profile propagation across graph breaks (build-selected runtime)."""

    def _trt_engines(self, module):
        return [
            m.engine for m in module.modules() if isinstance(m, TorchTensorRTModule)
        ]

    def test_submodule_profile_propagation(self):
        class GraphBreakMLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = torch.nn.Linear(16, 32)
                self.l2 = torch.nn.Linear(32, 16)

            def forward(self, x):
                a = self.l1(x)
                b = torch.sin(a)  # forced to run in torch -> graph break
                return self.l2(torch.relu(b))

        model = GraphBreakMLP().eval().cuda().half()
        trt_gm = _compile_mlp(model, torch_executed_ops={"torch.ops.aten.sin.default"})

        engines = self._trt_engines(trt_gm)
        # Expect more than one TRT engine (graph break) and every engine has
        # the two profiles propagated.
        self.assertGreaterEqual(len(engines), 2)
        for e in engines:
            self.assertEqual(e.num_optimization_profiles, 2)

        decode_in = torch.randn(4, 1, 16, dtype=torch.float16, device="cuda")
        prefill_in = torch.randn(4, 32, 16, dtype=torch.float16, device="cuda")
        with optimization_profile(trt_gm, "auto"):
            out_d = trt_gm(decode_in)
            out_p = trt_gm(prefill_in)
        self.assertTrue(torch.allclose(out_d, model(decode_in), atol=1e-2))
        self.assertTrue(torch.allclose(out_p, model(prefill_in), atol=1e-2))

    def test_reshaped_dynamic_submodule_input(self):
        # The reshape makes the post-graph-break submodule input dim a *derived*
        # symbolic expression of the source seq symbol (16 * seq), not the source
        # symbol itself. This exercises per-profile bound evaluation of derived
        # dynamic dims when propagating profiles to intermediate submodules.
        class ReshapeGraphBreakMLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(4, 32)
                self.fc2 = torch.nn.Linear(32, 32)

            def forward(self, x):  # x: (4, seq, 16)
                x = self.fc1(x.reshape(-1, 4))  # (16 * seq, 32): derived dynamic dim
                x = torch.relu(x)  # forced to run in torch -> graph break
                return self.fc2(x)

        model = ReshapeGraphBreakMLP().eval().cuda().half()
        trt_gm = _compile_mlp(model, torch_executed_ops={"torch.ops.aten.relu.default"})

        engines = self._trt_engines(trt_gm)
        # Graph break -> multiple TRT engines; every engine (including the one
        # fed the reshaped derived-dynamic tensor) carries both profiles.
        self.assertGreaterEqual(len(engines), 2)
        for e in engines:
            self.assertEqual(e.num_optimization_profiles, 2)

        decode_in = torch.randn(4, 1, 16, dtype=torch.float16, device="cuda")
        prefill_in = torch.randn(4, 32, 16, dtype=torch.float16, device="cuda")
        with optimization_profile(trt_gm, "auto"):
            out_d = trt_gm(decode_in)
            # seq=1 -> derived dim 16*1=16; both engines auto-select decode
            for e in engines:
                self.assertEqual(e._active_profile_index, DECODE_IDX)
            out_p = trt_gm(prefill_in)
            # seq=32 -> derived dim 16*32=512; both engines auto-select prefill
            for e in engines:
                self.assertEqual(e._active_profile_index, PREFILL_IDX)
        self.assertTrue(torch.allclose(out_d, model(decode_in), atol=1e-2))
        self.assertTrue(torch.allclose(out_p, model(prefill_in), atol=1e-2))


if __name__ == "__main__":
    run_tests()
