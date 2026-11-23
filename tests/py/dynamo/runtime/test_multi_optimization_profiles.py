# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import unittest

import torch
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import TorchTensorRTModule
from torch_tensorrt.dynamo.runtime._TRTEngine import TRTEngine
from torch_tensorrt.runtime import optimization_profile

import tensorrt as trt  # isort: skip  # imported after torch_tensorrt for RTX alias

# Profiles are an ordered list; the list index is the optimization-profile
# index selected at runtime. Order matters on a rescan: the decode profile
# ([1, 1]) and prefill profile ([1, 64]) overlap at seq=1, and a rescan takes the
# lowest fitting index, so declaring decode FIRST (index 0) lets it win that
# overlap. Auto-selection is sticky, though, so it only rescans once the active
# profile stops fitting -- prefill accepts seq=1 and would keep it.
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
            {
                "min_shape": (4, 1, 16),
                "opt_shape": (4, 1, 16),
                "max_shape": (4, 1, 16),
            },  # decode
            {
                "min_shape": (4, 1, 16),
                "opt_shape": (4, 48, 16),
                "max_shape": (4, 64, 16),
            },  # prefill
        ],
    )


def _compile_mlp_single_profile(model, **kwargs):
    """Compile with a static input, so the engine has exactly one profile."""
    example = torch.randn(4, 1, 16, dtype=torch.float16, device="cuda")
    ep = torch.export.export(model, (example,))
    return torch_tensorrt.dynamo.compile(
        ep,
        arg_inputs=[example],
        min_block_size=1,
        enabled_precisions={torch.float16},
        **kwargs,
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
        self.assertEqual(inp.profiles[DECODE_IDX]["max_shape"], (4, 1, 16))
        self.assertEqual(inp.profiles[PREFILL_IDX]["opt_shape"], (4, 48, 16))

    def test_min_zero_rejected(self):
        with self.assertRaises(ValueError):
            Input(
                profiles=[
                    {"min_shape": (0, 1), "opt_shape": (1, 1), "max_shape": (2, 1)}
                ]
            )

    def test_min_opt_max_ordering(self):
        with self.assertRaises(ValueError):
            Input(
                profiles=[
                    {"min_shape": (4, 1), "opt_shape": (4, 8), "max_shape": (4, 2)}
                ]
            )

    def test_empty_profiles_rejected(self):
        with self.assertRaises(ValueError):
            Input(profiles=[])

    def test_mutual_exclusion(self):
        with self.assertRaises(ValueError):
            Input(
                shape=(1, 2),
                profiles=[{"min_shape": (1,), "opt_shape": (1,), "max_shape": (1,)}],
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
                {
                    "min_shape": (1, 1),
                    "opt_shape": (1, 1),
                    "max_shape": (1, 1),
                },  # decode
                {
                    "min_shape": (1, 1),
                    "opt_shape": (1, 128),
                    "max_shape": (1, 512),
                },  # prefill
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
                    {
                        "min_shape": (1, 1),
                        "opt_shape": (1, 8),
                        "max_shape": (1, 16),
                    },
                    {
                        "min_shape": (1, 1),
                        "opt_shape": (1, 4),
                        "max_shape": (1, 32),
                    },
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

    def test_infer_outputs_accepts_a_shape_of_another_profile(self):
        # seq=32 fits only the prefill profile, which a run in auto mode switches
        # to. infer_outputs must accept it while decode is still active. A build
        # with the C++ runtime backs the module with the C++ engine, so also drive
        # the Python engine from the same serialized info.
        (module,) = [
            m for m in self.trt_gm.modules() if isinstance(m, TorchTensorRTModule)
        ]
        python_engine = TRTEngine(module._pack_engine_info())
        python_engine._auto_select_profiles = True
        with optimization_profile(self.trt_gm, "auto"):
            for engine in (module.engine, python_engine):
                self.assertEqual(engine._active_profile_index, DECODE_IDX)
                (output,) = engine.infer_outputs([list(self.prefill_in.shape)])
                self.assertEqual(list(getattr(output, "shape", output)), [4, 32, 16])

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

    def test_engine_level_out_of_range_index_raises(self):
        # Driving the engine directly skips optimization_profile()'s validation,
        # so the engine has to reject an index it cannot honor rather than warn
        # and keep running on whatever profile is loaded. These engines have two
        # profiles, so there is no single-profile tolerance to fall back on.
        # Only the exception type differs by runtime: ValueError from the Python
        # engine, RuntimeError from the C++ TORCHTRT_CHECK.
        for e in self._trt_engines(self.trt_gm):
            with self.assertRaises((ValueError, RuntimeError)):
                e.set_active_profile(e.num_optimization_profiles)
            with self.assertRaises((ValueError, RuntimeError)):
                e.set_active_profile(-1)

    def test_single_profile_engine_tolerates_a_nonzero_pin(self):
        # A pin reaches every engine in the module, so an index meant for a
        # multi-profile sibling lands on single-profile engines too. Those have
        # profile 0 and nothing to switch to: they must warn and stay on 0 rather
        # than fail an execution that was never about them (the C++ and Python
        # runtimes and the ExecuTorch backend all make this same exception).
        # Only reachable by driving the engine directly: set_optimization_profile
        # range-checks the index against this engine's profile count first, which
        # the tail of this test pins down.
        trt_gm = _compile_mlp_single_profile(self.model)
        static_in = torch.randn(4, 1, 16, dtype=torch.float16, device="cuda")
        engines = self._trt_engines(trt_gm)
        self.assertGreaterEqual(len(engines), 1)

        for e in engines:
            self.assertEqual(e.num_optimization_profiles, 1)
            if isinstance(e, TRTEngine):
                # Python runtime: the warning is a Python log record. The C++
                # runtime logs through LOG_WARNING, which assertLogs cannot see.
                with self.assertLogs("torch_tensorrt", level="WARNING") as logs:
                    e.set_active_profile(1)
                self.assertTrue(
                    any("Ignoring optimization profile" in m for m in logs.output)
                )
            else:
                e.set_active_profile(1)
            self.assertEqual(e._active_profile_index, 0)
            # Only a positive pin is tolerated; a negative one is still an error.
            with self.assertRaises((ValueError, RuntimeError)):
                e.set_active_profile(-1)

        # Tolerating the pin must leave the engine runnable, on profile 0.
        ref = self.model(static_in)
        out = trt_gm(static_in)
        self.assertTrue(torch.allclose(out, ref, atol=1e-2))
        for e in self._trt_engines(trt_gm):
            self.assertEqual(e._active_profile_index, 0)

        # The tolerance is engine-level only: the public entry point still
        # range-checks, so a user pin of 1 here is a mistake, not a no-op.
        with self.assertRaises(ValueError):
            with optimization_profile(trt_gm, 1):
                trt_gm(static_in)

    def test_non_int_profile_raises(self):
        with self.assertRaises(TypeError):
            with optimization_profile(self.trt_gm, "decode"):
                self.trt_gm(self.decode_in)

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
            {
                "min_shape": (1, 1),
                "opt_shape": (1, 128),
                "max_shape": (1, 512),
            },  # prefill
            {"min_shape": (1, 1), "opt_shape": (1, 1), "max_shape": (1, 16)},  # decode
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


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestProfilesWholeGraph(TestCase):
    """require_full_compilation converts a fully supported graph as one engine
    without partitioning."""

    def setUp(self):
        self.model = MLP().eval().cuda().half()
        example = torch.randn(4, 48, 16, dtype=torch.float16, device="cuda")
        self.ep = torch.export.export(
            self.model,
            (example,),
            dynamic_shapes=({1: torch.export.Dim("seq", min=1, max=64)},),
        )

    def _compile(self, profiles):
        trt_mod = torch_tensorrt.dynamo.compile(
            self.ep,
            arg_inputs=[Input(dtype=torch.float16, profiles=profiles)],
            min_block_size=1,
            enabled_precisions={torch.float16},
            require_full_compilation=True,
        )
        self.assertIsInstance(trt_mod, TorchTensorRTModule)
        engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(
            bytes(trt_mod.serialized_engine)
        )
        return trt_mod, engine

    def test_single_profile_sets_engine_profile(self):
        profile = {
            "min_shape": (4, 1, 16),
            "opt_shape": (4, 1, 16),
            "max_shape": (4, 64, 16),
        }
        trt_mod, engine = self._compile([profile])
        self.assertEqual(engine.num_optimization_profiles, 1)
        self.assertEqual(
            [tuple(s) for s in engine.get_tensor_profile_shape("x", 0)],
            [profile["min_shape"], profile["opt_shape"], profile["max_shape"]],
        )
        decode_in = torch.randn(4, 1, 16, dtype=torch.float16, device="cuda")
        self.assertTrue(
            torch.allclose(trt_mod(decode_in), self.model(decode_in), atol=1e-2)
        )

    def test_several_profiles_set_engine_profiles(self):
        decode = {
            "min_shape": (4, 1, 16),
            "opt_shape": (4, 1, 16),
            "max_shape": (4, 8, 16),
        }
        prefill = {
            "min_shape": (4, 16, 16),
            "opt_shape": (4, 32, 16),
            "max_shape": (4, 64, 16),
        }
        trt_mod, engine = self._compile([decode, prefill])
        self.assertEqual(engine.num_optimization_profiles, 2)
        for idx, profile in enumerate((decode, prefill)):
            self.assertEqual(
                [tuple(s) for s in engine.get_tensor_profile_shape("x", idx)],
                [profile["min_shape"], profile["opt_shape"], profile["max_shape"]],
            )
            inp = torch.randn(profile["opt_shape"], dtype=torch.float16, device="cuda")
            with optimization_profile(trt_mod, idx):
                self.assertTrue(
                    torch.allclose(trt_mod(inp), self.model(inp), atol=1e-2)
                )

    def _check_both_paths(self, ep, arg_inputs, kwarg_inputs, args, kwargs, model):
        expected = model(*args, **kwargs)
        flat_args = torch.utils._pytree.tree_leaves((args, kwargs))
        for require_full_compilation in (True, False):
            with self.subTest(require_full_compilation=require_full_compilation):
                trt_mod = torch_tensorrt.dynamo.compile(
                    ep,
                    arg_inputs=arg_inputs,
                    kwarg_inputs=kwarg_inputs,
                    min_block_size=1,
                    enabled_precisions={torch.float16},
                    require_full_compilation=require_full_compilation,
                )
                if require_full_compilation:
                    # The single engine module takes the flat inputs in export order.
                    self.assertIsInstance(trt_mod, TorchTensorRTModule)
                    out = trt_mod(*flat_args)
                else:
                    self.assertNotIsInstance(trt_mod, TorchTensorRTModule)
                    out = trt_mod(*args, **kwargs)
                self.assertTrue(torch.allclose(out, expected, atol=1e-2))

    def test_profiles_follow_kwarg_names(self):
        class TwoMLPs(torch.nn.Module):
            def __init__(self, mlp):
                super().__init__()
                self.mlp = mlp

            def forward(self, a, b):
                return self.mlp(a).mean(1) + self.mlp(b).mean(1)

        model = TwoMLPs(self.model)
        a = torch.randn(4, 32, 16, dtype=torch.float16, device="cuda")
        b = torch.randn(4, 4, 16, dtype=torch.float16, device="cuda")
        ep = torch.export.export(
            model,
            (),
            {"a": a, "b": b},
            dynamic_shapes={
                "a": {1: torch.export.Dim("sa", min=1, max=64)},
                "b": {1: torch.export.Dim("sb", min=1, max=64)},
            },
        )
        a_profile = {
            "min_shape": (4, 16, 16),
            "opt_shape": (4, 32, 16),
            "max_shape": (4, 64, 16),
        }
        b_profile = {
            "min_shape": (4, 2, 16),
            "opt_shape": (4, 4, 16),
            "max_shape": (4, 8, 16),
        }
        # Passed in a different order than at export.
        kwarg_inputs = {
            "b": Input(dtype=torch.float16, profiles=[b_profile]),
            "a": Input(dtype=torch.float16, profiles=[a_profile]),
        }
        self._check_both_paths(ep, [], kwarg_inputs, (), {"a": a, "b": b}, model)

    def test_profiles_follow_inputs_after_a_tuple(self):
        class PairThenY(torch.nn.Module):
            def __init__(self, mlp):
                super().__init__()
                self.mlp = mlp

            def forward(self, pair, y):
                p, q = pair
                return self.mlp(p).mean(1) + self.mlp(q).mean(1) + self.mlp(y).mean(1)

        model = PairThenY(self.model)
        p = torch.randn(4, 4, 16, dtype=torch.float16, device="cuda")
        q = torch.randn(4, 4, 16, dtype=torch.float16, device="cuda")
        y = torch.randn(4, 32, 16, dtype=torch.float16, device="cuda")
        ep = torch.export.export(
            model,
            ((p, q), y),
            dynamic_shapes=(
                (
                    {1: torch.export.Dim("sp", min=1, max=64)},
                    {1: torch.export.Dim("sq", min=1, max=64)},
                ),
                {1: torch.export.Dim("sy", min=1, max=64)},
            ),
        )
        plain = Input(
            min_shape=(4, 1, 16),
            opt_shape=(4, 4, 16),
            max_shape=(4, 64, 16),
            dtype=torch.float16,
        )
        y_profile = {
            "min_shape": (4, 16, 16),
            "opt_shape": (4, 32, 16),
            "max_shape": (4, 64, 16),
        }
        arg_inputs = [(plain, plain), Input(dtype=torch.float16, profiles=[y_profile])]
        self._check_both_paths(ep, arg_inputs, None, ((p, q), y), {}, model)

    def test_profiles_follow_kwarg_names_with_a_mutated_buffer(self):
        class CountingMLP(torch.nn.Module):
            def __init__(self, mlp):
                super().__init__()
                self.mlp = mlp
                self.register_buffer("calls", torch.zeros(1, device="cuda"))

            def forward(self, a, b):
                self.calls.add_(1)
                return self.mlp(a).sum(1) + self.mlp(b).sum(1)

        model = CountingMLP(self.model)
        a = torch.randn(4, 32, 16, dtype=torch.float16, device="cuda")
        b = torch.randn(4, 4, 16, dtype=torch.float16, device="cuda")
        ep = torch.export.export(
            model,
            (),
            {"a": a, "b": b},
            dynamic_shapes={
                "a": {1: torch.export.Dim("sa", min=1, max=64)},
                "b": {1: torch.export.Dim("sb", min=1, max=64)},
            },
        )
        decode = {
            "min_shape": (4, 1, 16),
            "opt_shape": (4, 1, 16),
            "max_shape": (4, 8, 16),
        }
        prefill = {
            "min_shape": (4, 16, 16),
            "opt_shape": (4, 32, 16),
            "max_shape": (4, 64, 16),
        }
        # The lifted buffer drops the input spec, so the kwargs are matched by name.
        trt_mod = torch_tensorrt.dynamo.compile(
            ep,
            arg_inputs=[],
            kwarg_inputs={
                "b": Input(dtype=torch.float16, profiles=[decode, decode]),
                "a": Input(dtype=torch.float16, profiles=[decode, prefill]),
            },
            min_block_size=1,
            enabled_precisions={torch.float16},
        )
        (engine_mod,) = [
            m for m in trt_mod.modules() if isinstance(m, TorchTensorRTModule)
        ]
        engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(
            bytes(engine_mod.serialized_engine)
        )
        self.assertEqual(engine.num_optimization_profiles, 2)
        for name, profile in (("a", prefill), ("b", decode)):
            self.assertEqual(
                [tuple(s) for s in engine.get_tensor_profile_shape(name, 1)],
                [profile["min_shape"], profile["opt_shape"], profile["max_shape"]],
            )

    def test_profiles_of_an_argument_export_renames(self):
        class Upper(torch.nn.Module):
            def __init__(self, mlp):
                super().__init__()
                self.mlp = mlp

            def forward(self, X):
                return self.mlp(X)

        X = torch.randn(4, 48, 16, dtype=torch.float16, device="cuda")
        ep = torch.export.export(
            Upper(self.model),
            (X,),
            dynamic_shapes=({1: torch.export.Dim("seq", min=1, max=64)},),
        )
        decode = {
            "min_shape": (4, 1, 16),
            "opt_shape": (4, 1, 16),
            "max_shape": (4, 8, 16),
        }
        prefill = {
            "min_shape": (4, 16, 16),
            "opt_shape": (4, 32, 16),
            "max_shape": (4, 64, 16),
        }
        # Export names the placeholder x, so the kwarg X is matched by position.
        for require_full_compilation in (True, False):
            with self.subTest(require_full_compilation=require_full_compilation):
                trt_mod = torch_tensorrt.dynamo.compile(
                    ep,
                    arg_inputs=[],
                    kwarg_inputs={
                        "X": Input(dtype=torch.float16, profiles=[decode, prefill])
                    },
                    min_block_size=1,
                    enabled_precisions={torch.float16},
                    require_full_compilation=require_full_compilation,
                )
                (engine_mod,) = [
                    m for m in trt_mod.modules() if isinstance(m, TorchTensorRTModule)
                ]
                engine = trt.Runtime(
                    trt.Logger(trt.Logger.WARNING)
                ).deserialize_cuda_engine(bytes(engine_mod.serialized_engine))
                self.assertEqual(engine.num_optimization_profiles, 2)

    def test_different_profile_counts_raise(self):
        class AddMLP(torch.nn.Module):
            def __init__(self, mlp):
                super().__init__()
                self.mlp = mlp

            def forward(self, x, y):
                return self.mlp(x) + y

        seq = torch.export.Dim("seq", min=1, max=64)
        example = torch.randn(4, 48, 16, dtype=torch.float16, device="cuda")
        ep = torch.export.export(
            AddMLP(self.model), (example, example), dynamic_shapes=({1: seq}, {1: seq})
        )
        profile = {
            "min_shape": (4, 1, 16),
            "opt_shape": (4, 48, 16),
            "max_shape": (4, 64, 16),
        }
        with self.assertRaisesRegex(ValueError, "same number of profiles"):
            torch_tensorrt.dynamo.compile(
                ep,
                arg_inputs=[
                    Input(dtype=torch.float16, profiles=[profile, profile]),
                    Input(dtype=torch.float16, profiles=[profile]),
                ],
                min_block_size=1,
                enabled_precisions={torch.float16},
                require_full_compilation=True,
            )


if __name__ == "__main__":
    run_tests()
