"""Whitebox tests for the RuntimeSettings data model + dispatch."""

import dataclasses
import io
import os
import tempfile
import unittest

import torch
import torch_tensorrt as torchtrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt.runtime import (
    RuntimeCache,
    RuntimeSettings,
    apply_runtime_settings,
    runtime_cache,
    runtime_config,
)


class SimpleModel(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x) + 1.0


def _compile_simple(*, runtime_settings=None):
    model = SimpleModel().eval().cuda()
    inputs = [
        torchtrt.Input(
            min_shape=(1, 3),
            opt_shape=(2, 3),
            max_shape=(4, 3),
            dtype=torch.float32,
        )
    ]
    compiled = torchtrt.compile(
        model,
        ir="dynamo",
        inputs=inputs,
        min_block_size=1,
    )
    torch._dynamo.reset()
    if runtime_settings is not None:
        apply_runtime_settings(compiled, runtime_settings)
    return compiled


class TestRuntimeSettingsDataModel(TestCase):
    """Pure dataclass behavior; no engine compile required."""

    def test_defaults_are_valid(self):
        from torch_tensorrt.dynamo._defaults import RUNTIME_CACHE_PATH

        rs = RuntimeSettings()
        self.assertEqual(rs.dynamic_shapes_kernel_specialization_strategy, "lazy")
        self.assertEqual(rs.cuda_graph_strategy, "disabled")
        # Defaults to the per-user temp path from _defaults.py (mirrors ENGINE_CACHE_DIR).
        self.assertEqual(rs.runtime_cache, RUNTIME_CACHE_PATH)

    def test_invalid_ds_strategy_raises_at_post_init(self):
        with self.assertRaises(ValueError):
            RuntimeSettings(dynamic_shapes_kernel_specialization_strategy="bogus")

    def test_invalid_cg_strategy_raises_at_post_init(self):
        with self.assertRaises(ValueError):
            RuntimeSettings(cuda_graph_strategy="bogus")

    def test_frozen(self):
        rs = RuntimeSettings()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            rs.cuda_graph_strategy = "whole_graph_capture"

    def test_merge_overrides(self):
        rs = RuntimeSettings()
        new = rs.merge(cuda_graph_strategy="whole_graph_capture")
        self.assertEqual(new.cuda_graph_strategy, "whole_graph_capture")
        # Original unchanged (frozen + replace).
        self.assertEqual(rs.cuda_graph_strategy, "disabled")

    def test_merge_unknown_key_raises(self):
        rs = RuntimeSettings()
        with self.assertRaises(TypeError):
            rs.merge(not_a_real_field=True)

    def test_equality_compares_all_fields(self):
        a = RuntimeSettings(cuda_graph_strategy="whole_graph_capture")
        b = RuntimeSettings(cuda_graph_strategy="whole_graph_capture")
        c = RuntimeSettings(cuda_graph_strategy="disabled")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_runtime_cache_as_path_string(self):
        rs = RuntimeSettings(runtime_cache="/tmp/whatever.bin")
        self.assertEqual(rs.runtime_cache, "/tmp/whatever.bin")


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "RuntimeSettings dispatch is exercised on TRT-RTX",
)
class TestRuntimeSettingsCompileTimeHint(TestCase):
    """Verify the compile-time hint primes the engine without a CM."""

    def test_compile_hint_sets_engine_settings(self):
        rs = RuntimeSettings(cuda_graph_strategy="whole_graph_capture")
        compiled = _compile_simple(runtime_settings=rs)
        from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (
            TorchTensorRTModule,
        )

        for _, mod in compiled.named_modules():
            if isinstance(mod, TorchTensorRTModule):
                self.assertEqual(
                    mod.runtime_settings.cuda_graph_strategy, "whole_graph_capture"
                )

    def test_runtime_config_cm_restores_on_exit(self):
        compiled = _compile_simple()
        from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (
            TorchTensorRTModule,
        )

        mod = next(
            m for _, m in compiled.named_modules() if isinstance(m, TorchTensorRTModule)
        )
        prior = mod.runtime_settings
        with runtime_config(compiled, cuda_graph_strategy="whole_graph_capture"):
            self.assertEqual(
                mod.runtime_settings.cuda_graph_strategy, "whole_graph_capture"
            )
        self.assertEqual(mod.runtime_settings, prior)


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "Multi-target tests require TRT-RTX",
)
class TestMultiTargetRuntimeConfig(TestCase):
    """`runtime_config([a, b], ...)` applies to engines under both targets."""

    def test_multi_target_runtime_config(self):
        model_a = _compile_simple()
        model_b = _compile_simple()
        with runtime_config(
            [model_a, model_b], cuda_graph_strategy="whole_graph_capture"
        ) as (m_a, m_b):
            self.assertIs(m_a, model_a)
            self.assertIs(m_b, model_b)
            from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (
                TorchTensorRTModule,
            )

            for target in (model_a, model_b):
                for _, mod in target.named_modules():
                    if isinstance(mod, TorchTensorRTModule):
                        self.assertEqual(
                            mod.runtime_settings.cuda_graph_strategy,
                            "whole_graph_capture",
                        )


class TestRuntimeConfigInvalidKey(TestCase):
    """Typo in a CM key should raise at construction, not silently no-op."""

    def test_unknown_kwarg_raises(self):
        # Use a Module that's not a TorchTensorRTModule -- we just need the
        # CM constructor to run; __enter__ won't find any engines.
        target = torch.nn.Linear(3, 3)
        with self.assertRaises(TypeError):
            runtime_config(target, not_a_real_field=True)


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "Lazy IExecutionContext count is meaningful only on TRT-RTX",
)
class TestLazyExecutionContextCreation(TestCase):
    """Regression guard: each setup creates exactly one IExecutionContext.

    On RTX, ``createExecutionContext`` JIT-compiles the specialized kernel set,
    so a redundant create doubles a non-trivial chunk of setup latency. The
    historical cpp-runtime path did two creates per engine setup -- one in the
    torchbind ctor with defaults, one in the post-construction
    ``update_runtime_settings`` dispatch. The lazy-create policy collapses these
    into a single create at first execute.
    """

    def _walk_engines(self, compiled):
        from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (
            TorchTensorRTModule,
        )

        for _, mod in compiled.named_modules():
            if isinstance(mod, TorchTensorRTModule):
                yield mod

    def test_one_context_create_with_default_settings(self):
        compiled = _compile_simple()
        ttrt_modules = list(self._walk_engines(compiled))
        self.assertTrue(ttrt_modules, "Expected at least one TorchTensorRTModule")
        # Both runtimes are now strictly lazy: nothing materializes a context
        # at setup. First execute is the single create site.
        # NCCL/multi-device engines are excluded -- they eagerly bind their
        # NCCL communicator (which materializes the context) at setup so the
        # cross-rank barrier completes before any execute. See
        # ``_TRTEngine._setup_engine`` and ``TRTEngine.cpp:bind_nccl_comm``.
        for mod in ttrt_modules:
            if getattr(mod.engine, "requires_native_multidevice", False):
                continue
            n = mod.engine.num_execution_contexts_created()
            self.assertEqual(n, 0, f"expected 0 contexts at setup, got {n}")

        inputs = [torch.randn(2, 3).cuda()]
        _ = compiled(*inputs)
        for mod in ttrt_modules:
            n = mod.engine.num_execution_contexts_created()
            self.assertEqual(
                n, 1, f"Expected exactly 1 create after first execute, got {n}"
            )

    def test_post_compile_settings_then_execute_is_one_create(self):
        """Compile + post-compile setter + first execute should observe a
        single ``createExecutionContext`` call. Replaces the redundant
        ``test_one_context_create_with_compile_time_settings`` -- with the
        compile-time hint dropped, the only way to apply user settings is via
        the post-compile setter, and the lazy-create policy keeps the count
        at one."""
        compiled = _compile_simple()
        rs = RuntimeSettings(cuda_graph_strategy="whole_graph_capture")
        for mod in self._walk_engines(compiled):
            mod.runtime_settings = rs
            # Settings-set must not eagerly create the context (lazy).
            self.assertEqual(mod.engine.num_execution_contexts_created(), 0)

        inputs = [torch.randn(2, 3).cuda()]
        _ = compiled(*inputs)
        for mod in self._walk_engines(compiled):
            n = mod.engine.num_execution_contexts_created()
            self.assertEqual(
                n,
                1,
                f"Setup + setter + first execute must perform exactly 1 createExecutionContext; got {n}",
            )

    def test_set_runtime_settings_lazy_recreate(self):
        """Changing settings invalidates the context but the recreate is lazy:
        the count bumps on the next execute, not on the set call."""
        compiled = _compile_simple()
        ttrt_modules = list(self._walk_engines(compiled))
        inputs = [torch.randn(2, 3).cuda()]
        _ = compiled(*inputs)
        for mod in ttrt_modules:
            self.assertEqual(mod.engine.num_execution_contexts_created(), 1)

        new_rs = RuntimeSettings(cuda_graph_strategy="whole_graph_capture")
        for mod in ttrt_modules:
            mod.runtime_settings = new_rs
            # Set itself never eagerly recreates on either runtime (lazy).
            self.assertEqual(mod.engine.num_execution_contexts_created(), 1)

        _ = compiled(*inputs)
        for mod in ttrt_modules:
            n = mod.engine.num_execution_contexts_created()
            self.assertEqual(
                n,
                2,
                f"Expected exactly 2 creates after settings flip + execute, got {n}",
            )

    def test_no_op_settings_change_does_not_recreate(self):
        """Re-applying the same RuntimeSettings is a no-op: no invalidate, no
        recreate, count is stable across follow-up executes."""
        rs = RuntimeSettings(cuda_graph_strategy="whole_graph_capture")
        compiled = _compile_simple(runtime_settings=rs)
        ttrt_modules = list(self._walk_engines(compiled))
        inputs = [torch.randn(2, 3).cuda()]
        _ = compiled(*inputs)
        baseline = [mod.engine.num_execution_contexts_created() for mod in ttrt_modules]

        for mod in ttrt_modules:
            mod.runtime_settings = rs  # identical to existing
        _ = compiled(*inputs)
        for mod, prior in zip(ttrt_modules, baseline):
            self.assertEqual(mod.engine.num_execution_contexts_created(), prior)


class TestStateDictRoundTripRuntimeSettings(TestCase):
    """Regression guard for the ``set_extra_state`` -> setter path.

    The B2 bug surfaced specifically through ``load_state_dict`` (which
    routes through ``set_extra_state``), not ``torch.save`` /
    ``torch.load`` (which goes through pickle and restores ``__dict__``
    wholesale, bypassing ``set_extra_state`` entirely). This test takes
    the ``state_dict`` path so it exercises the actual bug path.

    Catches: a future ``set_extra_state`` that overwrites / clobbers
    ``_implicit_cache_handle`` (or any future regression that leaves
    the slot in a state the setter can't handle)."""

    def test_setter_after_load_state_dict_does_not_raise(self):
        src = _compile_simple()
        state = src.state_dict()  # routes through get_extra_state
        dst = _compile_simple()
        dst.load_state_dict(state)  # routes through set_extra_state
        # B2 used to AttributeError on the next line because the slot
        # wasn't initialized after ``set_extra_state``.
        apply_runtime_settings(
            dst, RuntimeSettings(cuda_graph_strategy="whole_graph_capture")
        )


class TestNestedRuntimeConfigCudagraphs(TestCase):
    """Pinned-down composition contract: nesting ``runtime_config`` outside
    ``enable_cudagraphs`` applies settings state-only first; the wrapper's
    warm-up then materializes the context with the strategy already in
    effect (one ``createExecutionContext`` call total).

    Inverted nesting is documented as an anti-pattern but is hard to
    observe via create count -- the cudagraph wrapper replays the
    recorded graph without going through ``engine.context``, so a
    settings flip *inside* the cudagraphs CM doesn't trigger a recreate
    on the next call. The semantic bug there (stale strategy in the
    replayed graph) is not captured by ``num_execution_contexts_created``."""

    def _walk_engines(self, compiled):
        from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (
            TorchTensorRTModule,
        )

        for _, mod in compiled.named_modules():
            if isinstance(mod, TorchTensorRTModule):
                yield mod

    @unittest.skipIf(
        not ENABLED_FEATURES.tensorrt_rtx,
        "cuda_graph_strategy is TRT-RTX-only",
    )
    def test_nested_runtime_config_outside_cudagraphs_is_one_create(self):
        """``with runtime_config(...) as m: with enable_cudagraphs(m) as w:``
        applies settings state-only first; the cudagraphs warm-up then
        materializes the context with the strategy already in effect."""
        from torch_tensorrt.runtime import enable_cudagraphs

        compiled = _compile_simple()
        inputs = [torch.randn(2, 3).cuda()]
        with runtime_config(compiled, cuda_graph_strategy="whole_graph_capture") as m:
            with enable_cudagraphs(m) as w:
                _ = w(*inputs)
        for mod in self._walk_engines(compiled):
            n = mod.engine.num_execution_contexts_created()
            self.assertEqual(
                n,
                1,
                f"Nested form should create exactly 1 context, got {n}",
            )

    @unittest.skipIf(
        not ENABLED_FEATURES.tensorrt_rtx,
        "cuda_graph_strategy is TRT-RTX-only",
    )
    def test_collapsed_cudagraphs_with_strategy_is_one_create(self):
        """``with enable_cudagraphs(mod, cuda_graph_strategy=...) as w:`` --
        the collapsed form folds the prior ``set_cuda_graph_strategy`` CM
        into ``enable_cudagraphs``. Should yield the same single
        ``createExecutionContext`` call as the explicit nested form."""
        from torch_tensorrt.runtime import enable_cudagraphs

        compiled = _compile_simple()
        inputs = [torch.randn(2, 3).cuda()]
        with enable_cudagraphs(
            compiled, cuda_graph_strategy="whole_graph_capture"
        ) as w:
            _ = w(*inputs)
        for mod in self._walk_engines(compiled):
            n = mod.engine.num_execution_contexts_created()
            self.assertEqual(
                n,
                1,
                f"Collapsed form should create exactly 1 context, got {n}",
            )

    @unittest.skipIf(
        ENABLED_FEATURES.tensorrt_rtx,
        "Non-RTX-only fail-fast guard",
    )
    def test_enable_cudagraphs_strategy_kwarg_rejected_on_non_rtx(self):
        """``cuda_graph_strategy`` is a TRT-RTX-only knob. Passing the kwarg
        on a non-RTX build should fail fast at function-call time, not
        silently propagate to a no-op."""
        from torch_tensorrt.runtime import enable_cudagraphs

        compiled = _compile_simple()
        with self.assertRaises(RuntimeError) as cm:
            enable_cudagraphs(compiled, cuda_graph_strategy="whole_graph_capture")
        self.assertIn("TRT-RTX-only", str(cm.exception))


def _compile_with_inputs():
    """Compile SimpleModel and return (compiled, inputs) for save/load tests."""
    model = SimpleModel().eval().cuda()
    inputs = [torch.randn(2, 3).cuda()]
    compiled = torchtrt.compile(
        model,
        ir="dynamo",
        inputs=inputs,
        min_block_size=1,
    )
    torch._dynamo.reset()
    return compiled, inputs


def _save_load(compiled, inputs):
    """Save ``compiled`` to a temp file and return the loaded ExportedProgram and GraphModule."""
    with tempfile.NamedTemporaryFile(suffix=".ep", delete=False) as f:
        ep_path = f.name
    try:
        torchtrt.save(compiled, ep_path, arg_inputs=inputs)
        loaded_ep = torchtrt.load(ep_path)
    finally:
        try:
            os.unlink(ep_path)
        except OSError:
            pass
    loaded_gm = loaded_ep.module() if hasattr(loaded_ep, "module") else loaded_ep
    return loaded_ep, loaded_gm


class TestApplyRuntimeSettingsTypeErrors(TestCase):
    """Rejection of bad arguments; no engine compile required."""

    def test_settings_wrong_type_raises(self):
        model = torch.nn.Linear(3, 3).cuda()
        with self.assertRaises(TypeError) as cm:
            apply_runtime_settings(model, {"cuda_graph_strategy": "disabled"})
        self.assertIn("RuntimeSettings", str(cm.exception))

    def test_target_wrong_type_raises(self):
        with self.assertRaises(TypeError):
            apply_runtime_settings("not_a_module", RuntimeSettings(runtime_cache=None))

    def test_zero_engines_raises(self):
        model = torch.nn.Linear(3, 3).cuda()
        with self.assertRaises(RuntimeError) as cm:
            apply_runtime_settings(model, RuntimeSettings(runtime_cache=None))
        self.assertIn("no TRT engines", str(cm.exception))


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "apply_runtime_settings dispatch requires TRT-RTX",
)
class TestApplyRuntimeSettingsModuleOwned(TestCase):
    """Module-owned engines: string cache still accepted (module owns it)."""

    def test_module_path_string_accepted(self):
        compiled, inputs = _compile_with_inputs()
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            cache_path = f.name
        try:
            n = apply_runtime_settings(
                compiled, RuntimeSettings(runtime_cache=cache_path)
            )
            self.assertGreaterEqual(n, 1)
            _ = compiled(*inputs)
            self.assertTrue(os.path.exists(cache_path))
        finally:
            try:
                os.unlink(cache_path)
            except OSError:
                pass

    def test_returns_engine_count(self):
        compiled, _ = _compile_with_inputs()
        n = apply_runtime_settings(compiled, RuntimeSettings(runtime_cache=None))
        self.assertGreaterEqual(n, 1)


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "AOT load tests require TRT-RTX",
)
class TestApplyRuntimeSettingsModuleLess(TestCase):
    """Module-less engines from save/load."""

    def test_path_string_raises_for_module_less_engine(self):
        compiled, inputs = _compile_with_inputs()
        _, loaded_gm = _save_load(compiled, inputs)
        with self.assertRaises(TypeError) as cm:
            apply_runtime_settings(
                loaded_gm,
                RuntimeSettings(runtime_cache="/tmp/should_not_be_created.bin"),
            )
        msg = str(cm.exception)
        self.assertIn("runtime_cache", msg)
        self.assertIn("RuntimeCache", msg)

    def test_default_runtime_settings_raises_for_module_less_engine(self):
        compiled, inputs = _compile_with_inputs()
        _, loaded_gm = _save_load(compiled, inputs)
        with self.assertRaises(TypeError) as cm:
            apply_runtime_settings(loaded_gm, RuntimeSettings())
        self.assertIn("runtime_cache", str(cm.exception))

    def test_none_cache_applies_and_forward_runs(self):
        compiled, inputs = _compile_with_inputs()
        _, loaded_gm = _save_load(compiled, inputs)
        n = apply_runtime_settings(loaded_gm, RuntimeSettings(runtime_cache=None))
        self.assertGreaterEqual(n, 1)
        out = loaded_gm(*inputs)
        self.assertEqual(out.shape, inputs[0].shape)

    def test_runtime_cache_applies_and_persists(self):
        compiled, inputs = _compile_with_inputs()
        _, loaded_gm = _save_load(compiled, inputs)
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            cache_path = f.name
        try:
            cache = RuntimeCache(path=cache_path, autosave_on_del=False)
            n = apply_runtime_settings(
                loaded_gm,
                RuntimeSettings(runtime_cache=cache),
            )
            self.assertGreaterEqual(n, 1)
            _ = loaded_gm(*inputs)
            self.assertTrue(cache.has_cache())
            cache.save()
            size = os.path.getsize(cache_path)
            self.assertGreater(size, 0)
        finally:
            try:
                os.unlink(cache_path)
            except OSError:
                pass

    def test_cuda_graph_strategy_field_takes_effect(self):
        compiled, inputs = _compile_with_inputs()
        _, loaded_gm = _save_load(compiled, inputs)
        n = apply_runtime_settings(
            loaded_gm,
            RuntimeSettings(
                cuda_graph_strategy="whole_graph_capture",
                runtime_cache=None,
            ),
        )
        self.assertGreaterEqual(n, 1)
        out = loaded_gm(*inputs)
        self.assertEqual(out.shape, inputs[0].shape)

    def test_exported_program_reaches_same_engines_as_module(self):
        compiled, inputs = _compile_with_inputs()
        loaded_ep, loaded_gm = _save_load(compiled, inputs)
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            cache_path = f.name
        try:
            cache = RuntimeCache(path=cache_path, autosave_on_del=False)
            n_ep = apply_runtime_settings(
                loaded_ep, RuntimeSettings(runtime_cache=cache)
            )
            self.assertGreaterEqual(n_ep, 1)
            _ = loaded_gm(*inputs)
            self.assertTrue(cache.has_cache())
        finally:
            try:
                os.unlink(cache_path)
            except OSError:
                pass

    def test_warm_load_bytes_transferred(self):
        """Caller must call .load() / .load_from_stream() to warm the cache before attach.

        The module path auto-calls load() via _resolve_runtime_cache; there is no
        equivalent hook on the module-less path.  Verify warm bytes actually land:
        load_from_stream returns a byte count, so assertGreater(..., 0) is the
        non-vacuous check.
        """
        compiled, inputs = _compile_with_inputs()
        _, loaded_gm = _save_load(compiled, inputs)

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            cache_path = f.name
        try:
            cache_populate = RuntimeCache(path=cache_path, autosave_on_del=False)
            apply_runtime_settings(
                loaded_gm, RuntimeSettings(runtime_cache=cache_populate)
            )
            _ = loaded_gm(*inputs)
            cache_populate.save()
            with open(cache_path, "rb") as fh:
                saved_bytes = fh.read()
            self.assertGreater(len(saved_bytes), 0, "first save produced empty file")

            _, loaded_gm2 = _save_load(compiled, inputs)
            cache_warm = RuntimeCache(autosave_on_del=False)
            n_bytes = cache_warm.load_from_stream(io.BytesIO(saved_bytes))
            self.assertGreater(
                n_bytes,
                0,
                "load_from_stream transferred 0 bytes — warm load did nothing",
            )

            n_engines = apply_runtime_settings(
                loaded_gm2, RuntimeSettings(runtime_cache=cache_warm)
            )
            self.assertGreaterEqual(n_engines, 1)
            _ = loaded_gm2(*inputs)
            self.assertTrue(cache_warm.has_cache())
        finally:
            try:
                os.unlink(cache_path)
            except OSError:
                pass


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "Mixed target test requires TRT-RTX",
)
class TestApplyRuntimeSettingsMixedTarget(TestCase):
    """Module + module-less engines in one call: string must fail whole-call."""

    def test_mixed_target_string_fails_atomically(self):
        compiled, inputs = _compile_with_inputs()
        _, loaded_gm = _save_load(compiled, inputs)

        from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (
            TorchTensorRTModule,
        )

        prior = {
            mod: mod.runtime_settings
            for _, mod in compiled.named_modules()
            if isinstance(mod, TorchTensorRTModule) and mod.engine is not None
        }

        with self.assertRaises(TypeError):
            apply_runtime_settings(
                [compiled, loaded_gm],
                RuntimeSettings(runtime_cache="/tmp/should_not_apply.bin"),
            )

        for mod, saved in prior.items():
            self.assertEqual(mod.runtime_settings, saved)


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "CM raise tests require TRT-RTX",
)
class TestContextManagerRaisesOnModuleLess(TestCase):
    """runtime_config and runtime_cache raise on module-less engines."""

    def test_runtime_config_raises_on_loaded_gm(self):
        compiled, inputs = _compile_with_inputs()
        _, loaded_gm = _save_load(compiled, inputs)
        with self.assertRaises(TypeError) as cm:
            with runtime_config(loaded_gm, runtime_cache=None):
                pass
        msg = str(cm.exception)
        self.assertIn("apply_runtime_settings", msg)

    def test_runtime_cache_raises_on_loaded_gm(self):
        compiled, inputs = _compile_with_inputs()
        _, loaded_gm = _save_load(compiled, inputs)
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            cache_path = f.name
        try:
            with self.assertRaises(TypeError) as cm:
                with runtime_cache(loaded_gm, cache_path):
                    pass
            msg = str(cm.exception)
            self.assertIn("apply_runtime_settings", msg)
        finally:
            try:
                os.unlink(cache_path)
            except OSError:
                pass


if __name__ == "__main__":
    run_tests()
