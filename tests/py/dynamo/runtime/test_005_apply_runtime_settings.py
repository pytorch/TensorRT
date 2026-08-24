# type: ignore
"""Tests for apply_runtime_settings and the updated CM raise behaviour.

Tests that use save/load run both the module (in-process) and module-less
(AOT-loaded) paths.  The AOT tests additionally verify that the CM:

* raises TypeError on module-less engines (commit 3)
* names apply_runtime_settings in the error message
"""

import os
import tempfile
import unittest

import torch
import torch_tensorrt
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


def _compile_simple():
    model = SimpleModel().eval().cuda()
    inputs = [torch.randn(2, 3).cuda()]
    compiled = torch_tensorrt.compile(
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
        torch_tensorrt.save(compiled, ep_path, arg_inputs=inputs)
        loaded_ep = torch_tensorrt.load(ep_path)
    finally:
        try:
            os.unlink(ep_path)
        except OSError:
            pass
    loaded_gm = loaded_ep.module() if hasattr(loaded_ep, "module") else loaded_ep
    return loaded_ep, loaded_gm


# ---------------------------------------------------------------------------
# Tests that do NOT require an RTX build
# ---------------------------------------------------------------------------


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
        # A plain nn.Module has no TRT engines.
        model = torch.nn.Linear(3, 3).cuda()
        with self.assertRaises(RuntimeError) as cm:
            apply_runtime_settings(model, RuntimeSettings(runtime_cache=None))
        self.assertIn("no TRT engines", str(cm.exception))


# ---------------------------------------------------------------------------
# Tests that require TRT-RTX
# ---------------------------------------------------------------------------


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "apply_runtime_settings dispatch requires TRT-RTX",
)
class TestApplyRuntimeSettingsModuleOwned(TestCase):
    """Module-owned engines: string cache still accepted (module owns it)."""

    def test_module_path_string_accepted(self):
        compiled, inputs = _compile_simple()
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
        compiled, _ = _compile_simple()
        n = apply_runtime_settings(compiled, RuntimeSettings(runtime_cache=None))
        self.assertGreaterEqual(n, 1)


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "AOT load tests require TRT-RTX",
)
class TestApplyRuntimeSettingsModuleLess(TestCase):
    """Module-less engines from save/load."""

    def test_path_string_raises_for_module_less_engine(self):
        compiled, inputs = _compile_simple()
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
        # RuntimeSettings() default runtime_cache is a path string — a common footgun.
        compiled, inputs = _compile_simple()
        _, loaded_gm = _save_load(compiled, inputs)
        with self.assertRaises(TypeError) as cm:
            apply_runtime_settings(loaded_gm, RuntimeSettings())
        self.assertIn("runtime_cache", str(cm.exception))

    def test_none_cache_applies_and_forward_runs(self):
        compiled, inputs = _compile_simple()
        _, loaded_gm = _save_load(compiled, inputs)
        n = apply_runtime_settings(loaded_gm, RuntimeSettings(runtime_cache=None))
        self.assertGreaterEqual(n, 1)
        out = loaded_gm(*inputs)
        self.assertEqual(out.shape, inputs[0].shape)

    def test_runtime_cache_applies_and_persists(self):
        compiled, inputs = _compile_simple()
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
        """Non-cache field applied to a module-less engine; proves field-agnostic dispatch."""
        compiled, inputs = _compile_simple()
        _, loaded_gm = _save_load(compiled, inputs)
        n = apply_runtime_settings(
            loaded_gm,
            RuntimeSettings(
                cuda_graph_strategy="whole_graph_capture",
                runtime_cache=None,
            ),
        )
        self.assertGreaterEqual(n, 1)
        # Forward must still run after strategy change.
        out = loaded_gm(*inputs)
        self.assertEqual(out.shape, inputs[0].shape)

    def test_exported_program_reaches_same_engines_as_module(self):
        """apply_runtime_settings on ExportedProgram reaches the engines ep.module() uses."""
        compiled, inputs = _compile_simple()
        loaded_ep, loaded_gm = _save_load(compiled, inputs)
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            cache_path = f.name
        try:
            cache = RuntimeCache(path=cache_path, autosave_on_del=False)
            # Apply via ExportedProgram.
            n_ep = apply_runtime_settings(
                loaded_ep, RuntimeSettings(runtime_cache=cache)
            )
            self.assertGreaterEqual(n_ep, 1)
            # Running via ep.module() uses the same engine objects -> cache populated.
            _ = loaded_gm(*inputs)
            self.assertTrue(cache.has_cache())
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
        compiled, inputs = _compile_simple()
        _, loaded_gm = _save_load(compiled, inputs)

        from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (
            TorchTensorRTModule,
        )

        # Snapshot prior settings on the module engine.
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

        # Module engine settings must be unchanged (validate-before-mutate).
        for mod, saved in prior.items():
            self.assertEqual(mod.runtime_settings, saved)


@unittest.skipIf(
    not ENABLED_FEATURES.tensorrt_rtx,
    "CM raise tests require TRT-RTX",
)
class TestContextManagerRaisesOnModuleLess(TestCase):
    """runtime_config and runtime_cache raise on module-less engines."""

    def test_runtime_config_raises_on_loaded_gm(self):
        compiled, inputs = _compile_simple()
        _, loaded_gm = _save_load(compiled, inputs)
        with self.assertRaises(TypeError) as cm:
            with runtime_config(loaded_gm, runtime_cache=None):
                pass
        msg = str(cm.exception)
        self.assertIn("apply_runtime_settings", msg)

    def test_runtime_cache_raises_on_loaded_gm(self):
        compiled, inputs = _compile_simple()
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
