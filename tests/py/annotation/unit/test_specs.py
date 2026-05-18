"""Unit tests for CustomPluginSpec op_name invariants.

These two properties have no equivalent in the integration suite: e2e tests
only register one plugin at a time and never inspect op_name directly.  A
broken op_name causes TRT to silently fail to find or alias plugins, which
is extremely hard to diagnose from e2e output alone.
"""

import unittest

import torch_tensorrt.annotation as tta


class TestKernelSpecValidation(unittest.TestCase):
    """Negative tests for TritonSpec / CuTileSpec / CuTeDSLSpec construction."""

    def test_rejects_non_callable_launch_fn(self):
        with self.assertRaises(TypeError):
            tta.TritonSpec(launch_fn="not_callable")

    def test_rejects_non_list_configs(self):
        def kernel(x, out): pass
        with self.assertRaises(TypeError):
            tta.TritonSpec(launch_fn=kernel, configs={"BLOCK": 128})


class TestCustomPluginSpecValidation(unittest.TestCase):
    """Negative tests for user-facing API errors not exercised by integration tests."""

    def test_rejects_missing_meta_impl(self):
        def kernel(x, out): pass
        with self.assertRaises((TypeError, ValueError)):
            tta.custom_plugin(tta.triton(kernel))

    def test_rejects_none_meta_impl(self):
        def kernel(x, out): pass
        with self.assertRaises(ValueError) as ctx:
            tta.custom_plugin(tta.triton(kernel), meta_impl=None)
        self.assertIn("meta_impl", str(ctx.exception))

    def test_rejects_non_callable_meta_impl(self):
        def kernel(x, out): pass
        with self.assertRaises(TypeError) as ctx:
            tta.custom_plugin(tta.triton(kernel), meta_impl="not_callable")
        self.assertIn("callable", str(ctx.exception))

    def test_rejects_invalid_spec_type_lists_all_backends(self):
        with self.assertRaises(TypeError) as ctx:
            tta.custom_plugin("not_a_spec", meta_impl=lambda x: x.new_empty(x.shape))
        msg = str(ctx.exception)
        self.assertIn("TritonSpec", msg)
        self.assertIn("CuTileSpec", msg)
        self.assertIn("CuTeDSLSpec", msg)

    def test_rejects_invalid_element_in_list(self):
        def kernel(x, out): pass
        with self.assertRaises(TypeError):
            tta.custom_plugin([tta.triton(kernel), "invalid"], meta_impl=lambda x: x.new_empty(x.shape))

    def test_rejects_empty_spec_list(self):
        with self.assertRaises(ValueError) as ctx:
            tta.custom_plugin([], meta_impl=lambda x: x.new_empty(x.shape))
        self.assertIn("empty", str(ctx.exception))


class TestCustomPluginSpecOpName(unittest.TestCase):

    def test_op_name_uses_tta_custom_namespace(self):
        def kernel(x, out): pass
        descriptor = tta.custom_plugin(tta.triton(kernel), meta_impl=lambda x: x.new_empty(x.shape))
        ns, name = descriptor.op_name.split("::", 1)
        self.assertEqual(ns, "tta_custom")
        self.assertTrue(len(name) > 0)

    def test_op_name_differs_across_kernel_functions(self):
        """Different kernels must get different op_names — a collision causes TRT
        to silently execute the wrong plugin."""
        def kernel_a(x, out): pass
        def kernel_b(x, out): pass
        meta = lambda x: x.new_empty(x.shape)
        op_a = tta.custom_plugin(tta.triton(kernel_a), meta_impl=meta).op_name
        op_b = tta.custom_plugin(tta.triton(kernel_b), meta_impl=meta).op_name
        self.assertNotEqual(op_a, op_b)
