import unittest
from unittest import mock

from torch_tensorrt.dynamo.conversion.plugins import _custom_op


class TestCustomOpRegistration(unittest.TestCase):
    @mock.patch.object(_custom_op, "generate_plugin_converter")
    @mock.patch.object(_custom_op, "generate_plugin")
    def test_forwards_aot_callbacks_to_existing_plugin_generator(
        self, generate_plugin, generate_plugin_converter
    ):
        aot_impl = mock.Mock(name="aot_impl")
        autotune = mock.Mock(name="autotune")

        _custom_op.custom_op(
            "torchtrt_ex::aot_op",
            aot_impl=aot_impl,
            autotune=autotune,
            supports_dynamic_shapes=True,
        )

        generate_plugin.assert_called_once_with(
            "torchtrt_ex::aot_op", aot_impl=aot_impl, autotune=autotune
        )
        generate_plugin_converter.assert_called_once_with(
            "torchtrt_ex::aot_op",
            None,
            _custom_op.ConverterPriority.STANDARD,
            True,
            False,
            use_aot_if_available=True,
        )

    @mock.patch.object(_custom_op, "generate_plugin_converter")
    @mock.patch.object(_custom_op, "generate_plugin")
    def test_preserves_jit_defaults(self, generate_plugin, _):
        _custom_op.custom_op("torchtrt_ex::jit_op")

        generate_plugin.assert_called_once_with(
            "torchtrt_ex::jit_op", aot_impl=None, autotune=None
        )


if __name__ == "__main__":
    unittest.main()
