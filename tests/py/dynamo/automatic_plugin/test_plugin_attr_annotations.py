"""Regression tests for the annotations we hand TRT for scalar plugin attributes.

numpy 2.5 redefined ``npt.NDArray`` as a PEP 695 alias, which made
``typing.get_origin(npt.NDArray[np.float64])`` return the alias instead of
``np.ndarray``. TRT's plugin validator keys off that origin, so every plugin
with a scalar attribute started failing with

    ValueError: Attribute 'alpha' of type NDArray[numpy.float64] is not a
    supported serializable type.

These tests are CPU-only and version-agnostic: they pin the property TRT
actually relies on rather than the spelling of the annotation.
"""

import typing
import unittest

import numpy as np
import pytest
from torch.testing._internal.common_utils import run_tests

from torch_tensorrt.dynamo.conversion.plugins._generate_plugin import (
    _TORCH_SCHEMA_TYPE_TO_PLUGIN_ATTR_TYPE,
    np_scalar_attr_annotation,
)

_DTYPES = (np.float64, np.int64, np.bool_)


class TestScalarAttrAnnotation(unittest.TestCase):
    def test_origin_is_ndarray(self):
        """TRT's ``_is_npt_ndarray`` accepts an annotation only if its origin is
        ``np.ndarray`` — via ``typing.get_origin`` or ``__origin__``."""
        for dtype in _DTYPES:
            with self.subTest(dtype=dtype):
                ann = np_scalar_attr_annotation(dtype)
                self.assertIs(typing.get_origin(ann), np.ndarray)
                self.assertIs(getattr(ann, "__origin__", None), np.ndarray)

    def test_dtype_is_recoverable(self):
        """TRT's ``_infer_numpy_type`` digs the dtype out of the second arg."""
        for dtype in _DTYPES:
            with self.subTest(dtype=dtype):
                args = typing.get_args(np_scalar_attr_annotation(dtype))
                self.assertEqual(len(args), 2)
                self.assertEqual(typing.get_args(args[1])[0], dtype)

    def test_schema_type_map_uses_it(self):
        """The map the plugin generator reads must go through the same helper."""
        self.assertEqual(
            set(_TORCH_SCHEMA_TYPE_TO_PLUGIN_ATTR_TYPE), {"float", "int", "bool"}
        )
        for annotation in _TORCH_SCHEMA_TYPE_TO_PLUGIN_ATTR_TYPE.values():
            self.assertIs(typing.get_origin(annotation), np.ndarray)

    def test_trt_validator_accepts_it(self):
        """End of the chain: TRT's own helpers must accept what we emit."""
        utils = pytest.importorskip(
            "tensorrt_bindings.plugin._utils",
            reason="TensorRT plugin bindings not available",
        )
        validate = pytest.importorskip(
            "tensorrt_bindings.plugin._validate",
            reason="TensorRT plugin bindings not available",
        )
        for dtype in _DTYPES:
            with self.subTest(dtype=dtype):
                ann = np_scalar_attr_annotation(dtype)
                self.assertTrue(utils._is_numpy_array(ann))
                self.assertTrue(utils._is_npt_ndarray(ann))
                self.assertIn(
                    utils._infer_numpy_type(ann), validate.SERIALIZABLE_NP_DTYPES
                )


if __name__ == "__main__":
    run_tests()
