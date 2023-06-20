import logging
import unittest
from typing import Optional

import torch
import torch_tensorrt.fx.diagnostics as diagnostics
from torch_tensorrt.fx.passes.pass_utils import (
    override_alternative_batch_size,
    override_alternative_batch_size_exception_should_throw,
    validate_variable_batch_sizes,
)

diagnostics.set_current_collector(
    diagnostics.ZipDiagnosticsCollector(writer=diagnostics.get_current_writer())
)


_LOGGER: logging.Logger = logging.getLogger(__name__)
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)  # configure root logger


class BatchSizeError(Exception):
    pass


class PassUtilsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_run_alternative_batch_size(self):
        class TestModule(torch.nn.Module):
            should_fail_at_bs: Optional[int] = None

            def __init__(self):
                super().__init__()

            def forward(self, x, y, z):
                if x.shape[0] == self.should_fail_at_bs:
                    raise BatchSizeError(self.should_fail_at_bs)

                return x + y + z

        def gen_input(bs: int):
            return [
                torch.rand(bs, 64),
                torch.rand(bs, 64),
                torch.rand(bs, 64),
            ]

        @validate_variable_batch_sizes(1)
        def model_transform_pass_good(model, input):
            """
            This is a good transformation. Meaning that the model it
            produces will not fail at any batch sizes
            """
            model.should_fail_at_bs = None
            return model

        @validate_variable_batch_sizes(1)
        def model_transform_pass_bad(model, input):
            """
            This is a bad transformation. Meaning that the model it produces
            will fail when the given input batch size is 1
            """
            model.should_fail_at_bs = 1
            return model

        model = TestModule()
        input = gen_input(bs=10)

        with diagnostics.collect_when(diagnostics.CollectionConditions.always()):

            with override_alternative_batch_size_exception_should_throw(True):
                # This should succeed: the validate_inference decorator will
                # run both bs=10 and bs=1 successfully
                model_transform_pass_good(model, input)

                # This should fail: the validate_inference decorator will run the
                # model (post transform) at bs=1.
                model.should_fail_at_bs = None  # reset
                self.assertRaises(
                    BatchSizeError, lambda: model_transform_pass_bad(model, input)
                )

                # Test override_alternative_batch_size can disable run alt bs:
                # This should success: the validate_inference decorator will
                # NOT run alternative batch size, because it is disabled via
                # override_alternative_batch_size.
                model.should_fail_at_bs = None  # reset
                with override_alternative_batch_size(alternative_batch_size=-1):
                    model_transform_pass_bad(model, input)

            # Test that by default alt bs failures won't cause exception
            # thrown, because of no
            # `override_alternative_batch_size_exception_should_throw`
            model_transform_pass_bad(model, input)
