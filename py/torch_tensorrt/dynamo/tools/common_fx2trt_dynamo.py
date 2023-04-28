import logging
import time
import unittest
from typing import Callable, List, Optional, Set, Tuple

import torch
import torch.fx
from torch.testing._internal.common_utils import TestCase
from torch_tensorrt.fx.utils import LowerPrecision
import torch_tensorrt.fx.tracer.dispatch_tracer.aten_tracer as aten_tracer

from torch_tensorrt.dynamo.torch_compile.conversion import convert_module

_LOGGER: logging.Logger = logging.getLogger(__name__)

@unittest.skipIf(not torch.cuda.is_available(), "Skip because CUDA is not available")
class TRTTestCase(TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(3)

    def run_test(
        self,
        mod,
        inputs,
        expected_ops,
        unexpected_ops,
        rtol,
        atol,
        precision=LowerPrecision.FP32,
    ):
        with torch.no_grad():
            cuda_inputs = []
            for i in inputs:
                cuda_inputs.append(i.cuda())

            mod.eval()
            if len(expected_ops):
                self.assert_has_op(mod, expected_ops)
            if unexpected_ops:
                self.assert_unexpected_op(mod, unexpected_ops)
            start = time.perf_counter()
            interpreter_result = convert_module(mod)
            sec = time.perf_counter() - start
            _LOGGER.info(f"Interpreter run time(s): {sec}")

            ref_outputs = mod(*inputs)

            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            outputs = interpreter_result.graph_module(*cuda_inputs)
            end_event.record()
            torch.cuda.synchronize()
            _LOGGER.info(
                f"TRT run time(s)= {(start_event.elapsed_time(end_event) * 1.0e-3)}"
            )

            if type(outputs) not in (list, tuple):
                outputs = [outputs]
            if type(ref_outputs) not in (
                list,
                tuple,
                torch.return_types.max,
                torch.return_types.min,
            ):
                ref_outputs = [ref_outputs]
            for out, ref in zip(outputs, ref_outputs):
                if not isinstance(ref, torch.Tensor):
                    ref = torch.tensor([ref])
                ref = ref.cpu()  # to_dtype test has cases with gpu output
                if ref.dtype == torch.int64:
                    ref = ref.int()  # convert torch.max's index output tensor to int32
                torch.testing.assert_close(
                    out.cpu(), ref, rtol=rtol, atol=atol, equal_nan=True
                )


class DynamoTestCase(TRTTestCase):
    
    def run_test(
        self,
        mod,
        inputs,
        expected_ops,
        unexpected_ops=None,
        apply_passes=None,
        test_explicit_batch_dim=True,
        test_explicit_precision=False,
        rtol=1e-03,
        atol=1e-03,
        precision=LowerPrecision.FP32,
    ):
        mod.eval()
        mod = self.generate_graph(mod, inputs, expected_ops, unexpected_ops, None)

        if apply_passes is not None:
            pass_tracer = chain_passes(*apply_passes)
            mod = pass_tracer(mod, inputs)

        if test_explicit_batch_dim:
            interp = TRTInterpreter(
                mod,
                InputTensorSpec.from_tensors(inputs),
                explicit_batch_dimension=True,
            )
            super().run_test(
                mod, inputs, expected_ops, unexpected_ops, rtol, atol, precision
            )

        if test_explicit_precision:
            interp = TRTInterpreter(
                mod,
                InputTensorSpec.from_tensors(inputs),
                explicit_precision=test_explicit_precision,
            )
            super().run_test(
                mod, inputs, expected_ops, unexpected_ops, interp, rtol, atol
            )

            interp = TRTInterpreter(
                mod,
                InputTensorSpec.from_tensors(inputs),
                explicit_batch_dimension=True,
                explicit_precision=test_explicit_precision,
            )
            super().run_test(
                mod, inputs, expected_ops, unexpected_ops, interp, rtol, atol, precision
            )

    def run_test_with_dynamic_shape(
        self,
        mod,
        input_specs,
        expected_ops,
        unexpected_ops=None,
        rtol=1e-03,
        atol=1e-03,
    ):
        mod.eval()
        inputs = InputTensorSpec.create_inputs_from_specs(input_specs)
        mod = self.generate_graph(mod, inputs, expected_ops, unexpected_ops, None)

        interp = TRTInterpreter(
            mod,
            input_specs,
            explicit_batch_dimension=True,
        )
        # Since the lowering is based on optimal shape. We need to test with
        # different shape(for ex. max shape) for testing dynamic shape
        inputs_max = InputTensorSpec.create_inputs_from_max_specs(input_specs)
        super().run_test(
            mod, inputs_max, expected_ops, unexpected_ops, interp, rtol, atol
        )
