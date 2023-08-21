import logging
import time
import unittest
from typing import Callable, List, Optional, Set, Tuple

import torch
import torch_tensorrt.fx.tracer.dispatch_tracer.aten_tracer as aten_tracer
from torch.fx.passes.infra.pass_base import PassResult
from torch.testing._internal.common_utils import TestCase
from torch_tensorrt import Input

# Use interpreter, input spec, and test case from fx_ts_compat to test Dynamo Converter Registry
from torch_tensorrt.dynamo.conversion import TRTInterpreter
from torch_tensorrt.dynamo.runtime import PythonTorchTensorRTModule
from torch_tensorrt.fx.passes.lower_basic_pass_aten import (
    compose_bmm,
    compose_chunk,
    compose_getitem_slice,
    remove_ops,
    replace_aten_op_with_indices,
    replace_aten_reshape_alias_with_replace,
    replace_builtin_ops,
    replace_native_layernorm_with_layernorm,
    replace_transpose_mm_op_with_linear,
    run_const_fold,
)
from torch_tensorrt.fx.passes.pass_utils import chain_passes

_LOGGER: logging.Logger = logging.getLogger(__name__)


def fetch_attr(mod, target):
    """
    Fetch an attribute from the ``Module`` hierarchy of ``mod.module``.

    Args:
        target (str): The fully-qualfiied name of the attribute to fetch

    Return:
        Any: The value of the attribute.
    """
    target_atoms = target.split(".")
    attr_itr = mod
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(
                f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}"
            )
        attr_itr = getattr(attr_itr, atom)
    return attr_itr


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
        interpreter,
        rtol,
        atol,
        precision=torch.float,
        check_dtype=True,
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
            interpreter_result = interpreter.run(precision=precision)
            sec = time.perf_counter() - start
            _LOGGER.info(f"Interpreter run time(s): {sec}")
            trt_mod = PythonTorchTensorRTModule(
                interpreter_result.engine,
                interpreter_result.input_names,
                interpreter_result.output_names,
            )

            ref_outputs = mod(*inputs)

            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            outputs = trt_mod(*cuda_inputs)
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
                    out.cpu(),
                    ref,
                    rtol=rtol,
                    atol=atol,
                    equal_nan=True,
                    check_dtype=check_dtype,
                )

    def run_test_custom_compare_results(
        self,
        mod,
        inputs,
        expected_ops,
        interpreter,
        comparators: List[Tuple[Callable, List]],
        fp16_mode=False,
    ):
        """
        Runs the test and compares the result using the provided comparators.
        The size of comparators must be equal to the number of outputs from 'mod'.

        mod          - a model to run.
        inputs       - a list of the model inputs.
        expected ops - a list of ops that should be verified.
        interpreter  - used for converting the model to TRT.
        comparators  - a list of (func, args) pairs corresponding to each of
                       the module outputs. usage: func(x, y, *args)

        """
        with torch.no_grad():
            cuda_inputs = []
            for i in inputs:
                cuda_inputs.append(i.cuda())

            mod.eval()
            if len(expected_ops):
                self.assert_has_op(mod, expected_ops)

            interpreter_result = interpreter.run(
                precision=torch.half if fp16_mode else torch.float
            )
            trt_mod = PythonTorchTensorRTModule(
                interpreter_result.engine,
                interpreter_result.input_names,
                interpreter_result.output_names,
            )
            res_trt = trt_mod(*cuda_inputs).cpu()
            res_cpu = mod(*inputs)
            assert len(res_trt) == len(res_cpu)
            assert len(res_cpu) == len(comparators)
            for output_trt, output_cpu, comparator in zip(
                res_trt, res_cpu, comparators
            ):
                comp_func = comparator[0]
                args = comparator[1]
                self.assertTrue(comp_func(output_trt, output_cpu, *args))

    def run_test_with_error(self, mod, inputs, interpreter, expect_error):
        with self.assertRaises(expect_error):
            with torch.no_grad():
                cuda_inputs = []
                for i in inputs:
                    cuda_inputs.append(i.cuda())

                mod.eval()
                interpreter.run(precision=torch.float)

    def assert_has_op(self, mod, ops):
        ops_in_mod = set()

        for node in mod.graph.nodes:
            if node.op == "call_module":
                ops_in_mod.add(type(fetch_attr(mod, node.target)))
            elif node.op in {"call_function", "call_method"}:
                ops_in_mod.add(node.target)

        self.assertTrue(
            ops_in_mod >= ops, f"expected ops {ops}, actuall ops {ops_in_mod}"
        )

    def assert_unexpected_op(self, mod, ops):
        for node in mod.graph.nodes:
            if node.op == "call_module":
                if type(fetch_attr(mod, node.target)) in ops:
                    return False
            elif node.op in {"call_function", "call_method"}:
                if node.target in ops:
                    return False
        return True


class DispatchTestCase(TRTTestCase):
    def generate_graph(
        self,
        mod: torch.nn.Module,
        original_inputs: List[torch.Tensor],
        expected_ops: Set[Callable],
        unexpected_ops: Optional[Set[Callable]] = None,
        customized_passes: List[Callable] = None,
        disable_passes: bool = False,
    ):
        # Torchdynamo+aot proxytensor tracer
        # Below are common passes
        passes_list = [
            compose_bmm,
            compose_chunk,
            compose_getitem_slice,
            replace_aten_reshape_alias_with_replace,
            replace_aten_op_with_indices,
            replace_transpose_mm_op_with_linear,  # after compose_bmm
            replace_native_layernorm_with_layernorm,
            remove_ops,
            replace_builtin_ops,  # after replace_native_layernorm_with_layernorm
        ]
        # Combine with customized passes specific to any model
        if customized_passes:
            passes_list.extend(customized_passes)

        if disable_passes:
            passes_list = []

        fx_module, _ = aten_tracer.trace(mod, original_inputs)
        for passes in passes_list:
            pr: PassResult = passes(fx_module)
            fx_module = pr.graph_module
        fx_module(*original_inputs)

        fx_module = run_const_fold(fx_module)
        _LOGGER.info(f"FX graph= {fx_module.graph}")

        if len(expected_ops):
            self.assert_has_op(fx_module, expected_ops)
        if unexpected_ops:
            self.assert_unexpected_op(fx_module, unexpected_ops)

        return fx_module

    def run_test(
        self,
        mod,
        inputs,
        expected_ops,
        unexpected_ops=None,
        apply_passes=None,
        rtol=1e-03,
        atol=1e-03,
        precision=torch.float,
        check_dtype=True,
        disable_passes=False,
    ):
        mod.eval()
        mod = self.generate_graph(
            mod,
            inputs,
            expected_ops,
            unexpected_ops,
            None,
            disable_passes=disable_passes,
        )

        if apply_passes is not None:
            pass_tracer = chain_passes(*apply_passes)
            mod = pass_tracer(mod, inputs)

        interp = TRTInterpreter(
            mod,
            Input.from_tensors(inputs),
        )
        super().run_test(
            mod,
            inputs,
            expected_ops,
            unexpected_ops,
            interp,
            rtol,
            atol,
            precision,
            check_dtype,
        )

    def run_test_with_dynamic_shape(
        self,
        mod,
        input_specs,
        expected_ops,
        unexpected_ops=None,
        rtol=1e-03,
        atol=1e-03,
        disable_passes=False,
    ):
        mod.eval()
        inputs = [spec.example_tensor("opt_shape") for spec in input_specs]
        mod = self.generate_graph(
            mod,
            inputs,
            expected_ops,
            unexpected_ops,
            None,
            disable_passes=disable_passes,
        )

        interp = TRTInterpreter(
            mod,
            input_specs,
        )
        # Since the lowering is based on optimal shape. We need to test with
        # different shape(for ex. max shape) for testing dynamic shape
        inputs_max = [spec.example_tensor("max_shape") for spec in input_specs]
        super().run_test(
            mod, inputs_max, expected_ops, unexpected_ops, interp, rtol, atol
        )
