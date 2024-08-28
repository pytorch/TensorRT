# type: ignore

import logging
import time
import unittest
from typing import Callable, List, Optional, Tuple

import torch
import torch_tensorrt
from torch.fx.passes.shape_prop import ShapeProp
from torch.testing._internal.common_utils import TestCase
from torch_tensorrt import Input
from torch_tensorrt._enums import dtype
from torch_tensorrt.dynamo import _defaults
from torch_tensorrt.dynamo._settings import CompilationSettings

# Use interpreter, input spec, and test case from fx_ts_compat to test Dynamo Converter Registry
from torch_tensorrt.dynamo.conversion import TRTInterpreter
from torch_tensorrt.dynamo.conversion._conversion import infer_module_output_dtypes
from torch_tensorrt.dynamo.lowering import (
    get_decompositions,
    post_lowering,
    pre_export_lowering,
)
from torch_tensorrt.dynamo.runtime import PythonTorchTensorRTModule
from torch_tensorrt.dynamo.utils import ATOL, RTOL, get_torch_inputs

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
        interpreter,
        rtol=RTOL,
        atol=ATOL,
        check_dtype=True,
        pyt_inputs=None,
        rt_cls=PythonTorchTensorRTModule,
    ):
        with torch.no_grad():
            cuda_inputs = []
            for i in inputs:
                cuda_inputs.append(i.cuda())

            start = time.perf_counter()
            interpreter_result = interpreter.run()
            sec = time.perf_counter() - start
            _LOGGER.info(f"Interpreter run time(s): {sec}")
            trt_mod = rt_cls(
                serialized_engine=interpreter_result.serialized_engine,
                input_binding_names=list(interpreter_result.input_names),
                output_binding_names=list(interpreter_result.output_names),
                name="test_engine",
            )
            mod = mod.cuda()
            if pyt_inputs is not None:
                pyt_inputs_cuda = [
                    i.cuda() if isinstance(i, torch.Tensor) else i for i in pyt_inputs
                ]
                ref_outputs = mod(*pyt_inputs_cuda)
            else:
                ref_outputs = mod(*cuda_inputs)

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
                    if len(out.shape) == 0:
                        ref = torch.tensor(ref)
                    else:
                        ref = torch.tensor([ref])
                ref = ref.cpu()  # to_dtype test has cases with gpu output
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
        rt_cls=PythonTorchTensorRTModule,
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

            if len(expected_ops):
                self.assert_has_op(mod, expected_ops)

            interpreter_result = interpreter.run()
            trt_mod = rt_cls(
                serialized_engine=interpreter_result.serialized_engine,
                input_binding_names=list(interpreter_result.input_names),
                output_binding_names=list(interpreter_result.output_names),
                name="test_engine",
            )
            res_trt = trt_mod(*cuda_inputs).cpu()
            res_cpu = mod(*cuda_inputs).cpu()
            assert len(res_trt) == len(res_cpu)
            comparator = comparators

            if len(cuda_inputs) == 1:
                for comparator in comparators:
                    comp_func = comparator[0]
                    args = comparator[1]
                    self.assertTrue(comp_func(res_trt, res_cpu, *args))
            else:
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
            ops_in_mod >= ops, f"expected ops {ops}, actual ops {ops_in_mod}"
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
        use_dynamo_tracer: bool,
        enable_passes: bool,
        propagate_shapes: bool = False,
    ):
        mod = mod.eval()
        torch_inputs = get_torch_inputs(original_inputs, _defaults.DEVICE)
        if use_dynamo_tracer:
            exported_program = torch_tensorrt.dynamo.trace(mod, tuple(original_inputs))
            exported_program = pre_export_lowering(exported_program)
            exported_program = exported_program.run_decompositions(
                get_decompositions(False)
            )
            fx_module = exported_program.module()
        else:
            fx_module = torch.fx.symbolic_trace(mod)

        if enable_passes:
            fx_module = post_lowering(fx_module)

        if propagate_shapes:
            # TODO: This is currently being used to test embedding_bag_aten due to https://github.com/pytorch/TensorRT/issues/2843
            try:
                ShapeProp(fx_module).propagate(*torch_inputs)
            except (RuntimeError, AssertionError):
                _LOGGER.warning(
                    "Shape Propagation failed on Graph, skipping it",
                    exc_info=False,
                )
        return fx_module

    def run_test(
        self,
        mod,
        inputs,
        rtol=RTOL,
        atol=ATOL,
        precision=dtype.f32,
        check_dtype=True,
        use_dynamo_tracer=False,
        enable_passes=False,
        propagate_shapes=False,
        int32_reqd=False,
    ):
        mod = self.generate_graph(
            mod,
            inputs,
            use_dynamo_tracer=use_dynamo_tracer,
            enable_passes=enable_passes,
            propagate_shapes=propagate_shapes,
        )

        # Previous instance of the interpreter auto-casted 64-bit inputs
        # We replicate this behavior here
        compilation_settings = CompilationSettings(
            enabled_precisions={dtype._from(precision)},
            truncate_double=True,
            debug=True,
        )

        num_inputs = len(inputs)
        trt_inputs = inputs
        dtype_to_change = []
        if int32_reqd:
            dtype_to_change = [torch.int64, torch.float64]
        else:
            dtype_to_change = [
                torch.float64,
            ]
        for num_input in range(num_inputs):
            input = inputs[num_input]
            if input.dtype in dtype_to_change:
                dtype_32bit = (
                    torch.float32 if (input.dtype == torch.float64) else torch.int32
                )
                trt_inputs = (
                    list(trt_inputs[:num_input])
                    + [
                        input.to(dtype_32bit),
                    ]
                    + list(trt_inputs[num_input + 1 :])
                )

        trt_input_specs = [Input.from_tensor(i) for i in trt_inputs]
        input_specs = [Input.from_tensor(i) for i in inputs]

        output_dtypes = None
        if check_dtype:
            output_dtypes = infer_module_output_dtypes(
                mod,
                input_specs,
                compilation_settings.device,
                truncate_double=compilation_settings.truncate_double,
            )

        _LOGGER.debug(f"Compilation settings: {compilation_settings}")
        _LOGGER.debug(f"Inputs: {input_specs}")
        _LOGGER.debug(f"Output types: {output_dtypes}")

        interp = TRTInterpreter(
            mod,
            trt_input_specs,
            output_dtypes=output_dtypes,
            compilation_settings=compilation_settings,
        )

        super().run_test(
            mod,
            trt_inputs,
            interp,
            rtol,
            atol,
            check_dtype,
            pyt_inputs=inputs,
        )

    def run_test_compare_tensor_attributes_only(
        self,
        mod,
        inputs,
        expected_ops,
        comparators: List[Tuple[Callable, List]],
        precision=torch.float,
        output_dtypes=None,
        use_dynamo_tracer=False,
        enable_passes=False,
    ):
        mod = self.generate_graph(
            mod,
            inputs,
            use_dynamo_tracer=use_dynamo_tracer,
            enable_passes=enable_passes,
        )
        # Previous instance of the interpreter auto-casted 64-bit inputs
        # We replicate this behavior here
        compilation_settings = CompilationSettings(
            enabled_precisions={dtype._from(precision)},
            truncate_double=True,
            debug=True,
        )

        interp = TRTInterpreter(
            mod,
            Input.from_tensors(inputs),
            output_dtypes=output_dtypes,
            compilation_settings=compilation_settings,
        )
        super().run_test_custom_compare_results(
            mod, inputs, expected_ops, interp, comparators
        )

    def run_test_with_dynamic_shape(
        self,
        mod,
        input_specs,
        rtol=RTOL,
        atol=ATOL,
        output_dtypes=None,
        use_dynamo_tracer=False,
        enable_passes=False,
        use_example_tensors=True,
        pyt_inputs=None,
        propagate_shapes=False,
        check_dtype=True,
    ):
        mod = self.generate_graph(
            mod,
            input_specs,
            use_dynamo_tracer=use_dynamo_tracer,
            enable_passes=enable_passes,
            propagate_shapes=propagate_shapes,
        )

        # Previous instance of the interpreter auto-casted 64-bit inputs
        # We replicate this behavior here
        compilation_settings = CompilationSettings(truncate_double=True)

        if check_dtype:
            output_dtypes = infer_module_output_dtypes(
                mod,
                input_specs,
                compilation_settings.device,
                truncate_double=compilation_settings.truncate_double,
            )

        interp = TRTInterpreter(
            mod,
            input_specs,
            output_dtypes=output_dtypes,
            compilation_settings=compilation_settings,
        )
        # Since the lowering is based on optimal shape. We need to test with
        # different shape(for ex. max shape) for testing dynamic shape
        inputs_max = [
            (
                spec.example_tensor("max_shape")
                if spec.shape_mode == Input._ShapeMode.DYNAMIC
                else spec.example_tensor()
            )
            for spec in input_specs
        ]
        if not use_example_tensors:
            inputs_max = [spec.torch_tensor for spec in input_specs]
        super().run_test(mod, inputs_max, interp, rtol, atol, pyt_inputs=pyt_inputs)
