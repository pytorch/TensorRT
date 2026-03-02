# type: ignore

import logging
import time
import unittest
from typing import Any, Callable, List, Optional, Sequence, Tuple

import torch
import torch_tensorrt
from torch.fx.experimental.proxy_tensor import unset_fake_temporarily
from torch.fx.passes.shape_prop import ShapeProp
from torch.testing._internal.common_utils import TestCase
from torch_tensorrt import Input
from torch_tensorrt._Device import Device
from torch_tensorrt._enums import dtype
from torch_tensorrt.dynamo import _defaults
from torch_tensorrt.dynamo._defaults import default_device
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo._tracer import get_dynamic_shapes_args

# Use interpreter, input spec, and test case from fx_ts_compat to test Dynamo Converter Registry
from torch_tensorrt.dynamo.conversion import TRTInterpreter
from torch_tensorrt.dynamo.conversion._conversion import infer_module_output_dtypes
from torch_tensorrt.dynamo.lowering import (
    get_decompositions,
    post_lowering,
    pre_export_lowering,
)
from torch_tensorrt.dynamo.lowering.passes import remove_num_users_is_0_nodes
from torch_tensorrt.dynamo.runtime import PythonTorchTensorRTModule
from torch_tensorrt.dynamo.utils import ATOL, RTOL, get_model_device, get_torch_inputs

_LOGGER: logging.Logger = logging.getLogger(__name__)

# this is the post lowering pass list for the converter test
post_lowering_pass_list_for_converter_test = [
    remove_num_users_is_0_nodes,
]


# this method is only used in our converter test to infer the module output dtypes via dummy inference
# which is due to fx.symbolic_trace does not have the meta['val'] info in the node
# TODO: lan to remove this once our converter test is moved from fx.symbolic_trace to dynamo trace
def infer_module_output_dtypes_for_test(
    module: torch.fx.GraphModule,
    inputs: Sequence[Input],
    device: Device,
    kwarg_inputs: Optional[dict[str, Any]] = None,
    truncate_double: bool = False,
) -> List[dtype]:
    """
    This function performs model inference to determine the output dtypes
    and truncates them accordingly. inputs can be either arg_inputs or flattened input list.
    If it is flattened list, kwarg_inputs should be None, as it is already included in the flattened input.
    """
    # TODO: We can also determine output dtypes from the module.graph based on node metadata.
    # However, our converter tests use fx.symbolic_trace which sometimes does not provide metadata,
    # so we stick to the model inference approach currently.
    with unset_fake_temporarily():
        # Get the device on which the model exists
        # For large models, this can be done on CPU to save GPU memory allocation for TRT.
        device = get_model_device(module)
        torch_inputs = get_torch_inputs(inputs, device)
        if kwarg_inputs is None:
            kwarg_inputs = {}
        torch_kwarg_inputs = get_torch_inputs(kwarg_inputs, device)
        module_outputs = module(*torch_inputs, **torch_kwarg_inputs)
        if not isinstance(module_outputs, (list, tuple)):
            module_outputs = [module_outputs]

    # Int64 outputs can sometimes be generated from within other operators
    # such as aten.sum - such outputs can be truncated
    output_dtypes = []
    for output in module_outputs:
        output_ = output
        # We don't need to check if output is nested here because the input module will be flattened
        if not isinstance(output, torch.Tensor):
            if isinstance(output, str):
                raise ValueError(
                    f"Received an output type {type(output)} that's not in the acceptable datatypes (https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)"
                )
            else:
                output_ = torch.tensor(output)

        if truncate_double and output_.dtype == dtype.float64:
            output_dtypes.append(dtype.float32)
        else:
            output_dtypes.append(dtype._from(output_.dtype))

    return output_dtypes


# this is to enable dynamo tracer as True in the converter test files batch by batch
def get_use_dynamo_tracer(use_dynamo_tracer: Any) -> bool:
    # if in our converter tests we specifically set use_dynamo_tracer field, honor it
    if use_dynamo_tracer is not None and isinstance(use_dynamo_tracer, bool):
        return use_dynamo_tracer
    # if in our converter tests, we did not specify use_dynamo_tracer field
    import inspect
    import os
    import re

    filename = os.path.basename(inspect.stack()[2].filename)
    # enable converter test files which starts with test_a*.py to use dynamo tracer
    pattern = re.compile("^test_([a])+")
    if pattern.match(filename):
        return True
    else:
        return False


# this method is only used in our converter test to infer the module output dtypes via dummy inference
# which is due to fx.symbolic_trace does not have the meta['val'] info in the node
# TODO: lan to remove this once our converter test is moved from fx.symbolic_trace to dynamo trace
def infer_module_output_dtypes_for_test(
    module: torch.fx.GraphModule,
    inputs: Sequence[Input],
    device: Device,
    kwarg_inputs: Optional[dict[str, Any]] = None,
    truncate_double: bool = False,
) -> List[dtype]:
    """
    This function performs model inference to determine the output dtypes
    and truncates them accordingly. inputs can be either arg_inputs or flattened input list.
    If it is flattened list, kwarg_inputs should be None, as it is already included in the flattened input.
    """
    # TODO: We can also determine output dtypes from the module.graph based on node metadata.
    # However, our converter tests use fx.symbolic_trace which sometimes does not provide metadata,
    # so we stick to the model inference approach currently.
    with unset_fake_temporarily():
        # Get the device on which the model exists
        # For large models, this can be done on CPU to save GPU memory allocation for TRT.
        device = get_model_device(module)
        torch_inputs = get_torch_inputs(inputs, device)
        if kwarg_inputs is None:
            kwarg_inputs = {}
        torch_kwarg_inputs = get_torch_inputs(kwarg_inputs, device)
        module_outputs = module(*torch_inputs, **torch_kwarg_inputs)
        if not isinstance(module_outputs, (list, tuple)):
            module_outputs = [module_outputs]

    # Int64 outputs can sometimes be generated from within other operators
    # such as aten.sum - such outputs can be truncated
    output_dtypes = []
    for output in module_outputs:
        output_ = output
        # We don't need to check if output is nested here because the input module will be flattened
        if not isinstance(output, torch.Tensor):
            if isinstance(output, str):
                raise ValueError(
                    f"Received an output type {type(output)} that's not in the acceptable datatypes (https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)"
                )
            else:
                output_ = torch.tensor(output)

        if truncate_double and output_.dtype == dtype.float64:
            output_dtypes.append(dtype.float32)
        else:
            output_dtypes.append(dtype._from(output_.dtype))

    return output_dtypes


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
            serialized_engine = interpreter_result.engine.serialize()
            trt_mod = rt_cls(
                serialized_engine=serialized_engine,
                input_binding_names=list(interpreter_result.input_names),
                output_binding_names=list(interpreter_result.output_names),
                name="test_engine",
                requires_output_allocator=interpreter_result.requires_output_allocator,
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
            serialized_engine = interpreter_result.engine.serialize()
            trt_mod = rt_cls(
                serialized_engine=serialized_engine,
                input_binding_names=list(interpreter_result.input_names),
                output_binding_names=list(interpreter_result.output_names),
                name="test_engine",
                requires_output_allocator=interpreter_result.requires_output_allocator,
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
        settings: CompilationSettings = CompilationSettings(),
        torch_export_dynamic_shapes: Optional[Any] = None,
    ):
        mod = mod.eval()
        if use_dynamo_tracer:
            if torch_export_dynamic_shapes is None:
                torch_export_dynamic_shapes = get_dynamic_shapes_args(
                    mod, original_inputs
                )
            device = default_device()
            torch_export_inputs = get_torch_inputs(original_inputs, device)
            exported_program = torch.export.export(
                mod,
                tuple(torch_export_inputs),
                dynamic_shapes=torch_export_dynamic_shapes,
            )
            # for the dynamo tracer, with dynamic shapes,
            # we need to apply the post lowering passes especially remove _assert_scalar nodes
            for dynamic_shape_spec in torch_export_dynamic_shapes.values():
                if len(dynamic_shape_spec) > 0:
                    enable_passes = True
                    break
            if enable_passes:
                exported_program = pre_export_lowering(exported_program, settings)
                exported_program = exported_program.run_decompositions(
                    get_decompositions(False, settings.decompose_attention)
                )
            fx_module = exported_program.module()
        else:
            fx_module = torch.fx.symbolic_trace(mod)

        if enable_passes:
            fx_module = post_lowering(fx_module, settings)

        if propagate_shapes:
            # TODO: This is currently being used to test embedding_bag_aten due to https://github.com/pytorch/TensorRT/issues/2843
            try:
                device = get_model_device(fx_module)
                torch_inputs = get_torch_inputs(original_inputs, device)
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
        use_dynamo_tracer=None,
        enable_passes=False,
        propagate_shapes=False,
        int32_reqd=False,
        immutable_weights=True,
        use_explicit_typing=False,
        decompose_attention=False,
    ):
        # TODO: lan to remove this and set use_dynamo_traccer to True by default
        # once all the converter test files are moved to use_dynamo_tracer
        use_dynamo_tracer = get_use_dynamo_tracer(use_dynamo_tracer)
        # Previous instance of the interpreter auto-casted 64-bit inputs
        # We replicate this behavior here
        compilation_settings = CompilationSettings(
            enabled_precisions={dtype._from(precision)},
            truncate_double=True,
            immutable_weights=immutable_weights,
            use_explicit_typing=use_explicit_typing,
            decompose_attention=decompose_attention,
        )

        mod = self.generate_graph(
            mod,
            inputs,
            use_dynamo_tracer=use_dynamo_tracer,
            enable_passes=enable_passes,
            propagate_shapes=propagate_shapes,
            settings=compilation_settings,
        )

        for pass_func in post_lowering_pass_list_for_converter_test:
            mod = pass_func(mod, compilation_settings)
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
            if use_dynamo_tracer:
                output_dtypes = infer_module_output_dtypes(
                    mod,
                    truncate_double=compilation_settings.truncate_double,
                )
            else:
                output_dtypes = infer_module_output_dtypes_for_test(
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
        immutable_weights=True,
    ):
        # Previous instance of the interpreter auto-casted 64-bit inputs
        # We replicate this behavior here
        compilation_settings = CompilationSettings(
            enabled_precisions={dtype._from(precision)},
            truncate_double=True,
            immutable_weights=immutable_weights,
        )

        mod = self.generate_graph(
            mod,
            inputs,
            use_dynamo_tracer=use_dynamo_tracer,
            enable_passes=enable_passes,
            settings=compilation_settings,
        )

        for pass_func in post_lowering_pass_list_for_converter_test:
            mod = pass_func(mod, compilation_settings)

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
        use_dynamo_tracer=None,
        enable_passes=False,
        use_example_tensors=True,
        pyt_inputs=None,
        propagate_shapes=False,
        check_dtype=True,
        immutable_weights=True,
        torch_export_dynamic_shapes=None,
    ):
        # TODO: lan to remove this and set use_dynamo_traccer to True by default
        # once all the converter test files are moved to use_dynamo_tracer
        use_dynamo_tracer = get_use_dynamo_tracer(use_dynamo_tracer)

        # Previous instance of the interpreter auto-casted 64-bit inputs
        # We replicate this behavior here
        compilation_settings = CompilationSettings(
            truncate_double=True,
            immutable_weights=immutable_weights,
        )
        mod = self.generate_graph(
            mod,
            input_specs,
            use_dynamo_tracer=use_dynamo_tracer,
            enable_passes=enable_passes,
            propagate_shapes=propagate_shapes,
            settings=compilation_settings,
            torch_export_dynamic_shapes=torch_export_dynamic_shapes,
        )
        for pass_func in post_lowering_pass_list_for_converter_test:
            mod = pass_func(mod, compilation_settings)

        if check_dtype:
            if use_dynamo_tracer:
                output_dtypes = infer_module_output_dtypes(
                    mod,
                    truncate_double=compilation_settings.truncate_double,
                )
            else:
                output_dtypes = infer_module_output_dtypes_for_test(
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
