import base64
import logging
from typing import Any, Dict, List

import sympy
import torch
from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import TorchTensorRTModule

logger = logging.getLogger(__name__)


def _apply_symbolic_shape_expressions(
    inputs: List[torch.Tensor], shape_info: Dict[str, List[Dict[str, Any]]]
) -> List[torch.Tensor]:
    """
    Apply symbolic shape expressions to create output fake tensors.

    This applies the shape expressions captured at compile time to the current
    input fake tensors' symbolic context, using the input alignment to map
    symbolic dimensions.

    Args:
        inputs: Input fake tensors with current symbolic shapes
        shape_info: Dict with 'inputs' and 'outputs' keys containing shape_exprs and dtype info

    Returns:
        List of output fake tensors with symbolic shapes
    """
    from torch._guards import detect_fake_mode

    logger.debug(
        f"[torch.ops.tensorrt.execute_engine]: Meta kernel found the following input FakeTensors: {inputs}"
    )

    input_info = shape_info.get("inputs", [])
    output_info = shape_info.get("outputs", [])

    fake_mode = detect_fake_mode(inputs)
    if fake_mode is None:
        # No fake mode - shouldn't happen, but fall back to concrete shapes
        outputs = []
        for info in output_info:
            shape = [
                int(s) if not hasattr(s, "is_Symbol") else 1
                for s in info["shape_exprs"]
            ]
            outputs.append(
                torch.empty(shape, dtype=info["dtype"], device=inputs[0].device)
            )
        return outputs

    # Shape symbols are local to a ShapeEnv. The expressions in shape_info came
    # from the ShapeEnv used to compile the engine, so they must not be inserted
    # verbatim into the unrelated ShapeEnv doing this fake execution. Both
    # environments can contain a symbol called u0 for different quantities.
    #
    # Give every compile-time symbol a private identity before relating input
    # expressions to their runtime counterparts. Sympy symbols compare by name
    # and assumptions, whereas Dummy objects are unique even when their printed
    # names match a runtime symbol.
    compile_symbols = {
        symbol
        for info in (*input_info, *output_info)
        for expr in info["shape_exprs"]
        if not isinstance(expr, int)
        for symbol in expr.free_symbols
    }
    compile_symbol_namespace = {
        symbol: sympy.Dummy(symbol.name, integer=True) for symbol in compile_symbols
    }

    def in_compile_namespace(expr: sympy.Expr) -> sympy.Expr:
        return expr.xreplace(compile_symbol_namespace)

    # Prefer the ShapeEnv owned by the active FakeTensorMode. Inputs may all be
    # statically shaped even when a data-dependent engine output is symbolic.
    shape_env = getattr(fake_mode, "shape_env", None)
    compile_input_symbols = set()
    compile_to_runtime: Dict[sympy.Expr, sympy.Expr] = {}
    runtime_expr_to_symint = {}
    composite_input_equations = []

    # Align inputs: for each captured input, match it with the corresponding runtime input
    for inp_tensor, inp_info in zip(inputs, input_info):
        for d, compile_expr in zip(inp_tensor.shape, inp_info["shape_exprs"]):
            if isinstance(compile_expr, int):
                continue

            compile_expr = in_compile_namespace(compile_expr)
            compile_input_symbols.update(compile_expr.free_symbols)
            if isinstance(d, torch.SymInt):
                runtime_expr = d.node.expr
                runtime_expr_to_symint[runtime_expr] = d
                if shape_env is None:
                    shape_env = d.node.shape_env
            else:
                runtime_expr = sympy.Integer(d)

            if compile_expr.is_Symbol:
                if compile_expr in compile_to_runtime:
                    residual = compile_to_runtime[compile_expr] - runtime_expr
                    # Two runtime SymInts can be tied to the same compile-time
                    # symbol yet be distinct symbol objects here -- e.g. a
                    # shared torch.export.Dim gets reallocated as separate,
                    # SymInts on retrace/re-export. Preserve the compile-time
                    # equality by installing it as a guard in the active
                    # ShapeEnv instead of requiring the symbols to have
                    # already been merged.
                    error_message = (
                        "[torch.ops.tensorrt.execute_engine]: Runtime input shapes "
                        f"disagree on compile-time symbol {compile_expr}: already "
                        f"mapped to {compile_to_runtime[compile_expr]} from an earlier "
                        f"input, but this input maps it to {runtime_expr}"
                    )
                    if shape_env is not None:
                        residual = shape_env.simplify(residual)
                    residual = sympy.simplify(residual)
                    if residual != 0 and (
                        shape_env is None
                        or not shape_env.guard_or_defer_runtime_assert(
                            sympy.Eq(residual, 0), error_message
                        )
                    ):
                        raise RuntimeError(error_message)
                compile_to_runtime[compile_expr] = runtime_expr
            else:
                # Store the difference: sympy.Eq can auto-collapse to a bare
                # Boolean with no .lhs/.rhs (e.g. Eq(2*d, 7) -> False).
                composite_input_equations.append(compile_expr - runtime_expr)

            logger.debug(
                f"[torch.ops.tensorrt.execute_engine]: Meta kernel captured input shape mapping from {compile_expr} to {runtime_expr}"
            )

    # A constrained input can be recorded as an expression such as 2*s0.
    # Solve those equations for compile-time symbols not mapped by a direct
    # symbolic dimension.
    unresolved_input_symbols = compile_input_symbols - compile_to_runtime.keys()
    if composite_input_equations and unresolved_input_symbols:
        equations = [
            equation.xreplace(compile_to_runtime)
            for equation in composite_input_equations
        ]
        solutions = sympy.solve(
            equations,
            tuple(sorted(unresolved_input_symbols, key=str)),
            dict=True,
        )
        if len(solutions) == 1:
            # Underdetermined systems still return one dict, but a value can
            # contain other unresolved compile-time symbols, e.g. solve(s0+s1-10,
            # (s0,s1)) -> {s0: 10-s1}. Only accept fully resolved values.
            for symbol, value in solutions[0].items():
                if not (value.free_symbols & compile_input_symbols):
                    compile_to_runtime[symbol] = value

    # Validate every composite equation, even if unresolved_input_symbols was
    # empty above (a symbol already mapped elsewhere doesn't mean this
    # relationship was actually satisfied by these runtime inputs).
    for diff in composite_input_equations:
        residual = sympy.simplify(diff.xreplace(compile_to_runtime))
        unresolved_symbols = residual.free_symbols & compile_input_symbols
        if unresolved_symbols:
            raise RuntimeError(
                "[torch.ops.tensorrt.execute_engine]: Could not verify the compile-time "
                f"input relationship {diff} == 0 against these runtime shapes "
                f"(unresolved residual: {residual})"
            )
        if shape_env is not None:
            residual = sympy.simplify(shape_env.simplify(residual))
        if residual == 0:
            continue
        error_message = (
            "[torch.ops.tensorrt.execute_engine]: Runtime input shapes violate "
            f"a relationship captured at compile time: {diff} == 0 does not hold "
            f"for these inputs (residual {residual} != 0)"
        )
        if shape_env is not None and shape_env.guard_or_defer_runtime_assert(
            sympy.Eq(residual, 0), error_message
        ):
            continue
        if residual.is_number:
            raise RuntimeError(error_message)
        # Still symbolic: can't prove it holds, so fail closed.
        raise RuntimeError(
            "[torch.ops.tensorrt.execute_engine]: Could not verify the compile-time "
            f"input relationship {diff} == 0 against these runtime shapes "
            f"(unresolved residual: {residual})"
        )

    # Symbols which occur only in engine outputs represent quantities created
    # by the engine, such as the row count of nonzero. Allocate fresh runtime
    # symbols once per fake invocation, preserve sharing between output
    # expressions, and mark them as valid tensor sizes.
    output_only_symbols = {
        symbol
        for info in output_info
        for expr in info["shape_exprs"]
        if not isinstance(expr, int)
        for symbol in in_compile_namespace(expr).free_symbols
        if symbol not in compile_input_symbols
    }
    if output_only_symbols and shape_env is None:
        raise RuntimeError(
            "[torch.ops.tensorrt.execute_engine]: No shape_env available during meta kernel execution"
        )
    for symbol in sorted(output_only_symbols, key=str):
        runtime_symint = shape_env.create_unbacked_symint()
        shape_env._constrain_range_for_size(runtime_symint.node.expr)
        compile_to_runtime[symbol] = runtime_symint.node.expr
        runtime_expr_to_symint[runtime_symint.node.expr] = runtime_symint

    # Create output fake tensors with symbolic shapes
    logger.debug(f"Deserialized output shape expressions: {output_info}")
    outputs = []
    with fake_mode:
        for output_num, info in enumerate(output_info):
            output_shape = []
            for expr in info["shape_exprs"]:
                if isinstance(expr, int):
                    # Concrete dimension
                    output_shape.append(expr)
                else:
                    compile_expr = in_compile_namespace(expr)
                    missing_input_symbols = (
                        compile_expr.free_symbols
                        & compile_input_symbols - compile_to_runtime.keys()
                    )
                    if missing_input_symbols:
                        raise RuntimeError(
                            "[torch.ops.tensorrt.execute_engine]: Could not map "
                            f"compile-time input symbols {missing_input_symbols} "
                            f"while applying output expression {expr}"
                        )

                    runtime_expr = sympy.simplify(
                        compile_expr.xreplace(compile_to_runtime)
                    )
                    logger.debug(
                        f"Remapped symbolic output expression {expr} to {runtime_expr}"
                    )
                    if runtime_expr.is_number:
                        output_shape.append(int(runtime_expr))
                    elif runtime_expr in runtime_expr_to_symint:
                        output_shape.append(runtime_expr_to_symint[runtime_expr])
                    else:
                        try:
                            output_shape.append(
                                shape_env.create_symintnode(runtime_expr, hint=None)
                            )
                        except Exception as e:
                            raise RuntimeError(
                                f"[torch.ops.tensorrt.execute_engine]: Failed to create SymInt for remapped expression {runtime_expr} (captured as {expr}). Error: {e}"
                            ) from e

            outputs.append(
                torch.empty(output_shape, dtype=info["dtype"], device=inputs[0].device)
            )
    logger.debug(
        f"[torch.ops.tensorrt.execute_engine]: Meta kernel found the following output FakeTensors: {outputs}"
    )
    return outputs


@torch.library.register_fake("aten::cudnn_grid_sampler")  # type: ignore
def fake_aten_cudnn_grid_sampler(
    input: torch.Tensor,
    grid: torch.Tensor,
    interpolation_mode: int = 0,
    padding_mode: int = 0,
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Meta kernel for aten::cudnn_grid_sampler to enable FakeTensor/compile flows.
    Shapes follow grid_sampler semantics:
      - 2D: input [N, C, H_in, W_in], grid [N, H_out, W_out, 2] -> output [N, C, H_out, W_out]
      - 3D: input [N, C, D_in, H_in, W_in], grid [N, D_out, H_out, W_out, 3] -> output [N, C, D_out, H_out, W_out]
    """
    if grid.dim() == 4:
        n, h_out, w_out, _ = grid.shape
        c = input.shape[1]
        out_shape = [n, c, h_out, w_out]
    elif grid.dim() == 5:
        n, d_out, h_out, w_out, _ = grid.shape
        c = input.shape[1]
        out_shape = [n, c, d_out, h_out, w_out]
    else:
        raise RuntimeError(
            f"aten::cudnn_grid_sampler: unexpected grid rank {grid.dim()}"
        )
    return torch.empty(out_shape, dtype=input.dtype, device=input.device)


@torch.library.register_fake("tensorrt::execute_engine")  # type: ignore
def fake_tensorrt_execute_engine(
    inputs: List[torch.Tensor], fake_trt_engine: Any
) -> Any:
    """
    Meta kernel for TensorRT engine execution.

    Uses symbolic shape expressions captured at compile time to correctly infer
    output shapes while preserving symbolic SymInt relationships.
    """

    metadata = None
    if hasattr(fake_trt_engine, "real_obj"):
        # Wrapped C++ engine with real_obj
        trt_engine = fake_trt_engine.real_obj
        metadata = TorchTensorRTModule.decode_metadata(
            trt_engine.get_serialized_metadata()
        )
    else:
        metadata = TorchTensorRTModule.decode_metadata(
            fake_trt_engine.get_serialized_metadata()
        )

    shape_info = metadata.get("inout_symexprs") if metadata else None

    if shape_info:
        # Apply the symbolic shape expressions to create output fake tensors
        # shape_info now contains both 'inputs' and 'outputs' keys
        return _apply_symbolic_shape_expressions(inputs, shape_info)
    else:
        raise RuntimeError(
            "No symbolic shape expressions found in TensorRT engine metadata. "
            "This engine may have been compiled with an older version of Torch-TensorRT. "
            "Please recompile your model."
        )


@torch._library.register_fake_class("tensorrt::Engine")
class FakeTRTEngine:
    def __init__(self, engine_info: List[str]) -> None:
        self.version = engine_info[torch.ops.tensorrt.ABI_TARGET_IDX()]
        self.name = engine_info[torch.ops.tensorrt.NAME_IDX()]
        self.device_info = engine_info[torch.ops.tensorrt.DEVICE_IDX()]
        self.serialized_engine = engine_info[torch.ops.tensorrt.ENGINE_IDX()]
        self.in_binding_names = engine_info[
            torch.ops.tensorrt.INPUT_BINDING_NAMES_IDX()
        ]
        self.out_binding_names = engine_info[
            torch.ops.tensorrt.OUTPUT_BINDING_NAMES_IDX()
        ]
        self.hardware_compatible = engine_info[torch.ops.tensorrt.HW_COMPATIBLE_IDX()]
        self.serialized_metadata = engine_info[
            torch.ops.tensorrt.SERIALIZED_METADATA_IDX()
        ]
        self.requires_output_allocator = engine_info[
            torch.ops.tensorrt.REQUIRES_OUTPUT_ALLOCATOR_IDX()
        ]
        self.target_platform = engine_info[torch.ops.tensorrt.TARGET_PLATFORM_IDX()]

    @classmethod
    def __obj_unflatten__(cls, flattened_tq: Any) -> Any:
        engine_idx = torch.ops.tensorrt.ENGINE_IDX()
        engine_info = [info[1] for info in flattened_tq]
        engine_info[engine_idx] = base64.b64decode(engine_info[engine_idx])

        return cls(engine_info)

    def enable_profiling(self) -> Any:
        pass

    def disable_profiling(self) -> Any:
        pass

    def dump_engine_layer_info_to_file(self, path: str) -> Any:
        pass

    def dump_engine_layer_info(self) -> Any:
        pass

    def get_engine_layer_info(self) -> Any:
        pass

    def profile_path_prefix_getter(self) -> Any:
        pass

    def profile_path_prefix_setter(self) -> Any:
        pass

    def device_memory_budget_getter(self) -> Any:
        pass

    def device_memory_budget_setter(self) -> Any:
        pass

    def streamable_device_memory_budget_getter(self) -> Any:
        pass

    def automatic_device_memory_budget_getter(self) -> Any:
        pass

    def infer_outputs(self, input_shapes: List[Any]) -> Any:
        pass

    def reset_captured_graph(self) -> Any:
        pass

    def get_serialized_metadata(self) -> Any:
        return self.serialized_metadata

    def __setstate__(self, serialized_state: List[str]) -> Any:
        pass

    def __getstate__(self) -> Any:
        pass


@torch.library.custom_op(  # type: ignore[misc]
    "tensorrt::no_op_placeholder_for_execute_engine", mutates_args=()
)
def no_op_placeholder_for_execute_engine(
    inputs: List[torch.Tensor],
    abi_version: str,
    name: str,
    serialized_device_info: str,
    serialized_engine: torch.Tensor,
    serialized_in_binding_names: str,
    serialized_out_binding_names: str,
    serialized_hardware_compatible: str,
    serialized_metadata: str,
    serialized_target_platform: str,
    serialized_require_output_allocator: str,
    serialized_resource_allocation_strategy: str,
    serialized_requires_native_multidevice: str,
    serialized_aliased_io: str,
) -> List[torch.Tensor]:
    raise RuntimeError(
        "The saved model is cross compiled for windows in Linux, should only be loadded in Windows via torch_tensorrt.load_cross_compiled_exported_program() api."
    )


@torch.library.register_fake("tensorrt::no_op_placeholder_for_execute_engine")  # type: ignore
def fake_no_op_placeholder_for_execute_engine(
    inputs: List[torch.Tensor],
    abi_version: str,
    name: str,
    serialized_device_info: str,
    serialized_engine: torch.Tensor,
    serialized_in_binding_names: str,
    serialized_out_binding_names: str,
    serialized_hardware_compatible: str,
    serialized_metadata: str,
    serialized_target_platform: str,
    serialized_require_output_allocator: str,
    serialized_resource_allocation_strategy: str,
    serialized_requires_native_multidevice: str,
    serialized_aliased_io: str,
) -> List[torch.Tensor]:
    """Fake kernel for no_op_placeholder_for_execute_engine.

    Allows ExecuTorch ExportPass subclasses (e.g. RemoveMixedTypeOperators) to
    trace through this op during to_edge_transform_and_lower without hitting the
    C++ schema validator.  Output shapes are inferred from the serialized metadata
    embedded in the op's string args, same as fake_tensorrt_execute_engine.
    """
    from torch_tensorrt.dynamo.runtime._serialized_engine_layout import (
        deserialize_binding_names,
    )
    from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (
        TorchTensorRTModule,
        deserialize_aliased_io,
    )

    metadata = TorchTensorRTModule.decode_metadata(serialized_metadata)
    shape_info = metadata.get("inout_symexprs") if metadata else None
    if shape_info:
        outputs = _apply_symbolic_shape_expressions(inputs, shape_info)
        # Append the engine's aliased (KV-cache) outputs so the getitem indices
        # produced when to_edge re-traces this op stay in range: the aliased
        # outputs are network bindings appended after the fx output boundary, so
        # their shape/dtype come from the aliased input binding.
        aliased_io = deserialize_aliased_io(serialized_aliased_io)
        if aliased_io:
            in_names = deserialize_binding_names(serialized_in_binding_names)
            out_names = deserialize_binding_names(serialized_out_binding_names)
            for out_name in out_names:
                if out_name in aliased_io:
                    in_name = aliased_io[out_name][0]
                    if in_name in in_names:
                        outputs.append(
                            torch.empty_like(inputs[in_names.index(in_name)])
                        )
        return outputs
    else:
        raise RuntimeError(
            "No symbolic shape expressions found in TensorRT engine metadata. "
            "This engine may have been compiled with an older version of Torch-TensorRT. "
            "Please recompile your model."
        )
