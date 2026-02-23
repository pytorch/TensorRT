import base64
import logging
from typing import Any, Dict, List

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

    # Build a mapping from compile-time symbolic expressions to runtime SymInts
    # by aligning captured input info with actual runtime input tensors
    symbol_to_symint = {}
    symbol_to_concrete = {}
    shape_env = None

    # Align inputs: for each captured input, match it with the corresponding runtime input
    for idx, (inp_tensor, inp_info) in enumerate(zip(inputs, input_info)):
        for d, s in zip(inp_tensor.shape, inp_info["shape_exprs"]):
            if isinstance(d, torch.SymInt):
                symbol_to_symint[s] = d
                if shape_env is None:
                    shape_env = d.node.shape_env

            elif isinstance(d, int):
                symbol_to_concrete[s] = d

            logger.debug(
                f"[torch.ops.tensorrt.execute_engine]: Meta kernel captured and mapped symbol from input {inp_tensor} (compile time symbol: {s}, new symbol: {d})"
            )

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
                    logger.debug(f"Symbolic expression: {expr}")
                    # Symbolic expression (sympy expr)

                    # Check if this expression uses any symbols that are now concrete
                    has_concrete_symbols = any(
                        sym in symbol_to_concrete for sym in expr.free_symbols
                    )

                    if has_concrete_symbols:
                        # Case 2: Some compile-time symbols are now concrete ints
                        # Evaluate the expression to a concrete value
                        try:
                            # Build substitution dict with concrete values
                            subs_dict = {}
                            for sym in expr.free_symbols:
                                if sym in symbol_to_concrete:
                                    subs_dict[sym] = symbol_to_concrete[sym]
                                elif sym in symbol_to_symint:
                                    subs_dict[sym] = symbol_to_symint[sym].node.hint
                                else:
                                    subs_dict[sym] = sym

                            val = expr.subs(subs_dict)
                            concrete_dim = int(val)
                            output_shape.append(concrete_dim)
                            logger.debug(
                                f"Evaluated {expr} to concrete value {concrete_dim} using concrete mappings"
                            )
                        except Exception as e:
                            raise RuntimeError(
                                f"[torch.ops.tensorrt.execute_engine]: Failed to evaluate symbolic expression {expr} "
                                f"with concrete values. Free symbols: {expr.free_symbols}, "
                                f"Concrete mappings: {symbol_to_concrete}, "
                                f"SymInt mappings: {list(symbol_to_symint.keys())}. Error: {e}"
                            )
                    elif expr in symbol_to_symint:
                        # Case 1a: Direct mapping - compile-time symbol is represented by runtime SymInt
                        output_shape.append(symbol_to_symint[expr])
                        logger.debug(
                            f"Reused SymInt from input: {expr} -> {symbol_to_symint[expr]}"
                        )
                    elif shape_env is not None:
                        # Case 1b: Create new SymInt from expression using existing SymInts
                        try:
                            # Calculate hint by substituting known values
                            hint_val = expr.subs(
                                {
                                    sym: symbol_to_symint[sym].node.hint
                                    for sym in expr.free_symbols
                                    if sym in symbol_to_symint
                                }
                            )
                            hint = int(hint_val) if hint_val.is_number else None

                            # Create new SymInt from the expression
                            output_symint = shape_env.create_symintnode(expr, hint=hint)
                            output_shape.append(output_symint)
                            logger.debug(
                                f"Created new SymInt for {expr} with hint {hint}"
                            )
                        except Exception as e:
                            raise RuntimeError(
                                f"[torch.ops.tensorrt.execute_engine]: Failed to create SymInt for expression {expr}. "
                                f"Error: {e}"
                            )
                    else:
                        raise RuntimeError(
                            "[torch.ops.tensorrt.execute_engine]: No shape_env available during meta kernel execution"
                        )

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


@torch.library.custom_op(  # type: ignore
    "tensorrt::no_op_placeholder_for_execute_engine", mutates_args=()
)
def no_op_placeholder_for_execute_engine(
    inputs: List[torch.Tensor],
    abi_version: str,
    name: str,
    serialized_device_info: str,
    serialized_engine: str,
    serialized_in_binding_names: str,
    serialized_out_binding_names: str,
    serialized_hardware_compatible: str,
    serialized_metadata: str,
    serialized_target_platform: str,
    serialized_require_output_allocator: str,
) -> List[torch.Tensor]:
    raise RuntimeError(
        "The saved model is cross compiled for windows in Linux, should only be loadded in Windows via torch_tensorrt.load_cross_compiled_exported_program() api."
    )
