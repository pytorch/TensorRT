import logging
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.proxy_tensor import unset_fake_temporarily
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo.utils import (
    COMPLEX_TO_REAL_DTYPE,
    contains_sym_int,
    extract_var_range_info,
    extract_var_range_info_for_profile,
)

logger = logging.getLogger(__name__)

# Per-profile source-symbol bounds, indexed by optimization-profile index:
#   [ {symbol_name: {"min": int, "opt": int, "max": int}}, ... ]
# Built at the top level from ``Input.profiles`` and the export symbols, then
# propagated to submodule inputs by substituting into their SymInt expressions.
ProfileSourceBounds = List[Dict[str, Dict[str, int]]]


def _build_submodule_profiles(
    input_shape: torch.Size,
    union_min: Sequence[int],
    union_opt: Sequence[int],
    union_max: Sequence[int],
    profile_source_bounds: ProfileSourceBounds,
    num_profiles: int,
) -> Optional[List[Dict[str, Tuple[int, ...]]]]:
    """Evaluate per-profile min/opt/max for a (possibly symbolic) submodule shape.

    For each profile (by index), substitute the profile's source-symbol values
    into each SymInt dim and evaluate. Static dims and dims whose
    symbols cannot be resolved fall back to the union value. Returns an ordered
    list of profile dicts (one per profile index).
    """
    profiles: List[Dict[str, Tuple[int, ...]]] = []
    for i in range(num_profiles):
        sym_bounds = profile_source_bounds[i] if i < len(profile_source_bounds) else {}
        p_min: List[int] = []
        p_opt: List[int] = []
        p_max: List[int] = []
        for d, dim in enumerate(input_shape):
            if isinstance(dim, torch.SymInt):
                try:
                    p_min.append(
                        extract_var_range_info_for_profile(
                            dim, {s: b["min"] for s, b in sym_bounds.items()}
                        )
                    )
                    p_opt.append(
                        extract_var_range_info_for_profile(
                            dim, {s: b["opt"] for s, b in sym_bounds.items()}
                        )
                    )
                    p_max.append(
                        extract_var_range_info_for_profile(
                            dim, {s: b["max"] for s, b in sym_bounds.items()}
                        )
                    )
                except KeyError:
                    # Symbol(s) not present in this profile's source bounds;
                    # fall back to the union range for this dim.
                    p_min.append(int(union_min[d]))
                    p_opt.append(int(union_opt[d]))
                    p_max.append(int(union_max[d]))
            else:
                p_min.append(int(dim))
                p_opt.append(int(dim))
                p_max.append(int(dim))
        profiles.append(
            {
                "min": tuple(p_min),
                "opt": tuple(p_opt),
                "max": tuple(p_max),
            }
        )
    return profiles


def construct_dynamic_input(
    input_shape: torch.Size,
    input_dtype: torch.dtype,
    name: str = "",
    is_shape_tensor: bool = False,
    profile_source_bounds: Optional[ProfileSourceBounds] = None,
    num_profiles: int = 0,
) -> Input:
    """
    Constructs a torch_tensorrt.Input based on a symbolic input
    Args:
        input_shape: A symbolic shape / regular shape of a tensor (which can have a  mix of SymInt nodes and static values)
        profile_source_bounds: Optional per-profile source-symbol bounds used to
            propagate optimization profiles to this (intermediate) input.
        num_profiles: Number of profiles to emit when ``profile_source_bounds``
            is provided.
    Returns:
        A dynamic shaped torch_tensorrt.Input which has the properties of the symbolic shaped input.
    """
    min_shape = []
    opt_shape = []
    max_shape = []
    for d, dim in enumerate(input_shape):
        if isinstance(dim, torch.SymInt):
            min_max_opt = extract_var_range_info(dim)
            unwrapped_min_max_opt: Dict[str, int] = {}
            if "min" not in min_max_opt or min_max_opt["min"] is None:
                logger.warning(
                    f"Dynamic input {name} (shape: {input_shape}) has no min bound for dim {d}, attempting to use a sane default (min: 1). Please set a lower bound using torch._dynamo.mark_dynamic or torch.export.Dim"
                )
                unwrapped_min_max_opt["min"] = 1
            else:
                unwrapped_min_max_opt["min"] = min_max_opt["min"]

            if "max" not in min_max_opt or min_max_opt["max"] is None:
                logger.warning(
                    f"Dynamic input {name} (shape: {input_shape}) has no max bound for dim {d}, attempting to use a sane default (max: min({unwrapped_min_max_opt['min']}) * 2^12). Please set an upper bound using torch._dynamo.mark_dynamic or torch.export.Dim"
                )
                unwrapped_min_max_opt["max"] = unwrapped_min_max_opt["min"] * (2**12)
            else:
                unwrapped_min_max_opt["max"] = min_max_opt["max"]

            # if opt not exist, set it to the mean of min and max
            if "opt" not in min_max_opt or min_max_opt["opt"] is None:
                logger.info(
                    f"Dynamic input {name} (shape: {input_shape}) has no opt target i.e. which shape to specialize for, for dim {d}, attempting to use a sane default (opt: min({min_max_opt['min']}) + max({min_max_opt['max']}) / 2). If you want to specialized further, use torch_tensorrt.compile"
                )
                unwrapped_min_max_opt["opt"] = int(
                    unwrapped_min_max_opt["min"] + unwrapped_min_max_opt["max"] / 2
                )
            else:
                unwrapped_min_max_opt["opt"] = min_max_opt["opt"]

            min_shape.append(unwrapped_min_max_opt["min"])
            opt_shape.append(unwrapped_min_max_opt["opt"])
            max_shape.append(unwrapped_min_max_opt["max"])
        else:
            min_shape.append(dim)
            opt_shape.append(dim)
            max_shape.append(dim)

    # Multi-profile propagation: emit profiles for this intermediate input by
    # substituting source-symbol values into its SymInt dims.
    if profile_source_bounds and num_profiles:
        profiles = _build_submodule_profiles(
            input_shape,
            min_shape,
            opt_shape,
            max_shape,
            profile_source_bounds,
            num_profiles,
        )
        if profiles is not None:
            try:
                return Input(
                    profiles=profiles,
                    dtype=input_dtype,
                    name=name,
                    is_shape_tensor=is_shape_tensor,
                )
            except (ValueError, TypeError) as e:
                # Non-affine / non-monotonic expressions can yield invalid
                # per-profile bounds; fall back to the single union profile.
                logger.warning(
                    f"Could not propagate optimization profiles to submodule input "
                    f"'{name}' (shape {input_shape}): {e}. Falling back to the union "
                    "range for this input."
                )

    return Input(
        min_shape=min_shape,
        opt_shape=opt_shape,
        max_shape=max_shape,
        dtype=input_dtype,
        name=name,
        is_shape_tensor=is_shape_tensor,
    )


def get_input(
    input_shape: torch.Size,
    dtype: torch.dtype,
    name: str = "",
    is_shape_tensor: bool = False,
    profile_source_bounds: Optional[ProfileSourceBounds] = None,
    num_profiles: int = 0,
) -> Input:
    """
    Based on type of dimensions in the input_shape, construct regular or dynamic shaped inputs
    """
    if dtype in COMPLEX_TO_REAL_DTYPE:
        real_dtype = COMPLEX_TO_REAL_DTYPE[dtype]
        real_shape = torch.Size(list(input_shape) + [2])
        logger.info(
            f"Input '{name}' has complex dtype {dtype}. TensorRT does not support complex "
            f"tensors natively; it will be implicitly unpacked to a real tensor of shape "
            f"{real_shape} and dtype {real_dtype} (last dim = [real, imag])."
        )
        dtype = real_dtype
        input_shape = real_shape

    if contains_sym_int(input_shape):
        return construct_dynamic_input(
            input_shape,
            dtype,
            name=name,
            is_shape_tensor=is_shape_tensor,
            profile_source_bounds=profile_source_bounds,
            num_profiles=num_profiles,
        )
    else:
        return Input(
            shape=input_shape, dtype=dtype, name=name, is_shape_tensor=is_shape_tensor
        )


def construct_submodule_inputs(
    module: torch.fx.GraphModule,
    profile_source_bounds: Optional[ProfileSourceBounds] = None,
    num_profiles: int = 0,
) -> Sequence[Input]:
    """
    Construct torch_tensorrt Inputs based on the module inputs.
    The module inputs will have meta data which has the shape and dtype info
    Args:
        module: Input FX GraphModule
        profile_source_bounds: Optional per-profile source-symbol bounds. When
            provided (multi-profile compile), each dynamic submodule input gets
            ``profiles`` derived by substituting source symbols into its
            symbolic shape.
        num_profiles: Number of profiles corresponding to
            ``profile_source_bounds``.
    Returns:
        Sequence of torch_tensorrt.Input's representing inputs to given module
    """
    with unset_fake_temporarily():
        torchtrt_inputs = []
        module_inputs = [
            node for node in module.graph.nodes if node.op == "placeholder"
        ]
        for input in module_inputs:
            if input.meta:
                if "val" in input.meta:
                    input_meta = input.meta["val"]
                    if isinstance(input_meta, (FakeTensor, torch.Tensor)):
                        input_shape = input_meta.size()
                        torchtrt_inputs.append(
                            get_input(
                                input_shape,
                                input_meta.dtype,
                                name=input.name,
                                profile_source_bounds=profile_source_bounds,
                                num_profiles=num_profiles,
                            )
                        )
                    elif isinstance(input_meta, torch.SymInt):
                        # Assuming sym_integers | shape inputs always have torch.int64 dtype
                        torchtrt_inputs.append(
                            get_input(
                                [input_meta],
                                torch.int64,
                                name=input.name,
                                is_shape_tensor=True,
                                profile_source_bounds=profile_source_bounds,
                                num_profiles=num_profiles,
                            )
                        )
                    elif isinstance(input_meta, torch.SymFloat):
                        torchtrt_inputs.append(
                            get_input(
                                [1],
                                torch.float32,
                                name=input.name,
                                is_shape_tensor=False,  # Only SymInt inputs are treated as shape tensors
                            )
                        )
                    else:
                        raise ValueError(
                            f"The meta val for input node {input.target} is of type : {type(input_meta)}. Supported types: torch.Tensor|FakeTensor|torch.SymInt"
                        )

                elif "tensor_meta" in input.meta:
                    input_meta = input.meta["tensor_meta"]
                    input_shape = input_meta.shape
                    torchtrt_inputs.append(
                        get_input(input_shape, input_meta.dtype, name=input.name)
                    )
                else:
                    raise AssertionError(
                        f"Input {input.name} does not contain val and tensor_meta fields in the metadata. Please ensure you have exported the graph correctly"
                    )
            else:
                raise AssertionError(
                    f"Input {input.name} does not contain metadata. Please ensure you have exported the graph correctly"
                )

        return torchtrt_inputs


def build_profile_source_bounds(
    module: torch.fx.GraphModule,
    top_level_inputs: Sequence[Input],
    num_profiles: int,
) -> ProfileSourceBounds:
    """Map export source symbols to per-profile bounds from top-level inputs.

    For each top-level placeholder whose ``Input`` declares ``profiles``, read the
    export ``SymInt`` for each dynamic dim and record, per profile, the
    ``min`` / ``opt`` / ``max`` value of the corresponding source symbol. The
    result feeds :func:`construct_submodule_inputs` so intermediate engines
    inherit the same profiles (by index) via symbolic substitution.

    Args:
        module: Top-level (partitioned) GraphModule whose placeholders carry the
            export symbolic shapes.
        top_level_inputs: Ordered top-level Inputs (arg inputs followed by kwarg
            inputs), aligned with the module placeholders.
        num_profiles: Number of optimization profiles.
    Returns:
        A list indexed by optimization-profile index:
        ``[{symbol_name: {"min": int, "opt": int, "max": int}}, ...]``
    """
    bounds: ProfileSourceBounds = [{} for _ in range(num_profiles)]
    if not num_profiles:
        return bounds

    placeholders = [n for n in module.graph.nodes if n.op == "placeholder"]
    with unset_fake_temporarily():
        for ph, inp in zip(placeholders, top_level_inputs):
            profiles = getattr(inp, "profiles", None)
            if not profiles:
                continue
            if not ph.meta or "val" not in ph.meta:
                continue
            val = ph.meta["val"]
            if not isinstance(val, (FakeTensor, torch.Tensor)):
                continue
            for d, dim in enumerate(val.size()):
                if not isinstance(dim, torch.SymInt):
                    continue
                expr = dim.node.expr
                # Top-level dynamic dims map directly to a single source symbol.
                if not getattr(expr, "is_symbol", False):
                    continue
                sym_name = expr.name
                for i, prof in enumerate(profiles):
                    if i >= len(bounds) or d >= len(prof["min"]):
                        continue
                    bounds[i][sym_name] = {
                        "min": int(prof["min"][d]),
                        "opt": int(prof["opt"][d]),
                        "max": int(prof["max"][d]),
                    }
    return bounds


def run_shape_analysis(
    parent_module: torch.fx.GraphModule,
    inputs: Sequence[Input],
    kwarg_inputs: Optional[dict[str, Any]] = None,
) -> Tuple[Dict[Any, Sequence[Any]], Dict[Any, Sequence[Any]]]:
    submod_inputs_shape_map: Dict[Any, Sequence[Any]] = {}
    submod_outputs_shape_map: Dict[Any, Sequence[Any]] = {}
    sub_inputs: Sequence[torch.Tensor] = []
    sub_outputs: Sequence[torch.Tensor] = []

    # Register a hook to capture IO shapes for submodules
    def get_submodule_io(
        self: Any, inputs: Sequence[torch.Tensor], outputs: Sequence[torch.Tensor]
    ) -> None:
        nonlocal sub_inputs, sub_outputs
        sub_inputs = inputs
        sub_outputs = outputs
        return

    if kwarg_inputs is None:
        kwarg_inputs = {}
    # Iterate through submodules (both Torch and TRT) and store IO shapes
    for name, _ in parent_module.named_children():
        submodule = getattr(parent_module, name)
        handle = submodule.register_forward_hook(get_submodule_io)
        parent_module(*inputs, **kwarg_inputs)
        handle.remove()
        submod_inputs_shape_map[name] = (
            [input.shape for input in sub_inputs]
            if isinstance(sub_inputs, (tuple, list))
            else [sub_inputs.shape]
        )
        submod_outputs_shape_map[name] = (
            [output.shape for output in sub_outputs]
            if isinstance(sub_outputs, (tuple, list))
            else [sub_outputs.shape]
        )

    return submod_inputs_shape_map, submod_outputs_shape_map


def get_graph_converter_support(
    graph_module: torch.fx.GraphModule,
    torch_executed_ops: Optional[Set[str]] = None,
) -> Tuple[int, int]:
    """Helper function to get converter support overview pre-partitioning

    Args:
        graph_module: FX GraphModule to determine support for
        verbose: Bool representing whether to print operator support
        torch_executed_ops: Collection of operations to run in Torch, regardless of converter coverage
    Returns:
        The number of supported call_function nodes in the graph
    """
    from ._global_partitioner import TorchTensorRTOperatorSupport

    # Instantiate operator support object and module dictionary
    op_support = TorchTensorRTOperatorSupport(torch_executed_ops=torch_executed_ops)
    module_dict = dict(graph_module.named_modules())

    number_of_supported_nodes = 0
    total_functional_nodes = 0

    # Iterate over all nodes in the graph, enumerating call_function nodes
    for node in graph_module.graph.nodes:
        if node.op == "call_function":
            total_functional_nodes += 1

            if op_support.is_node_supported(module_dict, node):
                number_of_supported_nodes += 1

    # Print node support overview prior to partitioning
    op_support.print_support_overview(print_node_support=True)

    return number_of_supported_nodes, total_functional_nodes
