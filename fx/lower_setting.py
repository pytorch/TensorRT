import dataclasses as dc
from typing import Type, Set, Optional, Sequence, List

from fx2trt_oss.fx import InputTensorSpec
from fx2trt_oss.fx.passes.lower_basic_pass import (
    fuse_permute_linear,
    fuse_permute_matmul,
)
from fx2trt_oss.fx.utils import LowerPrecision
from torch import nn
from torch.fx.passes.pass_manager import PassManager


@dc.dataclass
class LowerSetting:
    """
    Basic configuration for lowering stack.

    Args:
    max_batch_size: The maximum batch size which can be used at execution time,
    and also the batch size for which the ICudaEngine will be optimized.

    input_specs: Specs for inputs to engine, can either be a single size or a
    range defined by Min, Optimal, Max sizes.

    explicit_batch_dimension: Use explicit batch dimension during lowering.

    explicit_precision: Use explicit precision during lowering.

    lower_precision: lower precision dtype during lowering.

    max_workspace_size: The maximum workspace size. The maximum GPU temporary
    memory which the TensorRT engine can use at execution time.

    strict_type_constraints: Require TensorRT engine to strictly follow data type
    setting at execution time.

    customized_fuse_pass: List of custmozied pass to apply during lowering process.

    lower_basic_fuse_pass: Enable basic pass fuse duirng lowering, i.e. fuse multiple operations
    as (a->b->c->d)=>(e). Current basic fuse patterns are:
    permute->linear
    permute->matmul

    verbose_log: Enable TensorRT engine verbose log mode.

    algo_selector: Enable TensorRT algorithm selector at execution time.

    timing_cache_prefix: TensorRT timing cache file path. TensorRT engine will use timing
    cache file at execution time if valid timing cache file is provided.

    save_timing_cache: Save updated timing cache data into timing cache file if the timing
    cache file is provided.

    ast_rewriter_allow_list (Optional[Set[nn.Module]]): Optional allow list of
    modules that need AST rewriting. This is aiming to eliminate input variable involve in
    exception checking control flow.

    leaf_module_list (Optional[Set[nn.Module]]): Optional leaf module list where
    modules will not be traced into.

    cuda_graph_batch_size (int): Cuda graph batch size, default to be -1.

    verbose_profile (bool): verbosity of profiler, default to False.

    min_acc_module_size(int): minimal number of nodes for an accelerate submodule.
    """

    max_batch_size: int = 2048
    input_specs: List[InputTensorSpec] = dc.field(default_factory=list)
    explicit_batch_dimension: bool = True
    explicit_precision: bool = False
    lower_precision: LowerPrecision = LowerPrecision.FP32
    max_workspace_size: int = 1 << 30
    strict_type_constraints: bool = False
    customized_fuse_pass: PassManager = PassManager.build_from_passlist([])
    lower_basic_fuse_pass: PassManager = PassManager.build_from_passlist(
        [fuse_permute_matmul, fuse_permute_linear]
    )
    verbose_log: bool = False
    algo_selector = None
    timing_cache_prefix: str = ""
    save_timing_cache: bool = False
    ast_rewriter_allow_list: Optional[Set[Type[nn.Module]]] = None
    leaf_module_list: Optional[Set[Type[nn.Module]]] = None
    cuda_graph_batch_size: int = -1
    verbose_profile: bool = False
    min_acc_module_size: int = 10
