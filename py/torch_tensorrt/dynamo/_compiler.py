import torch
import logging
from typing import Sequence, Any

from torch_tensorrt import EngineCapability, Device

from torch_tensorrt.dynamo import create_backend

logger = logging.getLogger(__name__)


def compile(
    gm: torch.Module,
    example_inputs: Sequence[Any],
    *,
    device=Device._current_device(),
    disable_tf32=False,
    sparse_weights=False,
    enabled_precisions=set(),
    refit=False,
    debug=False,
    capability=EngineCapability.default,
    num_avg_timing_iters=1,
    workspace_size=20 << 30,
    dla_sram_size=1048576,
    dla_local_dram_size=1073741824,
    dla_global_dram_size=536870912,
    calibrator=None,
    truncate_long_and_double=False,
    require_full_compilation=False,
    min_block_size=3,
    torch_executed_ops=[],
    torch_executed_modules=[],
):
    custom_backend = create_backend(
        device=device,
        disable_tf32=disable_tf32,
        sparse_weights=sparse_weights,
        enabled_precisions=enabled_precisions,
        refit=refit,
        debug=debug,
        capability=capability,
        num_avg_timing_iters=num_avg_timing_iters,
        workspace_size=workspace_size,
        dla_sram_size=dla_sram_size,
        dla_local_dram_size=dla_local_dram_size,
        dla_global_dram_size=dla_global_dram_size,
        calibrator=calibrator,
        truncate_long_and_double=truncate_long_and_double,
        require_full_compilation=require_full_compilation,
        min_block_size=min_block_size,
        torch_executed_ops=torch_executed_ops,
        torch_executed_modules=torch_executed_modules,
    )

    model = torch.compile(gm, backend=custom_backend)

    # Ensure compilation occurs by calling the function with provided inputs
    model(*example_inputs)

    return model
