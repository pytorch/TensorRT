import torch
from torch.fx import passes
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes import LoweringPassSignature

PRE_DEBUG_NAME = {
    0: "exported_program",
    1: "after_remove_detach,",
}

POST_DEBUG_NAME = {
    0: "after_decomposition",
    1: "after_remove_input_alias_fixing_clones",
    2: "after_constant_fold",
    3: "after_repair_input_as_output",
    4: "after_fuse_prims_broadcast",
    5: "after_replace_max_pool_with_indices",
    6: "after_remove_assert_nodes",
    7: "after_accumulate_fp32_matmul",
    8: "after_remove_num_users_is_0_nodes",
}


def get_draw_fx_graph_pass_post_lowering(
    idx: int, path_prefix: str
) -> LoweringPassSignature:

    def draw_fx_graph_pass(
        gm: torch.fx.GraphModule, settings: CompilationSettings
    ) -> torch.fx.GraphModule:
        path = f"{path_prefix}_{POST_DEBUG_NAME[idx]}.svg"
        g = passes.graph_drawer.FxGraphDrawer(gm, POST_DEBUG_NAME[idx])
        with open(path, "wb") as f:
            f.write(g.get_dot_graph().create_svg())
        return gm

    return draw_fx_graph_pass


def get_draw_fx_graph_pass_pre_lowering(
    idx: int, path_prefix: str
) -> LoweringPassSignature:

    def draw_fx_graph_pass(
        gm: torch.fx.GraphModule, settings: CompilationSettings
    ) -> torch.fx.GraphModule:
        path = f"{path_prefix}_{PRE_DEBUG_NAME[idx]}.svg"
        g = passes.graph_drawer.FxGraphDrawer(gm, PRE_DEBUG_NAME[idx])
        with open(path, "wb") as f:
            f.write(g.get_dot_graph().create_svg())
        return gm

    return draw_fx_graph_pass
