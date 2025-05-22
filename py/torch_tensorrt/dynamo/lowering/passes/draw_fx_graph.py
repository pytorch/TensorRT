import torch
from torch.fx import passes
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes import (
    LoweringPassSignature,
    post_lowering_pass_list,
    pre_lowering_pass_list,
)

PRE_DEBUG_NAME = {
    i + 1: f"after_{p.__name__}" for i, p in enumerate(pre_lowering_pass_list)
}
PRE_DEBUG_NAME[0] = "exported_program"

POST_DEBUG_NAME = {
    i + 1: f"after_{p.__name__}" for i, p in enumerate(post_lowering_pass_list)
}
POST_DEBUG_NAME[0] = "after_decomposition"


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
