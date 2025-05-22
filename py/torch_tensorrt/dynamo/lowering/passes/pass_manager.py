from typing import Any, Callable, List, Optional

import torch
from torch.fx import passes
from torch.fx.passes.pass_manager import PassManager
from torch_tensorrt.dynamo._settings import CompilationSettings


def get_draw_fx_graph_pass_lowering(
    idx: int, path_prefix: str, post: bool
) -> Callable[[torch.fx.GraphModule, CompilationSettings], torch.fx.GraphModule]:
    from torch_tensorrt.dynamo.lowering.passes import (
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

    def draw_fx_graph_pass(
        gm: torch.fx.GraphModule, settings: CompilationSettings
    ) -> torch.fx.GraphModule:
        DEBUG_NAME = POST_DEBUG_NAME[idx] if post else PRE_DEBUG_NAME[idx]
        path = f"{path_prefix}_{DEBUG_NAME}.svg"
        g = passes.graph_drawer.FxGraphDrawer(gm, DEBUG_NAME)
        with open(path, "wb") as f:
            f.write(g.get_dot_graph().create_svg())
        return gm

    return draw_fx_graph_pass


class DynamoPassManager(PassManager):  # type: ignore[misc]
    def __init__(
        self,
        passes: Optional[
            List[
                Callable[
                    [torch.fx.GraphModule, CompilationSettings], torch.fx.GraphModule
                ]
            ]
        ] = None,
    ):
        super().__init__(passes)

    @classmethod
    def build_from_passlist(
        cls,
        passes: Optional[
            List[
                Callable[
                    [torch.fx.GraphModule, CompilationSettings], torch.fx.GraphModule
                ]
            ]
        ],
    ) -> Any:
        pm = DynamoPassManager(passes)
        return pm

    def add_pass_with_index(
        self,
        lowering_pass: Callable[
            [torch.fx.GraphModule, CompilationSettings], torch.fx.GraphModule
        ],
        index: Optional[int] = None,
    ) -> None:
        if index is None:
            self.passes.append(lowering_pass)
            index = -1
        else:
            self.passes.insert(index, lowering_pass)

    def remove_pass_with_index(self, index: int) -> None:
        del self.passes[index]

    def insert_debug_pass(
        self, index: List[int], filename_prefix: str, post: bool = True
    ) -> None:

        for i in range(len(index)):

            debug_pass = get_draw_fx_graph_pass_lowering(
                index[i], filename_prefix, post
            )
            self.add_pass_with_index(debug_pass, index[i] + i)

    def __call__(self, gm: Any, settings: CompilationSettings) -> Any:
        self.validate()
        out = gm
        for _pass in self.passes:
            out = _pass(out, settings)
        return out

    def __str__(self) -> str:
        return str(self.passes)
