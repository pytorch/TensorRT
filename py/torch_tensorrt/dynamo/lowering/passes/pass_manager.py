from typing import Any, Callable, List, Optional, Sequence

import torch
from torch.fx.passes.pass_manager import PassManager
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.draw_fx_graph import (
    get_draw_fx_graph_pass_post_lowering,
    get_draw_fx_graph_pass_pre_lowering,
)


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
            [torch.fx.GraphModule, CompilationSettings, Sequence[torch.Tensor]],
            torch.fx.GraphModule,
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
            if post:
                debug_pass = get_draw_fx_graph_pass_post_lowering(
                    index[i], filename_prefix
                )
            else:
                debug_pass = get_draw_fx_graph_pass_pre_lowering(
                    index[i], filename_prefix
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
