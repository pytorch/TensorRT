from inspect import signature
from typing import TYPE_CHECKING, Any

# Keep this or otherwise torch throws an import error later
import torch._inductor.fx_passes.reinplace
from torch._inductor.fx_utils import FakeTensorUpdater as _torch_FakeTensorUpdater
from torch._inductor.virtualized import V

if TYPE_CHECKING:
    from torch._subclasses import FakeTensorMode
    from torch.fx import GraphModule


class FakeTensorUpdater(_torch_FakeTensorUpdater):  # type: ignore[misc]
    def __init__(self, gm: "GraphModule") -> None:
        # TODO: remove this method entirely in favor of the parent class after
        # https://github.com/pytorch/pytorch/pull/159523 is in a PyTorch release.
        if "gm" in signature(_torch_FakeTensorUpdater).parameters:
            super().__init__(gm)
        else:
            super().__init__(gm.graph)

    def incremental_update(self, fake_mode: "FakeTensorMode") -> Any:
        """Wrap incremental_update to accept fake_mode as an argument."""
        with V.set_fake_mode(fake_mode):
            return super().incremental_update()
