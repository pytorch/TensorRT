from inspect import signature
from typing import TYPE_CHECKING, Any

# Keep this or otherwise torch throws an import error later
import torch._inductor.fx_passes.reinplace  # noqa: F401
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

        # Nodes can arrive from upstream already missing ``meta['val']`` (e.g.
        # the write-back ``copy_`` left by functionalization of an in-place
        # custom op whose result is returned directly). The parent snapshots
        # every existing node as processed, so it would never fill them in.
        # Unmark them so ``incremental_update`` populates their metadata via
        # ordinary fake-tensor propagation.
        for node in gm.graph.nodes:
            if node.op == "call_function" and "val" not in node.meta:
                self.processed_hashes.discard(self.hash_node(node))

    def incremental_update(self, fake_mode: "FakeTensorMode") -> Any:
        """Wrap incremental_update to accept fake_mode as an argument."""
        with V.set_fake_mode(fake_mode):
            return super().incremental_update()
