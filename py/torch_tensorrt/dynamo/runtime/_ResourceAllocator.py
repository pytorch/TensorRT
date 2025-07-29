from typing import Any

import torch


class ResourceAllocationStrategy(torch.nn.Module):  # type: ignore[misc]
    """
    ResourceAllocationStrategy is a context manager module that temporarily enables dynamic resource allocation
    for all TRT submodules of the given compiled_module. When entering the context,
    it sets these submodules to use dynamically allocated resources. Upon exiting, it restores them to their
    original (static) resource allocation mode.
    """

    def __init__(
        self,
        compiled_module: torch.nn.Module,
        dynamically_allocate_resources: bool = True
    ) -> None:
        super(ResourceAllocationStrategy, self).__init__()
        self.compiled_module = compiled_module
        self.dynamically_allocate_resources = dynamically_allocate_resources

    def __enter__(self) -> None:
        print("Entering resource allocator context")
        for name, submodule in self.compiled_module.named_modules():
            if "_run_on_acc" in name:
                submodule.use_dynamically_allocated_resources(dynamically_allocate_resources=self.dynamically_allocate_resources)

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        for name, submodule in self.compiled_module.named_modules():
            if "_run_on_acc" in name:
                submodule.use_dynamically_allocated_resources(dynamically_allocate_resources=self.dynamically_allocate_resources)
