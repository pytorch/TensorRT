from typing import Any, Callable

import torch
import torch.nn as nn
import torch_tensorrt
from torch_tensorrt._enums import dtype
from torch_tensorrt.dynamo import partitioning
from torch_tensorrt.dynamo._compiler import convert_module
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    DYNAMO_CONVERTERS as CONVERTERS,
)
from torch_tensorrt.dynamo.lowering import (
    get_decompositions,
    pre_export_lowering,
)
from torch_tensorrt.dynamo.partitioning._hierarchical_partitioner import (
    hierarchical_adjacency_partition,
)
from torch_tensorrt.dynamo.utils import (
    get_output_metadata,
)
from torchvision import models


class InductorModule(torch.nn.Module):  # type: ignore[misc]
    """Wrapper module for inductor compiled function."""

    def __init__(self, func: Callable[..., Any]) -> None:
        super().__init__()
        self.func = func

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        return x


def main():
    # Create model
    model = SimpleModel().cuda()
    # model = models.efficientnet_b0(pretrained=True).cuda()
    model = model.eval()

    # Create example input
    example_input = torch.randn(1, 3, 224, 224).cuda()

    exported_program = torch.export.export(model, (example_input,))
    exported_program = pre_export_lowering(exported_program)
    exported_program = exported_program.run_decompositions(get_decompositions())

    gm = exported_program.module()

    print("Original Model Structure:\n", gm)

    original_output = model(example_input)

    # 1. Partition the model into blocks that can be executed by different backends
    partitioned_model, op_support = hierarchical_adjacency_partition(
        gm,
        verbose=True,
        min_block_size=1,
        backend_priority=["inductor", "tensorrt"],
        backend_support_map={
            "inductor": {
                "torch.ops.aten.convolution.default",
            },
            "tensorrt": CONVERTERS.keys(),
        },
        torch_executed_ops={
            "torch.ops.aten._native_batch_norm_legit_no_training.default"
        },
        require_full_compilation=False,
        skip_fusion=True,
    )

    print("1. Partitioned Model Structure:\n", partitioned_model)

    # 2. Compile each submodule with the corresponding backend
    submodule_node_dict = {}
    for node in partitioned_model.graph.nodes:
        if "_run_on_acc" not in node.name:
            continue
        submodule_node_dict[node.name] = node

    # Store compiled replicas of Torch subgraphs
    compiled_modules = {}

    for name, _ in partitioned_model.named_children():
        submodule = getattr(partitioned_model, name)
        if not isinstance(submodule, torch.fx.graph_module.GraphModule):
            continue

        if "_run_on_acc" not in name:
            submodule.to("cuda")
            continue

        if name not in submodule_node_dict:
            raise ValueError(
                f"node_name: {name} does not exist in the submodule node dictionary"
            )

        # set the submodule metadata back to the parent module_node
        metadata_list = get_output_metadata(submodule)
        assert len(metadata_list) > 0
        metadata_keys = ["val", "tensor_meta"]
        for key in metadata_keys:
            if key not in submodule_node_dict[name].meta:
                meta_val_list = [
                    metadata[key] for metadata in metadata_list if key in metadata
                ]
                submodule_node_dict[name].meta[key] = meta_val_list
                break

        # Get the submodule inputs for min, opt, max shapes of the graph inputs
        submodule_inputs = partitioning.construct_submodule_inputs(submodule)
        assert submodule_inputs is not None

        # compile submodule with pytorch inductor backend
        if "_run_on_acc_inductor" in name:
            sub_inputs = []
            for input in submodule_inputs:
                sub_input = input.torch_tensor.to(
                    dtype.to(input.dtype, t=torch.dtype)
                ).cuda()
                sub_inputs.append(sub_input)

            compiled_func = torch._inductor.compile(
                submodule,
                sub_inputs,
            )
            # Wrap the compiled function to be a torch.nn.Module
            compiled_submodule = InductorModule(compiled_func)

        # compile submodule with tensorrt backend
        elif "_run_on_acc_tensorrt" in name:
            compiled_submodule = convert_module(
                submodule,
                submodule_inputs,
                name=name,
            )
        else:
            raise ValueError(f"Unknown backend for submodule: {name}")

        compiled_modules[name] = compiled_submodule

    # Replace all FX Modules with compiled Modules
    for name, compiled_module in compiled_modules.items():
        setattr(partitioned_model, name, compiled_module)

    print("2. Compiled Model Structure:\n", partitioned_model)

    with torch.no_grad():
        partitioned_output = partitioned_model(example_input)
        print(
            "3. Verify that Partitioned output == Original output:",
            torch.allclose(partitioned_output, original_output, 1e-2, 1e-2),
        )


if __name__ == "__main__":
    main()
