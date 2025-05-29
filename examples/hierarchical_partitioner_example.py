# from torch_tensorrt.dynamo.partitioning._global_partitioner import partition
import torch
import torch.nn as nn
import torch_tensorrt
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    DYNAMO_ATEN_CONVERTERS,
)
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    DYNAMO_CONVERTERS as CONVERTERS,
)
from torch_tensorrt.dynamo.lowering import (
    get_decompositions,
    post_lowering,
    pre_export_lowering,
)
from torch_tensorrt.dynamo.partitioning._adjacency_partitioner import partition
from torch_tensorrt.dynamo.partitioning._hierarchical_partitioner import (
    hierarchical_partition,
)


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

    print(gm.graph)

    # Partition the model using the adjacency partitioner
    # partitioned_model, op_support = partition(
    #     gm,
    #     verbose=True,
    #     min_block_size=1,
    #     torch_executed_ops=[
    #         torch.ops.aten.relu.default,
    #     ],
    # )

    partitioned_model, op_support = hierarchical_partition(
        gm,
        verbose=True,
        min_block_size=1,
        backend_priority=["mlir", "tensorrt"],  # , "inductor"],
        backend_support_map={
            "mlir": {
                # operator.getitem,
                torch.ops.aten.conv2d.default,
                torch.ops.aten.convolution.default,
            },
            "tensorrt": set(DYNAMO_ATEN_CONVERTERS.keys()),
            # "inductor": {
            #     torch.ops.aten.relu.default,
            # },
        },
        torch_executed_ops=[
            torch.ops.aten._native_batch_norm_legit_no_training.default
        ],
        require_full_compilation=False,
        skip_fusion=False,
    )

    print("\nPartitioned Model Structure:")
    print(partitioned_model)

    with torch.no_grad():
        output = partitioned_model(example_input)
        print("Partitioned output:", output)
        print(
            "Partitioned output == original output:",
            torch.allclose(model(example_input), output, 1e-2, 1e-2),
        )


if __name__ == "__main__":
    main()
