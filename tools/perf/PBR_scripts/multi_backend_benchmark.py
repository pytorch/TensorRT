import argparse
import os
import timeit
from typing import Any, Callable

import custom_models as cm
import pandas as pd
import tensorrt as trt
import torch
import torch_tensorrt
from hook_in_engine_to_torch_trt import record_perf
from torch_tensorrt._enums import dtype
from torch_tensorrt.dynamo import partitioning
from torch_tensorrt.dynamo.conversion import convert_module
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    DYNAMO_CONVERTERS as CONVERTERS,
)
from torch_tensorrt.dynamo.lowering import (
    get_decompositions,
    post_lowering,
    pre_export_lowering,
)
from torch_tensorrt.dynamo.partitioning._hierarchical_partitioner import (
    hierarchical_adjacency_partition,
)
from torch_tensorrt.dynamo.utils import (
    get_output_metadata,
)


def multi_backend_test(
    model_name,
    batch_size,
    iterations,
    optimization_level,
    use_python_runtime,
    sdpa_backend,
    enable_cuda_graph,
    output_folder,
):
    class InductorModule(torch.nn.Module):  # type: ignore[misc]
        """Wrapper module for inductor compiled function."""

        def __init__(self, func: Callable[..., Any]) -> None:
            super().__init__()
            self.func = func

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            return self.func(*args, **kwargs)

    # Create model
    if model_name == "sd2.1_unet":
        model = cm.StableDiffusion2_1_Unet().cuda()
        latent_sample = torch.randn((batch_size, 4, 64, 64), dtype=torch.float16).cuda()
        timestep = torch.randint(0, 1000, (batch_size,), dtype=torch.float16).cuda()
        encoder_hidden_states = torch.randn(
            (batch_size, 1, 1024), dtype=torch.float16
        ).cuda()
        inputs = (latent_sample, timestep, encoder_hidden_states)
    elif model_name == "sd2.1_vae_decoder":
        model = cm.StableDiffusion2_1_VaeDecoder().cuda()
        latent_sample = torch.randn((batch_size, 4, 64, 64), dtype=torch.float16).cuda()
        inputs = (latent_sample,)
    elif model_name == "google_vit":
        model = cm.GoogleViTForImageClassification().cuda()
        inputs = (torch.randn((batch_size, 3, 224, 224), dtype=torch.float16).cuda(),)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model = model.eval()

    start_compile_time = timeit.default_timer()

    exported_program = torch.export.export(model, inputs)
    exported_program = pre_export_lowering(exported_program)

    decomps = get_decompositions()
    # remove the scaled_dot_product_attention.default decomposition
    decomps.pop(torch.ops.aten.scaled_dot_product_attention.default)
    exported_program = exported_program.run_decompositions(decomps)

    gm = exported_program.module()

    # print("Original Model Structure:\n", gm)

    compilation_options = {
        "enabled_precisions": {dtype.f16},
        "min_block_size": 1,
        "optimization_level": optimization_level,
        "immutable_weights": True,
        "truncate_double": True,
        "use_python_runtime": use_python_runtime,
    }
    settings = torch_tensorrt.dynamo.CompilationSettings(**compilation_options)

    gm = post_lowering(gm, settings)

    # 1. Partition the model into blocks that can be executed by different backends
    partitioned_model, op_support = hierarchical_adjacency_partition(
        gm,
        min_block_size=1,
        backend_priority=(
            ["inductor", "tensorrt"] if sdpa_backend == "inductor" else ["tensorrt"]
        ),
        backend_support_map={
            "inductor": {
                "torch.ops.aten.scaled_dot_product_attention.default",
            },
            "tensorrt": CONVERTERS.keys(),
        },
        require_full_compilation=False,
        torch_executed_ops=(
            {"torch.ops.aten.scaled_dot_product_attention.default"}
            if sdpa_backend == "torch_eager"
            else set()
        ),
        skip_fusion=True,
    )

    # print("1. Partitioned Model Structure:\n", partitioned_model)

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
                settings=settings,
                name=name,
            )
        else:
            raise ValueError(f"Unknown backend for submodule: {name}")

        compiled_modules[name] = compiled_submodule

    # Replace all FX Modules with compiled Modules
    for name, compiled_module in compiled_modules.items():
        setattr(partitioned_model, name, compiled_module)

    end_compile_time = timeit.default_timer()

    # print("2. Compiled Model Structure:\n", partitioned_model)

    if enable_cuda_graph:
        with torch_tensorrt.runtime.enable_cudagraphs(
            partitioned_model
        ) as cudagraphs_partitioned_model:
            result = record_perf(
                cudagraphs_partitioned_model,
                f"SDPA in {sdpa_backend}, rest in TRT",
                inputs,
                "fp16",
                iterations,
                batch_size,
                compile_time_s=end_compile_time - start_compile_time,
            )
    else:
        result = record_perf(
            partitioned_model,
            f"SDPA in {sdpa_backend}, rest in TRT",
            inputs,
            "fp16",
            iterations,
            batch_size,
            compile_time_s=end_compile_time - start_compile_time,
        )
    print(result)
    summary = pd.DataFrame([result])
    summary.insert(
        loc=0,
        column="model_name",
        value=model_name,
    )
    print(summary)
    runtime_type = "python_runtime" if use_python_runtime else "cpp_runtime"
    if sdpa_backend == "tensorrt":
        graph_break_type = "without_graph_break"
    else:
        graph_break_type = f"with_graph_break_sdpa_in_{sdpa_backend}"

    log_name = os.path.join(
        output_folder,
        f"{model_name}_{graph_break_type}_bs{batch_size}_fp16_optlevel{optimization_level}_{runtime_type}.csv",
    )
    summary.to_csv(log_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--use_python_runtime", action="store_true")
    parser.add_argument("--enable_cuda_graph", action="store_true")
    # default to be tensorrt
    parser.add_argument(
        "--sdpa_backend",
        type=str,
        choices=["tensorrt", "inductor", "torch_eager"],
        default="tensorrt",
    )
    parser.add_argument("--optimization_level", type=int)
    parser.add_argument("--output_folder", type=str, required=True)
    args = parser.parse_args()

    if args.sdpa_backend not in ["tensorrt", "inductor", "torch_eager"]:
        raise ValueError(
            f"Invalid SDPA backend: {args.sdpa_backend}. Valid choices are: tensorrt, inductor, torch_eager"
        )

    multi_backend_test(
        args.model_name,
        args.batch_size,
        args.iterations,
        args.optimization_level,
        args.use_python_runtime,
        args.sdpa_backend,
        args.enable_cuda_graph,
        args.output_folder,
    )
