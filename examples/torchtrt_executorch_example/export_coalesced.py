"""
.. _executorch_export_coalesced:

Exporting a Coalesced TensorRT + CUDA Model to ExecuTorch (.pte)
================================================================

This example exports one graph split across two ExecuTorch backends. TensorRT
takes the operators it can convert, and everything left over goes to
ExecuTorch's own CUDA backend, which compiles it with AOTInductor. Both
delegates end up inside a single ``.pte`` and run in the same method.

The model is ``cos(erfinv(tanh(x)))``. TensorRT has no converter for
``erfinv``, so that operator is the one the CUDA backend has to claim. That
makes the split real rather than incidental.

A value produced by one delegate is consumed by the other on the device, inside
a single method, so this is what proves the two backends can complete each
other rather than only work on their own.

The CUDA backend also writes an ``aoti_cuda_blob.ptd`` next to the ``.pte`` for
its external weights. This model has no weights, so that file is empty of
tensors and the reference runner does not need it.

Besides the ``.pte`` this writes ``<model_path>.expected``, holding the output
shape and the eager reference value for an all-ones input. The reference runner
gate reads that file instead of hard-coding a number, so the expected value
cannot drift away from the model.

Prerequisites
-------------
Install Torch-TensorRT with the ExecuTorch extra before running this example::

    pip install -e ".[executorch]"

ExecuTorch's CUDA backend also needs a CUDA toolkit (``nvcc``) at export time,
for the AOTInductor compile.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch_tensorrt
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.exir._serialize._program import deserialize_pte_binary

SHAPE = (64, 64)


class CoalescedModel(torch.nn.Module):
    def forward(self, x):
        return torch.cos(torch.erfinv(torch.tanh(x)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", default="coalesced.pte", help="Path to save the .pte"
    )
    args = parser.parse_args()
    model_path = Path(args.model_path)

    with torch.no_grad():
        model = CoalescedModel().eval().cuda()
        example_input = (torch.randn(SHAPE).cuda(),)

        exported_program = torch.export.export(model, example_input)
        trt_gm = torch_tensorrt.dynamo.compile(
            exported_program,
            arg_inputs=example_input,
            min_block_size=1,
            truncate_double=True,
        )
        torch_tensorrt.save(
            trt_gm,
            str(model_path),
            output_format="executorch",
            arg_inputs=example_input,
            retrace=False,
            # Catch-all, so every operator TensorRT rejected goes to the CUDA
            # backend instead of falling back to a portable CPU kernel.
            partitioners=[
                CudaPartitioner(
                    [CudaBackend.generate_method_name_compile_spec("forward")]
                )
            ],
        )

        # Both delegates must really be in the file. A partitioner change that
        # quietly routed the whole graph to TensorRT would otherwise leave a
        # green job that no longer tests the coalesced path at all.
        program = deserialize_pte_binary(model_path.read_bytes()).program
        delegates = [d.id for plan in program.execution_plan for d in plan.delegates]
        missing = [
            name for name in ("TensorRTBackend", "CudaBackend") if name not in delegates
        ]
        if missing:
            sys.exit(
                f"{model_path} is not coalesced: missing {missing}, found {delegates}"
            )

        # The reference runners fill every input element with 1.0, and this model
        # is elementwise, so a single number describes the whole expected output.
        reference = model(torch.ones(SHAPE).cuda())
        # Enforced, not assumed. Writing one element as the expected value for the
        # whole tensor is only sound while that holds, and the gate compares every
        # value the runner prints against it. Editing the model into something
        # non-uniform would otherwise still write a plausible reference file, and the
        # gate would then either fail blaming the runner or pass proving nothing.
        if not torch.isfinite(reference).all():
            sys.exit(
                f"{model_path} produced a non-finite reference output, which the gate reads as "
                "zero and would then accept a run of zeros from a dead delegate."
            )
        if reference.unique().numel() != 1:
            sys.exit(
                f"{model_path} produced a non-uniform reference output "
                f"({reference.unique().numel()} distinct values), so one number cannot "
                "describe it. Update this script to write the values the gate should expect."
            )
        # Appended to the whole path, not substituted for the extension, so this agrees
        # with the shell that reads it back. with_suffix() replaces whatever the last
        # extension is, so a path like m.v2 wrote m.expected while the gate looked for
        # m.v2.expected and stopped.
        expected_path = model_path.with_name(model_path.name + ".expected")
        expected_path.write_text(
            "[{}]\n{:.4f}\n".format(
                ",".join(str(dim) for dim in reference.shape),
                reference.flatten()[0].item(),
            )
        )

    print(f"Saved {model_path} with delegates {delegates}.")
    print(f"Saved {expected_path} with the eager reference output.")


if __name__ == "__main__":
    main()
