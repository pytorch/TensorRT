# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
.. _executorch_export_device_resident:

Exporting a Coalesced Model That Keeps Inputs and Outputs on the GPU
====================================================================

Same graph and same two backends as :ref:`executorch_export_coalesced`, but the
method boundary does no device copies. The caller hands in a CUDA tensor and gets
a CUDA tensor back.

Note the program carries more delegate *entries* than backends: TensorRT claims
the operators on either side of ``erfinv``, so the graph splits into two TensorRT
engines around one CUDA region and the delegate list reads
``['TensorRTBackend', 'CudaBackend', 'TensorRTBackend']``. The check below is
therefore a membership test, not a count.

By default ExecuTorch inserts ``et_copy._h2d_copy`` before a delegate that
consumes a method input and ``et_copy._d2h_copy`` after one that produces a
method output, so a method is safe to call with CPU tensors. For a pipeline that
already has its data on the GPU those copies are pure overhead, and they are what
``PropagateDeviceConfig`` turns off.

Skipping the copies is not enough on its own. Memory planning allocates the
graph inputs by default, and a program that owns its input buffer cannot take a
GPU tensor: setting one fails with error ``0x10`` (``NotSupported``), and the
log says "Error copying tensor data at input 0". So ``alloc_graph_input=False``
is required. ``enable_non_cpu_memory_planning``, on by default, is what lets
planning place tensors in GPU memory at all.

The outputs can go either way. This script also passes
``alloc_graph_output=False``, so the program does not own its output buffers: a
C++ caller sets them with ``Module::set_output``, and ExecuTorch's Python
runtime allocates GPU memory for them itself. Leaving the outputs planned
instead puts them in the program's own GPU memory, and that works from Python
and C++ too.

This script asserts on the serialized program rather than trusting the flags:
the exported ``.pte`` must contain neither copy operator, and it must still carry
both delegates. Checking ``tensor.is_cuda`` at runtime is not enough on its own,
because a round trip that ends back on the device would still look correct.

Prerequisites
-------------
Install Torch-TensorRT with its ``executorch`` extra, as the Installation section of
the ExecuTorch deployment guide shows:
https://docs.pytorch.org/TensorRT/user_guide/runtime_performance/executorch.html#installation

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
from executorch.exir import ExecutorchBackendConfig
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from executorch.exir.passes.propagate_device_pass import PropagateDeviceConfig
from executorch.exir.schema import DeviceType, Tensor

SHAPE = (64, 64)

# The operator names ExecuTorch's PropagateDevicePass inserts at the method
# boundary. Asserting on these by name is the point of this example: they are
# what the skip flags are supposed to remove.
BOUNDARY_COPY_OPS = ("_h2d_copy", "_d2h_copy")


class CoalescedModel(torch.nn.Module):
    def forward(self, x):
        return torch.cos(torch.erfinv(torch.tanh(x)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="coalesced_device_resident.pte",
        help="Path to save the .pte",
    )
    args = parser.parse_args()
    model_path = Path(args.model_path)
    # Appended to the whole path, not substituted for the extension, so this agrees with
    # export_coalesced.py and with any shell that reads it back. with_suffix() replaces whatever
    # the last extension is, so a path like m.v2 wrote m.expected while a reader looking for
    # m.v2.expected found nothing. Appending also cannot land on the model path itself, so the
    # reference can never overwrite the program.
    expected_path = model_path.with_name(model_path.name + ".expected")

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
        # Write beside the target and move it into place only after the checks below pass. Saving
        # straight over the target would let a rejected export replace a good program, and leave
        # the reference file describing something the program no longer is.
        staged_path = model_path.with_name(model_path.name + ".staged")
        torch_tensorrt.save(
            trt_gm,
            str(staged_path),
            output_format="executorch",
            arg_inputs=example_input,
            retrace=False,
            partitioners=[
                CudaPartitioner(
                    [CudaBackend.generate_method_name_compile_spec("forward")]
                )
            ],
            backend_config=ExecutorchBackendConfig(
                propagate_device_config=PropagateDeviceConfig(
                    skip_h2d_for_method_inputs=True,
                    skip_d2h_for_method_outputs=True,
                ),
                enable_non_cpu_memory_planning=True,
                # The runtime refuses to copy the caller's CUDA tensor into a
                # planned input, so the input must not be planned. Outputs work
                # either way.
                memory_planning_pass=MemoryPlanningPass(
                    alloc_graph_input=False, alloc_graph_output=False
                ),
            ),
        )

        program = deserialize_pte_binary(staged_path.read_bytes()).program

        # Still coalesced. A partitioner or config change that routed the whole
        # graph to one backend would otherwise leave this example passing while
        # testing something else entirely.
        delegates = [d.id for plan in program.execution_plan for d in plan.delegates]
        missing = [
            name for name in ("TensorRTBackend", "CudaBackend") if name not in delegates
        ]
        if missing:
            staged_path.unlink(missing_ok=True)
            sys.exit(
                f"{model_path} is not coalesced: missing {missing}, found {delegates}"
            )

        # The assertion this example exists for. Read the operator table of every
        # execution plan and reject the program if either boundary copy survived.
        found = sorted(
            {
                f"{operator.name}.{operator.overload}"
                for plan in program.execution_plan
                for operator in plan.operators
                if any(copy_op in operator.name for copy_op in BOUNDARY_COPY_OPS)
            }
        )
        if found:
            staged_path.unlink(missing_ok=True)
            sys.exit(
                f"FATAL: {model_path} still copies across the method boundary: {found}. "
                "The skip flags did not take, so this program would not keep a CUDA "
                "input on the device."
            )

        # And the boundary really is device-resident, read off the program rather
        # than assumed. A tensor with no extra_tensor_info defaults to CPU in the
        # schema, so a missing record is a failure here, not something to skip.
        host_tensors = []
        for plan in program.execution_plan:
            for kind, indices in (("input", plan.inputs), ("output", plan.outputs)):
                for index in indices:
                    value = plan.values[index].val
                    if not isinstance(value, Tensor):
                        continue
                    info = value.extra_tensor_info
                    device = DeviceType.CPU if info is None else info.device_type
                    if device != DeviceType.CUDA:
                        host_tensors.append(
                            f"{kind}[{index}]={DeviceType(device).name}"
                        )
        if host_tensors:
            staged_path.unlink(missing_ok=True)
            sys.exit(
                f"FATAL: {model_path} has non-CUDA method boundary tensors: "
                f"{host_tensors}"
            )

        # Every check passed, so this program may take the target's place.
        staged_path.replace(model_path)

        reference = model(torch.ones(SHAPE).cuda())
        expected_path.write_text(
            "[{}]\n{:.4f}\n".format(
                ",".join(str(dim) for dim in reference.shape),
                reference.flatten()[0].item(),
            )
        )

    print(f"Saved {model_path} with delegates {delegates}.")
    print(
        f"No {' or '.join(BOUNDARY_COPY_OPS)} in the program: boundary is device-resident."
    )
    print(f"Saved {expected_path} with the eager reference output.")


if __name__ == "__main__":
    main()
