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

Two settings are needed, not one. Skipping the copy is not enough on its own:
memory planning allocates graph inputs and outputs by default, so the runtime
would still reserve its own buffer and fill it from the caller's memory with a
host memcpy, which is undefined for device memory and puts the copy straight
back. ``alloc_graph_input=False`` and ``alloc_graph_output=False`` are what stop
that, and ``enable_non_cpu_memory_planning`` is required for planning to run over
non-CPU tensors at all.

Those two settings choose one of two valid arrangements, and it is worth knowing
which. With them off, as here, nobody allocates the boundary buffers and the
CALLER supplies their addresses, which is what a C++ consumer does and what
ExecuTorch's own reference runner is built for. The other arrangement leaves
planning on, so the program's own device arena owns them; ExecuTorch's own CUDA
example takes that one, calling ``to_executorch()`` with no planning overrides at
all.

The first arrangement cannot work from Python as things stand, because the
bindings do not supply an address for a non-planned output and there is no way to
give them one. Measured: with ExecuTorch's CUDA backend owning the last partition
a program exported this way fails every time, and the C++ error says why, that the
data pointer cannot be null. It appears to work when the TensorRT delegate owns
the last partition only because that delegate does not check, and the buffer it is
handed is host memory.

So run a program exported this way from C++, and export with planning left on if
Python is the target.

This script asserts on the serialized program rather than trusting the flags:
the exported ``.pte`` must contain neither copy operator, and it must still carry
both delegates. Checking ``tensor.is_cuda`` at runtime is not enough on its own,
because a round trip that ends back on the device would still look correct.

Prerequisites
-------------
Install Torch-TensorRT with the ExecuTorch extra before running this example::

    pip install -e ".[executorch]" \
        --extra-index-url https://download.pytorch.org/whl/nightly/cu130

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
                # Without these the runtime allocates its own buffer for the
                # input and output it was just told not to copy, and fills the
                # input from the caller's memory with a host memcpy.
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
