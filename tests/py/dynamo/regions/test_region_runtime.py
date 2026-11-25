# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Guard region manifests against mutation, and refuse CUDA graphs over PyTorch regions."""

import pytest
import torch
import torch_tensorrt
from torch_tensorrt import region
from torch_tensorrt.dynamo.region_compilation import (
    normalize_region_scopes,
)
from torch_tensorrt.dynamo.region_compilation._validation import (
    validate_region_mutations,
)
from torch_tensorrt.region._session import RegionCompilationSession

from ._utils import compile_regions


class MutationModel(torch.nn.Module):
    def __init__(
        self, target="buffer", placement="before", annotation="live", backend="eager"
    ):
        super().__init__()
        self.register_buffer("state", torch.zeros(2, 8))
        self.target = target
        self.placement = placement
        self.annotation = annotation
        self.backend = backend

    def forward(self, x):
        mutated = self.state if self.target == "buffer" else x
        if self.target == "temporary":
            mutated = x.clone()
        if self.placement == "before":
            mutated.add_(1)
        if self.annotation == "live":
            with region.execute_in_torch(backend=self.backend):
                result = torch.relu(x + mutated)
        else:
            if self.annotation == "empty":
                with region.execute_in_torch():
                    pass
            result = torch.relu(x + mutated)
        if self.placement == "after":
            mutated.add_(1)
        return result + mutated


def functional_export(model):
    with RegionCompilationSession():
        exported = torch.export.export(model, (torch.randn(2, 8),), strict=True)
        return normalize_region_scopes(exported).run_decompositions()


@pytest.mark.parametrize("annotation,target", [("none", "buffer"), ("empty", "input")])
def test_no_live_region_preserves_existing_mutation_support(annotation, target):
    exported = functional_export(MutationModel(target=target, annotation=annotation))
    validate_region_mutations(exported)


@pytest.mark.parametrize("placement", ["before", "after"])
def test_local_temporary_mutation_remains_supported(placement):
    exported = functional_export(MutationModel("temporary", placement))
    validate_region_mutations(exported)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "target,placement,backend",
    [
        ("buffer", "before", "eager"),
        ("buffer", "after", "inductor"),
        ("input", "after", "eager"),
        ("input", "before", "inductor"),
    ],
)
def test_public_compile_rejects_mutation_before_buffer_lifting(
    monkeypatch, target, placement, backend
):
    from torch_tensorrt.dynamo import _compiler

    def unexpected_lifting(*args, **kwargs):
        raise AssertionError("Mutation reached buffer lifting before region validation")

    monkeypatch.setattr(_compiler, "lift_mutated_buffers", unexpected_lifting)
    model = MutationModel(target, placement, backend=backend).cuda().eval()
    with pytest.raises(region.RegionError, match=f"mutation of {target}"):
        torch_tensorrt.compile(
            model,
            inputs=[torch.randn(2, 8, device="cuda")],
            ir="dynamo",
            strict=True,
            offload_module_to_cpu=False,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cudagraphs_reject_a_model_with_pytorch_regions():
    class Model(torch.nn.Module):
        def forward(self, x):
            with region.execute_in_torch():
                x = torch.sin(x)
            return x + 1

    x = torch.randn(2, 8, device="cuda")
    compiled = compile_regions(Model().cuda().eval(), [x])
    with pytest.raises(region.RegionError, match="CUDA graphs do not support"):
        torch_tensorrt.runtime.enable_cudagraphs(compiled)
    # The refused call leaves the CUDA graphs mode as it was.
    assert not torch_tensorrt.runtime.get_cudagraphs_mode()
    assert not torch_tensorrt.runtime.get_whole_cudagraphs_mode()
    with torch.no_grad():
        torch.testing.assert_close(compiled(x), torch.sin(x) + 1, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("retrace", [False, True])
def test_saving_a_model_compiled_with_regions_is_rejected(tmp_path, retrace):
    class Model(torch.nn.Module):
        def forward(self, x):
            with region.execute_in_torch():
                x = torch.sin(x)
            return x + 1

    x = torch.randn(2, 8, device="cuda")
    compiled = compile_regions(Model().cuda().eval(), [x])
    with pytest.raises(
        region.RegionError, match="Saving a model compiled with regions"
    ):
        torch_tensorrt.save(
            compiled, str(tmp_path / "model.pt2"), arg_inputs=[x], retrace=retrace
        )
    # The exporter refuses on its own too, as torch_tensorrt.executorch.export uses it.
    from torch_tensorrt.dynamo._exporter import export

    with pytest.raises(
        region.RegionError, match="Saving a model compiled with regions"
    ):
        export(compiled)
