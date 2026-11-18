# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import errno
import gc
import inspect
import json
import logging
import os
import platform
from types import SimpleNamespace

import pytest
import torch
import torch_tensorrt
from safetensors.torch import save_file
from torch.multiprocessing.reductions import StorageWeakRef
from torch_tensorrt._utils import _cache_root
from torch_tensorrt.dynamo import _low_memory, low_memory
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.tools import map_weights, weight_mapping
from torch_tensorrt.dynamo.tools.weight_mapping import release_mapped_storage

needs_file_deletion = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="low_memory raises NotImplementedError on Windows",
)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 8)
        self.register_buffer("scale", torch.full((8,), 2.0), persistent=False)

    def forward(self, x):
        return self.proj(x) * self.scale


def _loaded():
    torch.manual_seed(0)
    return _Model().eval()


def _tensors(model):
    return [*model.parameters(), *model.buffers()]


def test_compile_settings_hold_the_given_settings_and_the_defaults():
    assert dict(low_memory().compile_settings) == {
        "offload_module_to_cpu": True,
        "enable_resource_partitioning": True,
    }
    settings = low_memory(
        offload_module_to_cpu=False,
        cpu_memory_budget=3 << 30,
        release_weights_after_build=0,
    ).compile_settings
    assert dict(settings) == {
        "offload_module_to_cpu": False,
        "enable_resource_partitioning": True,
        "cpu_memory_budget": 3 << 30,
        "release_weights_after_build": False,
    }
    assert low_memory(max_partition_size=1 << 29).compile_settings[
        "max_partition_size"
    ] == (1 << 29)
    assert (
        CompilationSettings(release_weights_after_build=0).release_weights_after_build
        is False
    )


def test_model_weights_on_meta_keeps_offload_module_to_cpu_on(caplog):
    with caplog.at_level(logging.WARNING, logger="torch_tensorrt.dynamo._low_memory"):
        block = low_memory(model_weights_on_meta=True, offload_module_to_cpu=False)
    assert block.compile_settings["offload_module_to_cpu"] is True
    assert "device mismatch" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="torch_tensorrt.dynamo._low_memory"):
        block = low_memory(model_weights_on_meta=True)
    assert block.compile_settings["offload_module_to_cpu"] is True
    assert not caplog.records


def test_the_settings_are_compile_settings():
    # compile takes **kwargs, so a renamed setting would be ignored there, not rejected.
    block = {
        name
        for name, p in inspect.signature(low_memory).parameters.items()
        if p.kind is p.KEYWORD_ONLY and name != "model_weights_on_meta"
    }
    assert block and block <= set(
        inspect.signature(torch_tensorrt.dynamo.compile).parameters
    )


def test_bad_arguments_raise():
    with pytest.raises(TypeError, match="optimisation_level"):
        low_memory(optimisation_level=1)
    # Only settings that lower compilation memory: the rest go to compile itself.
    with pytest.raises(TypeError, match="min_block_size"):
        low_memory(min_block_size=1)
    # low_memory(...) takes no positional arguments (the model and checkpoint go to low_mem.map_weights).
    with pytest.raises(TypeError, match="positional"):
        low_memory(_loaded())
    with pytest.raises(TypeError, match="positional"):
        low_memory(_Model, "w.safetensors")
    with pytest.raises(TypeError, match="checkpoint"):
        low_memory(checkpoint="w.safetensors")


@needs_file_deletion
def test_checkpoint_key_without_checkpoint_raises():
    model = _loaded()
    with pytest.raises(ValueError, match="pass checkpoint= too"):
        with low_memory() as low_mem:
            low_mem.map_weights(model, checkpoint_key=str)


def test_map_weights_on_windows_raises_while_settings_only_block_works(monkeypatch):
    monkeypatch.setattr(_low_memory.platform, "system", lambda: "Windows")
    with low_memory() as low_mem:
        model = _Model().eval()
        assert not model.proj.weight.is_meta
        assert low_mem.compile_settings["offload_module_to_cpu"] is True
        with pytest.raises(NotImplementedError, match="Windows can't delete"):
            low_mem.map_weights(model)


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_compile_inside_low_memory_without_map_weights():
    x = torch.randn(3, 8)
    # 1. Default model_weights_on_meta=False: construct/load and compile a normal model inside the block without map_weights.
    with low_memory() as low_mem:
        torch.manual_seed(0)
        inside_model = _Model().eval()
        assert not inside_model.proj.weight.is_meta
        with torch.no_grad():
            expected_inside = inside_model(x)
        trt_inside = torch_tensorrt.dynamo.compile(
            torch.export.export(inside_model, (x,)),
            inputs=[x.cuda()],
            min_block_size=1,
            **low_mem.compile_settings,
        )
    torch.testing.assert_close(
        trt_inside(x.cuda()).cpu(), expected_inside, rtol=1e-3, atol=1e-3
    )

    # 2. model_weights_on_meta=True: a model already built outside the block still compiles inside it
    #    without calling map_weights (compile automatically stops meta parameter placement).
    outside_model = _loaded()
    with torch.no_grad():
        expected_outside = outside_model(x)
    with low_memory(model_weights_on_meta=True) as low_mem:
        trt_outside = torch_tensorrt.dynamo.compile(
            torch.export.export(outside_model, (x,)),
            inputs=[x.cuda()],
            min_block_size=1,
            **low_mem.compile_settings,
        )
    torch.testing.assert_close(
        trt_outside(x.cuda()).cpu(), expected_outside, rtol=1e-3, atol=1e-3
    )


@needs_file_deletion
def test_creating_a_model_inside_the_block_places_only_parameters_on_meta_until_mapped(
    tmp_path,
):
    class TiedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Linear(8, 8, bias=False)
            self.head = torch.nn.Linear(8, 8, bias=False)
            self.head.weight = self.embed.weight
            self.register_buffer(
                "inv_freq", torch.arange(8, dtype=torch.float32), persistent=False
            )

    ref = TiedModel().eval()
    save_file({"embed.weight": ref.embed.weight}, str(tmp_path / "model.safetensors"))

    with low_memory(model_weights_on_meta=True) as low_mem:
        model = TiedModel().eval()
        # Parameters land on meta and stay tied; buffers stay real.
        assert model.embed.weight.is_meta
        assert model.head.weight is model.embed.weight
        assert not model.inv_freq.is_meta
        model.to(torch.bfloat16)  # dtype tweak while still on meta
        assert model.embed.weight.is_meta and model.embed.weight.dtype == torch.bfloat16

        low_mem.map_weights(model, checkpoint=tmp_path)
        assert not model.embed.weight.is_meta
        assert model.embed.weight.dtype == torch.bfloat16
        assert model.head.weight is model.embed.weight
        # Meta parameter registration is now OFF for subsequent modules (such as compilation).
        after = torch.nn.Linear(4, 4)
        assert not after.weight.is_meta


@needs_file_deletion
def test_exception_before_map_weights_restores_normal_parameter_registration():
    with pytest.raises(RuntimeError, match="early boom"):
        with low_memory(model_weights_on_meta=True):
            inside = torch.nn.Linear(4, 4)
            assert inside.weight.is_meta
            raise RuntimeError("early boom")
    outside = torch.nn.Linear(4, 4)
    assert not outside.weight.is_meta


@needs_file_deletion
def test_from_pretrained_warns_that_feature_is_not_effective(caplog):
    class PretrainedDummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(4, 4)
            self.l2 = torch.nn.Linear(4, 4)

        @classmethod
        def from_config(cls):
            return cls()

        @classmethod
        def from_pretrained(cls, path):
            instance = cls()
            instance.name_or_path = path
            return instance

    # 1. Constructing via from_config under model_weights_on_meta=True emits no warning.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="torch_tensorrt.dynamo._low_memory"):
        with low_memory(model_weights_on_meta=True):
            PretrainedDummy.from_config()
    assert "from_pretrained" not in caplog.text

    # 2. Calling from_pretrained inside low_memory(model_weights_on_meta=True) warns once per class.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="torch_tensorrt.dynamo._low_memory"):
        with low_memory(model_weights_on_meta=True):
            PretrainedDummy.from_pretrained("some/repo")
    warnings = [r for r in caplog.records if "not effective" in r.message]
    assert len(warnings) == 1
    assert "PretrainedDummy.from_pretrained" in warnings[0].message
    assert "from_config" in warnings[0].message

    # 3. Passing a model already loaded via from_pretrained to map_weights also warns.
    caplog.clear()
    loaded = PretrainedDummy.from_pretrained("some/repo").eval()
    with caplog.at_level(logging.WARNING, logger="torch_tensorrt.dynamo._low_memory"):
        with low_memory() as low_mem:
            low_mem.map_weights(loaded)
    assert any("from_config" in r.message for r in caplog.records)


@needs_file_deletion
def test_huggingface_config_source_is_used_when_checkpoint_is_omitted(tmp_path):
    reference = _loaded()
    save_file(reference.state_dict(), str(tmp_path / "model.safetensors"))
    x = torch.randn(3, 8)
    with torch.no_grad():
        expected = reference(x)

    with low_memory(model_weights_on_meta=True) as low_mem:
        model = _Model().eval()
        model.config = SimpleNamespace(_name_or_path=str(tmp_path))
        assert model.proj.weight.is_meta and not model.scale.is_meta
        low_mem.map_weights(model)
        with torch.no_grad():
            torch.testing.assert_close(model(x), expected, rtol=0, atol=0)


@needs_file_deletion
def test_hub_repo_id_downloads_single_and_sharded_checkpoints(tmp_path, monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    from huggingface_hub.errors import EntryNotFoundError

    reference = _loaded()
    single_dir = tmp_path / "single"
    single_dir.mkdir()
    single_file = single_dir / "model.safetensors"
    save_file(reference.state_dict(), str(single_file))

    sharded_dir = tmp_path / "sharded"
    sharded_dir.mkdir()
    save_file(
        {"proj.weight": reference.proj.weight}, str(sharded_dir / "s1.safetensors")
    )
    save_file({"proj.bias": reference.proj.bias}, str(sharded_dir / "s2.safetensors"))
    index_file = sharded_dir / "model.safetensors.index.json"
    index_file.write_text(
        json.dumps(
            {
                "weight_map": {
                    "proj.weight": "s1.safetensors",
                    "proj.bias": "s2.safetensors",
                }
            }
        )
    )

    calls = []

    def fake_try_cache(repo_id, filename, **kwargs):
        return None

    def fake_download(repo_id, filename, **kwargs):
        calls.append((repo_id, filename))
        if repo_id == "org/single-model":
            if filename == "model.safetensors.index.json":
                raise EntryNotFoundError("no index")
            assert filename == "model.safetensors"
            return str(single_file)
        if repo_id == "org/sharded-model":
            return str(sharded_dir / filename)
        raise AssertionError(repo_id)

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", fake_try_cache)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    for repo_id in ("org/single-model", "org/sharded-model"):
        with low_memory(model_weights_on_meta=True) as low_mem:
            model = _Model().eval()
            low_mem.map_weights(model, checkpoint=repo_id)
            torch.testing.assert_close(
                model.proj.weight, reference.proj.weight, rtol=0, atol=0
            )
            torch.testing.assert_close(
                model.proj.bias, reference.proj.bias, rtol=0, atol=0
            )

    assert ("org/sharded-model", "s1.safetensors") in calls
    assert ("org/sharded-model", "s2.safetensors") in calls
    assert not weight_mapping._is_hub_repo_id(str(tmp_path / "missing.safetensors"))


@needs_file_deletion
def test_mapping_a_loaded_model_moves_weights_into_a_file_and_frees_the_loaded_copy(
    caplog,
):
    model = _loaded()
    x = torch.randn(3, 8)
    with torch.no_grad():
        expected = model(x)
    loaded = [StorageWeakRef(t.untyped_storage()) for t in _tensors(model)]
    with caplog.at_level(logging.WARNING, logger="torch_tensorrt.dynamo._low_memory"):
        with low_memory() as low_mem:
            low_mem.map_weights(model)
            assert low_mem.model is model
            assert low_mem.weights_file.parent == _cache_root() / "offloaded_weights"
            assert low_mem.weights_file.is_file()
            gc.collect()
            assert all(ref.expired() for ref in loaded)
            # Every parameter and buffer, the non-persistent one too, is a view of the file.
            assert all(
                release_mapped_storage(t.untyped_storage()) is not None
                for t in _tensors(model)
            )
            assert low_mem.report.mapped_bytes == sum(t.nbytes for t in _tensors(model))
            with torch.no_grad():
                torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
    assert "still in memory" not in caplog.text


@needs_file_deletion
def test_the_file_is_deleted_once_the_model_is():
    model = _loaded()
    with low_memory() as low_mem:
        low_mem.map_weights(model)
        path = low_mem.weights_file
    assert path.is_file()  # the model still reads it after the block
    model(torch.randn(3, 8))
    del model, low_mem
    gc.collect()
    assert not path.exists()


@needs_file_deletion
def test_weights_held_outside_the_model_are_reported(caplog):
    model = torch.nn.Sequential(
        torch.nn.Linear(512, 512), torch.nn.ReLU(), torch.nn.Linear(512, 512)
    )
    held = model(
        torch.randn(3, 512)
    )  # gradients on: the graph holds the last layer's weight
    with caplog.at_level(logging.WARNING, logger="torch_tensorrt.dynamo._low_memory"):
        with low_memory() as low_mem:
            low_mem.map_weights(model)
    assert "1.0 MiB of the model's loaded weights are still in memory" in caplog.text
    del held


@needs_file_deletion
def test_an_expanded_buffer_is_not_reported(caplog):
    # SigLIP's position ids: a view of an arange, which keeps the arange.
    model = _loaded()
    model.register_buffer(
        "position_ids", torch.arange(8).expand((1, -1)), persistent=False
    )
    with caplog.at_level(logging.WARNING, logger="torch_tensorrt.dynamo._low_memory"):
        with low_memory() as low_mem:
            low_mem.map_weights(model)
            assert (
                release_mapped_storage(model.position_ids.untyped_storage()) is not None
            )
            assert torch.equal(model.position_ids, torch.arange(8).expand((1, -1)))
    assert "still in memory" not in caplog.text


@needs_file_deletion
def test_a_model_on_meta_without_a_checkpoint_raises():
    with pytest.raises(ValueError, match="meta device.*proj.weight.*checkpoint="):
        with low_memory(model_weights_on_meta=True) as low_mem:
            model = _Model().eval()
            low_mem.map_weights(model)


@needs_file_deletion
def test_a_model_on_meta_reads_its_weights_from_the_checkpoint(tmp_path):
    reference = _loaded()
    save_file(reference.state_dict(), str(tmp_path / "model.safetensors"))  # float32
    x = torch.randn(3, 8)
    # The folder, as Hugging Face saves a checkpoint; each tensor is cast to the model's dtype.
    with low_memory(model_weights_on_meta=True) as low_mem:
        model = _Model().to(torch.bfloat16).eval()
        low_mem.map_weights(model, checkpoint=tmp_path)
        assert not any(t.is_meta for t in _tensors(model))
        assert model.proj.weight.dtype == torch.bfloat16
        assert low_mem.weights_file.parent == _cache_root() / "offloaded_weights"
        with torch.no_grad():
            torch.testing.assert_close(
                model(x.bfloat16()).float(), reference(x), rtol=2e-2, atol=2e-2
            )
    assert os.listdir(tmp_path) == ["model.safetensors"]  # only read


@needs_file_deletion
def test_checkpoint_key_gives_the_checkpoints_names(tmp_path):
    reference = _loaded()
    whole_model = {f"model.{k}": v for k, v in reference.state_dict().items()}
    save_file(whole_model, str(tmp_path / "whole.safetensors"))
    with low_memory(model_weights_on_meta=True) as low_mem:
        model = _Model().eval()
        low_mem.map_weights(
            model,
            tmp_path / "whole.safetensors",
            checkpoint_key=lambda name: f"model.{name}",
        )
        torch.testing.assert_close(
            model.proj.weight, reference.proj.weight, rtol=0, atol=0
        )


@needs_file_deletion
def test_a_tensor_on_meta_the_checkpoint_does_not_store_raises(tmp_path):
    save_file(_loaded().state_dict(), str(tmp_path / "model.safetensors"))
    # All tensors on meta, including the non-persistent scale buffer, which no checkpoint stores.
    with torch.device("meta"):
        model = _Model().eval()
    with pytest.raises(
        ValueError, match="scale is not in the source.*init_empty_weights"
    ):
        with low_memory() as low_mem:
            low_mem.map_weights(model, checkpoint=tmp_path / "model.safetensors")


@needs_file_deletion
def test_a_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No safetensors checkpoint"):
        with low_memory(model_weights_on_meta=True) as low_mem:
            model = _Model().eval()
            low_mem.map_weights(model, checkpoint=tmp_path / "missing.safetensors")


@pytest.mark.unit
@needs_file_deletion
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_compile_a_model_built_inside_the_block_from_its_checkpoint(tmp_path):
    reference = _loaded()
    save_file(reference.state_dict(), str(tmp_path / "model.safetensors"))
    x = torch.randn(3, 8)
    with torch.no_grad():
        expected = reference(x)
    with low_memory(model_weights_on_meta=True) as low_mem:
        model = _Model().eval()
        low_mem.map_weights(model, checkpoint=tmp_path / "model.safetensors")
        trt_model = torch_tensorrt.dynamo.compile(
            torch.export.export(model, (x,)),
            inputs=[x.cuda()],
            min_block_size=1,
            **low_mem.compile_settings,
        )
    torch.testing.assert_close(
        trt_model(x.cuda()).cpu(), expected, rtol=1e-3, atol=1e-3
    )


@needs_file_deletion
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_model_on_the_gpu_moves_to_the_cpu():
    model = _loaded().cuda()
    x = torch.randn(3, 8, device="cuda")
    with torch.no_grad():
        expected = model(x)
    with low_memory() as low_mem:
        low_mem.map_weights(model)
        assert not any(t.is_cuda for t in _tensors(model))
        with torch.no_grad():
            torch.testing.assert_close(model(x.cpu()), expected.cpu())


@needs_file_deletion
def test_a_failed_write_names_tmpdir_and_leaves_the_model_loaded(monkeypatch):
    model = _loaded()
    weight = model.proj.weight.untyped_storage().data_ptr()

    def disk_full(module, path, source=None):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(_low_memory, "save_weights_for_mapping", disk_full)
    with pytest.raises(OSError, match="No space left on device.*TMPDIR"):
        with low_memory() as low_mem:
            low_mem.map_weights(model)
    assert model.proj.weight.untyped_storage().data_ptr() == weight


@needs_file_deletion
def test_a_failed_mapping_deletes_the_file(monkeypatch):
    model = _loaded()
    paths = []
    new_path = _low_memory._new_weights_file_path
    monkeypatch.setattr(
        _low_memory,
        "_new_weights_file_path",
        lambda: paths.append(new_path()) or paths[-1],
    )

    def fail(module, path):
        assert os.path.isfile(path)
        raise RuntimeError("cannot map")

    monkeypatch.setattr(_low_memory, "map_weights", fail)
    with pytest.raises(RuntimeError, match="cannot map"):
        with low_memory() as low_mem:
            low_mem.map_weights(model)
    assert paths and not paths[0].exists()


@needs_file_deletion
def test_files_of_processes_that_no_longer_run_are_deleted():
    model = _loaded()
    directory = _cache_root() / "offloaded_weights"
    directory.mkdir(parents=True, exist_ok=True)
    # No process can have an id above the kernel's limit.
    gone = int(open("/proc/sys/kernel/pid_max").read()) + 1
    left_behind = [
        directory / f"weights_{gone}_test.safetensors",
        directory / f"weights_{gone}_test.safetensors.0123.partial",
    ]
    mine = directory / f"weights_{os.getpid()}_test.safetensors"
    for path in [*left_behind, mine]:
        path.write_bytes(b"\0")
    try:
        with low_memory() as low_mem:
            low_mem.map_weights(model)
        assert not any(path.exists() for path in left_behind)
        assert mine.exists()
    finally:
        for path in [*left_behind, mine]:
            path.unlink(missing_ok=True)


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_weight_the_graph_never_reads_may_stay_on_meta(tmp_path):
    class WithUnusedHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)
            self.head = torch.nn.Linear(8, 4)  # never called in forward

        def forward(self, x):
            return torch.relu(self.proj(x))

    torch.manual_seed(0)
    reference = WithUnusedHead().eval()
    proj_only = {
        k: v for k, v in reference.state_dict().items() if k.startswith("proj.")
    }
    save_file(proj_only, str(tmp_path / "proj.safetensors"))
    with torch.device("meta"):
        model = WithUnusedHead().eval()
    map_weights(model, tmp_path / "proj.safetensors", strict=False)
    assert model.head.weight.is_meta
    x = torch.randn(3, 8)
    trt_model = torch_tensorrt.dynamo.compile(
        torch.export.export(model, (x,)), inputs=[x.cuda()], min_block_size=1
    )
    torch.testing.assert_close(
        trt_model(x.cuda()).cpu(), reference(x), rtol=1e-3, atol=1e-3
    )


@needs_file_deletion
def test_leaving_drops_the_pages_also_on_an_exception(monkeypatch):
    model = _loaded()
    weight = model.proj.weight.detach().clone()
    dropped = []

    def spy(storage):
        dropped.append(release_mapped_storage(storage))
        return dropped[-1]

    monkeypatch.setattr(_low_memory, "release_mapped_storage", spy)
    low_mem = low_memory()
    with pytest.raises(RuntimeError, match="boom"):
        with low_mem:
            low_mem.map_weights(model)
            raise RuntimeError("boom")
    assert dropped and None not in dropped  # every one a mapped weight
    torch.testing.assert_close(low_mem.model.proj.weight, weight)


@needs_file_deletion
def test_a_block_that_keeps_weights_keeps_their_pages(monkeypatch):
    model = _loaded()
    dropped = []
    monkeypatch.setattr(_low_memory, "release_mapped_storage", dropped.append)
    with low_memory(release_weights_after_build=False) as low_mem:
        low_mem.map_weights(model)
    assert not dropped


def test_a_program_with_weights_on_meta_does_not_compile():
    with torch.device("meta"):
        model = _Model().eval()
    exported = torch.export.export(model, (torch.randn(3, 8, device="meta"),))
    with pytest.raises(ValueError, match="meta device.*proj.weight"):
        torch_tensorrt.dynamo.compile(
            exported, inputs=[torch.randn(3, 8)], min_block_size=1
        )


def test_a_dry_run_accepts_weights_on_meta():
    with torch.device("meta"):
        model = _Model().eval()
    exported = torch.export.export(model, (torch.randn(3, 8, device="meta"),))
    torch_tensorrt.dynamo.compile(
        exported, inputs=[torch.randn(3, 8)], min_block_size=1, dryrun=True
    )


@pytest.mark.unit
@needs_file_deletion
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_compile_with_the_block_settings(monkeypatch):
    released = []
    real = torch_tensorrt.dynamo._weight_release.release_mapped_storage

    def spy(storage):
        released.append(real(storage))
        return released[-1]

    monkeypatch.setattr(
        torch_tensorrt.dynamo._weight_release, "release_mapped_storage", spy
    )
    x = torch.randn(3, 8)
    model = _loaded()
    with torch.no_grad():
        expected = model(x)
    with low_memory() as low_mem:
        low_mem.map_weights(model)
        exported = torch.export.export(model, (x,))
        trt_model = torch_tensorrt.compile(
            exported,
            ir="dynamo",
            inputs=[x.cuda()],
            min_block_size=1,
            **low_mem.compile_settings,
        )
    torch.testing.assert_close(
        trt_model(x.cuda()).cpu(), expected, rtol=1e-3, atol=1e-3
    )
    # By default, mapped weights' pages are dropped.
    assert released and None not in released
    with torch.no_grad():
        torch.testing.assert_close(low_mem.model(x), expected)

    # With release_weights_after_build=False in the settings passed to compile, none are.
    released.clear()
    model_keep = _loaded()
    with low_memory(release_weights_after_build=False) as low_mem:
        low_mem.map_weights(model_keep)
        exported = torch.export.export(model_keep, (x,))
        torch_tensorrt.compile(
            exported,
            ir="dynamo",
            inputs=[x.cuda()],
            min_block_size=1,
            **low_mem.compile_settings,
        )
    assert not released

    # An ordinary model's weights are left alone by default.
    plain = _Model().eval()
    exported = torch.export.export(plain, (x,))
    torch_tensorrt.dynamo.compile(exported, inputs=[x.cuda()], min_block_size=1)
    assert plain.proj.weight.untyped_storage().nbytes() > 0


@pytest.mark.unit
@needs_file_deletion
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_the_block_does_not_change_other_compiles():
    """A model compiled inside the block without its settings keeps its own: a GPU model isn't offloaded."""
    loaded = _loaded()
    gpu_model = _Model().eval().cuda()
    x = torch.randn(3, 8, device="cuda")
    with low_memory() as low_mem:
        low_mem.map_weights(loaded)
        torch_tensorrt.dynamo.compile(
            torch.export.export(gpu_model, (x,)),
            inputs=[x],
            min_block_size=1,
            torch_executed_ops={torch.ops.aten.mul.Tensor},
        )
    assert all(p.is_cuda for p in gpu_model.parameters())
    gpu_model(x)


@pytest.mark.unit
@needs_file_deletion
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_pytorch_part_reading_a_mapped_weight_runs():
    x = torch.randn(3, 8)
    model = _loaded()
    with torch.no_grad():
        expected = model(x)
    with low_memory() as low_mem:
        low_mem.map_weights(model)
        trt_model = torch_tensorrt.dynamo.compile(
            torch.export.export(model, (x,)),
            inputs=[x.cuda()],
            min_block_size=1,
            torch_executed_ops={torch.ops.aten.mul.Tensor},  # reads the mapped scale
            **low_mem.compile_settings,
        )
    torch.testing.assert_close(
        trt_model(x.cuda()).cpu(), expected, rtol=1e-3, atol=1e-3
    )


def _resident_mib(path):
    """MiB of the file at `path` resident in this process: its mappings' Rss in /proc/self/smaps."""
    total, in_file = 0, False
    for line in open("/proc/self/smaps"):
        fields = line.split()
        if fields and "-" in fields[0] and len(fields) >= 5:  # a mapping's header line
            in_file = fields[-1] == path
        elif in_file and fields[0] == "Rss:":
            total += int(fields[1])
    return total / 1024


class _Big(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(512, 512, bias=False)  # 1 MiB of float32 weights

    def forward(self, x):
        return self.proj(x)


@pytest.mark.unit
@needs_file_deletion
@pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.exists("/proc/self/smaps"),
    reason="needs CUDA and /proc/self/smaps",
)
def test_the_weights_file_leaves_memory(tmp_path):
    save_file(_Big().eval().state_dict(), str(tmp_path / "model.safetensors"))
    x = torch.randn(3, 512)
    with low_memory(model_weights_on_meta=True) as low_mem:
        model = _Big().eval()
        low_mem.map_weights(model, checkpoint=tmp_path)
        path = os.path.realpath(low_mem.weights_file)
        with torch.no_grad():
            model(x)  # reads every weight in
        assert _resident_mib(path) >= 0.9
        torch_tensorrt.dynamo.compile(
            torch.export.export(model, (x,)),
            inputs=[x.cuda()],
            min_block_size=1,
            **low_mem.compile_settings,
        )
        # Read inside the block: leaving it drops the pages too, so a reading after it can't tell that
        # compile released them. Only the pages at the weight's ends, which may hold a neighbour's data,
        # can stay.
        assert _resident_mib(path) < 0.05
