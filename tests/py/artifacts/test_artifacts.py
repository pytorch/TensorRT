import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "_artifact_test_loader", Path(__file__).parents[1] / "reporting" / "isolation.py"
)
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)
r = loader.load_reporting()
a = loader.load_artifacts()


def test_fingerprints_and_atomic_json(tmp_path):
    assert a.fingerprint({"b": 2, "a": 1}) == a.fingerprint({"a": 1, "b": 2})
    assert a.fingerprint({"a": 1}) != a.fingerprint({"a": 2})
    path = tmp_path / "report.json"
    a.atomic_write_json(path, r.AnalysisReport(job_id="test"))
    assert r.from_json(path.read_text()).job_id == "test"
    before = path.read_bytes()
    with pytest.raises(r.ContractError):
        a.atomic_write_json(path, {"bad": float("nan")})
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_job_directory_manifest_and_create_only_finalization(tmp_path):
    job = a.JobDirectory(tmp_path, "job-1", {"settings": {"precision": "fp16"}})
    assert not job.final_path.exists()
    (job.path / "artifacts" / "model.pt2").write_bytes(b"model")
    a.atomic_write_json(job.path / "analysis.json", r.AnalysisReport(job_id="job-1"))
    result = job.finalize()
    assert result == tmp_path / "job-1"
    manifest = r.ArtifactManifest.from_json((result / "manifest.json").read_text())
    assert a.verify_manifest(manifest, result) == []
    assert {entry.path for entry in manifest.entries} == {
        "request.json",
        "analysis.json",
        "artifacts/model.pt2",
    }
    assert manifest.request_fingerprint == a.fingerprint(
        {"settings": {"precision": "fp16"}}
    )
    with pytest.raises(FileExistsError):
        a.JobDirectory(tmp_path, "job-1")
    with pytest.raises(RuntimeError):
        job.finalize()
    (result / "artifacts" / "model.pt2").write_bytes(b"other")
    assert a.verify_manifest(manifest, result) == ["artifacts/model.pt2: hash mismatch"]


def test_failure_preserves_partial_diagnostics(tmp_path):
    with pytest.raises(r.InvalidInputError, match="invalid settings"):
        with a.JobDirectory(tmp_path, "failed") as job:
            (job.path / "diagnostics" / "failure.log").write_text("partial evidence")
            raise r.InvalidInputError("invalid settings")
    manifest = r.ArtifactManifest.from_json(
        (tmp_path / "failed" / "manifest.json").read_text()
    )
    assert manifest.status == r.ReportStatus.FAILED
    assert manifest.failure.category == r.FailureCategory.INVALID_INPUT
    assert manifest.entries[0].path == "diagnostics/failure.log"


def test_abort_produces_cancelled_retryable_manifest(tmp_path):
    result = a.JobDirectory(tmp_path, "cancelled").abort()
    manifest = r.ArtifactManifest.from_json((result / "manifest.json").read_text())
    assert manifest.status == r.ReportStatus.CANCELLED
    assert manifest.failure.category == r.FailureCategory.CANCELLED
    assert manifest.failure.retry_unchanged
    assert not manifest.failure.retry_after_change


def test_finalization_rejects_existing_empty_destination(tmp_path):
    job = a.JobDirectory(tmp_path, "race")
    job.final_path.mkdir()
    with pytest.raises(FileExistsError):
        job.finalize()
    assert job.final_path.is_dir()
    assert not list(job.final_path.iterdir())


def test_symlinks_are_never_hashed_or_followed(tmp_path):
    outside = tmp_path / "outside"
    outside.write_text("private")
    root = tmp_path / "job"
    root.mkdir()
    (root / "leak").symlink_to(outside)
    with pytest.raises((OSError, ValueError)):
        a.hash_file(root / "leak")
    with pytest.raises(ValueError, match="symlinks"):
        a.build_manifest(root, "job")
    entry = r.ArtifactEntry(
        "leak", "leak", a.hash_file(outside), outside.stat().st_size
    )
    assert a.verify_manifest(r.ArtifactManifest(entries=[entry]), root) == [
        "leak: symlink"
    ]
    with pytest.raises(ValueError, match="symlinks"):
        a.atomic_write_json(root / "leak", {})


@pytest.mark.parametrize("job_id", ["../out", "/out", "", "a/b", "..", "a\\b"])
def test_job_ids_cannot_escape_root(tmp_path, job_id):
    with pytest.raises(ValueError):
        a.JobDirectory(tmp_path, job_id)


def test_manifest_cannot_omit_or_alter_files(tmp_path):
    job = a.JobDirectory(tmp_path, "integrity")
    (job.path / "artifact").write_bytes(b"evidence")
    with pytest.raises(ValueError, match="every file"):
        job.finalize(r.ArtifactManifest(job_id="integrity"))
    manifest = a.build_manifest(job.path, "integrity")
    (job.path / "artifact").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        job.finalize(manifest)


def test_tensor_fingerprints_cover_values_types_and_aliasing():
    torch = pytest.importorskip("torch")
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    assert a.hash_weights({"x": tensor}) == a.hash_weights({"x": tensor.clone()})
    assert a.hash_weights({"x": tensor}) != a.hash_weights({"x": tensor + 1})
    assert a.hash_weights({"x": tensor}) != a.hash_weights({"x": tensor.double()})
    a.hash_weights({"bf16": tensor.bfloat16()})
    assert a.hash_inputs((tensor, tensor)) != a.hash_inputs(
        (tensor.clone(), tensor.clone())
    )
    assert a.hash_inputs((tensor,)) != a.hash_inputs([tensor])
    assert a.hash_inputs((tensor,)) == a.hash_inputs((tensor.clone(),))


def test_graph_fingerprint_changes_with_structure():
    torch = pytest.importorskip("torch")
    left = torch.fx.symbolic_trace(lambda x: x + 1)
    same = torch.fx.symbolic_trace(lambda x: x + 1)
    right = torch.fx.symbolic_trace(lambda x: x + 2)
    assert a.hash_graph(left) == a.hash_graph(same)
    assert a.hash_graph(left) != a.hash_graph(right)


def test_graph_fingerprint_includes_nested_graph_bodies():
    torch = pytest.importorskip("torch")

    def nested(constant):
        child = torch.fx.symbolic_trace(lambda x: x + constant)
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        graph.output(graph.call_module("branch", (x,)))
        return torch.fx.GraphModule({"branch": child}, graph)

    left, right = nested(1), nested(2)
    assert str(left.graph) == str(right.graph)
    assert a.hash_graph(left) != a.hash_graph(right)


def test_environment_discovery_does_not_require_cuda():
    environment = a.collect_environment("env")
    assert environment.job_id == "env"
    assert environment.python_version
    assert r.EnvironmentReport.from_json(environment.to_json()) == environment


def test_environment_distinguishes_build_runtime_api_and_driver_versions(monkeypatch):
    import sys
    from types import SimpleNamespace

    props = SimpleNamespace(
        name="test", uuid="GPU-test", major=8, minor=9, total_memory=1024
    )
    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_properties=lambda index: props,
        cudart=lambda: SimpleNamespace(cudaRuntimeGetVersion=lambda: (0, 12090)),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            __version__="test", version=SimpleNamespace(cuda="12.8"), cuda=cuda
        ),
    )

    def driver_query(pointer):
        pointer._obj.value = 13000
        return 0

    monkeypatch.setattr(
        a.ctypes,
        "CDLL",
        lambda *args, **kwargs: SimpleNamespace(cuDriverGetVersion=driver_query),
    )
    nvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlSystemGetDriverVersion=lambda: b"580.1",
    )
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    environment = a.collect_environment()
    assert environment.torch_cuda_build_version == "12.8"
    assert environment.cuda_runtime_version == "12.9"
    assert environment.cuda_driver_version == "13.0"
    assert environment.nvidia_driver_version == "580.1"
    cuda.cudart = lambda: SimpleNamespace()
    environment = a.collect_environment()
    assert environment.cuda_runtime_version is None
    assert (
        "Runtime API query unavailable"
        in environment.unavailable["cuda_runtime_version"]
    )
    assert environment.torch_cuda_build_version == "12.8"
