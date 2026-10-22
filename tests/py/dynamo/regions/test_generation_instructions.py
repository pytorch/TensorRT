# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import json
import sys
import zipfile
from importlib import resources

import pytest

from torch_tensorrt.dynamo.region_compilation._generation import _instructions
from torch_tensorrt.dynamo.region_compilation._generation._backends import get_backend


@pytest.mark.parametrize("backend", ["triton", "cutile", "cuda_cpp"])
def test_backend_resources_are_selected_without_cwd_dependency(
    monkeypatch, tmp_path, backend
):
    monkeypatch.chdir(tmp_path)
    adapter = get_backend(backend)
    bundle = _instructions.load_instructions(backend, adapter.instruction_resources)
    names = {item["name"] for item in bundle["resources"]}
    assert "SKILL.md" in names
    backend_references = {
        f"references/{name}.md" for name in ("triton", "cutile", "cuda_cpp")
    }
    assert names & backend_references == {f"references/{backend}.md"}
    assert json.loads(json.dumps(bundle)) == bundle


def test_instruction_snapshots_and_policy_digest_follow_content(tmp_path):
    custom = tmp_path / "policy.md"
    custom.write_text("Prefer column tiles.")
    first = _instructions.load_instructions(
        "triton", ("references/triton.md",), "Optimize the reduction.", (str(custom),)
    )
    again = _instructions.load_instructions(
        "triton", ("references/triton.md",), "Optimize the reduction.", (str(custom),)
    )
    assert first["sha256"] == again["sha256"]
    custom.write_text("Prefer row tiles.")
    changed = _instructions.load_instructions(
        "triton", ("references/triton.md",), "Optimize the reduction.", (str(custom),)
    )
    assert first["sha256"] != changed["sha256"]
    assert "Prefer column tiles." in first["text"]
    assert "Prefer row tiles." in changed["text"]
    assert len({item["name"] for item in first["resources"]}) == len(first["resources"])


@pytest.mark.parametrize("instructions", ["", " \n\t"])
def test_blank_inline_instructions_are_optional(instructions):
    assert _instructions.load_instructions(
        "triton", instructions=instructions
    ) == _instructions.load_instructions("triton", instructions=None)


@pytest.mark.parametrize("name", ["../outside.md", "/tmp/outside.md", r"..\outside.md"])
def test_package_resources_cannot_escape(name):
    with pytest.raises(ValueError, match="package-relative"):
        _instructions.load_instructions("custom", (name,))


def test_instruction_size_is_bounded(tmp_path):
    custom = tmp_path / "large.md"
    custom.write_bytes(b"x" * (_instructions._MAX_RESOURCE_BYTES + 1))
    with pytest.raises(ValueError, match="256 KiB"):
        _instructions.load_instructions("triton", instruction_files=(str(custom),))


def test_packaged_resources_can_be_loaded_from_a_zip(monkeypatch, tmp_path):
    """The loader consumes package resources, not paths relative to the checkout."""
    package = "kernel_instruction_fixture"
    archive = tmp_path / "instructions.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(f"{package}/__init__.py", "")
        stream.writestr(f"{package}/backend.md", "Use the fixture backend contract.")
    monkeypatch.syspath_prepend(str(archive))
    importlib.invalidate_caches()
    try:
        bundle = _instructions.load_instructions("fixture", (f"{package}:backend.md",))
        assert bundle["resources"][-1]["text"] == "Use the fixture backend contract."
        assert bundle["resources"][-1]["origin"] == "package"
    finally:
        sys.modules.pop(package, None)


def test_builtins_load_through_resource_api(monkeypatch, tmp_path):
    archive = tmp_path / "builtin.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        for name in _instructions._COMMON_RESOURCES:
            stream.writestr(f"skills/kernel-author/{name}", f"Resource {name}")
    with zipfile.ZipFile(archive) as stream:
        root = zipfile.Path(stream)
        monkeypatch.setattr(resources, "files", lambda _: root)
        bundle = _instructions.load_instructions("test")
    assert len(bundle["resources"]) == 3
    assert "Resource SKILL.md" in bundle["text"]


def test_evidence_pages_reconstruct_exact_unicode_text():
    catalog = _instructions.EvidenceCatalog(page_chars=5)
    text = "First αβγ\nSecond line\n"
    catalog.add("attempt-1/compiler", text)
    offset = 0
    pages = []
    while True:
        page = catalog.read("attempt-1/compiler", offset)
        assert len(page["text"]) <= 5
        assert page["sha256"] == catalog.catalog()[0]["sha256"]
        pages.append(page["text"])
        if page["next_offset"] is None:
            break
        offset = page["next_offset"]
    assert "".join(pages) == text


def test_evidence_resources_are_immutable_and_unknown_paths_are_rejected(tmp_path):
    catalog = _instructions.EvidenceCatalog()
    catalog.add("task", "first")
    catalog.add("task", "first")
    with pytest.raises(ValueError, match="already registered"):
        catalog.add("task", "replacement")
    path = tmp_path / "secret.txt"
    path.write_text("outside evidence")
    with pytest.raises(ValueError, match="Unknown evidence"):
        catalog.read(str(path))
    assert catalog.read("task")["text"] == "first"
    metadata = catalog.catalog()
    metadata.clear()
    assert len(catalog.catalog()) == 1


@pytest.mark.parametrize("offset", [-1, 4, True, "0", None])
def test_evidence_rejects_invalid_offsets(offset):
    catalog = _instructions.EvidenceCatalog()
    catalog.add("task", "abc")
    with pytest.raises(ValueError, match="offset"):
        catalog.read("task", offset)


@pytest.mark.parametrize("size", [0, -1, 65537, True, 1.5])
def test_evidence_rejects_invalid_page_limits(size):
    with pytest.raises(ValueError, match="page_chars"):
        _instructions.EvidenceCatalog(page_chars=size)
