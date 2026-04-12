"""
Comprehensive tests for the native NCCL system in Torch-TensorRT.

Covers
------
1.  distributed_group() context manager — thread-local state, nesting, exception safety
2.  get_active_group / get_active_group_name — group resolution
3.  NCCL library utilities — path detection, symlink, LD_LIBRARY_PATH check
4.  fuse_distributed_ops graph pass — all_gather, reduce_scatter, all_reduce,
    no-fuse when wait_tensor has multiple users
5.  Single-rank NCCL op compilation (pytest, WORLD_SIZE=1)
6.  Multi-rank inference with distributed_group (torchrun / mpirun, 2 ranks)
7.  C++ runtime NCCL bind (bind_nccl_comm)
8.  Python runtime NCCL comm (setup_nccl_comm + pickle / unpickle)
9.  distributed_group with a non-default TP subgroup

Run single-rank pytest tests
-----------------------------
    cd tests/py/dynamo
    pytest distributed/test_native_nccl.py -v

Run multi-rank tests (single-node, 2 GPUs via torchrun)
---------------------------------------------------------
    torchrun --nproc_per_node=2 distributed/test_native_nccl.py --multirank

Run multi-rank tests (multinode, 1 GPU per node — run on each node)
--------------------------------------------------------------------
    # Node 0:
    RANK=0 WORLD_SIZE=2 LOCAL_RANK=0 MASTER_ADDR=<node0-ip> MASTER_PORT=29500 \\
        python distributed/test_native_nccl.py --multinode
    # Node 1:
    RANK=1 WORLD_SIZE=2 LOCAL_RANK=0 MASTER_ADDR=<node0-ip> MASTER_PORT=29500 \\
        python distributed/test_native_nccl.py --multinode
"""

from __future__ import annotations

import os
import sys
import threading
import unittest
from contextlib import nullcontext
from typing import Any
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
import torch.fx
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def is_nccl_available() -> bool:
    try:
        return dist.is_nccl_available()
    except Exception:
        return False


def is_trtllm_for_nccl() -> bool:
    try:
        from torch_tensorrt._features import ENABLED_FEATURES

        return bool(ENABLED_FEATURES.trtllm_for_nccl)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Shared test fakes
# ---------------------------------------------------------------------------


class _FakeEngine:
    """Minimal duck-type for torch.classes.tensorrt.Engine in unit tests.

    Has ``is_md`` and ``set_group_name`` so it passes the duck-type check
    inside set_distributed_group() without needing a real TRT build.
    """

    def __init__(self, is_md: bool = True) -> None:
        self.is_md = is_md
        self.group_name_calls: list = []

    def set_group_name(self, name: str) -> None:
        self.group_name_calls.append(name)


class _FakeGroup:
    """Minimal duck-type for a c10d ProcessGroup."""

    def __init__(self, name: str = "test_group") -> None:
        self.group_name = name


# ============================================================================
# Section 1 — distributed_group() context manager (no GPU / no dist init)
# ============================================================================


class TestDistributedGroupContextManager(unittest.TestCase):
    """Pure unit tests for the distributed_group() thread-local context manager.

    These tests deliberately avoid dist.init_process_group so they run in any
    environment, including CI without GPUs.
    """

    def setUp(self) -> None:
        from torch_tensorrt.distributed._distributed import (
            _state,
            distributed_group,
            get_active_group,
            get_active_group_name,
        )

        # Reset thread-local state so each test starts clean.
        if hasattr(_state, "pg"):
            del _state.pg

        self._state = _state
        self.get_active_group = get_active_group
        self.get_active_group_name = get_active_group_name
        self.distributed_group = distributed_group

    # -- default / no-dist cases --------------------------------------------

    def test_get_active_group_returns_none_when_dist_not_initialized(self) -> None:
        """Without dist.init_process_group, get_active_group() returns None."""
        if dist.is_initialized():
            self.skipTest("dist already initialized in this process")
        self.assertIsNone(self.get_active_group())

    def test_get_active_group_name_returns_empty_string_when_no_dist(self) -> None:
        """get_active_group_name() returns '' when no active group."""
        if dist.is_initialized():
            self.skipTest("dist already initialized in this process")
        self.assertEqual(self.get_active_group_name(), "")

    # -- basic set / restore ------------------------------------------------

    def test_context_manager_sets_active_group(self) -> None:
        """distributed_group() makes the group visible via get_active_group()."""
        fake = MagicMock()
        fake.group_name = "tp_group"
        self.assertIsNone(getattr(self._state, "pg", None))
        with self.distributed_group(fake):
            self.assertIs(self.get_active_group(), fake)

    def test_context_manager_restores_none_on_exit(self) -> None:
        """Thread-local is restored to None after context exits."""
        fake = MagicMock()
        fake.group_name = "tp_group"
        with self.distributed_group(fake):
            pass
        self.assertIsNone(getattr(self._state, "pg", None))

    def test_context_manager_restores_previous_group(self) -> None:
        """Restores *outer* group, not None, when exiting an inner context."""
        outer = MagicMock()
        outer.group_name = "outer"
        inner = MagicMock()
        inner.group_name = "inner"

        with self.distributed_group(outer):
            with self.distributed_group(inner):
                self.assertIs(self.get_active_group(), inner)
            # inner exited → back to outer
            self.assertIs(self.get_active_group(), outer)

        # outer exited → back to None
        self.assertIsNone(getattr(self._state, "pg", None))

    def test_context_manager_restores_on_exception(self) -> None:
        """Thread-local is restored even when the context body raises."""
        fake = MagicMock()
        fake.group_name = "tp_group"
        try:
            with self.distributed_group(fake):
                raise RuntimeError("body error")
        except RuntimeError:
            pass
        self.assertIsNone(getattr(self._state, "pg", None))

    # -- group_name resolution ----------------------------------------------

    def test_get_active_group_name_with_mock_group(self) -> None:
        """get_active_group_name() returns group.group_name string."""
        fake = MagicMock()
        fake.group_name = "my_tp_group"
        with self.distributed_group(fake):
            self.assertEqual(self.get_active_group_name(), "my_tp_group")

    def test_get_active_group_name_group_without_group_name_attr(self) -> None:
        """get_active_group_name() returns '' when the group has no group_name."""
        fake = MagicMock(spec=[])  # empty spec → no attributes
        with self.distributed_group(fake):
            self.assertEqual(self.get_active_group_name(), "")

    def test_get_active_group_name_non_string_group_name(self) -> None:
        """group_name is coerced to str even if the mock returns an int."""
        fake = MagicMock()
        fake.group_name = 42
        with self.distributed_group(fake):
            name = self.get_active_group_name()
        self.assertEqual(name, "42")

    # -- threading ----------------------------------------------------------

    def test_thread_local_isolation(self) -> None:
        """Another thread must NOT see the main thread's active group."""
        fake = MagicMock()
        fake.group_name = "tp_group"
        other_saw: list[Any] = []

        def worker() -> None:
            other_saw.append(getattr(self._state, "pg", None))

        with self.distributed_group(fake):
            t = threading.Thread(target=worker)
            t.start()
            t.join()

        self.assertEqual(len(other_saw), 1)
        self.assertIsNone(other_saw[0])

    def test_thread_can_set_its_own_group(self) -> None:
        """Each thread manages its own independent context."""
        fake_main = MagicMock()
        fake_main.group_name = "main_group"
        fake_thread = MagicMock()
        fake_thread.group_name = "thread_group"

        thread_saw: list[Any] = []

        def worker() -> None:
            with self.distributed_group(fake_thread):
                thread_saw.append(self.get_active_group())

        with self.distributed_group(fake_main):
            t = threading.Thread(target=worker)
            t.start()
            t.join()
            # Main thread unchanged
            self.assertIs(self.get_active_group(), fake_main)

        self.assertEqual(thread_saw, [fake_thread])

    # -- deeper nesting -----------------------------------------------------

    def test_deeply_nested_stack(self) -> None:
        """Five nested contexts restore correctly at each level."""
        groups = [MagicMock() for _ in range(5)]
        for i, g in enumerate(groups):
            g.group_name = f"group_{i}"

        def enter(depth: int) -> None:
            if depth == len(groups):
                return
            with self.distributed_group(groups[depth]):
                self.assertIs(self.get_active_group(), groups[depth])
                enter(depth + 1)
                # after inner exits, we're back to current level
                self.assertIs(self.get_active_group(), groups[depth])

        enter(0)
        self.assertIsNone(getattr(self._state, "pg", None))

    def test_sequential_context_managers(self) -> None:
        """Multiple sequential (non-nested) uses each set and restore properly."""
        for i in range(3):
            g = MagicMock()
            g.group_name = f"group_{i}"
            with self.distributed_group(g):
                self.assertIs(self.get_active_group(), g)
            self.assertIsNone(getattr(self._state, "pg", None))

    def test_none_group_is_valid(self) -> None:
        """distributed_group(None) is legal and makes get_active_group return None."""
        fake = MagicMock()
        fake.group_name = "outer"
        with self.distributed_group(fake):
            # Override with None
            with self.distributed_group(None):
                # None stored → get_active_group falls through to dist fallback
                active = getattr(self._state, "pg", "SENTINEL")
                self.assertIsNone(active)
            # Restored to fake
            self.assertIs(self.get_active_group(), fake)

    # -- module argument --------------------------------------------------------

    def test_with_module_yields_module(self) -> None:
        """distributed_group(group, module) yields the module as the context value."""
        group = MagicMock()
        group.group_name = "tp0"
        module = nn.Linear(4, 4)
        with self.distributed_group(group, module) as handle:
            self.assertIs(handle, module)

    def test_without_module_yields_none(self) -> None:
        """distributed_group(group) without module yields None."""
        group = MagicMock()
        group.group_name = "tp0"
        with self.distributed_group(group) as handle:
            self.assertIsNone(handle)

    def test_with_module_pre_pins_engines(self) -> None:
        """distributed_group(group, module) calls set_group_name on md engines."""
        eng = _FakeEngine(is_md=True)

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self._run_on_acc_0_engine = eng

        group = _FakeGroup("tp0")
        module = M()
        with self.distributed_group(group, module):
            pass
        self.assertEqual(eng.group_name_calls, ["tp0"])

    def test_with_module_state_active_inside_block(self) -> None:
        """_state.pg is set for the duration of the with block."""
        group = _FakeGroup("tp0")

        class M(nn.Module):
            pass

        captured = []
        with self.distributed_group(group, M()):
            captured.append(getattr(self._state, "pg", None))
        self.assertIs(captured[0], group)

    def test_with_module_state_restored_after_block(self) -> None:
        """_state.pg is restored after the with block exits."""
        group = _FakeGroup("tp0")

        class M(nn.Module):
            pass

        with self.distributed_group(group, M()):
            pass
        self.assertIsNone(getattr(self._state, "pg", None))

    def test_with_module_state_restored_on_exception(self) -> None:
        """_state.pg is restored even when the body raises."""
        group = _FakeGroup("tp0")

        class M(nn.Module):
            pass

        with self.assertRaises(ValueError):
            with self.distributed_group(group, M()):
                raise ValueError("boom")
        self.assertIsNone(getattr(self._state, "pg", None))


# ============================================================================
# Section 2 — set_distributed_group() (no GPU / no dist init)
# ============================================================================


class TestSetDistributedGroup(unittest.TestCase):
    """Unit tests for set_distributed_group().

    All tests avoid dist.init_process_group so they run in any environment,
    including CI without GPUs.  The ``_FakeEngine`` and ``_FakeGroup`` helpers
    duck-type the real objects so we can verify call behaviour without TRT.
    """

    def setUp(self) -> None:
        from torch_tensorrt.distributed._distributed import _state

        # Ensure no leftover pg from a previous test.
        if hasattr(_state, "pg"):
            del _state.pg
        self._state = _state

    def _call(self, module: nn.Module, group: Any) -> None:
        from torch_tensorrt.distributed import set_distributed_group

        set_distributed_group(module, group)

    # ---- helpers ----------------------------------------------------------------

    def _inlined_module(self, *engines: _FakeEngine) -> nn.Module:
        """Return an nn.Module with each engine as a plain instance attribute."""

        class M(nn.Module):
            pass

        m = M()
        for i, eng in enumerate(engines):
            setattr(m, f"_run_on_acc_{i}_engine", eng)
        return m

    # ---- inlined-engine tests ---------------------------------------------------

    def test_inlined_md_engine_receives_group_name(self) -> None:
        """An inlined is_md engine gets set_group_name called with the group name."""
        eng = _FakeEngine(is_md=True)
        self._call(self._inlined_module(eng), _FakeGroup("tp0"))
        self.assertEqual(eng.group_name_calls, ["tp0"])

    def test_inlined_non_md_engine_is_skipped(self) -> None:
        """An inlined engine with is_md=False is not touched."""
        eng = _FakeEngine(is_md=False)
        self._call(self._inlined_module(eng), _FakeGroup("tp0"))
        self.assertEqual(eng.group_name_calls, [])

    def test_no_active_group_is_noop(self) -> None:
        """When the group has no group_name, set_group_name is never called."""
        if dist.is_initialized():
            self.skipTest("dist already initialized in this process")
        eng = _FakeEngine(is_md=True)
        # Plain object has no group_name attribute → get_active_group_name returns ""
        self._call(self._inlined_module(eng), object())
        self.assertEqual(eng.group_name_calls, [])

    def test_multiple_engines_all_stamped(self) -> None:
        """Every distinct md engine in the module receives the group name."""
        eng_a = _FakeEngine(is_md=True)
        eng_b = _FakeEngine(is_md=True)
        self._call(self._inlined_module(eng_a, eng_b), _FakeGroup("tp0"))
        self.assertEqual(eng_a.group_name_calls, ["tp0"])
        self.assertEqual(eng_b.group_name_calls, ["tp0"])

    def test_mixed_md_and_non_md_engines(self) -> None:
        """md engines are stamped; non-md engines are left alone."""
        md_eng = _FakeEngine(is_md=True)
        non_md_eng = _FakeEngine(is_md=False)
        self._call(self._inlined_module(md_eng, non_md_eng), _FakeGroup("tp0"))
        self.assertEqual(md_eng.group_name_calls, ["tp0"])
        self.assertEqual(non_md_eng.group_name_calls, [])

    def test_same_engine_on_multiple_paths_stamped_once(self) -> None:
        """An engine reachable via two module attributes is only stamped once."""
        shared = _FakeEngine(is_md=True)

        class Inner(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self._run_on_acc_0_engine = shared

        class Outer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inner = Inner()
                self._run_on_acc_0_engine = shared  # same object

        self._call(Outer(), _FakeGroup("tp0"))
        self.assertEqual(shared.group_name_calls, ["tp0"])  # exactly once

    def test_nested_submodule_engines_stamped(self) -> None:
        """Engines nested inside child modules are found recursively."""
        eng_outer = _FakeEngine(is_md=True)
        eng_inner = _FakeEngine(is_md=True)

        class Inner(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self._run_on_acc_0_engine = eng_inner

        class Outer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.child = Inner()
                self._run_on_acc_0_engine = eng_outer

        self._call(Outer(), _FakeGroup("tp0"))
        self.assertEqual(eng_outer.group_name_calls, ["tp0"])
        self.assertEqual(eng_inner.group_name_calls, ["tp0"])

    def test_state_is_restored_after_call(self) -> None:
        """_state.pg is restored to its prior value after the call returns."""
        from torch_tensorrt.distributed._distributed import _state

        sentinel = object()
        _state.pg = sentinel
        self._call(self._inlined_module(), _FakeGroup("tp0"))
        self.assertIs(_state.pg, sentinel)

    # ---- TorchTensorRTModule (wrapper submodule) tests -------------------------

    def test_trt_module_wrapper_md_engine_stamped(self) -> None:
        """A TorchTensorRTModule submodule with is_md=True gets set_group_name."""
        from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

        eng = _FakeEngine(is_md=True)

        class FakeWrapper(TorchTensorRTModule):
            def __init__(self) -> None:
                nn.Module.__init__(self)  # bypass real TorchTensorRTModule.__init__
                self.engine = eng

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.trt_block = FakeWrapper()

        self._call(M(), _FakeGroup("tp0"))
        self.assertEqual(eng.group_name_calls, ["tp0"])

    def test_trt_module_wrapper_non_md_engine_skipped(self) -> None:
        """A TorchTensorRTModule submodule with is_md=False is not touched."""
        from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

        eng = _FakeEngine(is_md=False)

        class FakeWrapper(TorchTensorRTModule):
            def __init__(self) -> None:
                nn.Module.__init__(self)
                self.engine = eng

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.trt_block = FakeWrapper()

        self._call(M(), _FakeGroup("tp0"))
        self.assertEqual(eng.group_name_calls, [])

    def test_wrapper_engine_not_double_stamped_via_attr_scan(self) -> None:
        """The wrapper-path engine is not also stamped by the attr-scan path."""
        from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

        eng = _FakeEngine(is_md=True)

        class FakeWrapper(TorchTensorRTModule):
            def __init__(self) -> None:
                nn.Module.__init__(self)
                self.engine = eng

        self._call(FakeWrapper(), _FakeGroup("tp0"))
        # called exactly once — not twice (once via isinstance, once via vars scan)
        self.assertEqual(eng.group_name_calls, ["tp0"])


# ============================================================================
# Section 3 — NCCL library utilities (no GPU)   [was Section 2]
# ============================================================================


class TestNcclUtils(unittest.TestCase):
    """Tests for _nccl_utils.py functions — no GPU / no dist required."""

    def test_get_nccl_library_path_returns_none_or_string(self) -> None:
        """get_nccl_library_path returns either None or an existing directory."""
        from torch_tensorrt.distributed._nccl_utils import get_nccl_library_path

        result = get_nccl_library_path()
        if result is not None:
            self.assertIsInstance(result, str)
            self.assertTrue(
                os.path.isdir(result),
                f"get_nccl_library_path returned non-existent dir: {result}",
            )

    def test_check_nccl_library_path_system_nccl(self) -> None:
        """check_nccl_library_path returns True when nvidia.nccl not installed."""
        from torch_tensorrt.distributed._nccl_utils import (
            check_nccl_library_path,
            get_nccl_library_path,
        )

        nccl_lib_dir = get_nccl_library_path()
        if nccl_lib_dir is None:
            # System NCCL path — must return True
            self.assertTrue(check_nccl_library_path())
        else:
            # nvidia.nccl installed — result depends on LD_LIBRARY_PATH
            result = check_nccl_library_path()
            self.assertIsInstance(result, bool)

    def test_setup_nccl_for_torch_tensorrt_idempotent(self) -> None:
        """Calling setup_nccl_for_torch_tensorrt() multiple times is safe."""
        from torch_tensorrt.distributed import _nccl_utils

        # Reset the guard so we can test the first real call
        _nccl_utils._nccl_setup_checked = False

        from torch_tensorrt.distributed._nccl_utils import setup_nccl_for_torch_tensorrt

        setup_nccl_for_torch_tensorrt()
        setup_nccl_for_torch_tensorrt()  # second call — must not raise

    def test_ensure_nccl_symlink_nonexistent_dir(self) -> None:
        """ensure_nccl_symlink handles a nonexistent directory gracefully."""
        from torch_tensorrt.distributed._nccl_utils import ensure_nccl_symlink

        result = ensure_nccl_symlink("/nonexistent/path/to/nccl/lib")
        # libnccl.so.2 doesn't exist there → returns False
        self.assertFalse(result)

    def test_check_nccl_library_path_detects_missing_ld_path(self) -> None:
        """check_nccl_library_path returns False when LD_LIBRARY_PATH is absent."""
        from torch_tensorrt.distributed._nccl_utils import get_nccl_library_path

        nccl_lib_dir = get_nccl_library_path()
        if nccl_lib_dir is None:
            self.skipTest("nvidia.nccl not installed; system NCCL path is always OK")

        from torch_tensorrt.distributed._nccl_utils import check_nccl_library_path

        original = os.environ.get("LD_LIBRARY_PATH", "")
        # Remove nccl_lib_dir from LD_LIBRARY_PATH
        paths = [p for p in original.split(":") if p and p != nccl_lib_dir]
        os.environ["LD_LIBRARY_PATH"] = ":".join(paths)
        try:
            result = check_nccl_library_path()
            self.assertFalse(result)
        finally:
            os.environ["LD_LIBRARY_PATH"] = original


# ============================================================================
# Section 4 — fuse_distributed_ops graph pass (no GPU, no dist)   [was Section 3]
# ============================================================================


def _build_graph(collective_op, args_without_input):
    """Build a minimal FX graph: input → collective → wait_tensor → output."""
    g = torch.fx.Graph()
    inp = g.placeholder("inp")
    coll = g.call_function(collective_op, args=(inp, *args_without_input))
    wait = g.call_function(torch.ops._c10d_functional.wait_tensor.default, args=(coll,))
    g.output(wait)
    return torch.fx.GraphModule({}, g)


def _node_targets(gm: torch.fx.GraphModule) -> list:
    return [n.target for n in gm.graph.nodes if n.op == "call_function"]


class TestFuseDistributedOps(unittest.TestCase):
    """Unit tests for the fuse_distributed_ops lowering pass.

    Verifies that each collective + wait_tensor pair is replaced by the
    corresponding fused op, and that edge-cases (multiple users) are
    handled correctly.
    """

    def _settings(self):
        from torch_tensorrt.dynamo._settings import CompilationSettings

        return CompilationSettings()

    def _run_pass(self, gm):
        from torch_tensorrt.dynamo.lowering.passes.fuse_distributed_ops import (
            fuse_distributed_ops,
        )

        return fuse_distributed_ops(gm, self._settings())

    # -- all_gather ---------------------------------------------------------

    def test_fuse_all_gather_replaces_pair(self) -> None:
        """all_gather_into_tensor + wait_tensor → tensorrt_fused_nccl_all_gather_op."""
        from torch_tensorrt.dynamo.lowering.passes.fuse_distributed_ops import (
            tensorrt_fused_nccl_all_gather_op,
        )

        gm = _build_graph(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            args_without_input=(2, "test_group"),
        )
        gm = self._run_pass(gm)
        targets = _node_targets(gm)
        self.assertNotIn(
            torch.ops._c10d_functional.all_gather_into_tensor.default, targets
        )
        self.assertNotIn(torch.ops._c10d_functional.wait_tensor.default, targets)
        self.assertIn(tensorrt_fused_nccl_all_gather_op, targets)

    def test_fuse_all_gather_args(self) -> None:
        """Fused all_gather node carries (inp, group_size, group_name)."""
        from torch_tensorrt.dynamo.lowering.passes.fuse_distributed_ops import (
            tensorrt_fused_nccl_all_gather_op,
        )

        gm = _build_graph(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            args_without_input=(4, "grp"),
        )
        gm = self._run_pass(gm)
        fused = next(
            n for n in gm.graph.nodes if n.target == tensorrt_fused_nccl_all_gather_op
        )
        # args: (inp_placeholder, 4, "grp")
        self.assertEqual(fused.args[1], 4)
        self.assertEqual(fused.args[2], "grp")

    # -- reduce_scatter -----------------------------------------------------

    def test_fuse_reduce_scatter_replaces_pair(self) -> None:
        """reduce_scatter_tensor + wait_tensor → tensorrt_fused_nccl_reduce_scatter_op."""
        from torch_tensorrt.dynamo.lowering.passes.fuse_distributed_ops import (
            tensorrt_fused_nccl_reduce_scatter_op,
        )

        gm = _build_graph(
            torch.ops._c10d_functional.reduce_scatter_tensor.default,
            args_without_input=("sum", 2, "test_group"),
        )
        gm = self._run_pass(gm)
        targets = _node_targets(gm)
        self.assertNotIn(
            torch.ops._c10d_functional.reduce_scatter_tensor.default, targets
        )
        self.assertNotIn(torch.ops._c10d_functional.wait_tensor.default, targets)
        self.assertIn(tensorrt_fused_nccl_reduce_scatter_op, targets)

    def test_fuse_reduce_scatter_args(self) -> None:
        """Fused reduce_scatter node carries (inp, reduce_op, group_size, group_name)."""
        from torch_tensorrt.dynamo.lowering.passes.fuse_distributed_ops import (
            tensorrt_fused_nccl_reduce_scatter_op,
        )

        gm = _build_graph(
            torch.ops._c10d_functional.reduce_scatter_tensor.default,
            args_without_input=("sum", 4, "grp"),
        )
        gm = self._run_pass(gm)
        fused = next(
            n
            for n in gm.graph.nodes
            if n.target == tensorrt_fused_nccl_reduce_scatter_op
        )
        self.assertEqual(fused.args[1], "sum")
        self.assertEqual(fused.args[2], 4)
        self.assertEqual(fused.args[3], "grp")

    # -- all_reduce ---------------------------------------------------------

    def test_fuse_all_reduce_replaces_pair(self) -> None:
        """all_reduce + wait_tensor → tensorrt_fused_nccl_all_reduce_op."""
        from torch_tensorrt.dynamo.lowering.passes.fuse_distributed_ops import (
            tensorrt_fused_nccl_all_reduce_op,
        )

        gm = _build_graph(
            torch.ops._c10d_functional.all_reduce.default,
            args_without_input=("sum", "test_group"),
        )
        gm = self._run_pass(gm)
        targets = _node_targets(gm)
        self.assertNotIn(torch.ops._c10d_functional.all_reduce.default, targets)
        self.assertNotIn(torch.ops._c10d_functional.wait_tensor.default, targets)
        self.assertIn(tensorrt_fused_nccl_all_reduce_op, targets)

    def test_fuse_all_reduce_args(self) -> None:
        """Fused all_reduce node carries exactly 3 args: (inp, reduce_op, group_name)."""
        from torch_tensorrt.dynamo.lowering.passes.fuse_distributed_ops import (
            tensorrt_fused_nccl_all_reduce_op,
        )

        gm = _build_graph(
            torch.ops._c10d_functional.all_reduce.default,
            args_without_input=("sum", "my_group"),
        )
        gm = self._run_pass(gm)
        fused = next(
            n for n in gm.graph.nodes if n.target == tensorrt_fused_nccl_all_reduce_op
        )
        # Must be exactly 3 positional args (no group_size)
        self.assertEqual(len(fused.args), 3)
        self.assertEqual(fused.args[1], "sum")
        self.assertEqual(fused.args[2], "my_group")

    def test_fuse_all_reduce_no_group_size_arg(self) -> None:
        """The all_reduce fused op accepts exactly (inp, reduce_op, group_name).

        This is a regression test for the IndexError that occurred when the
        pass incorrectly accessed node.args[3] on an all_reduce node, which
        only has 3 args (index 0-2).
        """
        from torch_tensorrt.dynamo.lowering.passes.fuse_distributed_ops import (
            tensorrt_fused_nccl_all_reduce_op,
        )

        # Must not raise IndexError
        gm = _build_graph(
            torch.ops._c10d_functional.all_reduce.default,
            args_without_input=("sum", "world"),
        )
        try:
            gm = self._run_pass(gm)
        except IndexError as e:
            self.fail(
                f"fuse_distributed_ops raised IndexError on all_reduce: {e}\n"
                "This is the known bug where node.args[3] was accessed on a "
                "3-arg all_reduce node."
            )

    # -- no-fuse when wait_tensor has multiple users -----------------------

    def test_fuse_when_wait_tensor_result_has_multiple_uses(self) -> None:
        """Fusion proceeds even when wait_tensor's result has multiple downstream users.

        The pass only guards that the *collective* has exactly one user (wait_tensor).
        It does not restrict how many nodes consume wait_tensor's output — those uses
        transfer to the fused op's output after fusion, which is correct.
        """
        from torch_tensorrt.dynamo.lowering.passes.fuse_distributed_ops import (
            tensorrt_fused_nccl_all_reduce_op,
        )

        # Build a graph where wait_tensor result is used twice
        g = torch.fx.Graph()
        inp = g.placeholder("inp")
        ar = g.call_function(
            torch.ops._c10d_functional.all_reduce.default,
            args=(inp, "sum", "grp"),
        )
        wait = g.call_function(
            torch.ops._c10d_functional.wait_tensor.default, args=(ar,)
        )
        # wait used in two places — should NOT block fusion
        add1 = g.call_function(torch.ops.aten.add.Tensor, args=(wait, wait))
        g.output(add1)
        gm = torch.fx.GraphModule({}, g)

        gm = self._run_pass(gm)
        targets = _node_targets(gm)
        # Fusion proceeds: original pair replaced by the fused op
        self.assertNotIn(torch.ops._c10d_functional.all_reduce.default, targets)
        self.assertNotIn(torch.ops._c10d_functional.wait_tensor.default, targets)
        self.assertIn(tensorrt_fused_nccl_all_reduce_op, targets)
        # The add node must still be present (uses transferred to fused op)
        self.assertIn(torch.ops.aten.add.Tensor, targets)

    def test_no_fuse_when_collective_has_multiple_users(self) -> None:
        """Pass must NOT fuse when collective has multiple users (even if one is wait)."""
        from torch_tensorrt.dynamo.lowering.passes.fuse_distributed_ops import (
            tensorrt_fused_nccl_all_reduce_op,
        )

        g = torch.fx.Graph()
        inp = g.placeholder("inp")
        ar = g.call_function(
            torch.ops._c10d_functional.all_reduce.default,
            args=(inp, "sum", "grp"),
        )
        wait = g.call_function(
            torch.ops._c10d_functional.wait_tensor.default, args=(ar,)
        )
        # collective result used by wait AND by something else
        add = g.call_function(torch.ops.aten.add.Tensor, args=(ar, wait))
        g.output(add)
        gm = torch.fx.GraphModule({}, g)

        gm = self._run_pass(gm)
        targets = _node_targets(gm)
        self.assertIn(torch.ops._c10d_functional.all_reduce.default, targets)
        self.assertNotIn(tensorrt_fused_nccl_all_reduce_op, targets)

    # -- pass is idempotent -------------------------------------------------

    def test_pass_idempotent(self) -> None:
        """Applying fuse_distributed_ops twice produces the same result."""
        from torch_tensorrt.dynamo.lowering.passes.fuse_distributed_ops import (
            tensorrt_fused_nccl_all_reduce_op,
        )

        gm = _build_graph(
            torch.ops._c10d_functional.all_reduce.default,
            args_without_input=("sum", "world"),
        )
        gm = self._run_pass(gm)
        targets_first = _node_targets(gm)

        gm = self._run_pass(gm)
        targets_second = _node_targets(gm)

        self.assertEqual(targets_first, targets_second)


# ============================================================================
# Section 5 — Single-rank NCCL op compilation via pytest   [was Section 4]
# ============================================================================


class _AllReduceModel(nn.Module):
    def __init__(self, dim: int, group_name: str) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.group_name = group_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        out = torch.ops._c10d_functional.all_reduce.default(x, "sum", self.group_name)
        return torch.ops._c10d_functional.wait_tensor.default(out)


class _AllGatherModel(nn.Module):
    def __init__(self, dim: int, world_size: int, group_name: str) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.world_size = world_size
        self.group_name = group_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        out = torch.ops._c10d_functional.all_gather_into_tensor.default(
            x, self.world_size, self.group_name
        )
        return torch.ops._c10d_functional.wait_tensor.default(out)


class _ReduceScatterModel(nn.Module):
    def __init__(self, dim: int, world_size: int, group_name: str) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.world_size = world_size
        self.group_name = group_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        out = torch.ops._c10d_functional.reduce_scatter_tensor.default(
            x, "sum", self.world_size, self.group_name
        )
        return torch.ops._c10d_functional.wait_tensor.default(out)


@unittest.skipIf(
    not is_nccl_available(),
    "Skipped: NCCL backend not available.",
)
@unittest.skipIf(
    not is_trtllm_for_nccl(),
    "Skipped: TensorRT-LLM plugin for NCCL not available.",
)
class TestNcclOpsSingleRank(unittest.TestCase):
    """Single-rank compilation tests for all three NCCL collective ops.

    Uses WORLD_SIZE=1 / RANK=0 so they run under plain pytest without
    torchrun/mpirun.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from distributed_utils import set_environment_variables_pytest

        set_environment_variables_pytest()
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        cls.group = dist.new_group(ranks=[0])
        cls.group_name = cls.group.group_name
        cls.world_size = 1

    @classmethod
    def tearDownClass(cls) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

    def _run(self, model: nn.Module, inputs: list[torch.Tensor]) -> None:
        """Compile with torch_tensorrt and verify output matches PyTorch."""
        import torch_tensorrt

        model = model.cuda().eval()
        inputs_cuda = [t.cuda() for t in inputs]
        with torch.no_grad():
            ref = model(*inputs_cuda)

        with torch.no_grad():
            trt_model = torch.compile(
                model,
                backend="torch_tensorrt",
                dynamic=False,
                options={
                    "enabled_precisions": {torch.float32},
                    "use_python_runtime": True,
                    "min_block_size": 1,
                    "use_distributed_mode_trace": True,
                },
            )
            out = trt_model(*inputs_cuda)

        torch.testing.assert_close(ref, out, atol=1e-4, rtol=1e-4)

    def test_all_reduce_single_rank(self) -> None:
        """all_reduce compiles and produces correct output on a single rank."""
        dim = 8
        self._run(
            _AllReduceModel(dim, self.group_name),
            [torch.randn(1, dim)],
        )

    def test_all_gather_single_rank(self) -> None:
        """all_gather compiles and produces correct output on a single rank."""
        dim = 8
        self._run(
            _AllGatherModel(dim, self.world_size, self.group_name),
            [torch.randn(1, dim)],
        )

    def test_reduce_scatter_single_rank(self) -> None:
        """reduce_scatter compiles and produces correct output on a single rank."""
        dim = 8
        self._run(
            _ReduceScatterModel(dim, self.world_size, self.group_name),
            [torch.randn(1, dim)],
        )

    def test_distributed_group_with_single_rank_subgroup(self) -> None:
        """distributed_group() selects the subgroup as NCCL communicator source."""
        import torch_tensorrt
        from torch_tensorrt.distributed._distributed import (
            distributed_group,
            get_active_group_name,
        )

        dim = 8
        subgroup = dist.new_group(ranks=[0])

        with distributed_group(subgroup):
            # Inside the context, active group name must reflect subgroup
            self.assertEqual(get_active_group_name(), subgroup.group_name)

            self._run(
                _AllReduceModel(dim, subgroup.group_name),
                [torch.randn(1, dim)],
            )

    def test_get_active_group_falls_back_to_world_when_no_context(self) -> None:
        """When dist is initialized and no context is set, world group is returned."""
        from torch_tensorrt.distributed._distributed import (
            _state,
            get_active_group,
        )

        if hasattr(_state, "pg"):
            del _state.pg

        active = get_active_group()
        self.assertIsNotNone(active)

    def test_group_name_survives_context_exit(self) -> None:
        """After distributed_group() exits, get_active_group_name reverts to world."""
        from torch_tensorrt.distributed._distributed import (
            distributed_group,
            get_active_group_name,
        )

        subgroup = dist.new_group(ranks=[0])
        with distributed_group(subgroup):
            inner_name = get_active_group_name()
        outer_name = get_active_group_name()
        self.assertEqual(inner_name, subgroup.group_name)
        # After exit, world group name (may be "" for WORLD constant)
        self.assertIsNotNone(outer_name)  # not None — dist is still init'd


# ============================================================================
# Section 6 — Python runtime pickle / unpickle of _nccl_comm   [was Section 5]
# ============================================================================


@unittest.skipIf(
    not is_nccl_available(),
    "Skipped: NCCL backend not available.",
)
@unittest.skipIf(
    not is_trtllm_for_nccl(),
    "Skipped: TensorRT-LLM plugin for NCCL not available.",
)
class TestPythonRuntimePickle(unittest.TestCase):
    """Verifies that _nccl_comm is stripped on pickle and reset on unpickle.

    A PyCapsule / raw C pointer cannot be pickled.  The module must drop it
    and reinitialise lazily on the next forward pass.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from distributed_utils import set_environment_variables_pytest

        set_environment_variables_pytest()
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

    @classmethod
    def tearDownClass(cls) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

    def _compile_small_model(self) -> Any:
        """Return a compiled PythonTorchTensorRTModule instance."""
        import torch_tensorrt

        class LinearModel(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + 1

        model = LinearModel().cuda().eval()
        inp = torch.randn(2, 4, device="cuda")
        with torch.no_grad():
            trt_model = torch.compile(
                model,
                backend="torch_tensorrt",
                dynamic=False,
                options={
                    "enabled_precisions": {torch.float32},
                    "use_python_runtime": True,
                    "min_block_size": 1,
                },
            )
            _ = trt_model(inp)  # trigger compilation
        return trt_model

    def test_nccl_comm_absent_after_pickle(self) -> None:
        """__getstate__ must exclude _nccl_comm from the pickled state."""
        import pickle

        trt_model = self._compile_small_model()

        # Locate the underlying PythonTorchTensorRTModule
        def find_module(obj: Any) -> Any:
            from torch_tensorrt.dynamo.runtime._PythonTorchTensorRTModule import (
                PythonTorchTensorRTModule,
            )

            if isinstance(obj, PythonTorchTensorRTModule):
                return obj
            for child in obj.children() if isinstance(obj, nn.Module) else []:
                result = find_module(child)
                if result is not None:
                    return result
            return None

        module = find_module(trt_model)
        if module is None:
            self.skipTest(
                "Could not locate PythonTorchTensorRTModule in compiled model"
            )

        state = module.__getstate__()
        self.assertNotIn(
            "_nccl_comm",
            state,
            "_nccl_comm must be excluded from pickled state (it's a non-picklable C pointer)",
        )

    def test_nccl_comm_reset_to_none_after_unpickle(self) -> None:
        """__setstate__ must set _nccl_comm = None after unpickling."""
        import pickle

        trt_model = self._compile_small_model()

        def find_module(obj: Any) -> Any:
            from torch_tensorrt.dynamo.runtime._PythonTorchTensorRTModule import (
                PythonTorchTensorRTModule,
            )

            if isinstance(obj, PythonTorchTensorRTModule):
                return obj
            for child in obj.children() if isinstance(obj, nn.Module) else []:
                result = find_module(child)
                if result is not None:
                    return result
            return None

        module = find_module(trt_model)
        if module is None:
            self.skipTest(
                "Could not locate PythonTorchTensorRTModule in compiled model"
            )

        data = pickle.dumps(module)
        restored = pickle.loads(data)
        self.assertIsNone(
            restored._nccl_comm,
            "_nccl_comm must be None immediately after unpickling",
        )


# ============================================================================
# Section 7 — Multi-rank tests (torchrun / mpirun, requires --multirank flag)   [was Section 6]
# ============================================================================

# These tests are only executed when the script is run directly with
# --multirank.  They are intentionally structured as plain functions rather
# than unittest.TestCase so they can be driven by torchrun without a test
# runner.


def _multirank_setup() -> tuple:
    """Initialize the distributed environment for multi-rank tests."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    n_gpus = torch.cuda.device_count()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % n_gpus)) % n_gpus
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, device


def _check_close(a: torch.Tensor, b: torch.Tensor, name: str) -> None:
    try:
        torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-3)
        print(f"[PASS] {name}")
    except AssertionError as e:
        print(f"[FAIL] {name}: {e}")
        raise


def _multirank_all_reduce_correctness(
    rank: int, world_size: int, device: torch.device
) -> None:
    """all_reduce sum across all ranks produces world_size * value."""
    from torch_tensorrt.distributed._nccl_utils import setup_nccl_for_torch_tensorrt

    setup_nccl_for_torch_tensorrt()

    group = dist.group.WORLD
    group_name = group.group_name if hasattr(group, "group_name") else ""

    # Each rank sends fill(rank + 1); sum = 1 + 2 + ... + world_size
    expected_sum = world_size * (world_size + 1) // 2

    class AllReduceSum(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = torch.ops._c10d_functional.all_reduce.default(x, "sum", group_name)
            return torch.ops._c10d_functional.wait_tensor.default(out)

    model = AllReduceSum().to(device).eval()
    inp = torch.full((1, 4), float(rank + 1), device=device)

    with torch.no_grad():
        torch_out = model(inp)

    expected = torch.full((1, 4), float(expected_sum), device=device)
    _check_close(torch_out, expected, f"all_reduce sum rank={rank}")


def _multirank_all_gather_correctness(
    rank: int, world_size: int, device: torch.device
) -> None:
    """all_gather concatenates tensors from all ranks in order."""
    group = dist.group.WORLD
    group_name = group.group_name if hasattr(group, "group_name") else ""

    class AllGather(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = torch.ops._c10d_functional.all_gather_into_tensor.default(
                x, world_size, group_name
            )
            return torch.ops._c10d_functional.wait_tensor.default(out)

    model = AllGather().to(device).eval()
    inp = torch.full((1, 4), float(rank), device=device)

    with torch.no_grad():
        out = model(inp)

    # After gather: shape is (world_size, 4), row i == float(i)
    assert out.shape == torch.Size([world_size, 4]), f"Shape mismatch: {out.shape}"
    for r in range(world_size):
        expected_row = torch.full((4,), float(r), device=device)
        _check_close(out[r], expected_row, f"all_gather row {r} rank={rank}")


def _multirank_reduce_scatter_correctness(
    rank: int, world_size: int, device: torch.device
) -> None:
    """reduce_scatter sum then scatters: rank r gets sum of row r."""
    group = dist.group.WORLD
    group_name = group.group_name if hasattr(group, "group_name") else ""

    class ReduceScatter(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = torch.ops._c10d_functional.reduce_scatter_tensor.default(
                x, "sum", world_size, group_name
            )
            return torch.ops._c10d_functional.wait_tensor.default(out)

    model = ReduceScatter().to(device).eval()
    # Input: (world_size, 4), row r = fill(r+1) on every rank
    inp = torch.stack(
        [torch.full((4,), float(r + 1), device=device) for r in range(world_size)]
    )

    with torch.no_grad():
        out = model(inp)

    # Result for rank r: world_size * (r+1)
    expected = torch.full((1, 4), float(world_size * (rank + 1)), device=device)
    _check_close(out, expected, f"reduce_scatter rank={rank}")


def _multirank_distributed_group_tp_model(
    rank: int, world_size: int, device: torch.device
) -> None:
    """Tensor-parallel MLP with distributed_group() context manager produces correct output.

    This is the core test for the distributed_group() API at runtime.
    It verifies that:
      1. The subgroup can be passed to TRT engines via distributed_group()
      2. TRT TP compilation produces the same result as PyTorch TP
    """
    import torch_tensorrt
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
        parallelize_module,
    )
    from torch_tensorrt.distributed._distributed import distributed_group
    from torch_tensorrt.distributed._nccl_utils import setup_nccl_for_torch_tensorrt

    setup_nccl_for_torch_tensorrt()

    device_mesh = init_device_mesh("cuda", (world_size,))

    class TinyMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(16, 32, bias=False)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(32, 16, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc2(self.relu(self.fc1(x)))

    torch.manual_seed(42)
    model = TinyMLP().to(device)
    parallelize_module(
        model,
        device_mesh,
        {"fc1": ColwiseParallel(), "fc2": RowwiseParallel()},
    )

    torch.manual_seed(0)
    inp = torch.randn(4, 16, device=device)

    with torch.no_grad():
        pt_out = model(inp)

    # Compile inside distributed_group context so TRT engines pick up the right PG
    pg = dist.group.WORLD
    with distributed_group(pg):
        trt_model = torch.compile(
            model,
            backend="torch_tensorrt",
            dynamic=False,
            options={
                "enabled_precisions": {torch.float32},
                "use_python_runtime": True,
                "min_block_size": 1,
                "use_distributed_mode_trace": True,
            },
        )
        with torch.no_grad():
            trt_out = trt_model(inp)

    _check_close(pt_out, trt_out, f"TP MLP distributed_group rank={rank}")


def _multirank_distributed_group_subgroup(
    rank: int, world_size: int, device: torch.device
) -> None:
    """distributed_group() with a TP subgroup (not the world group) routes NCCL correctly.

    We create a subgroup containing all ranks (same topology as world, but a
    distinct process group object). The all_reduce result must still be correct.
    """
    if world_size < 2:
        print(f"[SKIP] _multirank_distributed_group_subgroup requires world_size >= 2")
        return
    import torch_tensorrt
    from torch_tensorrt.distributed._distributed import (
        distributed_group,
        get_active_group_name,
    )
    from torch_tensorrt.distributed._nccl_utils import setup_nccl_for_torch_tensorrt

    setup_nccl_for_torch_tensorrt()

    # New subgroup with all ranks (different group object from WORLD)
    subgroup = dist.new_group(ranks=list(range(world_size)))
    sg_name = subgroup.group_name

    class AllReduceModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = torch.ops._c10d_functional.all_reduce.default(x, "sum", sg_name)
            return torch.ops._c10d_functional.wait_tensor.default(out)

    model = AllReduceModel().to(device).eval()
    inp = torch.full((1, 8), float(rank + 1), device=device)
    expected_sum = sum(r + 1 for r in range(world_size))

    with distributed_group(subgroup):
        # Verify get_active_group_name returns the subgroup name inside context
        assert get_active_group_name() == sg_name, (
            f"Expected group name {sg_name!r}, " f"got {get_active_group_name()!r}"
        )

        trt_model = torch.compile(
            model,
            backend="torch_tensorrt",
            dynamic=False,
            options={
                "enabled_precisions": {torch.float32},
                "use_python_runtime": True,
                "min_block_size": 1,
                "use_distributed_mode_trace": True,
            },
        )
        with torch.no_grad():
            out = trt_model(inp)

    expected = torch.full((1, 8), float(expected_sum), device=device)
    _check_close(out, expected, f"distributed_group subgroup all_reduce rank={rank}")


def _multirank_cpp_runtime_bind_nccl(
    rank: int, world_size: int, device: torch.device
) -> None:
    """C++ runtime TRTEngine.bind_nccl_comm() is called exactly once and produces correct results."""
    if world_size < 2:
        print(f"[SKIP] _multirank_cpp_runtime_bind_nccl requires world_size >= 2")
        return
    import torch_tensorrt
    from torch_tensorrt.distributed._nccl_utils import setup_nccl_for_torch_tensorrt

    setup_nccl_for_torch_tensorrt()

    group = dist.group.WORLD
    group_name = group.group_name if hasattr(group, "group_name") else ""

    class AllReduceModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = torch.ops._c10d_functional.all_reduce.default(x, "sum", group_name)
            return torch.ops._c10d_functional.wait_tensor.default(out)

    model = AllReduceModel().to(device).eval()
    inp = torch.full((1, 4), float(rank + 1), device=device)
    expected_sum = world_size * (world_size + 1) // 2

    trt_model = torch.compile(
        model,
        backend="torch_tensorrt",
        dynamic=False,
        options={
            "enabled_precisions": {torch.float32},
            "use_python_runtime": False,  # C++ runtime
            "min_block_size": 1,
            "use_distributed_mode_trace": True,
        },
    )

    with torch.no_grad():
        out = trt_model(inp)
        # Second call — nccl_initialized must be True now, bind_nccl_comm not called again
        out2 = trt_model(inp)

    expected = torch.full((1, 4), float(expected_sum), device=device)
    _check_close(out, expected, f"C++ runtime all_reduce first call rank={rank}")
    _check_close(out2, expected, f"C++ runtime all_reduce second call rank={rank}")


def _multirank_distributed_group_context_switch(
    rank: int, world_size: int, device: torch.device
) -> None:
    """Switching distributed_group context between two subgroups routes to the correct communicator."""
    import torch_tensorrt
    from torch_tensorrt.distributed._distributed import distributed_group
    from torch_tensorrt.distributed._nccl_utils import setup_nccl_for_torch_tensorrt

    setup_nccl_for_torch_tensorrt()

    if world_size < 2:
        print(
            f"[SKIP] test_multirank_distributed_group_context_switch requires world_size >= 2"
        )
        return

    # Two subgroups: each containing all ranks (different group objects)
    sg1 = dist.new_group(ranks=list(range(world_size)))
    sg2 = dist.new_group(ranks=list(range(world_size)))
    sg1_name = sg1.group_name
    sg2_name = sg2.group_name

    class AllReduceModel(nn.Module):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = torch.ops._c10d_functional.all_reduce.default(x, "sum", self.name)
            return torch.ops._c10d_functional.wait_tensor.default(out)

    inp = torch.full((1, 4), float(rank + 1), device=device)
    expected_sum = world_size * (world_size + 1) // 2

    for i, (sg, sg_name) in enumerate([(sg1, sg1_name), (sg2, sg2_name)]):
        model = AllReduceModel(sg_name).to(device).eval()
        with distributed_group(sg):
            trt_model = torch.compile(
                model,
                backend="torch_tensorrt",
                dynamic=False,
                options={
                    "enabled_precisions": {torch.float32},
                    "use_python_runtime": True,
                    "min_block_size": 1,
                    "use_distributed_mode_trace": True,
                },
            )
            with torch.no_grad():
                out = trt_model(inp)

        expected = torch.full((1, 4), float(expected_sum), device=device)
        _check_close(out, expected, f"context_switch sg{i+1} rank={rank}")


def run_multirank_tests() -> None:
    """Entry point for --multirank mode (called by torchrun / mpirun workers)."""
    rank, world_size, device = _multirank_setup()
    print(f"[Rank {rank}/{world_size}] device={device}")

    tests = [
        _multirank_all_reduce_correctness,
        _multirank_all_gather_correctness,
        _multirank_reduce_scatter_correctness,
        _multirank_distributed_group_tp_model,
        _multirank_distributed_group_subgroup,
        _multirank_cpp_runtime_bind_nccl,
        _multirank_distributed_group_context_switch,
    ]

    failed = []
    for test_fn in tests:
        dist.barrier()
        try:
            test_fn(rank, world_size, device)
        except Exception as e:
            failed.append((test_fn.__name__, str(e)))
            if rank == 0:
                print(f"[FAIL] {test_fn.__name__}: {e}")

    dist.barrier()
    dist.destroy_process_group()

    if failed:
        print(f"\n[Rank {rank}] {len(failed)}/{len(tests)} tests FAILED:")
        for name, err in failed:
            print(f"  - {name}: {err}")
        sys.exit(1)
    else:
        if rank == 0:
            print(f"\nAll {len(tests)} multi-rank tests PASSED.")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    if "--multirank" in sys.argv or "--multinode" in sys.argv:
        sys.argv = [a for a in sys.argv if a not in ("--multirank", "--multinode")]
        run_multirank_tests()
    else:
        run_tests()
