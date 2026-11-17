"""Observer isolation tests that run without importing Torch-TensorRT runtimes."""

import asyncio
import importlib.util
import sys
from contextvars import copy_context
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "_torch_tensorrt_observer_tests",
    Path(__file__).resolve().parents[3]
    / "py"
    / "torch_tensorrt"
    / "dynamo"
    / "observer.py",
)
observer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = observer
spec.loader.exec_module(observer)


def test_child_context_registration_does_not_change_parent():
    observed = observer.Observer(name="copied context")
    calls = []
    with observed.add(lambda: calls.append("parent")):
        child = copy_context()

        def run_child():
            with observed.add(lambda: calls.append("child")):
                observed.observe()

        child.run(run_child)
        assert calls == ["parent", "child"]
        calls.clear()
        observed.observe()
        assert calls == ["parent"]
    observed.observe()
    assert calls == ["parent"]


def test_async_siblings_inherit_parent_without_sharing_registrations():
    observed = observer.Observer(name="async")
    calls = []

    async def run():
        async def child(name):
            with observed.add(lambda value: calls.append((name, value))):
                await asyncio.sleep(0)
                observed.observe(name)

        with observed.add(lambda value: calls.append(("parent", value))):
            await asyncio.gather(child("left"), child("right"))
            observed.observe("after")

    asyncio.run(run())
    assert calls == [
        ("parent", "left"),
        ("left", "left"),
        ("parent", "right"),
        ("right", "right"),
        ("parent", "after"),
    ]


def test_nested_duplicate_callbacks_and_exception_cleanup():
    observed = observer.Observer(name="nesting")
    calls = []
    callback = lambda: calls.append("called")
    with observed.add(callback):
        with pytest.raises(ValueError):
            with observed.add(callback):
                observed.observe()
                assert calls == ["called", "called"]
                raise ValueError("leave inner context")
        calls.clear()
        observed.observe()
        assert calls == ["called"]
    calls.clear()
    observed.observe()
    assert calls == []


def test_registration_inside_callback_uses_a_stable_snapshot():
    observed = observer.Observer(name="reentrancy")
    calls = []

    def callback():
        calls.append("outer")
        with observed.add(lambda: calls.append("temporary")):
            pass

    with observed.add(callback):
        observed.observe()
        observed.observe()
    assert calls == ["outer", "outer"]


def test_post_observer_distinguishes_successful_none_from_failure():
    contexts = []
    failure = RuntimeError("original failure")

    @observer.observable()
    def observed(fail):
        if fail:
            raise failure
        return None

    with observed.observers.post.add(contexts.append):
        assert observed(False) is None
        with pytest.raises(RuntimeError) as caught:
            observed(True)
    assert caught.value is failure
    assert contexts[0].return_value is None and contexts[0].exception is None
    assert contexts[1].return_value is None and contexts[1].exception is failure
    assert contexts[1].args == (True,)


def test_optional_callback_failures_preserve_result_and_original_exception():
    def callback(context):
        raise ValueError("callback failure")

    @observer.observable()
    def observed(fail=False):
        if fail:
            raise RuntimeError("operation failure")
        return 7

    with observed.observers.pre.add(callback), observed.observers.post.add(callback):
        assert observed() == 7
        with pytest.raises(RuntimeError, match="operation failure"):
            observed(True)
