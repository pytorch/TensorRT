# Owner(s): ["oncall: gpu_enablement"]
import functools
import logging
import typing as t
from contextlib import contextmanager
from unittest import TestCase

import torch_tensorrt.fx.observer as ob
from torch_tensorrt.fx.observer import observable

_LOGGER: logging.Logger = logging.getLogger(__name__)


def set_observer_callback_rethrow(fn):
    """
    Specify that observer callback exceptions should be re-thrown (default
    behavior is to swallow) Re-throw is only for test purpose.
    """

    @functools.wraps(fn)
    def fn_(*args, **kwargs):
        try:
            ob.RETHROW_CALLBACK_EXCEPTION = True
            return fn(*args, **kwargs)
        finally:
            ob.RETHROW_CALLBACK_EXCEPTION = False

    return fn_


class ObserverTests(TestCase):
    @set_observer_callback_rethrow
    def test_basics(self):
        @observable()
        def foo(x, y, z):
            return x + y + z

        with execution_verifier() as verify_execution:

            @verify_execution
            def log_pre(ctx: ob.ObserveContext) -> None:
                _LOGGER.info(f"calling log: {ctx}")
                assert ctx.callable is foo.orig_func
                assert ctx.args == (1, 2)
                assert ctx.kwargs == {"z": 3}
                assert not ctx.return_value

            @verify_execution
            def log_post(ctx: ob.ObserveContext) -> None:
                _LOGGER.info(f"calling log: {ctx}")
                assert ctx.callable is foo.orig_func
                assert ctx.args == (1, 2)
                assert ctx.kwargs == {"z": 3}
                assert ctx.return_value == 6

            with foo.observers.pre.add(log_pre), foo.observers.post.add(log_post):
                foo(1, 2, z=3)

        with execution_verifier() as verify_execution:

            @verify_execution
            def log_pre(ctx: ob.ObserveContext) -> None:
                _LOGGER.info(f"calling log: {ctx}")

            @verify_execution
            def log_post(ctx: ob.ObserveContext) -> None:
                _LOGGER.info(f"calling log: {ctx}")

            foo.observers.pre.add(log_pre)
            foo.observers.post.add(log_post)
            foo(1, 2, 3)

        with execution_verifier() as verify_execution:

            @verify_execution
            def f1(ctx: ob.ObserveContext) -> None:
                _LOGGER.info(f"calling f1: {ctx}")

            @verify_execution
            def f2(ctx: ob.ObserveContext) -> None:
                _LOGGER.info(f"calling f2: {ctx}")

            # Test that we can register the same observation point twice
            with foo.observers.pre.add(f1):
                with foo.observers.pre.add(f2):
                    foo(1, 2, z=3)

    def test_observer_callbacks_should_not_throw(self):
        @observable()
        def foo(x, y, z):
            return x + y + z

        with execution_verifier() as verify_execution:

            @verify_execution
            def log_pre(ctx: ob.ObserveContext) -> None:
                _LOGGER.info(f"calling log: {ctx}")
                raise CallbackError("TEST CALLBACK EXCEPTION")

            with foo.observers.pre.add(log_pre):
                foo(1, 2, 3)


@contextmanager
def execution_verifier():
    _is_called: t.Dict[callable, bool] = {}

    def verify_executed(fn):
        _is_called[fn] = False

        @functools.wraps(fn)
        def fn_(*args, **kwargs):
            _is_called[fn] = True
            return fn(*args, **kwargs)

        return fn_

    try:
        yield verify_executed
    except:  # noqa: B001
        raise
    else:
        for fn, was_executed in _is_called.items():
            assert was_executed, f"{fn} was not executed"


class CallbackError(Exception):
    pass
