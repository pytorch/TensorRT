# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
import functools
import logging
import traceback
import typing as t
from contextvars import ContextVar
from dataclasses import dataclass, field

_LOGGER = logging.getLogger(__name__)

# A context variable to hold registered callbacks for all the observers for the
# current execution context. The callbacks list could have been a member
# variable on the observer instance, however, contextvars document advice
# against creating context variables not at module-global level.
# https://docs.python.org/3/library/contextvars.html#contextvars.ContextVar
_CALLBACKS: ContextVar[dict[object, tuple[t.Callable[..., t.Any], ...]] | None] = (
    ContextVar("_CALLBACKS", default=None)
)

TObserverCallback = t.TypeVar("TObserverCallback", bound=t.Callable[..., t.Any])
P = t.ParamSpec("P")
R = t.TypeVar("R")
R_co = t.TypeVar("R_co", covariant=True)

# Whether to rethrow the exception caught while calling observer callbacks.
# Default to False. True is only used during tests.
RETHROW_CALLBACK_EXCEPTION: bool = False


@dataclass(frozen=True)
class Observer(t.Generic[TObserverCallback]):
    """
    Usage:

    >>> some_observer: Observer = ...
    >>> with some_observer.add(callback_func):
    >>>     # do stuff, and when some_observer.observe() is called,
    >>>     # it will execute callback_func()
    >>>     ...

    """

    name: str = ""
    # Ensure each Observer instance is considered a distinct key when stored in
    # the `_CALLBACKS` dictionary.
    unique_id: object = field(default_factory=lambda: object())

    def add(self, callback: TObserverCallback) -> t.ContextManager[None]:
        # Context copies share values, so never mutate a registry or callback
        # collection stored in a ContextVar. Registration is copy-on-write.
        registry = dict(_CALLBACKS.get() or {})
        registry[self] = registry.get(self, ()) + (callback,)
        token = _CALLBACKS.set(registry)

        # Cannot decorate the outer `add` directly with `contextmanager`,
        # because if it were not used with a `with` statement, its body won't
        # be executed.
        @contextlib.contextmanager
        def _add() -> t.Iterator[None]:
            try:
                yield
            finally:
                _CALLBACKS.reset(token)

        return _add()

    def observe(self, *args: t.Any, **kwargs: t.Any) -> None:
        for callback in self._get_callbacks():
            with _log_error(
                "Error calling observer callback", rethrow=RETHROW_CALLBACK_EXCEPTION
            ):
                callback(*args, **kwargs)

    def _get_callbacks(self) -> tuple[TObserverCallback, ...]:
        """
        Gets the callbacks registered in current execution context. Any code
        that iterates over the returned immutable tuple is concurrency safe.
        """
        callbacks = (_CALLBACKS.get() or {}).get(self, ())
        return t.cast(tuple[TObserverCallback, ...], callbacks)


@dataclass(frozen=True)
class ObserveContext:
    """
    Passed to the registered callables that observes any function decorated by
    `observable`. See `observable` for detail.

    Attributes:
        callable: the observed callable object
        args: the args passed to the callable
        kwargs: the kwargs passed to the callable
        return_value: the return value returned by the callable, only available
            when observing the callable after its invocation (via
            `CallableObservers.post`)
    """

    callable: t.Callable[..., t.Any]
    args: tuple[t.Any, ...]
    kwargs: t.Mapping[str, t.Any]
    return_value: t.Any = None
    exception: t.Optional[BaseException] = None


def observable() -> t.Callable[[t.Callable[P, R]], ObservedCallable[P, R]]:
    """
    A decorator to turn a function into observable

    Example:

    >>> @observable()
    >>> def func_to_observe(x, y) -> int:
    >>>     ...
    >>>
    >>> def log(ctx: ObserveContext):
    >>>     print(
    >>>         f"called {ctx.callable.__name__} with {ctx.args} {ctx.kwargs}"
    >>>     )
    >>>
    >>> # register:
    >>> with func_to_observe.observers.pre.add(log):
    >>>     func_to_observe(1, 2)
    >>>     # print out "called func_to_observe with (1,2)
    >>> # here it won't print
    """

    def decorator(observed_func: t.Callable[P, R]) -> ObservedCallable[P, R]:
        return _make_observable(orig_func=observed_func)

    return decorator


@dataclass(frozen=True)
class CallableObservers:
    pre: Observer[t.Callable[[ObserveContext], None]]
    post: Observer[t.Callable[[ObserveContext], None]]


class ObservedCallable(t.Protocol[P, R_co]):
    """
    Interface for an observed callable
    """

    @property
    def observers(self) -> CallableObservers: ...

    @property
    def orig_func(self) -> t.Callable[P, R_co]: ...

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R_co: ...


def _make_observable(
    orig_func: t.Callable[P, R],
) -> ObservedCallable[P, R]:
    """
    A wrapper for a callable which is to be observed.
    """

    observers = CallableObservers(
        pre=Observer(),
        post=Observer(),
    )

    @functools.wraps(orig_func)
    def observed_func(*args: P.args, **kwargs: P.kwargs) -> R:
        observers.pre.observe(ObserveContext(orig_func, args, kwargs))
        return_value: t.Any = None
        exception: BaseException | None = None
        try:
            result = orig_func(*args, **kwargs)
            return_value = result
            return result
        except BaseException as error:
            exception = error
            raise
        finally:
            observers.post.observe(
                ObserveContext(orig_func, args, kwargs, return_value, exception)
            )

    observed_func.orig_func = orig_func
    observed_func.observers = observers

    return t.cast(ObservedCallable[P, R], observed_func)


@contextlib.contextmanager
def _log_error(msg: str, rethrow: bool = False) -> t.Iterator[None]:
    try:
        yield
    except Exception as e:
        _e = e  # noqa: F841
        _LOGGER.info(f"{msg} (This error is handled): {traceback.format_exc()}")
        if rethrow:
            raise
