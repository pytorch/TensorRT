from typing import Type, TypeVar

T = TypeVar("T")


_DEBUG_ENABLED_FUNCS = []
_DEBUG_ENABLED_CLS = []

def fn_supports_debugger(func):
    _DEBUG_ENABLED_FUNCS.append(func)
    return func

def cls_supports_debugger(cls:  Type[T]) ->  Type[T]:
    _DEBUG_ENABLED_CLS.append(cls)
    return cls
