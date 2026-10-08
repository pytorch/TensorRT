"""Shared LeRobot compatibility shims and loading helpers (PI05, GR00T)."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager


def _patch_draccus_for_py314() -> None:
    """draccus passes `Union[...]` as argparse `type=`; on Python 3.14 Union is
    no longer callable so add_argument raises. The type is never invoked when
    parsing from a config file, so wrap non-callables in a placeholder."""
    from draccus.wrappers.field_wrapper import FieldWrapper

    orig = FieldWrapper.arg_options.fget
    if getattr(orig, "_py314_patched", False):
        return

    def arg_options(self):
        opts = orig(self)
        if "type" in opts and not callable(opts["type"]):
            tpe = opts["type"]
            opts["type"] = lambda v, _t=tpe: v
        return opts

    arg_options._py314_patched = True
    FieldWrapper.arg_options = property(arg_options)


def apply_lerobot_compat() -> None:
    """Install LeRobot shims shared by every LeRobot-backed family. Idempotent."""
    _patch_draccus_for_py314()


@contextmanager
def skip_weight_init() -> Generator[None]:
    """Skip random weight init while constructing a model whose checkpoint is
    loaded right after (``from_pretrained``); init is the slow part of export."""
    from transformers.initialization import no_init_weights

    with no_init_weights():
        yield
