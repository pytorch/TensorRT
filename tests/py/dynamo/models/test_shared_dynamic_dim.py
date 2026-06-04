# type: ignore
"""
Tests for sharing a dynamic dimension across inputs via ``Input(name_dims=...)``.

Background: when a model takes multiple inputs whose dynamic axes must be
**equal at runtime** (e.g. HF encoders with ``input_ids`` / ``attention_mask``
both shaped ``[B, S]``), naming each axis independently makes ``torch.export``
mint an *independent* ``Dim`` per input. ``torch.export`` then fails its
constraint check for any forward() that broadcasts across those axes (here:
``embed(input_ids) * mask.unsqueeze(-1)``), raising ``ConstraintViolationError``.

``Input(name_dims={axis: name})`` lets the caller tag a dynamic axis with a
name; the same name across inputs is exported as a single shared ``Dim``. All
the dynamic-shape intent lives on the ``Input`` objects -- no separate
``dynamic_shapes`` argument and no ``torch.export`` knowledge required at the
call site.
"""

import unittest

import pytest
import torch
import torch.nn as nn
import torch_tensorrt as torchtrt
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity

assertions = unittest.TestCase()


class _SharedBatchEncoder(nn.Module):
    """HF-style encoder stand-in: two int64 inputs sharing the batch axis.

    The ``embed(input_ids) * mask.unsqueeze(-1)`` broadcast forces
    ``input_ids.size(0) == attention_mask.size(0)`` -- the relationship a shared
    named dimension expresses.
    """

    def __init__(self, vocab: int = 1024, hidden: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, input_ids, attention_mask):
        x = self.embed(input_ids)
        mask = attention_mask.unsqueeze(-1).to(x.dtype)
        return self.proj(x * mask)


def _named_input(name: str, seq: int = 16, batch_min: int = 1, batch_max: int = 4):
    """A dynamic int64 Input whose batch axis (0) is named "B" for sharing."""
    return torchtrt.Input(
        min_shape=(batch_min, seq),
        opt_shape=(batch_max, seq),
        max_shape=(batch_max, seq),
        dtype=torch.int64,
        name=name,
        name_dims={0: "B"},
    )


@pytest.mark.unit
@pytest.mark.critical
def test_name_dims_shared_batch_kwarg_inputs():
    """Shared batch axis declared via ``Input(name_dims={0: "B"})`` on both
    kwarg inputs -- same name => one exported symbol; engine matches eager."""
    model = _SharedBatchEncoder().eval().cuda()

    kwarg_inputs = {
        "input_ids": _named_input("input_ids"),
        "attention_mask": _named_input("attention_mask"),
    }

    trt_mod = torchtrt.compile(
        model,
        ir="dynamo",
        kwarg_inputs=kwarg_inputs,
        min_block_size=1,
        cache_built_engines=False,
        reuse_cached_engines=False,
    )

    # Sample at the optimization shape and at a smaller batch within the range.
    for bs in (4, 2):
        ids = torch.randint(0, 1024, (bs, 16), dtype=torch.int64, device="cuda")
        mask = torch.ones((bs, 16), dtype=torch.int64, device="cuda")

        with torch.no_grad():
            ref = model(input_ids=ids, attention_mask=mask)
            out = trt_mod(input_ids=ids, attention_mask=mask)

        cos_sim = cosine_similarity(ref, out)
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            f"name_dims shared batch (kwargs) out-of-tolerance at bs={bs}: cos_sim={cos_sim}",
        )


@pytest.mark.unit
def test_name_dims_shared_batch_positional_inputs():
    """Same feature with positional ``inputs=[...]`` instead of kwargs."""
    model = _SharedBatchEncoder().eval().cuda()

    positional_inputs = [
        _named_input("input_ids"),
        _named_input("attention_mask"),
    ]

    trt_mod = torchtrt.compile(
        model,
        ir="dynamo",
        inputs=positional_inputs,
        min_block_size=1,
        cache_built_engines=False,
        reuse_cached_engines=False,
    )

    for bs in (4, 2):
        ids = torch.randint(0, 1024, (bs, 16), dtype=torch.int64, device="cuda")
        mask = torch.ones((bs, 16), dtype=torch.int64, device="cuda")

        with torch.no_grad():
            ref = model(ids, mask)
            out = trt_mod(ids, mask)

        cos_sim = cosine_similarity(ref, out)
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            f"name_dims shared batch (positional) out-of-tolerance at bs={bs}: cos_sim={cos_sim}",
        )


@pytest.mark.unit
def test_name_dims_shared_batch_mixed_args_and_kwargs():
    """input_ids passed positionally, attention_mask as a kwarg; both share "B"."""
    model = _SharedBatchEncoder().eval().cuda()

    trt_mod = torchtrt.compile(
        model,
        ir="dynamo",
        inputs=[_named_input("input_ids")],
        kwarg_inputs={"attention_mask": _named_input("attention_mask")},
        min_block_size=1,
        cache_built_engines=False,
        reuse_cached_engines=False,
    )

    for bs in (4, 2):
        ids = torch.randint(0, 1024, (bs, 16), dtype=torch.int64, device="cuda")
        mask = torch.ones((bs, 16), dtype=torch.int64, device="cuda")

        with torch.no_grad():
            ref = model(ids, attention_mask=mask)
            out = trt_mod(ids, attention_mask=mask)

        cos_sim = cosine_similarity(ref, out)
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            f"name_dims shared batch (mixed) out-of-tolerance at bs={bs}: cos_sim={cos_sim}",
        )


@pytest.mark.unit
def test_name_dims_conflicting_ranges_raises():
    """Same name with different (min, max) across inputs is a user error."""
    from torch_tensorrt.dynamo._tracer import build_dim_registry

    seq = 16
    inputs = {
        "input_ids": torchtrt.Input(
            min_shape=(1, seq),
            opt_shape=(4, seq),
            max_shape=(4, seq),
            dtype=torch.int64,
            name="input_ids",
            name_dims={0: "B"},
        ),
        "attention_mask": torchtrt.Input(
            min_shape=(1, seq),
            opt_shape=(8, seq),
            max_shape=(8, seq),
            dtype=torch.int64,
            name="attention_mask",
            name_dims={0: "B"},
        ),
    }
    with assertions.assertRaises(ValueError):
        build_dim_registry((), inputs)


@pytest.mark.unit
def test_name_dims_rejected_on_static_axis():
    """Naming a static axis (min == max) is rejected at Input construction."""
    with assertions.assertRaises(ValueError):
        torchtrt.Input(
            min_shape=(1, 16),
            opt_shape=(1, 16),
            max_shape=(1, 16),
            dtype=torch.int64,
            name="x",
            name_dims={0: "B"},
        )


@pytest.mark.unit
def test_name_dims_rejected_on_out_of_range_axis():
    """An axis index outside the input's rank is rejected at construction."""
    with assertions.assertRaises(ValueError):
        torchtrt.Input(
            min_shape=(1, 16),
            opt_shape=(4, 16),
            max_shape=(4, 16),
            dtype=torch.int64,
            name="x",
            name_dims={5: "B"},  # rank is 2; axis 5 does not exist
        )


@pytest.mark.unit
def test_default_path_unchanged_for_static_inputs():
    """Sanity check: a fully static input with no name_dims is unchanged."""

    class StaticModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 8)

        def forward(self, x):
            return self.linear(x)

    model = StaticModel().eval().cuda()
    trt_mod = torchtrt.compile(
        model,
        ir="dynamo",
        inputs=[torchtrt.Input(shape=(2, 8), dtype=torch.float32, name="x")],
        min_block_size=1,
        cache_built_engines=False,
        reuse_cached_engines=False,
    )
    x = torch.randn((2, 8), device="cuda")
    with torch.no_grad():
        ref = model(x)
        out = trt_mod(x)
    assertions.assertTrue(cosine_similarity(ref, out) > COSINE_THRESHOLD)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
