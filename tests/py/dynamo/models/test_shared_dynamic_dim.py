# type: ignore
"""
Tests for the ``dynamic_shapes=`` passthrough kwarg on ``torch_tensorrt.compile``.

Background: when a model takes multiple inputs whose dynamic axes must be
**equal at runtime** (e.g. HF encoders with ``input_ids`` / ``attention_mask``
both shaped ``[B, S]``), the legacy auto-inference path in
``dynamo/_tracer.py`` mints an *independent* ``Dim`` per input. ``torch.export``
then fails its constraint check for any forward() that broadcasts across those
axes (here: ``embed(input_ids) * mask.unsqueeze(-1)``), raising
``ConstraintViolationError``.

These tests exercise the new ``dynamic_shapes=`` passthrough that lets the
caller supply a shared ``Dim`` directly to ``torch_tensorrt.compile`` --
mirroring the ``torch.export.export(dynamic_shapes=...)`` signature -- so the
shared-batch case compiles end to end without the caller having to pre-export
the module themselves.
"""
import unittest

import pytest
import torch
import torch.nn as nn
import torch_tensorrt as torchtrt
from torch.export import Dim
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity

assertions = unittest.TestCase()


class _SharedBatchEncoder(nn.Module):
    """HF-style encoder stand-in: two int64 inputs sharing the batch axis.

    The ``embed(input_ids) * mask.unsqueeze(-1)`` broadcast forces
    ``input_ids.size(0) == attention_mask.size(0)`` -- the relationship the
    auto-inference path cannot express.
    """

    def __init__(self, vocab: int = 1024, hidden: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, input_ids, attention_mask):
        x = self.embed(input_ids)
        mask = attention_mask.unsqueeze(-1).to(x.dtype)
        return self.proj(x * mask)


def _kwarg_inputs(seq: int = 16, batch_min: int = 1, batch_max: int = 4):
    return {
        "input_ids": torchtrt.Input(
            min_shape=(batch_min, seq),
            opt_shape=(batch_max, seq),
            max_shape=(batch_max, seq),
            dtype=torch.int64,
            name="input_ids",
        ),
        "attention_mask": torchtrt.Input(
            min_shape=(batch_min, seq),
            opt_shape=(batch_max, seq),
            max_shape=(batch_max, seq),
            dtype=torch.int64,
            name="attention_mask",
        ),
    }


@pytest.mark.unit
@pytest.mark.critical
def test_dynamic_shapes_passthrough_with_shared_batch_dim():
    """With ``dynamic_shapes={..: {0: batch}, ..: {0: batch}}`` (one shared
    ``Dim``), compile succeeds and the engine matches the eager model."""
    model = _SharedBatchEncoder().eval().cuda()

    batch = Dim("batch", min=1, max=4)
    dynamic_shapes = {
        "input_ids": {0: batch},
        "attention_mask": {0: batch},
    }

    trt_mod = torchtrt.compile(
        model,
        ir="dynamo",
        kwarg_inputs=_kwarg_inputs(),
        dynamic_shapes=dynamic_shapes,
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
            f"Shared-batch encoder out-of-tolerance at bs={bs}: cos_sim={cos_sim}",
        )


@pytest.mark.unit
def test_dynamic_shapes_passthrough_positional_tuple_form():
    """``torch.export`` also accepts ``dynamic_shapes`` as a tuple matching the
    positional-args order. Verify the passthrough handles that form too."""
    model = _SharedBatchEncoder().eval().cuda()

    batch = Dim("batch", min=1, max=4)
    seq = 16
    positional_inputs = [
        torchtrt.Input(
            min_shape=(1, seq),
            opt_shape=(4, seq),
            max_shape=(4, seq),
            dtype=torch.int64,
            name="input_ids",
        ),
        torchtrt.Input(
            min_shape=(1, seq),
            opt_shape=(4, seq),
            max_shape=(4, seq),
            dtype=torch.int64,
            name="attention_mask",
        ),
    ]
    # Tuple form: one entry per positional arg, in declaration order.
    dynamic_shapes = ({0: batch}, {0: batch})

    trt_mod = torchtrt.compile(
        model,
        ir="dynamo",
        inputs=positional_inputs,
        dynamic_shapes=dynamic_shapes,
        min_block_size=1,
        cache_built_engines=False,
        reuse_cached_engines=False,
    )

    for bs in (4, 2):
        ids = torch.randint(0, 1024, (bs, seq), dtype=torch.int64, device="cuda")
        mask = torch.ones((bs, seq), dtype=torch.int64, device="cuda")

        with torch.no_grad():
            ref = model(ids, mask)
            out = trt_mod(ids, mask)

        cos_sim = cosine_similarity(ref, out)
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            f"Tuple-form dynamic_shapes out-of-tolerance at bs={bs}: cos_sim={cos_sim}",
        )


@pytest.mark.unit
def test_dynamic_shapes_passthrough_mixed_args_and_kwargs():
    """One positional input, one kwarg input, sharing a batch ``Dim``. Uses the
    unified dict-by-parameter-name form, which spans both positional and keyword
    parameters."""
    model = _SharedBatchEncoder().eval().cuda()

    batch = Dim("batch", min=1, max=4)
    seq = 16

    # input_ids passed positionally, attention_mask as a kwarg.
    positional_inputs = [
        torchtrt.Input(
            min_shape=(1, seq),
            opt_shape=(4, seq),
            max_shape=(4, seq),
            dtype=torch.int64,
            name="input_ids",
        ),
    ]
    kwarg_inputs = {
        "attention_mask": torchtrt.Input(
            min_shape=(1, seq),
            opt_shape=(4, seq),
            max_shape=(4, seq),
            dtype=torch.int64,
            name="attention_mask",
        ),
    }
    dynamic_shapes = {
        "input_ids": {0: batch},
        "attention_mask": {0: batch},
    }

    trt_mod = torchtrt.compile(
        model,
        ir="dynamo",
        inputs=positional_inputs,
        kwarg_inputs=kwarg_inputs,
        dynamic_shapes=dynamic_shapes,
        min_block_size=1,
        cache_built_engines=False,
        reuse_cached_engines=False,
    )

    for bs in (4, 2):
        ids = torch.randint(0, 1024, (bs, seq), dtype=torch.int64, device="cuda")
        mask = torch.ones((bs, seq), dtype=torch.int64, device="cuda")

        with torch.no_grad():
            ref = model(ids, attention_mask=mask)
            out = trt_mod(ids, attention_mask=mask)

        cos_sim = cosine_similarity(ref, out)
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            f"Mixed args/kwargs out-of-tolerance at bs={bs}: cos_sim={cos_sim}",
        )


@pytest.mark.unit
def test_dynamic_shapes_default_path_unchanged_for_static_inputs():
    """Sanity check: when ``dynamic_shapes=None`` and inputs are fully static,
    behavior is unchanged from the legacy path."""

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
