# type: ignore
"""Tests for the ``dynamic_shapes=`` passthrough kwarg on ``torch_tensorrt.compile``."""

import unittest

import pytest
import torch
import torch.nn as nn
import torch_tensorrt as torchtrt
from torch.export import Dim
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity

assertions = unittest.TestCase()


class _SharedBatchEncoder(nn.Module):
    def __init__(self, vocab: int = 1024, hidden: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, input_ids, attention_mask):
        x = self.embed(input_ids)
        mask = attention_mask.unsqueeze(-1).to(x.dtype)
        return self.proj(x * mask)


def _plain_input(name: str, seq: int = 16, batch_min: int = 1, batch_max: int = 4):
    return torchtrt.Input(
        min_shape=(batch_min, seq),
        opt_shape=(batch_max, seq),
        max_shape=(batch_max, seq),
        dtype=torch.int64,
        name=name,
    )


def _kwarg_inputs(seq: int = 16, batch_min: int = 1, batch_max: int = 4):
    return {
        "input_ids": _plain_input("input_ids", seq, batch_min, batch_max),
        "attention_mask": _plain_input("attention_mask", seq, batch_min, batch_max),
    }


@pytest.mark.unit
@pytest.mark.critical
def test_dynamic_shapes_passthrough_with_shared_batch_dim():
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

    for bs in (4, 2):
        ids = torch.randint(0, 1024, (bs, 16), dtype=torch.int64, device="cuda")
        mask = torch.ones((bs, 16), dtype=torch.int64, device="cuda")
        with torch.no_grad():
            ref = model(input_ids=ids, attention_mask=mask)
            out = trt_mod(input_ids=ids, attention_mask=mask)
        cos_sim = cosine_similarity(ref, out)
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            f"dynamic_shapes shared-batch kwargs out-of-tolerance at bs={bs}: cos_sim={cos_sim}",
        )


@pytest.mark.unit
def test_dynamic_shapes_passthrough_positional_tuple_form():
    model = _SharedBatchEncoder().eval().cuda()

    batch = Dim("batch", min=1, max=4)
    seq = 16
    positional_inputs = [
        _plain_input("input_ids", seq),
        _plain_input("attention_mask", seq),
    ]
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
            f"dynamic_shapes tuple form out-of-tolerance at bs={bs}: cos_sim={cos_sim}",
        )


@pytest.mark.unit
def test_dynamic_shapes_passthrough_mixed_args_and_kwargs():
    model = _SharedBatchEncoder().eval().cuda()

    batch = Dim("batch", min=1, max=4)
    seq = 16
    dynamic_shapes = {
        "input_ids": {0: batch},
        "attention_mask": {0: batch},
    }

    trt_mod = torchtrt.compile(
        model,
        ir="dynamo",
        inputs=[_plain_input("input_ids", seq)],
        kwarg_inputs={"attention_mask": _plain_input("attention_mask", seq)},
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
            f"dynamic_shapes mixed args/kwargs out-of-tolerance at bs={bs}: cos_sim={cos_sim}",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
# type: ignore
"""Tests for the ``dynamic_shapes=`` passthrough kwarg on ``torch_tensorrt.compile``.

Background: when a model takes multiple inputs whose dynamic axes must be equal
at runtime (e.g. HF encoders with ``input_ids`` / ``attention_mask`` both shaped
``[B, S]``), the legacy auto-inference path can mint an independent ``Dim`` per
input. These tests exercise the explicit ``dynamic_shapes=`` passthrough that
lets callers supply a shared ``Dim`` directly to ``torch_tensorrt.compile``.
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
    """HF-style encoder stand-in: two int64 inputs sharing the batch axis."""

    def __init__(self, vocab: int = 1024, hidden: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, input_ids, attention_mask):
        x = self.embed(input_ids)
        mask = attention_mask.unsqueeze(-1).to(x.dtype)
        return self.proj(x * mask)


def _plain_input(name: str, seq: int = 16, batch_min: int = 1, batch_max: int = 4):
    return torchtrt.Input(
        min_shape=(batch_min, seq),
        opt_shape=(batch_max, seq),
        max_shape=(batch_max, seq),
        dtype=torch.int64,
        name=name,
    )


def _kwarg_inputs(seq: int = 16, batch_min: int = 1, batch_max: int = 4):
    return {
        "input_ids": _plain_input("input_ids", seq, batch_min, batch_max),
        "attention_mask": _plain_input("attention_mask", seq, batch_min, batch_max),
    }


@pytest.mark.unit
@pytest.mark.critical
def test_dynamic_shapes_passthrough_with_shared_batch_dim():
    """Dict-by-parameter dynamic_shapes form with one shared batch ``Dim``."""
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

    for bs in (4, 2):
        ids = torch.randint(0, 1024, (bs, 16), dtype=torch.int64, device="cuda")
        mask = torch.ones((bs, 16), dtype=torch.int64, device="cuda")

        with torch.no_grad():
            ref = model(input_ids=ids, attention_mask=mask)
            out = trt_mod(input_ids=ids, attention_mask=mask)

        cos_sim = cosine_similarity(ref, out)
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            f"dynamic_shapes shared-batch kwargs out-of-tolerance at bs={bs}: cos_sim={cos_sim}",
        )


@pytest.mark.unit
def test_dynamic_shapes_passthrough_positional_tuple_form():
    """Tuple-form ``dynamic_shapes`` matching positional argument order."""
    model = _SharedBatchEncoder().eval().cuda()

    batch = Dim("batch", min=1, max=4)
    seq = 16
    positional_inputs = [
        _plain_input("input_ids", seq),
        _plain_input("attention_mask", seq),
    ]
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
            f"dynamic_shapes tuple form out-of-tolerance at bs={bs}: cos_sim={cos_sim}",
        )


@pytest.mark.unit
def test_dynamic_shapes_passthrough_mixed_args_and_kwargs():
    """Dict-by-parameter dynamic_shapes form spanning args and kwargs."""
    model = _SharedBatchEncoder().eval().cuda()

    batch = Dim("batch", min=1, max=4)
    seq = 16
    positional_inputs = [_plain_input("input_ids", seq)]
    kwarg_inputs = {"attention_mask": _plain_input("attention_mask", seq)}
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
            f"dynamic_shapes mixed args/kwargs out-of-tolerance at bs={bs}: cos_sim={cos_sim}",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
