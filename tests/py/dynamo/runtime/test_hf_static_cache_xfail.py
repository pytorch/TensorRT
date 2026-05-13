# type: ignore
"""HuggingFace decoder with ``StaticCache`` — currently expected to fail.

This file documents the current gap between Torch-TensorRT's aliased-I/O
support and stock HuggingFace decoder-only LMs. The test compiles
``TorchExportableModuleWithStaticCache(GPT2)`` (HF's recommended export
path for static-cache models) and asserts that it fails with the specific
known error so we notice if/when it starts working.

What works today (covered by ``test_aliased_io.py``):

* ``register_buffer`` + slice-write KV cache (the pattern we built for).

What fails with stock HF + ``StaticCache``:

1. ``ExportedProgram.run_decompositions`` fails inside torch with
   ``AssertionError: expected compiled_fn to be GraphModule, got
   <class 'function'>`` (``_functorch/_aot_autograd/graph_compile.py``).
   This is a torch.export internal that surfaces when running
   decompositions on an EP whose body contains certain ATen patterns
   produced by the HF wrapper. Not Torch-TensorRT's fault.

2. Even bypassing the decomp issue, HF's ``StaticCache.update`` writes
   via ``aten.index_copy_(cache, dim=2, idx, k_or_v)``, NOT
   ``aten.slice_scatter``. Our converter today only matches
   ``slice_scatter``. The cache tensors also show up as ``c_*_lifted``
   constants rather than BUFFER inputs in the graph_signature.

To make this end-to-end we'd need:

* An ``index_copy`` converter with the same KV-eligibility check + alias
  recording.
* Extension of ``lift_mutated_buffers`` to recognize mutated lifted
  constants (in addition to mutated buffers).
* Either an upstream fix for the ``aot_stage2_export`` issue or a
  workaround that skips ``run_decompositions`` for already-decomposed EPs.

When the upstream issues are resolved or those features land, this
xfail test should start passing — flip it to a real test then.
"""
import unittest

import torch
import torch_tensorrt


def _can_import_hf():
    try:
        from transformers import GPT2Config  # noqa: F401
        from transformers.integrations.executorch import (  # noqa: F401
            TorchExportableModuleWithStaticCache,
        )

        return True
    except Exception:
        return False


@unittest.skipUnless(_can_import_hf(), "transformers not installed")
class TestHFStaticCacheCurrentLimitations(unittest.TestCase):
    def _make_wrapped(self):
        from transformers import GPT2Config, GPT2LMHeadModel
        from transformers.integrations.executorch import (
            TorchExportableModuleWithStaticCache,
        )

        config = GPT2Config(
            vocab_size=128, n_positions=64, n_embd=64, n_layer=2, n_head=4
        )
        model = GPT2LMHeadModel(config).eval().cuda()
        model.config.use_cache = True
        model.generation_config.use_cache = True
        model.generation_config.cache_implementation = "static"
        model.generation_config.cache_config = {
            "batch_size": 1,
            "max_cache_len": 32,
            "device": "cuda",
        }
        return TorchExportableModuleWithStaticCache(
            model, batch_size=1, max_cache_len=32, device="cuda"
        ).cuda()

    def test_compile_fails_with_known_error(self):
        """Compiling GPT2 + StaticCache currently raises during
        ``run_decompositions``. We assert the error so a future
        upstream/internal fix that unblocks compilation shows up as a
        test failure (signaling we should remove this xfail and write a
        real test)."""
        wrapped = self._make_wrapped()
        input_ids = torch.tensor([[1]], dtype=torch.long, device="cuda")
        cache_position = torch.tensor([0], dtype=torch.long, device="cuda")
        ep = torch.export.export(
            wrapped,
            (),
            {"input_ids": input_ids, "cache_position": cache_position},
            strict=False,
        )

        # Confirm the EP looks like we expect (HF emits index_copy_ for
        # the cache write, not slice_scatter, and treats caches as
        # lifted constants rather than BUFFER inputs).
        has_index_copy = any(
            n.op == "call_function" and "index_copy" in str(n.target)
            for n in ep.graph.nodes
        )
        self.assertTrue(
            has_index_copy,
            "Expected aten.index_copy_ in HF GPT2 + StaticCache EP — if this "
            "test starts failing here, HF may have switched to slice_scatter "
            "and our existing converter might now handle the model directly.",
        )

        with self.assertRaises(Exception) as ctx:
            torch_tensorrt.compile(
                ep,
                ir="dynamo",
                inputs=[input_ids, cache_position],
                enabled_precisions={torch.float32},
                min_block_size=1,
                use_python_runtime=False,
            )
        # The torch internal raises an AssertionError with a specific
        # message. Match loosely so the test isn't brittle to phrasing
        # changes — we only want to detect that the failure is the
        # known one rather than something new.
        msg = str(ctx.exception)
        self.assertTrue(
            "compiled_fn" in msg
            or "GraphModule" in msg
            or "index_copy" in msg.lower()
            or "aot_stage2_export" in msg,
            f"Compile failed but not with the known error pattern: {msg}",
        )


if __name__ == "__main__":
    unittest.main()
