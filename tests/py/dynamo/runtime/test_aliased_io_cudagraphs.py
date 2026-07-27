# type: ignore
"""CUDA Graphs + aliased I/O.

CUDA Graphs normally clones each input into a persistent buffer so binding
addresses stay stable across replays. That mechanism is incompatible with
aliased I/O — the engine would write to the persistent clone and the
user's input tensor wouldn't observe the mutation.

The runtime handles this by:

* For each input binding that appears as the target of an aliased output,
  *bypass* the persistent-buffer copy and bind directly to the user's
  tensor. The caller is already required to pass stable input addresses
  under cudagraphs; aliased I/O just makes that contract observable.
* Skip aliased outputs in the post-execution copy-back loop (their
  ``output_buffers`` slot is intentionally never populated; the mutation
  is already visible on the user's input).

These tests cover capture + replay correctness for both KV-cache patterns
(user-input and buffer-style).
"""

import unittest

import torch
import torch_tensorrt
from torch.export import export
from torch.testing._internal.common_utils import TestCase, run_tests


@unittest.skipIf(
    not torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime,
    "Torch-TensorRT runtime is not available",
)
class TestCudagraphsAliasedIO(TestCase):
    def setUp(self):
        # Ensure clean state regardless of prior test ordering.
        torch_tensorrt.runtime.set_cudagraphs_mode(False)

    def tearDown(self):
        torch_tensorrt.runtime.set_cudagraphs_mode(False)

    def _compile(self, model, args):
        ep = export(model, tuple(args))
        return torch_tensorrt.compile(
            ep,
            ir="dynamo",
            inputs=list(args),
            enabled_precisions={torch.float32},
            min_block_size=1,
            use_python_runtime=False,
        )

    def test_user_input_kv_capture_and_replay(self):
        """User passes the same cache tensor across multiple cudagraph
        replays; mutation should land on that tensor each time."""

        class M(torch.nn.Module):
            def forward(self, cache, update):
                cache[:, :, 3:4, :] = update
                return cache.sum()

        cache_sample = torch.zeros(1, 4, 16, 8, device="cuda")
        update_sample = torch.ones(1, 4, 1, 8, device="cuda")
        compiled = self._compile(
            M().cuda(), (cache_sample.clone(), update_sample.clone())
        )

        with torch_tensorrt.runtime.enable_cudagraphs(compiled) as cg:
            # Use the SAME cache tensor across calls (the cudagraphs
            # contract). Each call overwrites position 3 with the new
            # update value.
            cache = torch.zeros(1, 4, 16, 8, device="cuda")
            cache_id, cache_ptr = id(cache), cache.data_ptr()

            # Step 1: capture. cache[3, :] becomes all 1s — sum = 32.
            cg(cache, torch.ones(1, 4, 1, 8, device="cuda") * 1.0)
            self.assertAlmostEqual(cache.sum().item(), 32.0, places=3)
            self.assertEqual(id(cache), cache_id)
            self.assertEqual(cache.data_ptr(), cache_ptr)

            # Step 2: replay. cache[3, :] becomes all 5s — sum = 160.
            cg(cache, torch.ones(1, 4, 1, 8, device="cuda") * 5.0)
            self.assertAlmostEqual(cache.sum().item(), 160.0, places=3)

            # Step 3: replay again. cache[3, :] becomes all 0s — sum = 0.
            cg(cache, torch.zeros(1, 4, 1, 8, device="cuda"))
            self.assertAlmostEqual(cache.sum().item(), 0.0, places=3)

    def test_buffer_kv_capture_and_replay(self):
        """Buffer-backed KV cache: the buffer lives on the compiled
        module. Cudagraphs should still capture+replay and mutate the
        buffer in place."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(1, 4, 16, 8))

            def forward(self, x):
                self.cache[:, :, 3:4, :] = x
                return self.cache.sum()

        compiled = self._compile(M().cuda(), (torch.ones(1, 4, 1, 8, device="cuda"),))

        with torch_tensorrt.runtime.enable_cudagraphs(compiled) as cg:
            cg(torch.ones(1, 4, 1, 8, device="cuda") * 1.0)
            self.assertAlmostEqual(cg.cache.sum().item(), 32.0, places=3)
            cg(torch.ones(1, 4, 1, 8, device="cuda") * 5.0)
            self.assertAlmostEqual(cg.cache.sum().item(), 160.0, places=3)
            cg(torch.zeros(1, 4, 1, 8, device="cuda"))
            self.assertAlmostEqual(cg.cache.sum().item(), 0.0, places=3)

    def test_matches_non_cudagraphs(self):
        """Same inputs, same model — cudagraphs vs no cudagraphs should
        produce identical cache state and return values."""

        class M(torch.nn.Module):
            def forward(self, cache, update):
                cache[:, :, 3:4, :] = update
                return cache.sum() + cache.mean()

        cache_sample = torch.zeros(1, 4, 16, 8, device="cuda")
        update_sample = torch.ones(1, 4, 1, 8, device="cuda") * 3.0
        compiled = self._compile(
            M().cuda(), (cache_sample.clone(), update_sample.clone())
        )

        # No cudagraphs.
        cache_plain = torch.zeros(1, 4, 16, 8, device="cuda")
        update = torch.ones(1, 4, 1, 8, device="cuda") * 7.0
        ret_plain = compiled(cache_plain, update)
        ret_plain_val = ret_plain[0] if isinstance(ret_plain, tuple) else ret_plain

        # With cudagraphs.
        with torch_tensorrt.runtime.enable_cudagraphs(compiled) as cg:
            cache_cg = torch.zeros(1, 4, 16, 8, device="cuda")
            ret_cg = cg(cache_cg, update)
            ret_cg_val = ret_cg[0] if isinstance(ret_cg, tuple) else ret_cg

        self.assertTrue(torch.allclose(cache_plain, cache_cg))
        self.assertTrue(torch.allclose(ret_plain_val, ret_cg_val))


if __name__ == "__main__":
    run_tests()
