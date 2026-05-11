# type: ignore
"""Integration tests for the ``lift_mutable_buffers`` flag on
``convert_exported_program_to_serialized_trt_engine``.

The low-level entry point returns a serialized engine. The high-level
``torch_tensorrt.compile`` automatically lifts buffers and wraps the
result in ``BufferThreadingModule``; this lower-level surface exposes
the same lifting machinery but leaves runtime binding management to the
caller. These tests exercise that caller-managed workflow end-to-end:

1. Compile via the low-level API with ``lift_mutable_buffers=True``.
2. Introspect the resulting engine — confirm it has additional input
   bindings for each mutated buffer and an aliased output per binding.
3. Construct a ``TorchTensorRTModule`` (C++ runtime — required for
   aliased I/O) with the discovered bindings.
4. Thread the buffer values in on each call and verify in-place
   mutation works (cache state persists across calls).
"""
import torch
import torch_tensorrt
from torch.export import export
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo import convert_exported_program_to_serialized_trt_engine
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

import tensorrt as trt


def _introspect_engine(engine_bytes):
    """Deserialize and return (input_names, output_names, aliased_io)."""
    rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    eng = rt.deserialize_cuda_engine(engine_bytes)
    input_names = []
    output_names = []
    aliased_io = {}
    for i in range(eng.num_io_tensors):
        name = eng.get_tensor_name(i)
        if eng.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input_names.append(name)
        else:
            output_names.append(name)
            try:
                aliased_in = eng.get_aliased_input_tensor(name)
            except Exception:
                aliased_in = None
            if aliased_in:
                aliased_io[name] = (aliased_in, "kv_cache_update")
    return input_names, output_names, aliased_io


def _build_module(engine_bytes, input_names, output_names, aliased_io):
    """Wrap engine bytes in a TorchTensorRTModule (C++ runtime path).

    The user-output boundary is derived inside the module from
    ``output_binding_names`` + ``aliased_io`` (side-effect aliased
    outputs always live at the end of the binding list).
    """
    return TorchTensorRTModule(
        serialized_engine=engine_bytes,
        input_binding_names=input_names,
        output_binding_names=output_names,
        aliased_io=aliased_io,
    )


class TestLiftMutableBuffersAPI(TestCase):
    def test_flag_off_no_buffer_bindings(self):
        """Default ``lift_mutable_buffers=False`` keeps the buffer baked
        into the engine; bindings only cover user inputs/outputs."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(1, 4, 16, 8))

            def forward(self, x):
                self.cache[:, :, 3:4, :] = x
                return self.cache.sum()

        m = M().cuda()
        x = torch.ones(1, 4, 1, 8, device="cuda")
        ep = export(m, (x.clone(),))
        engine_bytes = convert_exported_program_to_serialized_trt_engine(
            ep,
            inputs=[x.clone()],
            enabled_precisions={torch.float32},
            min_block_size=1,
        )
        inputs, outputs, aliased = _introspect_engine(engine_bytes)
        # Only the user input survives; the buffer is folded into the engine.
        self.assertEqual(inputs, ["x"])
        self.assertEqual(len(outputs), 1)
        self.assertEqual(aliased, {})

    def test_flag_on_adds_buffer_binding_and_alias(self):
        """``lift_mutable_buffers=True`` adds the buffer as an input
        binding and the engine reports an aliased output for it."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(1, 4, 16, 8))

            def forward(self, x):
                self.cache[:, :, 3:4, :] = x
                return self.cache.sum()

        m = M().cuda()
        x = torch.ones(1, 4, 1, 8, device="cuda")
        ep = export(m, (x.clone(),))
        engine_bytes = convert_exported_program_to_serialized_trt_engine(
            ep,
            inputs=[x.clone()],
            enabled_precisions={torch.float32},
            min_block_size=1,
            lift_mutable_buffers=True,
        )
        inputs, outputs, aliased = _introspect_engine(engine_bytes)

        # User input first, lifted buffer appended after.
        self.assertEqual(inputs, ["x", "buf_cache"])
        # One user output + one aliased mutation output.
        self.assertEqual(len(outputs), 2)
        # The aliased output should point at the buffer binding.
        self.assertEqual(len(aliased), 1)
        out_name, (in_name, kind) = next(iter(aliased.items()))
        self.assertEqual(in_name, "buf_cache")
        self.assertEqual(kind, "kv_cache_update")

    def test_caller_threads_buffer_for_in_place_mutation(self):
        """End-to-end: caller takes the engine, threads the buffer in,
        observes in-place mutation."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(1, 4, 16, 8))

            def forward(self, x):
                self.cache[:, :, 3:4, :] = x
                return self.cache.sum()

        m = M().cuda()
        x_sample = torch.ones(1, 4, 1, 8, device="cuda")
        ep = export(m, (x_sample.clone(),))

        engine_bytes = convert_exported_program_to_serialized_trt_engine(
            ep,
            inputs=[x_sample.clone()],
            enabled_precisions={torch.float32},
            min_block_size=1,
            lift_mutable_buffers=True,
        )
        inputs, outputs, aliased = _introspect_engine(engine_bytes)
        module = _build_module(engine_bytes, inputs, outputs, aliased)

        # The caller owns the buffer. Pass it explicitly each call; the
        # engine writes through the aliased binding into its storage.
        cache = torch.zeros(1, 4, 16, 8, device="cuda")
        x = torch.ones(1, 4, 1, 8, device="cuda") * 7.0

        # Bindings order: ["x", "buf_cache"], so call as (x, cache).
        cache_id, cache_ptr = id(cache), cache.data_ptr()
        ret = module(x, cache)

        # Numerical: cache should now have 7s at position [3:4, :].
        eager_cache = torch.zeros(1, 4, 16, 8, device="cuda")
        eager_cache[:, :, 3:4, :] = x
        eager_ret = eager_cache.sum()

        self.assertTrue(torch.allclose(cache, eager_cache))
        ret_val = ret[0] if isinstance(ret, tuple) else ret
        self.assertTrue(torch.allclose(ret_val, eager_ret))
        # Identity preserved — same tensor, same storage.
        self.assertEqual(id(cache), cache_id)
        self.assertEqual(cache.data_ptr(), cache_ptr)

    def test_streaming_state_persists_across_calls(self):
        """Caller-managed buffer state should persist across repeated
        calls when the same tensor is threaded in each time."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(1, 4, 16, 8))

            def forward(self, x):
                self.cache[:, :, 3:4, :] = x
                return self.cache.sum()

        m = M().cuda()
        x_sample = torch.ones(1, 4, 1, 8, device="cuda")
        ep = export(m, (x_sample.clone(),))
        engine_bytes = convert_exported_program_to_serialized_trt_engine(
            ep,
            inputs=[x_sample.clone()],
            enabled_precisions={torch.float32},
            min_block_size=1,
            lift_mutable_buffers=True,
        )
        inputs, outputs, aliased = _introspect_engine(engine_bytes)
        module = _build_module(engine_bytes, inputs, outputs, aliased)

        # Caller's cache lives across iterations; engine writes through
        # the aliased binding each call. Cache slice at position 3 has
        # shape (1,4,1,8) = 32 elements, so sum = 32 * scalar.
        cache = torch.zeros(1, 4, 16, 8, device="cuda")
        module(torch.ones(1, 4, 1, 8, device="cuda") * 1.0, cache)
        self.assertAlmostEqual(cache.sum().item(), 32.0, places=3)
        module(torch.ones(1, 4, 1, 8, device="cuda") * 5.0, cache)
        self.assertAlmostEqual(cache.sum().item(), 160.0, places=3)
        module(torch.zeros(1, 4, 1, 8, device="cuda"), cache)
        self.assertAlmostEqual(cache.sum().item(), 0.0, places=3)


if __name__ == "__main__":
    run_tests()
