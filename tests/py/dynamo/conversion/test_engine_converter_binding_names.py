"""
Tests for user-supplied I/O binding names on the engine-converter API.

Architecture: the exported program carries split pytree specs — one for
``args``, one for ``kwargs``, one for the return value.  The user
provides binding names in the same shape via three parallel kwargs:

  * ``arg_input_binding_names`` — pytree matching ``arg_inputs``
  * ``kwarg_input_binding_names`` — pytree matching ``kwarg_inputs``
  * ``output_binding_names`` — pytree matching the model's return value

We ``tree_flatten`` each and verify spec equality against the exported
program's specs.  On success the flat lists concatenate (args then
kwargs) into FX's flattened placeholder order; the interpreter just
indexes positionally.
"""

import unittest

import torch
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo._compiler import (
    BindingNameMismatchError,
    _binding_name_specs,
    _resolve_pytree_binding_names,
    convert_exported_program_to_serialized_trt_engine,
)

import tensorrt as trt

DEVICE = torch.device("cuda", 0)


def _trace(model, args, kwargs=None):
    return torch.export.export(model, args, kwargs or {})


# ── Spec extraction + pytree flattening (no GPU required) ────────────────────


class TestSpecResolution(TestCase):
    def _positional_two_input_program(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        return _trace(M().eval(), (torch.randn(2, 3), torch.randn(2, 3)))

    def _kwarg_program(self):
        class M(torch.nn.Module):
            def forward(self, image, positions):
                return image + positions

        return _trace(
            M().eval(),
            args=(),
            kwargs={"image": torch.randn(2, 3), "positions": torch.randn(2, 3)},
        )

    def _dict_output_program(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return {"primary": torch.relu(x), "secondary": torch.tanh(x)}

        return _trace(M().eval(), (torch.randn(2, 3),))

    def test_specs_extracted_for_positional(self) -> None:
        args_spec, kwargs_spec, out_spec = _binding_name_specs(
            self._positional_two_input_program()
        )
        self.assertIsNotNone(args_spec)
        self.assertEqual(args_spec.num_leaves, 2)
        # kwargs spec is the empty-dict spec — 0 leaves.
        self.assertIsNotNone(kwargs_spec)
        self.assertEqual(kwargs_spec.num_leaves, 0)
        self.assertIsNotNone(out_spec)

    def test_specs_extracted_for_kwargs(self) -> None:
        args_spec, kwargs_spec, _ = _binding_name_specs(self._kwarg_program())
        self.assertEqual(args_spec.num_leaves, 0)
        self.assertEqual(kwargs_spec.num_leaves, 2)

    def test_positional_args_match(self) -> None:
        args_spec, _, _ = _binding_name_specs(self._positional_two_input_program())
        names = _resolve_pytree_binding_names(
            ("name_x", "name_y"), role="arg_input", expected_spec=args_spec
        )
        self.assertEqual(names, ["name_x", "name_y"])

    def test_kwargs_match(self) -> None:
        _, kwargs_spec, _ = _binding_name_specs(self._kwarg_program())
        names = _resolve_pytree_binding_names(
            {"image": "img", "positions": "pos"},
            role="kwarg_input",
            expected_spec=kwargs_spec,
        )
        self.assertEqual(sorted(names), ["img", "pos"])

    def test_dict_output_match(self) -> None:
        _, _, out_spec = _binding_name_specs(self._dict_output_program())
        names = _resolve_pytree_binding_names(
            {"primary": "p_out", "secondary": "s_out"},
            role="output",
            expected_spec=out_spec,
        )
        self.assertEqual(sorted(names), ["p_out", "s_out"])

    def test_spec_mismatch_raises(self) -> None:
        """User provides a dict where positional args were expected."""
        args_spec, _, _ = _binding_name_specs(self._positional_two_input_program())
        with self.assertRaises(BindingNameMismatchError) as ctx:
            _resolve_pytree_binding_names(
                {"x": "a", "y": "b"},
                role="arg_input",
                expected_spec=args_spec,
            )
        self.assertIn("does not match the exported program", str(ctx.exception))

    def test_arity_mismatch_raises(self) -> None:
        args_spec, _, _ = _binding_name_specs(self._positional_two_input_program())
        with self.assertRaises(BindingNameMismatchError):
            _resolve_pytree_binding_names(
                ("only_one",), role="arg_input", expected_spec=args_spec
            )

    def test_non_string_leaf_raises(self) -> None:
        args_spec, _, _ = _binding_name_specs(self._positional_two_input_program())
        with self.assertRaises(BindingNameMismatchError) as ctx:
            _resolve_pytree_binding_names(
                ("a", 2), role="arg_input", expected_spec=args_spec
            )
        self.assertIn("leaves must all be strings", str(ctx.exception))

    def test_duplicate_within_pytree_raises(self) -> None:
        args_spec, _, _ = _binding_name_specs(self._positional_two_input_program())
        with self.assertRaises(BindingNameMismatchError) as ctx:
            _resolve_pytree_binding_names(
                ("same", "same"), role="arg_input", expected_spec=args_spec
            )
        self.assertIn("duplicate", str(ctx.exception))


# ── End-to-end (requires CUDA + TRT) ─────────────────────────────────────────


@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestEngineConverterBindingNames(TestCase):
    def _two_output_model(self):
        class M(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return torch.relu(x), torch.tanh(x)

        x = torch.randn(2, 3, device=DEVICE, dtype=torch.float16)
        return M().eval().cuda().half(), (x,)

    def _kwarg_model(self):
        class M(torch.nn.Module):
            def forward(self, image: torch.Tensor, positions: torch.Tensor):
                return image + positions

        image = torch.randn(2, 3, device=DEVICE, dtype=torch.float16)
        positions = torch.randn(2, 3, device=DEVICE, dtype=torch.float16)
        return M().eval().cuda().half(), {"image": image, "positions": positions}

    def _deserialize(self, engine_bytes: bytes):
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        return runtime.deserialize_cuda_engine(engine_bytes)

    def _binding_names(self, engine, mode: trt.TensorIOMode):
        names = []
        for i in range(engine.num_io_tensors):
            n = engine.get_tensor_name(i)
            if engine.get_tensor_mode(n) == mode:
                names.append(n)
        return names

    def _run_engine(self, engine, named_inputs):
        """Execute an engine through the native TRT Python API.

        Confirms the requested binding names are actually addressable at
        execution time, not just present in the engine metadata.
        ``named_inputs`` is a dict {binding_name -> contiguous CUDA tensor};
        returns a dict {binding_name -> output tensor}.
        """
        context = engine.create_execution_context()
        for name, tensor in named_inputs.items():
            context.set_input_shape(name, tuple(tensor.shape))
            context.set_tensor_address(name, tensor.data_ptr())

        outputs = {}
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
                continue
            shape = tuple(context.get_tensor_shape(name))
            trt_dtype = engine.get_tensor_dtype(name)
            torch_dtype = {
                trt.DataType.FLOAT: torch.float32,
                trt.DataType.HALF: torch.float16,
                trt.DataType.INT32: torch.int32,
                trt.DataType.INT64: torch.int64,
                trt.DataType.BOOL: torch.bool,
                trt.DataType.BF16: torch.bfloat16,
            }[trt_dtype]
            out = torch.empty(shape, dtype=torch_dtype, device=DEVICE)
            context.set_tensor_address(name, out.data_ptr())
            outputs[name] = out

        stream = torch.cuda.Stream(device=DEVICE)
        with torch.cuda.stream(stream):
            ok = context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        self.assertTrue(ok, "execute_async_v3 returned False")
        return outputs

    def test_default_names_unchanged(self) -> None:
        model, inputs = self._two_output_model()
        program = _trace(model, inputs)
        engine_bytes = convert_exported_program_to_serialized_trt_engine(
            program,
            arg_inputs=inputs,
            require_full_compilation=True,
            min_block_size=1,
            use_python_runtime=False,
            immutable_weights=True,
        )
        engine = self._deserialize(engine_bytes)
        outs = self._binding_names(engine, trt.TensorIOMode.OUTPUT)
        self.assertEqual(outs, ["output0", "output1"])

    def test_user_supplied_arg_and_output_names(self) -> None:
        model, inputs = self._two_output_model()
        program = _trace(model, inputs)
        engine_bytes = convert_exported_program_to_serialized_trt_engine(
            program,
            arg_inputs=inputs,
            arg_input_binding_names=("input_image",),
            output_binding_names=("relu_out", "tanh_out"),
            require_full_compilation=True,
            min_block_size=1,
            use_python_runtime=False,
            immutable_weights=True,
        )
        engine = self._deserialize(engine_bytes)
        self.assertEqual(
            self._binding_names(engine, trt.TensorIOMode.INPUT), ["input_image"]
        )
        self.assertEqual(
            self._binding_names(engine, trt.TensorIOMode.OUTPUT),
            ["relu_out", "tanh_out"],
        )

        # Native-TRT execution: bind by the user-supplied names and verify
        # numerical match against the PyTorch eager outputs.
        (x,) = inputs
        outs = self._run_engine(engine, {"input_image": x.contiguous()})
        with torch.no_grad():
            ref_relu, ref_tanh = model(x)
        torch.testing.assert_close(outs["relu_out"], ref_relu, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(outs["tanh_out"], ref_tanh, rtol=1e-2, atol=1e-2)

    def test_kwarg_input_binding_names(self) -> None:
        """Model takes kwargs; user names them via kwarg_input_binding_names."""
        model, kwargs = self._kwarg_model()
        program = _trace(model, (), kwargs)
        engine_bytes = convert_exported_program_to_serialized_trt_engine(
            program,
            arg_inputs=(),
            kwarg_inputs=kwargs,
            kwarg_input_binding_names={"image": "img_in", "positions": "pos_in"},
            require_full_compilation=True,
            min_block_size=1,
            use_python_runtime=False,
            immutable_weights=True,
        )
        engine = self._deserialize(engine_bytes)
        self.assertEqual(
            sorted(self._binding_names(engine, trt.TensorIOMode.INPUT)),
            ["img_in", "pos_in"],
        )

        outs = self._run_engine(
            engine,
            {
                "img_in": kwargs["image"].contiguous(),
                "pos_in": kwargs["positions"].contiguous(),
            },
        )
        with torch.no_grad():
            ref = model(**kwargs)
        # Single-output model → exactly one output binding.
        only_out = next(iter(outs.values()))
        torch.testing.assert_close(only_out, ref, rtol=1e-2, atol=1e-2)

    def test_arity_mismatch_raises_before_engine_build(self) -> None:
        model, inputs = self._two_output_model()
        program = _trace(model, inputs)
        with self.assertRaises(BindingNameMismatchError):
            convert_exported_program_to_serialized_trt_engine(
                program,
                arg_inputs=inputs,
                output_binding_names=("only_one",),
                require_full_compilation=True,
                min_block_size=1,
                use_python_runtime=False,
                immutable_weights=True,
            )

    def test_duplicate_within_list_raises(self) -> None:
        model, inputs = self._two_output_model()
        program = _trace(model, inputs)
        with self.assertRaises(BindingNameMismatchError):
            convert_exported_program_to_serialized_trt_engine(
                program,
                arg_inputs=inputs,
                output_binding_names=("same", "same"),
                require_full_compilation=True,
                min_block_size=1,
                use_python_runtime=False,
                immutable_weights=True,
            )

    def test_cross_input_output_overlap_raises(self) -> None:
        model, inputs = self._two_output_model()
        program = _trace(model, inputs)
        with self.assertRaises(BindingNameMismatchError):
            convert_exported_program_to_serialized_trt_engine(
                program,
                arg_inputs=inputs,
                arg_input_binding_names=("shared_name",),
                output_binding_names=("shared_name", "tanh_out"),
                require_full_compilation=True,
                min_block_size=1,
                use_python_runtime=False,
                immutable_weights=True,
            )

    def test_pytree_dict_output(self) -> None:
        class M(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return {"primary": torch.relu(x), "secondary": torch.tanh(x)}

        x = torch.randn(2, 3, device=DEVICE, dtype=torch.float16)
        model = M().eval().cuda().half()
        program = _trace(model, (x,))
        engine_bytes = convert_exported_program_to_serialized_trt_engine(
            program,
            arg_inputs=(x,),
            output_binding_names={"primary": "p_out", "secondary": "s_out"},
            require_full_compilation=True,
            min_block_size=1,
            use_python_runtime=False,
            immutable_weights=True,
        )
        engine = self._deserialize(engine_bytes)
        self.assertEqual(
            sorted(self._binding_names(engine, trt.TensorIOMode.OUTPUT)),
            ["p_out", "s_out"],
        )

        # Bind by user-supplied names via native TRT and verify numerics.
        (input_name,) = self._binding_names(engine, trt.TensorIOMode.INPUT)
        outs = self._run_engine(engine, {input_name: x.contiguous()})
        with torch.no_grad():
            ref = model(x)
        torch.testing.assert_close(outs["p_out"], ref["primary"], rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(
            outs["s_out"], ref["secondary"], rtol=1e-2, atol=1e-2
        )


if __name__ == "__main__":
    run_tests()
