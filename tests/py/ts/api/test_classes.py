import copy
import unittest
from typing import Dict

import torch
import torch_tensorrt as torchtrt
from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import TorchTensorRTModule


@unittest.skipIf(
    not torchtrt.ENABLED_FEATURES.torchscript_frontend,
    "TorchScript Frontend is not available",
)
class TestInput(unittest.TestCase):
    def _verify_correctness(self, struct: torchtrt.Input, target: Dict) -> bool:
        internal = struct._to_internal()

        list_eq = lambda al, bl: all([a == b for (a, b) in zip(al, bl)])

        eq = lambda a, b: a == b

        def field_is_correct(field, equal_fn, a1, a2):
            equal = equal_fn(a1, a2)
            if not equal:
                print("\nField {} is incorrect: {} != {}".format(field, a1, a2))
            return equal

        min_ = field_is_correct("min", list_eq, internal.min, target["min"])
        opt_ = field_is_correct("opt", list_eq, internal.opt, target["opt"])
        max_ = field_is_correct("max", list_eq, internal.max, target["max"])
        is_dynamic_ = field_is_correct(
            "is_dynamic", eq, internal.input_is_dynamic, target["input_is_dynamic"]
        )
        explicit_set_dtype_ = field_is_correct(
            "explicit_dtype",
            eq,
            internal._explicit_set_dtype,
            target["explicit_set_dtype"],
        )
        dtype_ = field_is_correct(
            "dtype",
            eq,
            torchtrt.dtype._from(internal.dtype),
            torchtrt.dtype._from(target["dtype"]),
        )
        format_ = field_is_correct(
            "format",
            eq,
            torchtrt.memory_format._from(internal.format),
            torchtrt.memory_format._from(target["format"]),
        )

        return all(
            [min_, opt_, max_, is_dynamic_, explicit_set_dtype_, dtype_, format_]
        )

    def test_infer_from_example_tensor(self):
        shape = [1, 3, 255, 255]
        target = {
            "min": shape,
            "opt": shape,
            "max": shape,
            "input_is_dynamic": False,
            "dtype": torchtrt.dtype.half,
            "format": torchtrt.memory_format.contiguous,
            "explicit_set_dtype": True,
        }

        example_tensor = torch.randn(shape).half()
        i = torchtrt.Input.from_tensor(example_tensor)
        ts_i = torchtrt.ts.TorchScriptInput(
            shape=i.shape, dtype=i.dtype, format=i.format
        )
        self.assertTrue(self._verify_correctness(ts_i, target))

    def test_static_shape(self):
        shape = [1, 3, 255, 255]
        target = {
            "min": shape,
            "opt": shape,
            "max": shape,
            "input_is_dynamic": False,
            "dtype": torchtrt.dtype.unknown,
            "format": torchtrt.memory_format.contiguous,
            "explicit_set_dtype": False,
        }

        i = torchtrt.Input(shape)
        ts_i = torchtrt.ts.TorchScriptInput(
            shape=i.shape, dtype=i.dtype, format=i.format
        )
        self.assertTrue(self._verify_correctness(ts_i, target))

        i = torchtrt.Input(tuple(shape))
        ts_i = torchtrt.ts.TorchScriptInput(
            shape=i.shape, dtype=i.dtype, format=i.format
        )
        self.assertTrue(self._verify_correctness(ts_i, target))

        i = torchtrt.Input(torch.randn(shape).shape)
        ts_i = torchtrt.ts.TorchScriptInput(
            shape=i.shape, dtype=i.dtype, format=i.format
        )
        self.assertTrue(self._verify_correctness(ts_i, target))

        i = torchtrt.Input(shape=shape)
        ts_i = torchtrt.ts.TorchScriptInput(
            shape=i.shape, dtype=i.dtype, format=i.format
        )
        self.assertTrue(self._verify_correctness(ts_i, target))

        i = torchtrt.Input(shape=tuple(shape))
        ts_i = torchtrt.ts.TorchScriptInput(
            shape=i.shape, dtype=i.dtype, format=i.format
        )
        self.assertTrue(self._verify_correctness(ts_i, target))

        i = torchtrt.Input(shape=torch.randn(shape).shape)
        ts_i = torchtrt.ts.TorchScriptInput(
            shape=i.shape, dtype=i.dtype, format=i.format
        )
        self.assertTrue(self._verify_correctness(ts_i, target))

    def test_data_type(self):
        shape = [1, 3, 255, 255]
        target = {
            "min": shape,
            "opt": shape,
            "max": shape,
            "input_is_dynamic": False,
            "dtype": torchtrt.dtype.half,
            "format": torchtrt.memory_format.contiguous,
            "explicit_set_dtype": True,
        }

        i = torchtrt.Input(shape, dtype=torchtrt.dtype.half)
        ts_i = torchtrt.ts.TorchScriptInput(
            shape=i.shape, dtype=i.dtype, format=i.format
        )
        self.assertTrue(self._verify_correctness(ts_i, target))

        i = torchtrt.Input(shape, dtype=torch.half)
        ts_i = torchtrt.ts.TorchScriptInput(
            shape=i.shape, dtype=i.dtype, format=i.format
        )
        self.assertTrue(self._verify_correctness(ts_i, target))

    def test_tensor_format(self):
        shape = [1, 3, 255, 255]
        target = {
            "min": shape,
            "opt": shape,
            "max": shape,
            "input_is_dynamic": False,
            "dtype": torchtrt.dtype.unknown,
            "format": torchtrt.memory_format.channels_last,
            "explicit_set_dtype": False,
        }

        i = torchtrt.Input(shape, format=torchtrt.memory_format.channels_last)
        ts_i = torchtrt.ts.TorchScriptInput(
            shape=i.shape, dtype=i.dtype, format=i.format
        )
        self.assertTrue(self._verify_correctness(ts_i, target))

        i = torchtrt.Input(shape, format=torch.channels_last)
        ts_i = torchtrt.ts.TorchScriptInput(
            shape=i.shape, dtype=i.dtype, format=i.format
        )
        self.assertTrue(self._verify_correctness(ts_i, target))

    def test_dynamic_shape(self):
        min_shape = [1, 3, 128, 128]
        opt_shape = [1, 3, 256, 256]
        max_shape = [1, 3, 512, 512]
        target = {
            "min": min_shape,
            "opt": opt_shape,
            "max": max_shape,
            "input_is_dynamic": True,
            "dtype": torchtrt.dtype.unknown,
            "format": torchtrt.memory_format.contiguous,
            "explicit_set_dtype": False,
        }

        i = torchtrt.Input(
            min_shape=min_shape, opt_shape=opt_shape, max_shape=max_shape
        )
        ts_i = torchtrt.ts.TorchScriptInput(
            min_shape=i.shape["min_shape"],
            opt_shape=i.shape["opt_shape"],
            max_shape=i.shape["max_shape"],
            dtype=i.dtype,
            format=i.format,
        )
        self.assertTrue(self._verify_correctness(ts_i, target))

        i = torchtrt.Input(
            min_shape=tuple(min_shape),
            opt_shape=tuple(opt_shape),
            max_shape=tuple(max_shape),
        )
        ts_i = torchtrt.ts.TorchScriptInput(
            min_shape=i.shape["min_shape"],
            opt_shape=i.shape["opt_shape"],
            max_shape=i.shape["max_shape"],
            dtype=i.dtype,
            format=i.format,
        )
        self.assertTrue(self._verify_correctness(ts_i, target))

        tensor_shape = lambda shape: torch.randn(shape).shape
        i = torchtrt.Input(
            min_shape=tensor_shape(min_shape),
            opt_shape=tensor_shape(opt_shape),
            max_shape=tensor_shape(max_shape),
        )
        ts_i = torchtrt.ts.TorchScriptInput(
            min_shape=i.shape["min_shape"],
            opt_shape=i.shape["opt_shape"],
            max_shape=i.shape["max_shape"],
            dtype=i.dtype,
            format=i.format,
        )
        self.assertTrue(self._verify_correctness(ts_i, target))


@unittest.skipIf(
    not torchtrt.ENABLED_FEATURES.torchscript_frontend,
    "TorchScript Frontend is not available",
)
class TestTorchTensorRTModule(unittest.TestCase):
    @staticmethod
    def _get_trt_mod(via_ts: bool = False):
        class Test(torch.nn.Module):
            def __init__(self):
                super(Test, self).__init__()
                self.fc1 = torch.nn.Linear(10, 5)
                self.fc2 = torch.nn.Linear(5, 5)

            def forward(self, x):
                out = self.fc2(self.fc1(x))
                return out

        mod = torch.jit.script(Test())
        if via_ts:
            test_mod_engine_str = torchtrt.ts.convert_method_to_trt_engine(
                mod, "forward", inputs=[torchtrt.Input((2, 10))]
            )
        else:
            test_mod_engine_str = torchtrt.convert_method_to_trt_engine(
                mod, "forward", inputs=[torchtrt.Input((2, 10))]
            )
        return TorchTensorRTModule(
            name="test",
            serialized_engine=test_mod_engine_str,
            input_binding_names=["input_0"],
            output_binding_names=["output_0"],
        )

    def test_detect_invalid_input_binding(self):
        class Test(torch.nn.Module):
            def __init__(self):
                super(Test, self).__init__()
                self.fc1 = torch.nn.Linear(10, 5)
                self.fc2 = torch.nn.Linear(5, 5)

            def forward(self, x):
                out = self.fc2(self.fc1(x))
                return out

        mod = torch.jit.script(Test())
        test_mod_engine_str = torchtrt.ts.convert_method_to_trt_engine(
            mod, "forward", inputs=[torchtrt.Input((2, 10))]
        )
        with self.assertRaises(RuntimeError):
            TorchTensorRTModule(
                name="test",
                serialized_engine=test_mod_engine_str,
                input_binding_names=["x.1"],
                output_binding_names=["output_0"],
            )

    def test_detect_invalid_output_binding(self):
        class Test(torch.nn.Module):
            def __init__(self):
                super(Test, self).__init__()
                self.fc1 = torch.nn.Linear(10, 5)
                self.fc2 = torch.nn.Linear(5, 5)

            def forward(self, x):
                out = self.fc2(self.fc1(x))
                return out

        mod = torch.jit.script(Test())
        test_mod_engine_str = torchtrt.ts.convert_method_to_trt_engine(
            mod, "forward", inputs=[torchtrt.Input((2, 10))]
        )
        with self.assertRaises(RuntimeError):
            TorchTensorRTModule(
                name="test",
                serialized_engine=test_mod_engine_str,
                input_binding_names=["input_0"],
                output_binding_names=["z.1"],
            )

    def test_set_get_profile_path_prefix(self):
        for trt_mod in (
            TestTorchTensorRTModule._get_trt_mod(),
            TestTorchTensorRTModule._get_trt_mod(via_ts=True),
        ):
            trt_mod.engine.profile_path_prefix = "/tmp/"
            self.assertTrue(trt_mod.engine.profile_path_prefix == "/tmp/")

    @unittest.skipIf(
        torchtrt.ENABLED_FEATURES.tensorrt_rtx,
        "layer info is different for tensorrt_rtx",
    )
    def test_get_layer_info(self):
        """
        {
            "Layers": [
                "%26 : Tensor = aten::matmul(%x.1, %25)_myl0_0",
                "%31 : Tensor = aten::matmul(%28, %30)_myl0_1"
            ],
            "Bindings": [
                "input_0",
                "output_0"
            ]
        }
        """

        import json

        for trt_mod in (
            TestTorchTensorRTModule._get_trt_mod(),
            TestTorchTensorRTModule._get_trt_mod(via_ts=True),
        ):
            layer_info = trt_mod.get_layer_info()
            trt_json = json.loads(layer_info)
            [self.assertTrue(k in trt_json.keys(), f"Key {k} is missing") for k in ["Layers", "Bindings"]]
            self.assertTrue(len(trt_json["Layers"]) == 4, "Not enough layers found")
            self.assertTrue(len(trt_json["Bindings"]) == 2, "Not enough bindings found")


if __name__ == "__main__":
    unittest.main()
