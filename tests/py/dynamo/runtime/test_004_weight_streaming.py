import itertools
import os
import unittest

import torch
import torch_tensorrt as torchtrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt._utils import is_tensorrt_rtx
from torch_tensorrt.dynamo.utils import prepare_inputs

INPUT_SIZE = (64, 100)


class SampleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(100, 128)
        self.layer2 = torch.nn.Linear(30, 64)
        self.mat1 = torch.randn((128, 32)).cuda()
        self.mat2 = torch.randn((64, 512)).cuda()
        self.relu = torch.nn.ReLU()
        self.conv = torch.nn.Conv1d(64, 6, 3)

    def forward(self, x):
        out = self.layer1(x)
        out = torch.matmul(out, self.mat1)
        out = self.relu((out + 2.0) * 0.05)
        out = self.conv(out)
        out = self.layer2(out)
        out = torch.matmul(out, self.mat2)
        return out


class TestWeightStreamingPython(TestCase):
    @parameterized.expand(
        [
            ("python_runtime", True),
            ("cpp_runtime", False),
        ]
    )
    def test_weight_streaming_default(self, _, use_python_runtime):
        model = SampleModel().eval().cuda()
        input = [torch.randn(*INPUT_SIZE, dtype=torch.float32).cuda()]
        exp_program = torch.export.export(model, tuple(input))
        optimized_model = torchtrt.dynamo.compile(
            exp_program,
            inputs=input,
            min_block_size=1,
            cache_built_engines=False,
            reuse_cached_engines=False,
            use_python_runtime=use_python_runtime,
            use_explicit_typing=True,
            enable_weight_streaming=True,
        )
        # Checking if default weight streaming budget(automatic) is applied when compiler option was provided
        weight_streaming_ctx = torchtrt.runtime.weight_streaming(optimized_model)
        assert weight_streaming_ctx.device_budget > 0

        requested_budget = int(weight_streaming_ctx.total_device_budget * 0.5)
        weight_streaming_ctx.device_budget = requested_budget

        # Weight streaming context keeps current device budget size
        with torchtrt.runtime.weight_streaming(optimized_model) as weight_streaming_ctx:
            current_budget = weight_streaming_ctx.device_budget
            assert current_budget == requested_budget

            new_budget = int(weight_streaming_ctx.total_device_budget * 0.2)
            weight_streaming_ctx.device_budget = new_budget

        # Weight streaming budget is reverted after the exit from weight streaming context
        assert weight_streaming_ctx.device_budget == requested_budget

        ref = model(*input)
        out = optimized_model(*input)

        torch.testing.assert_close(
            out.cpu(),
            ref.cpu(),
            rtol=5e-03,
            atol=5e-03,
            equal_nan=True,
            check_dtype=True,
        )
        torch._dynamo.reset()

    @parameterized.expand(
        [
            ("python_runtime", True),
            ("cpp_runtime", False),
        ]
    )
    def test_weight_streaming_manual(self, _, use_python_runtime):
        model = SampleModel().eval().cuda()
        input = [torch.randn(*INPUT_SIZE, dtype=torch.float32).cuda()]
        exp_program = torch.export.export(model, tuple(input))
        optimized_model = torchtrt.dynamo.compile(
            exp_program,
            inputs=input,
            min_block_size=1,
            cache_built_engines=False,
            reuse_cached_engines=False,
            use_python_runtime=use_python_runtime,
            use_explicit_typing=True,
            enable_weight_streaming=True,
        )
        # Weight streaming budget is applied manually.
        with torchtrt.runtime.weight_streaming(optimized_model) as weight_streaming_ctx:
            streamable_budget = weight_streaming_ctx.total_device_budget

            requested_budget = int(streamable_budget * 0.7)
            weight_streaming_ctx.device_budget = requested_budget
            assert weight_streaming_ctx.device_budget == requested_budget

            optimized_model(*input)

            # Full streaming by applying 0 budget
            weight_streaming_ctx.device_budget = 0
            assert weight_streaming_ctx.device_budget == 0

            # Automatic weight streaming size
            requested_budget = (
                weight_streaming_ctx.get_automatic_weight_streaming_budget()
            )
            weight_streaming_ctx.device_budget = requested_budget
            assert weight_streaming_ctx.device_budget == requested_budget

            requested_budget = int(streamable_budget * 0.5)
            weight_streaming_ctx.device_budget = requested_budget
            assert weight_streaming_ctx.device_budget == requested_budget

            out = optimized_model(*input)

        ref = model(*input)
        torch.testing.assert_close(
            out.cpu(),
            ref.cpu(),
            rtol=5e-03,
            atol=5e-03,
            equal_nan=True,
            check_dtype=True,
        )
        torch._dynamo.reset()

    @parameterized.expand(
        [
            ("python_runtime", True, False),
            ("python_runtime_multi_rt", True, True),
            ("cpp_runtime", False, False),
            ("cpp_runtime_multi_rt", False, True),
        ]
    )
    def test_weight_streaming_invalid_usage(self, _, use_python_runtime, multi_rt):
        model = SampleModel().eval().cuda()
        input = [torch.randn(*INPUT_SIZE, dtype=torch.float32).cuda()]
        exp_program = torch.export.export(model, tuple(input))
        optimized_model = torchtrt.dynamo.compile(
            exp_program,
            inputs=input,
            min_block_size=1,
            cache_built_engines=False,
            reuse_cached_engines=False,
            torch_executed_ops=(
                {"torch.ops.aten.convolution.default"} if multi_rt else {}
            ),
            use_python_runtime=use_python_runtime,
            use_explicit_typing=True,
            enable_weight_streaming=True,
        )

        # Setting weight streaming context to unsupported module
        with torchtrt.runtime.weight_streaming(model) as weight_streaming_ctx:
            streamable_budget = weight_streaming_ctx.total_device_budget
            assert streamable_budget == 0

        with torchtrt.runtime.weight_streaming(optimized_model) as weight_streaming_ctx:
            streamable_budget = weight_streaming_ctx.total_device_budget

            # Values is larger than max budget disables weight streaming
            weight_streaming_ctx.device_budget = streamable_budget + 1
            assert weight_streaming_ctx.device_budget == streamable_budget

            try:
                # Runtime error if requested budget is negative
                weight_streaming_ctx.device_budget = -1
                assert False
            except RuntimeError:
                assert True

            optimized_model(*input)

        torch._dynamo.reset()

    @parameterized.expand(
        [
            ("python_runtime", True),
            ("cpp_runtime", False),
        ]
    )
    def test_weight_streaming_multi_rt(self, _, use_python_runtime):
        model = SampleModel().eval().cuda()
        input = [torch.randn(*INPUT_SIZE, dtype=torch.float32).cuda()]
        exp_program = torch.export.export(model, tuple(input))

        optimized_model = torchtrt.dynamo.compile(
            exp_program,
            inputs=input,
            min_block_size=1,
            cache_built_engines=False,
            reuse_cached_engines=False,
            torch_executed_ops={"torch.ops.aten.convolution.default"},
            use_python_runtime=use_python_runtime,
            use_explicit_typing=True,
            enable_weight_streaming=True,
        )

        with torchtrt.runtime.weight_streaming(optimized_model) as weight_streaming_ctx:
            streamable_budget = weight_streaming_ctx.total_device_budget
            for pct in [0.05, 0.2, 0.4, 0.8, 1.0]:
                requested_budget = int(streamable_budget * pct)
                weight_streaming_ctx.device_budget = requested_budget

                # Budget distribution to multiple submodule may result in integer differences of at most 1
                assert abs(weight_streaming_ctx.device_budget - requested_budget) <= 1
                out = optimized_model(*input)

        ref = model(*input)
        torch.testing.assert_close(
            out.cpu(),
            ref.cpu(),
            rtol=5e-03,
            atol=5e-03,
            equal_nan=True,
            check_dtype=True,
        )

        torch._dynamo.reset()

    @parameterized.expand(
        [
            ("python_runtime", True),
            ("cpp_runtime", False),
        ]
    )
    def test_weight_streaming_cudagraphs(self, _, use_python_runtime):
        model = SampleModel().eval().cuda()
        input = [torch.randn(*INPUT_SIZE, dtype=torch.float32).cuda()]
        exp_program = torch.export.export(model, tuple(input))

        optimized_model = torchtrt.dynamo.compile(
            exp_program,
            inputs=input,
            min_block_size=1,
            cache_built_engines=False,
            reuse_cached_engines=False,
            torch_executed_ops={"torch.ops.aten.convolution.default"},
            use_python_runtime=use_python_runtime,
            use_explicit_typing=True,
            enable_weight_streaming=True,
        )

        with torchtrt.runtime.enable_cudagraphs(optimized_model) as cudagraphs_module:
            with torchtrt.runtime.weight_streaming(
                cudagraphs_module
            ) as weight_streaming_ctx:
                streamable_budget = weight_streaming_ctx.total_device_budget

                requested_budget = int(streamable_budget * 0.7)
                weight_streaming_ctx.device_budget = requested_budget
                for _ in range(4):
                    cudagraphs_module(*input)

                requested_budget = int(streamable_budget * 0.5)
                weight_streaming_ctx.device_budget = requested_budget
                for _ in range(4):
                    out = cudagraphs_module(*input)

        ref = model(*input)
        torch.testing.assert_close(
            out.cpu(),
            ref.cpu(),
            rtol=5e-03,
            atol=5e-03,
            equal_nan=True,
            check_dtype=True,
        )
        torch._dynamo.reset()

    @parameterized.expand(
        [
            ("python_runtime", True),
            ("cpp_runtime", False),
        ]
    )
    @unittest.skipIf(is_tensorrt_rtx(), "TensorRT-RTX has bug on cudagraphs")
    def test_runtime_state_change(self, _, use_python_runtime):
        class SampleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(100, 128)
                self.layer2 = torch.nn.Linear(128, 64)
                self.relu = torch.nn.ReLU()

            def forward(self, x, b=None, c=None, d=None, e=[]):
                out = self.layer1(x)
                out = out + b
                if c is not None:
                    out = out * c
                out = self.relu((out + 2.0) * 0.05)
                if d is not None:
                    out = out - d["value"] + d["value2"]
                out = self.layer2(out)
                for n in e:
                    out += n
                return out

        model = SampleModel().eval().cuda()
        input_list = []
        for batch_size in [8, 12, 12, 8, 8]:
            args = [torch.rand((batch_size, 100)).to("cuda")]
            kwargs = {
                "b": torch.rand((1, 128)).to("cuda"),
                "d": {
                    "value": torch.rand(1).to("cuda"),
                    "value2": torch.tensor(1.2).to("cuda"),
                },
                "e": [torch.rand(1).to("cuda"), torch.rand(1).to("cuda")],
            }
            input_list.append((args, kwargs))

        kwarg_torchtrt_input = prepare_inputs(input_list[0][1])

        compile_spec = {
            "arg_inputs": [
                torchtrt.Input(
                    min_shape=(1, 100),
                    opt_shape=(64, 100),
                    max_shape=(128, 100),
                    dtype=torch.float32,
                    name="x",
                ),
            ],
            "kwarg_inputs": kwarg_torchtrt_input,
            "device": torchtrt.Device("cuda:0"),
            "enabled_precisions": {torch.float},
            "pass_through_build_failures": True,
            "min_block_size": 1,
            "ir": "dynamo",
            "cache_built_engines": False,
            "reuse_cached_engines": False,
            "use_explicit_typing": True,
            "enable_weight_streaming": True,
            "torch_executed_ops": {"torch.ops.aten.mul.Tensor"},
            "use_python_runtime": use_python_runtime,
        }
        exp_program = torchtrt.dynamo.trace(model, **compile_spec)
        optimized_model = torchtrt.dynamo.compile(
            exp_program,
            **compile_spec,
        )

        # List of tuples representing different configurations for three features:
        # Cuda graphs, pre-allocated output buffer, weight streaming change
        states = list(itertools.product((True, False), repeat=3))
        # Create pairs of configurations representing an initial state and a changed state
        states_permutations = itertools.permutations(states, 2)

        def test_trt_model(enable_weight_streaming, optimized_model, input_list):
            # Test dynamic input shapes and weight streaming adjustments during inference.
            out_list = []
            weight_streaming_ctx = torchtrt.runtime.weight_streaming(optimized_model)
            streamable_budget = weight_streaming_ctx.total_device_budget
            if enable_weight_streaming:
                weight_streaming_ctx.device_budget = int(streamable_budget * 0.8)
            else:
                weight_streaming_ctx.device_budget = streamable_budget
            for i in range(len(input_list)):
                if enable_weight_streaming and i == 4:
                    weight_streaming_ctx.device_budget = int(streamable_budget * 0.6)
                out_list.append(optimized_model(*input_list[i][0], **input_list[i][1]))
            return out_list

        ref_out_list = []
        for i in range(len(input_list)):
            ref_out_list.append(model(*input_list[i][0], **input_list[i][1]))

        pre_allocated_output_ctx = torchtrt.runtime.enable_pre_allocated_outputs(
            optimized_model
        )

        for init_state, changed_state in states_permutations:
            for cuda_graphs, pre_allocated_output, weight_streaming in [
                init_state,
                changed_state,
            ]:
                pre_allocated_output_ctx.set_pre_allocated_output(pre_allocated_output)
                if cuda_graphs:
                    with torchtrt.runtime.enable_cudagraphs(
                        optimized_model
                    ) as cudagraphs_module:
                        trt_out_list = test_trt_model(
                            weight_streaming, cudagraphs_module, input_list
                        )
                else:
                    trt_out_list = test_trt_model(
                        weight_streaming, optimized_model, input_list
                    )

                for torch_model_results, optimized_model_results in zip(
                    ref_out_list, trt_out_list
                ):
                    torch.testing.assert_close(
                        torch_model_results,
                        optimized_model_results,
                        rtol=5e-03,
                        atol=5e-03,
                        equal_nan=True,
                        check_dtype=True,
                    )

        torch._dynamo.reset()


if __name__ == "__main__":
    run_tests()
