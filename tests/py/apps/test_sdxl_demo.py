# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "sdxl_demo", Path(__file__).resolve().parents[3] / "examples/apps/sdxl_demo.py"
)
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


class TestSDXLTimings(unittest.TestCase):
    def test_nested_compile_is_not_counted_as_unet_inference(self):
        clock = [0.0]

        def advance(seconds):
            clock[0] += seconds

        class UNet:
            compiled = False

            def compile(self):
                advance(5)
                self.compiled = True

            def _forward(self):
                if not self.compiled:
                    self.compile()
                advance(2)

        class VAE:
            def decode(self):
                advance(3)

        class Pipe:
            def __init__(self):
                self.unet = UNet()
                self.vae = VAE()
                self.seeds = []

            def encode_prompt(self):
                advance(1)

            def __call__(self, **kwargs):
                self.seeds.append(kwargs["generator"])
                self.encode_prompt()
                self.unet._forward()
                self.vae.decode()
                advance(4)
                return SimpleNamespace(images=["image"])

        args = SimpleNamespace(
            prompt="castle",
            negative_prompt="blurry",
            height=512,
            width=512,
            num_inference_steps=10,
            seed=42,
        )
        torch = SimpleNamespace(
            cuda=SimpleNamespace(synchronize=lambda device: None),
            no_grad=nullcontext,
            Generator=lambda device: SimpleNamespace(manual_seed=lambda seed: seed),
        )
        pipe = Pipe()
        with patch.object(demo, "perf_counter", side_effect=lambda: clock[0]):
            image, first = demo.generate(pipe, args, torch, "tensorrt")
            _, warm = demo.generate(pipe, args, torch, "tensorrt")

        self.assertEqual(image, "image")
        self.assertEqual(first["total_s"], 15)
        self.assertEqual(first["compile_s"], 5)
        self.assertEqual(first["compile_calls"], 1)
        self.assertEqual(first["unet_s"], 2)
        self.assertEqual(first["text_encoding_s"], 1)
        self.assertEqual(first["vae_decode_s"], 3)
        self.assertEqual(first["other_s"], 4)
        self.assertEqual(warm["total_s"], 10)
        self.assertEqual(warm["compile_calls"], 0)
        self.assertEqual(pipe.seeds, [42, 42])
        self.assertNotIn("_forward", vars(pipe.unet))
        self.assertNotIn("compile", vars(pipe.unet))
        self.assertNotIn("decode", vars(pipe.vae))
        self.assertNotIn("encode_prompt", vars(pipe))

    def test_method_is_restored_after_failure(self):
        def fail():
            raise ValueError("generation failed")

        obj = SimpleNamespace(forward=fail)
        timings = demo.Timings()
        with self.assertRaisesRegex(ValueError, "generation failed"):
            with timings.method(obj, "forward", "unet"):
                obj.forward()
        self.assertIs(obj.forward, fail)
        self.assertEqual(timings.calls["unet"], 1)

    def test_benchmark_excludes_first_and_extra_warmup(self):
        image = SimpleNamespace(save=lambda path: path.touch())

        def result(seconds):
            return {
                "total_s": seconds,
                "compile_calls": 0,
                "compile_s": 0,
                "text_encoding_s": 0,
                "unet_s": seconds,
                "unet_calls": 10,
                "vae_decode_s": 0,
                "other_s": 0,
            }

        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                seed=42,
                compare=True,
                output=Path(directory) / "scene.png",
                warmup=1,
                runs=2,
            )
            with (
                patch.object(
                    demo,
                    "generate",
                    side_effect=[
                        (image, result(seconds)) for seconds in (100, 200, 2, 4)
                    ],
                ),
                patch("builtins.print"),
            ):
                report = demo.benchmark(None, args, None, "pytorch")
            self.assertEqual(report["first"]["total_s"], 100)
            self.assertEqual(report["mean_s"], 3)
            self.assertEqual(report["median_s"], 3)
            self.assertEqual(len(report["runs"]), 2)
            self.assertTrue((Path(directory) / "scene_pytorch.png").is_file())


if __name__ == "__main__":
    unittest.main()
