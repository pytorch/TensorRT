"""Unit tests for Global Performance Tuner route parsing and accuracy metrics."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings, settings_are_compatible
from torch_tensorrt.dynamo.tuning.accuracy import (
    compute_tensor_loss,
    loss_cos,
    loss_l0,
    loss_l1,
    loss_l2,
    loss_linf,
)
from torch_tensorrt.dynamo.tuning.routes import (
    BuildRouteKnobDatabase,
    expand_build_routes,
    expand_routes_mixed,
    identify_positive_knobs,
    resolve_tuning_expression,
)
from torch_tensorrt.dynamo.tuning.sweeper import validate_tuning_options


def _sample_knob_db_json() -> str:
    return json.dumps(
        {
            "tuner_version": "test-1.0",
            "tuner_options": [
                {
                    "option": "-slice_fusion",
                    "allowed_values": "-slice_fusion=[on|off]",
                    "default_value": "on",
                    "help": "slice fusion",
                },
                {
                    "option": "-copy_ppg",
                    "allowed_values": "-copy_ppg=[on|off]",
                    "default_value": "on",
                    "help": "copy ppg",
                },
                {
                    "option": "-kgen:codegen:cuda_tile",
                    "allowed_values": "-kgen:codegen:cuda_tile=[0|1|2|3]",
                    "default_value": "1",
                    "help": "cuda tile",
                },
            ],
        }
    )


class TestBuildRouteParsing(unittest.TestCase):
    def setUp(self) -> None:
        self.db = BuildRouteKnobDatabase()
        assert self.db.load_from_json(_sample_knob_db_json())

    def test_full_expansion_two_binary(self) -> None:
        exprs, routes = expand_build_routes(
            "-slice_fusion=[on|off] -copy_ppg=[on|off]", "full", self.db
        )
        self.assertEqual(len(exprs), 2)
        self.assertEqual(len(routes), 4)
        self.assertIn("-slice_fusion=on -copy_ppg=on", routes)
        self.assertIn("-slice_fusion=off -copy_ppg=off", routes)

    def test_fast_expansion_linear(self) -> None:
        exprs, routes = expand_build_routes(
            "-slice_fusion=[on|off] -copy_ppg=[on|off]", "fast", self.db
        )
        # baseline + one off for each binary knob = 3
        self.assertEqual(len(routes), 3)
        self.assertEqual(routes[0], "-slice_fusion=on -copy_ppg=on")

    def test_dry_run_rejects_mixed(self) -> None:
        with self.assertRaises(ValueError):
            expand_build_routes(
                "-slice_fusion=[on|off]", "mixed", self.db, dry_run=True
            )

    def test_unknown_knob(self) -> None:
        with self.assertRaises(ValueError):
            expand_build_routes("-not_a_real_knob=[on|off]", "fast", self.db)

    def test_fixed_and_variable(self) -> None:
        exprs, routes = expand_build_routes(
            "-slice_fusion=off -copy_ppg=[on|off]", "full", self.db
        )
        self.assertEqual(len(routes), 2)
        self.assertTrue(all(r.startswith("-slice_fusion=off") for r in routes))

    def test_identify_positive_knobs(self) -> None:
        exprs, routes = expand_build_routes(
            "-slice_fusion=[on|off] -copy_ppg=[on|off]", "fast", self.db
        )
        self.assertEqual(len(routes), 3)
        # baseline slow, first one-off faster -> positive slice_fusion
        gpu_times = [10.0, 5.0, 11.0]
        positive = identify_positive_knobs(exprs, gpu_times, self.db)
        self.assertEqual(positive, [0])
        mixed = expand_routes_mixed(exprs, self.db, positive)
        self.assertTrue(any("slice_fusion=off" in r for r in mixed))

    def test_identify_positive_knobs_2(self) -> None:
        exprs, routes = expand_build_routes(
            "-slice_fusion=[on|off] -copy_ppg=[on|off]", "fast", self.db
        )
        self.assertEqual(len(routes), 3)
        # baseline slow, first one-off faster -> positive slice_fusion
        gpu_times = [10.0, 5.0, 3.0]
        positive = identify_positive_knobs(exprs, gpu_times, self.db)
        self.assertEqual(positive, [0, 1])
        mixed = expand_routes_mixed(exprs, self.db, positive)
        self.assertTrue(any("slice_fusion=off" in r for r in mixed))
        self.assertTrue(any("copy_ppg=off" in r for r in mixed))


class TestAccuracyMetrics(unittest.TestCase):
    def test_perfect_match_zero_loss(self) -> None:
        t = torch.randn(8, 8)
        self.assertEqual(loss_l0(t, t), 0.0)
        self.assertEqual(loss_l1(t, t), 0.0)
        self.assertEqual(loss_l2(t, t), 0.0)
        self.assertEqual(loss_linf(t, t), 0.0)
        self.assertAlmostEqual(loss_cos(t, t), 0.0, places=5)

    def test_l0_fraction(self) -> None:
        ref = torch.zeros(4)
        actual = torch.tensor([0.0, 0.0, 1.0, 1.0])
        # atol=rtol=0 => half the elements differ
        self.assertAlmostEqual(loss_l0(actual, ref, atol=0.0, rtol=0.0), 0.5)

    def test_algorithm_dispatch(self) -> None:
        a = torch.ones(3)
        b = torch.zeros(3)
        self.assertGreater(compute_tensor_loss(a, b, "l1"), 0.0)
        self.assertGreater(compute_tensor_loss(a, b, "lInf"), 0.0)


class TestSettingsAndValidation(unittest.TestCase):
    def test_build_route_engine_invariant(self) -> None:
        a = CompilationSettings(build_route="")
        b = CompilationSettings(build_route="-slice_fusion=off")
        ok, incompatible = settings_are_compatible(a, b)
        self.assertFalse(ok)
        self.assertIn("build_route", incompatible)

    def test_validate_mutually_exclusive_exprs(self) -> None:
        with self.assertRaises(ValueError):
            validate_tuning_options(
                CompilationSettings(
                    tune_build_routes="-a=[on|off]",
                    tune_build_route_file="/tmp/x.txt",
                )
            )

    def test_validate_continue_requires_cache(self) -> None:
        with self.assertRaises(ValueError):
            validate_tuning_options(CompilationSettings(tuning_continue=True))

    def test_validate_dry_run_mixed(self) -> None:
        with self.assertRaises(ValueError):
            validate_tuning_options(
                CompilationSettings(tuning_dry_run=True, tuning_search="mixed")
            )

    def test_partition_cache_path(self) -> None:
        from torch_tensorrt.dynamo.tuning.cache import (
            resolve_partition_tuning_cache_path,
            subgraph_partition_key,
        )

        g = torch.fx.symbolic_trace(torch.nn.ReLU())
        key = subgraph_partition_key(g)
        path = resolve_partition_tuning_cache_path("/tmp/tune.jsonl", g)
        self.assertEqual(path, f"/tmp/tune.{key}.jsonl")
        self.assertIsNone(resolve_partition_tuning_cache_path(None, g))

    def test_tune_expr_from_file(self) -> None:
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write("-slice_fusion=[on|off]\n")
            f.write("-copy_ppg=[on|off]\n")
            path = f.name
        expr = resolve_tuning_expression(tune_build_route_file=path)
        self.assertIn("-slice_fusion=[on|off]", expr)
        self.assertIn("-copy_ppg=[on|off]", expr)


class TestGPTAvailabilityAndIntegration(unittest.TestCase):
    def test_capability_probe(self) -> None:
        from torch_tensorrt.dynamo.tuning import is_global_perf_tuner_available

        # Should not raise; result depends on local TensorRT build.
        available = is_global_perf_tuner_available()
        self.assertIsInstance(available, bool)

    def test_small_tune_sweep(self) -> None:
        from torch_tensorrt.dynamo.tuning import (
            get_all_build_routes,
            is_global_perf_tuner_available,
        )

        if not torch.cuda.is_available() or not is_global_perf_tuner_available():
            self.skipTest("Global Performance Tuner or CUDA unavailable")

        import torch_tensorrt

        knobs = get_all_build_routes()
        options = knobs.get("tuner_options", [])
        binary = None
        for opt in options:
            allowed = opt.get("allowed_values", "")
            if "=[on|off]" in allowed:
                binary = opt["option"]
                break
        if binary is None:
            self.skipTest("No binary on/off knob found in tuner database")

        class Tiny(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.relu(x + 1.0)

        model = Tiny().eval().cuda()
        x = torch.randn(1, 8, device="cuda")
        expr = f"{binary}=[on|off]"
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            cache_base = f.name

        compiled = torch_tensorrt.compile(
            model,
            ir="dynamo",
            arg_inputs=[x],
            min_block_size=1,
            tune_build_routes=expr,
            tuning_search="full",
            accuracy_threshold=0.5,
            accuracy_algorithm="cos",
            tuning_cache_file=cache_base,
        )
        with torch.no_grad():
            out = compiled(x)
            ref = model(x)
        self.assertTrue(torch.allclose(out, ref, rtol=1e-2, atol=1e-2))

        cache_dir = os.path.dirname(cache_base) or "."
        cache_stem = os.path.splitext(os.path.basename(cache_base))[0]
        partition_caches = [
            os.path.join(cache_dir, name)
            for name in os.listdir(cache_dir)
            if name.startswith(cache_stem + ".") and name.endswith(".jsonl")
        ]
        self.assertTrue(partition_caches, "expected per-partition tuning cache file")
        with open(partition_caches[0], "r", encoding="utf-8") as f:
            lines = [ln for ln in f.readlines() if ln.strip()]
        self.assertGreaterEqual(len(lines), 3)  # header + 2 iters


if __name__ == "__main__":
    unittest.main()
