"""Unit tests for Global Performance Tuning route parsing and accuracy metrics."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings, settings_are_compatible
from torch_tensorrt.dynamo.tuning.accuracy import (
    accuracy_failed,
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
from torch_tensorrt.dynamo.tuning.sweeper import (
    should_run_tuning,
    validate_tuning_options,
)


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

    def test_nonfinite_loss_always_fails(self) -> None:
        self.assertTrue(accuracy_failed({"output": float("nan")}, 1.0))
        self.assertTrue(accuracy_failed({"output": float("inf")}, None))

    def test_all_metrics_reject_broadcastable_shape_mismatch(self) -> None:
        actual = torch.ones(2, 1)
        reference = torch.ones(2, 3)
        for metric in (loss_l0, loss_l1, loss_l2, loss_linf, loss_cos):
            with self.subTest(metric=metric.__name__):
                with self.assertRaisesRegex(ValueError, "Shape mismatch"):
                    metric(actual, reference)


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

    def test_dry_run_alone_is_requested_but_invalid(self) -> None:
        settings = CompilationSettings(tuning_dry_run=True)
        self.assertTrue(should_run_tuning(settings))
        with self.assertRaisesRegex(ValueError, "requires tune_build_routes"):
            validate_tuning_options(settings)

    def test_interpret_routes_dry_run_only_settings_to_validation(self) -> None:
        from torch_tensorrt._Input import Input
        from torch_tensorrt.dynamo.conversion._conversion import (
            interpret_module_to_result,
        )

        with mock.patch("torch_tensorrt.dynamo.tuning.require_global_perf_tuning"):
            with self.assertRaisesRegex(ValueError, "requires tune_build_routes"):
                interpret_module_to_result(
                    torch.fx.symbolic_trace(torch.nn.ReLU()),
                    [Input((1,))],
                    CompilationSettings(tuning_dry_run=True),
                )

    def test_accuracy_options_must_be_finite_and_nonnegative(self) -> None:
        invalid_settings = (
            CompilationSettings(accuracy_threshold=float("nan")),
            CompilationSettings(accuracy_threshold=-1.0),
            CompilationSettings(accuracy_atol=float("inf")),
            CompilationSettings(accuracy_rtol=-1.0),
        )
        for settings in invalid_settings:
            with self.subTest(settings=settings):
                with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                    validate_tuning_options(settings)

    def test_child_exit_message(self) -> None:
        from torch_tensorrt.dynamo.tuning.sweeper import _child_exit_message

        self.assertIn("signal 6", _child_exit_message(-6))
        self.assertIn("code 1", _child_exit_message(1))

    def test_fx_meta_snapshot_drops_unpicklable_values(self) -> None:
        import io

        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch_tensorrt.dynamo.tuning.sweeper import _snapshot_fx_meta

        class Tiny(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.relu(x)

        gm = torch.fx.symbolic_trace(Tiny())
        mode = FakeTensorMode()
        with mode:
            fake = mode.from_tensor(torch.randn(2, 3))
        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue
            node.meta["val"] = fake
            node.meta["original_aten"] = torch.ops.aten.relu.default
            node.meta["keep"] = 1
        snap = _snapshot_fx_meta(gm)
        buf = io.BytesIO()
        torch.save(snap, buf)
        relu_idx = [i for i, n in enumerate(gm.graph.nodes) if n.name == "relu"][0]
        relu_meta = snap["node_meta_list"][relu_idx]
        self.assertNotIn("original_aten", relu_meta)
        self.assertEqual(relu_meta["keep"], 1)
        self.assertIs(type(relu_meta["val"]), torch.Tensor)
        self.assertEqual(tuple(relu_meta["val"].shape), (2, 3))

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


class TestTuningCacheAndIPC(unittest.TestCase):
    def test_trtexec_style_cache_is_read(self) -> None:
        from torch_tensorrt.dynamo.tuning.cache import read_cache, read_iterations

        header = {
            "tuner_version": "test-1.0",
            "searching_algorithm": "fast",
            "tuning_expr": "-a=[on|off]",
        }
        row = {
            "iter": 0,
            "build_route": "-a=on",
            "crash": False,
            "error_message": "",
            "accuracy_loss": None,
            "gpu_time": 1.0,
        }
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(json.dumps(header) + "\n")
            f.write(json.dumps(row) + "\n")
            path = f.name
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        parsed_header = read_cache(path)
        self.assertEqual(parsed_header.tuning_expr, "-a=[on|off]")
        self.assertEqual(parsed_header.completed_iterations, 1)
        self.assertEqual(read_iterations(path), [row])

    def test_generated_sample_values_are_repeatable_without_rng_side_effect(
        self,
    ) -> None:
        from torch_tensorrt._Input import Input
        from torch_tensorrt.dynamo.tuning.sweeper import (
            _materialize_sample_tensors,
        )

        input_spec = Input((2, 3))
        rng_before = torch.random.get_rng_state()
        first = _materialize_sample_tensors([input_spec])
        rng_after = torch.random.get_rng_state()
        second = _materialize_sample_tensors([input_spec])
        self.assertTrue(torch.equal(rng_before, rng_after))
        self.assertTrue(torch.equal(first[0], second[0]))

    def test_child_nonzero_exit_overrides_result_file(self) -> None:
        from torch_tensorrt._Input import Input
        from torch_tensorrt.dynamo.tuning import sweeper

        class Process:
            exitcode = -6

            def __init__(self, args):
                self.args = args

            def start(self):
                torch.save({"crashed": False}, self.args[1])

            def join(self, timeout=None):
                pass

            def is_alive(self):
                return False

        class Context:
            def Process(self, *, target, args, daemon):
                return Process(args)

        module = torch.fx.symbolic_trace(torch.nn.ReLU())
        with mock.patch.object(
            sweeper.multiprocessing, "get_context", return_value=Context()
        ):
            result = sweeper._spawn_route_trial(
                module=module,
                inputs=[Input((1,))],
                sample_tensors=[torch.ones(1)],
                trial_settings=CompilationSettings(),
                route="-a=on",
                iter_idx=0,
                input_binding_names=None,
                output_binding_names=None,
                join_timeout_s=None,
            )
        self.assertTrue(result["crashed"])
        self.assertIn("signal 6", result["error_message"])

    def test_corrupt_child_result_becomes_crashed_trial(self) -> None:
        from torch_tensorrt._Input import Input
        from torch_tensorrt.dynamo.tuning import sweeper

        class Process:
            exitcode = 0

            def __init__(self, args):
                self.args = args

            def start(self):
                with open(self.args[1], "wb") as f:
                    f.write(b"not a torch result")

            def join(self, timeout=None):
                pass

            def is_alive(self):
                return False

        class Context:
            def Process(self, *, target, args, daemon):
                return Process(args)

        module = torch.fx.symbolic_trace(torch.nn.ReLU())
        created_dirs = []
        real_temporary_directory = tempfile.TemporaryDirectory

        def temporary_directory(*args, **kwargs):
            directory = real_temporary_directory(*args, **kwargs)
            created_dirs.append(directory.name)
            return directory

        with (
            mock.patch.object(
                sweeper.multiprocessing, "get_context", return_value=Context()
            ),
            mock.patch.object(
                sweeper.tempfile,
                "TemporaryDirectory",
                side_effect=temporary_directory,
            ),
        ):
            result = sweeper._spawn_route_trial(
                module=module,
                inputs=[Input((1,))],
                sample_tensors=[torch.ones(1)],
                trial_settings=CompilationSettings(),
                route="-a=on",
                iter_idx=0,
                input_binding_names=None,
                output_binding_names=None,
                join_timeout_s=None,
            )
        self.assertTrue(result["crashed"])
        self.assertIn("failed to load child result", result["error_message"])
        self.assertTrue(created_dirs)
        self.assertFalse(os.path.exists(created_dirs[0]))

    def test_missing_child_result_becomes_crashed_trial(self) -> None:
        from torch_tensorrt._Input import Input
        from torch_tensorrt.dynamo.tuning import sweeper

        process = mock.Mock(exitcode=0)
        process.is_alive.return_value = False
        context = mock.Mock()
        context.Process.return_value = process
        with mock.patch.object(
            sweeper.multiprocessing, "get_context", return_value=context
        ):
            result = sweeper._spawn_route_trial(
                module=torch.fx.symbolic_trace(torch.nn.ReLU()),
                inputs=[Input((1,))],
                sample_tensors=[torch.ones(1)],
                trial_settings=CompilationSettings(),
                route="-a=on",
                iter_idx=0,
                input_binding_names=None,
                output_binding_names=None,
                join_timeout_s=None,
            )
        self.assertTrue(result["crashed"])
        self.assertIn("without a result file", result["error_message"])

    def test_malformed_child_result_becomes_crashed_trial(self) -> None:
        from torch_tensorrt._Input import Input
        from torch_tensorrt.dynamo.tuning import sweeper

        malformed = sweeper._empty_trial_result(crashed=False, gpu_time=1.0)
        malformed["serialized_engine"] = b"engine"
        malformed["input_names"] = 1

        class Process:
            exitcode = 0

            def __init__(self, args):
                self.args = args

            def start(self):
                torch.save(malformed, self.args[1])

            def join(self, timeout=None):
                pass

            def is_alive(self):
                return False

        context = mock.Mock()
        context.Process.side_effect = lambda *, target, args, daemon: Process(args)
        with mock.patch.object(
            sweeper.multiprocessing, "get_context", return_value=context
        ):
            result = sweeper._spawn_route_trial(
                module=torch.fx.symbolic_trace(torch.nn.ReLU()),
                inputs=[Input((1,))],
                sample_tensors=[torch.ones(1)],
                trial_settings=CompilationSettings(),
                route="-a=on",
                iter_idx=0,
                input_binding_names=None,
                output_binding_names=None,
                join_timeout_s=None,
            )
        self.assertTrue(result["crashed"])
        self.assertIn("malformed child result", result["error_message"])

    def test_child_publishes_result_atomically(self) -> None:
        from torch_tensorrt.dynamo.tuning import sweeper

        with tempfile.TemporaryDirectory() as directory:
            payload_path = os.path.join(directory, "payload.pt")
            result_path = os.path.join(directory, "result.pt")
            torch.save({}, payload_path)
            expected = sweeper._empty_trial_result(crashed=False, gpu_time=1.0)
            with mock.patch.object(
                sweeper, "_execute_route_trial", return_value=expected
            ):
                sweeper._tuning_trial_entry(payload_path, result_path)
            self.assertEqual(
                torch.load(result_path, weights_only=False)["gpu_time"], 1.0
            )
            self.assertFalse(any(".partial." in name for name in os.listdir(directory)))


def _successful_trial(gpu_time: float, engine: bytes = b"engine"):
    return {
        "crashed": False,
        "error_message": "",
        "accuracy_loss": None,
        "gpu_time": gpu_time,
        "serialized_engine": engine,
        "input_names": ["x"],
        "output_names": ["output"],
        "requires_output_allocator": False,
        "symbolic_shape_expressions": {},
        "requires_native_multidevice": False,
        "aliased_io": {},
    }


class TestSweepReliability(unittest.TestCase):
    def setUp(self) -> None:
        from torch_tensorrt._Input import Input

        self.module = torch.fx.symbolic_trace(torch.nn.ReLU())
        self.inputs = [Input((1,))]
        self.knobs = _sample_knob_db_json()

    def _patch_sweep(self, trial_side_effect):
        from torch_tensorrt.dynamo.tuning import sweeper

        return (
            mock.patch.object(sweeper, "require_global_perf_tuning"),
            mock.patch.object(
                sweeper, "get_all_build_routes_raw", return_value=self.knobs
            ),
            mock.patch.object(
                sweeper,
                "_materialize_sample_tensors",
                return_value=[torch.tensor([0.25])],
            ),
            mock.patch.object(
                sweeper, "_spawn_route_trial", side_effect=trial_side_effect
            ),
        )

    def _write_resume_cache(
        self,
        base_path,
        settings,
        expression,
        phase1_times,
        phase2_times=(),
    ):
        from torch_tensorrt.dynamo.tuning import cache
        from torch_tensorrt.dynamo.tuning.routes import (
            BuildRouteKnobDatabase,
            expand_build_routes,
            expand_routes_mixed,
            identify_positive_knobs,
        )

        db = BuildRouteKnobDatabase()
        self.assertTrue(db.load_from_json(self.knobs))
        path = cache.resolve_partition_tuning_cache_path(base_path, self.module)
        cache.write_header(
            path,
            {
                "tuner_version": db.tuner_version,
                "accuracy_algorithm": settings.accuracy_algorithm,
                "accuracy_parameter": {
                    "atol": settings.accuracy_atol,
                    "rtol": settings.accuracy_rtol,
                    "epsilon": settings.accuracy_threshold,
                },
                "searching_algorithm": "mixed",
                "tuning_expr": expression,
            },
        )
        exprs, phase1_routes = expand_build_routes(expression, "mixed", db)
        for index, gpu_time in enumerate(phase1_times):
            cache.append_iteration(
                path,
                iter_idx=index,
                build_route=phase1_routes[index],
                crashed=False,
                gpu_time_ms=gpu_time,
            )
        positive = identify_positive_knobs(exprs, phase1_times, db)
        all_phase2 = expand_routes_mixed(exprs, db, positive)
        phase2_routes = [
            route for route in all_phase2 if route not in set(phase1_routes)
        ]
        for index, gpu_time in enumerate(phase2_times):
            cache.append_iteration(
                path,
                iter_idx=len(phase1_routes) + index,
                build_route=phase2_routes[index],
                crashed=False,
                gpu_time_ms=gpu_time,
            )
        return phase1_routes, phase2_routes

    def test_mixed_resume_finishes_phase1_then_phase2(self) -> None:
        from torch_tensorrt.dynamo.tuning.sweeper import tune_subgraph

        expression = "-slice_fusion=[on|off] -copy_ppg=[on|off]"
        with tempfile.TemporaryDirectory() as directory:
            base = os.path.join(directory, "tune.jsonl")
            initial = CompilationSettings(
                tune_build_routes=expression,
                tuning_search="mixed",
                tuning_cache_file=base,
            )
            phase1, _ = self._write_resume_cache(base, initial, expression, [10.0])
            calls = []

            def trial(**kwargs):
                calls.append(kwargs)
                return _successful_trial(float(8 - len(calls)), bytes([len(calls)]))

            patches = self._patch_sweep(trial)
            with patches[0], patches[1], patches[2], patches[3]:
                result = tune_subgraph(
                    self.module,
                    self.inputs,
                    CompilationSettings(tuning_continue=True, tuning_cache_file=base),
                )
            called_routes = [call["route"] for call in calls]
            self.assertEqual(called_routes[:2], phase1[1:])
            self.assertEqual(len(called_routes), 3)
            self.assertNotIn(called_routes[-1], phase1)
            self.assertEqual(result.serialized_engine, bytes([len(calls)]))
            sample_ids = {id(call["sample_tensors"]) for call in calls}
            self.assertEqual(len(sample_ids), 1)

    def test_mixed_resume_continues_phase2_without_repeating(self) -> None:
        from torch_tensorrt.dynamo.tuning.sweeper import tune_subgraph

        expression = (
            "-slice_fusion=[on|off] -copy_ppg=[on|off] -kgen:codegen:cuda_tile=[0|1]"
        )
        with tempfile.TemporaryDirectory() as directory:
            base = os.path.join(directory, "tune.jsonl")
            initial = CompilationSettings(
                tune_build_routes=expression,
                tuning_search="mixed",
                tuning_cache_file=base,
            )
            phase1, phase2 = self._write_resume_cache(
                base, initial, expression, [10.0, 9.0, 8.0, 7.0], [6.0]
            )
            calls = []

            def trial(**kwargs):
                calls.append(kwargs)
                return _successful_trial(float(5 - len(calls)), bytes([len(calls)]))

            patches = self._patch_sweep(trial)
            with patches[0], patches[1], patches[2], patches[3]:
                tune_subgraph(
                    self.module,
                    self.inputs,
                    CompilationSettings(tuning_continue=True, tuning_cache_file=base),
                )
            self.assertEqual([call["route"] for call in calls], phase2[1:])
            self.assertEqual(len(phase1), 4)

    def test_only_final_winner_is_inserted_into_engine_cache(self) -> None:
        from torch_tensorrt.dynamo.tuning import sweeper

        calls = []

        def trial(**kwargs):
            calls.append(kwargs)
            return _successful_trial(
                2.0 if len(calls) == 1 else 1.0, bytes([len(calls)])
            )

        engine_cache = mock.Mock()
        engine_cache.get_hash.return_value = "winner-hash"
        runtime = mock.Mock()
        engine = mock.Mock()
        runtime.deserialize_cuda_engine.return_value = engine
        patches = self._patch_sweep(trial)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            mock.patch("tensorrt.Runtime", return_value=runtime),
            mock.patch(
                "torch_tensorrt.dynamo.conversion._conversion.insert_engine_to_cache"
            ) as insert,
        ):
            result = sweeper.tune_subgraph(
                self.module,
                self.inputs,
                CompilationSettings(
                    tune_build_routes="-slice_fusion=[on|off]",
                    tuning_search="full",
                    cache_built_engines=True,
                    immutable_weights=False,
                ),
                engine_cache=engine_cache,
            )
        self.assertEqual(result.serialized_engine, b"\x02")
        insert.assert_called_once()
        runtime.deserialize_cuda_engine.assert_called_once_with(b"\x02")
        self.assertIs(insert.call_args.args[1].engine, engine)

    def test_engine_cache_round_trip_uses_weight_stripped_engine(self) -> None:
        from torch_tensorrt.dynamo._engine_cache import BaseEngineCache
        from torch_tensorrt.dynamo.conversion import _conversion

        class MemoryEngineCache(BaseEngineCache):
            def __init__(self):
                self.blobs = {}

            def save(self, hash, blob, *args, **kwargs):
                self.blobs[hash] = blob

            def load(self, hash, *args, **kwargs):
                return self.blobs.get(hash)

        serialization_config = mock.Mock()
        engine = mock.Mock()
        engine.create_serialization_config.return_value = serialization_config
        engine.serialize_with_config.return_value = b"stripped"
        engine_cache = MemoryEngineCache()
        result = _conversion.TRTInterpreterResult(
            engine=engine,
            input_names=["x"],
            output_names=["output"],
            requires_output_allocator=False,
            requires_native_multidevice=False,
            aliased_io={},
        )
        settings = CompilationSettings(
            build_route="-slice_fusion=off", immutable_weights=False
        )
        winner_hash = engine_cache.get_hash(self.module, self.inputs, settings)
        inserted = _conversion.insert_engine_to_cache(
            winner_hash, result, engine_cache, settings, self.inputs
        )
        self.assertTrue(inserted)
        serialization_config.set_flag.assert_called_once()
        cache_entry = engine_cache.check(winner_hash)
        self.assertIsNotNone(cache_entry)
        assert cache_entry is not None
        self.assertEqual(cache_entry[0], b"stripped")
        self.assertEqual(cache_entry[4].build_route, settings.build_route)

    def test_nonfinite_gpu_time_cannot_win(self) -> None:
        from torch_tensorrt.dynamo.tuning import sweeper

        trials = iter(
            [
                _successful_trial(float("nan"), b"bad"),
                _successful_trial(1.0, b"good"),
            ]
        )
        patches = self._patch_sweep(lambda **kwargs: next(trials))
        with patches[0], patches[1], patches[2], patches[3]:
            result = sweeper.tune_subgraph(
                self.module,
                self.inputs,
                CompilationSettings(
                    tune_build_routes="-slice_fusion=[on|off]",
                    tuning_search="full",
                ),
            )
        self.assertEqual(result.serialized_engine, b"good")

    def test_reference_outputs_are_computed_once_and_reused(self) -> None:
        from torch_tensorrt.dynamo.tuning import sweeper

        calls = []
        reference_outputs = {"result": torch.tensor([1.25])}

        def trial(**kwargs):
            calls.append(kwargs)
            return _successful_trial(float(len(calls)))

        patches = self._patch_sweep(trial)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            mock.patch.object(
                sweeper,
                "_compute_reference_outputs",
                return_value=reference_outputs,
            ) as compute_reference,
        ):
            sweeper.tune_subgraph(
                self.module,
                self.inputs,
                CompilationSettings(
                    tune_build_routes="-slice_fusion=[on|off]",
                    tuning_search="full",
                    accuracy_threshold=0.5,
                ),
            )

        compute_reference.assert_called_once()
        self.assertEqual(len(calls), 2)
        self.assertTrue(
            all(call["reference_outputs"] is reference_outputs for call in calls)
        )

    def test_dry_run_never_spawns_a_trial(self) -> None:
        from torch_tensorrt.dynamo.tuning import sweeper

        with (
            mock.patch.object(sweeper, "require_global_perf_tuning"),
            mock.patch.object(
                sweeper, "get_all_build_routes_raw", return_value=self.knobs
            ),
            mock.patch.object(sweeper, "_spawn_route_trial") as spawn,
        ):
            with self.assertRaisesRegex(RuntimeError, "no engines were built"):
                sweeper.tune_subgraph(
                    self.module,
                    self.inputs,
                    CompilationSettings(
                        tune_build_routes="-slice_fusion=[on|off]",
                        tuning_dry_run=True,
                    ),
                )
        spawn.assert_not_called()


class TestGPTAvailabilityAndIntegration(unittest.TestCase):
    def test_capability_probe(self) -> None:
        from torch_tensorrt.dynamo.tuning import is_global_perf_tuning_available

        # Should not raise; result depends on local TensorRT build.
        available = is_global_perf_tuning_available()
        self.assertIsInstance(available, bool)

    def test_small_tune_sweep(self) -> None:
        from torch_tensorrt.dynamo.tuning import (
            get_all_build_routes,
            is_global_perf_tuning_available,
        )

        if not torch.cuda.is_available() or not is_global_perf_tuning_available():
            self.skipTest("Global Performance Tuning or CUDA unavailable")

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
            if name != os.path.basename(cache_base)
            and name.startswith(cache_stem + ".")
            and name.endswith(".jsonl")
        ]
        self.assertTrue(partition_caches, "expected per-partition tuning cache file")
        with open(partition_caches[0], "r", encoding="utf-8") as f:
            lines = [ln for ln in f.readlines() if ln.strip()]
        self.assertGreaterEqual(len(lines), 3)  # header + 2 iters


if __name__ == "__main__":
    unittest.main()
