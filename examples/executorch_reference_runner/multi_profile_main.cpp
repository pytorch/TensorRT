/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Selecting a TensorRT optimization profile per call, through the high-level
 * ExecuTorch Module API.
 *
 * Pairs with examples/torchtrt_executorch_example/export_multi_profile.py,
 * which writes a two-profile Gemma-3 engine taking two [1, seq] index tensors
 * (input_ids and position_ids) and returning the last position's logits:
 *
 *   profile 0 -> decode,  seq == 1
 *   profile 1 -> prefill, seq in [1, 256], tuned at 128
 *
 * Ends with per-call latency for decode on each profile, the same comparison
 * examples/dynamo/multi_optimization_profiles.py makes through the Python
 * runtime. For latency distributions see multi_profile_benchmark.cpp.
 *
 * Usage:
 *   example_executorch_multi_profile_runner --model_path=model.pte
 */

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>

#include <torch_tensorrt/executorch/TensorRTBackend.h>

using executorch::extension::Module;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using torch_tensorrt::executorch_backend::OptimizationProfileGuard;

namespace {

constexpr int32_t kDecodeProfile = 0;
constexpr int32_t kPrefillProfile = 1;
constexpr int32_t kPrefillSeq = 128;
constexpr int32_t kMaxSeq = 256;

// Timing loop at the end, matching examples/dynamo/multi_optimization_profiles.py.
constexpr int kWarmup = 20;
constexpr int kIters = 50;
constexpr int kRounds = 3;

using Clock = std::chrono::steady_clock;

const char* get_flag(int argc, char** argv, const char* flag, const char* def) {
  const size_t n = strlen(flag);
  for (int i = 1; i < argc; ++i) {
    if (strncmp(argv[i], flag, n) == 0 && argv[i][n] == '=') {
      return argv[i] + n + 1;
    }
  }
  return def;
}

// One [1, seq] index tensor. The dtype comes from the .pte's method signature
// rather than being assumed: the backend binds tensor pointers straight to
// TensorRT without converting, so a mismatch here is silent corruption.
class IndexTensor {
 public:
  IndexTensor(int32_t seq, exec_aten::ScalarType dtype, bool positions)
      : sizes_{1, seq},
        dim_order_{0, 1},
        strides_{seq, 1},
        data_(static_cast<size_t>(seq) * (dtype == exec_aten::ScalarType::Long ? 8 : 4)),
        impl_(dtype, 2, sizes_.data(), data_.data(), dim_order_.data(), strides_.data()) {
    for (int32_t i = 0; i < seq; ++i) {
      const int64_t v = positions ? i : (1 + (static_cast<int64_t>(i) * 7919) % 9000);
      if (dtype == exec_aten::ScalarType::Long) {
        reinterpret_cast<int64_t*>(data_.data())[i] = v;
      } else {
        reinterpret_cast<int32_t*>(data_.data())[i] = static_cast<int32_t>(v);
      }
    }
  }

  EValue evalue() {
    return EValue(exec_aten::Tensor(&impl_));
  }

 private:
  std::vector<exec_aten::SizesType> sizes_;
  std::vector<exec_aten::DimOrderType> dim_order_;
  std::vector<exec_aten::StridesType> strides_;
  std::vector<uint8_t> data_;
  exec_aten::TensorImpl impl_;
};

// Owns the (input_ids, position_ids) pair for one sequence length.
class Step {
 public:
  Step(int32_t seq, exec_aten::ScalarType dtype)
      : ids_(seq, dtype, false), positions_(seq, dtype, true), args_{ids_.evalue(), positions_.evalue()} {}

  const std::vector<EValue>& args() const {
    return args_;
  }

 private:
  IndexTensor ids_;
  IndexTensor positions_;
  std::vector<EValue> args_;
};

// The method returns the last position's logits, so the argmax is the token the
// model would emit next. Printing it makes it obvious when a profile switch
// changes shapes but not results.
void print_prediction(const char* label, const std::vector<EValue>& outputs) {
  if (outputs.empty() || !outputs[0].isTensor()) {
    return;
  }
  exec_aten::Tensor t = outputs[0].toTensor();
  // const_data_ptr<T>() is an unchecked cast, so reading a dtype we did not plan
  // for walks the buffer at the wrong stride and runs off the end. Name the ones
  // handled and skip anything else rather than printing corrupt numbers.
  const exec_aten::ScalarType dtype = t.scalar_type();
  if (dtype != exec_aten::ScalarType::Float && dtype != exec_aten::ScalarType::Half &&
      dtype != exec_aten::ScalarType::BFloat16) {
    ET_LOG(Info, "%s: logits dtype %d not handled by this example; skipping", label, static_cast<int>(dtype));
    return;
  }
  double best = -1e30;
  int64_t best_idx = -1;
  for (int64_t i = 0; i < t.numel(); ++i) {
    double v = 0.0;
    if (dtype == exec_aten::ScalarType::Half) {
      v = static_cast<double>(t.const_data_ptr<exec_aten::Half>()[i]);
    } else if (dtype == exec_aten::ScalarType::BFloat16) {
      v = static_cast<double>(t.const_data_ptr<exec_aten::BFloat16>()[i]);
    } else {
      v = static_cast<double>(t.const_data_ptr<float>()[i]);
    }
    if (v > best) {
      best = v;
      best_idx = i;
    }
  }
  fprintf(stderr, "%-28s logits=[", label);
  for (ssize_t d = 0; d < t.dim(); ++d) {
    fprintf(stderr, "%d%s", static_cast<int>(t.size(d)), d + 1 < t.dim() ? "," : "");
  }
  fprintf(stderr, "] next_token=%" PRId64 "\n", best_idx);
}

// Runs one forward with whatever profile guard the caller has in scope.
bool run_guarded(Module& module, const char* label, const Step& step) {
  auto result = module.forward(step.args());

  if (!result.ok()) {
    ET_LOG(Error, "%s: forward() failed: 0x%" PRIx32, label, static_cast<uint32_t>(result.error()));
    return false;
  }
  print_prediction(label, result.get());
  return true;
}

// Runs one forward with `profile` pinned. The guard applies to every TensorRT
// delegate this thread executes while it is in scope. It stores the request
// only; each delegate switches inside its own execute(), on the stream that
// execute() already selected.
bool run(Module& module, const char* label, int32_t profile, const Step& step) {
  OptimizationProfileGuard profile_guard(profile);
  return run_guarded(module, label, step);
}

// Same, but lets each delegate choose from the input shapes.
bool run_auto(Module& module, const char* label, const Step& step) {
  auto profile_guard = OptimizationProfileGuard::automatic();
  return run_guarded(module, label, step);
}

// Mean milliseconds per forward, best of kRounds. The profile is pinned around
// the whole loop rather than per call, which is the realistic serving pattern
// and keeps profile switches out of the measurement. Inputs are host tensors,
// so each forward() stages H2D and synchronizes before returning and the
// interval covers the whole round trip. Negative on failure.
double benchmark(Module& module, const Step& step, int32_t profile) {
  OptimizationProfileGuard profile_guard(profile);

  for (int i = 0; i < kWarmup; ++i) {
    if (!module.forward(step.args()).ok()) {
      return -1.0;
    }
  }
  double best = std::numeric_limits<double>::infinity();
  for (int round = 0; round < kRounds; ++round) {
    const auto start = Clock::now();
    for (int i = 0; i < kIters; ++i) {
      if (!module.forward(step.args()).ok()) {
        return -1.0;
      }
    }
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    best = std::min(best, ms / kIters);
  }
  return best;
}

// Decode is timed twice against the one engine: once on the prefill profile,
// which accepts seq == 1 and so runs it on kernels tuned for a kPrefillSeq
// prompt (what a single-profile engine gives you), and once on its own profile.
// Prefill appears once because the decode profile does not accept a kPrefillSeq
// input at all, so prefill has only one profile it can run on.
bool report_latency(Module& module, const Step& prefill, const Step& decode) {
  const double decode_on_prefill = benchmark(module, decode, kPrefillProfile);
  const double decode_on_decode = benchmark(module, decode, kDecodeProfile);
  const double prefill_on_prefill = benchmark(module, prefill, kPrefillProfile);
  if (decode_on_prefill < 0.0 || decode_on_decode < 0.0 || prefill_on_prefill < 0.0) {
    ET_LOG(Error, "latency benchmark: forward() failed");
    return false;
  }

  fprintf(stderr, "\nPer-call latency (ms), batch=1\n");
  fprintf(stderr, "%-24s%18s%10s\n", "call", "active profile", "ms");
  fprintf(stderr, "----------------------------------------------------\n");
  fprintf(stderr, "%-24s%18s%10.3f\n", "decode (seq=1)", "prefill", decode_on_prefill);
  fprintf(stderr, "%-24s%18s%10.3f\n", "decode (seq=1)", "decode", decode_on_decode);
  char prefill_label[32];
  snprintf(prefill_label, sizeof(prefill_label), "prefill (seq=%d)", kPrefillSeq);
  fprintf(stderr, "%-24s%18s%10.3f\n", prefill_label, "prefill", prefill_on_prefill);
  fprintf(
      stderr,
      "\nGiving decode its own profile: %.2fx faster per token (%+.3f ms)\n",
      decode_on_prefill / decode_on_decode,
      decode_on_prefill - decode_on_decode);
  return true;
}

} // namespace

int main(int argc, char** argv) {
  executorch::runtime::runtime_init();

  const char* model_path = get_flag(argc, argv, "--model_path", "model_gemma3_multi_profile.pte");
  Module module(model_path);

  const auto meta = module.method_meta("forward");
  if (!meta.ok()) {
    ET_LOG(Error, "could not read method_meta: 0x%" PRIx32, static_cast<uint32_t>(meta.error()));
    return 1;
  }
  const auto input0 = meta->input_tensor_meta(0);
  if (!input0.ok()) {
    ET_LOG(Error, "could not read input 0 metadata");
    return 1;
  }
  const exec_aten::ScalarType dtype = input0->scalar_type();

  Step prefill(kPrefillSeq, dtype);
  Step long_prefill(kMaxSeq, dtype);
  Step decode(1, dtype);

  // Long prompt: pin the prefill profile, whose kernels TensorRT tuned at a
  // 128-token sequence.
  bool ok = run(module, "pinned prefill (seq=128)", kPrefillProfile, prefill);

  // One token at a time: pin the decode profile, whose seq is pinned to 1 so
  // TensorRT could specialize it instead of serving it from prefill kernels.
  for (int token = 0; ok && token < 3; ++token) {
    ok = run(module, "pinned decode (seq=1)", kDecodeProfile, decode);
  }

  // Back to prefill at the profile's upper bound, to show the switch is per
  // call and not one-way.
  ok = ok && run(module, "pinned prefill (seq=256)", kPrefillProfile, long_prefill);

  // Auto-selection reads the input shapes instead. It is sticky: once the
  // prefill profile is loaded a seq == 1 input still fits it, so this stays on
  // profile 1 rather than dropping back to decode. Pin when that matters.
  ok = ok && run_auto(module, "auto (seq=1)", decode);

  // With no guard in scope every delegate runs profile 0, which here accepts
  // seq == 1 only.
  if (ok) {
    auto result = module.forward(decode.args());
    if (!result.ok()) {
      ET_LOG(Error, "no guard: forward() failed: 0x%" PRIx32, static_cast<uint32_t>(result.error()));
      ok = false;
    } else {
      print_prediction("no guard (seq=1)", result.get());
    }
  }

  // A pinned index the engine does not have is an input error, reported before
  // anything is enqueued.
  if (ok) {
    OptimizationProfileGuard profile_guard(99);
    auto result = module.forward(decode.args());
    if (result.ok()) {
      ET_LOG(Error, "expected profile 99 to be rejected");
      ok = false;
    } else {
      fprintf(stderr, "%-28s rejected as expected\n", "pinned profile 99");
    }
  }

  // Correctness is settled by here; what remains is what the choice is worth.
  ok = ok && report_latency(module, prefill, decode);

  if (!ok) {
    return 1;
  }
  ET_LOG(Info, "Multi-profile run completed.");
  return 0;
}
