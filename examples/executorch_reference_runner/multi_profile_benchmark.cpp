/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * What per-call optimization profile selection is worth.
 *
 * Runs the same prefill/decode serving loop two ways against the one
 * two-profile engine from
 * examples/torchtrt_executorch_example/export_multi_profile.py:
 *
 *   prefill-only - every call pinned to the prefill profile. The prefill
 *                  profile accepts seq == 1, so decode runs on kernels
 *                  TensorRT tuned for a 128-token prompt.
 *   switching    - prefill pinned to the prefill profile and each decode step
 *                  pinned to the decode profile, whose seq is pinned to 1.
 *
 * One engine, one set of weights, one export; the only difference is which
 * profile is loaded when the call runs. Decode is where the difference should
 * show, since that is the phase whose kernels the prefill profile mistunes.
 *
 * Measurement notes:
 *   - The two configurations are interleaved in short blocks so that drift and
 *     any other tenant on the GPU hit both roughly equally.
 *   - Interference can only add time, so the low percentiles are the signal.
 *     min and p10 are what to read; the tail says how busy the machine was.
 *   - The first call of each block is discarded: it inherits whichever profile
 *     the previous block left loaded, so it can carry a switch the block is
 *     not meant to be measuring.
 *
 * Usage:
 *   example_executorch_multi_profile_benchmark --model_path=model.pte
 */

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>

#include <torch_tensorrt/executorch/TensorRTBackend.h>

using executorch::extension::Module;
using executorch::runtime::EValue;
using torch_tensorrt::executorch_backend::OptimizationProfileGuard;

namespace {

constexpr int32_t kDecodeProfile = 0;
constexpr int32_t kPrefillProfile = 1;

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

// atoi() returns 0 for garbage, so an unusable value would otherwise pass for a
// deliberate 0 and surface much later as an empty sample set or a nan in the
// summary. Reject anything below `min` here, where the flag name is still known.
bool get_int_flag(int argc, char** argv, const char* flag, int def, int min, int& out) {
  const char* raw = get_flag(argc, argv, flag, nullptr);
  out = raw == nullptr ? def : atoi(raw);
  if (out < min) {
    ET_LOG(Error, "%s must be at least %d, got '%s'", flag, min, raw == nullptr ? "" : raw);
    return false;
  }
  return true;
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

// The (input_ids, position_ids) pair for one sequence length, held so the
// EValue vector handed to forward() can be reused without reallocating.
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

struct Stats {
  size_t n = 0;
  double min = 0.0;
  double p10 = 0.0;
  double p25 = 0.0;
  double median = 0.0;
  double p90 = 0.0;
};

double percentile(const std::vector<double>& sorted, double q) {
  return sorted[static_cast<size_t>(q * static_cast<double>(sorted.size() - 1))];
}

Stats summarize(std::vector<double> samples) {
  Stats s;
  if (samples.empty()) {
    return s;
  }
  std::sort(samples.begin(), samples.end());
  s.n = samples.size();
  s.min = samples.front();
  s.p10 = percentile(samples, 0.10);
  s.p25 = percentile(samples, 0.25);
  s.median = percentile(samples, 0.50);
  s.p90 = percentile(samples, 0.90);
  return s;
}

void print_stats(const char* label, const Stats& s) {
  printf(
      "  %-30s n=%-5zu min=%8.3f  p10=%8.3f  p25=%8.3f  median=%8.3f  p90=%8.3f\n",
      label,
      s.n,
      s.min,
      s.p10,
      s.p25,
      s.median,
      s.p90);
}

// One forward under `profile`, timed end to end. Inputs are host tensors, so
// execute() stages H2D and synchronizes before returning; the interval covers
// the whole round trip. Returns milliseconds, or a negative value on failure.
double timed_forward(Module& module, const std::vector<EValue>& args, int32_t profile) {
  const auto t0 = Clock::now();
  double elapsed_ms = 0.0;
  {
    OptimizationProfileGuard profile_guard(profile);
    auto result = module.forward(args);
    elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    if (!result.ok()) {
      ET_LOG(Error, "forward() failed: 0x%" PRIx32, static_cast<uint32_t>(result.error()));
      return -1.0;
    }
  }
  return elapsed_ms;
}

struct WorkloadResult {
  std::vector<double> prefill_ms;
  std::vector<double> decode_ms;
  double wall_ms = 0.0;
};

// `rounds` iterations of one prefill followed by `decode_steps` decode steps.
bool run_block(
    Module& module,
    const Step& prefill,
    const Step& decode,
    int32_t prefill_profile,
    int32_t decode_profile,
    int rounds,
    int decode_steps,
    WorkloadResult& out) {
  const auto start = Clock::now();
  for (int r = 0; r < rounds; ++r) {
    const double p = timed_forward(module, prefill.args(), prefill_profile);
    if (p < 0.0) {
      return false;
    }
    if (r != 0) { // first call of a block inherits the previous block's profile
      out.prefill_ms.push_back(p);
    }
    for (int d = 0; d < decode_steps; ++d) {
      const double t = timed_forward(module, decode.args(), decode_profile);
      if (t < 0.0) {
        return false;
      }
      if (r != 0 || d != 0) {
        out.decode_ms.push_back(t);
      }
    }
  }
  out.wall_ms += std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  return true;
}

void compare(const char* what, const Stats& prefill_only, const Stats& switching) {
  const double d_min = prefill_only.min - switching.min;
  const double d_p10 = prefill_only.p10 - switching.p10;
  printf(
      "  %-30s %+8.3f ms (min)  %+8.3f ms (p10)   %.2fx (min)\n",
      what,
      d_min,
      d_p10,
      switching.min > 0.0 ? prefill_only.min / switching.min : 0.0);
}

} // namespace

int main(int argc, char** argv) {
  executorch::runtime::runtime_init();

  const char* model_path = get_flag(argc, argv, "--model_path", "model_gemma3_multi_profile.pte");
  int prefill_seq = 0;
  int blocks = 0;
  int block_rounds = 0;
  int decode_steps = 0;
  int warmup = 0;
  if (!get_int_flag(argc, argv, "--prefill_seq", 128, 1, prefill_seq) ||
      !get_int_flag(argc, argv, "--blocks", 10, 1, blocks) ||
      // run_block() discards the first round of each block as warm-in, so a
      // single round would leave prefill with no samples at all.
      !get_int_flag(argc, argv, "--block_rounds", 3, 2, block_rounds) ||
      !get_int_flag(argc, argv, "--decode_steps", 16, 1, decode_steps) ||
      !get_int_flag(argc, argv, "--warmup", 20, 0, warmup)) {
    return 1;
  }

  Module module(model_path);

  // Take the input dtype from the method itself rather than assuming it:
  // Torch-TensorRT may narrow int64 indices to int32 during lowering.
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

  Step prefill(prefill_seq, dtype);
  Step decode(1, dtype);

  printf("model      : %s\n", model_path);
  printf("inputs     : 2 x [1, seq] %s\n", dtype == exec_aten::ScalarType::Long ? "int64" : "int32");
  printf(
      "workload   : %d interleaved blocks x %d rounds x (1 prefill seq=%d + %d decode seq=1) per config\n",
      blocks,
      block_rounds,
      prefill_seq,
      decode_steps);
  printf("units      : milliseconds per module.forward(); read min/p10, the tail is machine noise\n\n");

  for (int i = 0; i < warmup; ++i) {
    if (timed_forward(module, prefill.args(), kPrefillProfile) < 0.0 ||
        timed_forward(module, decode.args(), kDecodeProfile) < 0.0) {
      return 1;
    }
  }

  WorkloadResult prefill_only;
  WorkloadResult switching;
  for (int b = 0; b < blocks; ++b) {
    if (!run_block(
            module, prefill, decode, kPrefillProfile, kPrefillProfile, block_rounds, decode_steps, prefill_only) ||
        !run_block(module, prefill, decode, kPrefillProfile, kDecodeProfile, block_rounds, decode_steps, switching)) {
      return 1;
    }
  }

  const Stats po_prefill = summarize(prefill_only.prefill_ms);
  const Stats po_decode = summarize(prefill_only.decode_ms);
  const Stats sw_prefill = summarize(switching.prefill_ms);
  const Stats sw_decode = summarize(switching.decode_ms);

  printf("prefill-only (every call on the prefill profile)\n");
  print_stats("prefill (seq=128)", po_prefill);
  print_stats("decode  (seq=1)", po_decode);
  printf("\nswitching (each phase on its own profile)\n");
  print_stats("prefill (seq=128)", sw_prefill);
  print_stats("decode  (seq=1)", sw_decode);

  printf("\nwhat switching bought (positive = switching is faster)\n");
  compare("decode", po_decode, sw_decode);
  compare("prefill", po_prefill, sw_prefill);
  printf(
      "  %-30s prefill-only=%9.1f ms   switching=%9.1f ms   %+.1f%%\n",
      "wall time (see note)",
      prefill_only.wall_ms,
      switching.wall_ms,
      100.0 * (switching.wall_ms - prefill_only.wall_ms) / prefill_only.wall_ms);
  printf(
      "\nnote: wall time is contention-prone, and interleaving charges each prefill-only block\n"
      "      one switch back that a single-profile engine would never pay, so it reads a little\n"
      "      kinder to switching than reality. multi_profile_main.cpp times a clean request.\n");

  return 0;
}
