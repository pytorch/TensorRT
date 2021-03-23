#include "NvInfer.h"
#include "c10/cuda/CUDAStream.h"
#include "core/conversion/conversion.h"
#include "core/runtime/runtime.h"
#include "core/util/prelude.h"
#include "cuda_runtime_api.h"
#include "torch/csrc/jit/ir/ir.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/custom_class.h"

#include <math.h>
#include <vector>

namespace trtorch {
namespace tests {
namespace util {

std::vector<core::conversion::InputRange> toInputRanges(std::vector<at::Tensor> ten) {
  std::vector<core::conversion::InputRange> a;
  for (auto i : ten) {
    a.push_back(core::conversion::InputRange(core::util::toVec(i.sizes())));
  }
  return std::move(a);
}

std::vector<core::conversion::InputRange> toInputRangesDynamic(std::vector<at::Tensor> ten, bool dynamic_batch) {
  std::vector<core::conversion::InputRange> a;

  for (auto i : ten) {
    auto opt = core::util::toVec(i.sizes());

    if (dynamic_batch) {
      std::vector<int64_t> min_range(opt);
      std::vector<int64_t> max_range(opt);

      min_range[0] = ceil(opt[0] / 2.0);
      max_range[0] = 2 * opt[0];

      a.push_back(core::conversion::InputRange(min_range, opt, max_range));
    } else {
      std::vector<int64_t> min_range(opt);
      std::vector<int64_t> max_range(opt);

      min_range[1] = ceil(opt[1] / 2.0);
      max_range[1] = 2 * opt[1];

      a.push_back(core::conversion::InputRange(min_range, opt, max_range));
    }
  }

  return std::move(a);
}

std::vector<at::Tensor> RunEngine(std::string& eng, std::vector<at::Tensor> inputs) {
  LOG_DEBUG("Running TRT version");
  auto engine_ptr = c10::make_intrusive<trtorch::core::runtime::TRTEngine>("test_engine", eng);
  auto outputs = trtorch::core::runtime::execute_engine(inputs, engine_ptr);
  return outputs;
}

std::vector<at::Tensor> RunGraphEngine(
    std::shared_ptr<torch::jit::Graph>& g,
    core::conversion::GraphParams& named_params,
    std::vector<at::Tensor> inputs) {
  LOG_DEBUG("Running TRT version");
  auto in = toInputRanges(inputs);
  auto info = core::conversion::ConversionInfo(in);
  info.engine_settings.workspace_size = 1 << 20;
  std::string eng = core::conversion::ConvertBlockToEngine(g->block(), info, named_params);
  return RunEngine(eng, inputs);
}

std::vector<at::Tensor> RunGraphEngineDynamic(
    std::shared_ptr<torch::jit::Graph>& g,
    core::conversion::GraphParams& named_params,
    std::vector<at::Tensor> inputs,
    bool dynamic_batch) {
  LOG_DEBUG("Running TRT version");
  auto in = toInputRangesDynamic(inputs, dynamic_batch);
  auto info = core::conversion::ConversionInfo(in);
  info.engine_settings.workspace_size = 1 << 20;
  std::string eng = core::conversion::ConvertBlockToEngine(g->block(), info, named_params);
  return RunEngine(eng, inputs);
}

} // namespace util
} // namespace tests
} // namespace trtorch
