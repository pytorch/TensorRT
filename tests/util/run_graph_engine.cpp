#include "NvInfer.h"
#include "c10/cuda/CUDAStream.h"
#include "core/conversion/conversion.h"
#include "core/ir/ir.h"
#include "core/runtime/runtime.h"
#include "core/util/prelude.h"
#include "core/util/trt_util.h"
#include "cuda_runtime_api.h"
#include "torch/csrc/jit/ir/ir.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/custom_class.h"

#include <math.h>
#include <vector>

namespace torch_tensorrt {
namespace tests {
namespace util {

std::vector<core::ir::Input> toInputs(std::vector<at::Tensor> ten) {
  std::vector<core::ir::Input> a;
  for (auto i : ten) {
    a.push_back(core::ir::Input(core::util::toVec(i.sizes()), core::util::ScalarTypeToTRTDataType(i.scalar_type())));
  }
  return a;
}

std::vector<core::ir::Input> toInputsDynamic(std::vector<at::Tensor> ten, bool dynamic_batch) {
  std::vector<core::ir::Input> a;

  for (auto i : ten) {
    auto opt = core::util::toVec(i.sizes());
    auto dtype = core::util::ScalarTypeToTRTDataType(i.scalar_type());

    if (dynamic_batch) {
      std::vector<int64_t> min_range(opt);
      std::vector<int64_t> max_range(opt);

      min_range[0] = ceil(opt[0] / 2.0);
      max_range[0] = 2 * opt[0];

      a.push_back(core::ir::Input(min_range, opt, max_range, dtype));
    } else {
      std::vector<int64_t> min_range(opt);
      std::vector<int64_t> max_range(opt);

      min_range[1] = ceil(opt[1] / 2.0);
      max_range[1] = 2 * opt[1];

      a.push_back(core::ir::Input(min_range, opt, max_range, dtype));
    }
  }

  return a;
}

std::vector<at::Tensor> RunEngine(std::string& eng, std::vector<at::Tensor> inputs) {
  LOG_DEBUG("Running TRT version");
  auto cuda_device = core::runtime::CudaDevice(0, nvinfer1::DeviceType::kGPU);
  auto engine_ptr = c10::make_intrusive<torch_tensorrt::core::runtime::TRTEngine>("test_engine", eng, cuda_device);
  auto outputs = torch_tensorrt::core::runtime::execute_engine(inputs, engine_ptr);
  return outputs;
}

std::vector<const torch::jit::Value*> get_var_inputs(
    c10::ArrayRef<torch::jit::Value*> ins,
    core::ir::StaticParams& static_ins) {
  std::vector<const torch::jit::Value*> var_ins;
  for (auto in : ins) {
    if (static_ins.find(in) == static_ins.end()) {
      var_ins.push_back(in);
    }
  }
  return var_ins;
}

std::vector<at::Tensor> RunGraphEngine(
    std::shared_ptr<torch::jit::Graph>& g,
    core::ir::StaticParams& named_params,
    std::vector<at::Tensor> inputs,
    nvinfer1::DataType op_precision = nvinfer1::DataType::kFLOAT) {
  LOG_DEBUG("Running TRT version");
  auto var_ins = get_var_inputs(g->inputs(), named_params);
  auto in = core::ir::pair_input_vals_with_specs(var_ins, toInputs(inputs));
  auto info = core::conversion::ConversionInfo();
  info.inputs = std::move(in);
  info.engine_settings.enabled_precisions.insert(op_precision);
  std::string eng = core::conversion::ConvertBlockToEngine(g->block(), info, named_params);
  return RunEngine(eng, inputs);
}

std::vector<at::Tensor> RunGraphEngineDynamic(
    std::shared_ptr<torch::jit::Graph>& g,
    core::ir::StaticParams& named_params,
    std::vector<at::Tensor> inputs,
    bool dynamic_batch) {
  LOG_DEBUG("Running TRT version");
  auto var_ins = get_var_inputs(g->inputs(), named_params);
  auto in = core::ir::pair_input_vals_with_specs(var_ins, toInputsDynamic(inputs, dynamic_batch));
  auto info = core::conversion::ConversionInfo();
  info.inputs = std::move(in);
  std::string eng = core::conversion::ConvertBlockToEngine(g->block(), info, named_params);
  return RunEngine(eng, inputs);
}

} // namespace util
} // namespace tests
} // namespace torch_tensorrt
