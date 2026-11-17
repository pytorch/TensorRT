// SPDX-License-Identifier: BSD-3-Clause
#include <executorch/extension/cuda/caller_stream.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/platform/runtime.h>

#include "benchmark.h"
using namespace pbr;
using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::runtime::Error;
using executorch::runtime::EValue;

int main(int argc, char** argv) {
  try {
    require(argc == 5,
            "Usage: benchmark_et MODEL_DIR FIXTURE OUTPUT MODEL_NAME");
    fs::path model = argv[1];
    std::string family = argv[4];
    require(family == "segformer" || family == "bevformer", "Unknown model");
    Report report(argv[3]);
    executorch::runtime::runtime_init();
    cuda_check(cudaSetDevice(0), "select GPU");
    Stream stream;
    cuda_check(cudaDeviceSynchronize(), "initialize CUDA");
    executorch::extension::cuda::CallerStreamGuard guard(stream.value);
    report.snapshot("cuda_initialized");
    Module module(
        (model / "model.pte").string(),
        family == "bevformer" ? (model / "aoti_cuda_blob.ptd").string() : "");
    require(module.load() == Error::Ok, "Program load failed");
    require(module.load_method("forward") == Error::Ok, "Method load failed");
    cuda_check(cudaDeviceSynchronize(), "method load");
    report.snapshot("method_loaded");
    auto meta = module.method_meta("forward");
    require(meta.ok(), "Missing method metadata");
    require(meta->num_inputs() == (family == "bevformer" ? 5 : 1) &&
                meta->num_outputs() == (family == "bevformer" ? 3 : 1),
            "Model IO count mismatch");
    Fixtures fixtures(argv[2], meta->num_inputs(), meta->num_outputs());
    std::vector<TensorPtr> tensors, outputs;
    std::vector<EValue> inputs;
    for (size_t i = 0; i < fixtures.inputs.size() + fixtures.outputs.size();
         ++i) {
      bool input = i < fixtures.inputs.size();
      size_t j = input ? i : i - fixtures.inputs.size();
      const auto desc =
          input ? meta->input_tensor_meta(j) : meta->output_tensor_meta(j);
      const auto& fixture = input ? fixtures.inputs[j] : fixtures.expected[j];
      require(desc.ok() && !desc->is_memory_planned() &&
                  desc->scalar_type() == executorch::aten::ScalarType::Float,
              "Requires caller-owned FP32 IO");
      require(desc->nbytes() == fixture.bytes() &&
                  desc->sizes().size() == fixture.shape.size(),
              "IO size mismatch");
      for (size_t d = 0; d < fixture.shape.size(); ++d)
        require(
            desc->sizes()[d] == fixture.shape[d] && desc->dim_order()[d] == d,
            "Requires matching contiguous IO");
      std::vector<executorch::aten::SizesType> shape(desc->sizes().begin(),
                                                     desc->sizes().end());
      auto tensor = executorch::extension::from_blob(
          input ? fixtures.inputs[j].data.data() : fixtures.outputs[j].data(),
          shape, desc->scalar_type());
      if (input) {
        tensors.push_back(tensor);
        inputs.emplace_back(*tensor);
      } else {
        outputs.push_back(tensor);
        require(module.set_output("forward", EValue(*tensor), j) == Error::Ok,
                "Output binding failed");
      }
    }
    PinnedIO pinned(fixtures);
    report.snapshot("fixtures_loaded");
    report.measure(fixtures, [&]() {
      auto result = module.forward(inputs);
      cuda_check(cudaDeviceSynchronize(), "complete inference");
      require(result.ok() && result->size() == outputs.size(),
              "Forward failed");
      for (size_t i = 0; i < outputs.size(); ++i) {
        require((*result)[i].isTensor(), "Expected tensor output");
        const auto t = (*result)[i].toTensor();
        require(t.const_data_ptr<float>() == fixtures.outputs[i].data() &&
                    t.nbytes() == fixtures.expected[i].bytes(),
                "Caller output binding changed");
      }
    });
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
