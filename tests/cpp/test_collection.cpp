#include <string>
#include <thread>
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/script.h"
#include "torch_tensorrt/torch_tensorrt.h"


TEST(CppAPITests, TestCollection) {


  std::string path =
  // "/opt/trtorch/tuple2model.ts";
  // "/opt/trtorch/tuple2_list2_v3.ts";
  // "/opt/trtorch/tuple2_tuple2_v3.ts";
  "/opt/trtorch/tuple2_v3.ts";
  // "/opt/trtorch/list2_list2_v3.ts";
  torch::Tensor in0 = torch::randn({1, 3, 512, 512}, torch::kCUDA).to(torch::kFloat);
  std::vector<at::Tensor> inputs;
  inputs.push_back(in0);

  torch::jit::Module mod;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    mod = torch::jit::load(path);
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
  }
  mod.eval();
  mod.to(torch::kCUDA);
  

  std::vector<torch::jit::IValue> inputs_;

  for (auto in : inputs) {
    inputs_.push_back(torch::jit::IValue(in.clone()));
  }


  std::vector<torch::jit::IValue> complex_inputs, complex_inputs_list;
  std::vector<torch::jit::IValue> tuple;
  std::tuple<torch::jit::IValue, torch::jit::IValue> input_tuple(in0, in0);
  // auto input_list = c10::impl::GenericList(c10::TensorType::get());
  // input_list.push_back(inputs_[0]);
  // input_list.push_back(inputs_[0]);

  // torch::jit::IValue input_list_ivalue = torch::jit::IValue(input_list);

  complex_inputs.push_back(input_tuple);
  complex_inputs_list.push_back(in0);
  complex_inputs_list.push_back(in0);



  auto out = mod.forward(complex_inputs);
  LOG_DEBUG("Finish torchscirpt forward");


  auto input_shape = torch_tensorrt::Input(in0.sizes(), torch_tensorrt::DataType::kUnknown);

  auto input_shape_ivalue = torch::jit::IValue(std::move(c10::make_intrusive<torch_tensorrt::Input>(input_shape)));


  c10::TypePtr elementType = input_shape_ivalue.type();
  auto list = c10::impl::GenericList(elementType);
  list.push_back(input_shape_ivalue);
  list.push_back(input_shape_ivalue);

  std::tuple<torch::jit::IValue, torch::jit::IValue> input_shape_tuple(input_shape_ivalue, input_shape_ivalue);


  torch::jit::IValue complex_input_shape(input_shape_tuple);
  std::tuple<torch::jit::IValue> input_tuple2(complex_input_shape);
  torch::jit::IValue complex_input_shape2(input_tuple2);
  // torch::jit::IValue complex_input_shape(list);

  auto compile_settings = torch_tensorrt::ts::CompileSpec(complex_input_shape2);
  compile_settings.require_full_compilation = false;
  compile_settings.min_block_size = 1;

  // compile_settings.torch_executed_modules.push_back("model1");
  // compile_settings.torch_executed_ops.push_back("aten::sub");


  // // FP16 execution
  // compile_settings.enabled_precisions = {torch::kHalf};
  // // Compile module
  auto trt_mod = torch_tensorrt::torchscript::compile(mod, compile_settings);
  LOG_DEBUG("Finish compile");
  auto trt_out = trt_mod.forward(complex_inputs);
  // auto trt_out = trt_mod.forward(complex_inputs_list);


  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(out.toTensor(), trt_out.toTensor(), 1e-5));
}