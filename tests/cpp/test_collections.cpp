#include <string>
#include <thread>
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/script.h"
#include "torch_tensorrt/torch_tensorrt.h"

TEST(CppAPITests, TestCollectionStandardTensorInput) {
  std::string path = "tests/modules/standard_tensor_input_scripted.jit.pt";
  torch::Tensor in0 = torch::randn({1, 3, 512, 512}, torch::kCUDA).to(torch::kHalf);
  std::vector<at::Tensor> inputs;
  inputs.push_back(in0);
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

  auto out = mod.forward(inputs_);

  std::vector<torch_tensorrt::Input> input_range;
  input_range.push_back({in0.sizes(), torch::kF16});
  input_range.push_back({in0.sizes(), torch::kF16});
  torch_tensorrt::ts::CompileSpec compile_settings(input_range);
  compile_settings.min_block_size = 1;

  // // FP16 execution
  compile_settings.enabled_precisions = {torch::kHalf};
  // // Compile module
  auto trt_mod = torch_tensorrt::torchscript::compile(mod, compile_settings);
  auto trt_out = trt_mod.forward(inputs_);

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(out.toTensor(), trt_out.toTensor(), 1e-5));
}

TEST(CppAPITests, TestCollectionTupleInput) {
  std::string path = "tests/modules/tuple_input_scripted.jit.pt";
  torch::Tensor in0 = torch::randn({1, 3, 512, 512}, torch::kCUDA).to(torch::kHalf);

  torch::jit::Module mod;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    mod = torch::jit::load(path);
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
  }
  mod.eval();
  mod.to(torch::kCUDA);

  std::vector<torch::jit::IValue> complex_inputs, complex_inputs_list;
  std::tuple<torch::jit::IValue, torch::jit::IValue> input_tuple(in0, in0);

  complex_inputs.push_back(input_tuple);

  auto out = mod.forward(complex_inputs);

  auto input_shape = torch_tensorrt::Input(in0.sizes(), torch_tensorrt::DataType::kHalf);

  auto input_shape_ivalue = torch::jit::IValue(std::move(c10::make_intrusive<torch_tensorrt::Input>(input_shape)));

  std::tuple<torch::jit::IValue, torch::jit::IValue> input_shape_tuple(input_shape_ivalue, input_shape_ivalue);

  torch::jit::IValue complex_input_shape(input_shape_tuple);
  std::tuple<torch::jit::IValue> input_tuple2(complex_input_shape);
  torch::jit::IValue complex_input_shape2(input_tuple2);

  auto compile_settings = torch_tensorrt::ts::CompileSpec(complex_input_shape2);
  compile_settings.min_block_size = 1;

  // // FP16 execution
  compile_settings.enabled_precisions = {torch::kHalf};
  // // Compile module
  auto trt_mod = torch_tensorrt::torchscript::compile(mod, compile_settings);
  auto trt_out = trt_mod.forward(complex_inputs);

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(out.toTensor(), trt_out.toTensor(), 1e-5));
}

TEST(CppAPITests, TestCollectionListInput) {
  std::string path = "tests/modules/list_input_scripted.jit.pt";
  torch::Tensor in0 = torch::randn({1, 3, 512, 512}, torch::kCUDA).to(torch::kHalf);
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

  std::vector<torch::jit::IValue> complex_inputs;
  auto input_list = c10::impl::GenericList(c10::TensorType::get());
  input_list.push_back(inputs_[0]);
  input_list.push_back(inputs_[0]);

  torch::jit::IValue input_list_ivalue = torch::jit::IValue(input_list);

  complex_inputs.push_back(input_list_ivalue);

  auto out = mod.forward(complex_inputs);

  auto input_shape = torch_tensorrt::Input(in0.sizes(), torch_tensorrt::DataType::kHalf);
  auto input_shape_ivalue = torch::jit::IValue(std::move(c10::make_intrusive<torch_tensorrt::Input>(input_shape)));

  c10::TypePtr elementType = input_shape_ivalue.type();
  auto list = c10::impl::GenericList(elementType);
  list.push_back(input_shape_ivalue);
  list.push_back(input_shape_ivalue);

  torch::jit::IValue complex_input_shape(list);
  std::tuple<torch::jit::IValue> input_tuple2(complex_input_shape);
  torch::jit::IValue complex_input_shape2(input_tuple2);

  auto compile_settings = torch_tensorrt::ts::CompileSpec(complex_input_shape2);
  compile_settings.min_block_size = 1;
  // compile_settings.torch_executed_ops.push_back("aten::__getitem__");

  // // FP16 execution
  compile_settings.enabled_precisions = {torch::kHalf};
  // // Compile module
  auto trt_mod = torch_tensorrt::torchscript::compile(mod, compile_settings);
  LOG_DEBUG("Finish compile");
  auto trt_out = trt_mod.forward(complex_inputs);

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(out.toTensor(), trt_out.toTensor(), 1e-5));
}

TEST(CppAPITests, TestCollectionTupleInputOutput) {
  std::string path = "tests/modules/tuple_input_output_scripted.jit.pt";

  torch::Tensor in0 = torch::randn({1, 3, 512, 512}, torch::kCUDA).to(torch::kHalf);

  torch::jit::Module mod;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    mod = torch::jit::load(path);
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
  }
  mod.eval();
  mod.to(torch::kCUDA);

  std::vector<torch::jit::IValue> complex_inputs, complex_inputs_list;
  std::tuple<torch::jit::IValue, torch::jit::IValue> input_tuple(in0, in0);

  complex_inputs.push_back(input_tuple);

  auto out = mod.forward(complex_inputs);

  auto input_shape = torch_tensorrt::Input(in0.sizes(), torch_tensorrt::DataType::kHalf);

  auto input_shape_ivalue = torch::jit::IValue(std::move(c10::make_intrusive<torch_tensorrt::Input>(input_shape)));

  std::tuple<torch::jit::IValue, torch::jit::IValue> input_shape_tuple(input_shape_ivalue, input_shape_ivalue);

  torch::jit::IValue complex_input_shape(input_shape_tuple);
  std::tuple<torch::jit::IValue> input_tuple2(complex_input_shape);
  torch::jit::IValue complex_input_shape2(input_tuple2);
  // torch::jit::IValue complex_input_shape(list);

  auto compile_settings = torch_tensorrt::ts::CompileSpec(complex_input_shape2);
  compile_settings.min_block_size = 1;

  // compile_settings.torch_executed_ops.push_back("prim::TupleConstruct");

  // // FP16 execution
  compile_settings.enabled_precisions = {torch::kHalf};
  // // Compile module
  auto trt_mod = torch_tensorrt::torchscript::compile(mod, compile_settings);
  auto trt_out = trt_mod.forward(complex_inputs);

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(
      out.toTuple()->elements()[0].toTensor(), trt_out.toTuple()->elements()[0].toTensor(), 1e-5));
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(
      out.toTuple()->elements()[1].toTensor(), trt_out.toTuple()->elements()[1].toTensor(), 1e-5));
}

TEST(CppAPITests, TestCollectionListInputOutput) {
  std::string path = "tests/modules/list_input_output_scripted.jit.pt";
  torch::Tensor in0 = torch::randn({1, 3, 512, 512}, torch::kCUDA).to(torch::kHalf);
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

  std::vector<torch::jit::IValue> complex_inputs;
  auto input_list = c10::impl::GenericList(c10::TensorType::get());
  input_list.push_back(inputs_[0]);
  input_list.push_back(inputs_[0]);

  torch::jit::IValue input_list_ivalue = torch::jit::IValue(input_list);

  complex_inputs.push_back(input_list_ivalue);

  auto out = mod.forward(complex_inputs);

  auto input_shape = torch_tensorrt::Input(in0.sizes(), torch_tensorrt::DataType::kHalf);

  auto input_shape_ivalue = torch::jit::IValue(std::move(c10::make_intrusive<torch_tensorrt::Input>(input_shape)));

  c10::TypePtr elementType = input_shape_ivalue.type();
  auto list = c10::impl::GenericList(elementType);
  list.push_back(input_shape_ivalue);
  list.push_back(input_shape_ivalue);

  torch::jit::IValue complex_input_shape(list);
  std::tuple<torch::jit::IValue> input_tuple2(complex_input_shape);
  torch::jit::IValue complex_input_shape2(input_tuple2);

  auto compile_settings = torch_tensorrt::ts::CompileSpec(complex_input_shape2);
  compile_settings.min_block_size = 1;

  // // FP16 execution
  compile_settings.enabled_precisions = {torch::kHalf};
  // // Compile module
  auto trt_mod = torch_tensorrt::torchscript::compile(mod, compile_settings);
  auto trt_out = trt_mod.forward(complex_inputs);

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(
      out.toList().vec()[0].toTensor(), trt_out.toList().vec()[0].toTensor(), 1e-5));
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(
      out.toList().vec()[1].toTensor(), trt_out.toList().vec()[1].toTensor(), 1e-5));
}

TEST(CppAPITests, TestCollectionComplexModel) {
  std::string path = "tests/modules/list_input_tuple_output_scripted.jit.pt";
  torch::Tensor in0 = torch::randn({1, 3, 512, 512}, torch::kCUDA).to(torch::kHalf);
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

  std::vector<torch::jit::IValue> complex_inputs;
  auto input_list = c10::impl::GenericList(c10::TensorType::get());
  input_list.push_back(inputs_[0]);
  input_list.push_back(inputs_[0]);

  torch::jit::IValue input_list_ivalue = torch::jit::IValue(input_list);

  complex_inputs.push_back(input_list_ivalue);

  auto out = mod.forward(complex_inputs);

  auto input_shape = torch_tensorrt::Input(in0.sizes(), torch_tensorrt::DataType::kHalf);

  auto input_shape_ivalue = torch::jit::IValue(std::move(c10::make_intrusive<torch_tensorrt::Input>(input_shape)));

  c10::TypePtr elementType = input_shape_ivalue.type();
  auto list = c10::impl::GenericList(elementType);
  list.push_back(input_shape_ivalue);
  list.push_back(input_shape_ivalue);

  torch::jit::IValue complex_input_shape(list);
  std::tuple<torch::jit::IValue> input_tuple2(complex_input_shape);
  torch::jit::IValue complex_input_shape2(input_tuple2);

  auto compile_settings = torch_tensorrt::ts::CompileSpec(complex_input_shape2);
  compile_settings.min_block_size = 1;

  // // FP16 execution
  compile_settings.enabled_precisions = {torch::kHalf};
  // // Compile module
  auto trt_mod = torch_tensorrt::torchscript::compile(mod, compile_settings);
  auto trt_out = trt_mod.forward(complex_inputs);

  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(
      out.toTuple()->elements()[0].toTensor(), trt_out.toTuple()->elements()[0].toTensor(), 1e-5));
  ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(
      out.toTuple()->elements()[1].toTensor(), trt_out.toTuple()->elements()[1].toTensor(), 1e-5));
}