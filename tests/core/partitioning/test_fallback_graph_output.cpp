#include <string>
#include <unordered_set>
#include "core/compiler.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/script.h"

#ifndef DISABLE_TEST_IN_CI

// TEST(Partitioning, ComputeResNet50FallbackGraphCorrectly) {
//   torch::jit::script::Module mod;
//   try {
//     mod = torch::jit::load("tests/modules/resnet50_traced.jit.pt");
//   } catch (const c10::Error& e) {
//     std::cerr << "error loading the model\n";
//     return;
//   }
//
//   const std::vector<std::vector<int64_t>> input_shapes = {{1, 3, 224, 224}};
//   std::vector<torch::jit::IValue> jit_inputs_ivalues;
//   std::vector<torch::jit::IValue> trt_inputs_ivalues;
//   for (auto in_shape : input_shapes) {
//     auto in = at::randint(5, in_shape, {at::kCUDA});
//     jit_inputs_ivalues.push_back(in.clone());
//     trt_inputs_ivalues.push_back(in.clone());
//   }
//
//   std::vector<torch_tensorrt::core::ir::Input> input_ranges{torch_tensorrt::core::ir::Input({1, 3, 224, 224})};
//
//   torch_tensorrt::core::CompileSpec cfg(input_ranges);
//   cfg.partition_info.enabled = true;
//   cfg.partition_info.forced_fallback_operators.push_back("aten::add");
//
//   auto jit_results = mod.forward(jit_inputs_ivalues).toTensor();
//   auto trt_mod = torch_tensorrt::core::CompileGraph(mod, cfg);
//   auto trt_results = trt_mod.forward(trt_inputs_ivalues).toTensor();
//   ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results, trt_results, 2e-6));
// }
//
// TEST(Partitioning, ComputeMobileNetFallbackGraphCorrectly) {
//   torch::jit::script::Module mod;
//   try {
//     mod = torch::jit::load("tests/modules/mobilenet_v2_traced.jit.pt");
//   } catch (const c10::Error& e) {
//     std::cerr << "error loading the model\n";
//     return;
//   }
//
//   const std::vector<std::vector<int64_t>> input_shapes = {{1, 3, 224, 224}};
//   std::vector<torch::jit::IValue> jit_inputs_ivalues;
//   std::vector<torch::jit::IValue> trt_inputs_ivalues;
//   for (auto in_shape : input_shapes) {
//     auto in = at::randint(5, in_shape, {at::kCUDA});
//     jit_inputs_ivalues.push_back(in.clone());
//     trt_inputs_ivalues.push_back(in.clone());
//   }
//
//   std::vector<torch_tensorrt::core::ir::Input> input_ranges{torch_tensorrt::core::ir::Input({1, 3, 224, 224})};
//   auto g = mod.get_method("forward").graph();
//   torch_tensorrt::core::CompileSpec cfg(input_ranges);
//   cfg.partition_info.enabled = true;
//   cfg.partition_info.forced_fallback_operators.push_back("aten::hardtanh");
//
//   auto jit_results = mod.forward(jit_inputs_ivalues).toTensor();
//   auto trt_mod = torch_tensorrt::core::CompileGraph(mod, cfg);
//   auto trt_results = trt_mod.forward(trt_inputs_ivalues).toTensor();
//   ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results, trt_results, 2e-6));
// }

/*
The following test is ambigious and somehow works in TRT 8.2, which might have a bug.
This FP16 model has inputs and weights configured to be FP16 but the builder precision
is set to FP32. So during shape analysis, when the Pyt/TRT segments (are run as pytorch
modules), the inputs of each segments are configured to be FP16 but after TRT conversion
and inference, TRT segments generate float outputs which become float inputs to following
segments. Hence type check fails during runtime at
https://github.com/pytorch/TensorRT/blob/master/core/runtime/execute_engine.cpp#L91
TO DO: Resolve type system check in partitioning
*/

// TEST(Partitioning, ComputeResNet50HalfFallbackGraphCorrectly) {
//   torch::jit::script::Module mod;
//   try {
//     mod = torch::jit::load("tests/modules/resnet50_traced.jit.pt");
//   } catch (const c10::Error& e) {
//     std::cerr << "error loading the model\n";
//     return;
//   }
//
//   mod.to(torch::kHalf);
//
//   const std::vector<std::vector<int64_t>> input_shapes = {{1, 3, 224, 224}};
//   std::vector<torch::jit::IValue> jit_inputs_ivalues;
//   std::vector<torch::jit::IValue> trt_inputs_ivalues;
//   for (auto in_shape : input_shapes) {
//     auto in = at::randint(5, in_shape, {at::kCUDA}).to(torch::kHalf);
//     jit_inputs_ivalues.push_back(in.clone());
//     trt_inputs_ivalues.push_back(in.clone());
//   }
//
//   auto in_shape = torch_tensorrt::core::ir::Input({1, 3, 224, 224});
//   in_shape.dtype = nvinfer1::DataType::kHALF;
//
//   std::vector<torch_tensorrt::core::ir::Input> input_ranges({in_shape});
//   auto g = mod.get_method("forward").graph();
//   torch_tensorrt::core::CompileSpec cfg(input_ranges);
//   cfg.partition_info.enabled = true;
//   cfg.partition_info.forced_fallback_operators.push_back("aten::add");
//
//   auto jit_results = mod.forward(jit_inputs_ivalues).toTensor();
//   auto trt_mod = torch_tensorrt::core::CompileGraph(mod, cfg);
//   auto trt_results = trt_mod.forward(trt_inputs_ivalues).toTensor();
//   // Lower threshold because FP16
//   ASSERT_TRUE(torch_tensorrt::tests::util::almostEqual(jit_results, trt_results, 2e-1));
// }
#endif
