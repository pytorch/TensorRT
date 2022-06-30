#include <stdlib.h>
#include <iostream>
#include <sstream>

#include "NvInfer.h"
#include "third_party/args/args.hpp"
#include "torch/script.h"

#include "torch_tensorrt/logging.h"
#include "torch_tensorrt/ptq.h"
#include "torch_tensorrt/torch_tensorrt.h"

#include "accuracy.h"
#include "fileio.h"
#include "luts.h"
#include "parser_util.h"

int main(int argc, char** argv) {
  torchtrt::logging::set_is_colored_output_on(true);
  torchtrt::logging::set_reportable_log_level(torchtrt::logging::Level::kWARNING);
  torchtrt::logging::set_logging_prefix("");

  args::ArgumentParser parser(
      "torchtrtc is a compiler for TorchScript, it will compile and optimize TorchScript programs to run on NVIDIA GPUs using TensorRT",
      "");
  args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});

  args::Group group(parser, "Verbiosity of the compiler", args::Group::Validators::AtMostOne);
  args::Flag verbose(
      group, "verbose", "Dumps debugging information about the compilation process onto the console", {'v', "verbose"});
  args::Flag warning(
      group,
      "warning",
      "Disables warnings generated during compilation onto the console (warnings are on by default)",
      {'w', "warnings"});
  args::Flag info(group, "info", "Dumps info messages generated during compilation onto the console", {"i", "info"});

  args::Flag build_debuggable_engine(
      parser, "build-debuggable-engine", "Creates a debuggable engine", {"build-debuggable-engine"});

  args::Flag allow_gpu_fallback(
      parser,
      "allow-gpu-fallback",
      "(Only used when targeting DLA (device-type)) Lets engine run layers on GPU if they are not supported on DLA",
      {"allow-gpu-fallback"});

  args::Flag require_full_compilation(
      parser,
      "require-full-compilation",
      "Require that the model should be fully compiled to TensorRT or throw an error",
      {"require-full-compilation"});

  args::ValueFlag<std::string> check_method_op_support(
      parser,
      "method_name",
      "Check the support for end to end compilation of a specified method in the TorchScript module",
      {"check-method-support"});

  args::Flag disable_tf32(
      parser, "disable-tf32", "Prevent Float32 layers from using the TF32 data format", {"disable-tf32"});

  args::Flag sparse_weights(
      parser, "sparse-weights", "Enable sparsity for weights of conv and FC layers", {"sparse-weights"});

  args::ValueFlagList<std::string> enabled_precisions(
      parser,
      "precision",
      "(Repeatable) Enabling an operating precision for kernels to use when building the engine (Int8 requires a calibration-cache argument) [ float | float32 | f32 | fp32 | half | float16 | f16 | fp16 | int8 | i8 | char ] (default: float)",
      {'p', "enable-precision"});
  args::ValueFlag<std::string> device_type(
      parser,
      "type",
      "The type of device the engine should be built for [ gpu | dla ] (default: gpu)",
      {'d', "device-type"});
  args::ValueFlag<uint64_t> gpu_id(
      parser, "gpu_id", "GPU id if running on multi-GPU platform (defaults to 0)", {"gpu-id"});
  args::ValueFlag<uint64_t> dla_core(
      parser, "dla_core", "DLACore id if running on available DLA (defaults to 0)", {"dla-core"});

  args::ValueFlag<std::string> engine_capability(
      parser,
      "capability",
      "The type of device the engine should be built for [ standard | safety | dla_standalone ]",
      {"engine-capability"});

  args::ValueFlag<std::string> calibration_cache_file(
      parser,
      "file_path",
      "Path to calibration cache file to use for post training quantization",
      {"calibration-cache-file"});

  args::ValueFlagList<std::string> torch_executed_ops(
      parser,
      "op_name",
      "(Repeatable) Operator in the graph that should always be run in PyTorch for execution (partial compilation must be enabled)",
      {"teo", "torch-executed-op"});

  args::ValueFlagList<std::string> torch_executed_mods(
      parser,
      "module_name",
      "(Repeatable) Module that should always be run in Pytorch for execution (partial compilation must be enabled)",
      {"tem", "torch-executed-mod"});

  args::ValueFlag<uint64_t> min_block_size(
      parser,
      "num_ops",
      "Minimum number of contiguous TensorRT supported ops to compile a subgraph to TensorRT",
      {"mbs", "min-block-size"});

  args::Flag embed_engine(
      parser,
      "embed-engine",
      "Whether to treat input file as a serialized TensorRT engine and embed it into a TorchScript module (device spec must be provided)",
      {"embed-engine"});

  args::ValueFlag<uint64_t> num_min_timing_iters(
      parser, "num_iters", "Number of minimization timing iterations used to select kernels", {"num-min-timing-iter"});
  args::ValueFlag<uint64_t> num_avg_timing_iters(
      parser, "num_iters", "Number of averaging timing iterations used to select kernels", {"num-avg-timing-iters"});
  args::ValueFlag<uint64_t> workspace_size(
      parser, "workspace_size", "Maximum size of workspace given to TensorRT", {"workspace-size"});
  args::ValueFlag<double> atol(
      parser,
      "atol",
      "Absolute tolerance threshold for acceptable numerical deviation from standard torchscript output (default 1e-8)",
      {"atol"});
  args::ValueFlag<double> rtol(
      parser,
      "rtol",
      "Relative tolerance threshold for acceptable numerical deviation from standard torchscript output (default 1e-5)",
      {"rtol"});

  args::Flag no_threshold_check(
      parser, "no-threshold-check", "Skip checking threshold compliance", {"no-threshold-check", "no-threshold-check"});
  args::Flag truncate_long_and_double(
      parser,
      "truncate-long-double",
      "Truncate weights that are provided in 64bit to 32bit (Long, Double to Int, Float)",
      {"truncate", "truncate-long-double", "truncate-64bit"});

  args::Flag save_engine(
      parser,
      "save_engine",
      "Instead of compiling a full a TorchScript program, save the created engine to the path specified as the output path",
      {"save-engine"});
  args::Positional<std::string> input_path(parser, "input_file_path", "Path to input TorchScript file");
  args::Positional<std::string> output_path(
      parser, "output_file_path", "Path for compiled TorchScript (or TensorRT engine) file");
  args::PositionalList<std::string> input_shapes(
      parser,
      "input_specs",
      "Specs for inputs to engine, can either be a single size or a range defined by Min, Optimal, Max sizes, e.g. \"(N,..,C,H,W)\" \"[(MIN_N,..,MIN_C,MIN_H,MIN_W);(OPT_N,..,OPT_C,OPT_H,OPT_W);(MAX_N,..,MAX_C,MAX_H,MAX_W)]\". Data Type and format can be specified by adding an \"@\" followed by dtype and \"%\" followed by format to the end of the shape spec. e.g. \"(3, 3, 32, 32)@f16\%NHWC\"");

  try {
    parser.ParseCLI(argc, argv);
  } catch (args::Help const&) {
    std::cout << parser;
    return 0;
  } catch (args::ParseError const& e) {
    torchtrt::logging::log(torchtrt::logging::Level::kERROR, e.what());
    std::cerr << std::endl << parser;
    return 1;
  }

  if (verbose) {
    torchtrt::logging::set_reportable_log_level(torchtrt::logging::Level::kDEBUG);
  } else if (info) {
    torchtrt::logging::set_reportable_log_level(torchtrt::logging::Level::kINFO);
  } else if (warning) {
    torchtrt::logging::set_reportable_log_level(torchtrt::logging::Level::kERROR);
  }

  auto real_input_path = torchtrtc::fileio::resolve_path(args::get(input_path));

  if (check_method_op_support) {
    torch::jit::Module mod;
    try {
      // Deserialize the ScriptModule from a file using torch::jit::load().
      mod = torch::jit::load(real_input_path);
    } catch (const c10::Error& e) {
      torchtrt::logging::log(torchtrt::logging::Level::kERROR, "Error loading the model (path may be incorrect)");
      return 1;
    }

    auto method = args::get(check_method_op_support);
    auto result = torchtrt::ts::check_method_operator_support(mod, method);
    if (result) {
      std::cout << "The method is supported end to end by Torch-TensorRT" << std::endl;
      return 0;
    } else {
      torchtrt::logging::log(torchtrt::logging::Level::kERROR, "Method is not currently supported by Torch-TensorRT");
      return 1;
    }
  }

  auto real_output_path = torchtrtc::fileio::resolve_path(args::get(output_path));

  // Instead of compiling, just embed engine in a PyTorch module
  if (embed_engine) {
    auto device_str = args::get(device_type);
    std::transform(
        device_str.begin(), device_str.end(), device_str.begin(), [](unsigned char c) { return std::tolower(c); });

    torchtrt::Device device;

    if (gpu_id) {
      device.gpu_id = args::get(gpu_id);
      torchtrt::set_device(device.gpu_id);
    }

    if (device_str == "gpu") {
      device.device_type = torchtrt::Device::DeviceType::kGPU;
    } else if (device_str == "dla") {
      device.device_type = torchtrt::Device::DeviceType::kDLA;
      if (dla_core) {
        device.dla_core = args::get(dla_core);
      }
    } else {
      torchtrt::logging::log(
          torchtrt::logging::Level::kERROR, "Invalid device type, options are [ gpu | dla ] found: " + device_type);
      std::cerr << std::endl << parser;
      return 1;
    }

    std::string serialized_engine = torchtrtc::fileio::read_buf(real_input_path);
    auto trt_mod = torchtrt::ts::embed_engine_in_new_module(serialized_engine, device);
    trt_mod.save(real_output_path);
    return 0;
  }

  std::vector<torchtrt::Input> ranges;
  for (const auto spec : args::get(input_shapes)) {
    ranges.push_back(torchtrtc::parserutil::parse_input(spec));
    std::stringstream ss;
    ss << "Parsed Input: " << ranges.back();
    torchtrt::logging::log(torchtrt::logging::Level::kDEBUG, ss.str());
  }

  auto compile_settings = torchtrt::ts::CompileSpec(ranges);

  if (build_debuggable_engine) {
    compile_settings.debug = true;
  }

  if (allow_gpu_fallback) {
    compile_settings.device.allow_gpu_fallback = true;
  }

  if (disable_tf32) {
    compile_settings.disable_tf32 = true;
  }

  if (sparse_weights) {
    compile_settings.sparse_weights = true;
  }

  std::string calibration_cache_file_path = "";
  if (calibration_cache_file) {
    calibration_cache_file_path = torchtrtc::fileio::resolve_path(args::get(calibration_cache_file));
  }

  auto calibrator = torchtrt::ptq::make_int8_cache_calibrator(calibration_cache_file_path);

  compile_settings.require_full_compilation = require_full_compilation;

  if (torch_executed_ops || torch_executed_mods) {
    if (require_full_compilation) {
      torchtrt::logging::log(
          torchtrt::logging::Level::kERROR,
          "Ops or modules to run in torch were provided but full compilation was requested. Please remove --require-full-compilation to run specified ops and modules in torch.");
      exit(1);
    }

    compile_settings.min_block_size = min_block_size;

    for (const auto _op : args::get(torch_executed_ops)) {
      compile_settings.torch_executed_ops.push_back(_op);
    }

    for (const auto _mod : args::get(torch_executed_mods)) {
      compile_settings.torch_executed_modules.push_back(_mod);
    }
  }

  if (enabled_precisions) {
    for (const auto precision : args::get(enabled_precisions)) {
      auto dtype = torchtrtc::parserutil::parse_dtype(precision);
      if (dtype == torchtrt::DataType::kFloat) {
        compile_settings.enabled_precisions.insert(torch::kF32);
      } else if (dtype == torchtrt::DataType::kHalf) {
        compile_settings.enabled_precisions.insert(torch::kF16);
      } else if (dtype == torchtrt::DataType::kChar) {
        compile_settings.enabled_precisions.insert(torch::kI8);
        if (calibration_cache_file) {
          compile_settings.ptq_calibrator = calibrator;
        } else {
          torchtrt::logging::log(
              torchtrt::logging::Level::kINFO,
              "Int8 precision has been enabled but no calibrator provided. This assumes the network has Q/DQ nodes obtained from Quantization aware training. For more details, refer to https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work-with-qat-networks");
        }
      } else {
        std::stringstream ss;
        ss << "Invalid precision given for enabled kernel precision, options are [ float | float32 | f32 | fp32 | half | float16 | f16 | fp16 | char | int8 | i8 ], found: ";
        ss << dtype;
        torchtrt::logging::log(torchtrt::logging::Level::kERROR, ss.str());
        std::cerr << std::endl << parser;
        return 1;
      }
    }
  }

  if (device_type) {
    auto device = args::get(device_type);
    std::transform(device.begin(), device.end(), device.begin(), [](unsigned char c) { return std::tolower(c); });

    if (gpu_id) {
      compile_settings.device.gpu_id = args::get(gpu_id);
      torchtrt::set_device(compile_settings.device.gpu_id);
    }

    if (device == "gpu") {
      compile_settings.device.device_type = torchtrt::Device::DeviceType::kGPU;
    } else if (device == "dla") {
      compile_settings.device.device_type = torchtrt::Device::DeviceType::kDLA;
      if (dla_core) {
        compile_settings.device.dla_core = args::get(dla_core);
      }
    } else {
      torchtrt::logging::log(
          torchtrt::logging::Level::kERROR, "Invalid device type, options are [ gpu | dla ] found: " + device);
      std::cerr << std::endl << parser;
      return 1;
    }
  }

  if (engine_capability) {
    auto capability = args::get(engine_capability);
    std::transform(
        capability.begin(), capability.end(), capability.begin(), [](unsigned char c) { return std::tolower(c); });
    if (capability == "standard") {
      compile_settings.capability = torchtrt::EngineCapability::kSTANDARD;
    } else if (capability == "safety") {
      compile_settings.capability = torchtrt::EngineCapability::kSAFETY;
    } else if (capability == "dla_standalone") {
      compile_settings.capability = torchtrt::EngineCapability::kDLA_STANDALONE;
    } else {
      torchtrt::logging::log(
          torchtrt::logging::Level::kERROR,
          "Invalid engine capability, options are [ standard | safety | dla_standalone ]");
      std::cerr << std::endl << parser;
      return 1;
    }
  }

  if (num_min_timing_iters) {
    compile_settings.num_min_timing_iters = args::get(num_min_timing_iters);
  }

  if (num_avg_timing_iters) {
    compile_settings.num_avg_timing_iters = args::get(num_avg_timing_iters);
  }

  if (workspace_size) {
    compile_settings.workspace_size = args::get(workspace_size);
  }

  if (truncate_long_and_double) {
    compile_settings.truncate_long_and_double = true;
  }

  torch::jit::Module mod;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    mod = torch::jit::load(real_input_path);
  } catch (const c10::Error& e) {
    torchtrt::logging::log(torchtrt::logging::Level::kERROR, "Error loading the model (path may be incorrect)");
    return 1;
  }

  if (require_full_compilation) {
    if (!torchtrt::ts::check_method_operator_support(mod, "forward")) {
      torchtrt::logging::log(torchtrt::logging::Level::kERROR, "Module is not currently supported by Torch-TensorRT");
      return 1;
    }
  }

  if (save_engine) {
    auto engine = torchtrt::ts::convert_method_to_trt_engine(mod, "forward", compile_settings);
    std::ofstream out(real_output_path);
    out << engine;
    out.close();
    return 0;
  } else {
    auto trt_mod = torchtrt::ts::compile(mod, compile_settings);

    if (!no_threshold_check &&
        (compile_settings.enabled_precisions.size() == 1 &&
         compile_settings.enabled_precisions.find(torchtrt::DataType::kFloat) !=
             compile_settings.enabled_precisions.end())) {
      double atol_val = 1e-8;
      double rtol_val = 1e-5;
      if (atol) {
        atol_val = args::get(atol);
      }
      if (rtol) {
        rtol_val = args::get(rtol);
      }

      std::vector<torch::jit::IValue> jit_inputs_ivalues;
      std::vector<torch::jit::IValue> trt_inputs_ivalues;

      for (auto i : ranges) {
        auto in = at::randn(i.opt_shape, {at::kCUDA});
        in = in.to(torchtrtc::luts::to_torch_dtype(i.dtype));
        jit_inputs_ivalues.push_back(in.clone());
        trt_inputs_ivalues.push_back(in.clone());
      }

      mod.to({at::kCUDA});
      torch::jit::IValue jit_results_ivalues = mod.forward(jit_inputs_ivalues);
      std::vector<at::Tensor> jit_results;
      if (jit_results_ivalues.isTensor()) {
        jit_results.push_back(jit_results_ivalues.toTensor());
      } else {
        auto results = jit_results_ivalues.toTuple()->elements();
        for (auto r : results) {
          jit_results.push_back(r.toTensor());
        }
      }

      torch::jit::IValue trt_results_ivalues = trt_mod.forward(trt_inputs_ivalues);
      std::vector<at::Tensor> trt_results;
      if (trt_results_ivalues.isTensor()) {
        trt_results.push_back(trt_results_ivalues.toTensor());
      } else {
        auto results = trt_results_ivalues.toTuple()->elements();
        for (auto r : results) {
          trt_results.push_back(r.toTensor());
        }
      }

      for (size_t i = 0; i < trt_results.size(); i++) {
        std::ostringstream threshold_ss;
        threshold_ss << "atol: " << atol_val << " rtol: " << rtol_val;
        if (!torchtrtc::accuracy::almost_equal(
                jit_results[i], trt_results[i].reshape_as(jit_results[i]), atol_val, rtol_val)) {
          torchtrt::logging::log(
              torchtrt::logging::Level::kWARNING,
              std::string("Maximum numerical deviation for output exceeds tolerance thresholds (") +
                  threshold_ss.str() + std::string(")"));
        } else {
          torchtrt::logging::log(
              torchtrt::logging::Level::kDEBUG,
              std::string("Maximum numerical deviation within threshold limits ") + threshold_ss.str());
        }
      }
    } else {
      if (no_threshold_check) {
        torchtrt::logging::log(
            torchtrt::logging::Level::kWARNING, "Threshold check skipped, numerical precision is not checked");
      } else {
        torchtrt::logging::log(
            torchtrt::logging::Level::kWARNING,
            "Due to change in operating data type, numerical precision is not checked");
      }
    }

    trt_mod.save(real_output_path);
  }

  return 0;
}
