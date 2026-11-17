// SPDX-License-Identifier: BSD-3-Clause
#include <NvInfer.h>
#include <dlfcn.h>

#include "benchmark.h"
using namespace pbr;

struct Logger : nvinfer1::ILogger {
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kWARNING) std::cerr << message << '\n';
  }
};
int main(int argc, char** argv) {
  try {
    require(argc == 5,
            "Usage: benchmark_trt MODEL_DIR FIXTURE OUTPUT MODEL_NAME");
    fs::path model = argv[1];
    std::string family = argv[4];
    require(family == "segformer" || family == "bevformer", "Unknown model");
    Report report(argv[3]);
    cuda_check(cudaSetDevice(0), "select GPU");
    Stream stream;
    cuda_check(cudaDeviceSynchronize(), "initialize CUDA");
    report.snapshot("cuda_initialized");
    if (family == "bevformer") {
      auto plugin = dlopen((model / "libbev_geometry.so").c_str(),
                           RTLD_NOW | RTLD_GLOBAL);
      if (!plugin) throw std::runtime_error(dlerror());
    }
    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime(
        nvinfer1::createInferRuntime(logger));
    require(bool(runtime), "Cannot create TensorRT runtime");
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    {
      auto file = model / "model.engine";
      std::vector<char> bytes(fs::file_size(file));
      std::ifstream input(file, std::ios::binary);
      require(!bytes.empty() && bool(input.read(bytes.data(), bytes.size())),
              "Cannot read engine");
      engine.reset(runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
      require(
          bool(engine),
          "Engine deserialization failed; check target and TensorRT version");
    }
    std::unique_ptr<nvinfer1::IExecutionContext> context(
        engine->createExecutionContext());
    require(bool(context), "Cannot create context");
    cuda_check(cudaDeviceSynchronize(), "context load");
    report.snapshot("method_loaded");
    const std::vector<std::string> names =
        family == "segformer"
            ? std::vector<std::string>{"pixel_values", "output0"}
            : std::vector<std::string>{"images",   "lidar2img", "can_bus",
                                       "prev_bev", "has_prev",  "logits",
                                       "boxes",    "next_bev"};
    size_t ni = family == "segformer" ? 1 : 5,
           no = family == "segformer" ? 1 : 3;
    require(engine->getNbIOTensors() == int(names.size()),
            "Engine IO count mismatch");
    Fixtures fixtures(argv[2], ni, no);
    std::vector<std::unique_ptr<DeviceBuffer>> device;
    for (size_t i = 0; i < names.size(); ++i) {
      auto& t = i < ni ? fixtures.inputs[i] : fixtures.expected[i - ni];
      const char* name = names[i].c_str();
      auto shape = engine->getTensorShape(name);
      require(
          engine->getTensorIOMode(name) ==
                  (i < ni ? nvinfer1::TensorIOMode::kINPUT
                          : nvinfer1::TensorIOMode::kOUTPUT) &&
              engine->getTensorDataType(name) == nvinfer1::DataType::kFLOAT &&
              engine->getTensorLocation(name) ==
                  nvinfer1::TensorLocation::kDEVICE &&
              engine->getTensorFormat(name) == nvinfer1::TensorFormat::kLINEAR,
          "Unsupported engine IO");
      require(shape.nbDims == int(t.shape.size()), "Rank mismatch");
      for (int d = 0; d < shape.nbDims; ++d)
        require(shape.d[d] == t.shape[d], "Shape mismatch");
      device.push_back(std::make_unique<DeviceBuffer>(t.bytes()));
      require(context->setTensorAddress(name, device.back()->value),
              "Cannot bind tensor");
    }
    PinnedIO pinned(fixtures);
    report.snapshot("fixtures_loaded");
    report.measure(fixtures, [&]() {
      for (size_t i = 0; i < ni; ++i)
        cuda_check(
            cudaMemcpyAsync(device[i]->value, fixtures.inputs[i].data.data(),
                            fixtures.inputs[i].bytes(), cudaMemcpyHostToDevice,
                            stream.value),
            "H2D");
      require(context->enqueueV3(stream.value), "TensorRT execution failed");
      for (size_t i = 0; i < no; ++i)
        cuda_check(
            cudaMemcpyAsync(fixtures.outputs[i].data(), device[ni + i]->value,
                            fixtures.expected[i].bytes(),
                            cudaMemcpyDeviceToHost, stream.value),
            "D2H");
      cuda_check(cudaDeviceSynchronize(), "complete inference");
    });
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
