#pragma once
#include "ATen/tensor.h"
#include "NvInfer.h"

namespace trtorch {
namespace core {
namespace quantization {

enum class CalibratorKind {
  kEntropy,
  kMinMax,
}

in conveter or whatever
in order given std::vector<at::Tensor> -> map<input_name, at::Tensor>

struct QuantizationSettings {
  CalibratorKind calibrator_type = CalibratorKind::kEntropy;
  const std::string& calibration_cache_file = "";
  bool use_cache = false;
  std::unordered_map<std::string, at::Tensor> calibration_dataset;
};

class CalibrationBatchStream {

};

class Int8CalibratorImpl {
public:
  TRTInt8CalibratorImpl(QuantizationSettings& settings);
  int GetBatchSize() const;
  bool GetBatch(void* bindings[], const char* names[], int num_bindings);
  const void* ReadCalibrationCache(size_t& length);
  void WriteCalibrationCache(const void* cache, size_t length);
private:
  std::unordered_map<std::string, at::Tensor> dataset_;
  const std::string& cache_file_path_;
  std::vector<char> cache_;
  bool use_cache_;
  size_t cache_size_ = 0;
};

class TRTInt8EntropyCalibrator : nvinfer1::IInt8EntropyCalibrator2 {
public:
  TRTInt8EntropyCalibrator(Int8CalibratorImpl impl) : impl_(impl) {}
  int getBatchSize() const override {return impl_.GetBatchSize();}
  bool getBatch(void* bindings[], const char* names[], int nbBindings) override {return impl_.GetBatch(bindings, names, nbBindings)};
  const void* readCalibrationCache(size_t& length) override {return impl_.ReadCalibrationCache(size_t& length)};
  void writeCalibrationCache(const void* cache, size_t length) override {impl_.WriteCalibrationCache(const void* cache, size_t length)};
private:
  Int8CalibratorImpl impl_;
};

class TRTInt8MinMaxCalibrator : nvinfer1::IInt8MinMaxCalibrator {
public:
  TRTInt8EntropyCalibrator(Int8CalibratorImpl impl) : impl_(impl) {}
  int getBatchSize() const override {return impl_.GetBatchSize();}
  bool getBatch(void* bindings[], const char* names[], int nbBindings) override {return impl_.GetBatch(bindings, names, nbBindings)};
  const void* readCalibrationCache(size_t& length) override {return impl_.ReadCalibrationCache(size_t& length)};
  void writeCalibrationCache(const void* cache, size_t length) override {impl_.WriteCalibrationCache(const void* cache, size_t length)};
private:
  Int8CalibratorImpl impl_;
};

nvinfer1::IInt8Calibrator create_int8_calibrator(QuantizationSettings settings);

} // quantization
} // core
} // trtorch