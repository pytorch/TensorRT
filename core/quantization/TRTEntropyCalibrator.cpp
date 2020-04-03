#include "core/util/prelude.h"
#include "core/quantization/quantization.h"

namespace trtorch {
namespace core {
namespace quantization {

Int8CalibratorImpl::Int8CalibratorImpl(QuantizationSettings&& settings)
  : dataset_(std::move(settings.calibration_dataset),
    cache_file_path_(settings.calibration_cache_file),
    use_cache_(settings.use_cache) {
  buffers_.reserve(dataset_.size);

}

int Int8CalibratorImpl::GetBatchSize() const {

}

bool Int8CalibratorImpl::GetBatch(void* bindings[], const char* names[], int num_bindings) {
    if (!is_next_batch) {
        return false;
    }

    for (size_t i = 0; i < num_bindings; i++) {
        auto batch = next_binding_batch(names[i]);
        batch = batch.to(at::kCUDA).contiguous();
        bindings[i] = batch.data_ptr();
    }
    return true;
}

const void* Int8CalibratorImpl::ReadCalibrationCache(size_t& length) {
    cache_.clear();
    std::ifstream cache_file(cache_file_path_, std::ios::binary);
    cache_file >> std::noskipws;
    if (use_cache && cache_file.good()) {
        std::copy(std::istream_iterator<char>(input),
                  std::istream_iterator<char>(),
                  std::back_inserter(cache_));
    }
    cache_size_ = cache_.size();
    return cache_size ? cache_.data() : nullptr;
}

void Int8CalibratorImpl::WriteCalibrationCache(const void* cache, size_t length) {
    std::ofstream cache_file(cache_file_path_, std::ios::binary);
    cache_file.write(reinterpret_cast<const char*>(cache_), cache_size_);
}

nvinfer1::IInt8Calibrator create_int8_calibrator(QuantizationSettings settings) {
  auto calibrator_impl = Int8CalibratorImpl(settings);
  switch(settings.calibrator_type) {
  case CalibratorKind::kMinMax:
    return TRTInt8MinMaxCalibrator(std::move(calibrator_impl));
  case CalibratorKind::kEntropy:
  default:
    return TRTInt8EntropyCalibrator(std::move(calibrator_impl));
  }
}

} // quantization
} // core
} // trtorch
