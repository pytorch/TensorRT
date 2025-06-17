#pragma once

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "NvInfer.h"

namespace torch_tensorrt {
namespace core {
namespace runtime {

enum TraceFormat { kPERFETTO, kTREX };

struct TRTEngineProfiler : public nvinfer1::IProfiler {
  struct Record {
    float time{0};
    int count{0};
  };
  void set_profile_format(TraceFormat format);
  virtual void reportLayerTime(const char* layerName, float ms) noexcept;
  TRTEngineProfiler(
      const std::string& name,
      const std::vector<TRTEngineProfiler>& srcProfilers = std::vector<TRTEngineProfiler>());
  friend std::ostream& operator<<(std::ostream& out, const TRTEngineProfiler& value);
  friend void dump_trace(const std::string& path, const TRTEngineProfiler& value);

 private:
  std::string name;
  std::vector<std::string> layer_names;
  std::map<std::string, Record> profile;
  TraceFormat profile_format = TraceFormat::kPERFETTO;
};

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt