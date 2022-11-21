#include <algorithm>
#include <fstream>
#include <iomanip>

#include "core/runtime/TRTEngineProfiler.h"

namespace torch_tensorrt {
namespace core {
namespace runtime {

void TRTEngineProfiler::reportLayerTime(const char* layer_name, float ms) noexcept {
  profile[layer_name].count++;
  profile[layer_name].time += ms;
  if (std::find(layer_names.begin(), layer_names.end(), layer_name) == layer_names.end()) {
    layer_names.push_back(layer_name);
  }
}

TRTEngineProfiler::TRTEngineProfiler(const std::string& name, const std::vector<TRTEngineProfiler>& srcProfilers)
    : name(name) {
  for (const auto& srcProfiler : srcProfilers) {
    for (const auto& rec : srcProfiler.profile) {
      auto it = profile.find(rec.first);
      if (it == profile.end()) {
        profile.insert(rec);
      } else {
        it->second.time += rec.second.time;
        it->second.count += rec.second.count;
      }
    }
  }
}

void dump_trace(const std::string& path, const TRTEngineProfiler& value) {
  std::stringstream out;
  out << "[" << std::endl;
  double ts = 0.0;
  for (size_t i = 0; i < value.layer_names.size(); i++) {
    auto layer_name = value.layer_names[i];
    auto elem = value.profile.at(layer_name);

    out << "  {" << std::endl;
    out << "    \"name\": \"" << layer_name << "\"," << std::endl;
    out << "    \"ph\": \"X\"," << std::endl;
    out << "    \"ts\": " << ts * 1000 << "," << std::endl;
    out << "    \"dur\": " << elem.time * 1000 << "," << std::endl;
    out << "    \"tid\": 1," << std::endl;
    out << "    \"pid\": \"" << value.name << " Engine Execution\"," << std::endl;
    out << "    \"args\": {}" << std::endl;
    out << "  }," << std::endl;

    ts += elem.time;
  }
  out.seekp(-2, out.cur);
  out << "\n]" << std::endl;
  std::ofstream f(path);
  f << out.str();
  f.close();
  return;
}

std::ostream& operator<<(std::ostream& out, const TRTEngineProfiler& value) {
  out << "========== " << value.name << " profile ==========" << std::endl;
  float totalTime = 0;
  std::string layer_name = "TensorRT layer name";
  int max_layer_name_len = std::max(static_cast<int>(layer_name.size()), 70);
  for (const auto& elem : value.profile) {
    totalTime += elem.second.time;
    max_layer_name_len = std::max(max_layer_name_len, static_cast<int>(elem.first.size()));
  }

  auto old_settings = out.flags();
  auto old_precision = out.precision();
  // Output header
  {
    out << std::setfill(' ') << std::setw(max_layer_name_len) << layer_name << " ";
    out << std::setw(12) << "Runtime, "
        << "%"
        << " ";
    out << std::setw(12) << "Invocations"
        << " ";
    out << std::setw(12) << "Runtime, ms" << std::endl;
  }
  for (size_t i = 0; i < value.layer_names.size(); i++) {
    layer_name = value.layer_names[i];
    auto elem = value.profile.at(layer_name);
    out << std::setw(max_layer_name_len) << layer_name << " ";
    out << std::setw(12) << std::fixed << std::setprecision(1) << (elem.time * 100.0F / totalTime) << "%"
        << " ";
    out << std::setw(12) << elem.count << " ";
    out << std::setw(12) << std::fixed << std::setprecision(2) << elem.time << std::endl;
  }
  out.flags(old_settings);
  out.precision(old_precision);
  out << "========== " << value.name << " total runtime = " << totalTime << " ms ==========" << std::endl;

  return out;
}

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt