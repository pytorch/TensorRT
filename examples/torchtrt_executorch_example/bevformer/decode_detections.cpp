// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// BEVFormer-Tiny NMSFreeCoder postprocessing. No Torch, Python, CUDA or MMCV.
// Convention follows fundamentalvision/BEVFormer at 66b65f3: sigmoid/top-300,
// center-range filtering, exp sizes, atan2 yaw, then bottom-center z adjustment.
// This decodes predictions; it does NOT establish detection accuracy.
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;
std::vector<float> read(const fs::path& path, size_t count) {
  if (fs::file_size(path) != count * sizeof(float)) throw std::runtime_error("Bad size: " + path.string());
  std::vector<float> values(count);
  std::ifstream file(path, std::ios::binary);
  if (!file.read(reinterpret_cast<char*>(values.data()), count * sizeof(float)))
    throw std::runtime_error("Cannot read " + path.string());
  for (float value : values) if (!std::isfinite(value)) throw std::runtime_error("Nonfinite raw prediction");
  return values;
}

int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::runtime_error("Usage: decode_detections raw-output-dir new-output.tsv");
    if (fs::exists(argv[2])) throw std::runtime_error("Output exists; choose a new filename");
    constexpr size_t queries = 900, classes = 10, keep = 300;
    auto scores = read(fs::path(argv[1]) / "output_0.bin", queries * classes);
    const auto boxes = read(fs::path(argv[1]) / "output_1.bin", queries * 10);
    for (float& score : scores) score = 1.0f / (1.0f + std::exp(-score));
    std::vector<size_t> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + keep, indices.end(),
      [&](size_t a, size_t b) { return scores[a] != scores[b] ? scores[a] > scores[b] : a < b; });
    std::ofstream out(argv[2]);
    if (!out) throw std::runtime_error("Cannot create output");
    out << "query\tlabel\tscore\tcx\tcy\tz_bottom\tw\tl\th\tyaw\tvx\tvy\n" << std::setprecision(9);
    size_t count = 0;
    for (size_t k = 0; k < keep; ++k) {
      const size_t index = indices[k], query = index / classes;
      const float* b = boxes.data() + query * 10;
      // Filter CENTER coordinates before converting z to bottom-center.
      if (b[0] < -61.2f || b[0] > 61.2f || b[1] < -61.2f || b[1] > 61.2f || b[4] < -10.f || b[4] > 10.f)
        continue;
      const float w = std::exp(b[2]), l = std::exp(b[3]), h = std::exp(b[5]);
      const float yaw = std::atan2(b[6], b[7]), z = b[4] - h * .5f;
      if (!std::isfinite(w) || !std::isfinite(l) || !std::isfinite(h) || !std::isfinite(z))
        throw std::runtime_error("Nonfinite decoded box");
      out << query << '\t' << index % classes << '\t' << scores[index] << '\t'
          << b[0] << '\t' << b[1] << '\t' << z << '\t' << w << '\t' << l << '\t'
          << h << '\t' << yaw << '\t' << b[8] << '\t' << b[9] << '\n';
      ++count;
    }
    if (!out) throw std::runtime_error("Failed writing decoded predictions");
    std::cout << "Decoded " << count << " of top " << keep << " predictions\n";
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
