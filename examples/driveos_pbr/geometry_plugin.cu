// SPDX-License-Identifier: Apache-2.0
// Geometry adapted from BEVFormer/OpenMMLab; copyright (c) OpenMMLab.
// BEVFormer by Zhiqi Li. See models/bevformer/LICENSE and upstream sources.
// Experimental fixed-shape BEVFormer-Tiny geometry plugin.
// Math follows the repository's BEVFormer Geometry adapter; no PyTorch/AOTI
// runtime.
#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cmath>
#include <new>
using namespace nvinfer1;
namespace bev_plugin {
constexpr char kName[] = "BEVFormerTinyGeometry";
constexpr char kNamespace[] = "";
constexpr int kQ = 2500;
__global__ void project(const float* matrices, float* refs, float* mask,
                        float* base) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= 6 * kQ * 4) return;
  const int h = t % 4, q = t / 4 % kQ, cam = t / (4 * kQ);
  const float x = ((float(q % 50) + .5f) * 0.02f) * 102.4f - 51.2f;
  const float y = ((float(q / 50) + .5f) * 0.02f) * 102.4f - 51.2f;
  // torch.linspace uses its start for the first half and end for the second.
  const float step = 7.f / 3.f;
  const float z0 = h < 2 ? .5f + float(h) * step : 7.5f - float(3 - h) * step;
  const float z = (z0 / 8.f) * 8.f - 5.f;
  const float* m = matrices + cam * 16;
  float v[3];
  for (int r = 0; r < 3; ++r) {
    // Preserve the paired accumulation used by the tested FP32 geometry.
    float a = 0.f;
    a = fmaf(m[4 * r], x, a);
    a = fmaf(m[4 * r + 1], y, a);
    a = a + fmaf(m[4 * r + 2], z, m[4 * r + 3]);
    v[r] = a;
  }
  float d = fmaxf(v[2], 1.e-5f);
  float u = (v[0] / d) / 800.f, w = (v[1] / d) / 480.f;
  refs[2 * t] = u;
  refs[2 * t + 1] = w;
  mask[t] =
      v[2] > 1.e-5f && u > 0.f && u < 1.f && w > 0.f && w < 1.f ? 1.f : 0.f;
  if (cam == 0 && h == 0) {
    base[2 * q] = (float(q % 50) + .5f) * 0.02f;
    base[2 * q + 1] = (float(q / 50) + .5f) * 0.02f;
  }
}
__global__ void rotate(const float* previous, const float* bus, float* result,
                       float* shift) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= kQ * 256) return;
  const int q = i / 256, channel = i % 256;
  const float a = bus[17] * float(3.14159265358979323846 / 180.0);
  const float c = cosf(a), s = sinf(a);
  const float tx = 75.f - 75.f * c + 75.f * s, ty = 75.f - 75.f * s - 75.f * c;
  const float x = float(q % 50) + .5f - 25.f, y = float(q / 50) + .5f - 25.f;
  const float gx = ((x * c + y * (-s)) + tx) / 25.f;
  const float gy = ((x * s + y * c) + ty) / 25.f;
  const float ix = ((gx + 1.f) * 50.f - 1.f) * .5f;
  const float iy = ((gy + 1.f) * 50.f - 1.f) * .5f;
  const int sx = __float2int_rn(ix), sy = __float2int_rn(iy);
  result[i] = (sx >= 0 && sx < 50 && sy >= 0 && sy < 50)
                  ? previous[(sy * 50 + sx) * 256 + channel]
                  : 0.f;
  if (i == 0) {
    const float cy = cosf(bus[16]), syaw = sinf(bus[16]);
    shift[0] = (bus[0] * syaw - bus[1] * cy) / 102.4f;
    shift[1] = (bus[0] * cy + bus[1] * syaw) / 102.4f;
  }
}
bool inputShapes(const PluginTensorDesc* in, int n) {
  if (n != 3) return false;
  const int rank[3] = {4, 2, 3};
  const int dims[3][4] = {{1, 6, 4, 4}, {1, 18, 0, 0}, {1, 2500, 256, 0}};
  for (int i = 0; i < 3; ++i) {
    if (in[i].type != DataType::kFLOAT ||
        in[i].format != TensorFormat::kLINEAR || in[i].dims.nbDims != rank[i])
      return false;
    for (int j = 0; j < rank[i]; ++j)
      if (in[i].dims.d[j] != dims[i][j]) return false;
  }
  return true;
}
class Geometry final : public IPluginV3,
                       public IPluginV3OneCore,
                       public IPluginV3OneBuild,
                       public IPluginV3OneRuntime {
  PluginFieldCollection fields_{0, nullptr};

 public:
  IPluginCapability* getCapabilityInterface(
      PluginCapabilityType t) noexcept override {
    switch (t) {
      case PluginCapabilityType::kCORE:
        return static_cast<IPluginV3OneCore*>(this);
      case PluginCapabilityType::kBUILD:
        return static_cast<IPluginV3OneBuild*>(this);
      case PluginCapabilityType::kRUNTIME:
        return static_cast<IPluginV3OneRuntime*>(this);
    }
    return nullptr;
  }
  IPluginV3* clone() noexcept override { return new (std::nothrow) Geometry(); }
  char const* getPluginName() const noexcept override { return kName; }
  char const* getPluginVersion() const noexcept override { return "1"; }
  char const* getPluginNamespace() const noexcept override {
    return kNamespace;
  }
  int32_t getNbOutputs() const noexcept override { return 5; }
  int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t ni,
                          DynamicPluginTensorDesc const*,
                          int32_t no) noexcept override {
    if (ni != 3 || no != 5) return 1;
    PluginTensorDesc desc[3];
    for (int i = 0; i < 3; ++i) desc[i] = in[i].desc;
    return inputShapes(desc, ni) ? 0 : 1;
  }
  int32_t getOutputDataTypes(DataType* out, int32_t no, DataType const* in,
                             int32_t ni) const noexcept override {
    if (ni != 3 || no != 5) return 1;
    for (int i = 0; i < ni; ++i)
      if (in[i] != DataType::kFLOAT) return 1;
    for (int i = 0; i < no; ++i) out[i] = DataType::kFLOAT;
    return 0;
  }
  int32_t getOutputShapes(DimsExprs const*, int32_t ni, DimsExprs const*,
                          int32_t ns, DimsExprs* out, int32_t no,
                          IExprBuilder& b) noexcept override {
    if (ni != 3 || no != 5 || ns != 0) return 1;
    const int ranks[5] = {5, 4, 3, 4, 2};
    const int dims[5][5] = {{6, 1, 2500, 4, 2},
                            {6, 1, 2500, 4, 0},
                            {1, 2500, 256, 0, 0},
                            {1, 2500, 1, 2, 0},
                            {1, 2, 0, 0, 0}};
    for (int i = 0; i < 5; ++i) {
      out[i].nbDims = ranks[i];
      for (int j = 0; j < ranks[i]; ++j) out[i].d[j] = b.constant(dims[i][j]);
    }
    return 0;
  }
  bool supportsFormatCombination(int32_t p, DynamicPluginTensorDesc const* io,
                                 int32_t ni, int32_t no) noexcept override {
    return ni == 3 && no == 5 && p >= 0 && p < 8 &&
           io[p].desc.type == DataType::kFLOAT &&
           io[p].desc.format == TensorFormat::kLINEAR;
  }
  int32_t onShapeChange(PluginTensorDesc const* in, int32_t ni,
                        PluginTensorDesc const*, int32_t no) noexcept override {
    return no == 5 && inputShapes(in, ni) ? 0 : 1;
  }
  int32_t enqueue(PluginTensorDesc const*, PluginTensorDesc const*,
                  void const* const* in, void* const* out, void*,
                  cudaStream_t stream) noexcept override {
    project<<<(6 * kQ * 4 + 255) / 256, 256, 0, stream>>>(
        (const float*)in[0], (float*)out[0], (float*)out[1], (float*)out[3]);
    if (cudaGetLastError() != cudaSuccess) return 1;
    rotate<<<(kQ * 256 + 255) / 256, 256, 0, stream>>>(
        (const float*)in[2], (const float*)in[1], (float*)out[2],
        (float*)out[4]);
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
  }
  IPluginV3* attachToContext(IPluginResourceContext*) noexcept override {
    return clone();
  }
  PluginFieldCollection const* getFieldsToSerialize() noexcept override {
    return &fields_;
  }
};
class GeometryCreator final : public IPluginCreatorV3One {
  PluginFieldCollection fields_{0, nullptr};

 public:
  char const* getPluginName() const noexcept override { return kName; }
  char const* getPluginVersion() const noexcept override { return "1"; }
  char const* getPluginNamespace() const noexcept override {
    return kNamespace;
  }
  PluginFieldCollection const* getFieldNames() noexcept override {
    return &fields_;
  }
  IPluginV3* createPlugin(char const*, PluginFieldCollection const* fc,
                          TensorRTPhase) noexcept override {
    if (fc && fc->nbFields) return nullptr;
    return new (std::nothrow) Geometry();
  }
};
REGISTER_TENSORRT_PLUGIN(GeometryCreator);
}  // namespace bev_plugin
