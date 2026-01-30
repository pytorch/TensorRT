#include "NvInfer.h"
#include "NvInferRuntimeCommon.h"
#include "core/conversion/converters/converters.h"
#include "core/plugins/plugins.h"
#include "core/util/prelude.h"
#include "torch/torch.h"

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

/*
 * Helper functions
 */
#ifndef TRT_MAJOR_RTX
void create_plugin(
    ConversionCtx* ctx,
    const torch::jit::Node* n,
    nvinfer1::ITensor* in,
    const char* name,
    std::vector<int64_t> in_shape,
    std::vector<int64_t> out_shape,
    std::vector<int64_t> out_size,
    std::vector<double> scales,
    std::string mode,
    bool align_corners,
    bool use_scales = false) {
  LOG_WARNING("Interpolation layer will be run through ATen, not TensorRT. Performance may be lower than expected");
  nvinfer1::PluginFieldCollection fc;
  std::vector<nvinfer1::PluginField> f;

  std::vector<int32_t> in_shape_casted(in_shape.begin(), in_shape.end());
  f.emplace_back(
      nvinfer1::PluginField("in_shape", in_shape_casted.data(), nvinfer1::PluginFieldType::kINT32, in_shape.size()));

  std::vector<int32_t> out_shape_casted(out_shape.begin(), out_shape.end());
  f.emplace_back(
      nvinfer1::PluginField("out_shape", out_shape_casted.data(), nvinfer1::PluginFieldType::kINT32, out_shape.size()));

  std::vector<int32_t> out_size_casted(out_size.begin(), out_size.end());
  f.emplace_back(
      nvinfer1::PluginField("out_size", out_size_casted.data(), nvinfer1::PluginFieldType::kINT32, out_size.size()));

  f.emplace_back(nvinfer1::PluginField("scales", scales.data(), nvinfer1::PluginFieldType::kFLOAT64, scales.size()));
  f.emplace_back(nvinfer1::PluginField("mode", &mode, nvinfer1::PluginFieldType::kCHAR, 1));

  int32_t align_corners_casted = static_cast<int32_t>(align_corners);
  f.emplace_back(nvinfer1::PluginField("align_corners", &align_corners_casted, nvinfer1::PluginFieldType::kINT32, 1));

  int32_t use_scales_casted = static_cast<int32_t>(use_scales);
  f.emplace_back(nvinfer1::PluginField("use_scales", &use_scales_casted, nvinfer1::PluginFieldType::kINT32, 1));

  fc.nbFields = f.size();
  fc.fields = f.data();
  auto creator = plugins::impl::TorchTRTPluginRegistry::getInstance().getPluginCreator("Interpolate", "1");
  auto interpolate_plugin = creator->createPlugin(name, &fc, nvinfer1::TensorRTPhase::kBUILD);

  auto resize_layer = ctx->net->addPluginV3(reinterpret_cast<nvinfer1::ITensor* const*>(&in), 1, nullptr, 0, *interpolate_plugin);
  TORCHTRT_CHECK(resize_layer, "Unable to create interpolation plugin from node" << *n);

  resize_layer->setName(util::node_info(n).c_str());

  auto layer_output = ctx->AssociateValueAndTensor(n->outputs()[0], resize_layer->getOutput(0));

  LOG_DEBUG("Output tensor shape: " << layer_output->getDimensions());
}
#endif

void resize_layer_size(
    ConversionCtx* ctx,
    const torch::jit::Node* n,
    nvinfer1::ITensor* in,
    std::vector<int64_t> out_shape,
    std::vector<float> scales,
    nvinfer1::InterpolationMode mode,
    bool align_corners = false) {
  TORCHTRT_CHECK((out_shape.size() > 0) ^ (scales.size() > 0), "only one of out_shape or scales should be defined");
  auto resize_layer = ctx->net->addResize(*in);
  TORCHTRT_CHECK(resize_layer, "Unable to create interpolation (resizing) layer from node" << *n);
  if (out_shape.size() > 0) {
    auto th_dynamic_shape_mask = torch::zeros(out_shape.size(), torch::kInt32);
    auto th_static_shape_mask = torch::zeros(out_shape.size(), torch::kInt32);
    for (size_t idx = 0; idx < out_shape.size(); ++idx) {
      if (out_shape[idx] == -1) {
        th_dynamic_shape_mask[idx] = 1;
      } else {
        th_static_shape_mask[idx] = out_shape[idx];
      }
    }

    auto dynamic_shape_mask = tensor_to_const(ctx, th_dynamic_shape_mask);
    auto static_shape_mask = tensor_to_const(ctx, th_static_shape_mask);
    nvinfer1::ITensor* input_shape = getShapeOutput(ctx, in);
    auto dynamic_shape =
        ctx->net->addElementWise(*input_shape, *dynamic_shape_mask, nvinfer1::ElementWiseOperation::kPROD)
            ->getOutput(0);
    auto target_output_shape =
        ctx->net->addElementWise(*dynamic_shape, *static_shape_mask, nvinfer1::ElementWiseOperation::kSUM)
            ->getOutput(0);
    resize_layer->setInput(1, *target_output_shape);
  } else {
    resize_layer->setScales(scales.data(), scales.size());
    if (align_corners) {
      LOG_WARNING("interpolate with align_corners and scale_factor works differently in TensorRT and PyTorch.");
    }
  }

  resize_layer->setResizeMode(mode);
  resize_layer->setName(util::node_info(n).c_str());

  if (align_corners) {
    resize_layer->setCoordinateTransformation(nvinfer1::ResizeCoordinateTransformation::kALIGN_CORNERS);
  } else {
    if (mode == nvinfer1::InterpolationMode::kLINEAR) {
      resize_layer->setCoordinateTransformation(nvinfer1::ResizeCoordinateTransformation::kHALF_PIXEL);
    } else {
      // kASYMMETRIC is the default transformation in TensorRT
      resize_layer->setCoordinateTransformation(nvinfer1::ResizeCoordinateTransformation::kASYMMETRIC);
    }
  }
  auto layer_output = ctx->AssociateValueAndTensor(n->outputs()[0], resize_layer->getOutput(0));

  LOG_DEBUG("Output tensor shape: " << layer_output->getDimensions());
}

/*
 * Interpolate Converter
 */

auto interpolate_registrations TORCHTRT_UNUSED =
    RegisterNodeConversionPatterns()
        .pattern(
            {"aten::upsample_nearest1d(Tensor self, int[1] output_size, float? scales=None) -> Tensor",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensor();
               auto in_shape = util::toVec(in->getDimensions());

               if (args[1].IValue()->isNone() && args[2].IValue()->isNone()) {
                 TORCHTRT_THROW_ERROR(
                     "Unable to convert node: " << util::node_info(n)
                                                << "\nOne of output_size or scales should be defined");
               } else if (!args[2].IValue()->isNone()) {
                 // Case 1: user uses scales
                 float scale = args[2].IValue()->toDouble();
                 std::vector<float> padded_scales(in_shape.size(), 1);
                 padded_scales[padded_scales.size() - 1] = scale;
                 resize_layer_size(ctx, n, in, {}, padded_scales, nvinfer1::InterpolationMode::kNEAREST);
               } else {
                 // Case 2: user uses output size
                 auto out_size = util::toVec(util::toDims(args[1].unwrapToIntList()));
                 TORCHTRT_ASSERT(
                     out_size.size() == 1, "aten::upsample_nearest1d input Tensor and output size dimension mismatch");

                 auto out_shape = in_shape;
                 std::copy(out_size.begin(), out_size.end(), out_shape.begin() + (in_shape.size() - out_size.size()));
                 resize_layer_size(ctx, n, in, out_shape, {}, nvinfer1::InterpolationMode::kNEAREST);
               }

               return true;
             }})
        .pattern(
            {"aten::upsample_nearest1d.vec(Tensor input, int[]? output_size, float[]? scale_factors) -> Tensor",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensor();
               auto in_shape = util::toVec(in->getDimensions());

               if (args[1].IValue()->isNone() && args[2].IValue()->isNone()) {
                 TORCHTRT_THROW_ERROR(
                     "Unable to convert node: " << util::node_info(n)
                                                << "\nOne of output_size or scale_factors should be defined");
               } else if (!args[2].IValue()->isNone()) {
                 // Case 1: user uses scales
                 auto scale_factors = args[2].unwrapToDoubleList();
                 TORCHTRT_ASSERT(scale_factors.size() == 1, "Number of scale factors should match the input size");
                 float scale = scale_factors[0];
                 std::vector<float> padded_scales(in_shape.size(), 1);
                 padded_scales[padded_scales.size() - 1] = scale;
                 resize_layer_size(ctx, n, in, {}, padded_scales, nvinfer1::InterpolationMode::kNEAREST);
               } else {
                 // Case 2: user uses output size
                 auto out_size = util::toVec(util::toDims(args[1].unwrapToIntList()));
                 TORCHTRT_ASSERT(
                     out_size.size() == 1, "aten::upsample_nearest1d input Tensor and output size dimension mismatch");

                 auto out_shape = in_shape;
                 std::copy(out_size.begin(), out_size.end(), out_shape.begin() + (in_shape.size() - out_size.size()));
                 resize_layer_size(ctx, n, in, out_shape, {}, nvinfer1::InterpolationMode::kNEAREST);
               }

               return true;
             }})
        .pattern(
            {"aten::upsample_nearest2d(Tensor self, int[2] output_size, float? scales_h=None, float? scales_w=None) -> Tensor",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensor();
               auto in_shape = util::toVec(in->getDimensions());

               if (args[1].IValue()->isNone() && (args[2].IValue()->isNone() || args[3].IValue()->isNone())) {
                 TORCHTRT_THROW_ERROR(
                     "Unable to convert node: " << util::node_info(n)
                                                << "\nOne of output_size or scales should be defined");
               } else if (!args[2].IValue()->isNone() && !args[3].IValue()->isNone()) {
                 // Case 1: user uses scales
                 float scale_h = args[2].IValue()->toDouble();
                 float scale_w = args[3].IValue()->toDouble();
                 std::vector<float> padded_scales(in_shape.size(), 1);
                 padded_scales[padded_scales.size() - 2] = scale_h;
                 padded_scales[padded_scales.size() - 1] = scale_w;
                 resize_layer_size(ctx, n, in, {}, padded_scales, nvinfer1::InterpolationMode::kNEAREST);
               } else {
                 // Case 2: user uses output size
                 auto out_size = util::toVec(util::toDims(args[1].unwrapToIntList()));
                 TORCHTRT_ASSERT(
                     out_size.size() == 2, "aten::upsample_nearest2d input Tensor and output size dimension mismatch");

                 auto out_shape = in_shape;
                 std::copy(out_size.begin(), out_size.end(), out_shape.begin() + (in_shape.size() - out_size.size()));
                 resize_layer_size(ctx, n, in, out_shape, {}, nvinfer1::InterpolationMode::kNEAREST);
               }

               return true;
             }})
        .pattern(
            {"aten::upsample_nearest2d.vec(Tensor input, int[]? output_size, float[]? scale_factors) -> Tensor",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensor();
               auto in_shape = util::toVec(in->getDimensions());

               if (args[1].IValue()->isNone() && args[2].IValue()->isNone()) {
                 TORCHTRT_THROW_ERROR(
                     "Unable to convert node: " << util::node_info(n)
                                                << "\nOne of output_size or scale_factors should be defined");
               } else if (!args[2].IValue()->isNone()) {
                 // Case 1: user uses scales
                 auto scale_factors = args[2].unwrapToDoubleList();
                 TORCHTRT_ASSERT(scale_factors.size() == 2, "Number of scale factors should match the input size");
                 float scale_h = scale_factors[0];
                 float scale_w = scale_factors[1];
                 std::vector<float> padded_scales(in_shape.size(), 1);
                 padded_scales[padded_scales.size() - 2] = scale_h;
                 padded_scales[padded_scales.size() - 1] = scale_w;
                 resize_layer_size(ctx, n, in, {}, padded_scales, nvinfer1::InterpolationMode::kNEAREST);
               } else {
                 // Case 2: user uses output size
                 auto out_size = util::toVec(util::toDims(args[1].unwrapToIntList()));
                 TORCHTRT_ASSERT(
                     out_size.size() == 2, "aten::upsample_nearest2d input Tensor and output size dimension mismatch");

                 auto out_shape = in_shape;
                 std::copy(out_size.begin(), out_size.end(), out_shape.begin() + (in_shape.size() - out_size.size()));
                 resize_layer_size(ctx, n, in, out_shape, {}, nvinfer1::InterpolationMode::kNEAREST);
               }

               return true;
             }})
        .pattern(
            {"aten::upsample_nearest3d(Tensor self, int[3] output_size, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensor();
               auto in_shape = util::toVec(in->getDimensions());

               if (args[1].IValue()->isNone() &&
                   (args[2].IValue()->isNone() || args[3].IValue()->isNone() || args[4].IValue()->isNone())) {
                 TORCHTRT_THROW_ERROR(
                     "Unable to convert node: " << util::node_info(n)
                                                << "\nOne of output_size or scales should be defined");
               } else if (!args[2].IValue()->isNone() && !args[3].IValue()->isNone() && !args[4].IValue()->isNone()) {
                 // Case 1: user uses scales
                 float scale_d = args[2].IValue()->toDouble();
                 float scale_h = args[3].IValue()->toDouble();
                 float scale_w = args[4].IValue()->toDouble();
                 std::vector<float> padded_scales(in_shape.size(), 1);
                 padded_scales[padded_scales.size() - 3] = scale_d;
                 padded_scales[padded_scales.size() - 2] = scale_h;
                 padded_scales[padded_scales.size() - 1] = scale_w;
                 resize_layer_size(ctx, n, in, {}, padded_scales, nvinfer1::InterpolationMode::kNEAREST);
               } else {
                 // Case 2: user uses output size
                 auto out_size = util::toVec(util::toDims(args[1].unwrapToIntList()));
                 TORCHTRT_ASSERT(
                     out_size.size() == 3, "aten::upsample_nearest3d input Tensor and output size dimension mismatch");

                 auto out_shape = in_shape;
                 std::copy(out_size.begin(), out_size.end(), out_shape.begin() + (in_shape.size() - out_size.size()));
                 resize_layer_size(ctx, n, in, out_shape, {}, nvinfer1::InterpolationMode::kNEAREST);
               }

               return true;
             }})
        .pattern(
            {"aten::upsample_nearest3d.vec(Tensor input, int[]? output_size, float[]? scale_factors) -> Tensor", // FIX
                                                                                                                 // THIS
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensor();
               auto in_shape = util::toVec(in->getDimensions());

               if (args[1].IValue()->isNone() && args[2].IValue()->isNone()) {
                 TORCHTRT_THROW_ERROR(
                     "Unable to convert node: " << util::node_info(n)
                                                << "\nOne of output_size or scale_factors should be defined");
               } else if (!args[2].IValue()->isNone()) {
                 // Case 1: user uses scales
                 auto scale_factors = args[2].unwrapToDoubleList();
                 TORCHTRT_ASSERT(scale_factors.size() == 3, "Number of scale factors should match the input size");
                 float scale_d = scale_factors[0];
                 float scale_h = scale_factors[1];
                 float scale_w = scale_factors[2];
                 std::vector<float> padded_scales(in_shape.size(), 1);
                 padded_scales[padded_scales.size() - 3] = scale_d;
                 padded_scales[padded_scales.size() - 2] = scale_h;
                 padded_scales[padded_scales.size() - 1] = scale_w;
                 resize_layer_size(ctx, n, in, {}, padded_scales, nvinfer1::InterpolationMode::kNEAREST);
               } else {
                 // Case 2: user uses output size
                 auto out_size = util::toVec(util::toDims(args[1].unwrapToIntList()));
                 TORCHTRT_ASSERT(
                     out_size.size() == 3, "aten::upsample_nearest3d input Tensor and output size dimension mismatch");

                 auto out_shape = in_shape;
                 std::copy(out_size.begin(), out_size.end(), out_shape.begin() + (in_shape.size() - out_size.size()));
                 resize_layer_size(ctx, n, in, out_shape, {}, nvinfer1::InterpolationMode::kNEAREST);
               }

               return true;
             }})
        .pattern(
            {"aten::upsample_linear1d(Tensor self, int[1] output_size, bool align_corners, float? scales=None) -> Tensor",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensor();
               auto in_shape = util::toVec(in->getDimensions());
               bool align_corners = args[2].unwrapToBool();

               if (args[1].IValue()->isNone() && args[3].IValue()->isNone()) {
                 TORCHTRT_THROW_ERROR(
                     "Unable to convert node: " << util::node_info(n)
                                                << "\nOne of output_size or scales should be defined");
               } else if (!args[3].IValue()->isNone()) {
                 // Case 1: user uses scales
                 float scale = args[3].IValue()->toDouble();
                 std::vector<float> padded_scales(in_shape.size(), 1);
                 padded_scales[padded_scales.size() - 1] = scale;
                 resize_layer_size(ctx, n, in, {}, padded_scales, nvinfer1::InterpolationMode::kLINEAR, align_corners);
               } else {
                 // Case 2: user uses output size
                 auto out_size = util::toVec(util::toDims(args[1].unwrapToIntList()));
                 TORCHTRT_ASSERT(
                     out_size.size() == 1, "aten::upsample_linear1d input Tensor and output size dimension mismatch");

                 auto out_shape = in_shape;
                 std::copy(out_size.begin(), out_size.end(), out_shape.begin() + (in_shape.size() - out_size.size()));
                 resize_layer_size(ctx, n, in, out_shape, {}, nvinfer1::InterpolationMode::kLINEAR, align_corners);
               }

               return true;
             }})
        .pattern(
            {"aten::upsample_linear1d.vec(Tensor input, int[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensor();
               auto in_shape = util::toVec(in->getDimensions());
               bool align_corners = args[2].unwrapToBool();

               if (args[1].IValue()->isNone() && args[3].IValue()->isNone()) {
                 TORCHTRT_THROW_ERROR(
                     "Unable to convert node: " << util::node_info(n)
                                                << "\nOne of output_size or scale_factors should be defined");
               } else if (!args[3].IValue()->isNone()) {
                 // Case 1: user uses scales
                 auto scale_factors = args[3].unwrapToDoubleList();
                 TORCHTRT_ASSERT(scale_factors.size() == 1, "Number of scale factors should match the input size");
                 float scale = scale_factors[0];
                 std::vector<float> padded_scales(in_shape.size(), 1);
                 padded_scales[padded_scales.size() - 1] = scale;
                 resize_layer_size(ctx, n, in, {}, padded_scales, nvinfer1::InterpolationMode::kLINEAR, align_corners);
               } else {
                 // Case 2: user uses output size
                 auto out_size = util::toVec(util::toDims(args[1].unwrapToIntList()));
                 TORCHTRT_ASSERT(
                     out_size.size() == 1, "aten::upsample_linear1d input Tensor and output size dimension mismatch");

                 auto out_shape = in_shape;
                 std::copy(out_size.begin(), out_size.end(), out_shape.begin() + (in_shape.size() - out_size.size()));
                 resize_layer_size(ctx, n, in, out_shape, {}, nvinfer1::InterpolationMode::kLINEAR, align_corners);
               }

               return true;
             }})
        .pattern(
            {"aten::upsample_bilinear2d(Tensor self, int[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensor();
               auto in_shape = util::toVec(in->getDimensions());
               bool align_corners = args[2].unwrapToBool();

               if (args[1].IValue()->isNone() && (args[3].IValue()->isNone() || args[4].IValue()->isNone())) {
                 TORCHTRT_THROW_ERROR(
                     "Unable to convert node: " << util::node_info(n)
                                                << "\nOne of output_size or scales should be defined");
               } else if (!args[3].IValue()->isNone() && !args[4].IValue()->isNone()) {
                 // Case 1: user uses scales
                 float scale_h = args[3].IValue()->toDouble();
                 float scale_w = args[4].IValue()->toDouble();
                 std::vector<float> padded_scales(in_shape.size(), 1);
                 padded_scales[padded_scales.size() - 2] = scale_h;
                 padded_scales[padded_scales.size() - 1] = scale_w;
                 resize_layer_size(ctx, n, in, {}, padded_scales, nvinfer1::InterpolationMode::kLINEAR, align_corners);
               } else {
                 // Case 2: user uses output size
                 auto out_size = util::toVec(util::toDims(args[1].unwrapToIntList()));

                 TORCHTRT_ASSERT(
                     out_size.size() == 2, "aten::upsample_bilinear2d input Tensor and output size dimension mismatch");

                 auto out_shape = in_shape;
                 std::copy(out_size.begin(), out_size.end(), out_shape.begin() + (in_shape.size() - out_size.size()));
                 resize_layer_size(ctx, n, in, out_shape, {}, nvinfer1::InterpolationMode::kLINEAR, align_corners);
               }

               return true;
             }})
        .pattern(
            {"aten::upsample_bilinear2d.vec(Tensor input, int[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensor();
               auto in_shape = util::toVec(in->getDimensions());
               bool align_corners = args[2].unwrapToBool();

               if (args[1].IValue()->isNone() && args[3].IValue()->isNone()) {
                 TORCHTRT_THROW_ERROR(
                     "Unable to convert node: " << util::node_info(n)
                                                << "\nOne of output_size or scale_factors should be defined");
               } else if (!args[3].IValue()->isNone()) {
                 // Case 1: user uses scales
                 auto scale_factors = args[3].unwrapToDoubleList();
                 TORCHTRT_ASSERT(scale_factors.size() == 2, "Number of scale factors should match the input size");
                 float scale_h = scale_factors[0];
                 float scale_w = scale_factors[1];
                 std::vector<float> padded_scales(in_shape.size(), 1);
                 padded_scales[padded_scales.size() - 2] = scale_h;
                 padded_scales[padded_scales.size() - 1] = scale_w;
                 resize_layer_size(ctx, n, in, {}, padded_scales, nvinfer1::InterpolationMode::kLINEAR, align_corners);
               } else {
                 // Case 2: user uses output size
                 auto out_size = util::toVec(util::toDims(args[1].unwrapToIntList()));

                 TORCHTRT_ASSERT(
                     out_size.size() == 2, "aten::upsample_bilinear2d input Tensor and output size dimension mismatch");

                 auto out_shape = in_shape;
                 std::copy(out_size.begin(), out_size.end(), out_shape.begin() + (in_shape.size() - out_size.size()));
                 resize_layer_size(ctx, n, in, out_shape, {}, nvinfer1::InterpolationMode::kLINEAR, align_corners);
               }

               return true;
             }})
        .pattern(
            {"aten::upsample_trilinear3d(Tensor self, int[3] output_size, bool align_corners, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensor();
               auto in_shape = util::toVec(in->getDimensions());
               bool align_corners = args[2].unwrapToBool();

               if (args[1].IValue()->isNone() &&
                   (args[3].IValue()->isNone() || args[4].IValue()->isNone() || args[5].IValue()->isNone())) {
                 TORCHTRT_THROW_ERROR(
                     "Unable to convert node: " << util::node_info(n) << "\nOne of size or scales should be defined");
               } else if (!args[3].IValue()->isNone() && !args[4].IValue()->isNone() && !args[5].IValue()->isNone()) {
                 // Case 1: user uses scales
                 float scale_d = args[3].IValue()->toDouble();
                 float scale_h = args[4].IValue()->toDouble();
                 float scale_w = args[5].IValue()->toDouble();
                 std::vector<float> padded_scales(in_shape.size(), 1);
                 padded_scales[padded_scales.size() - 3] = scale_d;
                 padded_scales[padded_scales.size() - 2] = scale_h;
                 padded_scales[padded_scales.size() - 1] = scale_w;
                 resize_layer_size(ctx, n, in, {}, padded_scales, nvinfer1::InterpolationMode::kLINEAR, align_corners);
               } else {
                 // Case 2: user uses output size
                 auto out_size = util::toVec(util::toDims(args[1].unwrapToIntList()));
                 TORCHTRT_ASSERT(
                     out_size.size() == 3,
                     "aten::upsample_trilinear3d input Tensor and output size dimension mismatch");

                 auto out_shape = in_shape;
                 std::copy(out_size.begin(), out_size.end(), out_shape.begin() + (in_shape.size() - out_size.size()));
                 resize_layer_size(ctx, n, in, out_shape, {}, nvinfer1::InterpolationMode::kLINEAR, align_corners);
               }

               return true;
             }})
        .pattern(
            {"aten::upsample_trilinear3d.vec(Tensor input, int[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensor();
               auto in_shape = util::toVec(in->getDimensions());
               bool align_corners = args[2].unwrapToBool();

               if (args[1].IValue()->isNone() && args[3].IValue()->isNone()) {
                 TORCHTRT_THROW_ERROR(
                     "Unable to convert node: " << util::node_info(n)
                                                << "\nOne of size or scale_factors should be defined");
               } else if (!args[3].IValue()->isNone()) {
                 // Case 1: user uses scales
                 auto scale_factors = args[3].unwrapToDoubleList();
                 TORCHTRT_ASSERT(scale_factors.size() == 3, "Number of scale factors should match the input size");
                 float scale_d = scale_factors[0];
                 float scale_h = scale_factors[1];
                 float scale_w = scale_factors[2];
                 std::vector<float> padded_scales(in_shape.size(), 1);
                 padded_scales[padded_scales.size() - 3] = scale_d;
                 padded_scales[padded_scales.size() - 2] = scale_h;
                 padded_scales[padded_scales.size() - 1] = scale_w;
                 resize_layer_size(ctx, n, in, {}, padded_scales, nvinfer1::InterpolationMode::kLINEAR, align_corners);
               } else {
                 // Case 2: user uses output size
                 auto out_size = util::toVec(util::toDims(args[1].unwrapToIntList()));
                 TORCHTRT_ASSERT(
                     out_size.size() == 3,
                     "aten::upsample_trilinear3d input Tensor and output size dimension mismatch");

                 auto out_shape = in_shape;
                 std::copy(out_size.begin(), out_size.end(), out_shape.begin() + (in_shape.size() - out_size.size()));
                 resize_layer_size(ctx, n, in, out_shape, {}, nvinfer1::InterpolationMode::kLINEAR, align_corners);
               }

               return true;
             }})
        .pattern(
            {"aten::grid_sampler(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensorOrFreeze(ctx);
               auto grid = args[1].ITensorOrFreeze(ctx);
               auto interpolation_mode = args[2].unwrapToInt();
               auto padding_mode = args[3].unwrapToInt();
               auto align_corners = args[4].unwrapToBool();

               static const auto sample_map = std::map<int, nvinfer1::SampleMode>{
                   {0, nvinfer1::SampleMode::kFILL},
                   {1, nvinfer1::SampleMode::kCLAMP},
                   {2, nvinfer1::SampleMode::kREFLECT}};

               static const auto interpolation_map = std::map<int, nvinfer1::InterpolationMode>{
                   {0, nvinfer1::InterpolationMode::kLINEAR},
                   {1, nvinfer1::InterpolationMode::kNEAREST},
                   {2, nvinfer1::InterpolationMode::kCUBIC}};

               auto grid_sample_layer = ctx->net->addGridSample(*in, *grid);
               TORCHTRT_CHECK(
                   grid_sample_layer, "Unable to create grid_sample layer from node: " << util::node_info(n));

               grid_sample_layer->setAlignCorners(align_corners);
               grid_sample_layer->setSampleMode(sample_map.at(padding_mode));
               grid_sample_layer->setInterpolationMode(interpolation_map.at(interpolation_mode));

               auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], grid_sample_layer->getOutput(0));
               LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
               return true;
             }});

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
