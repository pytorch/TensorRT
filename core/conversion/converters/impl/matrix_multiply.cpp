#include "core/conversion/converters/converter_util.h"
#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

auto mm_registrations TORCHTRT_UNUSED =
    RegisterNodeConversionPatterns()
        .pattern(
            {"aten::matmul(Tensor self, Tensor other) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto self = args[0].ITensorOrFreeze(ctx);
               auto other = args[1].ITensorOrFreeze(ctx);
               // Ensure self and other tensors have same nbDims by expanding the dimensions (from 0 axis) if
               // necessary.
               if (self->getDimensions().nbDims < other->getDimensions().nbDims) {
                 self = addPadding(ctx, n, self, other->getDimensions().nbDims, false, false);
               } else {
                 other = addPadding(ctx, n, other, self->getDimensions().nbDims, false, false);
               }

               auto mm_layer = ctx->net->addMatrixMultiply(
                   *self, nvinfer1::MatrixOperation::kNONE, *other, nvinfer1::MatrixOperation::kNONE);

               TORCHTRT_CHECK(mm_layer, "Unable to create matrix multiplication node: " << *n);
               mm_layer->setName(util::node_info(n).c_str());
               auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], mm_layer->getOutput(0));

               LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::bmm(Tensor self, Tensor mat2) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto self = args[0].ITensorOrFreeze(ctx);
               nvinfer1::Dims selfDims = self->getDimensions();
               auto mat2 = args[1].ITensorOrFreeze(ctx);
               nvinfer1::Dims mat2Dims = mat2->getDimensions();

               // check dimensions
               TORCHTRT_CHECK(
                   selfDims.nbDims == 3,
                   "Expected 3-dimensional tensor, but got "
                       << selfDims.nbDims
                       << "-dimensional tensor for argument #1 'batch1' (while checking arguments for bmm)");
               TORCHTRT_CHECK(
                   mat2Dims.nbDims == 3,
                   "Expected 3-dimensional tensor, but got "
                       << mat2Dims.nbDims
                       << "-dimensional tensor for argument #2 'batch2' (while checking arguments for bmm)");

               // Self and mat2 should have same size at dimension 0
               TORCHTRT_CHECK(
                   selfDims.d[0] == mat2Dims.d[0],
                   "Expected tensor to have size " << selfDims.d[0] << " at dimension 0, but got size " << mat2Dims.d[0]
                                                   << " for argument #2 'batch2' (while checking arguments for bmm)");
               // The size of mat2 at dimension 1 should be the same as that of self at dimension 2.
               TORCHTRT_CHECK(
                   selfDims.d[2] == mat2Dims.d[1],
                   "Expected tensor to have size " << selfDims.d[2] << " at dimension 1, but got size " << mat2Dims.d[1]
                                                   << " for argument #2 'batch2' (while checking arguments for bmm)");

               auto mm_layer = ctx->net->addMatrixMultiply(
                   *self, nvinfer1::MatrixOperation::kNONE, *mat2, nvinfer1::MatrixOperation::kNONE);
               TORCHTRT_CHECK(mm_layer, "Unable to create matrix multiplication node: " << *n);

               mm_layer->setName(util::node_info(n).c_str());
               auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], mm_layer->getOutput(0));

               LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto self = args[0].ITensorOrFreeze(ctx);
               auto bat1 = args[1].ITensorOrFreeze(ctx);
               auto bat2 = args[2].ITensorOrFreeze(ctx);
               nvinfer1::Dims batch1Dims = bat1->getDimensions();
               nvinfer1::Dims batch2Dims = bat2->getDimensions();

               // check dimensions
               TORCHTRT_CHECK(
                   batch1Dims.nbDims == 3,
                   "Expected 3-dimensional tensor, but got "
                       << batch1Dims.nbDims
                       << "-dimensional tensor for argument 'batch1' (while checking arguments for baddbmm)");
               TORCHTRT_CHECK(
                   batch2Dims.nbDims == 3,
                   "Expected 3-dimensional tensor, but got "
                       << batch2Dims.nbDims
                       << "-dimensional tensor for argument 'batch2' (while checking arguments for baddbmm)");
               TORCHTRT_CHECK(
                   batch1Dims.d[0] == batch2Dims.d[0],
                   "Expected tensor to have size " << batch1Dims.d[0] << " at dimension 0, but got size "
                                                   << batch2Dims.d[0]
                                                   << " for argument 'batch2' (while checking arguments for baddbmm)");
               TORCHTRT_CHECK(
                   batch1Dims.d[2] == batch2Dims.d[1],
                   "Expected tensor to have size " << batch1Dims.d[2] << " at dimension 1, but got size "
                                                   << batch2Dims.d[1]
                                                   << " for argument 'batch2' (while checking arguments for baddbmm)");

               auto mm_layer = ctx->net->addMatrixMultiply(
                   *bat1, nvinfer1::MatrixOperation::kNONE, *bat2, nvinfer1::MatrixOperation::kNONE);
               TORCHTRT_CHECK(mm_layer, "Unable to create matrix multiplication for node: " << *n);
               mm_layer->setName((util::node_info(n) + "_matmul").c_str());

               auto mm_out = mm_layer->getOutput(0);

               auto alpha = args[4].unwrapToScalar();
               if (alpha.to<float>() != 1.) {
                 auto alpha_tensor = scalar_to_tensor(ctx, alpha);
                 auto alpha_layer = add_elementwise(
                     ctx,
                     nvinfer1::ElementWiseOperation::kPROD,
                     mm_out,
                     alpha_tensor,
                     util::node_info(n) + std::string("_alpha_mul"));
                 TORCHTRT_CHECK(alpha_layer, "Unable to create alpha_mul layer from node: " << *n);
                 mm_out = alpha_layer->getOutput(0);
               }

               auto beta = args[3].unwrapToScalar();
               // If beta is 0, then input will be ignored, and nan and inf in it will not be propagated.
               if (beta.to<float>() != 0.) {
                 if (beta.to<float>() != 1.) {
                   auto beta_tensor = scalar_to_tensor(ctx, beta);
                   auto beta_layer = add_elementwise(
                       ctx,
                       nvinfer1::ElementWiseOperation::kPROD,
                       self,
                       beta_tensor,
                       util::node_info(n) + std::string("_beta_mul"));
                   TORCHTRT_CHECK(beta_layer, "Unable to create beta_mul layer from node: " << *n);
                   self = beta_layer->getOutput(0);
                 }
                 auto self_add_layer = add_elementwise(
                     ctx,
                     nvinfer1::ElementWiseOperation::kSUM,
                     self,
                     mm_out,
                     util::node_info(n) + std::string("_self_add"));
                 TORCHTRT_CHECK(self_add_layer, "Unable to create self_add layer from node: " << *n);
                 mm_out = self_add_layer->getOutput(0);
               }

               auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], mm_out);
               LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
               return true;
             }});
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
