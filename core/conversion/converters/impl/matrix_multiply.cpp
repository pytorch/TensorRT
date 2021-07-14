#include <torch/torch.h>
#include "core/conversion/converters/converter_util.h"
#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

auto mm_registrations TRTORCH_UNUSED =
    RegisterNodeConversionPatterns()
        .pattern({"aten::matmul(Tensor self, Tensor other) -> (Tensor)",
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
                    TRTORCH_CHECK(mm_layer, "Unable to create matrix multiplication node: " << *n);
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
               TRTORCH_CHECK(
                   selfDims.nbDims == 3,
                   "Expected 3-dimensional tensor, but got "
                       << selfDims.nbDims
                       << "-dimensional tensor for argument #1 'batch1' (while checking arguments for bmm)");
               TRTORCH_CHECK(
                   mat2Dims.nbDims == 3,
                   "Expected 3-dimensional tensor, but got "
                       << mat2Dims.nbDims
                       << "-dimensional tensor for argument #2 'batch2' (while checking arguments for bmm)");

               // Self and mat2 should have same size at dimension 0
               TRTORCH_CHECK(
                   selfDims.d[0] == mat2Dims.d[0],
                   "Expected tensor to have size " << selfDims.d[0] << " at dimension 0, but got size " << mat2Dims.d[0]
                                                   << " for argument #2 'batch2' (while checking arguments for bmm)");
               // The size of mat2 at dimension 1 should be the same as that of self at dimension 2.
               TRTORCH_CHECK(
                   selfDims.d[2] == mat2Dims.d[1],
                   "Expected tensor to have size " << selfDims.d[2] << " at dimension 1, but got size " << mat2Dims.d[1]
                                                   << " for argument #2 'batch2' (while checking arguments for bmm)");

               auto mm_layer = ctx->net->addMatrixMultiply(
                   *self, nvinfer1::MatrixOperation::kNONE, *mat2, nvinfer1::MatrixOperation::kNONE);
               TRTORCH_CHECK(mm_layer, "Unable to create matrix multiplication node: " << *n);

               mm_layer->setName(util::node_info(n).c_str());
               auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], mm_layer->getOutput(0));

               LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto self = args[0].ITensorOrFreeze(ctx);
               auto mat1 = args[1].ITensorOrFreeze(ctx);
               auto mat2 = args[2].ITensorOrFreeze(ctx);
               auto beta = args[4].unwrapToScalar().to<float>();
               auto betaTensor = tensor_to_const(ctx, torch::tensor({beta}));
               auto alpha = args[5].unwrapToScalar().to<float>();
               auto alphaTensor = tensor_to_const(ctx, torch::tensor({alpha}));

               // Ensure self and other tensors have same nbDims by expanding the dimensions (from 0 axis) if
               // necessary.
               if (mat1->getDimensions().nbDims < mat2->getDimensions().nbDims) {
                 mat1 = addPadding(ctx, n, mat1, mat2->getDimensions().nbDims, false, false);
               } else {
                 mat2 = addPadding(ctx, n, mat2, mat1->getDimensions().nbDims, false, false);
               }

               auto mat2_dims = mat2->getDimensions();
               nvinfer1::Dims transposed_mat2_dims;
               for (int i = mat2_dims.nbDims - 1; i >= 0; i--) {
                 transposed_mat2_dims.d[i] = mat2_dims.d[mat2_dims.nbDims - 1 - i];
               }
               auto shuffle_layer = ctx->net->addShuffle(*mat2);
               shuffle_layer->setReshapeDimensions(transposed_mat2_dims);
               mat2 = shuffle_layer->getOutput(0);

               auto mm_layer = ctx->net->addMatrixMultiply(
                   *mat1, nvinfer1::MatrixOperation::kNONE, *mat2, nvinfer1::MatrixOperation::kNONE);
               TRTORCH_CHECK(mm_layer, "Unable to create matrix multiplication layer in node: " << *n);
               auto mm_scale_layer = add_elementwise(
                   ctx,
                   nvinfer1::ElementWiseOperation::kPROD,
                   mm_layer->getOutput(0),
                   alphaTensor,
                   util::node_info(n) + "_alphaScale");
               TRTORCH_CHECK(mm_scale_layer, "Unable to create alpha scaling layer in node: " << *n);
               auto beta_scale_layer = add_elementwise(
                   ctx, nvinfer1::ElementWiseOperation::kPROD, self, betaTensor, util::node_info(n) + "_betaScale");
               TRTORCH_CHECK(beta_scale_layer, "Unable to create beta scaling layer in node: " << *n);
               auto add_mm_layer = add_elementwise(
                   ctx,
                   nvinfer1::ElementWiseOperation::kSUM,
                   beta_scale_layer->getOutput(0),
                   mm_scale_layer->getOutput(0),
                   util::node_info(n));
               TRTORCH_CHECK(add_mm_layer, "Unable to create addmm layer in node: " << *n);

               auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], add_mm_layer->getOutput(0));

               LOG_DEBUG("[AddMM layer] Output tensor shape: " << out_tensor->getDimensions());
               return true;
             }});
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
