#include <ATen/ATen.h>
#include "core/conversion/converters/converter_util.h"
#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"
#include "torch/torch.h"

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {


nvinfer1::Dims get_update_dim(const int indicesNumber, nvinfer1::Dims baseDims, nvinfer1::Dims indicesDims) {
  std::vector<int> updateDims;
  for (int i = 0; i < indicesDims.nbDims; i++) {
    updateDims.push_back(indicesDims.d[i]);
  }
  for (int i = indicesNumber; i < baseDims.nbDims; i++) {
    updateDims.push_back(baseDims.d[i]);
  }
  nvinfer1::Dims result;
  result.nbDims = updateDims.size();
  for (size_t i = 0; i < updateDims.size(); i++) {
    result.d[i] = updateDims[i];
  }
  return result;
}


auto index_put_registrations TORCHTRT_UNUSED =
    RegisterNodeConversionPatterns()
        .pattern(
            {"aten::index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensorOrFreeze(ctx);
               auto ts = args[1].IValue()->toListRef();
               auto update = args[2].ITensorOrFreeze(ctx);
               auto accumulate = args[3].unwrapToBool();

               std::vector<nvinfer1::ITensor*> indices;
               for (auto t : ts) {
                 if (t.isTensor()) {
                   auto torch_tensor = t.toTensor();
                   indices.push_back(tensor_to_const(ctx, torch_tensor));
                 } else {
                   auto cont = t.toCustomClass<TensorContainer>();
                   indices.push_back(cont->tensor());
                 }
               }

               std::vector<nvinfer1::ITensor*> indices_casted;
               nvinfer1::Dims final_dim = indices[0]->getDimensions();
               for (size_t i = 0; i < indices.size(); i ++) {
                auto cur_indice = indices[i];
                TORCHTRT_CHECK(cur_indice->getType() != nvinfer1::DataType::kBOOL, "index data type should not be bool");

                if (util::broadcastable(final_dim, cur_indice->getDimensions(), /*multidirectional=*/true)) {
                  final_dim = util::broadcastDim(final_dim, cur_indice->getDimensions());
                } else {
                  TORCHTRT_CHECK( true,
                   "indices or update are not broadcastable");
                }
                // Set datatype for indices tensor to INT32
                auto identity = ctx->net->addIdentity(*cur_indice);
                identity->setOutputType(0, nvinfer1::DataType::kINT32);
                auto cur_casted = identity->getOutput(0);
                indices_casted.push_back(cur_casted);
               }

               auto dim_vector = util::toVec(final_dim);
               
               at::Tensor ones = torch::ones(dim_vector, torch::kInt32);
               auto weights = Weights(ctx, ones);
               auto const_layer = ctx->net->addConstant(weights.shape, weights.data);
               TORCHTRT_CHECK(const_layer, "Unable to create constant layer from node: " << *n);
               auto const_out = const_layer->getOutput(0);

               std::vector<nvinfer1::ITensor*> indices_expanded;

               nvinfer1::Dims unsqueenze_dims;
               unsqueenze_dims.nbDims = final_dim.nbDims + 1;
               for (int j = 0; j < final_dim.nbDims; j++) {
                unsqueenze_dims.d[j] = final_dim.d[j];
               }
               unsqueenze_dims.d[final_dim.nbDims] = 1;

               for (size_t i = 0; i < indices_casted.size(); i++) {
                //expand to broadcasted size
                auto expand_layer = add_elementwise( ctx, nvinfer1::ElementWiseOperation::kPROD, indices_casted[i], const_out, "expand"+ std::to_string(i));
                TORCHTRT_CHECK(expand_layer, "Unable to create element prod layer from node: " << *n << i);
                auto expand_out = expand_layer->getOutput(0);
                LOG_DEBUG("expanded indice shape: " << expand_out->getDimensions());

                // unsqeenze last dimension
                auto shuffle_layer = ctx->net->addShuffle(*expand_out);
                TORCHTRT_CHECK(shuffle_layer, "Unable to create shuffle layer from node: " << *n << i);
                shuffle_layer->setReshapeDimensions(unsqueenze_dims);
                auto shuffle_out = ctx->AssociateValueAndTensor(n->outputs()[0], shuffle_layer->getOutput(0));
                indices_expanded.push_back(shuffle_out);
               }

               auto concat_layer = ctx->net->addConcatenation(indices_expanded.data(), indices_expanded.size());
               concat_layer->setAxis(final_dim.nbDims);
               auto concate_indices = concat_layer->getOutput(0);
               LOG_DEBUG("concate indices shape: " << concate_indices->getDimensions());
               
               auto updateDim = get_update_dim(indices.size(), in->getDimensions(), final_dim);
               LOG_DEBUG("updated dim is: " << updateDim);

               auto expand_update = add_expand_layer(ctx,  update, updateDim);
               LOG_DEBUG("expand update shape: " << expand_update->getDimensions());

               if (!accumulate) {
                auto scatterLayer = ctx->net->addScatter(*in, *concate_indices, *expand_update, nvinfer1::ScatterMode::kND);
                TORCHTRT_CHECK(scatterLayer, "Unable to create scatter nd layer from node: " << *n);
                auto out = ctx->AssociateValueAndTensor(n->outputs()[0], scatterLayer->getOutput(0));
                LOG_DEBUG("Output tensor shape: " << out->getDimensions());
               } else {
                auto gather_layer = ctx->net->addGather(*in, *concate_indices, 0);
                TORCHTRT_CHECK(gather_layer, "Unable to create gather nd layer from node: " << *n);
                gather_layer->setMode(nvinfer1::GatherMode::kND);
                auto gather_output = gather_layer->getOutput(0);
                LOG_DEBUG("gather shape: " << gather_output->getDimensions());

                auto update_accumulate =
                  ctx->net->addElementWise(*gather_output, *expand_update, nvinfer1::ElementWiseOperation::kSUM);
                TORCHTRT_CHECK(update_accumulate, "Unable to create element sum layer from node: " << *n);
                auto update_accumulate_out = update_accumulate->getOutput(0);

                auto scatterLayer = ctx->net->addScatter(*in, *concate_indices, *update_accumulate_out, nvinfer1::ScatterMode::kND);
                TORCHTRT_CHECK(scatterLayer, "Unable to create scatter nd layer from node: " << *n);
                auto out = ctx->AssociateValueAndTensor(n->outputs()[0], scatterLayer->getOutput(0));
                LOG_DEBUG("Output tensor shape: " << out->getDimensions());
               }
               return true;
             }});
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
