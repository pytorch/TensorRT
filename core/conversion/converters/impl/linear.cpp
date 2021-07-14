#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

auto linear_registrations TRTORCH_UNUSED = RegisterNodeConversionPatterns().pattern(
    {"aten::linear(Tensor input, Tensor weight, Tensor? bias = None) -> (Tensor)",
     [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
       // PyTorch follows in: Nx*xIN, W: OUTxIN, B: OUT, out: Nx*xOUT
       // TensorRT inserts a flatten in when following conv
       auto in = args[0].ITensorOrFreeze(ctx);
       auto shape = util::toVec(in->getDimensions());

       LOG_DEBUG("Input tensor shape: " << in->getDimensions());

       TRTORCH_ASSERT(
           shape.size() >= 2,
           "aten::linear expects input tensors to be of shape [N,..., in features], but found input Tensor less than 2D");

       if (shape.size() < 4) {
         // Flatten
         std::vector<int64_t> new_shape;
         new_shape.push_back(shape[0]);
         new_shape.push_back(1);
         new_shape.push_back(1);
         new_shape.push_back(util::volume(util::toDims(shape)) / shape[0]);

         auto new_dims = util::toDims(new_shape);
         LOG_DEBUG(
             "Input shape is less than 4D got: "
             << util::toDims(shape) << ", inserting shuffle layer to reshape to 4D tensor shape: " << new_dims);
         auto in_shuffle = ctx->net->addShuffle(*in);
         in_shuffle->setReshapeDimensions(new_dims);
         in_shuffle->setName((util::node_info(n) + " [Input Reshape to " + util::toStr(new_dims) + ']').c_str());
         in = in_shuffle->getOutput(0);
       }

       // Get the bias
       Weights bias;
       if (!args[2].IValue()->isNone()) {
         bias = Weights(ctx, args[2].IValue()->toTensor());
       } else {
         bias = Weights();
       }

       // Handle case when weights of conv/deconv is an ITensor. This case happens for QAT networks where
       // conv_weights -> Quantize -> Dequantize -> new_conv_weights -> conv <- input
       // new_conv_weights will be an ITensor because it is an output of Dequantize layer defined in
       // impl/quantization.cpp
       if (args[1].isITensor()) {
         auto kernel_tensor = args[1].ITensor();
         auto kernel_dims = args[1].ITensor()->getDimensions();
         // Initialize a dummy constant kernel to pass it to INetwork->addConvolutionNd/addDeconvolutionNd API.
         auto kernel_weights = nvinfer1::Weights{nvinfer1::DataType::kFLOAT, nullptr, 0};
         auto fc_layer = ctx->net->addFullyConnected(*in, kernel_dims.d[0], kernel_weights, bias.data);
         fc_layer->setInput(1, *kernel_tensor);
         fc_layer->setName(util::node_info(n).c_str());
         auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], fc_layer->getOutput(0));
         LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
         return true;
       }

       auto w_tensor = args[1].IValue()->toTensor();
       Weights w = Weights(ctx, w_tensor);

       nvinfer1::ILayer* new_layer;
       if (!args[2].IValue()->isNone()) {
         Weights b(ctx, args[2].IValue()->toTensor());
         new_layer = ctx->net->addFullyConnected(*in, w.num_output_maps, w.data, b.data);
       } else {
         LOG_DEBUG("There is no bias for the linear layer");
         new_layer = ctx->net->addFullyConnected(*in, w.num_output_maps, w.data, Weights().data);
       }

       TRTORCH_CHECK(new_layer, "Unable to create linear layer from node: " << *n);

       new_layer->setName(util::node_info(n).c_str());
       auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], new_layer->getOutput(0));

       LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());

       return true;
     }});
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
