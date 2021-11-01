#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

auto linear_registrations TORCHTRT_UNUSED = RegisterNodeConversionPatterns().pattern(
    {"aten::linear(Tensor input, Tensor weight, Tensor? bias = None) -> (Tensor)",
     [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
       // PyTorch follows in: Nx*xIN, W: OUTxIN, B: OUT, out: Nx*xOUT
       // TensorRT inserts a flatten in when following conv
       auto in = args[0].ITensorOrFreeze(ctx);
       auto shape = util::toVec(in->getDimensions());

       LOG_DEBUG("Input tensor shape: " << in->getDimensions());

       TORCHTRT_ASSERT(
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

       TORCHTRT_CHECK(new_layer, "Unable to create linear layer from node: " << *n);

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
} // namespace torch_tensorrt
