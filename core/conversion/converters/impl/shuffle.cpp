#include "core/conversion/converters/converters.h"

#include "torch/torch.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

static auto shuffle_registrations = RegisterNodeConversionPatterns()
  .pattern({
    "aten::flatten.using_ints(Tensor self, int start_dim=0, int end_dim=-1) -> (Tensor)",
    [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
      auto in = args[0].ITensor();
      auto start_dim = args[1].unwrapToInt();
      auto end_dim = args[2].unwrapToInt();
      auto in_shape = util::toVec(in->getDimensions());
      auto out_shape = torch::flatten(torch::rand(in_shape), start_dim, end_dim).sizes();

      auto shuffle = ctx->net->addShuffle(*in);
      TRTORCH_CHECK(shuffle, "Unable to create shuffle layer from node: " << *n);
      shuffle->setReshapeDimensions(util::toDims(out_shape));
      shuffle->setName(util::node_info(n).c_str());

      auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], shuffle->getOutput(0));
      LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
      return true;
    }
  }).pattern({
    "aten::reshape(Tensor self, int[] shape) -> (Tensor)",
    [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
      auto in = args[0].ITensor();
      auto in_shape = util::toVec(in->getDimensions());
      auto new_shape = torch::reshape(torch::rand(in_shape), args[1].unwrapToIntList().vec()).sizes();

      auto shuffle = ctx->net->addShuffle(*in);
      TRTORCH_CHECK(shuffle, "Unable to create shuffle layer from node: " << *n);
      shuffle->setReshapeDimensions(util::toDims(new_shape));
      shuffle->setName(util::node_info(n).c_str());

      auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], shuffle->getOutput(0));
      LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());

      return true;
    }
  }).pattern({
    "aten::view(Tensor(a) self, int[] size) -> (Tensor(a))",
    [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
      auto in = args[0].ITensor();
      auto in_shape = util::toVec(in->getDimensions());
      auto ex_tensor = torch::rand(in_shape);
      auto new_shape = ex_tensor.view(args[1].unwrapToIntList().vec()).sizes();

      auto shuffle = ctx->net->addShuffle(*in);
      TRTORCH_CHECK(shuffle, "Unable to create shuffle layer from node: " << *n);
      shuffle->setReshapeDimensions(util::toDims(new_shape));
      shuffle->setName(util::node_info(n).c_str());

      auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], shuffle->getOutput(0));
      LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());

      return true;
    }
  });
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
