#include "NvInfer.h"
#include "core/conversion/converters/converters.h"
#include "core/conversion/tensorcontainer/TensorContainer.h"
#include "core/util/prelude.h"
#include "torch/torch.h"

#include <ATen/ATen.h>
#include <vector>

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

nvinfer1::ITensor* add_bias(
    nvinfer1::ITensor* a,
    nvinfer1::ITensor* b,
    std::string b_name,
    ConversionCtx* ctx,
    const torch::jit::Node* n) {
  auto a_dim = a->getDimensions();
  auto b_dim = b->getDimensions();

  LOG_DEBUG(b_name << " tensor shape: " << b_dim);

  TORCHTRT_CHECK(
      util::broadcastable(a_dim, b_dim, false),
      "bias " << b_name << " is not broadcastable - can't be added to previous matmul operation.");

  if (util::toVec(a_dim) != util::toVec(b_dim)) {
    LOG_DEBUG(b_name << "'s dimensions need to be reshaped");

    auto shuffle = ctx->net->addShuffle(*b);
    TORCHTRT_CHECK(shuffle, "Unable to create shuffle layer from node: " << *n);
    shuffle->setReshapeDimensions(util::toDimsPad(util::toVec(b_dim), a_dim.nbDims));

    b = shuffle->getOutput(0);
  }

  LOG_DEBUG(b_name << "'s shape: " << b->getDimensions());

  auto add = ctx->net->addElementWise(*a, *b, nvinfer1::ElementWiseOperation::kSUM);
  TORCHTRT_CHECK(add, "Unable to create ElementWise layer from node: " << *n);

  return add->getOutput(0);
}

auto lstm_cell_registrations TORCHTRT_UNUSED =
    RegisterNodeConversionPatterns()
        .pattern(
            {"aten::gru_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto input = args[0].ITensorOrFreeze(ctx);
               auto hx = args[1].ITensorOrFreeze(ctx);
               auto w_ih = args[2].ITensorOrFreeze(ctx);
               auto w_hh = args[3].ITensorOrFreeze(ctx);

               LOG_DEBUG("Input tensor shape: " << input->getDimensions());
               LOG_DEBUG("w_ih tensor shape: " << w_ih->getDimensions());
               LOG_DEBUG("w_hh tensor shape: " << w_hh->getDimensions());

               // calculate first half of gates
               auto mm1 = ctx->net->addMatrixMultiply(
                   *input, nvinfer1::MatrixOperation::kNONE, *w_ih, nvinfer1::MatrixOperation::kTRANSPOSE);
               TORCHTRT_CHECK(mm1, "Unable to create matrix multiplication node: " << *n);
               auto mm1_out = mm1->getOutput(0);

               auto out1 = (args[4].isIValue() && args[4].IValue()->isNone())
                   ? mm1_out
                   : add_bias(mm1_out, args[4].ITensorOrFreeze(ctx), "b_ih", ctx, n);

               // calculate second half of gates
               auto mm2 = ctx->net->addMatrixMultiply(
                   *hx, nvinfer1::MatrixOperation::kNONE, *w_hh, nvinfer1::MatrixOperation::kTRANSPOSE);
               TORCHTRT_CHECK(mm2, "Unable to create matrix multiplication node: " << *n);
               auto mm2_out = mm2->getOutput(0);

               auto out2 = (args[5].isIValue() && args[5].IValue()->isNone())
                   ? mm2_out
                   : add_bias(mm2_out, args[5].ITensorOrFreeze(ctx), "b_hh", ctx, n);

               // chunk the first and second half of gates into 3 parts
               auto dims = util::toVec(out1->getDimensions());
               auto batch = dims[0];
               auto hidden = dims[1] / 3;

               std::vector<int64_t> size_vec = {batch, hidden};
               std::vector<int64_t> stride_vec = {1, 1};
               std::vector<int64_t> offset0 = {0, 0};
               std::vector<int64_t> offset1 = {0, hidden};
               std::vector<int64_t> offset2 = {0, 2 * hidden};

               auto size = util::toDims(size_vec);
               auto stride = util::toDims(stride_vec);

               auto out1_r_elm = ctx->net->addSlice(*out1, util::toDims(offset0), size, stride);
               TORCHTRT_CHECK(out1_r_elm, "Unable to create Slice layer from node: " << *n);
               auto out1_z_elm = ctx->net->addSlice(*out1, util::toDims(offset1), size, stride);
               TORCHTRT_CHECK(out1_z_elm, "Unable to create Slice layer from node: " << *n);
               auto out1_n_elm = ctx->net->addSlice(*out1, util::toDims(offset2), size, stride);
               TORCHTRT_CHECK(out1_n_elm, "Unable to create Slice layer from node: " << *n);
               auto out1_r = out1_r_elm->getOutput(0);
               auto out1_z = out1_z_elm->getOutput(0);
               auto out1_n = out1_n_elm->getOutput(0);

               auto out2_r_elm = ctx->net->addSlice(*out2, util::toDims(offset0), size, stride);
               TORCHTRT_CHECK(out2_r_elm, "Unable to create Slice layer from node: " << *n);
               auto out2_z_elm = ctx->net->addSlice(*out2, util::toDims(offset1), size, stride);
               TORCHTRT_CHECK(out2_z_elm, "Unable to create Slice layer from node: " << *n);
               auto out2_n_elm = ctx->net->addSlice(*out2, util::toDims(offset2), size, stride);
               TORCHTRT_CHECK(out2_n_elm, "Unable to create Slice layer from node: " << *n);
               auto out2_r = out2_r_elm->getOutput(0);
               auto out2_z = out2_z_elm->getOutput(0);
               auto out2_n = out2_n_elm->getOutput(0);

               // compute the R gate
               auto add_r = ctx->net->addElementWise(*out1_r, *out2_r, nvinfer1::ElementWiseOperation::kSUM);
               TORCHTRT_CHECK(add_r, "Unable to create ElementWise layer from node: " << *n);
               auto activ_r = ctx->net->addActivation(*add_r->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
               TORCHTRT_CHECK(activ_r, "Unable to create sigmoid activation layer from node: " << *n);
               auto r_gate = activ_r->getOutput(0);

               // compute the Z gate
               auto add_z = ctx->net->addElementWise(*out1_z, *out2_z, nvinfer1::ElementWiseOperation::kSUM);
               TORCHTRT_CHECK(add_z, "Unable to create ElementWise layer from node: " << *n);
               auto activ_z = ctx->net->addActivation(*add_z->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
               TORCHTRT_CHECK(activ_z, "Unable to create sigmoid activation layer from node: " << *n);
               auto z_gate = activ_z->getOutput(0);

               // compute the N gate
               auto out2_n_transform =
                   ctx->net->addElementWise(*r_gate, *out2_n, nvinfer1::ElementWiseOperation::kPROD);
               TORCHTRT_CHECK(out2_n_transform, "Unable to create ElementWise layer from node: " << *n);
               auto add_n = ctx->net->addElementWise(
                   *out1_n, *out2_n_transform->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
               TORCHTRT_CHECK(add_n, "Unable to create ElementWise layer from node: " << *n);
               auto activ_n = ctx->net->addActivation(*add_n->getOutput(0), nvinfer1::ActivationType::kTANH);
               TORCHTRT_CHECK(activ_n, "Unable to create tanh activation layer from node: " << *n);
               auto n_gate = activ_n->getOutput(0);

               // compute new state H'
               auto interm_1 = ctx->net->addElementWise(*z_gate, *n_gate, nvinfer1::ElementWiseOperation::kPROD);
               TORCHTRT_CHECK(interm_1, "Unable to create ElementWise layer from node: " << *n);
               auto interm_2 = ctx->net->addElementWise(*z_gate, *hx, nvinfer1::ElementWiseOperation::kPROD);
               TORCHTRT_CHECK(interm_2, "Unable to create ElementWise layer from node: " << *n);
               auto interm_3 =
                   ctx->net->addElementWise(*n_gate, *interm_1->getOutput(0), nvinfer1::ElementWiseOperation::kSUB);
               TORCHTRT_CHECK(interm_3, "Unable to create ElementWise layer from node: " << *n);
               auto h_prime = ctx->net->addElementWise(
                   *interm_2->getOutput(0), *interm_3->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
               TORCHTRT_CHECK(h_prime, "Unable to create ElementWise layer from node: " << *n);
               auto h_prime_out = h_prime->getOutput(0);

               ctx->AssociateValueAndTensor(n->outputs()[0], h_prime_out);

               LOG_DEBUG("Output tensor [h'] shape: " << h_prime_out->getDimensions());

               return true;
             }})
        .pattern(
            {"aten::lstm_cell(Tensor input, Tensor[] hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> (Tensor, Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto input = args[0].ITensorOrFreeze(ctx);
               auto w_ih = args[2].ITensorOrFreeze(ctx);
               auto w_hh = args[3].ITensorOrFreeze(ctx);

               LOG_DEBUG("Input tensor shape: " << input->getDimensions());
               LOG_DEBUG("w_ih tensor shape: " << w_ih->getDimensions());
               LOG_DEBUG("w_hh tensor shape: " << w_hh->getDimensions());

               std::vector<nvinfer1::ITensor*> state;
               auto hx = args[1].IValue()->toListRef();
               for (unsigned int i = 0; i < hx.size(); i++) {
                 auto t = hx[i];

                 nvinfer1::ITensor* itensor;

                 if (t.isTensor()) {
                   itensor = tensor_to_const(ctx, t.toTensor());
                 } else {
                   auto cont = t.toCustomClass<TensorContainer>();
                   itensor = cont->tensor();
                 }

                 LOG_DEBUG("State tensor " << i << " shape: " << itensor->getDimensions());
                 state.push_back(itensor);
               }

               // calculate first half of gates
               auto mm1 = ctx->net->addMatrixMultiply(
                   *input, nvinfer1::MatrixOperation::kNONE, *w_ih, nvinfer1::MatrixOperation::kTRANSPOSE);
               TORCHTRT_CHECK(mm1, "Unable to create matrix multiplication node: " << *n);
               auto mm1_out = mm1->getOutput(0);

               auto out1 = (args[4].isIValue() && args[4].IValue()->isNone())
                   ? mm1_out
                   : add_bias(mm1_out, args[4].ITensorOrFreeze(ctx), "b_ih", ctx, n);

               // calculate second half of gates
               auto mm2 = ctx->net->addMatrixMultiply(
                   *state[0], nvinfer1::MatrixOperation::kNONE, *w_hh, nvinfer1::MatrixOperation::kTRANSPOSE);
               TORCHTRT_CHECK(mm2, "Unable to create matrix multiplication node: " << *n);
               auto mm2_out = mm2->getOutput(0);

               auto out2 = (args[5].isIValue() && args[5].IValue()->isNone())
                   ? mm2_out
                   : add_bias(mm2_out, args[5].ITensorOrFreeze(ctx), "b_hh", ctx, n);

               // get all 4 gates
               auto add = ctx->net->addElementWise(*out1, *out2, nvinfer1::ElementWiseOperation::kSUM);
               TORCHTRT_CHECK(add, "Unable to create ElementWise layer from node: " << *n);
               auto add_out = add->getOutput(0);

               // chunk Tensor into 4 parts and apply activation functions
               auto dims = util::toVec(add_out->getDimensions());
               auto batch = dims[0];
               auto hidden = dims[1] / 4;

               std::vector<int64_t> size_vec = {batch, hidden};
               std::vector<int64_t> stride_vec = {1, 1};
               std::vector<int64_t> offset0 = {0, 0};
               std::vector<int64_t> offset1 = {0, hidden};
               std::vector<int64_t> offset2 = {0, 2 * hidden};
               std::vector<int64_t> offset3 = {0, 3 * hidden};

               auto size = util::toDims(size_vec);
               auto stride = util::toDims(stride_vec);

               auto slice1 = ctx->net->addSlice(*add_out, util::toDims(offset0), size, stride);
               TORCHTRT_CHECK(slice1, "Unable to create Slice layer from node: " << *n);
               auto activ1 = ctx->net->addActivation(*slice1->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
               TORCHTRT_CHECK(activ1, "Unable to create sigmoid activation layer from node: " << *n);
               auto ingate = activ1->getOutput(0);

               auto slice2 = ctx->net->addSlice(*add_out, util::toDims(offset1), size, stride);
               TORCHTRT_CHECK(slice2, "Unable to create Slice layer from node: " << *n);
               auto activ2 = ctx->net->addActivation(*slice2->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
               TORCHTRT_CHECK(activ2, "Unable to create sigmoid activation layer from node: " << *n);
               auto forgetgate = activ2->getOutput(0);

               auto slice3 = ctx->net->addSlice(*add_out, util::toDims(offset2), size, stride);
               TORCHTRT_CHECK(slice3, "Unable to create Slice layer from node: " << *n);
               auto activ3 = ctx->net->addActivation(*slice3->getOutput(0), nvinfer1::ActivationType::kTANH);
               TORCHTRT_CHECK(activ3, "Unable to create tanh activation layer from node: " << *n);
               auto cellgate = activ3->getOutput(0);

               auto slice4 = ctx->net->addSlice(*add_out, util::toDims(offset3), size, stride);
               TORCHTRT_CHECK(slice4, "Unable to create Slice layer from node: " << *n);
               auto activ4 = ctx->net->addActivation(*slice4->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
               TORCHTRT_CHECK(activ4, "Unable to create sigmoid activation layer from node: " << *n);
               auto outgate = activ4->getOutput(0);

               // compute cy
               auto forget_cx = ctx->net->addElementWise(*forgetgate, *state[1], nvinfer1::ElementWiseOperation::kPROD);
               TORCHTRT_CHECK(forget_cx, "Unable to create ElementWise layer from node: " << *n);
               auto in_cell = ctx->net->addElementWise(*ingate, *cellgate, nvinfer1::ElementWiseOperation::kPROD);
               TORCHTRT_CHECK(in_cell, "Unable to create ElementWise layer from node: " << *n);
               auto cy = ctx->net->addElementWise(
                   *forget_cx->getOutput(0), *in_cell->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
               TORCHTRT_CHECK(cy, "Unable to create ElementWise layer from node: " << *n);
               auto cy_out = cy->getOutput(0);

               // compute hy
               auto cy_tanh = ctx->net->addActivation(*cy_out, nvinfer1::ActivationType::kTANH);
               TORCHTRT_CHECK(cy_tanh, "Unable to create tanh activation layer from node: " << *n);
               auto hy =
                   ctx->net->addElementWise(*outgate, *cy_tanh->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
               TORCHTRT_CHECK(hy, "Unable to create ElementWise layer from node: " << *n);
               auto hy_out = hy->getOutput(0);

               ctx->AssociateValueAndTensor(n->outputs()[0], hy_out);
               ctx->AssociateValueAndTensor(n->outputs()[1], cy_out);

               LOG_DEBUG("Output tensor [hy] shape: " << hy_out->getDimensions());
               LOG_DEBUG("Output tensor [cy] shape: " << cy_out->getDimensions());

               return true;
             }});
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
