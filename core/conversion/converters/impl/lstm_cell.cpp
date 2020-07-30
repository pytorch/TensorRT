#include "torch/torch.h"
#include "NvInfer.h"
#include "core/util/prelude.h"
#include "core/conversion/converters/converters.h"
#include "core/conversion/tensorcontainer/TensorContainer.h"

#include <ATen/ATen.h>
#include <vector>

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

auto lstm_cell_registrations TRTORCH_UNUSED = RegisterNodeConversionPatterns()
    .pattern({
        "aten::lstm_cell(Tensor input, Tensor[] hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> (Tensor, Tensor)",
        [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
            auto input = args[0].ITensorOrFreeze(ctx);
            auto w_ih = args[2].ITensorOrFreeze(ctx);
            auto w_hh = args[3].ITensorOrFreeze(ctx);
            auto b_ih = args[4].ITensorOrFreeze(ctx);
            auto b_hh = args[5].ITensorOrFreeze(ctx);

            LOG_DEBUG("Input tensor shape: " << input->getDimensions());
            LOG_DEBUG("w_ih tensor shape: " << w_ih->getDimensions());
            LOG_DEBUG("w_hh tensor shape: " << w_hh->getDimensions());
            LOG_DEBUG("b_ih tensor shape: " << b_ih->getDimensions());
            LOG_DEBUG("b_hh tensor shape: " << b_hh->getDimensions());
            
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
            auto mm1 = ctx->net->addMatrixMultiply(*input, nvinfer1::MatrixOperation::kNONE, *w_ih, nvinfer1::MatrixOperation::kTRANSPOSE);
            TRTORCH_CHECK(mm1, "Unable to create matrix multiplication node: " << *n);

            auto mm1_out = mm1->getOutput(0);
            auto mm1_dim = mm1_out->getDimensions();
            auto b_ih_dim = b_ih->getDimensions();

            TRTORCH_CHECK(util::broadcastable(mm1_dim, b_ih_dim, false));

            if (util::toVec(mm1_dim) != util::toVec(b_ih_dim)) {
                LOG_DEBUG("b_ih dimensions need to be reshaped");

                auto shuffle = ctx->net->addShuffle(*b_ih);
                TRTORCH_CHECK(shuffle, "Unable to create shuffle layer from node: " << *n);
                shuffle->setReshapeDimensions(util::toDimsPad(util::toVec(b_ih_dim), mm1_dim.nbDims));
                b_ih = shuffle->getOutput(0);
            }

            auto add1 = ctx->net->addElementWise(*mm1_out, *b_ih, nvinfer1::ElementWiseOperation::kSUM);
            TRTORCH_CHECK(add1, "Unable to create ElementWise layer from node: " << *n);
            auto add1_out = add2->getOutput(0);

            // calculate second half of gates
            auto mm2 = ctx->net->addMatrixMultiply(*state[0], nvinfer1::MatrixOperation::kNONE, *w_hh, nvinfer1::MatrixOperation::kTRANSPOE);
            TRTORCH_CHECK(mm2, "Unable to create matrix multiplication node: " << *n);

            auto mm2_out = mm2->getOutput(0);
            auto mm2_dim = mm2_out->getDimensions();
            auto b_hh_dim = b_hh->getDimensions();

            TRTORCH_CHECK(util::broadcastable(mm2_dim, b_hh_dim, false));

            if (util::toVec(mm2_dim) != util::toVec(b_hh_dim)) {
                LOG_DEBUG("b_hh dimensions need to be reshaped");

                auto shuffle = ctx->net->addShuffle(*b_hh);
                TRTORCH_CHECK(shuffle, "Unable to create shuffle layer from node: " << *n);
                shuffle->setReshapeDimensions(util::toDimsPad(util::toVec(b_hh_dim), mm2_dim.nbDims));
                b_hh = shuffle->getOutput(0);
            }

            auto add2 = ctx->net->addElementWise(*mm2_out, *b_ih, nvinfer1::ElementWiseOperation::kSUM);
            TRTORCH_CHECK(add2, "Unable to create ElementWise layer from node: " << *n);
            auto add2_out = add2->getOutput(0);

            // gates
            auto add3 = ctx->net->addElementWise(*add1_out, *add2_out, nvinfer1::ElementWiseOperation::kSUM);
            TRTORCH_CHECK(add3, "Unable to create ElementWise layer from node: " << *n);
            auto add3_out = add3->getOutput(0);

            // chunk Tensor into 4 parts and apply activation functions
            auto dims = util::toVec(add3_out->getDimensions());
            auto batch = dims[0];
            auto hidden = dims[1]/4;

            auto size = util::toDims(std::vector<int64_t>({batch, hidden}));
            auto stride = util::toDims(std::vector<int64_t>({1, 1}));

            auto slice1 = ctx->net->addSlice(*add3_out, util::toDims(std::vector<int64_t>({0, 0})), size, stride);
            TRTORCH_CHECK(slice1, "Unable to create Slice layer from node: " << *n);
            auto activ1 = ctx->net->addActivation(*slice1->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
            TRTORCH_CHECK(activ1, "Unable to create sigmoid activation layer from node: " << *n);
            auto ingate = activ1->getOutput(0);

            auto slice2 = ctx->net->addSlice(*add3_out, util::toDims(std::vector<int64_t>({0, hidden})), size, stride);
            TRTORCH_CHECK(slice2, "Unable to create Slice layer from node: " << *n);
            auto activ2 = ctx->net->addActivation(*slice2->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
            TRTORCH_CHECK(activ2, "Unable to create sigmoid activation layer from node: " << *n);
            auto forgetgate = activ2->getOutput(0);

            auto slice3 = ctx->net->addSlice(*add3_out, util::toDims(std::vector<int64_t>({0, 2*hidden})), size, stride);
            TRTORCH_CHECK(slice3, "Unable to create Slice layer from node: " << *n);
            auto activ3 = ctx->net->addActivation(*slice3->getOutput(0), nvinfer1::ActivationType::kTANH);
            TRTORCH_CHECK(activ3, "Unable to create tanh activation layer from node: " << *n);
            auto cellgate = activ3->getOutput(0);

            auto slice4 = ctx->net->addSlice(*add3_out, util::toDims(std::vector<int64_t>({0, 3*hidden})), size, stride);
            TRTORCH_CHECK(slice4, "Unable to create Slice layer from node: " << *n);
            auto activ4 = ctx->net->addActivation(*slice4->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
            TRTORCH_CHECK(activ4, "Unable to create sigmoid activation layer from node: " << *n);
            auto outgate = activ4->getOutput(0);

            // compute cy
            auto forget_cx = ctx->net->addElementWise(*forgetgate, *state[1], nvinfer1::ElementWiseOperation::kPROD);
            TRTORCH_CHECK(forget_cx, "Unable to create ElementWise layer from node: " << *n);
            auto in_cell = ctx->net->addElementWise(*ingate, *cellgate, nvinfer1::ElementWiseOperation::kPROD);
            TRTORCH_CHECK(in_cell, "Unable to create ElementWise layer from node: " << *n);
            auto cy = ctx->net->addElementWise(*forget_cx->getOutput(0), *in_cell->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
            TRTORCH_CHECK(cy, "Unable to create ElementWise layer from node: " << *n);
            auto cy_out = ctx->AssociateValueAndTensor(n->outputs()[1], cy->getOutput(0));

            // compute hy
            auto cy_tanh = ctx->net->addActivation(*cy_out, nvinfer1::ActivationType::kTANH);
            TRTORCH_CHECK(cy_tanh, "Unable to create tanh activation layer from node: " << *n);
            auto hy = ctx->net->addElementWise(*outgate, *cy_tanh->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
            TRTORCH_CHECK(hy, "Unable to create ElementWise layer from node: " << *n);
            auto hy_out = ctx->AssociateValueAndTensor(n->outputs()[0], hy->getOutput(0));

            LOG_DEBUG("Output tensor [hy] shape: " << hy_out->getDimensions());
            LOG_DEBUG("Output tensor [cy] shape: " << cy_out->getDimensions());

            return true;
    }
  });
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch