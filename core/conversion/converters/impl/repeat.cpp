#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"

#include <torch/torch.h>

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace converters {
namespace impl {

auto repeatinterleave TORCHTRT_UNUSED =
    RegisterNodeConversionPatterns()
        .pattern({
  "aten::repeat_interleave(Tensor self, int repeats, int? dim=None, int? output_size=None) -> Tensor",
            [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
              auto self = args[0].ITensorOrFreeze(ctx);
              auto repeats = args[1].unwrapToScalar().to<int>();

              auto input_shape = self->getDimensions();

              int dim;
              if (args[2].IValue()->isNone()) {
                dim = 0;

                // Flatten self tensor
                int size = 0;
                for (int i = 0; i < input_shape.nbDims; i++) {
                  size += input_shape.d[i];
                }
                auto flatten = ctx->net->addShuffle(*self);
                TORCHTRT_CHECK(flatten, "Unable to create shuffle layer from node: " << *n);
                flatten->setReshapeDimensions(util::toDims(std::vector<int64_t>({size})));
                self = flatten->getOutput(0);
                input_shape = self->getDimensions();
              }
              else {
                dim = args[2].unwrapToScalar().to<int>();
              }

              // Insert singleton dimension after desired repeat dimension
              std::vector<int64_t> repeat_shape_vec;
              for (int j = 0; j < input_shape.nbDims; j++) {
                repeat_shape_vec.push_back(input_shape.d[j]);
                if (j == dim) {
                  repeat_shape_vec.push_back(1);
                }
              }
              auto expand = ctx->net->addShuffle(*self);
              TORCHTRT_CHECK(expand, "Unable to create shuffle layer from node: " << *n);
              auto repeat_shape = util::toDims(repeat_shape_vec);
              expand->setReshapeDimensions(repeat_shape);

              // Expand on newly created singleton dimension
              repeat_shape.d[dim + 1] = repeats;
              std::vector<int64_t> start_vec(repeat_shape.nbDims, 0);
              auto start = util::toDims(start_vec);

              std::vector<int64_t> strides_vec(repeat_shape.nbDims, 1);
              strides_vec[dim + 1] = 0;
              auto strides = util::toDims(strides_vec);

              auto slice = ctx->net->addSlice(*expand->getOutput(0), start, repeat_shape, strides);

              // Collapse repeated dimension back into desired dimension
              std::vector<int64_t> collapse_shape_vec;
              for (int k = 0; k < repeat_shape.nbDims; k++) {
                if (k == dim) {
                  collapse_shape_vec.push_back(repeat_shape.d[k] * repeat_shape.d[++k]);
                } else {
                  collapse_shape_vec.push_back(repeat_shape.d[k]);
                }
              }
              auto collapse = ctx->net->addShuffle(*slice->getOutput(0));
              TORCHTRT_CHECK(collapse, "Unable to create shuffle layer from node: " << *n);
              collapse->setReshapeDimensions(util::toDims(collapse_shape_vec));

              collapse->setName(util::node_info(n).c_str());
              auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], collapse->getOutput(0));
              LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());

              return true;

            }});
} // namespace impl
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
