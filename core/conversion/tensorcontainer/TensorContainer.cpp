#include "core/conversion/tensorcontainer/TensorContainer.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace {

static auto tensor_container =
  torch::class_<TensorContainer>("_eval_ivalue_types", "TensorContainer")
      .def(torch::init<>());
} // namespace
} // conversion
} // core
} // trtorch