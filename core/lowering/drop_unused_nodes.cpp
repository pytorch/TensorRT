#include "torch/csrc/jit/ir/ir.h"

namespace torch {
namespace jit {

namespace prim {
using namespace ::c10::prim;
} // namespace prim

} // namespace jit
} // namespace torch

namespace torch_tensorrt {
namespace core {
namespace lowering {

// From torch/csrc/jit/interpreter.cpp
void DropUnusedNodes(torch::jit::Block* b) {
  auto create_drop_if_unused = [&](at::ArrayRef<torch::jit::Value*> values) -> torch::jit::Node* {
    std::vector<torch::jit::Value*> to_drop;
    for (auto v : values) {
      if (v->uses().size() == 0 && v->node()->kind() != torch::jit::prim::Constant)
        to_drop.push_back(v);
    }
    if (to_drop.size() == 0)
      return nullptr;
    return b->owningGraph()->create(torch::jit::prim::Drop, to_drop, 0);
  };

  if (auto d = create_drop_if_unused(b->inputs())) {
    b->prependNode(d);
  }
  for (auto n : b->nodes()) {
    if (auto d = create_drop_if_unused(n->outputs())) {
      d->insertAfter(n);
    }
    for (auto b : n->blocks()) {
      DropUnusedNodes(b);
    }
  }
}

} // namespace lowering
} // namespace core
} // namespace torch_tensorrt
