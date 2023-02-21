#include <torch/csrc/jit/passes/subgraph_rewrite.h>

#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace lowering {
namespace passes {

void ReplaceAtenPad(std::shared_ptr<torch::jit::Graph>& graph) {
  for (auto it = graph->block()->nodes().begin(), end = graph->block()->nodes().end(); it != end; ++it) {
    if (it->kind() == c10::Symbol::fromQualString("aten::pad")) {
      // aten::pad(Tensor self, int[] pad, str mode='constant', float? value=None) -> (Tensor)
      auto mode = it->inputs()[2];
      if (mode->type()->isSubtypeOf(c10::StringType::get())) {
        std::string mode_str = torch::jit::toIValue(mode)->to<std::string>();
        if (mode_str == "reflect") {
          auto pad = it->inputs()[1];
          c10::List<int64_t> pad_list = torch::jit::toIValue(pad)->to<c10::List<int64_t>>();
          if (pad_list.size() == 2) {
            // aten::reflection_pad1d(Tensor self, int[2] padding) -> (Tensor)
            torch::jit::Node* new_node;
            new_node = graph->create(
                c10::Symbol::fromQualString("aten::reflection_pad1d"),
                torch::jit::ArrayRef<torch::jit::Value*>({it->inputs()[0], it->inputs()[1]}),
                1);
            new_node->insertAfter(*it);
            new_node->outputs()[0]->setType(c10::TensorType::get());
            it->outputs()[0]->replaceAllUsesWith(new_node->outputs()[0]);
            auto pre = --it;
            ++it;
            it->destroy();
            it = pre;
          } else if (pad_list.size() == 4) {
            // aten::reflection_pad2d(Tensor self, int[4] padding) -> (Tensor)
            torch::jit::Node* new_node;
            new_node = graph->create(
                c10::Symbol::fromQualString("aten::reflection_pad2d"),
                torch::jit::ArrayRef<torch::jit::Value*>({it->inputs()[0], it->inputs()[1]}),
                1);
            new_node->insertAfter(*it);
            new_node->outputs()[0]->setType(c10::TensorType::get());
            it->outputs()[0]->replaceAllUsesWith(new_node->outputs()[0]);
            auto pre = --it;
            ++it;
            it->destroy();
            it = pre;
          } else if (pad_list.size() == 6) {
            LOG_ERROR("Torch-TRT doesn't support aten::reflection_pad3d currently.");
          }

        } else if (mode_str == "replicate") {
          auto pad = it->inputs()[1];
          c10::List<int64_t> pad_list = torch::jit::toIValue(pad)->to<c10::List<int64_t>>();
          if (pad_list.size() == 2) {
            // aten::replication_pad1d(Tensor self, int[2] padding) -> (Tensor)
            torch::jit::Node* new_node;
            new_node = graph->create(
                c10::Symbol::fromQualString("aten::replication_pad1d"),
                torch::jit::ArrayRef<torch::jit::Value*>({it->inputs()[0], it->inputs()[1]}),
                1);
            new_node->insertAfter(*it);
            new_node->outputs()[0]->setType(c10::TensorType::get());
            it->outputs()[0]->replaceAllUsesWith(new_node->outputs()[0]);
            auto pre = --it;
            ++it;
            it->destroy();
            it = pre;
          } else if (pad_list.size() == 4) {
            // aten::replication_pad2d(Tensor self, int[4] padding) -> (Tensor)
            torch::jit::Node* new_node;
            new_node = graph->create(
                c10::Symbol::fromQualString("aten::replication_pad2d"),
                torch::jit::ArrayRef<torch::jit::Value*>({it->inputs()[0], it->inputs()[1]}),
                1);
            new_node->insertAfter(*it);
            new_node->outputs()[0]->setType(c10::TensorType::get());
            it->outputs()[0]->replaceAllUsesWith(new_node->outputs()[0]);
            auto pre = --it;
            ++it;
            it->destroy();
            it = pre;
          } else if (pad_list.size() == 6) {
            // aten::replication_pad3d(Tensor self, int[6] padding) -> (Tensor)
            torch::jit::Node* new_node;
            new_node = graph->create(
                c10::Symbol::fromQualString("aten::replication_pad3d"),
                torch::jit::ArrayRef<torch::jit::Value*>({it->inputs()[0], it->inputs()[1]}),
                1);
            new_node->insertAfter(*it);
            new_node->outputs()[0]->setType(c10::TensorType::get());
            it->outputs()[0]->replaceAllUsesWith(new_node->outputs()[0]);
            auto pre = --it;
            ++it;
            it->destroy();
            it = pre;
          }

        } else if (mode_str == "constant") {
          // aten::constant_pad_nd(Tensor self, int[] pad, Scalar value=0) -> (Tensor)
          torch::jit::Node* new_node;
          new_node = graph->create(
              c10::Symbol::fromQualString("aten::constant_pad_nd"),
              torch::jit::ArrayRef<torch::jit::Value*>({it->inputs()[0], it->inputs()[1], it->inputs()[3]}),
              1);
          new_node->insertAfter(*it);
          new_node->outputs()[0]->setType(c10::TensorType::get());
          it->outputs()[0]->replaceAllUsesWith(new_node->outputs()[0]);
          auto pre = --it;
          ++it;
          it->destroy();
          it = pre;
        } else if (mode_str == "circular") {
          LOG_ERROR("Torch-TRT doesn't support circular padding currently.");
        }
      }
    }
  }
  LOG_GRAPH("Post map aten::pad -> aten::constant_pad_nd/aten::reflection_padXd/aten::replication_padXd: " << *graph);
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace torch_tensorrt
