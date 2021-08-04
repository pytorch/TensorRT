#include <stack>
#include <unordered_set>

#include "core/lowering/passes/passes.h"
#include "core/util/prelude.h"

namespace trtorch {
namespace core {
namespace lowering {
namespace passes {

std::string unmangle_cls_name(const std::string& name) {
  auto unmangled = name;

  std::size_t torch_prefix = unmangled.find("__torch__");
  if (torch_prefix != std::string::npos) {
    unmangled.erase(torch_prefix, 10);
  }

  std::size_t mangle_pos = unmangled.find("___torch_mangle_");
  if (mangle_pos != std::string::npos) {
    unmangled.erase(mangle_pos, 21);
  }
  return unmangled;
}

void NotateModuleForFallback(const torch::jit::Module& mod, std::string mod_name, const std::string& method_name, std::unordered_set<std::string> forced_fallback_modules) {
  auto cls_name = unmangle_cls_name(mod.type()->name()->qualifiedName());
  auto g = mod.get_method(method_name).graph();

  auto nodes = g->block()->nodes();
  bool changed_mod = false;
  for (const auto n : nodes) {
    if (n->kind() == torch::jit::prim::GetAttr) {
      auto out_type = unmangle_cls_name(c10::toString(n->output(0)->type()));
      if (forced_fallback_modules.find(out_type) != forced_fallback_modules.end()) {
        LOG_DEBUG("Marking module for fallback: " << n->s(c10::attr::name) << " (" << out_type << ") [owner: " << mod_name << " (" << cls_name << ")]");
        auto uses = n->output(0)->uses();
        for (const auto u : uses) {
          auto user = u.user;
          auto delim_start_n = g->create(torch::jit::prim::Enter, 0);
          delim_start_n->s_(c10::Symbol::attr("compilation_edge"), "start");
          auto num_end_outs = user->outputs().size();
          auto delim_end_n = g->create(torch::jit::prim::Exit, 0);
          delim_end_n->s_(c10::Symbol::attr("compilation_edge"), "end");
          delim_start_n->insertBefore(user);
          delim_end_n->insertAfter(user);
        }
        changed_mod = true;
      }
    }
  }

  if (changed_mod) {
    LOG_DEBUG(*g);
  }

  for (const auto sub_mod : mod.named_children()) {
    NotateModuleForFallback(sub_mod.value, sub_mod.name, method_name, forced_fallback_modules);
  }
}

void MarkNodesForFallback(std::shared_ptr<torch::jit::Graph>& g) {
  auto b = g->block();

  std::stack<bool> mark = std::stack<bool>({false});
  for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
    auto n = *it;
    if (!mark.top() && n->kind() == torch::jit::prim::Enter && n->hasAttributeS("compilation_edge")) {
      if (n->s(c10::Symbol::attr("compilation_edge")) == "start") {
          LOG_DEBUG("Starting to mark new segmented targeted for torch");
          mark.push(true);
          it.destroyCurrent();
      }
    } else if (mark.top() && n->kind() == torch::jit::prim::Enter && n->hasAttributeS("compilation_edge")) {
      if(n->s(c10::Symbol::attr("compilation_edge")) == "start") {
        LOG_DEBUG("Found the start of another segmented block targeted for torch while actively marking a block");
        mark.push(true);
        it.destroyCurrent();
      }
    } else if (mark.top() && n->kind() == torch::jit::prim::Exit && n->hasAttributeS("compilation_edge")) {
      if(n->s(c10::Symbol::attr("compilation_edge")) == "end") {
        LOG_DEBUG("Found the end of segmented block targeted for torch while actively marking a block");
        mark.pop();
        it.destroyCurrent();
      }
    } else if (!mark.top() && n->kind() == torch::jit::prim::Exit && n->hasAttributeS("compilation_edge")) {
      if(n->s(c10::Symbol::attr("compilation_edge")) == "end") {
        LOG_WARNING("Found the end of segmented block targeted for torch while not actively marking a block");
      }
    } else if (mark.top()) {
      LOG_GRAPH("Marking " << util::node_info(n) << " to run in PyTorch");
      n->i_(c10::Symbol::attr("to_compile"), (int64_t) false);
    }
  }

  LOG_GRAPH("Post marking ops for pytorch execution: " << *g);
}

} // Namespace passes
} // namespace lowering
} // namespace core
} // namespace trtorch
