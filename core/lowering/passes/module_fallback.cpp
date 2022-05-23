#include <stack>
#include <unordered_set>

#include "core/lowering/passes/passes.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
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

void NotateModuleForFallback(
    const torch::jit::Module& mod,
    std::string mod_name,
    std::string method_name,
    std::unordered_set<std::string> forced_fallback_modules) {
  auto cls_name = unmangle_cls_name(mod.type()->name()->qualifiedName());

  auto g = mod.get_method(method_name).graph();
  auto nodes = g->block()->nodes();
  bool changed_mod = false;
  for (const auto n : nodes) {
    if (n->kind() == torch::jit::prim::GetAttr) {
      auto out_type = unmangle_cls_name(c10::toString(n->output(0)->type()));
      if (forced_fallback_modules.find(out_type) != forced_fallback_modules.end()) {
        LOG_GRAPH(
            "Notating module for fallback: " << n->s(c10::attr::name) << " (" << out_type << ") [owner: " << mod_name
                                             << " (" << cls_name << ")]");
        auto uses = n->output(0)->uses();
        int k = 0;
        for (const auto u : uses) {
          auto compilation_context_node = g->createNone();
          auto compilation_context = compilation_context_node->outputs()[0];
          compilation_context->setDebugName("compilation_context_" + std::to_string(k++));
          auto user = u.user;
          auto delim_start_n = g->create(torch::jit::prim::Enter, {compilation_context});
          delim_start_n->s_(c10::Symbol::attr("compilation_edge"), "start");
          auto delim_end_n = g->create(torch::jit::prim::Exit, {compilation_context});
          delim_end_n->s_(c10::Symbol::attr("compilation_edge"), "end");
          compilation_context_node->insertBefore(user);
          delim_start_n->insertBefore(user);
          delim_end_n->insertAfter(user);
        }
        changed_mod = true;
      }
    }
  }

  if (changed_mod) {
    LOG_GRAPH("Notated graph: " << *g);
  }

  if (mod.named_children().size() > 0) {
    for (const auto n : nodes) {
      std::string sub_method_name = "";
      if (n->kind() == torch::jit::prim::CallMethod) {
        sub_method_name = n->s(c10::Symbol::attr("name"));
        auto sub_mod_val = n->input(0);
        auto sub_mod_src_n = sub_mod_val->node();
        if (!sub_mod_src_n->hasAttributeS("name")) {
          LOG_GRAPH("Node: " << util::node_info(sub_mod_src_n) << " manages a module with no name, skipping");
          break;
        }
        auto sub_mod_name = sub_mod_src_n->s(c10::Symbol::attr("name"));
        for (const auto sub_mod : mod.named_children()) {
          // Theres probably a way to directly access the module we care about
          if (sub_mod.name == sub_mod_name) {
            LOG_GRAPH(
                "Looking at <module>.<method>() next: " << sub_mod_name << "." << sub_method_name
                                                        << "() (lowering.passes.NotateModuleForFallback)");
            NotateModuleForFallback(sub_mod.value, sub_mod.name, sub_method_name, forced_fallback_modules);
          }
        }
      }
    }
  }
}

void MarkNodesForFallback(std::shared_ptr<torch::jit::Graph>& g, bool delete_delims) {
  auto b = g->block();

  std::stack<bool> mark = std::stack<bool>({false});
  for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
    auto n = *it;
    if (!mark.top() && n->kind() == torch::jit::prim::Enter && n->hasAttributeS("compilation_edge")) {
      if (n->s(c10::Symbol::attr("compilation_edge")) == "start") {
        LOG_GRAPH("Starting to mark new segmented block targeted for torch");
        mark.push(true);
        if (delete_delims) {
          it.destroyCurrent();
        }
      }
    } else if (mark.top() && n->kind() == torch::jit::prim::Enter && n->hasAttributeS("compilation_edge")) {
      if (n->s(c10::Symbol::attr("compilation_edge")) == "start") {
        LOG_GRAPH("Found the start of another segmented block targeted for torch while actively marking a block");
        mark.push(true);
        if (delete_delims) {
          it.destroyCurrent();
        }
      }
    } else if (mark.top() && n->kind() == torch::jit::prim::Exit && n->hasAttributeS("compilation_edge")) {
      if (n->s(c10::Symbol::attr("compilation_edge")) == "end") {
        LOG_GRAPH("Found the end of segmented block targeted for torch while actively marking a block");
        mark.pop();
        if (delete_delims) {
          it.destroyCurrent();
        }
      }
    } else if (!mark.top() && n->kind() == torch::jit::prim::Exit && n->hasAttributeS("compilation_edge")) {
      if (n->s(c10::Symbol::attr("compilation_edge")) == "end") {
        LOG_WARNING("Found the end of segmented block targeted for torch while not actively marking a block");
      }
    } else if (mark.top()) {
      LOG_GRAPH("Marking " << util::node_info(n) << " to run in PyTorch");
      n->i_(c10::Symbol::attr("to_compile"), (int64_t) false);
    }
  }

  LOG_GRAPH("After marking operations for torch fallback: " << *g);
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace torch_tensorrt
