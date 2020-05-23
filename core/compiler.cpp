#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "NvInfer.h"

#include "ATen/core/function_schema.h"

#include "torch/csrc/jit/frontend/function_schema_parser.h"
#include "torch/csrc/jit/ir/ir.h"
#include "torch/csrc/jit/passes/pass_manager.h"
#include "torch/csrc/jit/passes/lower_graph.h"
#include "torch/csrc/jit/passes/graph_fuser.h"

#include "core/util/prelude.h"
#include "core/compiler.h"

#include "core/lowering/lowering.h"
#include "core/conversion/conversion.h"
#include "core/execution/execution.h"

namespace trtorch {
namespace core {

c10::FunctionSchema GenerateGraphSchema(torch::jit::script::Module mod, std::string method_name, std::shared_ptr<torch::jit::Graph>& g) {

    std::vector<c10::Argument> args;
    for (auto in : g->inputs()) {
        args.push_back(c10::Argument(in->debugName(), in->type()));
    }

    std::vector<c10::Argument> returns;
    for (auto out : g->outputs()) {
        returns.push_back(c10::Argument(out->debugName(), out->type()));
    }

    return c10::FunctionSchema(method_name, method_name, args, returns);
}


void AddEngineToGraph(torch::jit::script::Module mod, std::shared_ptr<torch::jit::Graph>& g, std::string& serialized_engine) {
    execution::EngineID uid = execution::RegisterEngineFromSerializedEngine(serialized_engine);
    auto num_io = execution::GetEngineIO(uid);

    auto self = g->addInput("self.1");
    self->setType(mod.type());

    auto id_val = g->insertConstant(uid);

    std::vector<torch::jit::Value*> engine_inputs;
    engine_inputs.push_back(id_val);

    for (uint64_t i = 0; i < num_io.first; i++) {
        auto in_val = g->addInput("");
        in_val->setType(c10::TensorType::get());
        engine_inputs.push_back(in_val);
    }

    auto engine_node = g->create(c10::Symbol::fromQualString("trt::execute_engine"), torch::jit::ArrayRef<torch::jit::Value*>(engine_inputs), num_io.second);
    g->block()->appendNode(engine_node);

    if (engine_node->outputs().size() > 1) {
        auto return_tuple_node = g->createTuple(engine_node->outputs());
        g->block()->appendNode(return_tuple_node);
        g->registerOutput(return_tuple_node->outputs()[0]);
    } else {
        g->registerOutput(engine_node->outputs()[0]);
    }

    LOG_DEBUG(*g << "(AddEngineToGraph)\n");

    return;
}

bool CheckMethodOperatorSupport(const torch::jit::script::Module& mod,
                                std::string method_name) {
    // Go through Lowering to simplify graph and extract weight parameters
    auto graph_and_parameters = lowering::Lower(mod, method_name);

    auto g = graph_and_parameters.first;
    LOG_DEBUG(*g << "(CheckMethodOperatorSupport)\n");

    return conversion::VerifyConverterSupportForBlock(g->block());
}

std::string ConvertGraphToTRTEngine(const torch::jit::script::Module& mod,
                                    std::string method_name,
                                    ExtraInfo cfg) {

    // Go through Lowering to simplify graph and extract weight parameters
    auto graph_and_parameters = lowering::Lower(mod, method_name);

    auto convert_cfg = std::move(cfg.convert_info);
    auto g = graph_and_parameters.first;
    auto params = graph_and_parameters.second;
    auto named_params = conversion::get_named_params(g->inputs(), params);

    LOG_INFO(*g << "(CompileGraph)\n");

    auto engine = ConvertBlockToEngine(g->block(), convert_cfg, named_params);
    return std::move(engine);
}

torch::jit::script::Module CompileGraph(const torch::jit::script::Module& mod,
                                        ExtraInfo cfg) {
    // TODO: Should be doing a functional transform but need PR #31978
    // [jit] More robust mangling
    //torch::jit::script::Module new_mod = mod.clone();
    torch::jit::script::Module new_mod(mod._ivalue()->name() + "_trt");
    std::vector<std::shared_ptr<torch::jit::Graph>> graphs;
    for (const torch::jit::script::Method& method : mod.get_methods()) {
        auto engine = ConvertGraphToTRTEngine(mod, method.name(), cfg);
        auto new_g = std::make_shared<torch::jit::Graph>();
        AddEngineToGraph(new_mod, new_g, engine);
        auto new_method = new_mod._ivalue()->compilation_unit()->create_function(method.name(), new_g);
        auto schema = GenerateGraphSchema(new_mod, new_method->name(), new_g);
        new_mod.type()->addMethod(new_method);
        new_method->setSchema(schema);
    }

    return new_mod;
}

} // namespace core
} // namespace trtorch

