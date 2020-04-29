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

    for (auto o : engine_node->outputs()) {
        g->registerOutput(o);
    }

    LOG_DEBUG(*g << "(AddEngineToGraph)\n");

    return;
}

bool CheckMethodOperatorSupport(const torch::jit::script::Module& mod,
                                std::string method_name) {
    auto g = mod.get_method(method_name).graph();
    // Go through PyTorch Lowering to simplify graph and extract weight parameters
    auto graph_and_parameters = torch::jit::LowerGraph(*g, mod._ivalue());

    g = graph_and_parameters.first;

    // Go through TRTorch Lowering to reformat graph to be conversion friendly
    // and also segment for accelerators and executors (TRT-DLA, TRT-GPU, PYT)
    lowering::LowerGraph(g);

    auto params = graph_and_parameters.second;
    auto named_params = conversion::get_named_params(g->inputs(), params);
    LOG_DEBUG(*g << "(CheckMethodOperatorSupport)\n");

    // Is this necessary?
    lowering::LowerBlock(g->block());

    return conversion::VerifyConverterSupportForBlock(g->block());
}

std::string ConvertGraphToTRTEngine(const torch::jit::script::Module& mod,
                                    std::string method_name,
                                    ExtraInfo cfg) {
    auto convert_cfg = std::move(cfg.convert_info);

    auto g = mod.get_method(method_name).graph();
    // Go through PyTorch Lowering to simplify graph and extract weight parameters
    auto graph_and_parameters = torch::jit::LowerGraph(*g, mod._ivalue());

    g = graph_and_parameters.first;

    // Go through TRTorch Lowering to reformat graph to be conversion friendly
    // and also segment for accelerators and executors (TRT-DLA, TRT-GPU, PYT)
    lowering::LowerGraph(g);

    auto params = graph_and_parameters.second;
    auto named_params = conversion::get_named_params(g->inputs(), params);
    LOG_INFO(*g << "(CompileGraph)\n");

    // Is this necessary?
    lowering::LowerBlock(g->block());
    auto engine = ConvertBlockToEngine(g->block(), convert_cfg, named_params);
    return std::move(engine);
}

torch::jit::script::Module CompileGraph(const torch::jit::script::Module& mod,
                                        ExtraInfo cfg) {
    // TODO: Should be doing a functional transform but need PR #31978
    // [jit] More robust mangling
    // torch::jit::script::Module new_mod = mod.clone();
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

