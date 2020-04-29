#include <sstream>

#include "core/util/prelude.h"
#include "core/conversion/conversion.h"
#include "core/conversion/converters/converters.h"
#include "core/conversion/evaluators/evaluators.h"
#include "core/conversion/conversionctx/ConversionCtx.h"

namespace trtorch {
namespace core {
namespace conversion {

// Defined in core/conversion/conversion_blacklist.cpp
bool isNodeConversionBlacklisted(const torch::jit::Node* n);

bool OpSupported(const torch::jit::Node* n) {
    return evaluators::shouldEvalAtConversionTime(n) || converters::node_is_convertable(n);
}

c10::optional<torch::jit::IValue> EvaluateNode(ConversionCtx* ctx, const torch::jit::Node* n, int level=0, int limit=10) {
    // Check to see if you can just go through and eval all of these AOT (saves the recursion)
    // Also probably a better way to deal with the two error cases;
    TRTORCH_CHECK(level < limit, "Failed to evaluate node: " << *n              \
                           << "Reason: Exceeded evaluation stack limit (limit=" \
                           << limit << ")");

    LOG_DEBUG(ctx->logger, "Evaluating " << util::node_info(n));
    evaluators::kwargs eval_args;
    for (auto eval_in : n->inputs()) {
        if (eval_args.find(eval_in) != eval_args.end()) {
            // No need to evaluate nodes that already have been entered in the
            // args dict
            continue;
        }
        if (ctx->evaluated_value_map.find(eval_in) != ctx->evaluated_value_map.end()) {
            eval_args[eval_in] = &(ctx->evaluated_value_map[eval_in]);
        } else if (evaluators::shouldEvalAtConversionTime(eval_in->node())) {
            auto result = EvaluateNode(ctx, eval_in->node(), level++, limit);
            if (result) {
                // WARN: If the converter returns None then should pass through
                // but if repeated dep this section will get called each time
                ctx->evaluated_value_map[eval_in] = std::move(result.value());
                eval_args[eval_in] = &(ctx->evaluated_value_map[eval_in]);
            }
        } else {
            TRTORCH_THROW_ERROR("Failed to evaluate node: " << *n                               \
                                << "Reason: Node inputs cannot be evaluated at conversion time" \
                                << "File a bug: https://www.github.com/NVIDIA/TRTorch/issues");
            return {};
        }
    }
    auto eval = evaluators::EvalNode(n, eval_args);
    return eval;
}

void AddLayer(ConversionCtx* ctx, const torch::jit::Node* n) {
    LOG_INFO(ctx->logger,
             "Adding Layer " << util::node_info(n) << " (ctx.AddLayer)");
    converters::args node_args;
    for (auto input : n->inputs()) {
        auto input_node = input->node();
        if (ctx->value_tensor_map.find(input) != ctx->value_tensor_map.end()) {
            // Node input already has a coresponding tensor
            LOG_DEBUG(ctx->logger, "Node input is an already converted tensor");
            node_args.push_back(ctx->value_tensor_map[input]);
        } else if (ctx->evaluated_value_map.find(input) != ctx->evaluated_value_map.end()) {
            // Node input is a value that has already been evaluated
            LOG_DEBUG(ctx->logger, "Node input is a result of a previously evaluated value");
            node_args.push_back(&(ctx->evaluated_value_map[input]));
        } else if (evaluators::shouldEvalAtConversionTime(input_node)) {
            // Node input is a node that needs to be evaluated before
            // the node can be converted
            LOG_DEBUG(ctx->logger, "Node input is a value that needs to be evaluated");
            auto eval = EvaluateNode(ctx, input_node);
            if (eval) {
                LOG_DEBUG(ctx->logger, "Found the value to be: " << eval.value());
                ctx->evaluated_value_map[input] = std::move(eval.value());
                node_args.push_back(&(ctx->evaluated_value_map[input]));
            } else {
                LOG_DEBUG(ctx->logger, "Found the value is None");;
                node_args.push_back(converters::Arg());
            }
        } else {
            // Node input has not been converted yet or is a prim op
            TRTORCH_THROW_ERROR("Unable to retrieve all node inputs for node: " \
                                << util::node_info(n) << " (ctx.AddLayer)\nSpecifically failed to retrieve value for input: " \
                                << *input_node);
        }
    }

    if (n->inputs().size() != node_args.size()) {
        TRTORCH_THROW_ERROR("Unable to retrieve all node inputs for node: " << *n);
    }


    auto schema = n->maybeSchema();
    TRTORCH_CHECK(schema, "Unable to get schema for Node " << util::node_info(n) \
                  << " (conversion.AddLayer)");

    auto converter = converters::get_node_converter_for(schema);
    TRTORCH_CHECK(converter, "Unable to convert node: " << util::node_info(n) \
                  << " (conversion.AddLayer)\nSchema: " << *schema
                  << "\nConverter for " << schema->name()
                  << " requested, but no such converter was found.\nIf you need a converter for this operator, you can try implementing one yourself\n"
                  << "or request a converter: https://www.github.com/NVIDIA/TRTorch/issues");

    TRTORCH_CHECK(converter(ctx, n, node_args),
                  "Converter for " << *schema << " failed to convert node: "
                  << util::node_info(n) << "please report this error to https://www.github.com/NVIDIA/TRTorch/issues");
}

void AddInputs(ConversionCtx* ctx,
                at::ArrayRef<const torch::jit::Value*> inputs,
                std::vector<InputRange>& input_dims) {

    auto type_lut = torch::jit::script::string_to_type_lut();
    std::vector<const torch::jit::Value*> input_tensors;
    for (auto in : inputs) {
        // Disregarding inputs that are not tensors
        //
        // Ex.
        // self.1:__torch__.alexnet -> ignored
        // input.1:Tensor -> used
        auto pt_type = in->type();
        if (pt_type == type_lut["Tensor"]) {
            input_tensors.push_back(in);
        }
    }

    TRTORCH_CHECK(input_tensors.size() == input_dims.size(),
                  "Expected dimension specifications for all input tensors" \
                  << ", but found " << input_tensors.size()                 \
                  << " input tensors and "                                  \
                  << input_dims.size() << " dimension specs (conversion.AddInputs)");

    auto profile = ctx->builder->createOptimizationProfile();

    for (size_t i = 0; i < input_tensors.size(); i++) {
        auto in = input_tensors[i];
        auto dims = input_dims[i];
        LOG_INFO(ctx->logger,
                 "Adding Input " << in->debugName()  \
                 << " (conversion.AddInputs)");
        LOG_DEBUG(ctx->logger, "Input shape set to " << dims.input_shape);
        auto trt_in = ctx->net->addInput(in->debugName().c_str(),
                                         ctx->input_type, dims.input_shape);
        TRTORCH_CHECK(trt_in, "Failed to add input node: " << in->debugName() << " (conversion.AddInputs)");

        profile->setDimensions(trt_in->getName(), nvinfer1::OptProfileSelector::kMIN, dims.min);
        profile->setDimensions(trt_in->getName(), nvinfer1::OptProfileSelector::kOPT, dims.opt);
        profile->setDimensions(trt_in->getName(), nvinfer1::OptProfileSelector::kMAX, dims.max);

        ctx->value_tensor_map[in] = trt_in;
    }

    TRTORCH_CHECK(profile->isValid(), "Optimization profile is invalid, please check the input range provided (conversion.AddInputs)");

    ctx->cfg->addOptimizationProfile(profile);
}

void MarkOutputs(ConversionCtx* ctx, at::ArrayRef<const torch::jit::Value*> outputs) {
    for (auto out : outputs) {
        auto it = ctx->value_tensor_map.find(out);
        // Leaves the potential for unused outputs to be populated with nullptr "safely"
        TRTORCH_CHECK(it != ctx->value_tensor_map.end() && it->second,
                      "No corresponding output TRT Tensor found for TorchScript output: " << out->debugName());
        auto out_tensor = it->second;
        ctx->net->markOutput(*out_tensor);
        LOG_INFO(ctx->logger,
                 "Marking Output " << out->debugName() << " (ctx.MarkOutput)");
    }
}

void AddParamsToCtxValueMap(ConversionCtx* ctx, GraphParams& params) {
    for (auto p : params) {
        ctx->evaluated_value_map[p.first] = torch::jit::IValue(p.second.clone());
    }
}

void ConvertBlockToNetDef(ConversionCtx* ctx, const torch::jit::Block* b, ConversionInfo build_info, GraphParams& static_params) {
     LOG_INFO(ctx->logger, "Converting Block");

    auto inputs = b->inputs();
    AddParamsToCtxValueMap(ctx, static_params);
    AddInputs(ctx, inputs, build_info.input_ranges);

    auto nodes = b->nodes();

    for (const auto n : nodes) {
        bool to_eval = evaluators::shouldEvalAtConversionTime(n);
        bool blacklisted = isNodeConversionBlacklisted(n);
        if (!to_eval && !blacklisted) {
            // Should error out if something fails
            AddLayer(ctx, n);
        } else {
            std::string reason = "";
            if (to_eval) {
                reason += " (to be evaluated)";
            }
            if (blacklisted) {
                reason += " (explicitly blacklisted)";
            }
            LOG_DEBUG(ctx->logger,
                      "Skipping Node: " << util::node_info(n) << reason);
        }
    }

    for (const auto n : nodes) {
        if (converters::node_is_convertable(n)) {
            ctx->CheckLayerAddition(n);
        }
    }

    auto outputs = b->outputs();
    MarkOutputs(ctx, outputs);
}

// Converts a already lowered block (blocks with no sub blocks) to
// a serialized TensorRT engine that can be deserialized and run

// Probably should consolidate these two functions
std::string ConvertBlockToEngine(const torch::jit::Block* b, ConversionInfo build_info, GraphParams& static_params) {
    ConversionCtx ctx(build_info.engine_settings);
    ConvertBlockToNetDef(&ctx, b, build_info, static_params);
    std::string engine = ctx.SerializeEngine();
    return engine;
}

bool VerifyConverterSupportForBlock(const torch::jit::Block* b) {
    bool supported = true;
    std::set<std::string> unsupported_ops;
    for (const auto n : b->nodes()) {
        if (!OpSupported(n)) {
            auto schema = n->maybeSchema();
            TRTORCH_CHECK(schema, "Unable to get schema for Node " << util::node_info(n) \
                                    << " (conversion.VerifyCoverterSupportForBlock");
            std::stringstream ss;
            ss << *schema;
            unsupported_ops.insert(ss.str());
            supported = false;
        }
    }

    if (!supported) {
        std::stringstream unsupported_msg;
         unsupported_msg << "Method requested cannot be compiled by TRTorch.\nUnsupported operators listed below:" << std::endl;
        for (auto s : unsupported_ops) {
            unsupported_msg << "  -  " << s << std::endl;
        }
        unsupported_msg << "You can either implement converters for these ops in your application or request implementation" << std::endl;
        unsupported_msg <<  "https://www.github.com/nvidia/TRTorch/issues" << std::endl;
        LOG_ERROR(unsupported_msg.str());
    }
    return supported;
}

} // namespace conversion
} // namespace core
} // namespace trtorch
