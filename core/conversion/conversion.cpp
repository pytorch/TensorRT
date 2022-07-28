#include "core/conversion/conversion.h"
#include <ATen/core/operator_name.h>
#include <torch/torch.h>
#include <sstream>
#include "c10/util/intrusive_ptr.h"
#include "core/conversion/conversionctx/ConversionCtx.h"
#include "core/conversion/converters/converter_util.h"
#include "core/conversion/converters/converters.h"
#include "core/conversion/evaluators/evaluators.h"
#include "core/conversion/tensorcontainer/TensorContainer.h"
#include "core/conversion/var/Var.h"
#include "core/util/prelude.h"
#include "core/util/trt_util.h"

namespace torch_tensorrt {
namespace core {
namespace conversion {

// Defined in core/conversion/conversion_ignorelist.cpp
bool isNodeConversionIgnored(const torch::jit::Node* n);

bool OpSupported(const torch::jit::Node* n) {
  return evaluators::shouldEvalAtConversionTime(n) || converters::node_is_convertable(n);
}

bool SpecialCaseSupport(const torch::jit::Node* n) {
  return n->kind() == torch::jit::prim::Loop || n->kind() == torch::jit::prim::If;
}

c10::optional<torch::jit::IValue> EvaluateNode(ConversionCtx* ctx, const torch::jit::Node* n, int level, int limit) {
  // Check to see if you can just go through and eval all of these AOT (saves
  // the recursion) Also probably a better way to deal with the two error cases;
  TORCHTRT_CHECK(
      level < limit,
      "Failed to evaluate node: " << *n << "Reason: Exceeded evaluation stack limit (limit=" << limit << ")");

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
    } else if (ctx->value_tensor_map.find(eval_in) != ctx->value_tensor_map.end()) {
      eval_args[eval_in] = ctx->value_tensor_map[eval_in];
    } else if (evaluators::shouldEvalAtConversionTime(eval_in->node())) {
      auto result = EvaluateNode(ctx, eval_in->node(), level++, limit);
      if (result) {
        // WARN: If the converter returns None then should pass through
        // but if repeated dep this section will get called each time
        auto val = result.value();
        if (val.isCustomClass()) {
          auto cont = val.toCustomClass<TensorContainer>();
          ctx->AssociateValueAndTensor(eval_in, cont->tensor());
          eval_args[eval_in] = ctx->value_tensor_map[eval_in];
        } else {
          ctx->AssociateValueAndIValue(eval_in, val);
          eval_args[eval_in] = &(ctx->evaluated_value_map[eval_in]);
        }
      }
    } else {
      TORCHTRT_THROW_ERROR(
          "Failed to evaluate node: " << *n << "Reason: Node inputs cannot be evaluated at conversion time\n"
                                      << "File a bug: https://www.github.com/NVIDIA/Torch-TensorRT/issues");
      return {};
    }
  }
  auto eval = evaluators::EvalNode(n, eval_args);
  return eval;
}

void AddLayer(ConversionCtx* ctx, const torch::jit::Node* n) {
  LOG_INFO(ctx->logger, "Adding Layer " << util::node_info(n) << " (ctx.AddLayer)");
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
        if (!eval.value().isTensor()) {
          LOG_DEBUG(ctx->logger, "Found the value to be: " << eval.value());
          if (eval.value().isTuple() && eval.value().toTuple()->elements().size() == 1) {
            eval.value() = {eval.value().toTuple()->elements()[0]};
          }
        } else {
          LOG_DEBUG(ctx->logger, "Found the value to be a tensor (shape " << eval.value().toTensor().sizes() << ')');
        }
        ctx->AssociateValueAndIValue(input, eval.value());
        node_args.push_back(&(ctx->evaluated_value_map[input]));
      } else {
        LOG_DEBUG(ctx->logger, "Found the value is None");
        node_args.push_back(Var());
      }
    } else {
      // Node input has not been converted yet or is a prim op
      TORCHTRT_THROW_ERROR(
          "Unable to retrieve all node inputs for node: "
          << util::node_info(n) << " (ctx.AddLayer)\nSpecifically failed to retrieve value for input: %"
          << input->debugName());
    }
  }

  if (n->inputs().size() != node_args.size()) {
    TORCHTRT_THROW_ERROR("Unable to retrieve all node inputs for node: " << *n);
  }

  auto schema = n->maybeSchema();
  TORCHTRT_CHECK(schema, "Unable to get schema for Node " << util::node_info(n) << " (conversion.AddLayer)");

  auto converter = converters::get_node_converter_for(schema);
  TORCHTRT_CHECK(
      converter,
      "Unable to convert node: "
          << util::node_info(n) << " (conversion.AddLayer)\nSchema: " << *schema << "\nConverter for " << schema->name()
          << " requested, but no such converter was found.\nIf you need a converter for this operator, you can try implementing one yourself\n"
          << "or request a converter: https://www.github.com/NVIDIA/Torch-TensorRT/issues");

  TORCHTRT_CHECK(
      converter(ctx, n, node_args),
      "Converter for " << *schema << " failed to convert node: " << util::node_info(n)
                       << "please report this error to https://www.github.com/NVIDIA/Torch-TensorRT/issues");
}

void AddInputs(ConversionCtx* ctx, c10::ArrayRef<const torch::jit::Value*> inputs, ConversionInfo& conversion_info) {
  std::unordered_map<const torch::jit::Value*, ir::Input>& input_specs = conversion_info.inputs;
  std::unordered_map<const torch::jit::Value*, std::vector<ir::Input>> collection_input_spec =
      conversion_info.collection_input_spec_map;

  std::vector<const torch::jit::Value*> input_tensors;
  for (auto in : inputs) {
    // Disregarding inputs that are not tensors
    //
    // Ex.
    // self.1:__torch__.alexnet -> ignored
    // input.1:Tensor -> used
    if (in->type()->isSubtypeOf(c10::TensorType::get()) &&
        ctx->evaluated_value_map.find(in) == ctx->evaluated_value_map.end()) {
      input_tensors.push_back(in);
    }
  }

  std::stringstream ss;
  ss << "Input Dimension Specs: {" << std::endl;
  for (auto i : input_specs) {
    ss << "    " << i.first->debugName() << " : " << i.second << ",";
  }
  ss << '}';
  auto dbg_str = ss.str();
  LOG_DEBUG(ctx->logger, dbg_str);

  auto profile = ctx->builder->createOptimizationProfile();

  for (auto input : input_tensors) {
    const torch::jit::Value* in = input;
    TORCHTRT_CHECK(
        input_specs.find(in) != input_specs.end() || collection_input_spec.find(in) != collection_input_spec.end(),
        "Cannot find an input spec associated with input: " << in->debugName());
    ir::Input spec;
    if (input_specs.find(in) != input_specs.end()) {
      spec = input_specs.find(in)->second;
    } else {
      spec = collection_input_spec.find(in)->second[0]; // assume input is tensor
    }
    // ir::Input& spec = input_specs.find(in)->second;

    std::string name = std::string("input_") + std::to_string(ctx->num_inputs);
    LOG_INFO(
        ctx->logger,
        "Adding Input " << in->debugName() << " (named: " << name << "): " << spec
                        << " in engine (conversion.AddInputs)");

    auto trt_in = ctx->net->addInput(name.c_str(), spec.dtype, spec.input_shape);
    TORCHTRT_CHECK(trt_in, "Failed to add input node: " << in->debugName() << " (conversion.AddInputs)");
    trt_in->setAllowedFormats(1U << static_cast<int>(spec.format));

    profile->setDimensions(trt_in->getName(), nvinfer1::OptProfileSelector::kMIN, spec.min);
    profile->setDimensions(trt_in->getName(), nvinfer1::OptProfileSelector::kOPT, spec.opt);
    profile->setDimensions(trt_in->getName(), nvinfer1::OptProfileSelector::kMAX, spec.max);

    if (spec.input_is_dynamic) {
      ctx->input_is_dynamic = true;
    }

    ctx->RecordNewITensor(in, trt_in);
    ctx->num_inputs += 1;
  }

  TORCHTRT_CHECK(
      profile->isValid(),
      "Optimization profile is invalid, please check the input range provided (conversion.AddInputs)");

  ctx->cfg->addOptimizationProfile(profile);
#if NV_TENSORRT_MAJOR > 7 || (NV_TENSORRT_MAJOR == 7 && NV_TENSORRT_MINOR >= 1)
  if (ctx->enabled_precisions.find(nvinfer1::DataType::kINT8) != ctx->enabled_precisions.end()) {
    ctx->cfg->setCalibrationProfile(profile);
  }
#endif
}

void MarkOutputs(ConversionCtx* ctx, at::ArrayRef<const torch::jit::Value*> outputs) {
  for (auto out : outputs) {
    auto it = ctx->value_tensor_map.find(out);
    if (it == ctx->value_tensor_map.end()) {
      if (ctx->evaluated_value_map.find(out) != ctx->evaluated_value_map.end()) {
        auto out_ivalue = ctx->evaluated_value_map[out];
        if (out_ivalue.isCustomClass()) {
          std::string name = std::string("output_") + std::to_string(ctx->num_outputs);
          auto output_container = out_ivalue.toCustomClass<TensorContainer>();
          nvinfer1::ITensor* out_tensor = output_container.get()->tensor();
          out_tensor->setName(name.c_str());
          ctx->net->markOutput(*out_tensor);
          LOG_INFO(
              ctx->logger, "Marking Output " << out->debugName() << " named " << name << " in engine (ctx.MarkOutput)");
          ctx->num_outputs += 1;
        } else if (out_ivalue.isTuple()) {
          TORCHTRT_THROW_ERROR("Tuple type. Only a single tensor or a TensorList type is supported.");
        } else if (out_ivalue.isList()) {
          TORCHTRT_THROW_ERROR("List type. Only a single tensor or a TensorList type is supported.");
        } else if (out_ivalue.isScalar()) {
          TORCHTRT_THROW_ERROR("Scalar type. Only a single tensor or a TensorList type is supported.");
        } else if (out_ivalue.isTensor()) {
          // prim::NumToTensor will go to here
          std::string name = std::string("output_") + std::to_string(ctx->num_outputs);
          auto out_tensor = converters::tensor_to_const(ctx, out_ivalue.toTensor(), "");
          out_tensor->setName(name.c_str());
          ctx->net->markOutput(*out_tensor);
          LOG_INFO(
              ctx->logger, "Marking Output " << out->debugName() << " named " << name << " in engine (ctx.MarkOutput)");
          ctx->num_outputs += 1;
        } else {
          TORCHTRT_THROW_ERROR("Unknown output type. Only a single tensor or a TensorList type is supported.");
        }
      }
    } else {
      bool setOutput = false;
      auto num_inputs = ctx->net->getNbInputs();
      auto out_tensor = it->second;
      std::string name = std::string("output_") + std::to_string(ctx->num_outputs);

      // Check if the output tensor is one of the inputs to the network. If so, apply an identity layer to it.
      for (int64_t i = 0; i < num_inputs; i++) {
        if (out_tensor == ctx->net->getInput(i)) {
          LOG_DEBUG(
              "One of the inputs named "
              << ctx->net->getInput(i)->getName()
              << " to the network is marked as an output tensor. Applying an identity layer and marking this tensor as output");
          auto id_out_tensor = converters::applyIdentityOp(ctx, out_tensor, name);
          ctx->net->markOutput(*id_out_tensor);
          setOutput = true;
        }
      }

      if (!setOutput) {
        out_tensor->setName(name.c_str());
        ctx->net->markOutput(*out_tensor);
      }
      LOG_INFO(
          ctx->logger, "Marking Output " << out->debugName() << " named " << name << " in engine (ctx.MarkOutput)");
      ctx->num_outputs += 1;
    }
  }
}

void AddParamsToCtxValueMap(ConversionCtx* ctx, ir::StaticParams& params) {
  for (auto p : params) {
    ctx->evaluated_value_map[p.first] = std::move(p.second);
  }
}

void EvaluateLoopBlock(ConversionCtx* ctx, const torch::jit::Node* n);

void MapIValues(
    ConversionCtx* ctx,
    c10::ArrayRef<const torch::jit::Value*> in_list,
    c10::ArrayRef<const torch::jit::Value*> out_list,
    int64_t in_offset,
    int64_t out_offset) {
  std::vector<std::pair<const torch::jit::Value*, const torch::jit::Value*>> input_output_pairs;
  std::transform(
      in_list.begin() + in_offset,
      in_list.end(),
      out_list.begin() + out_offset,
      std::back_inserter(input_output_pairs),
      [](auto in, auto out) { return std::make_pair(in, out); });

  for (auto p : input_output_pairs) {
    if (ctx->evaluated_value_map.find(p.first) != ctx->evaluated_value_map.end()) {
      auto input = ctx->evaluated_value_map[p.first];
      ctx->evaluated_value_map[p.second] = torch::jit::IValue(input);
    } else if (ctx->value_tensor_map.find(p.first) != ctx->value_tensor_map.end()) {
      auto input = ctx->value_tensor_map[p.first];
      ctx->value_tensor_map[p.second] = input;
    } else {
      TORCHTRT_THROW_ERROR(
          "Cannot find Value " << p.first->debugName() << " either evaluated values or tensor maps (MapIValues)");
    }
  }
}

void EvaluateConditionalBlock(ConversionCtx* ctx, const torch::jit::Node* n, bool contained_in_loop = false) {
  bool output_type_includes_tensor = false;
  for (auto o : n->outputs()) {
    if (o->type()->isSubtypeOf(c10::TensorType::get())) {
      output_type_includes_tensor = true;
    }
  }
  TORCHTRT_CHECK(
      !(contained_in_loop && output_type_includes_tensor),
      "Torch-TensorRT.TorchScript currently cannot compile conditionals within loops");

  auto condition = ctx->evaluated_value_map[n->input(0)].toBool();
  LOG_DEBUG(ctx->logger, "(Conditional Evaluation) Evaluating block " << (int)condition);
  auto b = condition ? n->blocks()[0] : n->blocks()[1];

  for (const auto bn : b->nodes()) {
    if (bn->kind() == torch::jit::prim::Loop) {
      EvaluateLoopBlock(ctx, bn);
    } else if (bn->kind() == torch::jit::prim::If) {
      EvaluateConditionalBlock(ctx, bn, contained_in_loop);
    } else if (evaluators::shouldEvalAtConversionTime(bn)) {
      auto eval = EvaluateNode(ctx, bn);
      if (!eval.value().isTensor()) {
        LOG_DEBUG(ctx->logger, "(Conditional Evaluation) Found the value to be: " << eval.value());
        if (eval.value().isTuple() && eval.value().toTuple()->elements().size() == 1) {
          eval.value() = {eval.value().toTuple()->elements()[0]};
        }
      } else {
        LOG_DEBUG(
            ctx->logger,
            "(Conditional Evaluation) Found the value to be a tensor (shape " << eval.value().toTensor().sizes()
                                                                              << ')');
      }
      ctx->AssociateValueAndIValue(bn->output(0), eval.value());
    } else if (converters::node_is_convertable(bn)) {
      AddLayer(ctx, bn);
    } else {
      TORCHTRT_THROW_ERROR(
          "Torch-TensorRT.TorchScript is unable to compile this conditional, a converter or evaluator is not available for node "
          << *bn);
    }
  }

  MapIValues(ctx, b->outputs(), n->outputs(), 0, 0);
}

// TODO: With functionalization pass we may be able to make this into a regular
// evaluator later
void EvaluateLoopBlock(ConversionCtx* ctx, const torch::jit::Node* n) {
  auto max_trip_count = ctx->evaluated_value_map[n->input(0)];
  auto start_cond = ctx->evaluated_value_map[n->input(1)];
  ctx->evaluated_value_map[n->blocks()[0]->inputs()[0]] = torch::jit::IValue(0);
  auto trip_count = ctx->evaluated_value_map[n->blocks()[0]->inputs()[0]];

  MapIValues(ctx, n->inputs(), n->outputs(), 2, 0);

  LOG_DEBUG(ctx->logger, "(Loop Evaluation) Evaluating loop " << *n);
  LOG_DEBUG(ctx->logger, "(Loop Evaluation) Max Trip Count: " << max_trip_count.toInt());
  LOG_DEBUG(ctx->logger, "(Loop Evaluation) Start Condition: " << start_cond.toBool());
  LOG_DEBUG(ctx->logger, "(Loop Evaluation) Current Trip Count: " << trip_count.toInt());

  while (start_cond.toBool() && trip_count.toInt() < max_trip_count.toInt()) {
    MapIValues(ctx, n->outputs(), n->blocks()[0]->inputs(), 0, 1);
    for (auto bn : n->blocks()[0]->nodes()) {
      if (bn->kind() == torch::jit::prim::Loop) {
        EvaluateLoopBlock(ctx, bn);
      } else if (bn->kind() == torch::jit::prim::If) {
        EvaluateConditionalBlock(ctx, bn, true);
      } else {
        TORCHTRT_CHECK(
            evaluators::shouldEvalAtConversionTime(bn),
            "Torch-TensorRT.TorchScript currently can only compile loops that are evaluatable at conversion time but node "
                << *bn << " cannot be evaluated.");
        auto eval = EvaluateNode(ctx, bn);
        if (!eval.value().isTensor()) {
          LOG_DEBUG(ctx->logger, "(Loop Evaluation) Found the value to be: " << eval.value());
        } else {
          LOG_DEBUG(
              ctx->logger,
              "(Loop Evaluation) Found the value to be a tensor (shape " << eval.value().toTensor().sizes() << ')');
        }
        ctx->AssociateValueAndIValue(bn->output(0), eval.value());
      }
    }

    MapIValues(ctx, n->blocks()[0]->outputs(), n->outputs(), 1, 0);
    start_cond = ctx->evaluated_value_map[n->blocks()[0]->outputs()[0]];
    auto new_trip_count = torch::jit::IValue(trip_count.toInt() + 1);
    trip_count.swap(new_trip_count);
    LOG_DEBUG(ctx->logger, "(Loop Evaluation) Condition: " << start_cond.toBool());
    LOG_DEBUG(ctx->logger, "(Loop Evaluation) Current Trip Count: " << trip_count.toInt());
  }
}

void ConvertBlockToNetDef(
    ConversionCtx* ctx,
    const torch::jit::Block* b,
    ConversionInfo& build_info,
    ir::StaticParams& static_params) {
  LOG_INFO(ctx->logger, "Converting Block");
  LOG_DEBUG(ctx->logger, *b->owningGraph());

  auto inputs = b->inputs();
  AddParamsToCtxValueMap(ctx, static_params);
  AddInputs(ctx, inputs, build_info);

  auto nodes = b->nodes();

  for (const auto n : nodes) {
    bool to_eval = evaluators::shouldEvalAtConversionTime(n);
    bool ignored = isNodeConversionIgnored(n);
    if (n->kind() == torch::jit::prim::Loop) {
      EvaluateLoopBlock(ctx, n);
    } else if (n->kind() == torch::jit::prim::If) {
      EvaluateConditionalBlock(ctx, n);
    } else if (to_eval) {
      auto eval = EvaluateNode(ctx, n);
      if (eval) {
        if (n->outputs().size() > 1) { // For ListUnpack scenario
          if (eval.value().isTuple()) {
            auto eval_list = eval.value().toTuple();
            TORCHTRT_CHECK(
                eval_list->elements().size() == n->outputs().size(),
                "Size of evaluated results: " << eval_list->elements().size()
                                              << " and node outputs size: " << n->outputs().size() << " must match.");
            for (size_t i = 0; i < eval_list->elements().size(); i++) {
              auto eval_output = eval_list.get()->elements()[i];
              if (eval_output.isCustomClass()) {
                auto container = eval_output.toCustomClass<TensorContainer>();
                auto tensor = container->tensor();
                LOG_DEBUG(
                    ctx->logger, "Found the evaluated value(s) to be an ITensor of shape: " << tensor->getDimensions());
                ctx->AssociateValueAndTensor(n->output(i), tensor);
              } else {
                LOG_DEBUG(
                    ctx->logger,
                    "Found the evaluated value(s) to be " << eval_output << " for node: " << util::node_info(n));
                ctx->AssociateValueAndIValue(n->output(i), eval_output);
              }
            }
          } else {
            TORCHTRT_THROW_ERROR("Unsupported return type for evaluated node");
          }
        } else if (eval.value().isCustomClass()) {
          auto container = eval.value().toCustomClass<TensorContainer>();
          auto tensor = container->tensor();
          LOG_DEBUG(ctx->logger, "Found the value to be an ITensor of shape: " << tensor->getDimensions());
          ctx->AssociateValueAndTensor(n->output(0), tensor);
        } else if (!eval.value().isTensor()) {
          LOG_DEBUG(ctx->logger, "Found the value to be: " << eval.value());
          ctx->AssociateValueAndIValue(n->output(0), eval.value());
        } else {
          LOG_DEBUG(ctx->logger, "Found the value to be a tensor (shape " << eval.value().toTensor().sizes() << ')');
          ctx->AssociateValueAndIValue(n->output(0), eval.value());
        }
      }
    } else if (!ignored) {
      // Should error out if something fails
      AddLayer(ctx, n);
    } else {
      std::string reason = "";
      if (to_eval) {
        reason += " (to be evaluated)";
      }
      if (ignored) {
        reason += " (explicitly ignored)";
      }
      LOG_DEBUG(ctx->logger, "Skipping Node: " << util::node_info(n) << reason);
    }
  }

  for (const auto n : nodes) {
    ctx->CheckLayerAddition(n);
  }

  auto outputs = b->outputs();
  MarkOutputs(ctx, outputs);
}

// Converts a already lowered block (blocks with no sub blocks) to
// a serialized TensorRT engine that can be deserialized and run

// Probably should consolidate these two functions
std::string ConvertBlockToEngine(
    const torch::jit::Block* b,
    ConversionInfo build_info,
    ir::StaticParams& static_params) {
  ConversionCtx ctx(build_info.engine_settings);
  ConvertBlockToNetDef(&ctx, b, build_info, static_params);
  std::string engine = ctx.SerializeEngine();
  return engine;
}

std::unordered_map<c10::OperatorName, std::string> GetUnsupportedOpsInBlock(const torch::jit::Block* b) {
  std::unordered_map<c10::OperatorName, std::string> unsupported_ops;
  for (const auto n : b->nodes()) {
    auto schema = n->maybeSchema();
    // Some ops like torch::jit::prim::Loop, torch::jit::prim::If, torch::jit::prim::DictConstruct don't have a schema
    // but they are supported. torch::jit::prim::DictConstruct is supported via fallback only
    if (!OpSupported(n) && !SpecialCaseSupport(n)) {
      if (schema) {
        std::stringstream ss;
        ss << *schema;
        unsupported_ops[schema->operator_name()] = ss.str();
      } else {
        std::stringstream ss;
        ss << util::node_info(n);
        // operator.overload is a filler name just to call the constructor.
        c10::OperatorName op(ss.str(), "operator.overload");
        unsupported_ops[op] = ss.str();
      }
    }

    for (const auto sub_b : n->blocks()) {
      auto sub_b_unsupported_ops = GetUnsupportedOpsInBlock(sub_b);
      unsupported_ops.insert(sub_b_unsupported_ops.begin(), sub_b_unsupported_ops.end());
    }
  }
  return unsupported_ops;
}

std::set<std::string> ConvertableOpsInBlock(const torch::jit::Block* b) {
  std::set<std::string> convertable_ops;
  for (const auto n : b->nodes()) {
    if (n->kind() == torch::jit::prim::Loop || n->kind() == torch::jit::prim::If ||
        converters::node_is_convertable(n)) {
      if (n->blocks().size() > 0) {
        for (const auto sub_b : n->blocks()) {
          auto sub_b_convertable_ops = ConvertableOpsInBlock(sub_b);
          convertable_ops.insert(sub_b_convertable_ops.begin(), sub_b_convertable_ops.end());
        }
      }
      if (converters::node_is_convertable(n)) {
        auto schema = n->maybeSchema();
        TORCHTRT_CHECK(
            schema, "Unable to get schema for Node " << util::node_info(n) << " (conversion.CheckForConvertableOps)");
        std::stringstream ss;
        ss << *schema;
        convertable_ops.insert(ss.str());
      }
    }
  }
  return convertable_ops;
}

bool OutputIsCollection(const torch::jit::Block* b) {
  for (auto out : b->outputs()) {
    if (out->type()->kind() == torch::jit::TypeKind::TupleType ||
        out->type()->kind() == torch::jit::TypeKind::ListType) {
      return true;
    }
  }
  return false;
}

bool VerifyConverterSupportForBlock(const torch::jit::Block* b, bool suppress_errors) {
  auto unsupported_ops = GetUnsupportedOpsInBlock(b);
  if (unsupported_ops.size() != 0) {
    std::stringstream unsupported_msg;
    unsupported_msg
        << "Method requested cannot be compiled end to end by Torch-TensorRT.TorchScript.\nUnsupported operators listed below:"
        << std::endl;
    for (auto s : unsupported_ops) {
      unsupported_msg << "  - " << s.second << std::endl;
    }

    if (!suppress_errors) {
      unsupported_msg
          << "You can either implement converters for these ops in your application or request implementation"
          << std::endl;
      unsupported_msg << "https://www.github.com/nvidia/Torch-TensorRT/issues" << std::endl;
      unsupported_msg << std::endl << "In Module:" << std::endl;

      LOG_ERROR(unsupported_msg.str());
    } else {
      LOG_INFO(unsupported_msg.str());
    }

    std::unordered_map<std::string, std::unordered_set<std::string>> unsupported_node_locations;
    for (const auto n : b->nodes()) {
      auto schema = n->maybeSchema();
      if (schema) {
        for (const auto& x : unsupported_ops) {
          if (x.first == schema->operator_name()) {
            auto loc = unsupported_node_locations.find(x.second);
            if (loc == unsupported_node_locations.end()) {
              unsupported_node_locations.insert({x.second, {torch_tensorrt::core::util::GetPyTorchSourceCode(n)}});
            } else {
              loc->second.insert(torch_tensorrt::core::util::GetPyTorchSourceCode(n));
            }
          }
        }
      }
    }

    for (const auto& type : unsupported_node_locations) {
      std::stringstream traceback;
      traceback << "Unsupported operator: " << type.first << std::endl;
      for (const auto& str : type.second) {
        traceback << str;
      }

      auto tb_str = traceback.str();
      if (!suppress_errors) {
        LOG_ERROR(tb_str);
      } else {
        LOG_DEBUG(tb_str);
      }
    }

    return false;
  }

  if (ConvertableOpsInBlock(b).size() == 0) {
    std::stringstream unsupported_msg;
    unsupported_msg
        << "Method requested cannot be compiled by Torch-TensorRT.TorchScript.\nThere is no work to be done since the resulting compiled program will contain an engine that is empty."
        << std::endl;
    unsupported_msg
        << "This may be because there are no operators that can be added to the TensorRT graph or all operators have a resolved compile time value."
        << std::endl;
    if (!suppress_errors) {
      LOG_ERROR(unsupported_msg.str());
    }
    return false;
  }

  else {
    return true;
  }
}

} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
