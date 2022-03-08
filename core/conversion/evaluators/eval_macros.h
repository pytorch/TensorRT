#pragma once

#include "core/conversion/evaluators/evaluators.h"

#define DEFINE_GENERIC_TWO_INPUT_EVALUATOR(name, node_kind, operation, schemas)                        \
  auto name##_registrations TORCHTRT_UNUSED = RegisterNodeEvaluators().evaluator(                      \
      {c10::Symbol::fromQualString(node_kind),                                                         \
       [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {              \
         if (args.at(n->input(0)).IValue()->isInt()) {                                                 \
           auto a = args.at(n->input(0)).unwrapToInt();                                                \
           if (args.at(n->input(1)).IValue()->isInt()) {                                               \
             auto b = args.at(n->input(1)).unwrapToInt();                                              \
             return operation;                                                                         \
           } else if (args.at(n->input(1)).IValue()->isDouble()) {                                     \
             auto b = args.at(n->input(1)).unwrapToDouble();                                           \
             return operation;                                                                         \
           } else if (args.at(n->input(1)).IValue()->isBool()) {                                       \
             auto b = args.at(n->input(1)).unwrapToBool();                                             \
             return operation;                                                                         \
           } else {                                                                                    \
             TORCHTRT_THROW_ERROR(                                                                     \
                 "Unimplemented data type for "                                                        \
                 << node_kind << " evaluator b arg:" << args.at(n->input(1)).IValue()->type()->str()); \
             return {};                                                                                \
           }                                                                                           \
         } else if (args.at(n->input(0)).IValue()->isDouble()) {                                       \
           auto a = args.at(n->input(0)).unwrapToDouble();                                             \
           if (args.at(n->input(1)).IValue()->isInt()) {                                               \
             auto b = args.at(n->input(1)).unwrapToInt();                                              \
             return operation;                                                                         \
           } else if (args.at(n->input(1)).IValue()->isDouble()) {                                     \
             auto b = args.at(n->input(1)).unwrapToDouble();                                           \
             return operation;                                                                         \
           } else if (args.at(n->input(1)).IValue()->isBool()) {                                       \
             auto b = args.at(n->input(1)).unwrapToBool();                                             \
             return operation;                                                                         \
           } else {                                                                                    \
             TORCHTRT_THROW_ERROR(                                                                     \
                 "Unimplemented data type for "                                                        \
                 << node_kind << " evaluator b arg:" << args.at(n->input(1)).IValue()->type()->str()); \
             return {};                                                                                \
           }                                                                                           \
         } else if (args.at(n->input(0)).IValue()->isBool()) {                                         \
           auto a = args.at(n->input(0)).unwrapToBool();                                               \
           if (args.at(n->input(1)).IValue()->isInt()) {                                               \
             auto b = args.at(n->input(1)).unwrapToInt();                                              \
             return operation;                                                                         \
           } else if (args.at(n->input(1)).IValue()->isDouble()) {                                     \
             auto b = args.at(n->input(1)).unwrapToDouble();                                           \
             return operation;                                                                         \
           } else if (args.at(n->input(1)).IValue()->isBool()) {                                       \
             auto b = args.at(n->input(1)).unwrapToBool();                                             \
             return operation;                                                                         \
           } else {                                                                                    \
             TORCHTRT_THROW_ERROR(                                                                     \
                 "Unimplemented data type for "                                                        \
                 << node_kind << " evaluator b arg:" << args.at(n->input(1)).IValue()->type()->str()); \
             return {};                                                                                \
           }                                                                                           \
         } else if (args.at(n->input(0)).IValue()->isString()) {                                       \
           auto a = args.at(n->input(0)).unwrapToString();                                             \
           if (args.at(n->input(1)).IValue()->isString()) {                                            \
             auto b = args.at(n->input(1)).unwrapToString();                                           \
             return operation;                                                                         \
           } else {                                                                                    \
             TORCHTRT_THROW_ERROR(                                                                     \
                 "Unimplemented data type for "                                                        \
                 << node_kind << " evaluator b arg:" << args.at(n->input(1)).IValue()->type()->str()); \
             return {};                                                                                \
           }                                                                                           \
         } else {                                                                                      \
           TORCHTRT_THROW_ERROR(                                                                       \
               "Unimplemented data type for "                                                          \
               << node_kind << " evaluator a arg: " << args.at(n->input(0)).IValue()->type()->str());  \
           return {};                                                                                  \
         }                                                                                             \
       },                                                                                              \
       EvalOptions().validSchemas(schemas)});

#define DEFINE_ARITHMATIC_TWO_INPUT_EVALUATOR(name, node_kind, operation, schemas)                     \
  auto name##_registrations TORCHTRT_UNUSED = RegisterNodeEvaluators().evaluator(                      \
      {c10::Symbol::fromQualString(node_kind),                                                         \
       [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {              \
         if (args.at(n->input(0)).IValue()->isInt()) {                                                 \
           auto a = args.at(n->input(0)).unwrapToInt();                                                \
           if (args.at(n->input(1)).IValue()->isInt()) {                                               \
             auto b = args.at(n->input(1)).unwrapToInt();                                              \
             return operation;                                                                         \
           } else if (args.at(n->input(1)).IValue()->isDouble()) {                                     \
             auto b = args.at(n->input(1)).unwrapToDouble();                                           \
             return operation;                                                                         \
           } else if (args.at(n->input(1)).IValue()->isBool()) {                                       \
             auto b = args.at(n->input(1)).unwrapToBool();                                             \
             return operation;                                                                         \
           } else {                                                                                    \
             TORCHTRT_THROW_ERROR(                                                                     \
                 "Unimplemented data type for "                                                        \
                 << node_kind << " evaluator b arg:" << args.at(n->input(1)).IValue()->type()->str()); \
             return {};                                                                                \
           }                                                                                           \
         } else if (args.at(n->input(0)).IValue()->isDouble()) {                                       \
           auto a = args.at(n->input(0)).unwrapToDouble();                                             \
           if (args.at(n->input(1)).IValue()->isInt()) {                                               \
             auto b = args.at(n->input(1)).unwrapToInt();                                              \
             return operation;                                                                         \
           } else if (args.at(n->input(1)).IValue()->isDouble()) {                                     \
             auto b = args.at(n->input(1)).unwrapToDouble();                                           \
             return operation;                                                                         \
           } else if (args.at(n->input(1)).IValue()->isBool()) {                                       \
             auto b = args.at(n->input(1)).unwrapToBool();                                             \
             return operation;                                                                         \
           } else {                                                                                    \
             TORCHTRT_THROW_ERROR(                                                                     \
                 "Unimplemented data type for "                                                        \
                 << node_kind << " evaluator b arg:" << args.at(n->input(1)).IValue()->type()->str()); \
             return {};                                                                                \
           }                                                                                           \
         } else {                                                                                      \
           TORCHTRT_THROW_ERROR(                                                                       \
               "Unimplemented data type for "                                                          \
               << node_kind << " evaluator a arg: " << args.at(n->input(0)).IValue()->type()->str());  \
           return {};                                                                                  \
         }                                                                                             \
       },                                                                                              \
       EvalOptions().validSchemas(schemas)});

#define DEFINE_TWO_INPUT_SIMPLE_EVALUATOR(node_kind, node_name, operation, type, schemas) \
  auto node_kind##_registrations TORCHTRT_UNUSED = RegisterNodeEvaluators().evaluator(    \
      {c10::Symbol::fromQualString(node_name),                                            \
       [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> { \
         auto a = args.at(n->input(0)).unwrapTo<type>();                                  \
         auto b = args.at(n->input(1)).unwrapTo<type>();                                  \
         return operation;                                                                \
       },                                                                                 \
       EvalOptions().validSchemas(schemas)});
