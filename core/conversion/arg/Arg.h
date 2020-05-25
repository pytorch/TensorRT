#pragma once

#include <string>
#include <map>

#include "torch/csrc/jit/runtime/custom_operator.h"
#include "ATen/core/function_schema.h"

#include "core/util/prelude.h"
#include "core/conversion/conversionctx/ConversionCtx.h"

namespace trtorch {
namespace core {
namespace conversion {

class Arg {
public:
  enum Type {
    kITensor,
    kIValue,
    kNone
  };

  Arg();
  Arg(const torch::jit::IValue* p);
  Arg(nvinfer1::ITensor* p);
  Arg(const Arg& a);
  Arg& operator=(const Arg& a);
  Arg& operator=(const torch::jit::IValue* in);
  Arg& operator=(nvinfer1::ITensor* in);
  const torch::jit::IValue* IValue() const;
  nvinfer1::ITensor* ITensor() const;

  //TODO: Can we consolidate this in a way that prevents requesting invalid types
  at::Tensor unwrapToTensor(at::Tensor default_val);
  at::Tensor unwrapToTensor();
  int64_t unwrapToInt(int64_t default_val);
  int64_t unwrapToInt();
  double unwrapToDouble(double default_val);
  double unwrapToDouble();
  bool unwrapToBool(bool default_val);
  bool unwrapToBool();
  c10::Scalar unwrapToScalar(c10::Scalar default_val);
  c10::Scalar unwrapToScalar();
  c10::List<int64_t> unwrapToIntList(c10::List<int64_t> default_val);
  c10::List<int64_t> unwrapToIntList();
  c10::List<double> unwrapToDoubleList(c10::List<double> default_val);
  c10::List<double> unwrapToDoubleList();
  c10::List<bool> unwrapToBoolList(c10::List<bool> default_val);
  c10::List<bool> unwrapToBoolList();
  c10::List<at::Tensor> unwrapToTensorList(c10::List<at::Tensor> default_val);
  c10::List<at::Tensor> unwrapToTensorList();

  template<typename T>
  T unwrapTo(T default_val);
  template<typename T>
  T unwrapTo();

  bool isIValue() const;
  bool isITensor() const;
  bool isNone() const;
  Arg::Type type() const;
  std::string type_name() const;
private:
  union ArgContainer {
    const torch::jit::IValue* ivalue;
    nvinfer1::ITensor* tensor;
    void* none;
  };

  ArgContainer ptr_;
  Type type_;
};

} // namespace conversion
} // namespace core
} // namespace trtorch

#include "core/conversion/arg/Arg_inl.h"