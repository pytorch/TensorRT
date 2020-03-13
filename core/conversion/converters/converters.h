#pragma once

#include <string>
#include <map>

#include "torch/csrc/jit/custom_operator.h"
#include "ATen/core/function_schema.h"

#include "core/conversion/conversionctx/ConversionCtx.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {

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
        
    

typedef std::vector<Arg> args;
typedef std::function<bool(ConversionCtx*, const torch::jit::Node*, args&)> OpConverter;
struct ConversionPattern {    
    std::string signature;
    OpConverter converter;
};

void register_node_converter(torch::jit::FunctionSchema* signature, OpConverter& converter);
void register_node_converter(std::string signature, OpConverter& converter);
void register_node_converter(ConversionPattern p);

class RegisterNodeConversionPatterns {
public:
    RegisterNodeConversionPatterns() = default;
    RegisterNodeConversionPatterns(const RegisterNodeConversionPatterns&) = delete;
    RegisterNodeConversionPatterns& operator=(const RegisterNodeConversionPatterns&) = delete;
    RegisterNodeConversionPatterns(RegisterNodeConversionPatterns&&) noexcept;
    RegisterNodeConversionPatterns& operator=(RegisterNodeConversionPatterns&&) noexcept;
    RegisterNodeConversionPatterns&& pattern(ConversionPattern p) &&;
};

bool node_is_convertable(const torch::jit::Node* n);
OpConverter get_node_converter_for(const torch::jit::FunctionSchema* signature);

struct Weights {
    //TODO: Rebuild this in a way that makes sense for more than just conv2/3D and linear
    nvinfer1::Weights data;
    nvinfer1::Dims kernel_shape;
    nvinfer1::Dims shape;
    int64_t num_input_maps;
    int64_t num_output_maps;

    Weights();
    Weights(ConversionCtx* ctx, at::Tensor t);
    Weights(ConversionCtx* ctx, float val);
    friend std::ostream& operator<<(std::ostream& os, const Weights& w); 
};

inline nvinfer1::ITensor* tensor_to_const(ConversionCtx* ctx, at::Tensor t) {
    auto t_weights = Weights(ctx, t);
    return ctx->net->addConstant(t_weights.shape, t_weights.data)->getOutput(0);
}

} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
