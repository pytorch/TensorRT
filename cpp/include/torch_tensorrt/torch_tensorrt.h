/*
 * Copyright (c) NVIDIA Corporation.
 * All rights reserved.
 *
 * This library is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "torch/custom_class.h"

#include "torch_tensorrt/macros.h"

// Just include the .h?
#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace torch {
namespace jit {
struct Graph;
struct Module;
} // namespace jit
} // namespace torch

namespace c10 {
enum class DeviceType : int8_t;
enum class ScalarType : int8_t;
template <class>
class ArrayRef;
} // namespace c10

namespace nvinfer1 {
class IInt8Calibrator;
}
#endif // DOXYGEN_SHOULD_SKIP_THIS

namespace torch_tensorrt {
/**
 * Supported Data Types that can be used with TensorRT engines
 *
 * This class is compatable with c10::DataTypes (but will check for TRT
 * support) so there should not be a reason that you need to use this type
 * explictly.
 */
class DataType {
 public:
  /**
   * Underlying enum class to support the DataType Class
   *
   * In the case that you need to use the DataType class itself, interface
   * using this enum vs. normal instatination
   *
   * ex. torch_tensorrt::DataType type = DataType::kFloat;
   */
  enum Value : int8_t {
    /// FP32
    kFloat,
    /// FP16
    kHalf,
    /// INT8
    kChar,
    /// INT
    kInt,
    /// Bool
    kBool,
    /// Sentinel value
    kUnknown
  };

  /**
   * @brief Construct a new Data Type object
   *
   */
  DataType() = default;
  /**
   * @brief DataType constructor from enum
   *
   */
  constexpr DataType(Value t) : value(t) {}
  /**
   * @brief Construct a new Data Type object from torch type enums
   *
   * @param t
   */
  TORCHTRT_API DataType(c10::ScalarType t);
  /**
   * @brief Get the enum value of the DataType object
   *
   * @return Value
   */
  operator Value() const {
    return value;
  }
  explicit operator bool() = delete;
  /**
   * @brief Comparision operator for DataType
   *
   * @param other
   * @return true
   * @return false
   */
  constexpr bool operator==(DataType other) const {
    return value == other.value;
  }
  /**
   * @brief Comparision operator for DataType
   *
   * @param other
   * @return true
   * @return false
   */
  constexpr bool operator==(DataType::Value other) const {
    return value == other;
  }
  /**
   * @brief Comparision operator for DataType
   *
   * @param other
   * @return true
   * @return false
   */
  constexpr bool operator!=(DataType other) const {
    return value != other.value;
  }
  /**
   * @brief Comparision operator for DataType
   *
   * @param other
   * @return true
   * @return false
   */
  constexpr bool operator!=(DataType::Value other) const {
    return value != other;
  }

 private:
  friend TORCHTRT_API std::ostream& operator<<(std::ostream& os, const DataType& dtype);
  Value value;
};

/**
 * @brief Setting data structure for Target device
 */
struct Device {
  /**
   * Supported Device Types that can be used with TensorRT engines
   *
   * This class is compatable with c10::DeviceTypes (but will check for TRT
   * support) but the only applicable value is at::kCUDA, which maps to
   * DeviceType::kGPU
   *
   * To use the DataType class itself, interface using the enum vs. normal
   * instatination
   *
   * ex. torch_tensorrt::DeviceType type = DeviceType::kGPU;
   */
  class DeviceType {
   public:
    /**
     * Underlying enum class to support the DeviceType Class
     *
     * In the case that you need to use the DeviceType class itself, interface
     * using this enum vs. normal instatination
     *
     * ex. torch_tensorrt::DeviceType type = DeviceType::kGPU;
     */
    enum Value : int8_t {
      /// Target GPU to run engine
      kGPU,
      /// Target DLA to run engine
      kDLA,
    };

    /**
     * @brief Construct a new Device Type object
     *
     */
    DeviceType() = default;
    /**
     * @brief Construct a new Device Type object from internal enum
     *
     */
    constexpr DeviceType(Value t) : value(t) {}
    /**
     * @brief Construct a new Device Type object from torch device enums
     * Note: The only valid value is torch::kCUDA (torch::kCPU is not supported)
     *
     * @param t
     */
    DeviceType(c10::DeviceType t);
    /**
     * @brief Get the internal value from the Device object
     *
     * @return Value
     */
    operator Value() const {
      return value;
    }
    explicit operator bool() = delete;
    /**
     * @brief Comparison operator for DeviceType
     *
     * @param other
     * @return true
     * @return false
     */
    constexpr bool operator==(DeviceType other) const {
      return value == other.value;
    }
    /**
     * @brief Comparison operator for DeviceType
     *
     * @param other
     * @return true
     * @return false
     */
    constexpr bool operator!=(DeviceType other) const {
      return value != other.value;
    }

   private:
    Value value;
  };

  /**
   * @brief Setting data structure for device
   * This struct will hold Target device related parameters such as device_type, gpu_id, dla_core
   */
  DeviceType device_type;

  /*
   * Target gpu id
   */
  int64_t gpu_id;

  /*
   * When using DLA core on NVIDIA AGX platforms gpu_id should be set as Xavier device
   */
  int64_t dla_core;

  /**
   * (Only used when targeting DLA (device))
   * Lets engine run layers on GPU if they are not supported on DLA
   */
  bool allow_gpu_fallback;

  /**
   * Constructor for Device structure
   */
  Device() : device_type(DeviceType::kGPU), gpu_id(0), dla_core(0), allow_gpu_fallback(false) {}
};

/**
 * Emum for selecting engine capability
 */
enum class EngineCapability : int8_t {
  kSTANDARD,
  kSAFETY,
  kDLA_STANDALONE,
};

/**
 * @brief TensorFormat is an enum class which defines the memeory layout used to store Tensor Data
 * */
class TensorFormat {
 public:
  /**
   * Underlying enum class to support the TensorFormat Class
   *
   * In the case that you need to use the TensorFormat class itself, interface
   * using this enum vs. normal instatination
   *
   * ex. torch_tensorrt::TensorFormat type = TensorFormat::kContiguous;
   */
  enum Value : int8_t {
    /// Contiguous / NCHW / Linear
    kContiguous,
    /// Channel Last / NHWC
    kChannelsLast,
    /// Sentinel value
    kUnknown,
  };

  /**
   * @brief Construct a new TensorFormat object
   *
   */
  TensorFormat() = default;
  /**
   * @brief TensorFormat constructor from enum
   *
   */
  constexpr TensorFormat(Value t) : value(t) {}
  /**
   * @brief Construct a new TensorFormat object from torch type enums
   *
   * @param t
   */
  TORCHTRT_API TensorFormat(at::MemoryFormat t);
  /**
   * @brief Get the enum value of the TensorFormat object
   *
   * @return Value
   */
  operator Value() const {
    return value;
  }
  explicit operator bool() = delete;
  /**
   * @brief Comparision operator for TensorFormat
   *
   * @param other
   * @return true
   * @return false
   */
  constexpr bool operator==(TensorFormat other) const {
    return value == other.value;
  }
  /**
   * @brief Comparision operator for TensorFormat
   *
   * @param other
   * @return true
   * @return false
   */
  constexpr bool operator==(TensorFormat::Value other) const {
    return value == other;
  }
  /**
   * @brief Comparision operator for TensorFormat
   *
   * @param other
   * @return true
   * @return false
   */
  constexpr bool operator!=(TensorFormat other) const {
    return value != other.value;
  }
  /**
   * @brief Comparision operator for TensorFormat
   *
   * @param other
   * @return true
   * @return false
   */
  constexpr bool operator!=(TensorFormat::Value other) const {
    return value != other;
  }

 private:
  friend TORCHTRT_API std::ostream& operator<<(std::ostream& os, const TensorFormat& format);
  Value value;
};

/**
 * @brief A struct to hold an input range (used by TensorRT Optimization
 * profile)
 *
 * This struct can either hold a single vector representing an input shape,
 * signifying a static input shape or a set of three input shapes representing
 * the min, optiminal and max input shapes allowed for the engine.
 */
struct TORCHTRT_API Input : torch::CustomClassHolder {
  /// Minimum acceptable input size into the engine
  std::vector<int64_t> min_shape;
  /// Optimal input size into the engine (size optimized for given kernels accept any size in min max range)
  std::vector<int64_t> opt_shape;
  /// Maximum acceptable input size into the engine
  std::vector<int64_t> max_shape;
  /// Input shape to be fed to TensorRT, in the event of a dynamic shape, -1's will hold the place of variable
  /// dimensions
  std::vector<int64_t> shape;
  /// Expected data type for the input
  DataType dtype;
  /// Expected tensor format for the input
  TensorFormat format;

  Input() {}
  /**
   * @brief Construct a new Input spec object for static input size from
   * vector, optional arguments allow the user to configure expected input shape
   * tensor format. dtype (Expected data type for the input) defaults to PyTorch
   * / traditional TRT convection (FP32 for FP32 only, FP16 for FP32 and FP16, FP32 for Int8)
   *
   * @param shape Input tensor shape
   * @param format Expected tensor format for the input (Defaults to contiguous)
   */
  TORCHTRT_API Input(std::vector<int64_t> shape, TensorFormat format = TensorFormat::kContiguous);

  /**
   * @brief Construct a new Input spec object for static input size from
   * vector, optional arguments allow the user to configure expected input shape
   * tensor format
   *
   * @param shape Input tensor shape
   * @param dtype Expected data type for the input (Defaults to the type of the weights in the first tensor
   * calculation if detectable else Float32)
   * @param format Expected tensor format for the input (Defaults to contiguous)
   */
  TORCHTRT_API Input(std::vector<int64_t> shape, DataType dtype, TensorFormat format = TensorFormat::kContiguous);

  /**
   * @brief Construct a new Input spec object for static input size from
   * c10::ArrayRef (the type produced by tensor.sizes()), vector, optional arguments
   * allow the user to configure expected input shape tensor format
   * dtype (Expected data type for the input) defaults to PyTorch
   * / traditional TRT convection (FP32 for FP32 only, FP16 for FP32 and FP16, FP32 for Int8)
   *
   * @param shape Input tensor shape
   * @param format Expected tensor format for the input (Defaults to contiguous)
   */
  TORCHTRT_API Input(c10::ArrayRef<int64_t> shape, TensorFormat format = TensorFormat::kContiguous);

  /**
   * @brief Construct a new Input spec object for static input size from
   * c10::ArrayRef (the type produced by tensor.sizes()), vector, optional arguments
   * allow the user to configure expected input shape tensor format
   *
   * @param shape Input tensor shape
   * @param dtype Expected data type for the input (Defaults to the type of the weights in the first tensor
   * calculation if detectable else Float32)
   * @param format Expected tensor format for the input (Defaults to contiguous)
   */
  TORCHTRT_API Input(c10::ArrayRef<int64_t> shape, DataType dtype, TensorFormat format = TensorFormat::kContiguous);

  /**
   * @brief Construct a new Input spec object dynamic input size from
   * c10::ArrayRef (the type produced by tensor.sizes()) for min, opt, and max
   * supported sizes. dtype (Expected data type for the input) defaults to PyTorch
   * / traditional TRT convection (FP32 for FP32 only, FP16 for FP32 and FP16, FP32 for Int8)
   *
   * @param min_shape Minimum shape for input tensor
   * @param opt_shape Target optimization shape for input tensor
   * @param max_shape Maximum acceptible shape for input tensor
   * @param format Expected tensor format for the input (Defaults to contiguous)
   */
  TORCHTRT_API Input(
      std::vector<int64_t> min_shape,
      std::vector<int64_t> opt_shape,
      std::vector<int64_t> max_shape,
      TensorFormat format = TensorFormat::kContiguous);

  /**
   * @brief Construct a new Input spec object for a dynamic input size from vectors
   * for minimum shape, optimal shape, and max shape supported sizes optional arguments
   * allow the user to configure expected input shape tensor format
   *
   * @param min_shape Minimum shape for input tensor
   * @param opt_shape Target optimization shape for input tensor
   * @param max_shape Maximum acceptible shape for input tensor
   * @param dtype Expected data type for the input (Defaults to the type of the weights in the first tensor
   * calculation if detectable else Float32)
   * @param format Expected tensor format for the input (Defaults to contiguous)
   */
  TORCHTRT_API Input(
      std::vector<int64_t> min_shape,
      std::vector<int64_t> opt_shape,
      std::vector<int64_t> max_shape,
      DataType dtype,
      TensorFormat format = TensorFormat::kContiguous);

  /**
   * @brief Construct a new Input spec object dynamic input size from
   * c10::ArrayRef (the type produced by tensor.sizes()) for min, opt, and max
   * supported sizes. dtype (Expected data type for the input) defaults to PyTorch
   * / traditional TRT convection (FP32 for FP32 only, FP16 for FP32 and FP16, FP32 for Int8)
   *
   * @param min_shape Minimum shape for input tensor
   * @param opt_shape Target optimization shape for input tensor
   * @param max_shape Maximum acceptible shape for input tensor
   * @param format Expected tensor format for the input (Defaults to contiguous)
   */
  TORCHTRT_API Input(
      c10::ArrayRef<int64_t> min_shape,
      c10::ArrayRef<int64_t> opt_shape,
      c10::ArrayRef<int64_t> max_shape,
      TensorFormat format = TensorFormat::kContiguous);

  /**
   * @brief Construct a new Input spec object dynamic input size from
   * c10::ArrayRef (the type produced by tensor.sizes()) for min, opt, and max
   * supported sizes
   *
   * @param min_shape Minimum shape for input tensor
   * @param opt_shape Target optimization shape for input tensor
   * @param max_shape Maximum acceptible shape for input tensor
   * @param dtype Expected data type for the input (Defaults to the type of the weights in the first tensor
   * calculation if detectable else Float32)
   * @param format Expected tensor format for the input (Defaults to contiguous)
   */
  TORCHTRT_API Input(
      c10::ArrayRef<int64_t> min_shape,
      c10::ArrayRef<int64_t> opt_shape,
      c10::ArrayRef<int64_t> max_shape,
      DataType dtype,
      TensorFormat format = TensorFormat::kContiguous);

  /**
   * @brief Construct a new Input spec object using a torch tensor as an example
   * The tensor's shape, type and layout inform the spec's values
   *
   * Note: You cannot set dynamic shape through this method, you must use an alternative constructor
   *
   * @param tensor Reference tensor to set shape, type and layout
   */
  TORCHTRT_API Input(at::Tensor tensor);

 private:
  friend TORCHTRT_API std::ostream& operator<<(std::ostream& os, const Input& input);
  bool input_is_dynamic;
};

/**
 * @brief A struct to hold complex inputs
 *
 * This struct can either hold a complex inputs of shape or a flattened one,
 */
struct TORCHTRT_API GraphInputs {
  torch::jit::IValue input_signature; // nested Input, full input spec
  std::vector<Input> inputs; // flatten input spec
};

/**
 * @brief Get the build information for the library including the dependency
 * versions
 *
 * @return std::string
 */
TORCHTRT_API std::string get_build_info();

/**
 * @brief Dump the version information for Torch-TensorRT including base libtorch and
 * TensorRT versions to stdout
 *
 */
TORCHTRT_API void dump_build_info();

/**
 * @brief Set gpu device id
 *
 * @param gpu_id
 *
 * Sets gpu id using cudaSetDevice
 */
TORCHTRT_API void set_device(const int gpu_id);

namespace torchscript {
/**
 * Settings data structure for Torch-TensorRT TorchScript compilation
 *
 */
struct CompileSpec {
  /**
   * @brief Construct a new Compile Spec object
   * Convienence constructor to set fixed input size from vectors describing
   * size of input tensors. Each entry in the vector represents a input and
   * should be provided in call order.
   *
   * This constructor should be use as a convience in the case that all inputs are static sized and
   * you are okay with default input dtype and formats (FP32 for FP32 and INT8 weights, FP16 for FP16 weights,
   * contiguous)
   *
   * @param fixed_sizes
   */
  TORCHTRT_API CompileSpec(std::vector<std::vector<int64_t>> fixed_sizes);

  /**
   * @brief Construct a new Compile Spec object
   * Convienence constructor to set fixed input size from c10::ArrayRef's (the
   * output of tensor.sizes()) describing size of input tensors. Each entry in
   * the vector represents a input and should be provided in call order.
   *
   * This constructor should be use as a convience in the case that all inputs are static sized and
   * you are okay with default input dtype and formats (FP32 for FP32 and INT8 weights, FP16 for FP16 weights,
   * contiguous)
   *
   * @param fixed_sizes
   */
  TORCHTRT_API CompileSpec(std::vector<c10::ArrayRef<int64_t>> fixed_sizes);

  /**
   * @brief Construct a new Compile Spec object from input ranges.
   * Each entry in the vector represents a input and should be provided in call
   * order.
   *
   * Use this constructor to define inputs with dynamic shape, specific input types or tensor formats
   *
   * @param inputs
   */
  CompileSpec(std::vector<Input> inputs);

  /**
   * @brief Construct a new Compile Spec  object from IValue which represents the nesting of input tensors for a module.
   *
   * @param input_signature
   */
  CompileSpec(torch::jit::IValue input_signature);
  // Defaults should reflect TensorRT defaults for BuilderConfig

  /**
   * @brief Specifications for inputs to the engine, can store a IValue which has stored complex Input
   *  or a flatened Input
   */
  GraphInputs graph_inputs;
  /**
   * @brief The set of precisions TensorRT is allowed to use for kernels during compilation
   *
   */
  std::set<DataType> enabled_precisions = {DataType::kFloat};

  /**
   * Prevent Float32 layers from using TF32 data format
   *
   * TF32 computes inner products by rounding the inputs to 10-bit mantissas
   * before multiplying, but accumulates the sum using 23-bit mantissas.
   * This is the behavior of FP32 layers by default.
   */
  bool disable_tf32 = false;

  /**
   * Enable sparsity for weights of conv and FC layers
   */
  bool sparse_weights = false;

  /**
   * Build a refitable engine
   */
  bool refit = false;

  /**
   * Build a debugable engine
   */
  bool debug = false;

  /**
   * Truncate long/double type to int/float type
   */
  bool truncate_long_and_double = false;

  /**
   * Target Device
   */
  Device device;

  /**
   * Sets the restrictions for the engine (CUDA Safety)
   */
  EngineCapability capability = EngineCapability::kSTANDARD;

  /**
   * Number of averaging timing iterations used to select kernels
   */
  uint64_t num_avg_timing_iters = 1;

  /**
   * Maximum size of workspace given to TensorRT
   */
  uint64_t workspace_size = 0;

  /**
   * Fast software managed RAM used by DLA to communicate within a layer.
   */
  uint64_t dla_sram_size = 1048576;

  /**
   * Host RAM used by DLA to share intermediate tensor data across operations
   */
  uint64_t dla_local_dram_size = 1073741824;

  /**
   * host RAM used by DLA to store weights and metadata for execution
   */
  uint64_t dla_global_dram_size = 536870912;

  /**
   * Calibration dataloaders for each input for post training quantizatiom
   */
  nvinfer1::IInt8Calibrator* ptq_calibrator = nullptr;

  /**
   * Require the full module be compiled to TensorRT instead of potentially running unsupported operations in PyTorch
   */
  bool require_full_compilation = false;

  /**
   * Minimum number of contiguous supported operators to compile a subgraph to TensorRT
   */
  uint64_t min_block_size = 3;

  /**
   * List of aten operators that must be run in PyTorch. An error will be thrown if this list is not empty but
   * ``require_full_compilation`` is True
   */
  std::vector<std::string> torch_executed_ops;

  /**
   * List of modules that must be run in PyTorch. An error will be thrown if this list is not empty but
   * ``require_full_compilation`` is True
   */
  std::vector<std::string> torch_executed_modules;
};

/**
 * @brief Check to see if a module is fully supported by the compiler
 *
 * @param module: torch::jit::script::Module - Existing TorchScript module
 * @param method_name: std::string - Name of method to compile
 *
 * Takes a module and a method name and checks if the method graph contains
 * purely convertable operators
 *
 * Will print out a list of unsupported operators if the graph is unsupported
 *
 * @returns bool: Method is supported by Torch-TensorRT.TorchScript
 */
TORCHTRT_API bool check_method_operator_support(const torch::jit::Module& module, std::string method_name);

/**
 * @brief Compile a TorchScript module for NVIDIA GPUs using TensorRT
 *
 * @param module: torch::jit::Module - Existing TorchScript module
 * @param info: torch_tensorrt::CompileSpec - Compilation settings
 *
 * Takes a existing TorchScript module and a set of settings to configure the
 * compiler and will convert methods to JIT Graphs which call equivalent
 * TensorRT engines
 *
 * Converts specifically the forward method of a TorchScript Module
 *
 * @return: A new module trageting a TensorRT engine
 */
TORCHTRT_API torch::jit::Module compile(const torch::jit::Module& module, CompileSpec info);

/**
 * @brief Compile a TorchScript method for NVIDIA GPUs using TensorRT
 *
 * @param module: torch::jit::Module - Existing TorchScript module
 * @param method_name: std::string - Name of method to compile
 * @param info: torch_tensorrt::CompileSpec - Compilation settings
 *
 * Takes a existing TorchScript module and a set of settings to configure the
 * compiler and will convert selected method to a serialized TensorRT engine
 * which can be run with TensorRT
 *
 * @return: std::string: Serialized TensorRT engine equivilant to the method
 * graph
 */
TORCHTRT_API std::string convert_method_to_trt_engine(
    const torch::jit::Module& module,
    std::string method_name,
    CompileSpec info);

/**
 * @brief Take a previously created TensorRT engine and embed it in
 * in a TorchScript module
 *
 * @param engine: std::string - Pre-built serialized TensorRT engine
 * @param device: CompileSepc::Device - Device information
 *
 * Takes a pre-built serialized TensorRT engine and embeds it in a TorchScript
 * module. Registers execution of the engine as the forward method of the module
 * Forward is defined as: forward(Tensor[]) -> Tensor[]
 *
 * TensorRT bindings must have names with the following format:
 * - [symbol].[index in input / output array]
 * ex.
 * - [x.0, x.1, x.2] -> [y.0]
 *
 * @return: A new module targeting a TensorRT engine
 */
TORCHTRT_API torch::jit::Module embed_engine_in_new_module(const std::string& engine, Device device);
} // namespace torchscript
} // namespace torch_tensorrt
