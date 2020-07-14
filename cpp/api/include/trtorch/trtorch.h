/*
 * Copyright (c) NVIDIA Corporation.
 * All rights reserved.
 *
 * This library is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <vector>
#include <memory>

// Just include the .h?
#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace torch {
namespace jit {
struct Graph;
struct Module;
} // namespace jit
} // namespace torch

namespace c10 {
enum class DeviceType : int16_t;
enum class ScalarType : int8_t;
template <class>
class ArrayRef;
}

namespace nvinfer1 {
class IInt8Calibrator;
}
#endif //DOXYGEN_SHOULD_SKIP_THIS

#include "trtorch/macros.h"
namespace trtorch {
/**
 * Settings data structure for TRTorch compilation
 *
 */
struct TRTORCH_API ExtraInfo {
    /**
     * @brief A struct to hold an input range (used by TensorRT Optimization profile)
     *
     * This struct can either hold a single vector representing an input shape, signifying a
     * static input shape or a set of three input shapes representing the min, optiminal and max
     * input shapes allowed for the engine.
     */
    struct TRTORCH_API InputRange {
        /// Minimum acceptable input size into the engine
        std::vector<int64_t> min;
        /// Optimal input size into the engine (gets best performace)
        std::vector<int64_t> opt;
        /// Maximum acceptable input size into the engine
        std::vector<int64_t> max;
        /**
         * @brief Construct a new Input Range object for static input size from vector
         *
         * @param opt
         */
        InputRange(std::vector<int64_t> opt);
        /**
         * @brief Construct a new Input Range object static input size from c10::ArrayRef (the type produced by tensor.sizes())
         *
         * @param opt
         */
        InputRange(c10::ArrayRef<int64_t> opt);
        /**
         * @brief Construct a new Input Range object dynamic input size from vectors for min, opt, and max supported sizes
         *
         * @param min
         * @param opt
         * @param max
         */
        InputRange(std::vector<int64_t> min, std::vector<int64_t> opt, std::vector<int64_t> max);
        /**
         * @brief Construct a new Input Range object dynamic input size from c10::ArrayRef (the type produced by tensor.sizes()) for min, opt, and max supported sizes
         *
         * @param min
         * @param opt
         * @param max
         */
        InputRange(c10::ArrayRef<int64_t> min, c10::ArrayRef<int64_t> opt, c10::ArrayRef<int64_t> max);
    };

    /**
     * Supported Data Types that can be used with TensorRT engines
     *
     * This class is compatable with c10::DataTypes (but will check for TRT support)
     * so there should not be a reason that you need to use this type explictly.
     */
    class DataType {
    public:
        /**
         * Underlying enum class to support the DataType Class
         *
         * In the case that you need to use the DataType class itself, interface using
         * this enum vs. normal instatination
         *
         * ex. trtorch::DataType type = DataType::kFloat;
         */
        enum Value : int8_t {
            /// FP32
            kFloat,
            /// FP16
            kHalf,
            /// INT8
            kChar,
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
        DataType(c10::ScalarType t);
        /**
         * @brief Get the enum value of the DataType object
         *
         * @return Value
         */
        operator Value() const  { return value; }
        explicit operator bool() = delete;
        /**
         * @brief Comparision operator for DataType
         *
         * @param other
         * @return true
         * @return false
         */
        constexpr bool operator==(DataType other) const { return value == other.value; }
        /**
         * @brief Comparision operator for DataType
         *
         * @param other
         * @return true
         * @return false
         */
        constexpr bool operator==(DataType::Value other) const { return value == other; }
        /**
         * @brief Comparision operator for DataType
         *
         * @param other
         * @return true
         * @return false
         */
        constexpr bool operator!=(DataType other) const { return value != other.value; }
        /**
         * @brief Comparision operator for DataType
         *
         * @param other
         * @return true
         * @return false
         */
        constexpr bool operator!=(DataType::Value other) const { return value != other; }
    private:
        Value value;
    };

    /**
     * Supported Device Types that can be used with TensorRT engines
     *
     * This class is compatable with c10::DeviceTypes (but will check for TRT support)
     * but the only applicable value is at::kCUDA, which maps to DeviceType::kGPU
     *
     * To use the DataType class itself, interface using the enum vs. normal instatination
     *
     * ex. trtorch::DeviceType type = DeviceType::kGPU;
     */
    class DeviceType {
    public:
        /**
         * Underlying enum class to support the DeviceType Class
         *
         * In the case that you need to use the DeviceType class itself, interface using
         * this enum vs. normal instatination
         *
         * ex. trtorch::DeviceType type = DeviceType::kGPU;
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
        operator Value() const { return value; }
        explicit operator bool() = delete;
        /**
         * @brief Comparison operator for DeviceType
         *
         * @param other
         * @return true
         * @return false
         */
        constexpr bool operator==(DeviceType other) const { return value == other.value; }
        /**
         * @brief Comparison operator for DeviceType
         *
         * @param other
         * @return true
         * @return false
         */
        constexpr bool operator!=(DeviceType other) const { return value != other.value; }
    private:
        Value value;
    };

    /**
     * Emum for selecting engine capability
     */
    enum class EngineCapability : int8_t {
        kDEFAULT,
        kSAFE_GPU,
        kSAFE_DLA,
    };

    /**
     * @brief Construct a new Extra Info object from input ranges.
     * Each entry in the vector represents a input and should be provided in call order.
     *
     * Use this constructor if you want to use dynamic shape
     *
     * @param input_ranges
     */
    ExtraInfo(std::vector<InputRange> input_ranges)
        : input_ranges(std::move(input_ranges)) {}
    /**
     * @brief Construct a new Extra Info object
     * Convienence constructor to set fixed input size from vectors describing size of input tensors.
     * Each entry in the vector represents a input and should be provided in call order.
     *
     * @param fixed_sizes
     */
    ExtraInfo(std::vector<std::vector<int64_t>> fixed_sizes);
    /**
     * @brief Construct a new Extra Info object
     * Convienence constructor to set fixed input size from c10::ArrayRef's (the output of tensor.sizes()) describing size of input tensors.
     * Each entry in the vector represents a input and should be provided in call order.
     * @param fixed_sizes
     */
    ExtraInfo(std::vector<c10::ArrayRef<int64_t>> fixed_sizes);

    // Defaults should reflect TensorRT defaults for BuilderConfig

    /**
     * Sizes for inputs to engine, can either be a single size or a range
     * defined by Min, Optimal, Max sizes
     *
     * Order is should match call order
     */
    std::vector<InputRange> input_ranges;

    /**
     * Default operating precision for the engine
     */
    DataType op_precision = DataType::kFloat;

    /**
     * Build a refitable engine
     */
    bool refit = false;

    /**
     * Build a debugable engine
     */
    bool debug = false;

    /**
     * Restrict operating type to only set default operation precision (op_precision)
     */
    bool strict_types = false;

    /**
     * (Only used when targeting DLA (device))
     * Lets engine run layers on GPU if they are not supported on DLA
     */
    bool allow_gpu_fallback = true;

    /**
     * Target device type
     */
    DeviceType device = DeviceType::kGPU;

    /**
     * Sets the restrictions for the engine (CUDA Safety)
     */
    EngineCapability capability = EngineCapability::kDEFAULT;

    /**
     * Number of minimization timing iterations used to select kernels
     */
    uint64_t num_min_timing_iters = 2;
    /**
     * Number of averaging timing iterations used to select kernels
     */
    uint64_t num_avg_timing_iters = 1;

    /**
     * Maximum size of workspace given to TensorRT
     */
    uint64_t workspace_size = 0;

    /**
     * Maximum batch size (must be >= 1 to be set, 0 means not set)
     */
    uint64_t max_batch_size = 0;

    /**
     * Calibration dataloaders for each input for post training quantizatiom
     */
    nvinfer1::IInt8Calibrator* ptq_calibrator = nullptr;
};

/**
 * @brief Get the build information for the library including the dependency versions
 *
 * @return std::string
 */
TRTORCH_API std::string get_build_info();


/**
 * @brief Dump the version information for TRTorch including base libtorch and TensorRT versions
 * to stdout
 *
 */
TRTORCH_API void dump_build_info();

/**
 * @brief Check to see if a module is fully supported by the compiler
 *
 * @param module: torch::jit::script::Module - Existing TorchScript module
 * @param method_name: std::string - Name of method to compile
 *
 * Takes a module and a method name and checks if the method graph contains purely
 * convertable operators
 *
 * Will print out a list of unsupported operators if the graph is unsupported
 *
 * @returns bool: Method is supported by TRTorch
 */
TRTORCH_API bool CheckMethodOperatorSupport(const torch::jit::Module& module, std::string method_name);

/**
 * @brief Compile a TorchScript module for NVIDIA GPUs using TensorRT
 *
 * @param module: torch::jit::Module - Existing TorchScript module
 * @param info: trtorch::ExtraInfo - Compilation settings
 *
 * Takes a existing TorchScript module and a set of settings to configure the compiler
 * and will convert methods to JIT Graphs which call equivalent TensorRT engines
 *
 * Converts specifically the forward method of a TorchScript Module
 *
 * @return: A new module trageting a TensorRT engine
 */
TRTORCH_API torch::jit::Module CompileGraph(const torch::jit::Module& module, ExtraInfo info);

/**
 * @brief Compile a TorchScript method for NVIDIA GPUs using TensorRT
 *
 * @param module: torch::jit::Module - Existing TorchScript module
 * @param method_name: std::string - Name of method to compile
 * @param info: trtorch::ExtraInfo - Compilation settings
 *
 * Takes a existing TorchScript module and a set of settings to configure the compiler
 * and will convert selected method to a serialized TensorRT engine which can be run with
 * TensorRT
 *
 * @return: std::string: Serialized TensorRT engine equivilant to the method graph
 */
TRTORCH_API std::string ConvertGraphToTRTEngine(const torch::jit::Module& module, std::string method_name, ExtraInfo info);
} // namespace trtorch
