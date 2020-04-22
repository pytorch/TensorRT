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

#include "torch/torch.h"
#include "NvInfer.h"

// Just include the .h?
namespace torch {
namespace jit {
struct Graph;
namespace script {
struct Module;
} // namespace script
} // namespace jit
} // namespace torch

namespace c10 {
enum class DeviceType : int16_t;
enum class ScalarType : int8_t;
template <class>
class ArrayRef;
}

namespace nvinfer1 {
class IInt8EntropyCalibrator2;
}

#include "trtorch/macros.h"
#include "trtorch/logging.h"
#include "trtorch/ptq.h"
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
        std::vector<int64_t> min;
        std::vector<int64_t> opt;
        std::vector<int64_t> max;
        InputRange(std::vector<int64_t> opt);
        InputRange(c10::ArrayRef<int64_t> opt);
        InputRange(std::vector<int64_t> min, std::vector<int64_t> opt, std::vector<int64_t> max);
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

        DataType() = default;
        constexpr DataType(Value t) : value(t) {}
        DataType(c10::ScalarType t);
        operator Value() const  { return value; }
        explicit operator bool() = delete;
        constexpr bool operator==(DataType other) const { return value == other.value; }
        constexpr bool operator!=(DataType other) const { return value != other.value; }
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

        DeviceType() = default;
        constexpr DeviceType(Value t) : value(t) {}
        DeviceType(c10::DeviceType t);
        operator Value() const { return value; }
        explicit operator bool() = delete;
        constexpr bool operator==(DeviceType other) const { return value == other.value; }
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

    ExtraInfo(std::vector<InputRange> input_ranges)
        : input_ranges(std::move(input_ranges)) {}
    ExtraInfo(std::vector<std::vector<int64_t>> fixed_sizes);
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
    bool strict_type = false;

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
    uint64_t workspace_size = 1 << 20;

    /**
     * Calibration dataloaders for each input for post training quantizatiom
     */
    nvinfer1::IInt8Calibrator* ptq_calibrator;
};

/**
 * Get the version information for TRTorch including base libtorch and TensorRT versions
 */
TRTORCH_API std::string get_build_info();

/**
 * Dump the version information for TRTorch including base libtorch and TensorRT versions
 * to stdout
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
 */
TRTORCH_API bool CheckMethodOperatorSupport(const torch::jit::script::Module& module, std::string method_name);

/**
 * @brief Compile a TorchScript module for NVIDIA GPUs using TensorRT
 *
 * @param module: torch::jit::script::Module - Existing TorchScript module
 * @param info: trtorch::ExtraInfo - Compilation settings
 *
 * Takes a existing TorchScript module and a set of settings to configure the compiler
 * and will convert methods to JIT Graphs which call equivalent TensorRT engines
 *
 * Converts specifically the forward method of a TorchScript Module
 */
TRTORCH_API torch::jit::script::Module CompileGraph(const torch::jit::script::Module& module, ExtraInfo info);

/**
 * @brief Compile a TorchScript method for NVIDIA GPUs using TensorRT
 *
 * @param module: torch::jit::script::Module - Existing TorchScript module
 * @param method_name: std::string - Name of method to compile
 * @param info: trtorch::ExtraInfo - Compilation settings
 *
 * Takes a existing TorchScript module and a set of settings to configure the compiler
 * and will convert selected method to a serialized TensorRT engine which can be run with
 * TensorRT
 */
TRTORCH_API std::string ConvertGraphToTRTEngine(const torch::jit::script::Module& module, std::string method_name, ExtraInfo info);

namespace ptq {
/**
 * @brief A factory to build a post training quantization calibrator from a torch dataloader
 *
 * Creates a calibrator to use for post training quantization
 * If there are multiple inputs, the dataset should produce a example which is a vector (or similar container) of tensors vs a single tensor
 *
 * By default the returned calibrator uses TensorRT Entropy v2 algorithm to perform calibration. This is recommended for feed forward networks
 * You can override the algorithm selection (such as to use the MinMax Calibrator recomended for NLP tasks) by calling make_int8_calibrator with
 * the calibrator class as a template parameter.
 *
 * e.g. trtorch::ptq::make_int8_calibrator<nvinfer1::IInt8MinMaxCalibrator>(std::move(calibration_dataloader), calibration_cache_file, use_cache);
 */
template<typename Algorithm = nvinfer1::IInt8EntropyCalibrator2, typename DataLoader>
TRTORCH_API inline Int8Calibrator<Algorithm, DataLoader> make_int8_calibrator(DataLoader dataloader, const std::string& cache_file_path, bool use_cache) {
    return Int8Calibrator<Algorithm, DataLoader>(std::move(dataloader), cache_file_path, use_cache);
}

/**
 * @brief A factory to build a post training quantization calibrator from a torch dataloader that only uses the calibration cache
 *
 * Creates a calibrator to use for post training quantization which reads from a previously created calibration cache, therefore
 * you can have a calibration cache generating program that requires a dataloader and a dataset, then save the cache to use later
 * in a different program that needs to calibrate from scratch and not have the dataset dependency. However, the network should also
 *  be recalibrated if its structure changes, or the input data set changes, and it is the responsibility of the application to ensure this.
 *
 * By default the returned calibrator uses TensorRT Entropy v2 algorithm to perform calibration. This is recommended for feed forward networks
 * You can override the algorithm selection (such as to use the MinMax Calibrator recomended for NLP tasks) by calling make_int8_calibrator with
 * the calibrator class as a template parameter.
 *
 * e.g. trtorch::ptq::make_int8_cache_calibrator<nvinfer1::IInt8MinMaxCalibrator>(calibration_cache_file);
 */
template<typename Algorithm = nvinfer1::IInt8EntropyCalibrator2>
TRTORCH_API inline Int8CacheCalibrator<Algorithm> make_int8_cache_calibrator(const std::string& cache_file_path) {
    return Int8CacheCalibrator<Algorithm>(cache_file_path);
}
} // namespace ptq
} // namespace trtorch
