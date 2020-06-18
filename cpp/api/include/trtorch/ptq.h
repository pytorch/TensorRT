#pragma once

#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <sstream>

#include "trtorch/logging.h"

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace nvinfer1 {
class IInt8Calibrator;
class IInt8EntropyCalibrator2;
}

namespace torch {
class Tensor;
}

namespace trtorch {
namespace ptq {
bool get_batch_impl(void* bindings[], const char* names[], int nbBindings, torch::Tensor& data);
}
}
#endif //DOXYGEN_SHOULD_SKIP_THIS

namespace trtorch {
namespace ptq {

/**
 * @brief Generic Int8Calibrator implementation based on a specified
 * TensorRT calibration algorithm and a LibTorch DataLoader
 *
 * @tparam Algorithm: class nvinfer1::IInt8Calibrator (Default: nvinfer1::IInt8EntropyCalibrator2) - Algorithm to use
 * @tparam DataLoaderUniquePtr: std::unique_ptr<torch::data::DataLoader> - DataLoader type
 */
template<typename Algorithm, typename DataLoaderUniquePtr>
class Int8Calibrator : Algorithm {
    using DataLoader = typename DataLoaderUniquePtr::element_type;
    using Batch = typename DataLoader::super::BatchType;
public:
    /**
     * @brief Construct a new Int8Calibrator object
     *
     * Using the provided DataLoader, construct a calibrator that can be used for PTQ with TRTorch
     *
     * @param dataloader: std::unqiue_ptr<torch::data::DataLoader> - A unique pointer to the DataLoader, should be what is returned from the make_data_loader factory
     * @param cache_file_path: const std::string& - A path to store / find the calibration cache
     * @param use_cache : bool - Whether to use the cache (if it exists)
     */
    Int8Calibrator(DataLoaderUniquePtr dataloader, const std::string& cache_file_path, bool use_cache)
      : dataloader_(dataloader.get()), cache_file_path_(cache_file_path), use_cache_(use_cache) {
          for (auto batch : *dataloader_) {
            batched_data_.push_back(batch.data);
          }
          it_ = batched_data_.begin();
      }

    /**
     * @brief Get the Batch Size for the next batch (always 1 due to issues with TRT and explicit batch)
     *
     * @return int
     */
    int getBatchSize() const override {
        // HACK: TRTorch only uses explict batch sizing, INT8 Calibrator does not
        // work when reporting the batch size here and having explicity batching.
        // So we just report batch size 1 (warnings will still be printed out).
        return 1;
        //return static_cast<int>(dataloader_->options().batch_size);
    }

    /**
     * @brief Get the next Batch
     *
     * @param bindings: void*[] - An array of binding pointers (fed in from TensorRT calibrator), these buffers should be filed with batch data for each input
     * @param names: const char*[] - Names of bindings
     * @param nbBindings: int - Number of bindings
     * @return true - There is a new batch for the calibrator to consume
     * @return false - There is not a new batch for the calibrator to consume
     */
    bool getBatch(void* bindings[], const char* names[], int nbBindings) override {
        if (it_ != batched_data_.end()) {
            auto status = get_batch_impl(bindings, names, nbBindings, *it_);
            it_ = ++it_;
            return status;
        } else {
            // Reset iterator if incase calibrator is going to be used again
            it_ = batched_data_.begin();
            return false;
        }
    }

    /**
     * @brief Read calibration cache
     *
     * How to read from the calibration cache, only enabled if use_cache is set
     *
     * @param length
     * @return const void* - Pointer to cache data
     */
    const void* readCalibrationCache(size_t& length) override {
        if (use_cache_) {
            std::stringstream ss;
            ss << "Reading Calibration Cache from " << cache_file_path_;
            logging::log(logging::Level::kINFO, ss.str());

            cache_.clear();
            std::ifstream input(cache_file_path_, std::ios::binary);
            input >> std::noskipws;
            if (input.good()) {
                std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
                    std::back_inserter(cache_));
                logging::log(logging::Level::kDEBUG, "Cache read");
            }
            length = cache_.size();
            return length ? cache_.data() : nullptr;
        }
        return nullptr;
    }

    /**
     * @brief Write calibration cache
     *
     * Write a the calibration cache provided by TensorRT to a specified file
     *
     * @param cache: const void* - cache data
     * @param length: size_t - length of cache
     */
    void writeCalibrationCache(const void* cache, size_t length) override {
        std::ofstream cache_file(cache_file_path_, std::ios::binary);
        cache_file.write(reinterpret_cast<const char*>(cache), length);
        std::stringstream ss;
        ss << "Saved Calibration Cache to " << cache_file_path_;
        logging::log(logging::Level::kINFO, ss.str());
    }

    /**
     * @brief operator to cast to nvinfer1::IInt8Calibrator*
     *
     * Convience function to convert to a IInt8Calibrator* to easily be assigned to the ptq_calibrator field in ExtraInfo
     *
     * @return nvinfer1::IInt8Calibrator*
     */
    operator nvinfer1::IInt8Calibrator* () {
        return reinterpret_cast<nvinfer1::IInt8Calibrator*>(this);
    }

private:
    /// Pointer to the dataloader
    DataLoader* dataloader_;
    /// Path to cache file
    const std::string& cache_file_path_;
    /// Size of cache
    size_t cache_size_ = 0;
    /// Whether to use the cache or not
    bool use_cache_;
    /// Cache data
    std::vector<char> cache_;
    /// Batched Data
    std::vector<torch::Tensor> batched_data_;
    /// Iterator to move through dataset
    std::vector<torch::Tensor>::iterator it_;

};

/**
 * @brief Generic Int8Calibrator implementation based on a specified
 * TensorRT calibration algorithm that only reads from a calibration file
 *
 * @tparam Algorithm: class nvinfer1::IInt8Calibrator (Default: nvinfer1::IInt8EntropyCalibrator2) - Algorithm to use
 */
template<typename Algorithm>
class Int8CacheCalibrator : Algorithm {
public:
    /**
     * @brief Construct a new Int 8 Cache Calibrator object
     *
     * @param cache_file_path
     */
    Int8CacheCalibrator(const std::string& cache_file_path)
      : cache_file_path_(cache_file_path) {}

    /**
     * @brief Get the Batch Size for the next batch (always 1 due to issues with TRT and explicit batch)
     *
     * @return int
     */
    int getBatchSize() const override {
        // HACK: TRTorch only uses explict batch sizing, INT8 Calibrator does not
        // work when reporting the batch size here and having explicity batching.
        // So we just report batch size 1 (warnings will still be printed out).
        return 1;
    }

    /**
     * @brief Get the next Batch
     *
     * Not used always returns false
     *
     * @param bindings: void*[] - An array of binding pointers (fed in from TensorRT calibrator), these buffers should be filed with batch data for each input
     * @param names: const char*[] - Names of bindings
     * @param nbBindings: int - Number of bindings
     * @return false
     */
    bool getBatch(void* bindings[], const char* names[], int nbBindings) override {
        return false;
    }

    /**
     * @brief Read calibration cache
     *
     * How to read from the calibration cache, only enabled if use_cache is set
     *
     * @param length
     * @return const void* - Pointer to cache data
     */
    const void* readCalibrationCache(size_t& length) override {
        std::stringstream ss;
        ss << "Reading Calibration Cache from " << cache_file_path_;
        logging::log(logging::Level::kINFO, ss.str());

        cache_.clear();
        std::ifstream input(cache_file_path_, std::ios::binary);
        input >> std::noskipws;
        if (input.good()) {
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
                std::back_inserter(cache_));
            logging::log(logging::Level::kDEBUG, "Cache read");
        }
        length = cache_.size();
        return length ? cache_.data() : nullptr;
    }


    /**
     * @brief Write calibration cache
     *
     * Write a the calibration cache provided by TensorRT to a specified file
     *
     * @param cache: const void* - cache data
     * @param length: size_t - length of cache
     */
    void writeCalibrationCache(const void* cache, size_t length) override {
        std::ofstream cache_file(cache_file_path_, std::ios::binary);
        cache_file.write(reinterpret_cast<const char*>(cache), length);
        std::stringstream ss;
        ss << "Saved Calibration Cache to " << cache_file_path_;
        logging::log(logging::Level::kINFO, ss.str());
    }

    /**
     * @brief operator to cast to nvinfer1::IInt8Calibrator*
     *
     * Convience function to convert to a IInt8Calibrator* to easily be assigned to the ptq_calibrator field in ExtraInfo
     *
     * @return nvinfer1::IInt8Calibrator*
     */
    operator nvinfer1::IInt8Calibrator* () {
        return reinterpret_cast<nvinfer1::IInt8Calibrator*>(this);
    }

private:
    /// Path to cache file
    const std::string& cache_file_path_;
    /// Size of cache
    size_t cache_size_ = 0;
    /// Cache data
    std::vector<char> cache_;
};

} // namespace ptq
} // namespace trtorch