/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "torch_tensorrt/executorch/PooledScratchInstall.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <NvInfer.h>

#include <executorch/runtime/platform/log.h>

namespace torch_tensorrt {
namespace executorch_backend {
namespace {

// Collects what TensorRT reports while it is attached. Attaching it is
// ScopedRecorderAttachment's below; this is only the sink.
class CollectingErrorRecorder final : public nvinfer1::IErrorRecorder {
 public:
  int32_t getNbErrors() const noexcept override {
    return static_cast<int32_t>(errors_.size());
  }
  nvinfer1::ErrorCode getErrorCode(int32_t index) const noexcept override {
    return in_range(index) ? errors_[static_cast<size_t>(index)].first : nvinfer1::ErrorCode::kSUCCESS;
  }
  ErrorDesc getErrorDesc(int32_t index) const noexcept override {
    return in_range(index) ? errors_[static_cast<size_t>(index)].second.c_str() : "";
  }
  bool hasOverflowed() const noexcept override {
    return overflowed_;
  }
  void clear() noexcept override {
    errors_.clear();
  }
  bool reportError(nvinfer1::ErrorCode code, ErrorDesc desc) noexcept override {
    // noexcept, and a throwing push_back here would terminate: an allocation
    // failure while reporting an error costs the description, not the process.
    try {
      errors_.emplace_back(code, std::string(desc == nullptr ? "" : desc));
    } catch (...) {
      overflowed_ = true;
    }
    return false; // false asks TensorRT to keep going; the caller decides
  }
  RefCount incRefCount() noexcept override {
    return ++refs_;
  }
  RefCount decRefCount() noexcept override {
    return --refs_;
  }

  bool anything_reported() const {
    return !errors_.empty() || overflowed_;
  }

 private:
  bool in_range(int32_t index) const {
    return index >= 0 && static_cast<size_t>(index) < errors_.size();
  }

  std::vector<std::pair<nvinfer1::ErrorCode, std::string>> errors_;
  bool overflowed_ = false;
  RefCount refs_ = 1;
};

// Attaches `recorder` to `ctx` for the length of one TensorRT call and puts back
// whatever the caller had attached before.
//
// The restore is a destructor rather than a statement after the call. `recorder`
// is a stack object and `ctx` outlives the call, so a return added between the
// attach and the restore would leave TensorRT holding a pointer into a dead
// frame. Nor is the restore tidiness: while a recorder is attached TensorRT
// reports to it *instead of* to the ILogger, so one left in place would divert
// the diagnostics for setInputShape, setTensorAddress and enqueueV3 away from the
// backend's logger.
//
// It takes a reference of its own on the recorder it displaces and holds it
// across the swap. TensorRT drops a reference on the recorder it replaces and
// takes one again on the restore, so a recorder whose only reference is
// TensorRT's reaches zero on the way in: measured on a real execution context on
// TensorRT 11.2.1 and 10.16.1.11, the replaced recorder's count goes to 0 at the
// attach and incRefCount is then called on it again at the restore. Destroying an
// error recorder once its count reaches zero is what IErrorRecorder's own
// documentation describes and what TensorRT's samples do, so without this
// reference such a caller's recorder is freed at the attach and re-registered
// dangling at the restore, once per pooled call.
class ScopedRecorderAttachment {
 public:
  ScopedRecorderAttachment(nvinfer1::IExecutionContext& ctx, nvinfer1::IErrorRecorder& recorder)
      : ctx_(ctx), previous_(ctx.getErrorRecorder()) {
    if (previous_ != nullptr) {
      previous_->incRefCount();
    }
    ctx_.setErrorRecorder(&recorder);
  }
  ScopedRecorderAttachment(const ScopedRecorderAttachment&) = delete;
  ScopedRecorderAttachment& operator=(const ScopedRecorderAttachment&) = delete;

  ~ScopedRecorderAttachment() {
    ctx_.setErrorRecorder(previous_);
    if (previous_ != nullptr) {
      previous_->decRefCount();
    }
  }

 private:
  nvinfer1::IExecutionContext& ctx_;
  nvinfer1::IErrorRecorder* previous_;
};

} // namespace

// Declared in PooledScratchInstall.h. See that header for why the install is
// checked rather than made and trusted.
bool install_pooled_scratch(nvinfer1::IExecutionContext& ctx, void* buffer, std::size_t bytes, int device_id) {
  CollectingErrorRecorder recorder;
  {
    const ScopedRecorderAttachment attached(ctx, recorder);
    ctx.setDeviceMemoryV2(buffer, static_cast<int64_t>(bytes));
  }

  if (!recorder.anything_reported()) {
    return true;
  }
  ET_LOG(
      Error,
      "TensorRTBackend::execute: TensorRT refused the %zu-byte shared activation scratch buffer for device %d: %s",
      bytes,
      device_id,
      recorder.getNbErrors() > 0 ? recorder.getErrorDesc(0) : "the refusal could not be recorded");
  return false;
}

} // namespace executorch_backend
} // namespace torch_tensorrt
