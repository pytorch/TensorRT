#pragma once

#include <cstdint>
#include <tuple>

#include "torch/custom_class.h"

namespace torch_tensorrt {
namespace core {
namespace runtime {

/// Serializable holder for a CUDA stream handle, exposed as TorchBind class
/// torch.classes.tensorrt.StreamGuard.
///
/// Designed to be a graph attribute on stream-planned modules.  The four
/// stream-control ops (set_stream / sync_streams / enter_compute_stream /
/// exit_compute_stream) take a StreamGuard ScriptObject as their stream
/// argument and dereference it at op-call time, instead of baking a raw
/// cudaStream_t handle as a literal int into the FX graph.  This keeps the
/// graph free of dead pointers across save/load and AOTI codegen.
///
/// Pickling: __getstate__ records (device_index, auto_bind) and drops the
/// live handle.  __setstate__ either auto-creates a fresh stream on the
/// recorded device (auto_bind=true) or leaves the guard unbound (false), in
/// which case the user must call .bind(stream_handle) before invoking the
/// planned module.  set_stream / sync_streams raise if invoked on an
/// unbound guard.
struct StreamGuard : torch::CustomClassHolder {
  StreamGuard() = default;
  explicit StreamGuard(int64_t device_index, bool auto_bind = false);

  void bind(int64_t cuda_stream_handle);
  void clear();
  int64_t get_handle() const;
  bool is_bound() const;
  int64_t device_index() const;
  bool auto_bind() const;
  void set_auto_bind(bool value);

  // Materialize a fresh CUDA stream on the recorded device and bind to it.
  // Owns the underlying torch::cuda stream object via a static keep-alive
  // map so the cudaStream_t handle stays valid for the guard's lifetime.
  void auto_bind_fresh_stream();

  // Pickle hooks.  We record only (device_index, auto_bind) — never the live
  // handle, since cudaStream_t is process-local.
  std::tuple<int64_t, bool> __getstate__() const;
  void __setstate__(std::tuple<int64_t, bool> state);

  // torch.export uses __obj_flatten__ to lower TorchBind constants in the
  // graph to a list of (name, value) pairs.  We export the same fields as
  // pickle so re-export through retrace produces an equivalent guard.
  std::tuple<std::tuple<std::string, int64_t>, std::tuple<std::string, bool>> __obj_flatten__() const;

 private:
  int64_t handle_ = 0;
  int64_t device_index_ = 0;
  bool auto_bind_ = false;
};

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
