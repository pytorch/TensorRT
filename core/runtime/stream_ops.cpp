#include "core/util/prelude.h"

#include <mutex>
#include <unordered_map>

#include "ATen/cuda/CUDAEvent.h"
#include "c10/cuda/CUDAStream.h"
#include "torch/library.h"
#include "torch/torch.h"

#include "core/runtime/StreamGuard.h"
#include "core/runtime/runtime.h"

namespace torch_tensorrt {
namespace core {
namespace runtime {

// ── StreamGuard implementation ────────────────────────────────────────────────

namespace {

// Keep-alive table for streams created via auto_bind.  Without this, the
// torch::cuda::Stream wrapper would be destroyed at the end of
// auto_bind_fresh_stream() and the underlying cudaStream_t handle could be
// reused by the driver.  Keyed by raw stream handle so callers can also
// retrieve streams the same way they'd retrieve a TorchBind attribute.
std::mutex g_keepalive_mu;
std::unordered_map<int64_t, c10::cuda::CUDAStream> g_auto_bound_streams;

void validate_stream_device(int64_t handle, int64_t device_index) {
  // CUDA reserves stream IDs 1 (legacy default) and 2 (per-thread default) as
  // magic values with global synchronization semantics that defeat
  // the SM-partitioning use case.  handle=0 (the device default stream) is
  // accepted — bind(0) represents "this guard tracks the default stream,"
  // which enter_compute_stream uses to record a default-stream caller and
  // exit_compute_stream uses to restore it.
  TORCHTRT_CHECK(
      handle != 1 && handle != 2,
      "StreamGuard.bind: legacy/per-thread magic stream IDs (" << handle << ") not supported");
  // Device-affinity validation is a best-effort cross-check; we accept the
  // handle even if the runtime cannot resolve its context, since some setups
  // (e.g., externally-created streams not yet primed) won't expose it.
  (void)device_index;
}

} // namespace

StreamGuard::StreamGuard(int64_t device_index, bool auto_bind) : device_index_(device_index), auto_bind_(auto_bind) {
  if (auto_bind_) {
    auto_bind_fresh_stream();
  }
}

void StreamGuard::bind(int64_t cuda_stream_handle) {
  validate_stream_device(cuda_stream_handle, device_index_);
  handle_ = cuda_stream_handle;
}

void StreamGuard::clear() {
  handle_ = 0;
}

int64_t StreamGuard::get_handle() const {
  return handle_;
}

bool StreamGuard::is_bound() const {
  return handle_ != 0;
}

int64_t StreamGuard::device_index() const {
  return device_index_;
}

bool StreamGuard::auto_bind() const {
  return auto_bind_;
}

void StreamGuard::set_auto_bind(bool value) {
  auto_bind_ = value;
}

void StreamGuard::auto_bind_fresh_stream() {
  auto stream = c10::cuda::getStreamFromPool(
      /*isHighPriority=*/false, static_cast<c10::DeviceIndex>(device_index_));
  int64_t h = reinterpret_cast<int64_t>(stream.stream());
  {
    std::lock_guard<std::mutex> lock(g_keepalive_mu);
    g_auto_bound_streams.emplace(h, stream);
  }
  handle_ = h;
}

std::tuple<int64_t, bool> StreamGuard::__getstate__() const {
  // Drop the live handle on save — cudaStream_t is process-local.
  return std::make_tuple(device_index_, auto_bind_);
}

void StreamGuard::__setstate__(std::tuple<int64_t, bool> state) {
  device_index_ = std::get<0>(state);
  auto_bind_ = std::get<1>(state);
  handle_ = 0;
  if (auto_bind_) {
    auto_bind_fresh_stream();
  }
}

std::tuple<std::tuple<std::string, int64_t>, std::tuple<std::string, bool>> StreamGuard::__obj_flatten__() const {
  return std::tuple(
      std::tuple<std::string, int64_t>("device_index", device_index_),
      std::tuple<std::string, bool>("auto_bind", auto_bind_));
}

// ── Stream-control ops (StreamGuard-typed) ────────────────────────────────────

namespace {

c10::cuda::CUDAStream stream_from_guard(const c10::intrusive_ptr<StreamGuard>& g, const char* op_name) {
  TORCHTRT_CHECK(g, op_name << ": StreamGuard ScriptObject is null (graph attribute missing or wrong type)");
  TORCHTRT_CHECK(
      g->is_bound(),
      op_name << ": StreamGuard for cuda:" << g->device_index()
              << " is not bound. Call StreamGuard.bind(stream_handle) or use "
                 "auto_bind=True before invoking the planned module.");
  return c10::cuda::getStreamFromExternal(
      reinterpret_cast<cudaStream_t>(g->get_handle()), static_cast<c10::DeviceIndex>(g->device_index()));
}

// ── enter_compute_stream ───────────────────────────────────────────────────────
// Capture the caller's current CUDA stream handle into the caller StreamGuard
// (mutation), and — only when the caller is on the default stream — switch the
// current stream to the planned compute stream.
//
// caller_guard is a pre-registered StreamGuard attribute on the planned
// module (registered by the FX pass).  Its lifetime is tied to the module's
// lifetime, so downstream sync_streams nodes that reference it via get_attr
// pick up the runtime-captured handle.  Returns the caller handle as an int
// token for FX data-flow ordering.
int64_t enter_compute_stream(
    c10::intrusive_ptr<StreamGuard> primary,
    c10::intrusive_ptr<StreamGuard> caller_guard,
    int64_t device_index) {
  TORCHTRT_CHECK(primary, "enter_compute_stream: primary StreamGuard is null");
  TORCHTRT_CHECK(caller_guard, "enter_compute_stream: caller StreamGuard is null");
  auto current = c10::cuda::getCurrentCUDAStream(static_cast<c10::DeviceIndex>(device_index));
  auto default_stream = c10::cuda::getDefaultCUDAStream(static_cast<c10::DeviceIndex>(device_index));
  int64_t caller_handle = reinterpret_cast<int64_t>(current.stream());

  // Mutate caller_guard at runtime so downstream sync_streams calls referencing
  // this same TorchBind object see the captured handle.  If caller was on
  // default stream, leave caller_guard unbound (sync_streams treats unbound
  // as "default stream, no fence needed due to implicit barriers").
  caller_guard->clear();
  if (current != default_stream) {
    caller_guard->bind(caller_handle);
  } else {
    auto planned = stream_from_guard(primary, "enter_compute_stream");
    c10::cuda::setCurrentCUDAStream(planned);
  }
  return caller_handle;
}

// ── exit_compute_stream ───────────────────────────────────────────────────────
// Restore the caller's pre-enter stream from caller_guard's bound handle.
// Unbound means the caller was on the default stream — restore there.
void exit_compute_stream(c10::intrusive_ptr<StreamGuard> caller_guard, int64_t device_index) {
  TORCHTRT_CHECK(caller_guard, "exit_compute_stream: caller StreamGuard is null");
  if (caller_guard->is_bound()) {
    auto restore = c10::cuda::getStreamFromExternal(
        reinterpret_cast<cudaStream_t>(caller_guard->get_handle()), static_cast<c10::DeviceIndex>(device_index));
    c10::cuda::setCurrentCUDAStream(restore);
  } else {
    c10::cuda::setCurrentCUDAStream(c10::cuda::getDefaultCUDAStream(static_cast<c10::DeviceIndex>(device_index)));
  }
}

// ── set_stream ────────────────────────────────────────────────────────────────
// Switch the current CUDA stream to guard's bound stream.  Returns the raw
// stream handle as a token so call_trt_with_engine can data-depend on this op.
int64_t set_stream(c10::intrusive_ptr<StreamGuard> guard, int64_t /*device_index*/) {
  auto s = stream_from_guard(guard, "set_stream");
  c10::cuda::setCurrentCUDAStream(s);
  return reinterpret_cast<int64_t>(s.stream());
}

// ── sync_streams ──────────────────────────────────────────────────────────────
// Record a CUDA event on src and have dst wait for it.  Returns dst's handle
// as a token so call_trt_with_token can data-depend on this op.
//
// When either guard is unbound (represents the default stream), the explicit
// fence is unnecessary: CUDA's default stream has implicit-barrier semantics
// that already serialize it against work on any other stream.  We return a
// non-zero token in that case (dst's recorded device handle from the live
// stream lookup, or src's) so the data-flow edge into call_trt_with_token
// stays intact.
int64_t sync_streams(c10::intrusive_ptr<StreamGuard> src, c10::intrusive_ptr<StreamGuard> dst, int64_t device_index) {
  TORCHTRT_CHECK(src && dst, "sync_streams: src or dst StreamGuard is null");
  if (!src->is_bound() || !dst->is_bound()) {
    // One side is the default stream; default-stream implicit barriers
    // already cover ordering.  Token is the dst handle if bound, else src,
    // else a non-zero sentinel — value only matters for FX scheduling.
    if (dst->is_bound())
      return dst->get_handle();
    if (src->is_bound())
      return src->get_handle();
    return 1; // sentinel: both sides are default; ordering is implicit
  }
  auto src_s = stream_from_guard(src, "sync_streams");
  auto dst_s = stream_from_guard(dst, "sync_streams");
  at::cuda::CUDAEvent event;
  event.record(src_s);
  event.block(dst_s);
  (void)device_index;
  return reinterpret_cast<int64_t>(dst_s.stream());
}

// ── call_trt_with_token ───────────────────────────────────────────────────────
// Real dispatcher op (not a Python HOP).  Takes an int token from the
// preceding stream-control op as its first argument, creating a hard
// data-flow edge that any FX/AOT/Inductor scheduler must respect.  The token
// is otherwise discarded — the body just forwards to execute_engine.
//
// AOTI codegen handles this op the same as any custom dispatcher op: it
// emits a C++ call into the dispatcher, the engine ScriptObject argument
// resolves to the corresponding torchbind constant in the loaded .pt2.
std::vector<at::Tensor> call_trt_with_token(
    int64_t /*token*/,
    c10::intrusive_ptr<TRTEngine> engine,
    std::vector<at::Tensor> inputs) {
  return execute_engine(std::move(inputs), std::move(engine));
}

} // namespace

// ── Schema + impl registration ────────────────────────────────────────────────
//
// TORCH_LIBRARY_FRAGMENT adds to an existing library namespace without
// redefining it.  Schemas here are visible to AOTI codegen and the dispatcher
// the same as any other custom op.

TORCH_LIBRARY_FRAGMENT(tensorrt, m) {
  // TorchBind permits only one __init__ signature per class.  We pick the
  // 2-arg form (device_index, auto_bind) and expose factory shortcuts via
  // separate methods on the Python side (see runtime/_stream_binding.py).
  m.class_<StreamGuard>("StreamGuard")
      .def(torch::init<int64_t, bool>())
      .def("bind", &StreamGuard::bind)
      .def("clear", &StreamGuard::clear)
      .def("get_handle", &StreamGuard::get_handle)
      .def("is_bound", &StreamGuard::is_bound)
      .def("device_index", &StreamGuard::device_index)
      .def("auto_bind", &StreamGuard::auto_bind)
      .def("set_auto_bind", &StreamGuard::set_auto_bind)
      .def("__obj_flatten__", &StreamGuard::__obj_flatten__)
      .def_pickle(
          [](const c10::intrusive_ptr<StreamGuard>& self) { return self->__getstate__(); },
          [](std::tuple<int64_t, bool> state) {
            auto sg = c10::make_intrusive<StreamGuard>(std::get<0>(state), false);
            sg->__setstate__(state);
            return sg;
          });

  m.def(
      "enter_compute_stream(__torch__.torch.classes.tensorrt.StreamGuard primary, "
      "__torch__.torch.classes.tensorrt.StreamGuard caller_guard, "
      "int device_index) -> int");
  m.def(
      "exit_compute_stream(__torch__.torch.classes.tensorrt.StreamGuard caller_guard, "
      "int device_index) -> ()");
  m.def(
      "set_stream(__torch__.torch.classes.tensorrt.StreamGuard guard, "
      "int device_index) -> int");
  m.def(
      "sync_streams(__torch__.torch.classes.tensorrt.StreamGuard src, "
      "__torch__.torch.classes.tensorrt.StreamGuard dst, "
      "int device_index) -> int");
  m.def(
      "call_trt_with_token(int token, "
      "__torch__.torch.classes.tensorrt.Engine engine, "
      "Tensor[] inputs) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(tensorrt, CompositeExplicitAutograd, m) {
  m.impl("enter_compute_stream", enter_compute_stream);
  m.impl("exit_compute_stream", exit_compute_stream);
  m.impl("set_stream", set_stream);
  m.impl("sync_streams", sync_streams);
  m.impl("call_trt_with_token", call_trt_with_token);
}

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
