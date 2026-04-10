# Design Changes for PR #4157

This document contains the exact code changes for the redesigned Multi-Device TensorRT Runtime, addressing review comments.

**Note:** Build config changes (MODULE.bazel, pyproject.toml, setup.py, py/requirements.txt) and debug logging additions (backends.py, remove_sym_nodes.py, partitioning/common.py, utils.py) are NOT included — those are local environment changes.

---

## 1. `core/runtime/runtime.h`

**Change:** ABI version bump and renamed serialization indices.

```diff
-const std::string ABI_VERSION = "8";
+const std::string ABI_VERSION = "9";
```

```diff
-  RANK_IDX,
-  WORLD_SIZE_IDX,
+  IS_MD_ENGINE_IDX,
+  OPTIONAL_RANK_IDX,
+  OPTIONAL_WORLD_SIZE_IDX,
   SERIALIZATION_LEN,
```

---

## 2. `core/runtime/TRTEngine.h`

**Change:** Removed `set_rank`, `set_world_size`, `set_nccl_comm`, `init_nccl_comm`, `set_process_group_from_registry`. Added `detect_distributed_context` and `setup_nccl_comm`.

```diff
   // Distributed inference fields (-1 indicates non-distributed mode)
   int64_t rank = -1;
   int64_t world_size = -1;

-  // Set rank and world_size for distributed inference
-  void set_rank(int64_t rank_val);
-  void set_world_size(int64_t world_size_val);
-
 #ifdef ENABLE_TRT_NCCL_COLLECTIVES
   ncclComm_t nccl_comm = nullptr;
-  void set_nccl_comm(int64_t comm_ptr);
-  void init_nccl_comm(const std::string& group_name = "default");
-  bool set_process_group_from_registry(const std::string& group_name = "default");
+
+  // Detect rank and world_size from ProcessGroup
+  void detect_distributed_context(const std::string& group_name);
+
+  // Resolve ProcessGroup, get NCCL communicator, and bind to TRT context
+  void setup_nccl_comm(const std::string& group_name);
   bool set_nccl_communicator_to_trt_context();
 #endif
```

---

## 3. `core/runtime/TRTEngine.cpp`

### 3a. Constructor 2 (deserialization) — log build-time rank, don't overwrite

```diff
               : ResourceAllocationStrategy::kStatic)) {
-  // Load distributed info if available (backward compatible with older ABI versions)
-  if (serialized_info.size() > RANK_IDX && !serialized_info[RANK_IDX].empty()) {
-    this->rank = std::stoll(serialized_info[RANK_IDX]);
-  }
-  if (serialized_info.size() > WORLD_SIZE_IDX && !serialized_info[WORLD_SIZE_IDX].empty()) {
-    this->world_size = std::stoll(serialized_info[WORLD_SIZE_IDX]);
-  }
+  if (std::stoi(serialized_info[IS_MD_ENGINE_IDX])) {
+    int64_t build_rank = std::stoll(serialized_info[OPTIONAL_RANK_IDX]);
+    int64_t build_world_size = std::stoll(serialized_info[OPTIONAL_WORLD_SIZE_IDX]);
+    if (build_rank != this->rank) {
+      LOG_INFO(
+          "Distributed engine originally built on rank " << build_rank << " of " << build_world_size
+          << ", now running on rank " << this->rank << " of " << this->world_size);
+    } else {
+      LOG_INFO(
+          "Distributed engine: rank " << this->rank << " of " << this->world_size);
+    }
+  }
 }
```

### 3b. Constructor 3 — no distributed logic (removed detect_distributed_context call)

No changes to constructor 3. It is clean — no distributed code.

### 3c. Removed set_rank, set_world_size

```diff
-void TRTEngine::set_rank(int64_t rank_val) {
-  this->rank = rank_val;
-  LOG_DEBUG("Rank set on TRTEngine: " << this->rank);
-}
-
-void TRTEngine::set_world_size(int64_t world_size_val) {
-  this->world_size = world_size_val;
-  LOG_DEBUG("World size set on TRTEngine: " << this->world_size);
-}
```

### 3d. Removed set_nccl_comm, init_nccl_comm, set_process_group_from_registry

All three functions removed entirely.

### 3e. New: detect_distributed_context

```cpp
#ifdef ENABLE_TRT_NCCL_COLLECTIVES
void TRTEngine::detect_distributed_context(const std::string& group_name) {
  auto pg = c10d::resolve_process_group(group_name);
  if (pg) {
    this->rank = pg->getRank();
    this->world_size = pg->getSize();
    LOG_DEBUG("Detected distributed context: rank=" << this->rank << ", world_size=" << this->world_size);
  }
}
```

### 3f. New: setup_nccl_comm (replaces set_process_group_from_registry)

```cpp
void TRTEngine::setup_nccl_comm(const std::string& group_name) {
  auto pg = c10d::resolve_process_group(group_name);
  TORCHTRT_CHECK(pg != nullptr, "ProcessGroup '" << group_name << "' not found in registry");

  auto backend = pg->getBackend(c10d::ProcessGroup::BackendType::NCCL);
  TORCHTRT_CHECK(backend != nullptr, "ProcessGroup '" << group_name << "' has no NCCL backend");

  auto* nccl_pg = dynamic_cast<c10d::ProcessGroupNCCL*>(backend.get());
  TORCHTRT_CHECK(nccl_pg != nullptr, "Backend is not ProcessGroupNCCL");

  at::cuda::set_device(this->device_info.id);

  int64_t comm_ptr = nccl_pg->getCommPtr();
  TORCHTRT_CHECK(
      comm_ptr != 0,
      "NCCL communicator not initialized for device " << this->device_info.id
          << ". Ensure a collective operation has been performed first.");

  this->nccl_comm = reinterpret_cast<ncclComm_t>(comm_ptr);
  set_nccl_communicator_to_trt_context();
  LOG_INFO("NCCL comm set up (rank=" << this->rank << ", device=" << this->device_info.id << ")");
}
```

### 3g. set_nccl_communicator_to_trt_context — replaced try-catch with TORCHTRT_CHECK

```diff
 bool TRTEngine::set_nccl_communicator_to_trt_context() {
-  if (!exec_ctx) {
-    LOG_ERROR("Cannot set NCCL communicator: execution context is null");
-    return false;
-  }
-  if (this->nccl_comm == nullptr) {
-    LOG_WARNING(...);
-    return false;
-  }
-  try {
-    void* comm_ptr = static_cast<void*>(this->nccl_comm);
-    exec_ctx->setCommunicator(comm_ptr);
-    LOG_INFO(...);
-    return true;
-  } catch (const std::exception& e) {
-    LOG_ERROR(...);
-    return false;
-  }
+  TORCHTRT_CHECK(exec_ctx != nullptr, "Cannot set NCCL communicator: execution context is null");
+  TORCHTRT_CHECK(this->nccl_comm != nullptr, "NCCL communicator is not set");
+
+  void* comm_ptr = static_cast<void*>(this->nccl_comm);
+  exec_ctx->setCommunicator(comm_ptr);
+
+  LOG_INFO(
+      "NCCL communicator set on TensorRT execution context "
+      "(rank=" << this->rank << ", device=" << this->device_info.id << ")");
+  return true;
 }
```

### 3h. serialize() — write IS_MD_ENGINE and optional rank/world_size

```diff
   serialized_info[RESOURCE_ALLOCATION_STRATEGY_IDX] =
       this->resource_allocation_strategy == ResourceAllocationStrategy::kDynamic ? "1" : "0";
+  bool is_md = this->world_size > 1;
+  serialized_info[IS_MD_ENGINE_IDX] = is_md ? "1" : "0";
+  if (is_md) {
+    serialized_info[OPTIONAL_RANK_IDX] = std::to_string(this->rank);
+    serialized_info[OPTIONAL_WORLD_SIZE_IDX] = std::to_string(this->world_size);
+  }

   return serialized_info;
```

---

## 4. `core/runtime/register_jit_hooks.cpp`

### 4a. Removed old bindings, added new ones

```diff
         .def_readonly("rank", &TRTEngine::rank)
         .def_readonly("world_size", &TRTEngine::world_size)
-        .def("set_rank", &TRTEngine::set_rank)
-        .def("set_world_size", &TRTEngine::set_world_size)
 #ifdef ENABLE_TRT_NCCL_COLLECTIVES
-        .def("set_nccl_comm", &TRTEngine::set_nccl_comm)
         .def(
-            "init_nccl_comm",
-            [](c10::intrusive_ptr<TRTEngine> self, std::string group_name = "default") {
-              self->init_nccl_comm(group_name);
+            "detect_distributed_context",
+            [](c10::intrusive_ptr<TRTEngine> self, std::string group_name) {
+              self->detect_distributed_context(group_name);
+            })
+        .def(
+            "setup_nccl_comm",
+            [](c10::intrusive_ptr<TRTEngine> self, std::string group_name) {
+              self->setup_nccl_comm(group_name);
             })
 #endif
```

### 4b. Updated constant names

```diff
-  m.def("RANK_IDX", []() -> int64_t { return RANK_IDX; });
-  m.def("WORLD_SIZE_IDX", []() -> int64_t { return WORLD_SIZE_IDX; });
+  m.def("IS_MD_ENGINE_IDX", []() -> int64_t { return IS_MD_ENGINE_IDX; });
+  m.def("OPTIONAL_RANK_IDX", []() -> int64_t { return OPTIONAL_RANK_IDX; });
+  m.def("OPTIONAL_WORLD_SIZE_IDX", []() -> int64_t { return OPTIONAL_WORLD_SIZE_IDX; });
```

---

## 5. `core/runtime/execute_engine.cpp`

**Change:** Only binds NCCL comm to TRT context. Does NOT call `setup_nccl_comm` — Python handles it.

```diff
-      // Distributed setup - set NCCL communicator on TensorRT execution context
+      // Distributed setup - bind NCCL communicator to TRT execution context
+      // setup_nccl_comm must have been called from Python before first forward
 #ifdef ENABLE_TRT_NCCL_COLLECTIVES
-      if (compiled_engine->rank >= 0 && compiled_engine->world_size > 1) {
-        bool result = compiled_engine->set_nccl_communicator_to_trt_context();
-        if (!result) {
-          LOG_ERROR("Failed to set NCCL communicator on TRT context");
-          LOG_ERROR("This will cause collective operations to fail at runtime");
-          LOG_ERROR("Make sure to call module.init_nccl_comm() after compilation");
-        }
-      } else {
-        LOG_DEBUG(
-            "Single-device mode (rank=" << compiled_engine->rank << ", world_size=" << compiled_engine->world_size
-                                        << ") - skipping NCCL setup");
+      if (compiled_engine->world_size > 1 && compiled_engine->nccl_comm != nullptr) {
+        compiled_engine->set_nccl_communicator_to_trt_context();
       }
 #endif
```

---

## 6. `py/torch_tensorrt/dynamo/conversion/_conversion.py`

**Change:** Removed rank/world_size detection and passing to module constructor.

```diff
-    rank = -1
-    world_size = -1
     if settings.use_distributed_mode_trace:
-        import os
-        import torch.distributed as dist
         # Check if distributed backends are available
         ...

     return rt_cls(
         serialized_engine=...,
         ...
-        rank=rank,
-        world_size=world_size,
     )
```

---

## 7. `py/torch_tensorrt/dynamo/runtime/_TorchTensorRTModule.py`

### 7a. Updated constants

```diff
-    RANK_IDX = torch.ops.tensorrt.RANK_IDX()  # 11
-    WORLD_SIZE_IDX = torch.ops.tensorrt.WORLD_SIZE_IDX()  # 12
-    SERIALIZATION_LEN = torch.ops.tensorrt.SERIALIZATION_LEN()  # 13
+    IS_MD_ENGINE_IDX = torch.ops.tensorrt.IS_MD_ENGINE_IDX()  # 11
+    OPTIONAL_RANK_IDX = torch.ops.tensorrt.OPTIONAL_RANK_IDX()  # 12
+    OPTIONAL_WORLD_SIZE_IDX = torch.ops.tensorrt.OPTIONAL_WORLD_SIZE_IDX()  # 13
+    SERIALIZATION_LEN = torch.ops.tensorrt.SERIALIZATION_LEN()  # 14
```

### 7b. Constructor — removed rank/world_size args

```diff
     def __init__(
         self,
         serialized_engine: Optional[bytes] = None,
         ...
-        rank: int = -1,
-        world_size: int = 1,
     ):
```

Removed `self.rank = rank`, `self.world_size = world_size`, `self._nccl_setup_done`.

### 7c. New helper: _get_default_group_name

```python
def _get_default_group_name(self) -> str:
    """Get the group name of the default ProcessGroup."""
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        pg = dist.group.WORLD
        if pg is not None and hasattr(pg, "group_name"):
            return pg.group_name
    return ""
```

### 7d. setup_engine — calls detect_distributed_context

```diff
 def setup_engine(self) -> None:
     if self.engine is not None:
         return
     self.engine = torch.classes.tensorrt.Engine(self._pack_engine_info())
+
+    # Detect distributed context (rank/world_size) from ProcessGroup
+    group_name = self._get_default_group_name()
+    if group_name:
+        self.engine.detect_distributed_context(group_name)
```

### 7e. _pack_engine_info — uses dist.is_initialized

```diff
-        engine_info[RANK_IDX] = str(self.rank)
-        engine_info[WORLD_SIZE_IDX] = str(self.world_size)
+        import torch.distributed as dist
+        is_md = dist.is_initialized() and dist.get_world_size() > 1
+        engine_info[IS_MD_ENGINE_IDX] = str(int(is_md))
+        if is_md:
+            engine_info[OPTIONAL_RANK_IDX] = str(dist.get_rank())
+            engine_info[OPTIONAL_WORLD_SIZE_IDX] = str(dist.get_world_size())
```

### 7f. forward — lazy NCCL setup

```diff
 def forward(self, *inputs):
     if self.engine is None:
         raise RuntimeError("Engine has not been setup yet.")

+    # Lazy NCCL setup on first forward
+    if self.engine.world_size > 1 and not hasattr(self, '_nccl_initialized'):
+        group_name = self._get_default_group_name()
+        if group_name:
+            self.engine.setup_nccl_comm(group_name)
+            self._nccl_initialized = True
+
     assert len(inputs) == len(self.input_binding_names), ...
```

### 7g. Removed functions

- `_auto_init_distributed()` — replaced by lazy setup in forward
- `set_distributed_info()` — called removed `set_rank`/`set_world_size`
- `init_nccl_comm()` — replaced by `setup_nccl_comm` in forward
- `setup_nccl_for_torch_tensorrt` import — no longer needed

---

## 8. `py/torch_tensorrt/dynamo/runtime/_PythonTorchTensorRTModule.py`

### 8a. Constructor — removed rank/world_size args, auto-detect

```diff
     def __init__(
         self,
         ...
-        rank: int = -1,
-        world_size: int = 1,
     ):
         ...
-        self.rank = rank
-        self.world_size = world_size
+        # Auto-detect distributed context
+        import torch.distributed as dist
+        if dist.is_initialized():
+            self.rank = dist.get_rank()
+            self.world_size = dist.get_world_size()
+        else:
+            self.rank = -1
+            self.world_size = -1
         self._nccl_comm: Optional[Any] = None
```

### 8b. Simplified setup_nccl_comm

Replaced `setup_nccl`, `set_nccl_communicator`, `get_nccl_communicator`, `_get_nccl_comm_from_process_group`, `_create_nccl_comm_via_nccl_lib` with a single function:

```python
def setup_nccl_comm(self) -> None:
    """Set up NCCL communicator from PyTorch's ProcessGroup.

    In PythonTorchTensorRTModule, this is a single call that gets the NCCL comm
    and binds it to the TRT context. rank/world_size are already set in __init__
    via dist.get_rank().

    In TorchTensorRTModule (C++ runtime), this is split into two calls:
    - detect_distributed_context(group_name): sets rank/world_size on the C++ engine
      (called in setup_engine, needed for serialization before forward)
    - setup_nccl_comm(group_name): gets NCCL comm and binds to TRT context
      (called lazily on first forward)
    """
    if not self.is_distributed:
        return

    setup_nccl_for_torch_tensorrt()

    import torch.distributed as dist
    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed must be initialized before calling setup_nccl(). "
            "Call dist.init_process_group('nccl') first."
        )

    pg = dist.group.WORLD
    if pg is None or dist.get_backend(pg) != "nccl":
        raise RuntimeError("Default ProcessGroup must use NCCL backend")

    backend = pg._get_backend(torch.device("cuda"))

    # Force NCCL communicator initialization with a dummy collective
    dummy = torch.zeros(1, device="cuda")
    dist.all_reduce(dummy)

    comm_ptr = backend._comm_ptr()
    if comm_ptr is None or comm_ptr == 0:
        raise RuntimeError("Failed to get NCCL communicator from ProcessGroup")

    self._nccl_comm = comm_ptr

    # Bind communicator to TRT execution context (PyCapsule required by TRT Python API)
    if self.context is not None:
        import ctypes
        ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
        ctypes.pythonapi.PyCapsule_New.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p,
        ]
        comm_capsule = ctypes.pythonapi.PyCapsule_New(comm_ptr, None, None)
        self.context.set_communicator(comm_capsule)

    logger.info(f"NCCL comm set up (rank={self.rank}, world_size={self.world_size})")
```

### 8c. Removed functions

- `get_rank()`, `get_world_size()` — fields are public
- `set_nccl_communicator()` — merged into `setup_nccl`
- `get_nccl_communicator()` — `_nccl_comm` is accessible
- `has_native_trt_collectives` property — use `ENABLED_FEATURES.native_trt_collectives`
- `_create_nccl_comm_via_nccl_lib()` — removed `nccl.core` dependency
- `_get_nccl_comm_from_process_group()` — merged into `setup_nccl`

### 8d. _load_from_state_dict — auto-detect rank, log build-time rank

```diff
         self.target_platform = state_dict[prefix + "platform"]
-        self.rank = state_dict.get(prefix + "rank", -1)
-        self.world_size = state_dict.get(prefix + "world_size", -1)
+
+        build_rank = state_dict.get(prefix + "rank", -1)
+        build_world_size = state_dict.get(prefix + "world_size", -1)
+        import torch.distributed as dist
+        if dist.is_initialized():
+            self.rank = dist.get_rank()
+            self.world_size = dist.get_world_size()
+        else:
+            self.rank = -1
+            self.world_size = -1
+        if build_world_size > 1:
+            if build_rank != self.rank:
+                logger.info(
+                    f"Distributed engine originally built on rank {build_rank} of {build_world_size}, "
+                    f"now running on rank {self.rank} of {self.world_size}"
+                )
+            else:
+                logger.info(f"Distributed engine: rank {self.rank} of {self.world_size}")
```

### 8e. forward — uses ENABLED_FEATURES directly

```diff
             if self.is_distributed and self._nccl_comm is None:
                 nccl_type = (
                     "native TRT collectives"
-                    if self.has_native_trt_collectives
+                    if ENABLED_FEATURES.native_trt_collectives
                     else (
```

---

## 9. Remove `nccl.h` dependency — use `void*` for NCCL communicator

**Rationale:** `nccl.h` is not a Bazel dependency — it's picked up from the system/PyTorch install path. Using `void*` instead of `ncclComm_t` removes this fragile dependency. We don't own the communicator (PyTorch's ProcessGroupNCCL owns it), so we just pass it as an opaque pointer to TRT's `setCommunicator(void*)`.

### `core/runtime/TRTEngine.h`

```diff
 #ifdef ENABLE_TRT_NCCL_COLLECTIVES
-#include <nccl.h>
+// Using void* instead of ncclComm_t to avoid nccl.h dependency.
+// We don't own the communicator — it's owned by PyTorch's ProcessGroupNCCL.
 #endif
```

```diff
 #ifdef ENABLE_TRT_NCCL_COLLECTIVES
-  ncclComm_t nccl_comm = nullptr;
+  void* nccl_comm = nullptr;
```

### `core/runtime/TRTEngine.cpp`

In `setup_nccl_comm`:
```diff
-  this->nccl_comm = reinterpret_cast<ncclComm_t>(comm_ptr);
+  this->nccl_comm = reinterpret_cast<void*>(comm_ptr);
```

In `set_nccl_communicator_to_trt_context`:
```diff
-  void* comm_ptr = static_cast<void*>(this->nccl_comm);
-  exec_ctx->setCommunicator(comm_ptr);
+  exec_ctx->setCommunicator(this->nccl_comm);
```

Also update section 3f `setup_nccl_comm` code to use `void*`:

In section 3f above, replace:
```cpp
  this->nccl_comm = reinterpret_cast<ncclComm_t>(comm_ptr);
```
with:
```cpp
  this->nccl_comm = reinterpret_cast<void*>(comm_ptr);
```

And section 3g `set_nccl_communicator_to_trt_context` simplifies to:
```cpp
bool TRTEngine::set_nccl_communicator_to_trt_context() {
  TORCHTRT_CHECK(exec_ctx != nullptr, "Cannot set NCCL communicator: execution context is null");
  TORCHTRT_CHECK(this->nccl_comm != nullptr, "NCCL communicator is not set");

  exec_ctx->setCommunicator(this->nccl_comm);

  LOG_INFO(
      "NCCL communicator set on TensorRT execution context "
      "(rank=" << this->rank << ", device=" << this->device_info.id << ")");
  return true;
}
```

---

## Compatibility bug fixes (for PyTorch 2.10 / NGC 26.01)

These are separate from the design changes but needed to run on the test environment:

### `_FakeTensorUpdater.py` — guard for `torch._inductor.fx_passes.reinplace`

```python
is_scatter = False
if hasattr(torch._inductor.fx_passes, "reinplace"):
    is_scatter = (
        node.target
        is torch._inductor.fx_passes.reinplace._generalized_scatter
    )
```

### `fuse_distributed_ops.py` — handle all_reduce with 3 args

```python
fused_args = tuple(node.args)
if len(fused_args) < 4:
    logger.debug(f"all_reduce node has {len(fused_args)} args instead of 4")
```

### `_TorchTensorRTModule.py` — typo fix

```diff
-                    self.int_nccl_comm(pg)
+                    self.init_nccl_comm(pg)
```

(This line is now removed entirely in the redesign, but was needed for the original PR.)

---

## 10. Move `import torch.distributed as dist` to top-level

Both Python runtime modules had `import torch.distributed as dist` scattered as local imports
inside multiple functions. Moved to top-level since `torch.distributed` is part of PyTorch
(no external dependency).

### `_TorchTensorRTModule.py`

```diff
 import torch
+import torch.distributed as dist
 from torch_tensorrt._Device import Device
```

Removed local imports from `_pack_engine_info()` and `_get_default_group_name()`.

### `_PythonTorchTensorRTModule.py`

```diff
 import torch
+import torch.distributed as dist
 import torch_tensorrt
```

Removed local imports from `__init__()`, `setup_nccl_comm()`, and `_load_from_state_dict()`.

Added comment in `_load_from_state_dict` explaining the design rule:

```python
# Same rule as C++ TRTEngine: serialized rank/world_size are build-time
# metadata for logging. Runtime rank is auto-detected from ProcessGroup.
build_rank = state_dict.get(prefix + "rank", -1)
build_world_size = state_dict.get(prefix + "world_size", -1)
if dist.is_initialized():
    self.rank = dist.get_rank()
    self.world_size = dist.get_world_size()
else:
    self.rank = -1
    self.world_size = -1
if build_world_size > 1:
    if build_rank != self.rank:
        logger.info(
            f"Distributed engine originally built on rank {build_rank} of {build_world_size}, "
            f"now running on rank {self.rank} of {self.world_size}"
        )
    else:
        logger.info(f"Distributed engine: rank {self.rank} of {self.world_size}")
```

---

## 11. Function naming: `setup_nccl_comm` in both runtimes

Both runtime modules use `setup_nccl_comm` as the function name for setting up NCCL,
but they work differently due to the C++ vs Python runtime distinction:

### `_TorchTensorRTModule` (C++ runtime) — two separate calls

```python
# In setup_engine():
self.engine.detect_distributed_context(group_name)  # sets rank/world_size on C++ engine

# In forward() (lazily):
self.engine.setup_nccl_comm(group_name)  # gets NCCL comm, binds to TRT context
```

**Why split:** rank/world_size must be available for serialization before any forward call.
The NCCL communicator is only needed at execution time.

### `_PythonTorchTensorRTModule` (Python runtime) — single call

```python
# In forward() (lazily):
self.setup_nccl_comm()  # gets NCCL comm, converts to PyCapsule, sets on TRT context
```

**Why single:** rank/world_size are already set in `__init__` via `dist.get_rank()`.
No C++ engine to populate. Only need to get the NCCL comm and bind it.

Comment in `_PythonTorchTensorRTModule.setup_nccl_comm`:
```python
def setup_nccl_comm(self) -> None:
    """Set up NCCL communicator from PyTorch's ProcessGroup.

    In PythonTorchTensorRTModule, this is a single call that gets the NCCL comm
    and binds it to the TRT context. rank/world_size are already set in __init__
    via dist.get_rank().

    In TorchTensorRTModule (C++ runtime), this is split into two calls:
    - detect_distributed_context(group_name): sets rank/world_size on the C++ engine
      (called in setup_engine, needed for serialization before forward)
    - setup_nccl_comm(group_name): gets NCCL comm and binds to TRT context
      (called lazily on first forward)
    """
```

## 12. Move `setup_nccl_for_torch_tensorrt()` to user scripts

**Rationale:** `setup_nccl_for_torch_tensorrt()` sets `LD_LIBRARY_PATH` so TensorRT can find `libnccl.so`.
This is a one-time environment setup, not an engine-level concern. The reviewer said this
should be a utility the user calls, not hidden inside engine code.

### `_PythonTorchTensorRTModule.py` — removed call and import

```diff
-from torch_tensorrt.dynamo.runtime._nccl_utils import setup_nccl_for_torch_tensorrt
```

```diff
 def setup_nccl_comm(self) -> None:
     if not self.is_distributed:
         return

-    setup_nccl_for_torch_tensorrt()
-
     if not dist.is_initialized():
```

### Example scripts — added call after imports

`examples/distributed_inference/tensor_parallel_simple_example.py`:
```python
import torch_tensorrt
from torch_tensorrt.dynamo.runtime._nccl_utils import setup_nccl_for_torch_tensorrt

setup_nccl_for_torch_tensorrt()
```

`tools/llm/tensor_parallel_llama_llm.py`:
```python
import torch_tensorrt
from torch_tensorrt.dynamo.runtime._nccl_utils import setup_nccl_for_torch_tensorrt

setup_nccl_for_torch_tensorrt()
```

The user is now responsible for calling `setup_nccl_for_torch_tensorrt()` once before
distributed TRT inference. The function remains in `_nccl_utils` as a public utility.
