from __future__ import annotations

import base64
import copy
import logging
import pickle
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch_tensorrt._Device import Device
from torch_tensorrt._enums import Platform
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.runtime._serialized_engine_layout import (
    ABI_TARGET_IDX,
    ABI_VERSION,
    DEVICE_IDX,
    ENGINE_IDX,
    HW_COMPATIBLE_IDX,
    INPUT_BINDING_NAMES_IDX,
    NAME_IDX,
    OUTPUT_BINDING_NAMES_IDX,
    REQUIRES_NATIVE_MULTIDEVICE_IDX,
    REQUIRES_OUTPUT_ALLOCATOR_IDX,
    RESOURCE_ALLOCATION_STRATEGY_IDX,
    SERIALIZATION_LEN,
    SERIALIZED_METADATA_IDX,
    TARGET_PLATFORM_IDX,
    SerializedTensorRTEngineFmt,
    serialize_binding_names,
    serialize_device_info,
)
from torch_tensorrt.runtime._runtime_config import RuntimeSettings

logger = logging.getLogger(__name__)

SerializedTorchTensorRTModuleFmt = Tuple[
    str,
    Optional[SerializedTensorRTEngineFmt],
    List[str],
    List[str],
]


class TorchTensorRTModule(torch.nn.Module):  # type: ignore[misc]
    """``nn.Module`` that runs a TensorRT engine inside PyTorch.

    When the C++ Torch-TensorRT runtime is available, execution uses
    ``torch.classes.tensorrt.Engine`` and ``torch.ops.tensorrt.execute_engine``.
    When only the Python runtime is available, a Python :class:`TRTEngine` is
    registered under the same ``tensorrt::execute_engine`` op so that the same
    compiled graph works with either runtime transparently.

    Supports ``torch.save`` / ``torch.load`` via ``get_extra_state`` / ``set_extra_state``.
    """

    def __init__(
        self,
        serialized_engine: Optional[bytes] = None,
        input_binding_names: Optional[List[str]] = None,
        output_binding_names: Optional[List[str]] = None,
        *,
        name: str = "",
        settings: CompilationSettings = CompilationSettings(),
        requires_output_allocator: bool = False,
        requires_native_multidevice: bool = False,
        symbolic_shape_expressions: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ):
        """Takes a name, target device, serialized TensorRT engine, and binding names / order and constructs
        a PyTorch ``torch.nn.Module`` around it. Uses the Torch-TensorRT runtime extension to run the engines

        If binding names are not provided, it is assumed that the engine binding names follow the following convention:

            - [symbol].[index in input / output array]
                - ex. [x.0, x.1, x.2] -> [y.0]

        Arguments:
            serialized_engine (bytes): Serialized TensorRT engine in the form of a bytearray
            input_binding_names (List[str]): List of input TensorRT engine binding names in the order they would be passed to the TRT modules
            output_binding_names (List[str]): List of output TensorRT engine binding names in the order they should be returned

        Keyword Arguments:
            name (str): Name for module
            settings (torch_tensorrt.dynamo.CompilationSettings): Settings used to compile engine, assumes engine was built with default compilation settings if object not passed
            requires_output_allocator (bool): Boolean flag indicating if the converter creates operators which require an Output Allocator to run (e.g. data dependent operators)
            requires_native_multidevice (bool): Boolean flag indicating if the converter creates operators which require multiple devices to run (e.g. multi-device collective operations)
            symbolic_shape_expressions (List[Any]): List of symbolic shape expressions for each input binding

        Example:

            .. code-block:: py

                with io.BytesIO() as engine_bytes:
                    engine_bytes.write(trt_engine.serialize())
                    engine_str = engine_bytes.getvalue()

                trt_module = TorchTensorRTModule(
                    engine_str,
                    input_binding_names=["x"],
                    output_binding_names=["output"],
                    name="my_module",
                    settings=CompilationSettings(device=torch.cuda.current_device)
                )

        Args:
            serialized_engine: Raw TRT engine bytes (``None`` if restoring state only).
            input_binding_names: Input tensor names in ``forward`` order.
            output_binding_names: Output tensor names in return order.
            name: Logical name for logging and serialization.
            settings: Compilation/runtime settings (device, lazy init, cross-compile, etc.).
            requires_output_allocator: Engine needs TRT dynamic output allocation.
            symbolic_shape_expressions: Optional symbolic shape metadata from compile.
        """
        super().__init__()

        self.input_binding_names = (
            input_binding_names if input_binding_names is not None else []
        )
        self.output_binding_names = (
            output_binding_names if output_binding_names is not None else []
        )
        self.name = name
        self.hardware_compatible = settings.hardware_compatible
        self.settings = copy.deepcopy(settings)
        self.serialized_engine = serialized_engine
        self.engine: Optional[Any] = None
        self.requires_output_allocator = requires_output_allocator
        self.dynamically_allocate_resources = settings.dynamically_allocate_resources

        # Per-engine runtime mode controls.
        self._runtime_settings: RuntimeSettings = RuntimeSettings()
        # Engine-implicit ``RuntimeCache``: built lazily when the user
        # passes a path string via ``runtime_settings`` (the module owns the
        # wrapper; the engine just holds a reference).
        self._implicit_cache_handle: Any = None
        self.symbolic_shape_expressions = symbolic_shape_expressions
        self.requires_native_multidevice = requires_native_multidevice
        self.target_platform = (
            Platform.current_platform()
            if not self.settings.enable_cross_compile_for_windows
            else Platform.WIN_X86_64
        )
        self.profiling_enabled = False
        self.target_device = self._resolve_target_device()

        if (
            serialized_engine
            and not self.settings.lazy_engine_init
            and not self.settings.enable_cross_compile_for_windows
        ):
            self.setup_engine()

    def __deepcopy__(self, memo: dict[int, Any]) -> "TorchTensorRTModule":
        # The C++ TRTEngine is not safely deep-copyable for distributed (NCCL)
        # engines — creating a new execution context during copy can crash at
        # destruction time.  Since the exporter only reads the engine (never
        # executes it), sharing the same C++ object is safe.
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "engine":
                object.__setattr__(result, k, v)  # shallow: reuse the same C++ Engine
            else:
                object.__setattr__(result, k, copy.deepcopy(v, memo))
        return result

    def _resolve_target_device(self) -> torch.device:
        """Resolve the engine's target CUDA device from compilation settings."""
        if self.settings.device is not None:
            return torch.device(f"cuda:{self.settings.device.gpu_id}")
        return torch.device(f"cuda:{torch.cuda.current_device()}")

    def _pack_engine_info(self) -> List[str | bytes]:
        target_device = (
            self.settings.device
            if self.settings.device is not None
            else Device._current_device()
        )
        metadata = {
            "settings": self.settings,
            "inout_symexprs": self.symbolic_shape_expressions,
            "output_tensors_are_unowned": (
                False
                if self.engine is None
                else self.engine.are_output_tensors_unowned()
            ),
        }

        engine_info: List[str | bytes] = [""] * SERIALIZATION_LEN
        engine_info[ABI_TARGET_IDX] = (
            torch.ops.tensorrt.ABI_VERSION()
            if ENABLED_FEATURES.torch_tensorrt_runtime
            else ABI_VERSION
        )
        engine_info[NAME_IDX] = (
            self.name + "_engine" if self.name != "" else "tensorrt_engine"
        )
        engine_info[DEVICE_IDX] = (
            target_device._to_serialized_rt_device()
            if ENABLED_FEATURES.torch_tensorrt_runtime
            else serialize_device_info(target_device)
        )
        assert self.serialized_engine is not None
        engine_info[ENGINE_IDX] = self.serialized_engine
        engine_info[INPUT_BINDING_NAMES_IDX] = serialize_binding_names(
            self.input_binding_names
        )
        engine_info[OUTPUT_BINDING_NAMES_IDX] = serialize_binding_names(
            self.output_binding_names
        )
        engine_info[HW_COMPATIBLE_IDX] = str(int(self.hardware_compatible))
        engine_info[SERIALIZED_METADATA_IDX] = self.encode_metadata(metadata)
        engine_info[TARGET_PLATFORM_IDX] = (
            self.target_platform._to_serialized_rt_platform()
            if ENABLED_FEATURES.torch_tensorrt_runtime
            else str(self.target_platform)
        )
        engine_info[REQUIRES_OUTPUT_ALLOCATOR_IDX] = str(
            int(self.requires_output_allocator)
        )
        logger.info(
            f"PROVIDED RESOURCE ALLOCATION STRATEGY: {self.dynamically_allocate_resources}"
        )
        engine_info[RESOURCE_ALLOCATION_STRATEGY_IDX] = str(
            int(self.dynamically_allocate_resources)
        )
        engine_info[REQUIRES_NATIVE_MULTIDEVICE_IDX] = str(
            int(self.requires_native_multidevice)
        )
        # rank/world_size are runtime facts; queried from ProcessGroup at execution time.
        # RuntimeSettings are intentionally NOT serialized: they're per-engine, in-memory
        # init values, not part of the engine's identity (see pytorch/TensorRT#4310).
        return engine_info

    def get_engine(self) -> torch.classes.tensorrt.Engine:
        """Return the underlying engine, raising if it has not been set up.

        Used by every engine-accessing method except the hot ``forward`` path,
        which intentionally skips the check to avoid per-call overhead.
        """
        if self.engine is None:
            raise RuntimeError("Engine has not been setup yet.")
        return self.engine

    def get_streamable_device_memory_budget(self) -> Any:
        return self.get_engine().streamable_device_memory_budget

    def get_automatic_device_memory_budget(self) -> Any:
        return self.get_engine().automatic_device_memory_budget

    def get_device_memory_budget(self) -> Any:
        return self.get_engine().device_memory_budget

    def set_device_memory_budget(self, budget_bytes: int) -> int:
        engine = self.get_engine()
        if budget_bytes < 0:
            budget_bytes = self.get_streamable_device_memory_budget()
        engine.device_memory_budget = budget_bytes
        if engine.device_memory_budget != budget_bytes:
            logger.error(f"Failed to set weight streaming budget to {budget_bytes}")
            budget_bytes = engine.device_memory_budget
        if self.get_streamable_device_memory_budget() == budget_bytes:
            logger.warning("Weight streaming is disabled")
        return budget_bytes

    def _reset_captured_graph(self) -> None:
        self.get_engine().reset_captured_graph()

    def use_dynamically_allocated_resources(
        self, dynamically_allocate_resources: bool = False
    ) -> None:
        self.dynamically_allocate_resources = dynamically_allocate_resources
        self.get_engine().use_dynamically_allocated_resources(
            self.dynamically_allocate_resources
        )

    # --- runtime-settings dispatch ----------------------------------------

    @property
    def runtime_settings(self) -> RuntimeSettings:
        """The current ``RuntimeSettings``. Snapshot used by ``runtime_config`` CM enter/exit."""
        return self._runtime_settings

    @runtime_settings.setter
    def runtime_settings(self, rs: RuntimeSettings) -> None:
        """Apply ``RuntimeSettings`` to this engine (or stash for ``setup_engine``)."""
        # 1. Normalize: path-string -> managed handle; everything else passes through.
        rs_resolved = self._resolve_runtime_cache(rs)
        # 2. Push to the engine if it exists; if not we stash for later.
        if self.engine is not None:
            self._send_to_engine(rs_resolved)
        # 3. Store the resolved form so reads agree with what the engine sees.
        self._runtime_settings = rs_resolved

    def _resolve_runtime_cache(self, rs: RuntimeSettings) -> RuntimeSettings:
        """Normalize ``rs.runtime_cache`` to ``None`` | ``RuntimeCache`` (never a path str).

        Manages the ``_implicit_cache_handle`` slot as a side effect: builds a
        fresh wrapper for a new path, reuses the existing one for the same
        path, releases it (with save-on-swap) for non-path inputs.
        """
        from torch_tensorrt.runtime._runtime_cache import RuntimeCache

        rc = rs.runtime_cache

        # Branch 1: not a non-empty path string (None / empty / handle / torchbind).
        # Caller owns the handle's lifecycle -- release ours, pass rs through.
        # Guard: if ``rc`` IS our managed wrapper (e.g. setup_engine re-applying
        # a resolved rs), this is a no-op self-passback, not a release.
        if not (isinstance(rc, str) and rc):
            if rc is not self._implicit_cache_handle:
                self._set_managed_handle(None)
            return rs

        # Branch 2: same path + wrapper still usable -> reuse. Keeps the CM
        # enter/exit cycle cheap (no teardown/rebuild loses in-memory kernels).
        old = self._implicit_cache_handle
        if old is not None and old.path == rc and self._wrapper_still_attached(old):
            return rs.merge(runtime_cache=old)

        # Branch 3: build a fresh managed wrapper for this path. The facade
        # auto-picks the torchbind sibling on cpp rt and the pure-Python
        # handle on python-only rt. ``autosave_on_del=True`` is how implicit
        # caches persist across runs.
        new = RuntimeCache(path=rc, autosave_on_del=True)
        # Explicitly load bytes here (which lands either in the runtime cache
        # or pending bytes) as this cache is managed by this module, so a
        # correct initial state is set.
        try:
            new.load()
        except Exception as e:
            logger.warning(
                f"Failed to warm-load implicit runtime cache from {rc!r}: {e}"
            )
        self._set_managed_handle(new)
        return rs.merge(runtime_cache=new)

    def _set_managed_handle(self, new: Optional[Any]) -> None:
        """Install ``new`` as the implicit handle; save the prior wrapper if displaced.

        Save errors are swallowed (logged) so a transient disk failure during
        a settings swap can't crash the user's assignment.
        """
        old = self._implicit_cache_handle
        if old is not None and old is not new:
            try:
                old.save()
            except Exception as e:
                logger.warning(
                    f"Failed to save prior implicit runtime cache on swap: {e}"
                )
        self._implicit_cache_handle = new

    def _wrapper_still_attached(self, w: Any) -> bool:
        """Is ``w`` reusable for the current runtime? Python rt always
        accepts; cpp rt needs the torchbind sibling live (else the cpp
        engine has no way to hold the same underlying ``IRuntimeCache``).
        """
        return not ENABLED_FEATURES.torch_tensorrt_runtime or w.is_cpp_runtime()

    def _send_to_engine(self, rs: RuntimeSettings) -> None:
        """Push ``rs`` to whichever engine flavor is attached."""
        from torch_tensorrt.dynamo.runtime._TRTEngine import TRTEngine
        from torch_tensorrt.runtime._runtime_cache import _to_torchbind_handle
        from torch_tensorrt.runtime._runtime_config import (
            _CUDA_GRAPH_STRATEGY_MAP,
            _DYNAMIC_SHAPES_KERNEL_STRATEGY_MAP,
        )

        if isinstance(self.engine, TRTEngine):
            self.engine.update_runtime_settings(rs)
        else:
            # Strategies cross the boundary as ints (TorchBind ``int64_t``,
            # mirroring the nvinfer1 enum integers on the cpp side).
            self.engine.update_runtime_settings(
                _DYNAMIC_SHAPES_KERNEL_STRATEGY_MAP[
                    rs.dynamic_shapes_kernel_specialization_strategy
                ],
                _CUDA_GRAPH_STRATEGY_MAP[rs.cuda_graph_strategy],
                _to_torchbind_handle(rs.runtime_cache),
            )

    def setup_engine(self) -> None:
        """
        Setup engine for a module which has deferred engine setup.

        Will setup the TensorRT engine for this module in the case that setup has been
        deferred. In the case that the engine has already been setup, will return without
        changing anything. Assumes that serialized engine and settings have already been passed
        to the module.
        """
        if self.engine is not None:
            return

        if not ENABLED_FEATURES.torch_tensorrt_runtime:
            from torch_tensorrt.dynamo.runtime._TRTEngine import TRTEngine

            self.engine = TRTEngine(
                self._pack_engine_info(),
                profile_execution=self.profiling_enabled,
            )
        else:
            self.engine = torch.classes.tensorrt.Engine(self._pack_engine_info())

        # Re-apply via the setter: resolves any path-string runtime_cache,
        # dispatches to the engine, and writes back the resolved form.
        self.runtime_settings = self._runtime_settings

        # requires_native_multidevice is set by the C++ constructor from the serialized REQUIRES_NATIVE_MULTIDEVICE_IDX field.
        if self.engine.requires_native_multidevice:
            from torch_tensorrt.distributed._nccl_utils import (
                check_nccl_engine_requirements,
            )

            check_nccl_engine_requirements()

        # Store the active process group name on the C++ engine so that the
        # lazy NCCL setup in execute_engine() can find the right communicator
        # without needing any further Python involvement.
        if (
            ENABLED_FEATURES.torch_tensorrt_runtime
            and self.engine.requires_native_multidevice
        ):
            from torch_tensorrt.distributed._distributed import (
                get_active_group_name,
                register_md_engine,
            )

            group_name = get_active_group_name()
            if group_name:
                self.engine.set_group_name(group_name)

            # Register the C++ engine for teardown tracking so
            # distributed_context().__exit__ can release the NCCL comm even
            # for torch.compile models where the engine lives in dynamo's
            # code cache and isn't reachable via module tree walking.
            register_md_engine(self.engine)

    def encode_metadata(self, metadata: Any) -> str:
        metadata = copy.deepcopy(metadata)
        dumped_metadata = pickle.dumps(metadata)
        encoded_metadata = base64.b64encode(dumped_metadata).decode("utf-8")
        return encoded_metadata

    @staticmethod
    def decode_metadata(encoded_metadata: bytes) -> Any:
        dumped_metadata = base64.b64decode(encoded_metadata.encode("utf-8"))
        metadata = pickle.loads(dumped_metadata)
        return metadata

    def get_extra_state(self) -> SerializedTorchTensorRTModuleFmt:
        if self.engine is not None:
            engine_info = self._pack_engine_info()
            engine_bytes = engine_info[ENGINE_IDX]
            assert isinstance(engine_bytes, (bytes, bytearray))
            engine_info[ENGINE_IDX] = base64.b64encode(engine_bytes)
            return (
                self.name,
                engine_info,
                self.input_binding_names,
                self.output_binding_names,
            )
        elif self.serialized_engine:
            engine_info = self._pack_engine_info()
            engine_bytes = engine_info[ENGINE_IDX]
            assert isinstance(engine_bytes, bytes)
            engine_info[ENGINE_IDX] = base64.b64encode(engine_bytes)
            return (
                self.name,
                engine_info,
                self.input_binding_names,
                self.output_binding_names,
            )
        else:
            return (
                self.name,
                None,
                self.input_binding_names,
                self.output_binding_names,
            )

    def set_extra_state(self, state: SerializedTorchTensorRTModuleFmt) -> None:
        self.name = state[0]

        if state[1] is not None:
            serialized_engine_info: SerializedTensorRTEngineFmt = list(state[1])
            serialized_engine_info[ENGINE_IDX] = base64.b64decode(
                serialized_engine_info[ENGINE_IDX]
            )
            self.hardware_compatible = bool(
                int(serialized_engine_info[HW_COMPATIBLE_IDX])
            )
            self.requires_output_allocator = bool(
                int(serialized_engine_info[REQUIRES_OUTPUT_ALLOCATOR_IDX])
            )

            serialized_metadata = serialized_engine_info[SERIALIZED_METADATA_IDX]
            # ``_pack_engine_info`` packs the metadata as a ``str``
            # (base64-of-pickle.dumps decoded to utf-8); ``decode_metadata``
            # expects the same. The assertion was inherited from an older
            # bytes-typed format and never updated.
            assert isinstance(serialized_metadata, str)
            metadata = TorchTensorRTModule.decode_metadata(serialized_metadata)
            self.settings = metadata["settings"]
            self.symbolic_shape_expressions = metadata["inout_symexprs"]

            # RuntimeSettings are NOT serialized; restore defaults. Caller can
            # reapply via ``mod.runtime_settings = ...`` (per submodule) or a CM after load.
            self._runtime_settings = RuntimeSettings()
            # Mirror the settings reset on the implicit cache handle so a
            # stale wrapper from prior use doesn't survive load_state_dict and
            # silently write the fresh engine's cache bytes to the old path.
            self._implicit_cache_handle = None
            if not ENABLED_FEATURES.torch_tensorrt_runtime:
                from torch_tensorrt.dynamo.runtime._TRTEngine import TRTEngine

                self.engine = TRTEngine(serialized_engine_info)
            else:
                self.engine = torch.classes.tensorrt.Engine(serialized_engine_info)

            self.engine.set_output_tensors_as_unowned(
                metadata["output_tensors_are_unowned"]
            )
        else:
            self.engine = None
            self.settings = CompilationSettings()
            self.hardware_compatible = False

        self.input_binding_names = state[2]
        self.output_binding_names = state[3]
        self.target_device = self._resolve_target_device()

    def set_pre_allocated_outputs(self, enable: bool) -> None:
        self.get_engine().use_pre_allocated_outputs = enable

    @property
    def pre_allocated_outputs(self) -> Any:
        """Pre-allocated output tensors currently held by the underlying engine."""
        if self.engine is None:
            return []
        return getattr(self.engine, "pre_allocated_outputs", [])

    def set_use_output_allocator(self, enable: bool) -> None:
        self.get_engine().use_output_allocator_outputs = enable

    def forward(self, *inputs: Any) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        """Run the TensorRT engine on GPU tensors (non-tensor args are cast to CUDA tensors).

        Note: callers are responsible for ensuring the engine has been set up;
        the hot path intentionally omits a ``self.engine is None`` guard so
        that a properly-bound module avoids the per-call attribute check.
        """
        target = self.target_device
        binding_names = self.input_binding_names
        # len-check inlined (cheaper than keeping an f-string around the hot path)
        if len(inputs) != len(binding_names):
            raise AssertionError(
                f"Wrong number of inputs, expected {len(binding_names)} got {len(inputs)}."
            )

        # If the inputs are not Torch Tensors, which can occur in scenarios such as shape tensors
        # which are outputs of a preceding Torch subgraph (where the Dynamic input may be an integer)
        # directly cast the input to a Torch Tensor.
        #
        # This also avoids the need for type-checking inputs, since they are now explicitly casted to Torch tensors
        input_tensors: List[torch.Tensor] = []
        for i in inputs:
            if isinstance(i, torch.Tensor):
                if not i.is_cuda:
                    logger.warning(
                        "Input tensor is not on a CUDA device. Moving it to CUDA automatically. "
                        "For best performance, ensure all inputs are on the correct CUDA device "
                        "before calling the TensorRT engine (e.g. tensor.cuda() or tensor.to(device))."
                    )
                    i = i.cuda()
                input_tensors.append(i)
            else:
                input_tensors.append(torch.tensor(i).cuda())

        outputs = torch.ops.tensorrt.execute_engine(input_tensors, self.engine)
        if len(outputs) == 1:
            return outputs[0]

        return tuple(outputs)

    def enable_profiling(
        self,
        profiling_results_dir: Optional[str] = None,
        profile_format: str = "perfetto",
    ) -> None:
        """Enable engine profiling (optional path prefix and format for tracing output)."""
        engine = self.get_engine()

        if profiling_results_dir is not None:
            engine.profile_path_prefix = profiling_results_dir

        engine.enable_profiling()
        if hasattr(engine, "set_profile_format"):
            engine.set_profile_format(profile_format)
        self.profiling_enabled = True

    def set_output_tensors_as_unowned(self, enabled: bool) -> None:
        self.get_engine().set_output_tensors_as_unowned(enabled)

    def are_output_tensors_unowned(self) -> bool:
        return bool(self.get_engine().are_output_tensors_unowned())

    def disable_profiling(self) -> None:
        """Disable engine profiling and clear the profiling flag on this module."""
        engine = self.get_engine()
        engine.disable_profiling()
        self.profiling_enabled = False

    def get_layer_info(self) -> str:
        """Get a JSON string containing the layer information encoded by the TensorRT engine in this module

        Returns:

            str: A JSON string which contains the layer information of the engine incapsulated in this module
        """
        layer_info: str = self.get_engine().get_engine_layer_info()
        return layer_info

    def dump_layer_info(self) -> None:
        """Dump layer information encoded by the TensorRT engine in this module to STDOUT"""
        self.get_engine().dump_engine_layer_info()
