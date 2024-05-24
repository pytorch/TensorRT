from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Any, Optional, Type, Union

import numpy as np
import torch
from torch_tensorrt._features import ENABLED_FEATURES

import tensorrt as trt


class dtype(Enum):
    """Enum to set supported dtypes in the compiler"""

    # Supported types in Torch-TensorRT
    unknown = auto()
    u8 = auto()
    i8 = auto()
    i32 = auto()
    i64 = auto()
    f16 = auto()
    f32 = auto()
    f64 = auto()
    b = auto()
    bf16 = auto()
    # TODO: Enable FP8
    # f8 = auto()

    uint8 = u8
    int8 = i8

    int32 = i32

    long = i64
    int64 = i64

    half = f16
    fp16 = f16
    float16 = f16

    float = f32
    fp32 = f32
    float32 = f32

    double = f64
    fp64 = f64
    float64 = f64

    # TODO: Enable when FP8 is enabled
    # float8 = f8
    # fp8 = f8

    bfloat16 = bf16

    @staticmethod
    def _is_np_obj(t: Any) -> bool:
        if isinstance(t, np.dtype):
            return True
        elif isinstance(t, type):
            if issubclass(t, np.generic):
                return True
        return False

    @classmethod
    def _from(
        cls,
        t: Union[torch.dtype, trt.DataType, np.dtype, dtype, type],
        use_default: bool = False,
    ) -> dtype:
        # TODO: Ideally implemented with match statement but need to wait for Py39 EoL
        if isinstance(t, torch.dtype):
            if t == torch.uint8:
                return dtype.u8
            elif t == torch.int8:
                return dtype.i8
            elif t == torch.long:
                return dtype.i64
            elif t == torch.int32:
                return dtype.i32
            elif t == torch.half:
                return dtype.f16
            elif t == torch.float:
                return dtype.f32
            elif t == torch.float64:
                return dtype.f64
            elif t == torch.bool:
                return dtype.b
            elif t == torch.bfloat16:
                return dtype.bf16
            elif use_default:
                logging.warning(
                    f"Given dtype that does not have direct mapping to Torch-TensorRT supported types ({t}), defaulting to torch_tensorrt.dtype.float"
                )
                return dtype.float
            else:
                raise TypeError(
                    f"Provided an unsupported data type as a data type for translation (support: bool, int, long, half, float, bfloat16), got: {t}"
                )
        elif isinstance(t, trt.DataType):
            if t == trt.DataType.UINT8:
                return dtype.u8
            elif t == trt.DataType.INT8:
                return dtype.i8
            elif t == trt.DataType.INT32:
                return dtype.i32
            elif t == trt.DataType.INT64:
                return dtype.i64
            elif t == trt.DataType.HALF:
                return dtype.f16
            elif t == trt.DataType.FLOAT:
                return dtype.f32
            elif t == trt.DataType.BOOL:
                return dtype.b
            elif t == trt.DataType.BF16:
                return dtype.bf16
            else:
                raise TypeError(
                    f"Provided an unsupported data type as a data type for translation (support: bool, int, half, float, bfloat16), got: {t}"
                )

        elif dtype._is_np_obj(t):
            if t == np.uint8:
                return dtype.u8
            elif t == np.int8:
                return dtype.i8
            elif t == np.int32:
                return dtype.i32
            elif t == np.int64:
                return dtype.i64
            elif t == np.float16:
                return dtype.f16
            elif t == np.float32:
                return dtype.f32
            elif t == np.float64:
                return dtype.f64
            elif t == np.bool:
                return dtype.b
            elif use_default:
                logging.warning(
                    f"Given dtype that does not have direct mapping to Torch-TensorRT supported types ({t}), defaulting to torch_tensorrt.dtype.float"
                )
                return dtype.float
            else:
                raise TypeError(
                    "Provided an unsupported data type as an input data type (support: bool, int, long, half, float, bfloat16), got: "
                    + str(t)
                )

        elif isinstance(t, dtype):
            return t

        elif ENABLED_FEATURES.torchscript_frontend:
            from torch_tensorrt import _C

            if isinstance(t, _C.dtype):
                if t == _C.dtype.long:
                    return dtype.i64
                elif t == _C.dtype.int32:
                    return dtype.i32
                elif t == _C.dtype.int8:
                    return dtype.i8
                elif t == _C.dtype.half:
                    return dtype.f16
                elif t == _C.dtype.float:
                    return dtype.f32
                elif t == _C.dtype.double:
                    return dtype.f64
                elif t == _C.dtype.bool:
                    return dtype.b
                elif t == _C.dtype.unknown:
                    return dtype.unknown
                else:
                    raise TypeError(
                        f"Provided an unsupported data type as an input data type (support: bool, int32, long, half, float), got: {t}"
                    )
        # else: # commented out for mypy
        raise TypeError(
            f"Provided unsupported source type for dtype conversion (got: {t})"
        )

    @classmethod
    def try_from(
        cls,
        t: Union[torch.dtype, trt.DataType, np.dtype, dtype],
        use_default: bool = False,
    ) -> Optional[dtype]:
        try:
            casted_format = dtype._from(t, use_default=use_default)
            return casted_format
        except (ValueError, TypeError) as e:
            logging.debug(
                f"Conversion from {t} to torch_tensorrt.dtype failed", exc_info=True
            )
            return None

    def to(
        self,
        t: Union[Type[torch.dtype], Type[trt.DataType], Type[np.dtype], Type[dtype]],
        use_default: bool = False,
    ) -> Union[torch.dtype, trt.DataType, np.dtype, dtype]:
        # TODO: Ideally implemented with match statement but need to wait for Py39 EoL
        if t == torch.dtype:
            if self == dtype.u8:
                return torch.uint8
            elif self == dtype.i8:
                return torch.int8
            elif self == dtype.i32:
                return torch.int
            elif self == dtype.i64:
                return torch.long
            elif self == dtype.f16:
                return torch.half
            elif self == dtype.f32:
                return torch.float
            elif self == dtype.f64:
                return torch.double
            elif self == dtype.b:
                return torch.bool
            elif self == dtype.bf16:
                return torch.bfloat16
            elif use_default:
                logging.warning(
                    f"Given dtype that does not have direct mapping to torch ({self}), defaulting to torch.float"
                )
                return torch.float
            else:
                raise TypeError(f"Unsupported torch dtype (had: {self})")

        elif t == trt.DataType:
            if self == dtype.u8:
                return trt.DataType.UINT8
            if self == dtype.i8:
                return trt.DataType.INT8
            elif self == dtype.i32:
                return trt.DataType.INT32
            elif self == dtype.i64:
                return trt.DataType.INT64
            elif self == dtype.f16:
                return trt.DataType.HALF
            elif self == dtype.f32:
                return trt.DataType.FLOAT
            elif self == dtype.b:
                return trt.DataType.BOOL
            elif self == dtype.bf16:
                return trt.DataType.BF16
            elif use_default:
                return trt.DataType.FLOAT
            else:
                raise TypeError("Unsupported tensorrt dtype")

        elif t == np.dtype:
            if self == dtype.u8:
                return np.uint8
            elif self == dtype.i8:
                return np.int8
            elif self == dtype.i32:
                return np.int32
            elif self == dtype.i64:
                return np.int64
            elif self == dtype.f16:
                return np.float16
            elif self == dtype.f32:
                return np.float32
            elif self == dtype.f64:
                return np.float64
            elif self == dtype.b:
                return np.bool_
            elif use_default:
                return np.float32
            else:
                raise TypeError("Unspported numpy dtype")

        elif t == dtype:
            return self

        elif ENABLED_FEATURES.torchscript_frontend:
            from torch_tensorrt import _C

            if t == _C.dtype:
                if self == dtype.i64:
                    return _C.dtype.long
                elif self == dtype.i8:
                    return _C.dtype.int8
                elif self == dtype.i32:
                    return _C.dtype.int32
                elif self == dtype.f16:
                    return _C.dtype.half
                elif self == dtype.f32:
                    return _C.dtype.float
                elif self == dtype.f64:
                    return _C.dtype.double
                elif self == dtype.b:
                    return _C.dtype.bool
                elif self == dtype.unknown:
                    return _C.dtype.unknown
                else:
                    raise TypeError(
                        f"Provided an unsupported data type as an input data type (support: bool, int32, long, half, float), got: {self}"
                    )
        # else: # commented out for mypy
        raise TypeError(
            f"Provided unsupported destination type for dtype conversion {t}"
        )

    def try_to(
        self,
        t: Union[Type[torch.dtype], Type[trt.DataType], Type[np.dtype], Type[dtype]],
        use_default: bool,
    ) -> Optional[Union[torch.dtype, trt.DataType, np.dtype, dtype]]:
        try:
            casted_format = self.to(t, use_default)
            return casted_format
        except (ValueError, TypeError) as e:
            logging.debug(
                f"torch_tensorrt.dtype conversion to target type {t} failed",
                exc_info=True,
            )
            return None

    def __eq__(self, other: Union[torch.dtype, trt.DataType, np.dtype, dtype]) -> bool:
        other_ = dtype._from(other)
        return bool(self.value == other_.value)

    def __hash__(self) -> int:
        return hash(self.value)

    # Putting aliases here that mess with mypy
    bool = b
    int = i32


class memory_format(Enum):

    # TensorRT supported memory layouts
    linear = auto()
    chw2 = auto()
    hwc8 = auto()
    chw4 = auto()
    chw16 = auto()
    chw32 = auto()
    dhwc8 = auto()
    cdhw32 = auto()
    hwc = auto()
    dla_linear = auto()
    dla_hwc4 = auto()
    hwc16 = auto()
    dhwc = auto()

    # PyTorch aliases for TRT layouts
    contiguous = linear
    channels_last = hwc
    channels_last_3d = dhwc

    @classmethod
    def _from(
        cls, f: Union[torch.memory_format, trt.TensorFormat, memory_format]
    ) -> memory_format:
        # TODO: Ideally implemented with match statement but need to wait for Py39 EoL
        if isinstance(f, torch.memory_format):
            if f == torch.contiguous_format:
                return memory_format.contiguous
            elif f == torch.channels_last:
                return memory_format.channels_last
            elif f == torch.channels_last_3d:
                return memory_format.channels_last_3d
            else:
                raise TypeError(
                    f"Provided an unsupported memory format for tensor, got: {dtype}"
                )

        elif isinstance(f, trt.DataType):
            if f == trt.TensorFormat.LINEAR:
                return memory_format.linear
            elif f == trt.TensorFormat.CHW2:
                return memory_format.chw2
            elif f == trt.TensorFormat.HWC8:
                return memory_format.hwc8
            elif f == trt.TensorFormat.CHW4:
                return memory_format.chw4
            elif f == trt.TensorFormat.CHW16:
                return memory_format.chw16
            elif f == trt.TensorFormat.CHW32:
                return memory_format.chw32
            elif f == trt.TensorFormat.DHWC8:
                return memory_format.dhwc8
            elif f == trt.TensorFormat.CDHW32:
                return memory_format.cdhw32
            elif f == trt.TensorFormat.HWC:
                return memory_format.hwc
            elif f == trt.TensorFormat.DLA_LINEAR:
                return memory_format.dla_linear
            elif f == trt.TensorFormat.DLA_HWC4:
                return memory_format.dla_hwc4
            elif f == trt.TensorFormat.HWC16:
                return memory_format.hwc16
            elif f == trt.TensorFormat.DHWC:
                return memory_format.dhwc
            else:
                raise TypeError(
                    f"Provided an unsupported tensor format for tensor, got: {dtype}"
                )

        elif isinstance(f, memory_format):
            return f

        elif ENABLED_FEATURES.torchscript_frontend:
            from torch_tensorrt import _C

            if isinstance(f, _C.TensorFormat):
                if f == _C.TensorFormat.contiguous:
                    return memory_format.contiguous
                elif f == _C.TensorFormat.channels_last:
                    return memory_format.channels_last
                else:
                    raise ValueError(
                        "Provided an unsupported tensor format (support: NCHW/contiguous_format, NHWC/channel_last)"
                    )
        # else: # commented out for mypy
        raise TypeError("Provided unsupported source type for memory_format conversion")

    @classmethod
    def try_from(
        cls, f: Union[torch.memory_format, trt.TensorFormat, memory_format]
    ) -> Optional[memory_format]:
        try:
            casted_format = memory_format._from(f)
            return casted_format
        except (ValueError, TypeError) as e:
            logging.debug(
                f"Conversion from {f} to torch_tensorrt.memory_format failed",
                exc_info=True,
            )
            return None

    def to(
        self,
        t: Union[
            Type[torch.memory_format], Type[trt.TensorFormat], Type[memory_format]
        ],
    ) -> Union[torch.memory_format, trt.TensorFormat, memory_format]:
        if t == torch.memory_format:
            if self == memory_format.contiguous:
                return torch.contiguous_format
            elif self == memory_format.channels_last:
                return torch.channels_last
            elif self == memory_format.channels_last_3d:
                return torch.channels_last_3d
            else:
                raise TypeError("Unsupported torch dtype")

        elif t == trt.TensorFormat:
            if self == memory_format.linear:
                return trt.TensorFormat.LINEAR
            elif self == memory_format.chw2:
                return trt.TensorFormat.CHW2
            elif self == memory_format.hwc8:
                return trt.TensorFormat.HWC8
            elif self == memory_format.chw4:
                return trt.TensorFormat.CHW4
            elif self == memory_format.chw16:
                return trt.TensorFormat.CHW16
            elif self == memory_format.chw32:
                return trt.TensorFormat.CHW32
            elif self == memory_format.dhwc8:
                return trt.TensorFormat.DHWC8
            elif self == memory_format.cdhw32:
                return trt.TensorFormat.CDHW32
            elif self == memory_format.hwc:
                return trt.TensorFormat.HWC
            elif self == memory_format.dla_linear:
                return trt.TensorFormat.DLA_LINEAR
            elif self == memory_format.dla_hwc4:
                return trt.TensorFormat.DLA_HWC4
            elif self == memory_format.hwc16:
                return trt.TensorFormat.HWC16
            elif self == memory_format.dhwc:
                return trt.TensorFormat.DHWC
            else:
                raise TypeError("Unsupported tensorrt memory format")

        elif t == memory_format:
            return self

        elif ENABLED_FEATURES.torchscript_frontend:
            from torch_tensorrt import _C

            if t == _C.TensorFormat:
                if self == memory_format.contiguous:
                    return _C.TensorFormat.contiguous
                elif self == memory_format.channels_last:
                    return _C.TensorFormat.channels_last
                else:
                    raise ValueError(
                        "Provided an unsupported tensor format (support: NCHW/contiguous_format, NHWC/channel_last)"
                    )
        # else: # commented out for mypy
        raise TypeError(
            "Provided unsupported destination type for memory format conversion"
        )

    def try_to(
        self,
        t: Union[
            Type[torch.memory_format], Type[trt.TensorFormat], Type[memory_format]
        ],
    ) -> Optional[Union[torch.memory_format, trt.TensorFormat, memory_format]]:
        try:
            casted_format = self.to(t)
            return casted_format
        except (ValueError, TypeError) as e:
            logging.debug(
                f"torch_tensorrt.memory_format conversion to target type {t} failed",
                exc_info=True,
            )
            return None

    def __eq__(
        self, other: Union[torch.memory_format, trt.TensorFormat, memory_format]
    ) -> bool:
        other_ = memory_format._from(other)
        return self.value == other_.value

    def __hash__(self) -> int:
        return hash(self.value)


class DeviceType(Enum):
    UNKNOWN = auto()
    GPU = auto()
    DLA = auto()

    @classmethod
    def _from(cls, d: Union[trt.DeviceType, DeviceType]) -> DeviceType:
        if isinstance(d, trt.DeviceType):
            if d == trt.DeviceType.GPU:
                return DeviceType.GPU
            elif d == trt.DeviceType.DLA:
                return DeviceType.DLA
            else:
                raise ValueError(
                    "Provided an unsupported device type (support: GPU/DLA)"
                )

        elif isinstance(d, DeviceType):
            return d

        elif ENABLED_FEATURES.torchscript_frontend:
            from torch_tensorrt import _C

            if isinstance(d, _C.DeviceType):
                if d == _C.DeviceType.GPU:
                    return DeviceType.GPU
                elif d == _C.DeviceType.DLA:
                    return DeviceType.DLA
                else:
                    raise ValueError(
                        "Provided an unsupported device type (support: GPU/DLA)"
                    )
        # else: # commented out for mypy
        raise TypeError("Provided unsupported source type for DeviceType conversion")

    @classmethod
    def try_from(cls, d: Union[trt.DeviceType, DeviceType]) -> Optional[DeviceType]:
        try:
            casted_format = DeviceType._from(d)
            return casted_format
        except (ValueError, TypeError) as e:
            logging.debug(
                f"Conversion from {d} to torch_tensorrt.DeviceType failed",
                exc_info=True,
            )
            return None

    def to(
        self,
        t: Union[Type[trt.DeviceType], Type[DeviceType]],
        use_default: bool = False,
    ) -> Union[trt.DeviceType, DeviceType]:
        if t == trt.DeviceType:
            if self == DeviceType.GPU:
                return trt.DeviceType.GPU
            elif self == DeviceType.DLA:
                return trt.DeviceType.DLA
            elif use_default:
                return trt.DeviceType.GPU
            else:
                raise ValueError(
                    "Provided an unsupported device type (support: GPU/DLA)"
                )

        elif t == DeviceType:
            return self

        elif ENABLED_FEATURES.torchscript_frontend:
            from torch_tensorrt import _C

            if t == _C.DeviceType:
                if self == DeviceType.GPU:
                    return _C.DeviceType.GPU
                elif self == DeviceType.DLA:
                    return _C.DeviceType.DLA
                else:
                    raise ValueError(
                        "Provided an unsupported device type (support: GPU/DLA)"
                    )
        # else: # commented out for mypy
        raise TypeError(
            "Provided unsupported destination type for device type conversion"
        )

    def try_to(
        self,
        t: Union[Type[trt.DeviceType], Type[DeviceType]],
        use_default: bool = False,
    ) -> Optional[Union[trt.DeviceType, DeviceType]]:
        try:
            casted_format = self.to(t, use_default=use_default)
            return casted_format
        except (ValueError, TypeError) as e:
            logging.debug(
                f"torch_tensorrt.DeviceType conversion to target type {t} failed",
                exc_info=True,
            )
            return None

    def __eq__(self, other: Union[trt.DeviceType, DeviceType]) -> bool:
        other_ = DeviceType._from(other)
        return bool(self.value == other_.value)

    def __hash__(self) -> int:
        return hash(self.value)


class EngineCapability(Enum):
    STANDARD = auto()
    SAFETY = auto()
    DLA_STANDALONE = auto()

    @classmethod
    def _from(
        cls, c: Union[trt.EngineCapability, EngineCapability]
    ) -> EngineCapability:
        if isinstance(c, trt.EngineCapability):
            if c == trt.EngineCapability.STANDARD:
                return EngineCapability.STANDARD
            elif c == trt.EngineCapability.SAFETY:
                return EngineCapability.SAFETY
            elif c == trt.EngineCapability.DLA_STANDALONE:
                return EngineCapability.DLA_STANDALONE
            else:
                raise ValueError("Provided an unsupported engine capability")

        elif isinstance(c, EngineCapability):
            return c

        elif ENABLED_FEATURES.torchscript_frontend:
            from torch_tensorrt import _C

            if isinstance(c, _C.EngineCapability):
                if c == _C.EngineCapability.STANDARD:
                    return EngineCapability.STANDARD
                elif c == _C.EngineCapability.SAFETY:
                    return EngineCapability.SAFETY
                elif c == _C.EngineCapability.DLA_STANDALONE:
                    return EngineCapability.DLA_STANDALONE
                else:
                    raise ValueError("Provided an unsupported engine capability")
        # else: # commented out for mypy
        raise TypeError(
            "Provided unsupported source type for EngineCapability conversion"
        )

    @classmethod
    def try_from(
        c: Union[trt.EngineCapability, EngineCapability]
    ) -> Optional[EngineCapability]:
        try:
            casted_format = EngineCapability._from(c)
            return casted_format
        except (ValueError, TypeError) as e:
            logging.debug(
                f"Conversion from {c} to torch_tensorrt.EngineCapablity failed",
                exc_info=True,
            )
            return None

    def to(
        self, t: Union[Type[trt.EngineCapability], Type[EngineCapability]]
    ) -> Union[trt.EngineCapability, EngineCapability]:
        if t == trt.EngineCapability:
            if self == EngineCapability.STANDARD:
                return trt.EngineCapability.STANDARD
            elif self == EngineCapability.SAFETY:
                return trt.EngineCapability.SAFETY
            elif self == EngineCapability.DLA_STANDALONE:
                return trt.EngineCapability.DLA_STANDALONE
            else:
                raise ValueError("Provided an unsupported engine capability")

        elif t == EngineCapability:
            return self

        elif ENABLED_FEATURES.torchscript_frontend:
            from torch_tensorrt import _C

            if t == _C.EngineCapability:
                if self == EngineCapability.STANDARD:
                    return _C.EngineCapability.STANDARD
                elif self == EngineCapability.SAFETY:
                    return _C.EngineCapability.SAFETY
                elif self == EngineCapability.DLA_STANDALONE:
                    return _C.EngineCapability.DLA_STANDALONE
                else:
                    raise ValueError("Provided an unsupported engine capability")
        # else: # commented out for mypy
        raise TypeError(
            "Provided unsupported destination type for engine capablity type conversion"
        )

    def try_to(
        self, t: Union[Type[trt.EngineCapability], Type[EngineCapability]]
    ) -> Optional[Union[trt.EngineCapability, EngineCapability]]:
        try:
            casted_format = self.to(t)
            return casted_format
        except (ValueError, TypeError) as e:
            logging.debug(
                f"torch_tensorrt.EngineCapablity conversion to target type {t} failed",
                exc_info=True,
            )
            return None

    def __eq__(self, other: Union[trt.EngineCapability, EngineCapability]) -> bool:
        other_ = EngineCapability._from(other)
        return bool(self.value == other_.value)

    def __hash__(self) -> int:
        return hash(self.value)
