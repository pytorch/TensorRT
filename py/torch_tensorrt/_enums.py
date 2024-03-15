from typing import Self, Optional

from enum import Enum, auto
import torch
import numpy as np
import tensorrt as trt
import logging

from torch_tensorrt._features import ENABLED_FEATURES

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
    bool = auto()

    # TODO: Enable FP8 and BF16
    #f8 = auto()
    #bf16 = auto()

    uint8 = u8

    int = i32
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
    #float8 = f8
    #fp8 = f8

    # TODO: Enable when BF16 is enabled
    #bfloat16 = bf16


    @staticmethod
    def _from(t: torch.dtype | trt.DataType | np.dtype, use_default=False) -> Self:
        # TODO: Ideally implemented with match statement but need to wait for Py39 EoL
        if isinstance(t, torch.dtype):
            if t == torch.uint8:
                return dtype.u8
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
                return dtype.bool
            elif use_default:
                logging.warning("Given dtype that does not have direct mapping to Torch-TensorRT supported types ({}), defaulting to torch_tensorrt.dtype.float".format(t))
                return dtype.float
            else:
                raise TypeError(
                    "Provided an unsupported data type as an input data type (support: bool, int32, long, half, float), got: "
                    + str(t)
                )
        elif isinstance(t, trt.DataType):
            if t == trt.uint8:
                return dtype.u8
            elif t == trt.int8:
                return dtype.i8
            elif t == trt.int32:
                return dtype.i32
            elif t == trt.float16:
                return dtype.f16
            elif t == trt.float32:
                return dtype.f32
            elif (trt.__version__ >= "7.0" and t == torch.bool):
                return dtype.bool
            else:
                raise TypeError(
                    "Provided an unsupported data type as an input data type (support: bool, int32, half, float), got: "
                    + str(t)
                )

        elif isinstance(t, np.dtype):
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
                return dtype.bool
            elif use_default:
                logging.warning("Given dtype that does not have direct mapping to Torch-TensorRT supported types ({}), defaulting to torch_tensorrt.dtype.float".format(t))
                return dtype.float
            else:
                raise TypeError(
                    "Provided an unsupported data type as an input data type (support: bool, int32, long, half, float), got: "
                    + str(t)
                )

        elif isinstance(t, dtype):
            return t

        elif ENABLED_FEATURES.torchscript_frontend:
            from torch_tensorrt import _C
            if isinstance(t, _C.dtype):
                if t == _C.dtype.long:
                    return dtype.i64
                elif t ==  _C.dtype.int32:
                    return dtype.i32
                elif t == _C.dtype.half:
                    return dtype.f16
                elif t == _C.dtype.float:
                    return dtype.f32
                elif t == _C.dtype.double:
                    return dtype.f64
                elif t == _C.dtype.bool:
                    return dtype.bool
                elif t == _C.dtype.unknown:
                    return dtype.unknown
                else:
                    raise TypeError(
                        "Provided an unsupported data type as an input data type (support: bool, int32, long, half, float), got: "
                        + str(t)
                    )
        else:
            raise TypeError(
                "Provided unsupported source type for dtype conversion"
            )

    @staticmethod
    def try_from(t: torch.dtype | trt.DataType | np.dtype) -> Optional[Self]:
        try:
            casted_format = dtype._from(t)
            return casted_format
        except (ValueError, TypeError) as e:
            logging.debug(e)
            return None

    def to(self, t: type, use_default: bool=False) -> torch.dtype | trt.DataType | np.dtype:
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
            elif self == dtype.bool:
                return torch.bool
            elif use_default:
                logging.warning("Given dtype that does not have direct mapping to torch ({}), defaulting to torch.float".format(self))
                return torch.float
            else:
                raise TypeError("Unsupported torch dtype")

        elif t == trt.DataType:
            if self == dtype.u8:
                return trt.DataType.UINT8
            if self == dtype.i8:
                return trt.DataType.INT8
            elif self == dtype.i32:
                return trt.DataType.INT32
            elif self == dtype.f16:
                return trt.DataType.HALF
            elif self == dtype.f32:
                return trt.DataType.FLOAT
            elif self == dtype.bool:
                return trt.DataType.BOOL
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
            elif self == dtype.bool:
                return np.bool
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
                elif self == dtype.i32:
                    return _C.dtype.int32
                elif self == dtype.f16:
                    return _C.dtype.half
                elif self == dtype.f32:
                    return _C.dtype.float
                elif self == dtype.f64:
                    return _C.dtype.double
                elif self == dtype.bool:
                    return _C.dtype.bool
                elif self == dtype.unknown:
                    return _C.dtype.unknown
                else:
                    raise TypeError(
                        "Provided an unsupported data type as an input data type (support: bool, int32, long, half, float), got: "
                        + str(self)
                    )
        else:
            raise TypeError(
                "Provided unsupported destination type for dtype conversion"
            )

    def try_to(self, t: type, use_default: bool=False) -> Optional[torch.dtype | trt.DataType | np.dtype]:
        try:
            print(self)
            casted_format = self.to(t, use_default)
            return casted_format
        except (ValueError, TypeError) as e:
            logging.debug(e)
            return None

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

    @staticmethod
    def _from(f: torch.memory_format | trt.TensorFormat) -> Self:
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
                    "Provided an unsupported memory format for tensor, got: "
                    + str(dtype)
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
                    "Provided an unsupported tensor format for tensor, got: "
                    + str(dtype)
                )

        elif isinstance(f, memory_format):
            return f

        elif ENABLED_FEATURES.torchscript_frontend:
            from torch_tensorrt import _C
            if isinstance(f, _C.TensorFormat):
                if f == _C.TensorFormat.contiguous:
                    return memory_format.contiguous
                if f == _C.TensorFormat.channels_last:
                    return memory_format.channels_last
                else:
                    raise ValueError(
                        "Provided an unsupported tensor format (support: NCHW/contiguous_format, NHWC/channel_last)"
                    )
        else:
            raise TypeError(
                "Provided unsupported source type for memory_format conversion"
            )

    @staticmethod
    def try_from(f: torch.memory_format | trt.TensorFormat) -> Optional[Self]:
        try:
            casted_format = memory_format._from(f)
            return casted_format
        except (ValueError, TypeError) as e:
            logging.debug(e)
            return None


    def to(self, t: type) -> torch.memory_format | trt.TensorFormat:
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
            if self ==  memory_format.linear:
                return trt.TensorFormat.LINEAR
            elif self == memory_format.chw2:
                return trt.TensorFormat.CHW2
            elif self ==  memory_format.hwc8:
                return trt.TensorFormat.HWC8
            elif self == memory_format.chw4:
                return trt.TensorFormat.CHW4
            elif self ==  memory_format.chw16:
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
                raise TypeError("Unsupported tensorrt dtype")

        elif (t == memory_format):
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

        else:
            raise TypeError(
                "Provided unsupported destination type for memory format conversion"
            )

    def try_to(self, t: type) -> Optional[torch.memory_format | trt.TensorFormat]:
        try:
            casted_format = self.to(t)
            return casted_format
        except (ValueError, TypeError) as e:
            logging.debug(e)
            return None

class DeviceType(Enum):
    GPU = auto()
    DLA = auto()

    @staticmethod
    def _from(d: trt.DeviceType) -> Self:
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
                if d == _C.DeviceType.DLA:
                    return DeviceType.DLA
                else:
                    raise ValueError(
                        "Provided an unsupported device type (support: GPU/DLA)"
                    )

    @staticmethod
    def try_from(d: trt.DeviceType) -> Optional[Self]:
        try:
            casted_format = DeviceType._from(d)
            return casted_format
        except (ValueError, TypeError) as e:
            logging.debug(e)
            return None

    def to(self, t: type) -> trt.DeviceType:
        if t == trt.DeviceType:
            if self == DeviceType.GPU:
                return trt.DeviceType.GPU
            elif self == DeviceType.DLA:
                return trt.DeviceType.DLA
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
        else:
           raise TypeError(
                "Provided unsupported destination type for device type conversion"
            )

    def try_to(self, t: type) -> Optional[trt.DeviceType]:
        try:
            casted_format = self.to(t)
            return casted_format
        except (ValueError, TypeError) as e:
            logging.debug(e)
            return None