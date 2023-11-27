from enum import Enum, auto

# from tensorrt import DeviceType  # noqa: F401


class dtype(Enum):
    float32 = auto()
    half = auto()
    float16 = auto()
    float = auto()
    int8 = auto()
    int32 = auto()
    int = auto()
    int64 = auto()
    float64 = auto()
    bool = auto()
    unknown = auto()


class TensorFormat(Enum):
    contiguous = auto()
    channels_last = auto()


class EngineCapability(Enum):
    safe_gpu = auto()
    safe_dla = auto()
    default = auto()
