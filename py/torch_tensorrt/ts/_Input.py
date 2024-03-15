from typing import Any
import torch
from torch_tensorrt.ts import _enums

from torch_tensorrt import _C
from torch_tensorrt._Input import Input
from torch_tensorrt._enums import dtype, memory_format


class TorchScriptInput(Input):
    """
    Defines an input to a module in terms of expected shape, data type and tensor format.

    Attributes:
        shape_mode (torch_tensorrt.Input._ShapeMode): Is input statically or dynamically shaped
        shape (Tuple or Dict): Either a single Tuple or a dict of tuples defining the input shape.
            Static shaped inputs will have a single tuple. Dynamic inputs will have a dict of the form
            ``{
                "min_shape": Tuple,
                "opt_shape": Tuple,
                "max_shape": Tuple
            }``
        dtype (torch_tensorrt.dtype): The expected data type of the input tensor (default: torch_tensorrt.dtype.float32)
        format (torch_tensorrt.TensorFormat): The expected format of the input tensor (default: torch_tensorrt.TensorFormat.NCHW)
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """__init__ Method for torch_tensorrt.Input

        Input accepts one of a few construction patterns

        Args:
            shape (Tuple or List, optional): Static shape of input tensor

        Keyword Arguments:
            shape (Tuple or List, optional): Static shape of input tensor
            min_shape (Tuple or List, optional): Min size of input tensor's shape range
                Note: All three of min_shape, opt_shape, max_shape must be provided, there must be no positional arguments, shape must not be defined and implictly this sets Input's shape_mode to DYNAMIC
            opt_shape (Tuple or List, optional): Opt size of input tensor's shape range
                Note: All three of min_shape, opt_shape, max_shape must be provided, there must be no positional arguments, shape must not be defined and implictly this sets Input's shape_mode to DYNAMIC
            max_shape (Tuple or List, optional): Max size of input tensor's shape range
                Note: All three of min_shape, opt_shape, max_shape must be provided, there must be no positional arguments, shape must not be defined and implictly this sets Input's shape_mode to DYNAMIC
            dtype (torch.dtype or torch_tensorrt.dtype): Expected data type for input tensor (default: torch_tensorrt.dtype.float32)
            format (torch.memory_format or torch_tensorrt.TensorFormat): The expected format of the input tensor (default: torch_tensorrt.TensorFormat.NCHW)
            tensor_domain (Tuple(float, float), optional): The domain of allowed values for the tensor, as interval notation: [tensor_domain[0], tensor_domain[1]).
                Note: Entering "None" (or not specifying) will set the bound to [0, 2)

        Examples:
            - Input([1,3,32,32], dtype=torch.float32, format=torch.channel_last)
            - Input(shape=(1,3,32,32), dtype=torch_tensorrt.dtype.int32, format=torch_tensorrt.TensorFormat.NCHW)
            - Input(min_shape=(1,3,32,32), opt_shape=[2,3,32,32], max_shape=(3,3,32,32)) #Implicitly dtype=torch_tensorrt.dtype.float32, format=torch_tensorrt.TensorFormat.NCHW
        """
        super().__init__(*args, **kwargs)

    @staticmethod
    def _parse_dtype(d: Any) -> _enums.dtype:
        if isinstance(d, torch.dtype):
            if d == torch.long:
                return _enums.dtype.long
            elif d == torch.int32:
                return _enums.dtype.int32
            elif d == torch.half:
                return _enums.dtype.half
            elif d == torch.float:
                return _enums.dtype.float
            elif d == torch.float64:
                return _enums.dtype.double
            elif d == torch.bool:
                return _enums.dtype.bool
            else:
                raise TypeError(
                    "Provided an unsupported data type as an input data type (support: bool, int32, long, half, float), got: "
                    + str(d)
                )

        elif isinstance(d, dtype):
            return d.to(_enums.dtype)

        elif isinstance(d, _enums.dtype):
            return d

        else:
            raise TypeError(
                "Input data type needs to be specified with a torch.dtype or a torch_tensorrt.dtype, got: "
                + str(type(d))
            )

    @staticmethod
    def _to_torch_dtype(dtype: _enums.dtype) -> torch.dtype:
        if dtype == _enums.dtype.long:
            return torch.long
        elif dtype == _enums.dtype.int32:
            return torch.int32
        elif dtype == _enums.dtype.half:
            return torch.half
        elif dtype == _enums.dtype.float:
            return torch.float
        elif dtype == _enums.dtype.bool:
            return torch.bool
        elif dtype == _enums.dtype.double:
            return torch.float64
        else:
            # Default torch_dtype used in FX path
            return torch.float32

    def is_trt_dtype(self) -> bool:
        return bool(self.dtype != _enums.dtype.long)

    @staticmethod
    def _parse_format(format: Any) -> _enums.TensorFormat:
        if isinstance(format, torch.memory_format):
            if format == torch.contiguous_format:
                return _enums.TensorFormat.contiguous
            elif format == torch.channels_last:
                return _enums.TensorFormat.channels_last
            else:
                raise ValueError(
                    "Provided an unsupported tensor format (support: NCHW/contiguous_format, NHWC/channel_last)"
                )

        elif isinstance(format, _enums.TensorFormat):
            return format

        elif isinstance(format, memory_format):
            return format.to(_enums.TensorFormat)

        else:
            raise TypeError(
                "Tensor format needs to be specified with either torch.memory_format or torch_tensorrt.TensorFormat"
            )

    def _to_internal(self) -> _C.Input:
        internal_in = _C.Input()
        if self.shape_mode == Input._ShapeMode.DYNAMIC:
            if isinstance(self.shape, dict):
                if not TorchScriptInput._supported_input_size_type(self.shape["min_shape"]):
                    raise TypeError(
                        "Input shape specifications for inputs are required to be a List, tuple or torch.Size, found type: "
                        + str(type(self.shape["min_shape"]))
                        + " for min_shape"
                    )
                else:
                    internal_in.min = self.shape["min_shape"]

                if not TorchScriptInput._supported_input_size_type(self.shape["opt_shape"]):
                    raise TypeError(
                        "Input shape specifications for inputs are required to be a List, tuple or torch.Size, found type: "
                        + str(type(self.shape["opt_shape"]))
                        + " for opt_shape"
                    )
                else:
                    internal_in.opt = self.shape["opt_shape"]

                if not TorchScriptInput._supported_input_size_type(self.shape["max_shape"]):
                    raise TypeError(
                        "Input shape specifications for inputs are required to be a List, tuple or torch.Size, found type: "
                        + str(type(self.shape["max_shape"]))
                        + " for max_shape"
                    )
                else:
                    internal_in.max = self.shape["max_shape"]
                internal_in.input_is_dynamic = True
        else:
            if not TorchScriptInput._supported_input_size_type(self.shape):
                raise TypeError(
                    "Input shape specifications for inputs are required to be a List, tuple or torch.Size, found type: "
                    + str(type(self.shape))
                    + " for shape"
                )
            else:
                internal_in.opt = self.shape
            internal_in.input_is_dynamic = False

        internal_in.dtype = TorchScriptInput._parse_dtype(self.dtype)
        internal_in._explicit_set_dtype = self._explicit_set_dtype
        internal_in.format = TorchScriptInput._parse_format(self.format)

        internal_in.tensor_domain = TorchScriptInput._parse_tensor_domain(self.tensor_domain)
        return internal_in
