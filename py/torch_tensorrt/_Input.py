from enum import Enum
from typing import List, Dict, Any, Tuple

import torch

from torch_tensorrt import _enums
from torch_tensorrt import _C


class Input(object):
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

    class _ShapeMode(Enum):
        STATIC = 0
        DYNAMIC = 1

    shape_mode = None  #: (torch_tensorrt.Input._ShapeMode): Is input statically or dynamically shaped
    shape = None  #: (Tuple or Dict): Either a single Tuple or a dict of tuples defining the input shape. Static shaped inputs will have a single tuple. Dynamic inputs will have a dict of the form ``{ "min_shape": Tuple, "opt_shape": Tuple, "max_shape": Tuple }``
    dtype = (
        _enums.dtype.unknown
    )  #: The expected data type of the input tensor (default: torch_tensorrt.dtype.float32)
    _explicit_set_dtype = False
    format = (
        _enums.TensorFormat.contiguous
    )  #: The expected format of the input tensor (default: torch_tensorrt.TensorFormat.NCHW)

    DOMAIN_OFFSET = 2
    low_tensor_domain_incl = 0
    high_tensor_domain_excl = low_tensor_domain_incl + DOMAIN_OFFSET

    def __init__(self, *args, **kwargs):
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
            tensor_domain (Tuple(int, int), optional): The domain of allowed integer values for the tensor, as interval notation: [tensor_domain[0], tensor_domain[1]).
                Note: Entering one NoneType will set the other bound to the provided value +/- 2; entering two NoneTypes (or not specifying) will set the bound to [0, 2)

        Examples:
            - Input([1,3,32,32], dtype=torch.float32, format=torch.channel_last)
            - Input(shape=(1,3,32,32), dtype=torch_tensorrt.dtype.int32, format=torch_tensorrt.TensorFormat.NCHW)
            - Input(min_shape=(1,3,32,32), opt_shape=[2,3,32,32], max_shape=(3,3,32,32)) #Implicitly dtype=torch_tensorrt.dtype.float32, format=torch_tensorrt.TensorFormat.NCHW
        """
        if len(args) == 1:
            if not Input._supported_input_size_type(args[0]):
                raise TypeError(
                    "Input shape specifications for inputs are required to be a List, tuple or torch.Size, found type: "
                    + str(type(args[0]))
                )
            if any(k in kwargs for k in ["min_shape", "opt_shape", "max_shape"]):
                raise ValueError(
                    "Found that both shape (as a positional argument), and one or more of min_shape, opt_shape, max_shape were specified\nclass Input expects that only either shape or all three of min_shape, opt_shape, max_shape are defined"
                )
            self.shape = tuple(args[0])
            self.shape_mode = Input._ShapeMode.STATIC

        elif len(args) == 0:
            if not ("shape" in kwargs) and not (
                all(k in kwargs for k in ["min_shape", "opt_shape", "max_shape"])
            ):
                raise ValueError(
                    "Missing required arguments for class Input\nEither shape or all three of min_shape, opt_shape, max_shape must be defined"
                )
            elif ("shape" in kwargs) and all(
                k in kwargs for k in ["min_shape", "opt_shape", "max_shape"]
            ):
                raise ValueError(
                    "Found that both shape, and one or more of min_shape, opt_shape, max_shape were specified\nclass Input expects that only either shape or all three of min_shape, opt_shape, max_shape are defined"
                )

            if "shape" in kwargs:
                if not Input._supported_input_size_type(kwargs["shape"]):
                    raise TypeError(
                        "Input shape specifications for inputs are required to be a List, tuple or torch.Size, found type: "
                        + str(type(kwargs["shape"]))
                    )
                self.shape = tuple(kwargs["shape"])
                self.shape_mode = Input._ShapeMode.STATIC
            else:
                if not Input._supported_input_size_type(kwargs["min_shape"]):
                    raise TypeError(
                        "Input shape specifications for inputs are required to be a List, tuple or torch.Size, found type: "
                        + str(type(kwargs["min_shape"]))
                        + " for min_shape"
                    )
                if not Input._supported_input_size_type(kwargs["opt_shape"]):
                    raise TypeError(
                        "Input shape specifications for inputs are required to be a List, tuple or torch.Size, found type: "
                        + str(type(kwargs["opt_shape"]))
                        + " for opt_shape"
                    )
                if not Input._supported_input_size_type(kwargs["max_shape"]):
                    raise TypeError(
                        "Input shape specifications for inputs are required to be a List, tuple or torch.Size, found type: "
                        + str(type(kwargs["max_shape"]))
                        + " for max_shape"
                    )

                self.shape = {
                    "min_shape": tuple(kwargs["min_shape"]),
                    "opt_shape": tuple(kwargs["opt_shape"]),
                    "max_shape": tuple(kwargs["max_shape"]),
                }
                self.shape_mode = Input._ShapeMode.DYNAMIC

        else:
            raise ValueError(
                "Unexpected number of positional arguments for class Input \n    Found {} arguments, expected either zero or a single positional arguments".format(
                    len(args)
                )
            )

        if "dtype" in kwargs:
            self.dtype = Input._parse_dtype(kwargs["dtype"])
            self._explicit_set_dtype = True

        if "format" in kwargs:
            self.format = Input._parse_format(kwargs["format"])

        if "tensor_domain" in kwargs:
            domain = kwargs["tensor_domain"]
        else:
            domain = None

        self.tensor_domain = Input._parse_tensor_domain(domain)

    def __str__(self) -> str:
        if self.shape_mode == Input._ShapeMode.STATIC:
            return "Input(shape={}, dtype={}, format={}, domain=[{}, {}))".format(
                self.shape,
                str(self.dtype),
                str(self.format),
                str(self.tensor_domain[0]),
                str(self.tensor_domain[1]),
            )
        elif self.shape_mode == Input._ShapeMode.DYNAMIC:
            return "Input(min_shape={}, opt_shape={}, max_shape={}, dtype={}, format={}, domain=[{}, {}))".format(
                self.shape["min_shape"],
                self.shape["opt_shape"],
                self.shape["max_shape"],
                str(self.dtype),
                str(self.format),
                str(self.tensor_domain[0]),
                str(self.tensor_domain[1]),
            )
        else:
            raise RuntimeError("Unknown input shape mode")

    def _to_internal(self) -> _C.Input:
        internal_in = _C.Input()
        if self.shape_mode == Input._ShapeMode.DYNAMIC:
            if not Input._supported_input_size_type(self.shape["min_shape"]):
                raise TypeError(
                    "Input shape specifications for inputs are required to be a List, tuple or torch.Size, found type: "
                    + str(type(self.shape["min_shape"]))
                    + " for min_shape"
                )
            else:
                internal_in.min = self.shape["min_shape"]

            if not Input._supported_input_size_type(self.shape["opt_shape"]):
                raise TypeError(
                    "Input shape specifications for inputs are required to be a List, tuple or torch.Size, found type: "
                    + str(type(self.shape["opt_shape"]))
                    + " for opt_shape"
                )
            else:
                internal_in.opt = self.shape["opt_shape"]

            if not Input._supported_input_size_type(self.shape["max_shape"]):
                raise TypeError(
                    "Input shape specifications for inputs are required to be a List, tuple or torch.Size, found type: "
                    + str(type(self.shape["max_shape"]))
                    + " for max_shape"
                )
            else:
                internal_in.max = self.shape["max_shape"]
            internal_in.input_is_dynamic = True
        else:
            if not Input._supported_input_size_type(self.shape):
                raise TypeError(
                    "Input shape specifications for inputs are required to be a List, tuple or torch.Size, found type: "
                    + str(type(self.shape))
                    + " for shape"
                )
            else:
                internal_in.opt = self.shape
            internal_in.input_is_dynamic = False

        if self.dtype != _enums.dtype.unknown:
            self._explicit_set_dtype = True
        else:
            self._explicit_set_dtype = False

        internal_in.dtype = Input._parse_dtype(self.dtype)
        internal_in._explicit_set_dtype = self._explicit_set_dtype
        internal_in.format = Input._parse_format(self.format)

        internal_in.tensor_domain = Input._parse_tensor_domain(self.tensor_domain)
        return internal_in

    @staticmethod
    def _supported_input_size_type(input_size: Any) -> bool:
        if isinstance(input_size, torch.Size):
            return True
        elif isinstance(input_size, tuple):
            return True
        elif isinstance(input_size, list):
            return True
        else:
            return False

    @staticmethod
    def _parse_dtype(dtype: Any) -> _enums.dtype:
        if isinstance(dtype, torch.dtype):
            if dtype == torch.long:
                return _enums.dtype.long
            elif dtype == torch.int32:
                return _enums.dtype.int32
            elif dtype == torch.half:
                return _enums.dtype.half
            elif dtype == torch.float:
                return _enums.dtype.float
            elif dtype == torch.bool:
                return _enums.dtype.bool
            else:
                raise TypeError(
                    "Provided an unsupported data type as an input data type (support: bool, int32, long, half, float), got: "
                    + str(dtype)
                )

        elif isinstance(dtype, _enums.dtype):
            return dtype

        else:
            raise TypeError(
                "Input data type needs to be specified with a torch.dtype or a torch_tensorrt.dtype, got: "
                + str(type(dtype))
            )

    def is_trt_dtype(self) -> bool:
        return self.dtype != _enums.dtype.long

    @staticmethod
    def _parse_format(format: Any) -> _enums.TensorFormat:
        if isinstance(format, torch.memory_format):
            if format == torch.contiguous_format:
                return _enums.TensorFormat.contiguous
            elif format == torch.channels_last:
                return _enums.TensorFormat.channels_last
            else:
                raise ValueError(
                    "Provided an unsupported tensor format (support: NHCW/contiguous_format, NHWC/channel_last)"
                )

        elif isinstance(format, _enums.TensorFormat):
            return format

        else:
            raise TypeError(
                "Tensor format needs to be specified with either torch.memory_format or torch_tensorrt.TensorFormat"
            )

    @staticmethod
    def _parse_tensor_domain(domain: Tuple[int, int]) -> Tuple:
        """
        Produce a tuple of integers which specifies a tensor domain in the interval format: [lo, hi)

        Args:
            domain (Tuple[int, int]): A tuple of integers (or NoneTypes) to verify

        Returns:
            A tuple of two int32_t-valid integers
        """
        if domain is None:
            domain_lo = None
            domain_hi = None
        elif len(domain) == 2:
            domain_lo, domain_hi = domain
        else:
            raise ValueError(
                f"Expected 2 values for domain, got {len(domain)}: {domain}"
            )

        lo_domain_missing = domain_lo is None
        hi_domain_missing = domain_hi is None
        valid_type_lo = (domain_lo is None) or isinstance(domain_lo, int)
        valid_type_hi = (domain_hi is None) or isinstance(domain_hi, int)

        if not valid_type_lo:
            raise ValueError(
                f"Expected integer value for tensor domain low specifier, got {domain_lo}"
            )
        elif not valid_type_hi:
            raise ValueError(
                f"Expected integer value for tensor domain high specifier, got {domain_hi}"
            )

        if lo_domain_missing and hi_domain_missing:
            result_domain = (
                Input.low_tensor_domain_incl,
                Input.high_tensor_domain_excl,
            )
        elif lo_domain_missing:
            result_domain = (domain_hi - Input.DOMAIN_OFFSET, domain_hi)
        elif hi_domain_missing:
            result_domain = (domain_lo, domain_lo + Input.DOMAIN_OFFSET)
        else:
            if domain_hi <= domain_lo:
                raise ValueError(
                    "Expected provided integer range to have low tensor domain value "
                    + f"< high tensor domain value, got invalid range [{domain_lo}, {domain_hi})"
                )
            result_domain = (domain_lo, domain_hi)

        def val_exceeds_int32_t_repr(value: int):
            """Returns true if the input integer value exceeds C++ int32_t repr, else false"""
            if not isinstance(value, int):
                raise ValueError(f"Expected int input, got {type(value)}")
            return (value == 0x80000000) or (abs(value) > 0x80000000)

        if isinstance(result_domain[0], int) and val_exceeds_int32_t_repr(
            result_domain[0]
        ):
            raise ValueError(
                f"Determined low-bound on tensor domain does not fit in a 32-bit signed int"
            )
        elif isinstance(result_domain[1], int) and val_exceeds_int32_t_repr(
            result_domain[1]
        ):
            raise ValueError(
                f"Determined high-bound on tensor domain does not fit in a 32-bit signed int"
            )

        return result_domain

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> "Input":
        """
        Produce a Input which contains the information of the given PyTorch tensor.

        Args:
            tensor (torch.Tensor): A PyTorch tensor.

        Returns:
            A Input object.
        """
        if not any(
            [
                t.is_contiguous(memory_format=torch.contiguous_format),
                t.is_contiguous(memory_format=torch.channels_last),
            ]
        ):
            raise ValueError(
                "Tensor does not have a supported memory format, supported formats are contiguous or channel_last"
            )
        frmt = (
            torch.contiguous_format
            if t.is_contiguous(memory_format=torch.contiguous_format)
            else torch.channels_last
        )
        return cls(shape=t.shape, dtype=t.dtype, format=frmt)

    @classmethod
    def from_tensors(cls, ts: torch.Tensor) -> List["Input"]:
        """
        Produce a list of Inputs which contain
        the information of all the given PyTorch tensors.

        Args:
            tensors (Iterable[torch.Tensor]): A list of PyTorch tensors.

        Returns:
            A list of Inputs.
        """

        assert isinstance(ts, (list, tuple))
        return [cls.from_tensor(t) for t in ts]

    def example_tensor(self, optimization_profile_field: str = None) -> torch.Tensor:
        """
        Get an example tensor of the shape specified by the Input object

        Args:
            optimization_profile_field (Optional(str)): Name of the field to use for shape in the case the Input is dynamically shaped

        Returns:
            A PyTorch Tensor
        """
        if optimization_profile_field is not None:
            try:
                assert any(
                    [
                        optimization_profile_field == field_name
                        for field_name in ["min_shape", "opt_shape", "max_shape"]
                    ]
                )
            except:
                raise ValueError(
                    "Invalid field name, expected one of min_shape, opt_shape, max_shape"
                )

        if (
            optimization_profile_field is not None
            and self.shape_mode == Input._ShapeMode.STATIC
        ):
            raise ValueError(
                "Specified a optimization profile field but the input is static"
            )

        if (
            optimization_profile_field is None
            and self.shape_mode == Input._ShapeMode.DYNAMIC
        ):
            raise ValueError(
                "Requested an example tensor from a dynamic shaped input but did not specific which profile field to use."
            )

        if self.shape_mode == Input._ShapeMode.STATIC:
            return torch.randn(self.shape).to(dtype=self.dtype)
        else:
            return torch.randn(self.shape[optimization_profile_field]).to(
                dtype=self.dtype
            )
