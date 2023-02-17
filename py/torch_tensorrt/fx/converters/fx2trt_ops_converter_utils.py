import tensorrt as trt
import torch


def get_arg(args, kwargs, name, pos, default):
    if name in kwargs:
        return kwargs[name]
    elif len(args) > pos:
        return args[pos]
    else:
        return default


def add_missing_trt_tensors(network, tensors):
    """Creates missing TensorRT tensors as constants and attaches them to the Torch Tensors"""
    with use_shape_wrapping(False):
        trt_tensors = [None] * len(tensors)

        dtype = check_torch_dtype(*tensors)

        for i, t in enumerate(tensors):
            trt_tensor = None

            # GET TRT TENSOR (OR CREATE TRT CONSTANT)

            # get tensor w/ _trt
            # or... add constant for scalar primitive
            if isinstance(t, float) or isinstance(t, int):
                shape = (1,)
                scalar = t * torch.ones(shape, dtype=dtype).cpu().numpy()
                trt_tensor = network.add_constant(shape, scalar).get_output(0)
            elif hasattr(t, "_trt"):
                trt_tensor = t._trt

            # or... add constant for leaf tensor w/o _trt
            else:

                # remove all preceding ones, these can be re-inserted later when broadcasting
                num_preceding_ones = 0
                for j in range(len(t.shape)):
                    if int(t.shape[j]) == 1:
                        num_preceding_ones += 1
                    else:
                        break
                shape = tuple(t.shape[num_preceding_ones:])

                weight = t.detach().cpu().numpy()
                t._trt = network.add_constant(shape, weight).get_output(0)
                trt_tensor = t._trt

            assert trt_tensor is not None

            trt_tensors[i] = trt_tensor

        return trt_tensors


def broadcast_trt_tensors(network, trt_tensors, broadcast_ndim):
    """Broadcast TensorRT tensors to the specified dimension by pre-padding shape 1 dims"""
    with use_shape_wrapping(False):
        broadcasted_trt_tensors = [None] * len(trt_tensors)

        for i, t in enumerate(trt_tensors):

            if len(t.shape) < broadcast_ndim:
                # append 1 size dims to front
                diff = broadcast_ndim - len(t.shape)
                shape = tuple([1] * diff + list(t.shape))
                layer = network.add_shuffle(t)
                layer.reshape_dims = shape
                trt_tensor = layer.get_output(0)
            else:
                trt_tensor = t

            broadcasted_trt_tensors[i] = trt_tensor

        return broadcasted_trt_tensors


def check_torch_dtype(*tensors):
    dtype = None
    for t in tensors:
        if isinstance(t, torch.Tensor):
            if dtype is None:
                dtype = t.dtype
            else:
                assert dtype == t.dtype  # , 'Tensor data types must match')
    assert (
        dtype is not None
    )  # , 'Data type could not be inferred from any item in list')
    return dtype


class use_shape_wrapping:

    stack = [True]  # default true

    def __init__(self, value: bool):
        self._value = value

    def __enter__(self, *args, **kwargs):
        self.stack.insert(0, self._value)

    def __exit__(self, *args, **kwargs):
        self.stack.pop(0)


def convert_AdaptiveAvgPool2d(method_args):
    network = method_args[0]
    module = method_args[1]
    input = method_args[2]
    input_trt = add_missing_trt_tensors(method_args[0], [input])[0]
    output_size = module.output_size
    if not isinstance(output_size, tuple):
        output_size = (output_size,) * 2

    stride = (
        input_trt.shape[-2] // output_size[-2],
        input_trt.shape[-1] // output_size[-1],
    )

    kernel_size = stride
    layer = network.add_pooling(
        input=input_trt, type=trt.PoolingType.AVERAGE, window_size=kernel_size
    )
    layer.stride = stride

    output = layer.get_output(0)
    return output


def convert_AdaptiveAvgPool3d(method_args):
    network = method_args[0]
    module = method_args[1]
    input = method_args[2]

    input_trt = add_missing_trt_tensors(network, [input])[0]

    output_size = module.output_size
    if not isinstance(output_size, tuple):
        output_size = (output_size,) * 3

    stride = (
        input_trt.shape[-3] // output_size[-3],
        input_trt.shape[-2] // output_size[-2],
        input_trt.shape[-1] // output_size[-1],
    )

    kernel_size = stride
    layer = network.add_pooling_nd(
        input=input_trt,
        type=trt.PoolingType.AVERAGE,
        window_size=kernel_size,
    )
    layer.stride_nd = stride

    output = layer.get_output(0)
    return output
