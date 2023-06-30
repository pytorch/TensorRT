from torch_tensorrt.fx.types import (
    TRTDataType,
    TRTNetwork,
    TRTTensor,
)


def cast_trt_tensor(
    network: TRTNetwork,
    input_val: TRTTensor,
    dtype: TRTDataType,
    name: str,
) -> TRTTensor:
    """
    Given a TRT Tensor, convert that Tensor to the specified dtype
    Adds an Identity layer to the network which performs the conversion
    Args:
        network (TRTNetwork): A TensorRT network
        input_val (TRTTensor): A TRT Tensor to cast to a new data type
        dtype (TRTDataType): The TRTDataType to cast the input Tensor to
        name (str): Name of the calling layer
    Returns:
        A TensorRT ITensor which has been casted to the specified dtype
    """
    #
    if input_val.dtype != dtype:
        identity_layer = network.add_identity(input_val)
        identity_layer.set_output_type(0, dtype)
        identity_layer.name = (
            f"Cast ITensor {input_val.name} from {input_val.dtype} to {dtype} - {name}"
        )
        return identity_layer.get_output(0)
    else:
        return input_val


def broadcastable(
    a: TRTTensor,
    b: TRTTensor,
) -> bool:
    "Check if two tensors are broadcastable according to torch rules"
    a_shape = tuple(a.shape)
    b_shape = tuple(b.shape)
    # check from the trailing
    diff = len(a_shape) - len(b_shape)
    if diff == 0:
        return True
    if diff > 0:
        max = len(a_shape)
        min = len(b_shape)
        greater_tensor = a_shape
        lesser_tensor = b_shape
    elif diff < 0:
        max = len(b_shape)
        min = len(a_shape)
        greater_tensor = b_shape
        lesser_tensor = a_shape
    j = min - 1
    for i in range(max - 1, diff - 1, -1):
        if not (
            greater_tensor[i] != lesser_tensor[j]
            and (greater_tensor[i] == 1 or lesser_tensor[i] == 1)
        ):
            return False
    return True
