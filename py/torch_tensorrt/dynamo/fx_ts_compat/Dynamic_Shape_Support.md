# PyTorch Operations Dynamic Shape Support Summary



 | Operation | Test Method | Supports Dynamic Shape | Shape | Num of dimensions | Reason |
| --- | --- | --- | --- | --- | --- |
| adaptive_avgpool |     | partially | (-1, -1, 256, 256) | 2   | AdaptiveAvgPool2d and AdaptiveAvgPool3d currently doesn't support dynamic shapes for last two dims. |
| any |     | no  |     |     | torch.zeros(tuple(\[*input_t.shape\])). Trying to create tensor with negative dimension -1: \[-1, -1, -1, -1\] |
| as_strided |     | no  |     |     | RuntimeError: setStorage: sizes \[2, 3\], strides \[1, 2\], storage offset 0, and itemsize 8 requiring a storage size of 48 are out of bounds for storage of size 16 |
| avg_pool | avg_pool2d | yes | (-1,-,1,-1,-1) | 4   |     |
|     | avg_pool1d | partially | (-1, 3, 3) | 1   |     |
| batchnorm |     | partially | (-1, 3, -1, -1) | 3   | "Channel dim can't be dynamic for batch norm." |
| binary_ops |     | yes | (-1,-,1,-1,-1) | 4   |     |
| cat |     | yes | (-1,-,1,-1,-1) | 4   |     |
| chunk |     | partially | (-1, 1, 3, -1) | any (not chunk dim) | AssertionError: Can't chunk on dynamic shape dimension! |
| clamp |     | yes | (-1,-,1,-1,-1) |     |     |
| convolution | conv2d | partially | (-1, 3, -1, -1) | 3   | AssertionError: Channel dim can't be dynamic for convolution. |
|     | conv1d | partially | (-1, 3, 3) | 1   |     |
|     | conv3d | partially | (-1,-,1,-1,-1) | 4   | AssertionError: Channel dim can't be dynamic for convolution. |
| dequantize |     | yes | (-1,-,1,-1,-1) | 4   |     |
| eimsum |     | yes | (-1,-,1,-1,-1) | 4   |     |
| elu |     | yes | (-1,-,1,-1,-1) | 4   |     |
| embedding |     | yes | (-1,-,1,-1,-1) | 4   |     |
| eq  | SimpleConverter | yes | (-1,-,1,-1,-1) | 4   |     |
|     | ConstInputConverter | yes | (-1,-,1,-1,-1) | 4   |     |
|     | EqMethodConverter | no  | limitation in converter |     | RuntimeError: Trying to create tensor with negative dimension -1: \[-1, -1, -1, -1\] |
|     | EqOperatorConverter | no  | limitation in converter |     | RuntimeError: Trying to create tensor with negative dimension -1: \[-1, -1, -1, -1\] |
|     | EqOperatorConstant | partially | (3,-1) | 1   |     |
|     | EqConverter | no  | limitation in converter |     | RuntimeError: Trying to create tensor with negative dimension -1: \[-1, -1, -1, -1\] |
| expand |     | no  |     |     | Dynamic shape is not suitable for the expand operation. |
| flatten |     | yes | (-1, -1, -1, -1, -1) | 5   |     |
| gelu |     | yes | (-1,-,1,-1,-1) | 4   |     |
| getitem |     | yes | (-1,-,1,-1,-1) | 4   |     |
| gt  | EqOperatorSimpleConverter | yes | (-1,-,1,-1,-1) | 4   |     |
|     | ConstInputConverter | yes | (-1,-,1,-1,-1) | 4   |     |
|     | GtConverter | no  | limitation in converter |     | RuntimeError: Trying to create tensor with negative dimension -1: \[-1, -1, -1, -1\] |
|     | GtMethodConverter | no  | limitation in converter |     | RuntimeError: Trying to create tensor with negative dimension -1: \[-1, -1, -1, -1\] |
|     | GtOperator | no  | limitation in converter |     | RuntimeError: Trying to create tensor with negative dimension -1: \[-1, -1, -1, -1\] |
|     | EqOperator | no  | limitation in converter |     | RuntimeError: Trying to create tensor with negative dimension -1: \[-1, -1, -1, -1\] |
| hardsigmoid |     | yes | (-1,-,1,-1,-1) | 4   |     |
| hardtanh |     | yes | (-1,-,1,-1,-1) | 4   |     |
| interpolate |     | yes | (-1,-,1,-1,-1) | 4   |     |
| isinf |     | yes | (-1,-,1,-1,-1) | 4   |     |
| leaky_relu |     | yes | (-1,-,1,-1,-1) | 4   |     |
| linear |     | partially | (-1, 3, 5) | 1   | AssertionError: Currently we only support one dynmaic dim for linear and it can't be the last dim. |
| logical_and |     | yes | (-1, -1, -1, -1) | 4   |     |
| logical_or |     | yes | (-1, -1, -1, -1) | 4   |     |
| logical_xor |     | yes | (-1, -1, -1, -1) | 4   |     |
| lt  |     | yes | (-1, -1, -1, -1) | 4   |     |
| masked_fill |     | no  | limitation in converter |     | RuntimeError: Trying to create tensor with negative dimension -1: \[-1, -1, -1, -1\] |
| mat_mul |     | yes | batch dim |     |     |
| max | MaxFullReduce | yes | (-1, -1, -1, -1) | 4   |     |
|     | MaxDimReduce | yes | (-1, -1, -1, -1) | 4   |     |
|     | MaxMethod | yes | (-1, -1, -1, -1) | 4   |     |
| maximum |     | yes | (-1, -1, -1, -1) | 4   |     |
| maxpool | max_pool1d | partially | (1, 1, -1) | 1   | shape is not set to (-1, -1, -1) as reshape dimension with, more than one -1 wildcard is not allowed while adding unsqueeze layer |
|     | max_pool2d | yes | (-1, -1, -1, -1) | 4   |     |
|     | max_pool3d | yes | (-1, -1, -1, -1, -1) | 5   |     |
| min | MinFullReduce | yes | (-1, -1, -1, -1) | 4   |     |
|     | MinDimReduce | yes | (-1, -1, -1, -1) | 4   |     |
|     | MinMethod | yes | (-1, -1, -1, -1) | 4   |     |
| minimum |     | yes | (-1, -1, -1, -1) | 4   |     |
| narrow |     | partially | (-1, 3, -1, -1) | 3   | AssertionError: Can't chunk on dynamic shape dimension! |
| ne  | NeFunctionConverter | yes | (-1, -1, -1, -1) | 4   |     |
|     | NeMethodConverter | yes | (-1, -1, -1, -1) | 4   |     |
|     | NeOperatorConverter | yes | (-1, -1, -1, -1) | 4   |     |
|     | ConstInputConverter | yes | (-1, -1, -1, -1) | 4   |     |
|     | NeOperatorConstantConverter | partially | (3, -1) | 1   |     |
| new_ones |     | yes | (-1, -1, -1, -1) | 4   |     |
| numel |     | no  | limitation in converter |     | RuntimeError: numel does not support dynamic shapes. |
| pad |     | no  | limitation in converter |     | test\_pad\_with\_dynamic\_shape\_four\_dimensions\_0\_2d (deeplearning.trt.torch\_tensorrt.py.torch\_tensorrt.fx.test.converters.acc\_op.test\_pad.TestPadConverter) ... \[07/15/2022-09:23:18\] \[TRT\] \[E\] 2: \[intInterval.cpp::max::26\] Error Code 2: Internal Error (Assertion !empty() failed. |
| permute |     | yes | (-1, -1, -1, -1) | 4   |     |
| prod |     | yes | (-1, -1, -1, -1) | 4   |     |
| quantize\_per\_tensor |     | yes | (-1, -1, -1, -1) | 4   |     |
| reduce op |     | yes | (-1, -1, -1, -1) | 4   |     |
| relu |     | yes | (-1, -1, -1, -1) | 4   |     |
| repeat interleave |     | partially | (-1, 3, 2) | 1   | AssertionError: Currently we don't support unsqueeze with more than one dynamic dims. |
| reshape |     | yes | (-1, -1, -1, -1) | 4   |     |
| selu |     | yes | (-1, -1, -1, -1) | 4   |     |
| sigmoid |     | yes | (-1,-,1,-1,-1) | 4   |     |
| silu |     | yes | (-1,-,1,-1,-1) | 4   |     |
| size |     | yes | (-1, -1, -1, -1) | 4   |     |
| softmax |     | yes | (-1, -1, -1, -1) | 4   |     |
| softsign |     | yes | (-1, -1, -1, -1) | 4   |     |
| split |     | partially | (-1, 10, -1) | 2   | AssertionError: Can't chunk on dynamic shape dimension! |
| squeeze |     | partially | (1, -1, 2) | 1   | AssertionError: Currently more than one dynamic dim for input to squeeze is not supported. |
| std |     | yes | (-1, -1, -1, -1) | 4   |     |
| tanh |     | yes | (-1, -1, -1, -1) | 4   |     |
| tile |     | yes | (-1, -1, -1, -1) | 4   |     |
| to_dtype | int | yes | (-1, -1, -1, -1) | 4   |     |
|     | float | yes | (-1, -1, -1, -1) | 4   |     |
| topk |     | yes | (-1, -1, -1, -1) | 4   |     |
| transpose_convolution | conv_transpose2d | partially | (-1, 3, -1, -1) | 3   |     |
|     | conv_transpose3d | partially | (-1, 3, -1, -1, -1) | 4   |     |
| type_as |     | yes | (-1, -1, -1, -1) | 4   | RuntimeError: ShapeProp error for: node=%type\_1 : \[#users=1\] = call\_method\[target=type\](args = (%input_1,), kwargs = {dtype: torch.float32}) with meta={} |
| unary ops |     | yes | (-1, -1, -1, -1) | 4   |     |
| unsqueeze |     | partially | (-1, 2, 3) | 1   | AssertionError: Currently we don't support unsqueeze with more than one dynamic dims. |
| where |     | no  | limitation in converter |     | torch.broadcast_shape can not handle -1 dimension in shape \[-1, 2, 2\] |



Binary Ops Include following operations:
|Binary Ops       |
|----------|
|add       |
|sub       |
|div       |
|mul       |
|floor_div |
|fmod      |
|floor_divide|
|pow       |


Unary Ops Include following operations:
|Unary Ops     |
|----------|
|rsqrt     |
|sin       |
|cos       |
|tan       |
|sinh      |
|cosh      |
|asin      |
|acos      |
|atan      |
|abs       |
|neg       |
|reciprocal|
|sqrt      |
|log       |
|exp       |
|floor     |
|ceil      |
|sign      |

Note: For more information about the test method, please refer to the operation test files. Additionally, test files include information about errors encountered during dynamic shape testing.
