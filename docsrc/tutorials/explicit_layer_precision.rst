.. _explicit_layer_precision:

Explicit Layer precision
====================================

There's a trade off when you run deep learning models in lower precision. Often, we can see improvements in latency
of the model, but accuracy can be compromised. In such cases, setting only some layers of the model to a specific precision
can help retain the model accuracy. TensorRT provides `setPrecision` API for setting our desired layer precision for any layer in the network.
TRTorch provides a new and convenient way to set precisions explicitly for layers directly in the Torchscript graph.
This precision configuration will be used when the TensorRT graph (`INetworkDefinition`) is constructed. For example let's consider a simple network with four layers.

```python
import torch
import trtorch

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 3)
        self.conv2 = nn.Conv2d(3, 3, 3)
        self.pool = nn.MaxPool2d(3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = torch.ops.trtorch.start_setting_precision(x, torch.float16)
        x = self.conv2(x)
        x = torch.ops.trtorch.stop_setting_precision(x)
        x = self.pool(x)
        x = self.relu(x)
        return x

scripted_model = Model().cuda().eval()
torch.jit.save(scripted_model, 'model.ts')
```

In this code, we can see that `self.conv2` is wrapped around `torch.ops.trtorch.start_setting_precision` and `torch.ops.trtorch.stop_setting_precision`. This enforces `conv2` layer to run in `FP16` precision, if there is a corresponding implementation in TensorRT. The precision of layers wrapped in between these nodes is explicitly set to the one that we provide to `torch.ops.trtorch.start_setting_precision`. We can also set a layer precision to INT8 by using `torch.ops.trtorch.start_setting_precision(x, torch.int8, input_min, input_max, output_min, output_max)`. The `input_min, input_max` correspond to the dynamic range of the input tensor for conv2 and `output_min, output_max` correspond to the dynamic range of the output of conv2. It is necessary to provide dynamic range for a layer set to INT8 precision when there is no calibrator provided. This can be beneficial if user wants to just observe performance (without the need to implement calibrator) or if they obtained dynamic ranges through other model quantization techniques.

* The function signature for `torch.ops.trtorch.start_setting_precision` is as follows

```
torch::Tensor start_setting_precision(torch::Tensor input, c10::ScalarType precision){...}
```

This doesn't modify the input tensor and behaves like an identity operation. The functionality in TRTorch converter is to set the next layer consuming the input to the `precision`. All the following layers would also have the `precision` explicitly set.

* The function signature for `torch.ops.trtorch.start_setting_precision_with_dr` is as follows

```
torch::Tensor start_setting_precision_with_dr(torch::Tensor input, c10::ScalarType precision, double input_min,
                                      double input_max, double output_min, double output_max){...}
```

This must be used when a user would like to set INT8 precision for the successive layers. (`input_min`, `input_max`) and (`output_min`, `output_max`) are the dynamic ranges for input and output tensors of the layer (for which INT8 precision is being set). Providing dynamic ranges is necessary for this operation.

* The function signature for `torch.ops.trtorch.stop_setting_precision` is as follows

```
torch::Tensor stop_setting_precision(torch::Tensor)
```

This op just signals the end of explicit layer precision by the user. This behaves as an identity operation as well. In the above `Model` class, this indicates that only `conv2` is explicitly set to `FP16` while others can use `FP32` or `FP16` based on whichever implementation is the fastest.

`torch.ops.trtorch.start_setting_precision`, `torch.ops.trtorch.start_setting_precision_with_dr`  and `torch.ops.trtorch.stop_setting_precision` are custom ops in Torchscript registered by TRTorch library. So you need to `import trtorch` to ensure Pytorch JIT understands it.

Here is the representation of these ops in a sample torchscript graph generated from scripting.

```
INFO: [TRTorch] - graph(%x.1 : Tensor,
      %40 : Float(3, strides=[1], requires_grad=1, device=cuda:0),
      %41 : Float(3, 3, 3, 3, strides=[27, 9, 3, 1], requires_grad=1, device=cuda:0),
      %42 : Float(3, strides=[1], requires_grad=1, device=cuda:0),
      %43 : Float(3, 3, 3, 3, strides=[27, 9, 3, 1], requires_grad=1, device=cuda:0)):
  %2 : int = prim::Constant[value=1]()
  %3 : float = prim::Constant[value=-6.]()
  %4 : float = prim::Constant[value=6.]()
  %16 : int = prim::Constant[value=0]()
  %32 : int = prim::Constant[value=3]()
  %33 : bool = prim::Constant[value=0]()
  %171 : int[] = prim::ListConstruct(%2, %2)
  %173 : int[] = prim::ListConstruct(%16, %16)
  %175 : int[] = prim::ListConstruct(%2, %2)
  %176 : bool = prim::Constant[value=0]()
  %177 : int[] = prim::Constant[value=[0, 0]]()
  %178 : Tensor = aten::_convolution(%x.1, %43, %42, %171, %173, %175, %176, %177, %2, %176, %176, %176, %176)
  %x1.1 : Tensor = trtorch::start_setting_precision_with_dr(%178, %2, %3, %4, %3, %4)
  %165 : int[] = prim::ListConstruct(%2, %2)
  %167 : int[] = prim::ListConstruct(%16, %16)
  %169 : int[] = prim::ListConstruct(%2, %2)
  %179 : bool = prim::Constant[value=0]()
  %180 : int[] = prim::Constant[value=[0, 0]]()
  %181 : Tensor = aten::_convolution(%x1.1, %41, %40, %165, %167, %169, %179, %180, %2, %179, %179, %179, %179)
  %x3.1 : Tensor = trtorch::stop_setting_precision(%181)
  %155 : int[] = prim::ListConstruct(%32, %32)
  %157 : int[] = prim::ListConstruct(%2, %2)
  %159 : int[] = prim::ListConstruct(%2, %2)
  %161 : int[] = prim::ListConstruct(%2, %2)
  %x4.1 : Tensor = aten::max_pool2d(%x3.1, %155, %157, %159, %161, %33)
  %result.3 : Tensor = aten::relu(%x4.1)
  return (%result.3)
```
