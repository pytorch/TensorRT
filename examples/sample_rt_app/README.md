# Sample application which uses TRTorch runtime library and plugin library.

This sample is a demonstration on how to use TRTorch runtime library `libtrtorchrt.so` along with plugin library `libtrtorch_plugins.so`

In this demo, we convert two models `ConvGelu` and `Norm` to TensorRT using TRTorch python API and perform inference using `samplertapp`. In these models, `Gelu` and `Norm` layer are expressed as plugins in the network.

## Generating Torch script modules with TRT Engines

The following command will generate `conv_gelu.jit` and `norm.jit` torchscript modules which contain TensorRT engines.

```
python network.py
```

## Sample runtime app

The main goal is to use TRTorch runtime library `libtrtorchrt.so`, a lightweight library sufficient enough to deploy your Torchscript programs containing TRT engines.

1) Download the latest release from TRTorch github repo and unpack the tar file.

```
cd examples/sample_rt_app
// Download latest TRTorch release tar file (libtrtorch.tar.gz) from https://github.com/NVIDIA/TRTorch/releases
tar -xvzf libtrtorch.tar.gz
```

 2) Build and run `samplertapp`

​     `samplertapp` is a binary which loads the torchscript modules `conv_gelu.jit` or `norm.jit` and runs the TRT engines on a random input using TRTorch runtime components. Checkout the `main.cpp` and `BUILD ` file for necessary code and compilation dependencies.

To build and run the app

```
cd TRTorch
bazel run //examples/sample_rt_app:samplertapp $PWD/examples/sample_rt_app/norm.jit
```
