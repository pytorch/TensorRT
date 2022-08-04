# torch_tensorrtrt_example

## Sample application which uses Torch-TensorRT runtime library and plugin library.

This sample is a demonstration on how to use Torch-TensorRT runtime library `libtorchtrt_runtime.so` along with plugin library `libtorchtrt_plugins.so`

In this demo, we convert two models `ConvGelu` and `Norm` to TensorRT using Torch-TensorRT python API and perform inference using `torchtrt_runtime_example`. In these models, `Gelu` and `Norm` layer are expressed as plugins in the network.

### Generating Torch script modules with TRT Engines

The following command will generate `conv_gelu.jit` and `norm.jit` torchscript modules which contain TensorRT engines.

```sh
python network.py
```

### `torchtrt_runtime_example`
The main goal is to use Torch-TensorRT runtime library `libtorchtrt_runtime.so`, a lightweight library sufficient enough to deploy your Torchscript programs containing TRT engines.

1) Download releases of LibTorch and Torch-TensorRT from https://pytorch.org and the Torch-TensorRT github repo and unpack both in the deps directory.

```sh
cd examples/torch_tensorrtrt_example/deps
// Download latest Torch-TensorRT release tar file (libtorch_tensorrt.tar.gz) from https://github.com/pytorch/TensorRT/releases
tar -xvzf libtorch_tensorrt.tar.gz
unzip libtorch-cxx11-abi-shared-with-deps-[PYTORCH_VERSION].zip
```

> If cuDNN and TensorRT are not installed on your system / in your LD_LIBRARY_PATH then do the following as well

```sh
cd deps
mkdir cudnn && tar -xvzf <cuDNN TARBALL> --directory cudnn --strip-components=1
mkdir tensorrt && tar -xvzf <TensorRT TARBALL> --directory tensorrt --strip-components=1
cd ..
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/deps/torch_tensorrt/lib:$(pwd)/deps/libtorch/lib:$(pwd)/deps/tensorrt/lib:$(pwd)/deps/cudnn/lib64:/usr/local/cuda/lib
```

This gives maximum compatibility with system configurations for running this example but in general you are better off adding `-Wl,-rpath $(DEP_DIR)/tensorrt/lib -Wl,-rpath $(DEP_DIR)/cudnn/lib64` to your linking command for actual applications

 2) Build and run `torchtrt_runtime_example`

​     `torchtrt_runtime_example` is a binary which loads the torchscript modules `conv_gelu.jit` or `norm.jit` and runs the TRT engines on a random input using Torch-TensorRT runtime components. Checkout the `main.cpp` and `Makefile ` file for necessary code and compilation dependencies.

To build and run the app

```sh
cd examples/torchtrt_runtime_example
make
# If paths are different than the ones below, change as necessary
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/deps/torch_tensorrt/lib:$(pwd)/deps/libtorch/lib:$(pwd)/deps/tensorrt/lib:$(pwd)/deps/cudnn/lib64:/usr/local/cuda/lib
./torchtrt_runtime_example $PWD/examples/torchtrt_runtime_example/norm.jit
```
