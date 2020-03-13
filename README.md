# TRTorch 

> Ahead of Time (AOT) compiling for PyTorch JIT

## Compiling TRTorch 

### Dependencies

- Libtorch 1.4.0
- CUDA 10.1
- cuDNN 7.6
- TensorRT 6.0.1.5

Install TensorRT, CUDA and cuDNN on the system before starting to compile.


``` shell
bazel build //:libtrtorch --cxxopt="-DNDEBUG"
```

### Debug build 
``` shell
bazel build //:libtrtorch --compilation_mode=dbg
```

A tarball with the include files and library can then be found in bazel-bin 

### Running TRTorch on a JIT Graph 

> Make sure to add LibTorch's version of CUDA 10.1 to your LD_LIBRARY_PATH `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/bazel-TRTorch/external/libtorch/lib`


``` shell
bazel run //cpp/trtorchexec -- $(realpath <PATH TO GRAPH>) <input-size>
```

## How do I add support for a new op...

### In TRTorch?

Thanks for wanting to contribute! There are two main ways to handle supporting a new op. Either you can write a converter for the op from scratch and register it in the NodeConverterRegistry or if you can map the op to a set of ops that already have converters you can write a graph rewrite pass which will replace your new op with an equivalent subgraph of supported ops. Its preferred to use graph rewriting because then we do not need to maintain a large library of op converters. 

### In my application?

> The Node Converter Registry is not exposed currently in the public API but you can try using internal headers. 

You can register a converter for your op using the NodeConverterRegistry inside your application. 

## Structure of the repo

| Component     | Description                                                  |
| ------------- | ------------------------------------------------------------ |
| [**core**]()  | Main JIT ingest, lowering, conversion and execution implementations |
| [**cpp**]()   | C++ API for TRTorch                                          |
| [**tests**]() | Unit test for TRTorch                                        |

## License

The TRTorch license can be found in the LICENSE file.
