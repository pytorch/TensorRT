import os
from setuptools import setup, Extension
from torch.utils import cpp_extension


# library_dirs should point to the libtrtorch.so, include_dirs should point to the dir that include the headers
# 1) download the latest package from https://github.com/pytorch/TensorRT/releases/
# 2) Extract the file from downloaded package, we will get the "trtorch" directory
# 3) Set trtorch_path to that directory
torchtrt_path = <PATH TO TORCHTRT>

ext_modules = [
    cpp_extension.CUDAExtension('elu_converter', ['./csrc/elu_converter.cpp'],
                                library_dirs=[(torchtrt_path + "/lib/")],
                                libraries=["torchtrt"],
                                include_dirs=[torchtrt_path + "/include/torch_tensorrt/"])
]

setup(
    name='elu_converter',
    ext_modules=ext_modules,
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
