import os
from setuptools import setup, Extension
from torch.utils import cpp_extension


# library_dirs should point to the libtrtorch.so, include_dirs should point to the dir that include the headers
# 1) download the latest package from https://github.com/NVIDIA/TRTorch/releases/
# 2) Extract the file from downloaded package, we will get the "trtorch" directory
# 3) Set trtorch_path to that directory
trtorch_path = <PATH TO TRTORCH>

ext_modules = [
    cpp_extension.CUDAExtension('elu_converter', ['./csrc/elu_converter.cpp'],
                                library_dirs=[(trtorch_path + "/lib/")],
                                libraries=["trtorch"],
                                include_dirs=[trtorch_path + "/include/trtorch/"])
]

setup(
    name='elu_converter',
    ext_modules=ext_modules,
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
