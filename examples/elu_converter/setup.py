import os
from setuptools import setup, Extension
from torch.utils import cpp_extension

dir_path = os.path.dirname(os.path.realpath(__file__))

ext_modules = [
    cpp_extension.CUDAExtension('elu_converter', ['elu_converter.cpp'],
                                library_dirs=[(dir_path + "/../../bazel-bin/cpp/api/lib/")],
                                libraries=["trtorch"],
                                include_dirs=[dir_path + "/../../"])
]

setup(
    name='elu_converter',
    ext_modules=ext_modules,
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
