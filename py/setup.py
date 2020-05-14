import os
import sys
import glob
import setuptools
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.develop import develop
from setuptools.command.install import install
from distutils.cmd import Command

from torch.utils import cpp_extension
from shutil import copyfile, rmtree

dir_path = os.path.dirname(os.path.realpath(__file__))

__version__ = '0.0.1'

def gen_version_file():
    if not os.path.exists(dir_path + '/trtorch/version.py'):
        os.mknod(dir_path + '/trtorch/version.py')

    with open(dir_path + '/trtorch/version.py', 'w') as f:
        print("creating version file")
        f.write("__version__ = \"" + __version__ + '\"')

def copy_libtrtorch():
    if not os.path.exists(dir_path + '/trtorch/lib'):
        os.makedirs(dir_path + '/trtorch/lib')

    print("copying library into module")
    copyfile(dir_path + "/../bazel-bin/cpp/api/lib/libtrtorch.so", dir_path + '/trtorch/lib/libtrtorch.so')

class DevelopCommand(develop):
    description = "Builds the package and symlinks it into the PYTHONPATH"

    def initialize_options(self):
        develop.initialize_options(self)

    def finalize_options(self):
        develop.finalize_options(self)

    def run(self):
        gen_version_file()
        copy_libtrtorch()
        develop.run(self)


class InstallCommand(install):
    description = "Builds the package"

    def initialize_options(self):
        install.initialize_options(self)

    def finalize_options(self):
        install.finalize_options(self)

    def run(self):
        gen_version_file()
        copy_libtrtorch()
        install.run(self)

class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    PY_CLEAN_FILES = ['./build', './dist', './trtorch/__pycache__', './trtorch/lib', './*.pyc', './*.tgz', './*.egg-info']
    description = "Command to tidy up the project root"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for path_spec in self.PY_CLEAN_FILES:
            # Make paths absolute and relative to this path
            abs_paths = glob.glob(os.path.normpath(os.path.join(dir_path, path_spec)))
            for path in [str(p) for p in abs_paths]:
                if not path.startswith(dir_path):
                    # Die if path in CLEAN_FILES is absolute + outside this directory
                    raise ValueError("%s is not a path inside %s" % (path, dir_path))
                print('Removing %s' % os.path.relpath(path))
                rmtree(path)

ext_modules = [
    cpp_extension.CUDAExtension('trtorch._C',
                                ['trtorch/csrc/trtorch_py.cpp'],
                                library_dirs=[
                                    dir_path + '/trtorch/lib/libtrtorch.so',
                                    dir_path + '/trtorch/lib/'
                                ],
                                libraries=[
                                    "trtorch"
                                ],
                                include_dirs=[
                                    dir_path + "/../",
                                ],
                                extra_compile_args=[
                                    "-D_GLIBCXX_USE_CXX11_ABI=0"
                                ],
                                extra_link_args=[
                                    "-D_GLIBCXX_USE_CXX11_ABI=0"
                                    "-Wl,--no-as-needed",
                                    "-ltrtorch"
                                ],
                                undef_macros=[ "NDEBUG" ]
                            )
]

setup(
    name='trtorch',
    version=__version__,
    author='NVIDIA Corporation.',
    author_email='narens@nvidia.com',
    url='https://github.com/nvidia/trtorch',
    description='A compiler backend for PyTorch JIT targeting NVIDIA GPUs',
    long_description='',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.4'],
    setup_requires=['pybind11>=2.4'],
    cmdclass={
        'install': InstallCommand,
        'clean': CleanCommand,
        'develop': DevelopCommand,
        'build_ext': cpp_extension.BuildExtension
    },
    zip_safe=False,
    license="BSD-3",
    packages=find_packages(),
    classifiers=["Intended Audience :: Developers",
                 "Intended Audience :: Science/Research",
                 "Operating System :: POSIX :: Linux",
                 "Programming Language :: C++",
                 "Programming Language :: Python",
                 "Programming Language :: Python :: Implementation :: CPython",
                 "Topic :: Scientific/Engineering",
                 "Topic :: Scientific/Engineering :: Artifical Intelligence",
                 "Topic :: Software Development",
                 "Topic :: Software Developement :: Libraries"],

)
