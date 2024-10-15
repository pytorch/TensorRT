.. _installation:

Installation
##################

Precompiled Binaries
---------------------

Torch-TensorRT 2.x is centered primarily around Python. As such, precompiled releases can be found on `pypi.org <https://pypi.org/project/torch-tensorrt/>`_

Dependencies
~~~~~~~~~~~~~~

You need to have CUDA, PyTorch, and TensorRT (python package is sufficient) installed to use Torch-TensorRT

    * https://developer.nvidia.com/cuda
    * https://pytorch.org


Installing Torch-TensorRT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can install the python package using

.. code-block:: sh

    python -m pip install torch torch-tensorrt tensorrt

Packages are uploaded for Linux on x86 and Windows

Installing Torch-TensorRT for a specific CUDA version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similar to PyTorch, Torch-TensorRT has builds compiled for different versions of CUDA. These are distributed on PyTorch's package index

For example CUDA 11.8

.. code-block:: sh

    python -m pip install torch torch-tensorrt tensorrt --extra-index-url https://download.pytorch.org/whl/cu118

Installing Nightly Builds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Torch-TensorRT distributed nightlies targeting the PyTorch nightly. These can be installed from the PyTorch nightly package index (separated by CUDA version)

.. code-block:: sh

    python -m pip install --pre torch torch-tensorrt tensorrt --extra-index-url https://download.pytorch.org/whl/nightly/cu124



.. _bin-dist:

C++ Precompiled Binaries (TorchScript Only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Precompiled tarballs for releases are provided here: https://github.com/pytorch/TensorRT/releases

.. _compile-from-source:

Compiling From Source
------------------------

Building on Linux
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _installing-deps:

Dependencies
^^^^^^^^^^^^^^

* Torch-TensorRT is built with **Bazel**, so begin by installing it.

    * The easiest way is to install bazelisk using the method of your choosing https://github.com/bazelbuild/bazelisk
    * Otherwise you can use the following instructions to install binaries https://docs.bazel.build/versions/master/install.html
    * Finally if you need to compile from source (e.g. aarch64 until bazel distributes binaries for the architecture) you can use these instructions

    .. code-block:: shell

        export BAZEL_VERSION=$(cat <PATH_TO_TORCHTRT_ROOT>/.bazelversion)
        mkdir bazel
        cd bazel
        curl -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-dist.zip
        unzip bazel-$BAZEL_VERSION-dist.zip
        bash ./compile.sh
        cp output/bazel /usr/local/bin/


* You will also need to have **CUDA** installed on the system (or if running in a container, the system must have the CUDA driver installed and the container must have CUDA)

    * Specify your CUDA version here if not the version used in the branch being built: https://github.com/pytorch/TensorRT/blob/4e5b0f6e860910eb510fa70a76ee3eb9825e7a4d/WORKSPACE#L46


* The correct **LibTorch** and **TensorRT** versions will be pulled down for you by bazel.

    NOTE: By default bazel will pull the latest nightly from pytorch.org. For building main, this is usually sufficient however if there is a specific PyTorch you are targeting,
    edit these locations with updated URLs/paths:

    * https://github.com/pytorch/TensorRT/blob/4e5b0f6e860910eb510fa70a76ee3eb9825e7a4d/WORKSPACE#L53C1-L53C1


* **TensorRT** is not required to be installed on the system to build Torch-TensorRT, in fact this is preferable to ensure reproducible builds. If versions other than the default are needed
  point the WORKSPACE file to the URL of the tarball or download the tarball for TensorRT from https://developer.nvidia.com and update the paths in the WORKSPACE file here https://github.com/pytorch/TensorRT/blob/4e5b0f6e860910eb510fa70a76ee3eb9825e7a4d/WORKSPACE#L71

    For example:

    .. code-block:: python

        http_archive(
            name = "tensorrt",
            build_file = "@//third_party/tensorrt/archive:BUILD",
            sha256 = "<TENSORRT SHA256>", # Optional but recommended
            strip_prefix = "TensorRT-<TENSORRT VERSION>",
            urls = [
                "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/<TENSORRT DOWNLOAD PATH>",
                # OR
                "file:///<ABSOLUTE PATH TO FILE>/TensorRT-<TENSORRT VERSION>.Linux.x86_64-gnu.cuda-<CUDA VERSION>.tar.gz"
            ],
        )

    Remember at runtime, these libraries must be added to your ``LD_LIBRARY_PATH`` explicitly

If you have a local version of TensorRT installed, this can be used as well by commenting out the above lines and uncommenting the following lines https://github.com/pytorch/TensorRT/blob/4e5b0f6e860910eb510fa70a76ee3eb9825e7a4d/WORKSPACE#L114C1-L124C3


Building the Package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once the WORKSPACE has been configured properly, all that is required to build torch-tensorrt is the following command

    .. code-block:: sh

        python -m pip install --pre . --extra-index-url https://download.pytorch.org/whl/nightly/cu124


If you use the ``uv`` (`https://docs.astral.sh/uv/ <https://docs.astral.sh/uv/>`_) tool to manage python and your projects, the command is slightly simpler


    .. code-block:: sh

        uv pip install -e .


To build the wheel file

    .. code-block:: sh

        python -m pip wheel --no-deps --pre . --extra-index-url https://download.pytorch.org/whl/nightly/cu124 -w dist

Additional Build Options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some features in the library are optional and allow builds to be lighter or more portable.

Python Only Distribution
............................

There are multiple features of the library which require C++ components to be enabled. This includes both the TorchScript frontend which accepts TorchScript modules for compilation
and the Torch-TensorRT runtime, the default executor for modules compiled with Torch-TensorRT, be it with the TorchScript or Dynamo frontend.

In the case you may want a build which does not require C++ you can disable these features and avoid building these components. As a result, the only available runtime will be the Python based on
which has implications for features like serialization.

.. code-block:: sh

    PYTHON_ONLY=1 python -m pip install --pre . --extra-index-url https://download.pytorch.org/whl/nightly/cu124


No TorchScript Frontend
............................

The TorchScript frontend is a legacy feature of Torch-TensorRT which is now in maintenance as TorchDynamo has become the preferred compiler technology for this project. It contains quite a bit
of C++ code that is no longer necessary for most users. Therefore you can exclude this component from your build to speed up build times. The C++ based runtime will still be available to use.

.. code-block:: sh

    NO_TORCHSCRIPT=1 python -m pip install --pre . --extra-index-url https://download.pytorch.org/whl/nightly/cu124


Building the C++ Library Standalone (TorchScript Only)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Release Build
............................

.. code-block:: shell

    bazel build //:libtorchtrt -c opt

A tarball with the include files and library can then be found in ``bazel-bin``

.. _build-from-archive-debug:

Debug Build
............................

To build with debug symbols use the following command

.. code-block:: shell

    bazel build //:libtorchtrt -c dbg

A tarball with the include files and library can then be found in ``bazel-bin``

Pre CXX11 ABI Build
............................

To build using the pre-CXX11 ABI use the ``pre_cxx11_abi`` config

.. code-block:: shell

    bazel build //:libtorchtrt --config pre_cxx11_abi -c [dbg/opt]

A tarball with the include files and library can then be found in ``bazel-bin``


.. _abis:

Choosing the Right ABI
^^^^^^^^^^^^^^^^^^^^^^^^

Likely the most complicated thing about compiling Torch-TensorRT is selecting the correct ABI. There are two options
which are incompatible with each other, pre-cxx11-abi and the cxx11-abi. The complexity comes from the fact that while
the most popular distribution of PyTorch (wheels downloaded from pytorch.org/pypi directly) use the pre-cxx11-abi, most
other distributions you might encounter (e.g. ones from NVIDIA - NGC containers, and builds for Jetson as well as certain
libtorch builds and likely if you build PyTorch from source) use the cxx11-abi. It is important you compile Torch-TensorRT
using the correct ABI to function properly. Below is a table with general pairings of PyTorch distribution sources and the
recommended commands:

+-------------------------------------------------------------+----------------------------------------------------------+--------------------------------------------------------------------+
| PyTorch Source                                              | Recommended Python Compilation Command                   | Recommended C++ Compilation Command                                |
+=============================================================+==========================================================+====================================================================+
| PyTorch whl file from PyTorch.org                           | python -m pip install .                                  | bazel build //:libtorchtrt -c opt --config pre_cxx11_abi           |
+-------------------------------------------------------------+----------------------------------------------------------+--------------------------------------------------------------------+
| libtorch-shared-with-deps-*.zip from PyTorch.org            | python -m pip install .                                  | bazel build //:libtorchtrt -c opt --config pre_cxx11_abi           |
+-------------------------------------------------------------+----------------------------------------------------------+--------------------------------------------------------------------+
| libtorch-cxx11-abi-shared-with-deps-*.zip from PyTorch.org  | python setup.py bdist_wheel --use-cxx11-abi              | bazel build //:libtorchtrt -c opt                                  |
+-------------------------------------------------------------+----------------------------------------------------------+--------------------------------------------------------------------+
| PyTorch preinstalled in an NGC container                    | python setup.py bdist_wheel --use-cxx11-abi              | bazel build //:libtorchtrt -c opt                                  |
+-------------------------------------------------------------+----------------------------------------------------------+--------------------------------------------------------------------+
| PyTorch from the NVIDIA Forums for Jetson                   | python setup.py bdist_wheel --use-cxx11-abi              | bazel build //:libtorchtrt -c opt                                  |
+-------------------------------------------------------------+----------------------------------------------------------+--------------------------------------------------------------------+
| PyTorch built from Source                                   | python setup.py bdist_wheel --use-cxx11-abi              | bazel build //:libtorchtrt -c opt                                  |
+-------------------------------------------------------------+----------------------------------------------------------+--------------------------------------------------------------------+

    NOTE: For all of the above cases you must correctly declare the source of PyTorch you intend to use in your WORKSPACE file for both Python and C++ builds. See below for more information



Building on Windows
~~~~~~~~~~~~~~~~~~~~~~~~~~~


* Microsoft VS 2022 Tools
* Bazelisk
* CUDA


Build steps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Open the app "x64 Native Tools Command Prompt for VS 2022" - note that Admin privileges may be necessary
* Ensure Bazelisk (Bazel launcher) is installed on your machine and available from the command line. Package installers such as Chocolatey can be used to install Bazelisk
* Install latest version of Torch (i.e. with ``pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124``)
* Clone the Torch-TensorRT repository and navigate to its root directory
* Run ``pip install ninja wheel setuptools``
* Run ``pip install --pre -r py/requirements.txt``
* Run ``set DISTUTILS_USE_SDK=1``
* Run ``python setup.py bdist_wheel``
* Run ``pip install dist/*.whl``

Advanced setup and Troubleshooting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the ``WORKSPACE`` file, the ``cuda_win``, ``libtorch_win``, and ``tensorrt_win`` are Windows-specific modules which can be customized. For instance, if you would like to build with a different version of CUDA, or your CUDA installation is in a non-standard location, update the `path` in the `cuda_win` module.

Similarly, if you would like to use a different version of pytorch or tensorrt, customize the `urls` in the ``libtorch_win`` and ``tensorrt_win`` modules, respectively.

Local versions of these packages can also be used on Windows. See ``toolchains\\ci_workspaces\\WORKSPACE.win.release.tmpl`` for an example of using a local version of TensorRT on Windows.


Alternative Build Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Building with CMake (TorchScript Only)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is possible to build the API libraries (in cpp/) and the torchtrtc executable using CMake instead of Bazel.
Currently, the python API and the tests cannot be built with CMake.
Begin by installing CMake.

    * Latest releases of CMake and instructions on how to install are available for different platforms
      [on their website](https://cmake.org/download/).

A few useful CMake options include:

    * CMake finders for TensorRT are provided in `cmake/Modules`. In order for CMake to use them, pass
      `-DCMAKE_MODULE_PATH=cmake/Modules` when configuring the project with CMake.
    * Libtorch provides its own CMake finder. In case CMake doesn't find it, pass the path to your install of
      libtorch with `-DTorch_DIR=<path to libtorch>/share/cmake/Torch`
    * If TensorRT is not found with the provided cmake finder, specify `-DTensorRT_ROOT=<path to TensorRT>`
    * Finally, configure and build the project in a build directory of your choice with the following command
      from the root of Torch-TensorRT project:

    .. code-block:: shell

        cmake -S. -B<build directory> \
            [-DCMAKE_MODULE_PATH=cmake/Module] \
            [-DTorch_DIR=<path to libtorch>/share/cmake/Torch] \
            [-DTensorRT_ROOT=<path to TensorRT>] \
            [-DCMAKE_BUILD_TYPE=Debug|Release]
        cmake --build <build directory>


Building Natively on aarch64 (Jetson)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Prerequisites
............................

Install or compile a build of PyTorch/LibTorch for aarch64

NVIDIA hosts builds the latest release branch for Jetson here:

    https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048


Environment Setup
............................

To build natively on aarch64-linux-gnu platform, configure the ``WORKSPACE`` with local available dependencies.

1. Replace ``WORKSPACE`` with the corresponding WORKSPACE file in ``//toolchains/jp_workspaces``

2. Configure the correct paths to directory roots containing local dependencies in the ``new_local_repository`` rules:

    NOTE: If you installed PyTorch using a pip package, the correct path is the path to the root of the python torch package.
    In the case that you installed with ``sudo pip install`` this will be ``/usr/local/lib/python3.8/dist-packages/torch``.
    In the case you installed with ``pip install --user`` this will be ``$HOME/.local/lib/python3.8/site-packages/torch``.

In the case you are using NVIDIA compiled pip packages, set the path for both libtorch sources to the same path. This is because unlike
PyTorch on x86_64, NVIDIA aarch64 PyTorch uses the CXX11-ABI. If you compiled for source using the pre_cxx11_abi and only would like to
use that library, set the paths to the same path but when you compile make sure to add the flag ``--config=pre_cxx11_abi``

.. code-block:: shell

    new_local_repository(
        name = "libtorch",
        path = "/usr/local/lib/python3.8/dist-packages/torch",
        build_file = "third_party/libtorch/BUILD"
    )

    new_local_repository(
        name = "libtorch_pre_cxx11_abi",
        path = "/usr/local/lib/python3.8/dist-packages/torch",
        build_file = "third_party/libtorch/BUILD"
    )


Compile C++ Library and Compiler CLI
........................................................

    NOTE: Due to shifting dependency locations between Jetpack 4.5 and 4.6 there is a now a flag to inform bazel of the Jetpack version

    .. code-block:: shell

        --platforms //toolchains:jetpack_x.x


Compile Torch-TensorRT library using bazel command:

.. code-block:: shell

   bazel build //:libtorchtrt --platforms //toolchains:jetpack_5.0

Compile Python API
............................

    NOTE: Due to shifting dependencies locations between Jetpack 4.5 and newer Jetpack versions there is now a flag for ``setup.py`` which sets the jetpack version (default: 5.0)

Compile the Python API using the following command from the ``//py`` directory:

.. code-block:: shell

    python3 setup.py install --use-cxx11-abi

If you have a build of PyTorch that uses Pre-CXX11 ABI drop the ``--use-cxx11-abi`` flag

If you are building for Jetpack 4.5 add the ``--jetpack-version 5.0`` flag
