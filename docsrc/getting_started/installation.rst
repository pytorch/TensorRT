.. _installation:

Installation
=============

Precompiled Binaries
*********************

Dependencies
---------------

You need to have either PyTorch or LibTorch installed based on if you are using Python or C++
and you must have CUDA, cuDNN and TensorRT installed.

    * https://www.pytorch.org
    * https://developer.nvidia.com/cuda
    * https://developer.nvidia.com/cudnn
    * https://developer.nvidia.com/tensorrt


Python Package
---------------

You can install the python package using

.. code-block:: sh

    pip3 install torch-tensorrt -f https://github.com/pytorch/TensorRT/releases

.. _bin-dist:

C++ Binary Distribution
------------------------

Precompiled tarballs for releases are provided here: https://github.com/pytorch/TensorRT/releases

.. _compile-from-source:

Compiling From Source
******************************************

.. _installing-deps:

Dependencies for Compilation
-------------------------------

Torch-TensorRT is built with Bazel, so begin by installing it.

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


You will also need to have CUDA installed on the system (or if running in a container, the system must have
the CUDA driver installed and the container must have CUDA)

The correct LibTorch version will be pulled down for you by bazel.

    NOTE: For best compatability with official PyTorch, use torch==1.10.0+cuda113, TensorRT 8.0 and cuDNN 8.2 for CUDA 11.3 however Torch-TensorRT itself supports
    TensorRT and cuDNN for other CUDA versions for usecases such as using NVIDIA compiled distributions of PyTorch that use other versions of CUDA
    e.g. aarch64 or custom compiled version of PyTorch.

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
| PyTorch Source                                              | Recommended C++ Compilation Command                      | Recommended Python Compilation Command                             |
+=============================================================+==========================================================+====================================================================+
| PyTorch whl file from PyTorch.org                           | bazel build //:libtorchtrt -c opt --config pre_cxx11_abi | python3 setup.py bdist_wheel                                       |
+-------------------------------------------------------------+----------------------------------------------------------+--------------------------------------------------------------------+
| libtorch-shared-with-deps-*.zip from PyTorch.org            | bazel build //:libtorchtrt -c opt --config pre_cxx11_abi | python3 setup.py bdist_wheel                                       |
+-------------------------------------------------------------+----------------------------------------------------------+--------------------------------------------------------------------+
| libtorch-cxx11-abi-shared-with-deps-*.zip from PyTorch.org  | bazel build //:libtorchtrt -c opt                        | python3 setup.py bdist_wheel --use-cxx11-abi                       |
+-------------------------------------------------------------+----------------------------------------------------------+--------------------------------------------------------------------+
| PyTorch preinstalled in an NGC container                    | bazel build //:libtorchtrt -c opt                        | python3 setup.py bdist_wheel --use-cxx11-abi                       |
+-------------------------------------------------------------+----------------------------------------------------------+--------------------------------------------------------------------+
| PyTorch from the NVIDIA Forums for Jetson                   | bazel build //:libtorchtrt -c opt                        | python3 setup.py bdist_wheel --jetpack-version 4.6 --use-cxx11-abi |
+-------------------------------------------------------------+----------------------------------------------------------+--------------------------------------------------------------------+
| PyTorch built from Source                                   | bazel build //:libtorchtrt -c opt                        | python3 setup.py bdist_wheel --use-cxx11-abi                       |
+-------------------------------------------------------------+----------------------------------------------------------+--------------------------------------------------------------------+

    NOTE: For all of the above cases you must correctly declare the source of PyTorch you intend to use in your WORKSPACE file for both Python and C++ builds. See below for more information

You then have two compilation options:

.. _build-from-archive:

**Building using cuDNN & TensorRT tarball distributions**
--------------------------------------------------------------

    This is recommended so as to build Torch-TensorRT hermetically and insures any compilation errors are not caused by version issues

    Make sure when running Torch-TensorRT that these versions of the libraries are prioritized in your ``$LD_LIBRARY_PATH``

You need to download the tarball distributions of TensorRT and cuDNN from the NVIDIA website.
    * https://developer.nvidia.com/cudnn
    * https://developer.nvidia.com/tensorrt

Place these files in a directory (the directories ``third_party/distdir/[x86_64-linux-gnu | aarch64-linux-gnu]`` exist for this purpose)

Then compile referencing the directory with the tarballs

    If you get errors regarding the packages, check their sha256 hashes and make sure they match the ones listed in ``WORKSPACE``

Release Build
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

    bazel build //:libtorchtrt -c opt --distdir third_party/distdir/[x86_64-linux-gnu | aarch64-linux-gnu]

A tarball with the include files and library can then be found in ``bazel-bin``

.. _build-from-archive-debug:

Debug Build
^^^^^^^^^^^^^^^^^^^^^^^^

To build with debug symbols use the following command

.. code-block:: shell

    bazel build //:libtorchtrt -c dbg --distdir third_party/distdir/[x86_64-linux-gnu | aarch64-linux-gnu]

A tarball with the include files and library can then be found in ``bazel-bin``

Pre CXX11 ABI Build
^^^^^^^^^^^^^^^^^^^^^^^^

To build using the pre-CXX11 ABI use the ``pre_cxx11_abi`` config

.. code-block:: shell

    bazel build //:libtorchtrt --config pre_cxx11_abi -c [dbg/opt] --distdir third_party/distdir/[x86_64-linux-gnu | aarch64-linux-gnu]

A tarball with the include files and library can then be found in ``bazel-bin``

.. _build-from-local:

**Building using locally installed cuDNN & TensorRT**
--------------------------------------------------------------

    If you encounter bugs and you compiled using this method please disclose that you used local sources in the issue (an ldd dump would be nice too)

Install TensorRT, CUDA and cuDNN on the system before starting to compile.

In WORKSPACE comment out:

.. code-block:: python

    # Downloaded distributions to use with --distdir
    http_archive(
        name="cudnn",
        urls=[
            "<URL>",
        ],
        build_file="@//third_party/cudnn/archive:BUILD",
        sha256="<TAR SHA256>",
        strip_prefix="cuda",
    )

    http_archive(
        name="tensorrt",
        urls=[
            "<URL>",
        ],
        build_file="@//third_party/tensorrt/archive:BUILD",
        sha256="<TAR SHA256>",
        strip_prefix="TensorRT-<VERSION>",
    )

and uncomment

.. code-block:: python

    # Locally installed dependencies
    new_local_repository(
        name="cudnn", path="/usr/", build_file="@//third_party/cudnn/local:BUILD"
    )

    new_local_repository(
        name="tensorrt", path="/usr/", build_file="@//third_party/tensorrt/local:BUILD"
    )

Release Build
^^^^^^^^^^^^^^^^^^^^^^^^

Compile using:

.. code-block:: shell

    bazel build //:libtorchtrt -c opt

A tarball with the include files and library can then be found in ``bazel-bin``

.. _build-from-local-debug:

Debug Build
^^^^^^^^^^^^

To build with debug symbols use the following command

.. code-block:: shell

    bazel build //:libtorchtrt -c dbg


A tarball with the include files and library can then be found in ``bazel-bin``

Pre CXX11 ABI Build
^^^^^^^^^^^^^^^^^^^^^^^^

To build using the pre-CXX11 ABI use the ``pre_cxx11_abi`` config

.. code-block:: shell

    bazel build //:libtorchtrt --config pre_cxx11_abi -c [dbg/opt]

**Building with CMake**
-----------------------

It is possible to build the API libraries (in cpp/) and the torchtrtc executable using CMake instead of Bazel.
Currently, the python API and the tests cannot be built with CMake.
Begin by installing CMake.

    * Latest releases of CMake and instructions on how to install are available for different platforms
      [on their website](https://cmake.org/download/).

A few useful CMake options include:

    * CMake finders for TensorRT and cuDNN are provided in `cmake/Modules`. In order for CMake to use them, pass
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

**Building the Python package**
--------------------------------

Begin by installing ``ninja``

You can build the Python package using ``setup.py`` (this will also build the correct version of ``libtorchtrt.so``)

.. code-block:: shell

    python3 setup.py [install/bdist_wheel]

Debug Build
^^^^^^^^^^^^

.. code-block:: shell

    python3 setup.py develop [--user]

This also compiles a debug build of ``libtorchtrt.so``

**Building Natively on aarch64 (Jetson)**
-------------------------------------------

Prerequisites
^^^^^^^^^^^^^^

Install or compile a build of PyTorch/LibTorch for aarch64

NVIDIA hosts builds the latest release branch for Jetson here:

    https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048


Enviorment Setup
^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    NOTE: Due to shifting dependency locations between Jetpack 4.5 and 4.6 there is a now a flag to inform bazel of the Jetpack version

    .. code-block:: shell

        --platforms //toolchains:jetpack_x.x


Compile Torch-TensorRT library using bazel command:

.. code-block:: shell

   bazel build //:libtorchtrt --platforms //toolchains:jetpack_5.0

Compile Python API
^^^^^^^^^^^^^^^^^^^^

    NOTE: Due to shifting dependencies locations between Jetpack 4.5 and newer Jetpack verisons there is now a flag for ``setup.py`` which sets the jetpack version (default: 5.0)

Compile the Python API using the following command from the ``//py`` directory:

.. code-block:: shell

    python3 setup.py install --use-cxx11-abi

If you have a build of PyTorch that uses Pre-CXX11 ABI drop the ``--use-cxx11-abi`` flag

If you are building for Jetpack 4.5 add the ``--jetpack-version 5.0`` flag
