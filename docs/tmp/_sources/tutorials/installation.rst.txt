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

    pip3 install trtorch -f https://github.com/NVIDIA/TRTorch/releases

.. _bin-dist:

C++ Binary Distribution
------------------------

Precompiled tarballs for releases are provided here: https://github.com/NVIDIA/TRTorch/releases

.. _compile-from-source:

Compiling From Source
******************************************

.. _installing-deps:

Dependencies for Compilation
-------------------------------

TRTorch is built with Bazel, so begin by installing it.

    * The easiest way is to install bazelisk using the method of you choosing https://github.com/bazelbuild/bazelisk
    * Otherwise you can use the following instructions to install binaries https://docs.bazel.build/versions/master/install.html
    * Finally if you need to compile from source (e.g. aarch64 until bazel distributes binaries for the architecture) you can use these instructions

    .. code-block:: shell

        export BAZEL_VERSION=$(cat <PATH_TO_TRTORCH_ROOT>/.bazelversion)
        mkdir bazel
        cd bazel
        curl -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-dist.zip
        unzip bazel-$BAZEL_VERSION-dist.zip
        bash ./compile.sh
        cp output/bazel /usr/local/bin/


You will also need to have CUDA installed on the system (or if running in a container, the system must have
the CUDA driver installed and the container must have CUDA)

The correct LibTorch version will be pulled down for you by bazel.

    NOTE: For best compatability with official PyTorch, use torch==1.9.1+cuda111, TensorRT 8.0 and cuDNN 8.2 for CUDA 11.1 however TRTorch itself supports
    TensorRT and cuDNN for CUDA versions other than 11.1 for usecases such as using NVIDIA compiled distributions of PyTorch that use other versions of CUDA
    e.g. aarch64 or custom compiled version of PyTorch.

You then have two compilation options:

.. _build-from-archive:

**Building using cuDNN & TensorRT tarball distributions**
--------------------------------------------------------------

    This is recommended so as to build TRTorch hermetically and insures any compilation errors are not caused by version issues

    Make sure when running TRTorch that these versions of the libraries are prioritized in your ``$LD_LIBRARY_PATH``

You need to download the tarball distributions of TensorRT and cuDNN from the NVIDIA website.
    * https://developer.nvidia.com/cudnn
    * https://developer.nvidia.com/tensorrt

Place these files in a directory (the directories ``thrid_party/distdir/[x86_64-linux-gnu | aarch64-linux-gnu]`` exist for this purpose)

Then compile referencing the directory with the tarballs

    If you get errors regarding the packages, check their sha256 hashes and make sure they match the ones listed in ``WORKSPACE``

Release Build
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

    bazel build //:libtrtorch -c opt --distdir thrid_party/distdir/[x86_64-linux-gnu | aarch64-linux-gnu]

A tarball with the include files and library can then be found in ``bazel-bin``

.. _build-from-archive-debug:

Debug Build
^^^^^^^^^^^^^^^^^^^^^^^^

To build with debug symbols use the following command

.. code-block:: shell

    bazel build //:libtrtorch -c dbg --distdir thrid_party/distdir/[x86_64-linux-gnu | aarch64-linux-gnu]

A tarball with the include files and library can then be found in ``bazel-bin``

Pre CXX11 ABI Build
^^^^^^^^^^^^^^^^^^^^^^^^

To build using the pre-CXX11 ABI use the ``pre_cxx11_abi`` config

.. code-block:: shell

    bazel build //:libtrtorch --config pre_cxx11_abi -c [dbg/opt] --distdir thrid_party/distdir/[x86_64-linux-gnu | aarch64-linux-gnu]

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
        name = "cudnn",
        urls = ["<URL>",],

        build_file = "@//third_party/cudnn/archive:BUILD",
        sha256 = "<TAR SHA256>",
        strip_prefix = "cuda"
    )

    http_archive(
        name = "tensorrt",
        urls = ["<URL>",],

        build_file = "@//third_party/tensorrt/archive:BUILD",
        sha256 = "<TAR SHA256>",
        strip_prefix = "TensorRT-<VERSION>"
    )

and uncomment

.. code-block:: python

    # Locally installed dependencies
    new_local_repository(
        name = "cudnn",
        path = "/usr/",
        build_file = "@//third_party/cudnn/local:BUILD"
    )

    new_local_repository(
    name = "tensorrt",
    path = "/usr/",
    build_file = "@//third_party/tensorrt/local:BUILD"
    )

Release Build
^^^^^^^^^^^^^^^^^^^^^^^^

Compile using:

.. code-block:: shell

    bazel build //:libtrtorch -c opt

A tarball with the include files and library can then be found in ``bazel-bin``

.. _build-from-local-debug:

Debug Build
^^^^^^^^^^^^

To build with debug symbols use the following command

.. code-block:: shell

    bazel build //:libtrtorch -c dbg


A tarball with the include files and library can then be found in ``bazel-bin``

Pre CXX11 ABI Build
^^^^^^^^^^^^^^^^^^^^^^^^

To build using the pre-CXX11 ABI use the ``pre_cxx11_abi`` config

.. code-block:: shell

    bazel build //:libtrtorch --config pre_cxx11_abi -c [dbg/opt]

**Building the Python package**
--------------------------------

Begin by installing ``ninja``

You can build the Python package using ``setup.py`` (this will also build the correct version of ``libtrtorch.so``)

.. code-block:: shell

    python3 setup.py [install/bdist_wheel]

Debug Build
^^^^^^^^^^^^

.. code-block:: shell

    python3 setup.py develop [--user]

This also compiles a debug build of ``libtrtorch.so``

**Building Natively on aarch64 (Jetson)**
-------------------------------------------

Prerequisites
^^^^^^^^^^^^^^

Install or compile a build of PyTorch/LibTorch for aarch64

NVIDIA hosts builds the latest release branch for Jetson here:

    https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-9-0-now-available/72048


Enviorment Setup
^^^^^^^^^^^^^^^^^

To build natively on aarch64-linux-gnu platform, configure the ``WORKSPACE`` with local available dependencies.

1. Disable the rules with ``http_archive`` for x86_64 by commenting the following rules:

.. code-block:: shell

    #http_archive(
    #    name = "libtorch",
    #    build_file = "@//third_party/libtorch:BUILD",
    #    strip_prefix = "libtorch",
    #    urls = ["https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.5.1.zip"],
    #    sha256 = "cf0691493d05062fe3239cf76773bae4c5124f4b039050dbdd291c652af3ab2a"
    #)

    #http_archive(
    #    name = "libtorch_pre_cxx11_abi",
    #    build_file = "@//third_party/libtorch:BUILD",
    #    strip_prefix = "libtorch",
    #    sha256 = "818977576572eadaf62c80434a25afe44dbaa32ebda3a0919e389dcbe74f8656",
    #    urls = ["https://download.pytorch.org/libtorch/cu102/libtorch-shared-with-deps-1.5.1.zip"],
    #)

    # Download these tarballs manually from the NVIDIA website
    # Either place them in the distdir directory in third_party and use the --distdir flag
    # or modify the urls to "file:///<PATH TO TARBALL>/<TARBALL NAME>.tar.gz

    #http_archive(
    #    name = "cudnn",
    #    urls = ["https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.0.1.13/10.2_20200626/cudnn-10.2-linux-x64-v8.0.1.13.tgz"],
    #    build_file = "@//third_party/cudnn/archive:BUILD",
    #    sha256 = "0c106ec84f199a0fbcf1199010166986da732f9b0907768c9ac5ea5b120772db",
    #    strip_prefix = "cuda"
    #)

    #http_archive(
    #    name = "tensorrt",
    #    urls = ["https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/7.1/tars/TensorRT-7.1.3.4.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.0.tar.gz"],
    #    build_file = "@//third_party/tensorrt/archive:BUILD",
    #    sha256 = "9205bed204e2ae7aafd2e01cce0f21309e281e18d5bfd7172ef8541771539d41",
    #    strip_prefix = "TensorRT-7.1.3.4"
    #)

    NOTE: You may also need to configure the CUDA version to 10.2 by setting the path for the cuda new_local_repository


2. Configure the correct paths to directory roots containing local dependencies in the ``new_local_repository`` rules:

    NOTE: If you installed PyTorch using a pip package, the correct path is the path to the root of the python torch package.
    In the case that you installed with ``sudo pip install`` this will be ``/usr/local/lib/python3.6/dist-packages/torch``.
    In the case you installed with ``pip install --user`` this will be ``$HOME/.local/lib/python3.6/site-packages/torch``.

In the case you are using NVIDIA compiled pip packages, set the path for both libtorch sources to the same path. This is because unlike
PyTorch on x86_64, NVIDIA aarch64 PyTorch uses the CXX11-ABI. If you compiled for source using the pre_cxx11_abi and only would like to
use that library, set the paths to the same path but when you compile make sure to add the flag ``--config=pre_cxx11_abi``

.. code-block:: shell

    new_local_repository(
        name = "libtorch",
        path = "/usr/local/lib/python3.6/dist-packages/torch",
        build_file = "third_party/libtorch/BUILD"
    )

    new_local_repository(
        name = "libtorch_pre_cxx11_abi",
        path = "/usr/local/lib/python3.6/dist-packages/torch",
        build_file = "third_party/libtorch/BUILD"
    )

    new_local_repository(
        name = "cudnn",
        path = "/usr/",
        build_file = "@//third_party/cudnn/local:BUILD"
    )

    new_local_repository(
        name = "tensorrt",
        path = "/usr/",
        build_file = "@//third_party/tensorrt/local:BUILD"
    )

Compile C++ Library and Compiler CLI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    NOTE: Due to shifting dependency locations between Jetpack 4.5 and 4.6 there is a now a flag to inform bazel of the Jetpack version

    .. code-block:: shell

        --platforms //toolchains:jetpack_4.x


Compile TRTorch library using bazel command:

.. code-block:: shell

   bazel build //:libtrtorch --platforms //toolchains:jetpack_4.6

Compile Python API
^^^^^^^^^^^^^^^^^^^^

    NOTE: Due to shifting dependencies locations between Jetpack 4.5 and Jetpack 4.6 there is now a flag for ``setup.py`` which sets the jetpack version (default: 4.6)

Compile the Python API using the following command from the ``//py`` directory:

.. code-block:: shell

    python3 setup.py install --use-cxx11-abi

If you have a build of PyTorch that uses Pre-CXX11 ABI drop the ``--use-cxx11-abi`` flag

If you are building for Jetpack 4.5 add the ``--jetpack-version 4.5`` flag