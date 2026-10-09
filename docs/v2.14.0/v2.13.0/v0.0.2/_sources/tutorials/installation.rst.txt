.. _installation:

Installation
=============

Dependencies
******************************************

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

    # Python 3.5
    pip3 install  https://github.com/NVIDIA/TRTorch/releases/download/v0.0.2/trtorch-0.0.2-cp35-cp35m-linux_x86_64.whl
    # Python 3.6
    pip3 install  https://github.com/NVIDIA/TRTorch/releases/download/v0.0.2/trtorch-0.0.2-cp36-cp36m-linux_x86_64.whl
    # Python 3.7
    pip3 install  https://github.com/NVIDIA/TRTorch/releases/download/v0.0.2/trtorch-0.0.2-cp37-cp37m-linux_x86_64.whl
    # Python 3.8
    pip3 install  https://github.com/NVIDIA/TRTorch/releases/download/v0.0.2/trtorch-0.0.2-cp38-cp38-linux_x86_64.whl

.. _bin-dist:

Binary Distribution
----------------------

Precompiled tarballs for releases are provided here: https://github.com/NVIDIA/TRTorch/releases

.. _compile-from-source:

Compiling From Source
----------------------

.. _installing-deps:

Dependencies for Compilation
******************************************

TRTorch is built with Bazel, so begin by installing it. https://docs.bazel.build/versions/master/install.html

You will also need to have CUDA installed on the system (or if running in a container, the system must have
the CUDA driver installed and the container must have CUDA)

The correct LibTorch version will be pulled down for you by bazel.

You then have two compilation options:

.. _build-from-archive:

**Building using cuDNN & TensorRT tarball distributions**
***********************************************************

    This is recommended so as to build TRTorch hermetically and insures any compilation errors are not caused by version issues

    Make sure when running TRTorch that these versions of the libraries are prioritized in your ``$LD_LIBRARY_PATH``

You need to download the tarball distributions of TensorRT and cuDNN from the NVIDIA website.
    * https://developer.nvidia.com/cudnn
    * https://developer.nvidia.com/tensorrt

Place these files in a directory (the directories ``thrid_party/distdir/[x86_64-linux-gnu | aarch64-linux-gnu]`` exist for this purpose)

Then compile referencing the directory with the tarballs

    If you get errors regarding the packages, check their sha256 hashes and make sure they match the ones listed in ``WORKSPACE``

.. code-block:: shell

    bazel build //:libtrtorch -c opt --distdir thrid_party/distdir/[x86_64-linux-gnu | aarch64-linux-gnu]

A tarball with the include files and library can then be found in ``bazel-bin``

.. _build-from-archive-debug:

Debug build
^^^^^^^^^^^^^

.. code-block:: shell

    bazel build //:libtrtorch -c dbg --distdir thrid_party/distdir/[x86_64-linux-gnu | aarch64-linux-gnu]

Pre CXX11 ABI build
^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

    bazel build //:libtrtorch --config pre_cxx11_abi -c [dbg/opt] --distdir thrid_party/distdir/[x86_64-linux-gnu | aarch64-linux-gnu]

A tarball with the include files and library can then be found in ``bazel-bin``

.. _build-from-local:

**Building using locally installed cuDNN & TensorRT**
******************************************************

    If you encounter bugs and you compiled using this method please disclose it in the issue (an ldd dump would be nice too)

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

Compile using:

.. code-block:: shell

    bazel build //:libtrtorch -c opt

A tarball with the include files and library can then be found in ``bazel-bin``

.. _build-from-local-debug:

Debug build
^^^^^^^^^^^^

.. code-block:: shell

    bazel build //:libtrtorch -c dbg


A tarball with the include files and library can then be found in ``bazel-bin``

Pre CXX11 ABI build
^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

    bazel build //:libtrtorch --config pre_cxx11_abi -c [dbg/opt]

Building the Python package
-----------------------------

Begin by installing ``ninja``

You can build the Python package using ``setup.py`` (this will also build the correct version of ``libtrtorch.so``)

.. code-block:: shell

    python3 setup.py [install/bdist_wheel]

Debug build
^^^^^^^^^^^^

.. code-block:: shell

    python3 setup.py develop [--user]

This also compiles a debug build of ``libtrtorch.so``
