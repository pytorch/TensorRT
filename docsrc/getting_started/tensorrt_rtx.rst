.. _Torch-TensorRT-RTX:

Torch-TensorRT-RTX
=====================

Overview
--------

TensorRT-RTX
~~~~~~~~~~~~

TensorRT for RTX builds on the proven performance of the NVIDIA TensorRT inference library, and simplifies the deployment of AI models on NVIDIA RTX GPUs across desktops, laptops, and workstations.

TensorRT for RTX is a drop-in replacement for NVIDIA TensorRT in applications targeting NVIDIA RTX GPUs from Turing through Blackwell generations. It introduces a Just-In-Time (JIT) optimizer in the runtime that compiles improved inference engines directly on the end-user’s RTX-accelerated PC in under 30 seconds. This eliminates the need for lengthy pre-compilation steps and enables rapid engine generation, improved application portability, and cutting-edge inference performance.

For detailed information about TensorRT-RTX, refer to:

* `TensorRT-RTX Documentation <https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/index.html>`_

Currently, Torch-TensorRT only supports TensorRT-RTX for experimental purposes.
Torch-TensorRT by default uses standard TensorRT during the build and run.

To use TensorRT-RTX:

- Build the wheel with the ``--use-rtx`` flag or set ``USE_TRT_RTX=true``.
- During runtime, set the ``USE_TRT_RTX=true`` environment variable to invoke TensorRT-RTX.

Prerequisites
-------------

Clone the Repository
~~~~~~~~~~~~~~~~~~~~~

First, clone the Torch-TensorRT repository:

.. code-block:: sh

   git clone https://github.com/pytorch/TensorRT.git
   cd TensorRT

Install System Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**In Linux:**

Install Python development headers (required for building Python extensions):

.. code-block:: sh

   # For Python 3.12 (adjust version number based on your Python version)
   sudo apt install python3.12-dev

Install CUDA Toolkit
~~~~~~~~~~~~~~~~~~~~

Download and install the CUDA Toolkit from the `NVIDIA Developer website <https://developer.nvidia.com/cuda-downloads>`_.

**Important:** Check the required CUDA version in the `MODULE.bazel <https://github.com/pytorch/TensorRT/blob/main/MODULE.bazel>`_ file. You must install the exact CUDA toolkit version specified there (for example, at the time of writing, CUDA 13.0 is required).

After installation, set the ``CUDA_HOME`` environment variable:

.. code-block:: sh

   export CUDA_HOME=/usr/local/cuda
   # Add this to your ~/.bashrc or ~/.zshrc to make it persistent

Install Python Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**It is strongly recommended to use a virtual environment** to avoid conflicts with system packages:

.. code-block:: sh

   # Create a virtual environment
   python -m venv .venv

   # Activate the virtual environment
   source .venv/bin/activate  # On Linux/Mac
   # OR on Windows:
   # .venv\Scripts\activate

Before building, install the required Python packages:

.. code-block:: sh

   # Install setuptools (provides distutils)
   pip install setuptools

   # Install PyTorch nightly build (check CUDA version in MODULE.bazel)
   # Replace cuXXX with your CUDA version (e.g., cu130 for CUDA 13.0)
   pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cuXXX

   # If you encounter version conflicts during the build, you may need to specify
   # the exact PyTorch version constraint. Check pyproject.toml for requirements.
   # For example, if pyproject.toml specifies torch>=2.10.0.dev,<2.11.0:
   # pip install --pre "torch>=2.10.0.dev,<2.11.0" torchvision --index-url https://download.pytorch.org/whl/nightly/cu130

   # Install additional build dependencies
   pip install pyyaml numpy

.. note::

   The PyTorch version requirement is defined in `pyproject.toml <https://github.com/pytorch/TensorRT/blob/main/pyproject.toml>`_ (build requirements) and `setup.py <https://github.com/pytorch/TensorRT/blob/main/setup.py>`_ (runtime requirements). If you encounter version-related errors during installation, refer to these files for the exact version constraints.

.. note::

   Remember to activate the virtual environment (``source .venv/bin/activate``) whenever you work with this project or run the build commands.

Install Bazel
~~~~~~~~~~~~~

Bazel is required to build the wheel with TensorRT-RTX.

**In Linux:**

.. code-block:: sh

   curl -L https://github.com/bazelbuild/bazelisk/releases/download/v1.26.0/bazelisk-linux-amd64 \
    -o bazelisk \
    && mv bazelisk /usr/bin/bazel \
    && chmod +x /usr/bin/bazel

**In Windows:**

.. code-block:: sh

   choco install bazelisk -y

Install TensorRT-RTX Tarball
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TensorRT-RTX tarball can be downloaded from https://developer.nvidia.com/tensorrt-rtx.
Currently, Torch-TensorRT uses TensorRT-RTX version **1.2.0.54**.

Once downloaded:

**In Linux:**

Make sure you add the lib path to the ``LD_LIBRARY_PATH`` environment variable.

.. code-block:: sh

    # If TensorRT-RTX is downloaded in /your_local_download_path/TensorRT-RTX-1.2.0.54
    export LD_LIBRARY_PATH=/your_local_download_path/TensorRT-RTX-1.2.0.54/lib:$LD_LIBRARY_PATH
    echo $LD_LIBRARY_PATH | grep TensorRT-RTX

**In Windows:**

Make sure you add the lib path to the Windows system variable ``PATH``.

.. code-block:: sh

    # If TensorRT-RTX is downloaded in C:\your_local_download_path\TensorRT-RTX-1.2.0.54
    set PATH="%PATH%;C:\your_local_download_path\TensorRT-RTX-1.2.0.54\lib"
    echo %PATH% | findstr TensorRT-RTX

Install TensorRT-RTX Wheel
~~~~~~~~~~~~~~~~~~~~~~~~~~

Currently, the `tensorrt_rtx` wheel is not published on PyPI.
You must install it manually from the downloaded tarball.

.. code-block:: sh

   # If the tarball is downloaded in /your_local_download_path/TensorRT-RTX-1.2.0.54
   python -m pip install /your_local_download_path/TensorRT-RTX-1.2.0.54/python/tensorrt_rtx-1.2.0.54-cp39-none-linux_x86_64.whl

Build Torch-TensorRT with TensorRT-RTX
--------------------------------------

Build Locally with TensorRT-RTX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before building, ensure you have completed all the prerequisite steps above, including:

- Cloning the repository
- Installing Python dependencies (setuptools, torch, pyyaml, numpy)
- Setting CUDA_HOME environment variable
- Installing the correct CUDA toolkit version
- Installing Python development headers
- Installing Bazel

Then build the wheel:

.. code-block:: sh

    # If you have previously built with standard TensorRT, make sure to clean the build environment,
    # otherwise it will use the existing .so built with standard TensorRT, which is not compatible with TensorRT-RTX.
    python setup.py clean
    bazel clean --expunge
    # Remove everything under build directory
    rm -rf build/*

    # Build wheel with TensorRT-RTX
    python setup.py bdist_wheel --use-rtx

    # Install the wheel (note: the wheel filename uses underscores, not hyphens)
    python -m pip install dist/torch_tensorrt-*.whl

Quick Start
-----------

.. code-block:: python

    # You must set USE_TRT_RTX=true to use TensorRT-RTX
    USE_TRT_RTX=true python examples/dynamo/torch_compile_resnet_example.py

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Missing distutils module**

If you encounter ``ModuleNotFoundError: No module named 'distutils'``, install setuptools:

.. code-block:: sh

    pip install setuptools

**Missing CUDA_HOME environment variable**

If you encounter ``OSError: CUDA_HOME environment variable is not set``, set the CUDA_HOME path:

.. code-block:: sh

    export CUDA_HOME=/usr/local/cuda

**CUDA version mismatch**

If you encounter errors about CUDA paths not existing (e.g., ``/usr/local/cuda-X.Y/ does not exist``), ensure you have the correct CUDA version installed. Check the required version in `MODULE.bazel <https://github.com/pytorch/TensorRT/blob/main/MODULE.bazel>`_. You may need to:

1. Update your NVIDIA drivers
2. Download and install the specific CUDA toolkit version required by MODULE.bazel
3. Clean and rebuild after installing the correct version

**PyTorch version mismatch**

If you encounter an error like ``ERROR: No matching distribution found for torch<X.Y.Z,>=X.Y.Z.dev`` (for example, ``torch<2.11.0,>=2.10.0.dev``), you need to install a compatible PyTorch nightly version.

First, check the exact version constraint in `pyproject.toml <https://github.com/pytorch/TensorRT/blob/main/pyproject.toml>`_, then install with that constraint:

.. code-block:: sh

    # Example: if pyproject.toml requires torch>=2.10.0.dev,<2.11.0
    # and MODULE.bazel specifies CUDA 13.0 (cu130):
    pip install --pre "torch>=2.10.0.dev,<2.11.0" torchvision --index-url https://download.pytorch.org/whl/nightly/cu130

Replace the version constraint and CUDA version (cuXXX) according to your project's requirements.

**Missing Python development headers**

If you encounter ``fatal error: Python.h: No such file or directory``, install the Python development package:

.. code-block:: sh

    # For Python 3.12 (adjust version based on your Python)
    sudo apt install python3.12-dev

Verifying TensorRT-RTX Linkage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you encounter load or link errors, check if `tensorrt_rtx` is linked correctly.
If not, clean up the environment and rebuild.

**In Linux:**

.. code-block:: sh

    # Ensure only tensorrt_rtx is installed (no standard tensorrt wheels)
    python -m pip list | grep tensorrt

    # Check if libtorchtrt.so links to the correct tensorrt_rtx shared object
    trt_install_path=$(python -m pip show torch-tensorrt | grep "Location" | awk '{print $2}')/torch_tensorrt

    # Verify libtensorrt_rtx.so.1 is linked, and libnvinfer.so.10 is NOT
    ldd $trt_install_path/lib/libtorchtrt.so

**In Windows:**

.. code-block:: sh

    # Check if tensorrt_rtx_1_0.dll is linked, and libnvinfer.dll is NOT
    cd py/torch_tensorrt
    dumpbin /DEPENDENTS torchtrt.dll
