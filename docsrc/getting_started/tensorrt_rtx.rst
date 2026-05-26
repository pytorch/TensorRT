.. _Torch-TensorRT-RTX:

Torch-TensorRT for RTX
########################

Torch-TensorRT supports TensorRT for RTX, which builds on the proven
performance of the NVIDIA TensorRT inference library, and simplifies the
deployment of AI models on NVIDIA RTX GPUs across desktops, laptops, and
workstations.

TensorRT for RTX is a drop-in replacement for NVIDIA TensorRT in applications
targeting NVIDIA RTX GPUs from Turing through Blackwell generations. It
introduces a Just-In-Time (JIT) optimizer in the runtime that compiles
improved inference engines directly on the end-user's RTX-accelerated PC in
under 30 seconds. This eliminates the need for lengthy pre-compilation steps
and enables rapid engine generation, improved application portability, and
cutting-edge inference performance.

Currently, Torch-TensorRT only supports TensorRT-RTX for experimental purposes;
Torch-TensorRT by default uses standard TensorRT during the build and run. For
detailed information about TensorRT-RTX itself, see the
`TensorRT-RTX documentation <https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/index.html>`_.

Precompiled Binaries
---------------------

Dependencies
~~~~~~~~~~~~~~

You need to have CUDA, PyTorch, and an NVIDIA RTX GPU (Turing through Blackwell)
to use Torch-TensorRT for RTX.

    * https://developer.nvidia.com/cuda
    * https://pytorch.org


Installing Torch-TensorRT for RTX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can install the python package using

.. code-block:: sh

    python -m pip install torch torch-tensorrt-rtx

Packages are uploaded for Linux on x86 and Windows.


Installing Nightly Builds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Torch-TensorRT for RTX distributes nightlies targeting the PyTorch nightly.
These can be installed from the PyTorch nightly package index (separated by
CUDA version):

.. code-block:: sh

    python -m pip install --pre torch torch_tensorrt_rtx --extra-index-url https://download.pytorch.org/whl/nightly/cu130

.. note::

   Both the stable and nightly ``torch_tensorrt_rtx`` wheels bundle
   ``tensorrt_rtx`` and its CUDA libs as dependencies — no manual
   TensorRT-RTX tarball download or ``LD_LIBRARY_PATH`` setup is required
   when installing via pip. The manual tarball flow is only needed when
   compiling from source (see below).


Import Test
~~~~~~~~~~~~

After installation, verify the import succeeds:

.. code-block:: sh

    python -c "import torch_tensorrt; print(torch_tensorrt.__version__)"

.. note::

   The Python import path is ``torch_tensorrt`` regardless of which flavor of
   TensorRT (``tensorrt`` or ``tensorrt_rtx``) is installed (the flavor is determined
   by the wheel name: ``torch-tensorrt`` vs. ``torch-tensorrt-rtx``).


Example: RTX-Only Features
--------------------------

The following minimal example compiles a toy convolutional model with dynamic
input shapes and demonstrates two TensorRT-RTX-only ``torch_tensorrt.compile``
keyword arguments:

* ``runtime_cache_path`` — path to an on-disk cache of JIT-compiled engines.
  The cache is populated on first use and reloaded on subsequent runs, so
  repeated invocations of the same compiled module skips JIT compilation. The
  default is a file under the system temp directory; set this to a persistent
  path (e.g. somewhere under your project) to share the cache across runs.
* ``dynamic_shapes_kernel_specialization_strategy`` — controls how TensorRT-RTX
  specializes kernels for the current runtime shape. Accepts ``"lazy"``
  (default; use a generic fallback kernel while a shape-specialized kernel
  compiles asynchronously), ``"eager"`` (block on the current shape until the
  specialized kernel is ready), or ``"none"`` (always use the generic kernel).

Both keyword arguments are silently ignored on standard-TensorRT builds; they
only take effect when the ``torch_tensorrt_rtx`` wheel is installed.

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch_tensorrt


    class ToyConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.conv(x))


    model = ToyConv().eval().cuda()

    inputs = [
        torch_tensorrt.Input(
            min_shape=(1, 3, 224, 224),
            opt_shape=(4, 3, 224, 224),
            max_shape=(8, 3, 224, 224),
            dtype=torch.float32,
        )
    ]

    compiled = torch_tensorrt.compile(
        model,
        ir="dynamo",
        inputs=inputs,
        enabled_precisions={torch.float32},
        use_python_runtime=True,
        # RTX-only: persist JIT-compiled engines across runs.
        runtime_cache_path="/tmp/my_rtx_cache.bin",
        # RTX-only: "lazy" | "eager" | "none".
        dynamic_shapes_kernel_specialization_strategy="eager",
    )

    out = compiled(torch.randn(4, 3, 224, 224).cuda())
    print(out.shape)


Compiling From Source
----------------------

The standard build prerequisites (Bazel, CUDA, Python, PyTorch nightly) are
unchanged for the TensorRT-RTX build — see :ref:`installing-deps` in the main
installation guide for those. Only the RTX-specific deltas are listed below.

RTX-Specific Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

Download the TensorRT-RTX tarball from https://developer.nvidia.com/tensorrt-rtx.
Torch-TensorRT currently uses TensorRT-RTX version **1.4.0.76**.

Once downloaded:

**On Linux**, add the tarball ``lib`` directory to ``LD_LIBRARY_PATH``:

.. code-block:: sh

    # If TensorRT-RTX is extracted in /your_local_download_path/TensorRT-RTX-1.4.0.76
    export LD_LIBRARY_PATH=/your_local_download_path/TensorRT-RTX-1.4.0.76/lib:$LD_LIBRARY_PATH

**On Windows**, add the tarball ``lib`` directory to the system ``PATH``:

.. code-block:: sh

    set PATH=%PATH%;C:\your_local_download_path\TensorRT-RTX-1.4.0.76\lib


Building the Wheel
~~~~~~~~~~~~~~~~~~~

.. warning::

    If you have previously built with standard TensorRT in this tree, clean the
    build environment first — otherwise ``bdist_wheel`` will reuse the existing
    ``.so`` built against standard TensorRT, which is not compatible with
    TensorRT-RTX.

.. code-block:: sh

    python setup.py clean
    bazel clean --expunge
    rm -rf build/*

Then build and install the wheel:

.. code-block:: sh

    # Either flag below works; pick one.
    python setup.py bdist_wheel --use-rtx
    # equivalently:
    USE_TRT_RTX=true python setup.py bdist_wheel

    # Note: the wheel filename uses underscores, not hyphens, and contains 'rtx'.
    python -m pip install dist/torch_tensorrt_rtx-*.whl


Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~~

**Missing distutils module**

If you encounter ``ModuleNotFoundError: No module named 'distutils'``, install
setuptools:

.. code-block:: sh

    pip install setuptools

**Missing CUDA_HOME environment variable**

If you encounter ``OSError: CUDA_HOME environment variable is not set``, set
the ``CUDA_HOME`` path:

.. code-block:: sh

    export CUDA_HOME=/usr/local/cuda

**CUDA version mismatch**

If you encounter errors about CUDA paths not existing (e.g.,
``/usr/local/cuda-X.Y/ does not exist``), ensure you have the correct CUDA
version installed. Check the required version in
`MODULE.bazel <https://github.com/pytorch/TensorRT/blob/main/MODULE.bazel>`_.
You may need to:

1. Update your NVIDIA drivers
2. Download and install the specific CUDA toolkit version required by ``MODULE.bazel``
3. Clean and rebuild after installing the correct version

**PyTorch version mismatch**

If you encounter an error like
``ERROR: No matching distribution found for torch<X.Y.Z,>=X.Y.Z.dev`` (for
example, ``torch<2.11.0,>=2.10.0.dev``), install a compatible PyTorch nightly.
First check the exact version constraint in
`pyproject.toml <https://github.com/pytorch/TensorRT/blob/main/pyproject.toml>`_,
then install with that constraint:

.. code-block:: sh

    # Example: if pyproject.toml requires torch>=2.12.0.dev,<2.13.0
    # and MODULE.bazel specifies CUDA 13.0 (cu130):
    pip install --pre "torch>=2.12.0.dev,<2.13.0" torchvision --index-url https://download.pytorch.org/whl/nightly/cu130

Replace the version constraint and CUDA version (cuXXX) according to your
project's requirements.

**Missing Python development headers**

If you encounter ``fatal error: Python.h: No such file or directory``, install
the Python development package:

.. code-block:: sh

    # For Python 3.12 (adjust version based on your Python)
    sudo apt install python3.12-dev


Verifying TensorRT-RTX Linkage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you encounter load or link errors, check that ``tensorrt_rtx`` is linked
correctly. If not, clean the environment and rebuild.

**Linux:**

.. code-block:: sh

    # Ensure only tensorrt_rtx is installed (no standard tensorrt wheels)
    python -m pip list | grep tensorrt

    # Check if libtorchtrt.so links to the correct tensorrt_rtx shared object
    trt_install_path=$(python -m pip show torch-tensorrt | grep "Location" | awk '{print $2}')/torch_tensorrt

    # Verify libtensorrt_rtx.so.1 is linked, and libnvinfer.so.10 is NOT
    ldd $trt_install_path/lib/libtorchtrt.so

**Windows:**

.. code-block:: sh

    # Check if tensorrt_rtx_1_0.dll is linked, and libnvinfer.dll is NOT
    cd py/torch_tensorrt
    dumpbin /DEPENDENTS torchtrt.dll
