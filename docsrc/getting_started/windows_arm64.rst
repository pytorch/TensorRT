Building Torch-TensorRT-RTX for Windows ARM64
#########################################

This guide describes local TensorRT-RTX builds for Windows ARM64. It covers:

* a native build on a Windows ARM64 machine; and
* a cross-build on a Windows x64 machine that targets Windows ARM64.

The commands below must be run from the root of the Torch-TensorRT repository
in a Visual Studio developer command prompt. The resulting wheel is written to
the ``dist`` directory.

Prerequisites
-------------

Both build types require:

* Visual Studio 2022 Build Tools with the C++ build tools and ARM64 compiler
  components;
* Bazelisk via choco;
* CUDA Toolkit 13.4 Preview. The Windows x86 installation contains the
  headers and ARM64 libraries required to cross-compile for Windows ARM64.
  Native Windows ARM64 builds also use CUDA 13.4;
* a compatible Windows ARM64 PyTorch 2.14 package containing the C++ headers
  and libraries under ``torch\include`` and ``torch\lib``. Nightly wheels are
  available from NVIDIA's ``nvtorch_oot_nightly`` index at
  ``https://pypi.nvidia.com/nvtorch_oot_nightly/torch/``;
* Python packages required by Torch-TensorRT-RTX. The currently available
  Windows ARM64 PyTorch wheel targets CPython 3.13, so native and cross-builds
  are currently supported only with Python 3.13.

If the CUDA Toolkit is installed somewhere other than the path in
``MODULE.bazel``, update the ``cuda_win_arm64`` repository path. Use the same
CUDA root for ``CUDA_PATH``, ``CUDA_HOME``, and
``TORCHTRT_TARGET_CUDA_ROOT`` where those variables are required below.

Native build on Windows ARM64
-----------------------------

1. Open an ARM64-native Visual Studio 2022 developer command prompt. You can
   also initialize one from ``cmd.exe``:

   .. code-block:: bat

      call "%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" arm64
      echo %VSCMD_ARG_HOST_ARCH%
      echo %VSCMD_ARG_TGT_ARCH%
      where cl.exe
      cl.exe

   Both printed architectures should be ``arm64``. Adjust ``BuildTools`` in
   the path if Visual Studio was installed in another edition. ``where
   cl.exe`` should resolve to an ARM64-native compiler path containing
   ``Hostarm64\arm64\cl.exe``, and the ``cl.exe`` banner should identify
   ARM64 as the target architecture.

2. Create and activate a virtual environment using a native ARM64 Python 3.13
   interpreter:

   .. code-block:: bat

      C:\path\to\arm64-python\python.exe -m venv .venv
      call .venv\Scripts\activate.bat
      python -c "import platform; print(platform.machine())"

   The last command must print ``ARM64``. Install the compatible ARM64 PyTorch
   wheel and the prerequisite packages listed above in this environment.

3. Configure CUDA and tell setuptools to reuse the active Visual Studio
   environment:

   .. code-block:: bat

      set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
      set "CUDA_HOME=%CUDA_PATH%"
      set DISTUTILS_USE_SDK=1

4. Build the ARM64 TensorRT-RTX wheel:

   .. code-block:: bat

      python -m pip install --upgrade pip
      python -m pip install numpy packaging pyyaml setuptools==72.1.0 wheel fmt build

      # Install the Windows ARM64 PyTorch 2.14 nightly.
      python -m pip install --pre "torch>=2.14.0.dev,<2.15.0" --index-url https://pypi.nvidia.com/nvtorch_oot_nightly/torch/

      # build the native windows on arm TensorRT-RTX wheel
      python setup.py bdist_wheel --use-rtx

   ``setup.py`` detects the native ARM64 Python interpreter, selects the
   Windows ARM64 Bazel platform, and produces a wheel tagged
   ``win_arm64``. Do not pass ``--windows-on-arm`` for a native build.

5. Confirm the wheel tag and test it in the ARM64 environment:

   .. code-block:: bat

      dir dist\*win_arm64.whl
      python -m pip install C:\path\to\the\generated-win_arm64.whl
      python -c "import torch_tensorrt; print(torch_tensorrt.__version__)"

Cross-build on Windows x64
--------------------------

The cross-build uses windows-x64 to run the build and the x64-hosted ARM64 MSVC
compiler to create ARM64 binaries. The host and target Python installations
must have the same major and minor version. Use Python 3.13 for both the x64
host and the ARM64 target because the available Windows ARM64 PyTorch wheel
and the resulting Torch-TensorRT wheel target CPython 3.13.

1. Open an x64-to-ARM64 Visual Studio 2022 cross-tools prompt, or initialize
   one from ``cmd.exe``:

   .. code-block:: bat

      call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64_arm64
      echo Host=%VSCMD_ARG_HOST_ARCH%
      echo Target=%VSCMD_ARG_TGT_ARCH%
      echo Tools=%VCToolsInstallDir%

   The host architecture should be ``x64`` and the target architecture should
   be ``arm64``.

2. Create and activate an environment using x64 Python 3.13. Install an x64
   PyTorch package so that ``setup.py`` can run, followed by the prerequisite
   packages listed above:

   .. code-block:: bat

      C:\path\to\x64-python\python.exe -m venv .venv-cross
      call .venv-cross\Scripts\activate.bat
      python -c "import platform; print(platform.machine())"

   The last command should print ``AMD64``.

3. Prepare the ARM64 target dependencies:

   * ``TORCHTRT_TARGET_TORCH_ROOT`` must contain ``include`` and ``lib``
     directories from an ARM64 PyTorch 2.14 installation. Download the ARM64
     wheel from ``https://pypi.nvidia.com/nvtorch_oot_nightly/torch/`` and
     install or extract it to create this target root.
   * ``TORCHTRT_TARGET_CUDA_ROOT`` must contain the CUDA headers and
     ``lib\arm64\cudart.lib``.
   * ``TORCHTRT_TARGET_PYTHON_ROOT`` must point to an ARM64 Python 3.13
     installation. It must contain the ARM64 Python headers and import library:
     ``include\Python.h`` and ``libs\python313.lib``. Do not use headers or
     libraries from the x64 host Python installation.

   Set the target roots and reuse the active cross-tools environment:

   .. code-block:: bat

      set "TORCHTRT_TARGET_TORCH_ROOT=C:/torchtrt-arm64-artifacts/pytorch/torch"
      set "TORCHTRT_TARGET_CUDA_ROOT=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/"
      set "TORCHTRT_TARGET_PYTHON_ROOT=C:\torchtrt-arm64-artifacts\python313-arm64"
      set "CUDA_PATH=%TORCHTRT_TARGET_CUDA_ROOT%"
      set "CUDA_HOME=%TORCHTRT_TARGET_CUDA_ROOT%"
      set DISTUTILS_USE_SDK=1

4. Cross-build the ARM64 TensorRT-RTX wheel:

   .. code-block:: bat
      python -m pip install --upgrade pip

      python -m pip install numpy packaging pyyaml setuptools==72.1.0 wheel fmt build

      # only for run the setup.py, not use this torch wheelto build windows on arm
      python -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130
      
      # cross-build the ARM64 TensorRT-RTX wheel
      python setup.py bdist_wheel --use-rtx --windows-on-arm

   The ``--windows-on-arm`` option explicitly selects ARM64 while the build is
   running under x64 Python. It also selects Bazel's x64-to-ARM64 MSVC
   toolchain and emits a ``win_arm64`` wheel.

5. Confirm that the generated wheel is tagged for ARM64:

   .. code-block:: bat

      dir dist\*win_arm64.whl

   An ARM64 wheel cannot be imported by the x64 host Python used for the
   cross-build. Copy the wheel to a Windows ARM64 machine, install it into a
   matching ARM64 Python environment, and run:

   .. code-block:: bat

      python -m pip install C:\path\to\the\generated-win_arm64.whl
      python -c "import torch_tensorrt; print(torch_tensorrt.__version__)"
