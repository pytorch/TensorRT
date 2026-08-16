.. _getting_started_windows:

Building Torch-TensorRT on Windows
====================================

Torch-TensorRT has community support for Windows platform using CMake

Prerequisite:

* Microsoft VS 2022 Tools
* Bazelisk
* CUDA


Build steps
-------------------

* Open the app "x64 Native Tools Command Prompt for VS 2022" - note that Admin priveleges may be necessary
* Ensure Bazelisk (Bazel launcher) is installed on your machine and available from the command line. Package installers such as Chocolatey can be used to install Bazelisk
* Install latest version of Torch (i.e. with `pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124`)
* Clone the Torch-TensorRT repository and navigate to its root directory
* Run `pip install ninja wheel setuptools`
* Run `pip install --pre -r py/requirements.txt`
* Run `set DISTUTILS_USE_SDK=1`
* Run `python setup.py bdist_wheel`
* Run `pip install dist/*.whl`

Advanced setup and Troubleshooting
-------------------
In the `WORKSPACE` file, the `cuda_win`, `libtorch_win`, and `tensorrt_win` are Windows-specific modules which can be customized. For instance, if you would like to build with a different version of CUDA, or your CUDA installation is in a non-standard location, update the `path` in the `cuda_win` module.

Similarly, if you would like to use a different version of pytorch or tensorrt, customize the `urls` in the `libtorch_win` and `tensorrt_win` modules, respectively.

Local versions of these packages can also be used on Windows. See `toolchains\ci_workspaces\WORKSPACE.win.release.tmpl` for an example of using a local version of TensorRT on Windows.
