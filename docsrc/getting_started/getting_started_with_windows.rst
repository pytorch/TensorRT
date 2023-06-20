.. _getting_started_windows:

Building Torch-TensorRT on Windows
====================================

Torch-TensorRT has community support for Windows platform using CMake

Prerequisite:

* Microsoft Visual Studio
* LibTorch
* TensorRT
* CUDA
* cuDNN


Build configuration
-------------------

* Open Microsoft Visual Studio
* Open Torch-TensorRT source code folder
* Open Manage configurations -> Edit JSON to open CMakeSettings.json file.
* Configure the CMake build configurations. Following is an example configuration:

.. code-block:: none

    {
      "configurations": [
        {
          "name": "x64-Debug",
          "generator": "Ninja",
          "configurationType": "Debug",
          "inheritEnvironments": [ "msvc_x64_x64" ],
          "buildRoot": "${projectDir}\\out\\build\\${name}",
          "installRoot": "${projectDir}\\out\\install\\${name}",
          "cmakeCommandArgs": "-S . -B out",
          "buildCommandArgs": "cmake --build out",
          "ctestCommandArgs": "",
          "variables": [
            {
              "name": "CMAKE_MODULE_PATH",
              "value": "$PWD\cmake\Modules",
              "type": "FILEPATH"
            },
            {
              "name": "Torch_DIR",
              "value": "<Path to libtorch>\share\cmake\Torch",
              "type": "FILEPATH"
            },
            {
              "name": "TensorRT_ROOT",
              "value": "<Path to TensorRT directory>",
              "type": "FILEPATH"
            },
            {
              "name": "CMAKE_BUILD_TYPE",
              "value": "Release",
              "type": " STRING"
            }
          ]
        }
      ]
    }


Compilation
-----------

* Click Build -> Build All or directly press Ctrl + Shift + B

Note: After successful compilation, the build artifacts will be present at buildRoot path configured.

Installation
------------

* Build -> Install Torch-TensorRT

Note: After successful installation, the artifacts will be present at installRoot.


Building With Visual Studio Code
==================================

1. Install Visual Studio Code
2. Install Build Tools for Visual Studio 2022

    - Select "Desktop Development with C++"
      > Currently, this installs MSVC v143 - 2022. There are also options to install previous 2019/2017/2015 editions of MSVC
      > License term "1b Build Tools additional use right" allows using Build Tools to compile Open Source Dependencies
      > Also allows using Build Tools to develop and test Open Source Dependencies, to the minor extend of ensuring compatibility with Build Tools

3. Install CUDA (e.g. 11.7.1)
4. Install cuDNN (e.g. 8.5.0.96)

    - Set ``cuDNN_ROOT_DIR``

5. Install `TensorRT` (e.g 8.5.1.7)

    - Set ``TensorRT_ROOT``
    - Add ``TensorRT_ROOT\lib`` to ``PATH``

6. Install "libtorch-win-shared-with-deps-latest.zip"

    - Select build targeting the appropriate CUDA version
    - Set ``Torch_DIR``
    - Add ``Torch_DIR\lib`` to ``PATH``

7. Clone TensorRT repo
8. Install C++ and CMake Tools extensions from MS

    - Change build to ``RelWithDebInfo``

9. Update ``.vscode\settings.json``

    - Clean, configure, build

e.g. /.vscode/settings.json

.. code-block:: json
    
    {
        "cmake.generator": "Ninja",
        "cmake.configureSettings": {
            "CMAKE_MODULE_PATH": {
                "type": "FILEPATH",
                "value": "$PWD\\cmake\\Modules"
            },
            "CMAKE_CXX_FLAGS": {
                "type": "STRING",
                "value": "-D_SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING"
            },
            "Torch_DIR": {
                "type": "FILEPATH",
                "value": "X:\\libtorch\\share\\cmake\\Torch"
            },
            "TensorRT_ROOT": {
                "type": "FILEPATH",
                "value": "X:\\path\\to\\tensorrt"
            },
            "cuDNN_ROOT_DIR": {
                "type": "FILEPATH",
                "value": "X:\\path\\to\\cudnn"
            },
            "CMAKE_CUDA_FLAGS": "-allow-unsupported-compiler"
        },
        "cmake.buildDirectory": "${workspaceFolder}/torch_tensorrt_build"
    }
