.. _getting_started_windows:

Building Torch-TensorRT on Windows
====================================

Torch-TensorRT has community support for Windows platform using CMake

Pre-requisite

* Microsoft Visual Studio
* LibTorch
* TensorRT
* CUDA
* cuDNN


.. Build configuration:

* Open Microsoft Visual Studio
* Open Torch-TensorRT source code folder
* Open Manage configurations -> Edit JSON to open CMakeSettings.json file.
* Configure the CMake build configurations. Following is an example configuration:


.. code-block:: JSON
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

.. Compilation:

* Build -> Build All OR Ctrl + Shift + B

Note: After successful compilation, the build artifacts will be present at buildRoot path configured.

.. Installation:

* Build -> Install Torch-TensorRT

Note: After successful installation, the artifacts will be present at installRoot.