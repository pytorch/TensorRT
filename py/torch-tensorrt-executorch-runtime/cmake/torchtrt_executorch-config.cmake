# CMake package for the Torch-TensorRT ExecuTorch delegate, as installed by the
# torch-tensorrt-executorch-runtime wheel.
#
# ExecuTorch ships its own backends as prebuilt shared libraries plus a CMake
# package, so a C++ app links executorch::backend_cuda and gets the backend
# registered. This file gives the TensorRT delegate the same treatment, so a C++
# app can link it out of the installed wheel instead of building this repo from
# source:
#
#   find_package(executorch REQUIRED COMPONENTS backend_cuda)
#   find_package(torchtrt_executorch REQUIRED)
#   target_link_libraries(my_app PRIVATE
#     executorch::runtime executorch::backend_cuda torchtrt::executorch_backend)
#
# There is nothing to include. The delegate exposes no public header: it
# registers itself with ExecuTorch's backend registry from a static initializer
# inside the shared library, and everything a caller does afterwards is
# ExecuTorch's own Runtime API.
#
# Point CMake at it with either of:
#   -Dtorchtrt_executorch_DIR=$(python -c "import torch_tensorrt_executorch_runtime as m, pathlib; print(pathlib.Path(m.__file__).parent / 'lib/cmake/torchtrt_executorch')")
#   -DCMAKE_PREFIX_PATH=$(python -c "import torch_tensorrt_executorch_runtime as m, pathlib; print(pathlib.Path(m.__file__).parent)")

# 3.28, not 3.24. This file alone would configure on 3.24, but its whole purpose is to be used
# alongside find_package(executorch COMPONENTS backend_cuda), and that package rejects anything
# below 3.28 because older versions write the $ORIGIN token in a runtime search path incorrectly.
# Advertising a floor the documented usage cannot meet just moves the failure later.
cmake_minimum_required(VERSION 3.28)

include(FindPackageHandleStandardArgs)

# The package root is found by walking up from this file until the delegate library turns up under
# lib/, rather than by counting "../.." a fixed number of times. This file installs to
# lib/cmake/torchtrt_executorch, the layout find_package searches under a prefix and the same one
# ExecuTorch uses for its own package, so the walk passes through a lib/ directory on the way out.
# Testing for the library rather than for a directory named lib is what keeps it from stopping there.
set(_torchtrt_executorch_root "${CMAKE_CURRENT_LIST_DIR}")
unset(TORCHTRT_EXECUTORCH_BACKEND_LIBRARY)
foreach(_ RANGE 4)
  if(EXISTS "${_torchtrt_executorch_root}/lib/libexecutorch_backend_tensorrt.so")
    set(TORCHTRT_EXECUTORCH_BACKEND_LIBRARY
      "${_torchtrt_executorch_root}/lib/libexecutorch_backend_tensorrt.so")
    break()
  endif()
  get_filename_component(_torchtrt_executorch_root "${_torchtrt_executorch_root}" DIRECTORY)
endforeach()

find_package_handle_standard_args(
  torchtrt_executorch
  REQUIRED_VARS TORCHTRT_EXECUTORCH_BACKEND_LIBRARY
)

if(NOT torchtrt_executorch_FOUND)
  return()
endif()

set(TORCHTRT_EXECUTORCH_LIBRARIES torchtrt::executorch_backend)

if(TARGET torchtrt::executorch_backend)
  # Another subproject already called find_package in this configure. Redefining
  # an imported target is an error, so keep the one that is there.
  message(STATUS "torchtrt_executorch: torchtrt::executorch_backend is already defined, reusing it")
  return()
endif()

add_library(torchtrt::executorch_backend SHARED IMPORTED)
set_target_properties(
  torchtrt::executorch_backend
  PROPERTIES
    IMPORTED_LOCATION "${TORCHTRT_EXECUTORCH_BACKEND_LIBRARY}"
    INTERFACE_COMPILE_FEATURES cxx_std_17
)

# Retain the static registration even when the consumer references no delegate symbol.
# Scope --no-as-needed to this library so unrelated dependencies can still be dropped.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set_property(
    TARGET torchtrt::executorch_backend
    APPEND
    PROPERTY
      INTERFACE_LINK_OPTIONS
      "LINKER:--push-state,--no-as-needed,${TORCHTRT_EXECUTORCH_BACKEND_LIBRARY},--pop-state"
  )
  # So the loader finds the library in the installed wheel at run time. The delegate is not a
  # dependency the consumer copies around: it lives in site-packages next to the executorch wheel
  # whose runtime it links, and both have to be found from the same place.
  set_property(
    TARGET torchtrt::executorch_backend
    APPEND
    PROPERTY INTERFACE_LINK_OPTIONS "LINKER:--enable-new-dtags,-rpath,${_torchtrt_executorch_root}/lib"
  )
endif()
