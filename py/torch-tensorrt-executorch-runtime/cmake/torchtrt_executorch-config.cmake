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
#   target_link_libraries(my_app PRIVATE executorch::runtime torchtrt::executorch_backend)
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
foreach(_ RANGE 4)
  if(EXISTS "${_torchtrt_executorch_root}/lib/libexecutorch_backend_tensorrt.so")
    break()
  endif()
  get_filename_component(_torchtrt_executorch_root "${_torchtrt_executorch_root}" DIRECTORY)
endforeach()

# Globbed rather than passed to find_library with a bare name, so a versioned soname
# (libfoo.so.1) is found too, the way ExecuTorch's own package config does it.
file(
  GLOB _torchtrt_executorch_matches
  "${_torchtrt_executorch_root}/lib/libexecutorch_backend_tensorrt.so"
  "${_torchtrt_executorch_root}/lib/libexecutorch_backend_tensorrt.so.*"
)
if(_torchtrt_executorch_matches)
  # Highest version first, so a prefix carrying two does not select by glob order.
  list(SORT _torchtrt_executorch_matches)
  list(REVERSE _torchtrt_executorch_matches)
  list(GET _torchtrt_executorch_matches 0 TORCHTRT_EXECUTORCH_BACKEND_LIBRARY)
endif()

find_package_handle_standard_args(
  torchtrt_executorch
  REQUIRED_VARS TORCHTRT_EXECUTORCH_BACKEND_LIBRARY
)

if(NOT torchtrt_executorch_FOUND)
  return()
endif()

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

# The delegate registers itself from a static initializer, so the dependency has to survive the link
# even if the consumer never names a symbol from it. --as-needed keeps a library only when something
# in the link references it, and today the delegate happens to export enough that it is kept anyway;
# that is incidental, not a guarantee, and it would stop being true the moment the exported surface
# shrinks. --no-as-needed makes the outcome independent of that. Bracketed with push-state and
# pop-state so the flag applies only to this library and does not change how the rest of the
# consumer's link line is treated.
#
# This is the shared-library counterpart of the --whole-archive that the in-repo source build needs,
# where the equivalent risk is real today: a static archive member nothing references IS dropped.
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
    PROPERTY INTERFACE_LINK_OPTIONS "LINKER:-rpath,${_torchtrt_executorch_root}/lib"
  )
endif()

set(TORCHTRT_EXECUTORCH_LIBRARIES torchtrt::executorch_backend)
