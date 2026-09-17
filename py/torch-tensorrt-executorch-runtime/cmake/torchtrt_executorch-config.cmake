# CMake package for the Torch-TensorRT ExecuTorch delegate, as installed by the
# torch-tensorrt-executorch-runtime wheel.
#
# ExecuTorch ships its own backends as prebuilt shared libraries plus a CMake
# package, so a C++ app links executorch::backend_cuda and gets the backend
# registered. This file gives the TensorRT delegate the same treatment, so a C++
# app can link it out of the installed wheel instead of building this repo from
# source:
#
#   find_package(executorch REQUIRED COMPONENTS backend_cuda kernels_optimized)
#   find_package(torchtrt_executorch REQUIRED)
#   target_link_libraries(my_app PRIVATE
#     executorch::runtime executorch::backend_cuda
#     executorch::kernels_optimized torchtrt::executorch_backend)
#
# kernels_optimized supplies the et_copy host/device copy operators.
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
  # Reusing is right when another subproject already ran this same config, and wrong when the name
  # belongs to something else. The in-tree build defines it as an interface library over a private
  # static copy, so a project that pulls that in and then calls find_package would silently link
  # the private copy and never touch the shared library in the wheel. Only an imported shared
  # library can be the one this config created.
  get_target_property(_torchtrt_executorch_existing_type torchtrt::executorch_backend TYPE)
  if(NOT _torchtrt_executorch_existing_type STREQUAL "SHARED_LIBRARY")
    message(FATAL_ERROR
      "torchtrt::executorch_backend already exists as a ${_torchtrt_executorch_existing_type}, "
      "not as the imported shared library this package provides. The in-tree delegate target and "
      "the installed one cannot both be used in a single configure: drop one of them.")
  endif()
  # Type alone does not identify it. A shared imported target of the same name pointing at another
  # file would pass, and the consumer would link that file while believing it linked this one.
  get_target_property(_torchtrt_executorch_existing_location
    torchtrt::executorch_backend IMPORTED_LOCATION)
  if(NOT _torchtrt_executorch_existing_location STREQUAL "${TORCHTRT_EXECUTORCH_BACKEND_LIBRARY}")
    message(FATAL_ERROR
      "torchtrt::executorch_backend already points at "
      "${_torchtrt_executorch_existing_location}, not at the "
      "${TORCHTRT_EXECUTORCH_BACKEND_LIBRARY} this package found. Two different delegates cannot "
      "both be used in a single configure: drop one of them.")
  endif()
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
  # This is the path on the machine that configured, and it lands in every consumer binary, so a
  # binary built here does not run anywhere else. That is the right default for building against an
  # installed wheel, which is what this package is for, and wrong for anything redistributable.
  # Not option(): inside a package config that creates a cache entry in the consumer's project
  # and, depending on the policy in force, overrides a plain variable the consumer already
  # set, so the documented opt-out could be ignored. Honour what the consumer set.
  if(NOT DEFINED TORCHTRT_EXECUTORCH_EMBED_RUNPATH)
    set(TORCHTRT_EXECUTORCH_EMBED_RUNPATH ON)
  endif()
  if(TORCHTRT_EXECUTORCH_EMBED_RUNPATH)
    set_property(
      TARGET torchtrt::executorch_backend
      APPEND
      PROPERTY INTERFACE_LINK_OPTIONS "LINKER:--enable-new-dtags,-rpath,${_torchtrt_executorch_root}/lib"
    )
  else()
    # This removes the run path this package adds, not the one CMake adds by itself:
    # linking an imported library records its directory as DT_RUNPATH regardless, which is
    # still this machine absolute path. Only CMAKE_SKIP_BUILD_RPATH in the consumer own
    # project removes that, and a package config has no business setting it there.
    message(STATUS
      "torchtrt_executorch: not adding a run path. CMake still records "
      "${_torchtrt_executorch_root}/lib as a build run path when linking an imported "
      "library, so a redistributable binary also needs CMAKE_SKIP_BUILD_RPATH, and then has "
      "to locate the delegate itself at run time.")
  endif()
endif()
