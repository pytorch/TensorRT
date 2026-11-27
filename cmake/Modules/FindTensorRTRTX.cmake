# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

find_path(TensorRTRTX_INCLUDE_DIR NvInfer.h
  HINTS "${TensorRTRTX_ROOT}" ENV TensorRTRTX_ROOT
  PATH_SUFFIXES include)
if(TensorRTRTX_INCLUDE_DIR)
  foreach(component MAJOR MINOR PATCH)
    file(STRINGS "${TensorRTRTX_INCLUDE_DIR}/NvInferVersion.h" _rtx_version
      REGEX "^#define TRT_${component}_RTX [0-9]+")
    string(REGEX REPLACE ".*TRT_${component}_RTX ([0-9]+).*" "\\1"
      TensorRTRTX_VERSION_${component} "${_rtx_version}")
  endforeach()
  set(TensorRTRTX_VERSION "${TensorRTRTX_VERSION_MAJOR}.${TensorRTRTX_VERSION_MINOR}.${TensorRTRTX_VERSION_PATCH}")
endif()
set(_rtx_name "tensorrt_rtx_${TensorRTRTX_VERSION_MAJOR}_${TensorRTRTX_VERSION_MINOR}")
find_library(TensorRTRTX_LIBRARY NAMES tensorrt_rtx "${_rtx_name}"
  HINTS "${TensorRTRTX_ROOT}" ENV TensorRTRTX_ROOT
  PATH_SUFFIXES lib lib64)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRTRTX
  REQUIRED_VARS TensorRTRTX_INCLUDE_DIR TensorRTRTX_LIBRARY TensorRTRTX_VERSION_MAJOR
  VERSION_VAR TensorRTRTX_VERSION)
if(TensorRTRTX_FOUND AND NOT TARGET TensorRTRTX::runtime)
  add_library(TensorRTRTX::runtime SHARED IMPORTED)
  set_target_properties(TensorRTRTX::runtime PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${TensorRTRTX_INCLUDE_DIR}"
    IMPORTED_LOCATION "${TensorRTRTX_LIBRARY}")
  if(WIN32)
    find_file(TensorRTRTX_DLL "${_rtx_name}.dll"
      HINTS "${TensorRTRTX_ROOT}" ENV TensorRTRTX_ROOT PATH_SUFFIXES bin lib)
    if(NOT TensorRTRTX_DLL)
      message(FATAL_ERROR "TensorRT-RTX runtime DLL ${_rtx_name}.dll was not found")
    endif()
    set_target_properties(TensorRTRTX::runtime PROPERTIES
      IMPORTED_IMPLIB "${TensorRTRTX_LIBRARY}"
      IMPORTED_LOCATION "${TensorRTRTX_DLL}")
  endif()
endif()
