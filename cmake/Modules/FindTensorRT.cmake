# This module defines the following variables:
#
# ::
#
#   TensorRT_INCLUDE_DIRS
#   TensorRT_LIBRARIES
#   TensorRT_FOUND
#
# ::
#
#   TensorRT_VERSION_STRING - version (x.y.z)
#   TensorRT_VERSION_MAJOR  - major version (x)
#   TensorRT_VERSION_MINOR  - minor version (y)
#   TensorRT_VERSION_PATCH  - patch version (z)
#
# Hints
# ^^^^^
# A user may set ``TensorRT_ROOT`` to an installation root to tell this module where to look.
#
set(_TensorRT_SEARCHES)

if(TensorRT_ROOT)
  set(_TensorRT_SEARCH_ROOT PATHS ${TensorRT_ROOT} NO_DEFAULT_PATH)
  list(APPEND _TensorRT_SEARCHES _TensorRT_SEARCH_ROOT)
endif()

# appends some common paths
set(_TensorRT_SEARCH_NORMAL
  PATHS "/usr"
)
list(APPEND _TensorRT_SEARCHES _TensorRT_SEARCH_NORMAL)

# Include dir
foreach(search ${_TensorRT_SEARCHES})
  find_path(TensorRT_INCLUDE_DIR NAMES NvInfer.h ${${search}} PATH_SUFFIXES include)
endforeach()

if(NOT TensorRT_LIBRARY)
  foreach(search ${_TensorRT_SEARCHES})
    find_library(TensorRT_LIBRARY NAMES nvinfer ${${search}} PATH_SUFFIXES lib)
  endforeach()
endif()

if(NOT TensorRT_nvinfer_plugin_LIBRARY)
  foreach(search ${_TensorRT_SEARCHES})
    find_library(TensorRT_nvinfer_plugin_LIBRARY NAMES nvinfer_plugin ${${search}} PATH_SUFFIXES lib)
  endforeach()
endif()

mark_as_advanced(TensorRT_INCLUDE_DIR)

if(TensorRT_INCLUDE_DIR AND EXISTS "${TensorRT_INCLUDE_DIR}/NvInfer.h")
  file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInfer.h" TensorRT_MAJOR REGEX "^#define NV_TENSORRT_MAJOR [0-9]+.*$")
  file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInfer.h" TensorRT_MINOR REGEX "^#define NV_TENSORRT_MINOR [0-9]+.*$")
  file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInfer.h" TensorRT_PATCH REGEX "^#define NV_TENSORRT_PATCH [0-9]+.*$")

  string(REGEX REPLACE "^#define NV_TENSORRT_MAJOR ([0-9]+).*$" "\\1" TensorRT_VERSION_MAJOR "${TensorRT_MAJOR}")
  string(REGEX REPLACE "^#define NV_TENSORRT_MINOR ([0-9]+).*$" "\\1" TensorRT_VERSION_MINOR "${TensorRT_MINOR}")
  string(REGEX REPLACE "^#define NV_TENSORRT_PATCH ([0-9]+).*$" "\\1" TensorRT_VERSION_PATCH "${TensorRT_PATCH}")
  set(TensorRT_VERSION_STRING "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}")
endif()

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(TensorRT REQUIRED_VARS TensorRT_LIBRARY TensorRT_INCLUDE_DIR VERSION_VAR TensorRT_VERSION_STRING)

if(TensorRT_FOUND)
  set(TensorRT_INCLUDE_DIRS ${TensorRT_INCLUDE_DIR})

  if(NOT TensorRT_LIBRARIES)
    set(TensorRT_LIBRARIES ${TensorRT_LIBRARY})
    if (TensorRT_nvinfer_plugin_LIBRARY)
      list(APPEND TensorRT_LIBRARIES ${TensorRT_nvinfer_plugin_LIBRARY})
    endif()
  endif()

  if(NOT TARGET TensorRT::TensorRT)
    add_library(TensorRT INTERFACE IMPORTED)
    add_library(TensorRT::TensorRT ALIAS TensorRT)
  endif()

  if(NOT TARGET TensorRT::nvinfer)
    add_library(TensorRT::nvinfer SHARED IMPORTED)
    if (WIN32)
      foreach(search ${_TensorRT_SEARCHES})
        find_file(TensorRT_LIBRARY_DLL
          NAMES nvinfer.dll
          PATHS ${${search}}
          PATH_SUFFIXES bin
        )
      endforeach()

      set_target_properties(TensorRT::nvinfer PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIRS}"
        IMPORTED_LOCATION "${TensorRT_LIBRARY_DLL}"
        IMPORTED_IMPLIB "${TensorRT_LIBRARY}"
      )
    else()
      set_target_properties(TensorRT::nvinfer PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIRS}"
        IMPORTED_LOCATION "${TensorRT_LIBRARY}"
      )
    endif()
    target_link_libraries(TensorRT INTERFACE TensorRT::nvinfer)
  endif()

  if(NOT TARGET TensorRT::nvinfer_plugin AND TensorRT_nvinfer_plugin_LIBRARY)
    add_library(TensorRT::nvinfer_plugin SHARED IMPORTED)
    if (WIN32)
      foreach(search ${_TensorRT_SEARCHES})
        find_file(TensorRT_nvinfer_plugin_LIBRARY_DLL
          NAMES nvinfer_plugin.dll
          PATHS ${${search}}
          PATH_SUFFIXES bin
        )
      endforeach()

      set_target_properties(TensorRT::nvinfer_plugin PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIRS}"
        IMPORTED_LOCATION "${TensorRT_nvinfer_plugin_LIBRARY_DLL}"
        IMPORTED_IMPLIB "${TensorRT_nvinfer_plugin_LIBRARY}"
      )
    else()
      set_target_properties(TensorRT::nvinfer_plugin PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIRS}"
        IMPORTED_LOCATION "${TensorRT_nvinfer_plugin_LIBRARY}"
      )
    endif()
    target_link_libraries(TensorRT INTERFACE TensorRT::nvinfer_plugin)
  endif()
endif()
