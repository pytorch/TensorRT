cmake_minimum_required(VERSION 3.17)
project(libtorchtrt_executorch LANGUAGES CXX)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Locate the in-tree TensorRTBackend.cpp source.  This file lives under
# core/runtime/executorch/ in the source tree and is the only source we
# need to build for the libtorch-free ExecuTorch backend.
get_filename_component(_torchtrt_source_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(_torchtrt_executorch_backend_source
    "${_torchtrt_source_root}/core/runtime/executorch/TensorRTBackend.cpp")
set(_torchtrt_executorch_backend_include_dir
    "${_torchtrt_source_root}/core/runtime/executorch")

list(APPEND CMAKE_MODULE_PATH "${_torchtrt_source_root}/cmake/Modules")

find_package(TensorRT REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

add_library(executorch_trt_backend STATIC "${_torchtrt_executorch_backend_source}")
set_target_properties(executorch_trt_backend
    PROPERTIES
        OUTPUT_NAME "executorch_trt_backend"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

target_include_directories(executorch_trt_backend
    PUBLIC
        "${_torchtrt_source_root}"
    PRIVATE
        "${_torchtrt_executorch_backend_include_dir}"
)

set(_torchtrt_executorch_link_libraries
    CUDA::cudart
    TensorRT::nvinfer
    Threads::Threads
)

if(TARGET executorch_core)
    list(APPEND _torchtrt_executorch_link_libraries executorch_core)
elseif(TARGET executorch)
    list(APPEND _torchtrt_executorch_link_libraries executorch)
elseif(DEFINED EXECUTORCH_ROOT AND EXISTS "${EXECUTORCH_ROOT}/runtime")
    get_filename_component(_executorch_include_parent "${EXECUTORCH_ROOT}" DIRECTORY)
    target_include_directories(executorch_trt_backend
        PUBLIC
            "${_executorch_include_parent}"
    )
    if(NOT DEFINED EXECUTORCH_CORE_LIBRARY)
        set(
            EXECUTORCH_CORE_LIBRARY
            "${EXECUTORCH_ROOT}/cmake-out/libexecutorch_core.a"
            CACHE FILEPATH
            "Path to the ExecuTorch core static library"
        )
    endif()
    if(NOT EXISTS "${EXECUTORCH_CORE_LIBRARY}")
        message(FATAL_ERROR "EXECUTORCH_CORE_LIBRARY='${EXECUTORCH_CORE_LIBRARY}' does not exist")
    endif()
    list(APPEND _torchtrt_executorch_link_libraries "${EXECUTORCH_CORE_LIBRARY}")
else()
    message(FATAL_ERROR "Add ExecuTorch before the Torch-TensorRT ExecuTorch backend or set EXECUTORCH_ROOT to an ExecuTorch source tree")
endif()

target_link_libraries(executorch_trt_backend
    PUBLIC
        ${_torchtrt_executorch_link_libraries}
)

add_library(torchtrt_executorch_backend INTERFACE)
add_library(torchtrt::executorch_backend ALIAS torchtrt_executorch_backend)
add_dependencies(torchtrt_executorch_backend executorch_trt_backend)

if(MSVC)
    target_link_libraries(torchtrt_executorch_backend
        INTERFACE
            executorch_trt_backend
            ${_torchtrt_executorch_link_libraries}
    )
else()
    target_link_libraries(torchtrt_executorch_backend
        INTERFACE
            "-Wl,--whole-archive"
            "$<TARGET_FILE:executorch_trt_backend>"
            "-Wl,--no-whole-archive"
            ${_torchtrt_executorch_link_libraries}
    )
endif()

install(
    TARGETS executorch_trt_backend
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)
