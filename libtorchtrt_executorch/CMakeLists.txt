cmake_minimum_required(VERSION 3.17)
project(libtorchtrt_executorch LANGUAGES CXX)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/Modules")

find_package(TensorRT REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED)
find_package(Threads REQUIRED)

set(_torchtrt_executorch_sources
    "${CMAKE_CURRENT_LIST_DIR}/core/runtime/executorch/TensorRTBackend.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/core/runtime/DeviceList.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/core/runtime/Platform.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/core/runtime/RTDevice.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/core/runtime/TRTEngine.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/core/runtime/TRTEngineProfiler.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/core/runtime/runtime.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/core/runtime/runtime_utils.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/core/util/Exception.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/core/util/trt_util.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/core/util/logging/TorchTRTLogger.cpp"
)

add_library(executorch_trt_backend STATIC ${_torchtrt_executorch_sources})
set_target_properties(executorch_trt_backend
    PROPERTIES
        OUTPUT_NAME "executorch_trt_backend"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

target_include_directories(executorch_trt_backend
    PUBLIC
        "${CMAKE_CURRENT_LIST_DIR}"
)

set(_torchtrt_executorch_link_libraries
    CUDA::cudart
    TensorRT::nvinfer
    torch
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
    message(FATAL_ERROR "Add ExecuTorch before libtorchtrt_executorch or set EXECUTORCH_ROOT to an ExecuTorch source tree")
endif()

if(NOT WIN32)
    list(APPEND _torchtrt_executorch_link_libraries stdc++fs)
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
