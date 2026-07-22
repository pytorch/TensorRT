cmake_minimum_required(VERSION 3.17)
project(torch_tensorrt_executorch LANGUAGES CXX)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

foreach(_torchtrt_cmake_module_dir IN ITEMS
    "${CMAKE_CURRENT_LIST_DIR}/cmake/Modules"
    "${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/Modules"
)
    if(EXISTS "${_torchtrt_cmake_module_dir}")
        list(APPEND CMAKE_MODULE_PATH "${_torchtrt_cmake_module_dir}")
    endif()
endforeach()

find_package(TensorRT REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

set(_torchtrt_executorch_sources
    "${CMAKE_CURRENT_LIST_DIR}/TensorRTBackend.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/TensorRTBlobHeader.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/TensorRTCallerStream.cpp"
)

add_library(executorch_trt_backend STATIC ${_torchtrt_executorch_sources})
set_target_properties(executorch_trt_backend
    PROPERTIES
        OUTPUT_NAME "executorch_trt_backend"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

target_include_directories(executorch_trt_backend
    PUBLIC
        "${CMAKE_CURRENT_LIST_DIR}/../../../include"
)

get_filename_component(_torchtrt_repo_root "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
if(EXISTS "${_torchtrt_repo_root}/core/runtime/TensorRTBindingNames.h")
    target_include_directories(executorch_trt_backend
        PUBLIC
            "${_torchtrt_repo_root}"
    )
endif()

target_compile_definitions(executorch_trt_backend
    PUBLIC
        C10_USING_CUSTOM_GENERATED_MACROS
)

set(_torchtrt_executorch_source_root "")
foreach(_executorch_root_candidate IN ITEMS "${EXECUTORCH_SOURCE_DIR}" "${EXECUTORCH_ROOT}")
    if(_executorch_root_candidate AND EXISTS "${_executorch_root_candidate}/runtime/core/portable_type/c10")
        set(_torchtrt_executorch_source_root "${_executorch_root_candidate}")
        get_filename_component(_executorch_include_parent "${_executorch_root_candidate}" DIRECTORY)
        target_include_directories(executorch_trt_backend
            PUBLIC
                "${_executorch_include_parent}"
                "${_executorch_root_candidate}/runtime/core/portable_type/c10"
        )
    endif()
endforeach()

set(_torchtrt_owns_extension_cuda FALSE)
if(NOT TARGET extension_cuda)
    if(NOT _torchtrt_executorch_source_root OR
       NOT EXISTS "${_torchtrt_executorch_source_root}/extension/cuda/caller_stream.cpp")
        message(FATAL_ERROR
            "Torch-TensorRT's ExecuTorch backend requires an ExecuTorch source "
            "checkout containing extension/cuda (ExecuTorch 1.4 or a commit "
            "including pytorch/executorch#20158 and #20498)."
        )
    endif()
    add_library(extension_cuda SHARED
        "${_torchtrt_executorch_source_root}/extension/cuda/caller_stream.cpp"
    )
    set(_torchtrt_owns_extension_cuda TRUE)
    target_compile_definitions(
        extension_cuda PRIVATE EXECUTORCH_EXTENSION_CUDA_BUILDING
    )
    target_include_directories(extension_cuda PUBLIC "${_executorch_include_parent}")
    target_link_libraries(extension_cuda PUBLIC CUDA::cudart)
endif()

get_target_property(_extension_cuda_type extension_cuda TYPE)
if(NOT _extension_cuda_type STREQUAL "SHARED_LIBRARY")
    message(FATAL_ERROR
        "ExecuTorch's extension_cuda target must be shared; got "
        "${_extension_cuda_type}. Static copies break caller-stream sharing."
    )
endif()

set(_torchtrt_executorch_link_libraries
    CUDA::cudart
    TensorRT::nvinfer
    Threads::Threads
    extension_cuda
)

if(TARGET executorch_core)
    list(APPEND _torchtrt_executorch_link_libraries executorch_core)
elseif(TARGET executorch)
    list(APPEND _torchtrt_executorch_link_libraries executorch)
elseif(DEFINED EXECUTORCH_ROOT AND EXISTS "${EXECUTORCH_ROOT}/runtime")
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
    message(FATAL_ERROR "Add ExecuTorch before torch_tensorrt_executorch or set EXECUTORCH_ROOT to an ExecuTorch source tree")
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
if(_torchtrt_owns_extension_cuda)
    install(
        TARGETS extension_cuda
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    )
endif()
