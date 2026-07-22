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

# Select ExecuTorch's shared extension_cuda so every CUDA-capable delegate shares
# one caller-stream thread-local. Exactly one shared library is allowed in the
# process; each branch rejects a static copy, which would duplicate the TLS and
# silently break the caller-stream handshake. Precedence: an ExecuTorch-provided
# target, then an explicit prebuilt shared library, then a source build.
set(_torchtrt_owns_extension_cuda FALSE)
if(TARGET extension_cuda)
    # Provided by ExecuTorch, e.g. add_subdirectory() with EXECUTORCH_BUILD_CUDA=ON.
    get_target_property(_extension_cuda_type extension_cuda TYPE)
    if(NOT _extension_cuda_type STREQUAL "SHARED_LIBRARY")
        message(FATAL_ERROR
            "ExecuTorch's extension_cuda target must be shared; got "
            "${_extension_cuda_type}. Static copies break caller-stream sharing."
        )
    endif()
elseif(DEFINED EXECUTORCH_EXTENSION_CUDA_LIBRARY)
    # Use a prebuilt shared library, e.g. the one shipped in libtorchtrt.tar.gz.
    # A caller-supplied override is authoritative: never silently fall back to a
    # different source-built library when its path is invalid.
    if(NOT EXISTS "${EXECUTORCH_EXTENSION_CUDA_LIBRARY}")
        message(FATAL_ERROR
            "EXECUTORCH_EXTENSION_CUDA_LIBRARY does not exist: "
            "${EXECUTORCH_EXTENSION_CUDA_LIBRARY}"
        )
    elseif(WIN32)
        # A .lib suffix alone cannot distinguish an import library from a static
        # archive, so accepting this override could duplicate caller-stream TLS.
        message(FATAL_ERROR
            "Prebuilt EXECUTORCH_EXTENSION_CUDA_LIBRARY overrides are not supported "
            "on Windows. Provide ExecuTorch's shared extension_cuda target or build "
            "the shared library from an ExecuTorch source checkout."
        )
    elseif(NOT EXECUTORCH_EXTENSION_CUDA_LIBRARY MATCHES "\\.(so(\\.[0-9]+)*|dylib)$")
        message(FATAL_ERROR
            "EXECUTORCH_EXTENSION_CUDA_LIBRARY must name a shared .so or .dylib; "
            "static archives would duplicate caller-stream TLS."
        )
    else()
        add_library(extension_cuda SHARED IMPORTED)
        set_target_properties(extension_cuda PROPERTIES
            IMPORTED_LOCATION "${EXECUTORCH_EXTENSION_CUDA_LIBRARY}"
        )
    endif()
elseif(_torchtrt_executorch_source_root AND
       EXISTS "${_torchtrt_executorch_source_root}/extension/cuda/caller_stream.cpp")
    # Build the single shared library from an ExecuTorch source checkout.
    add_library(extension_cuda SHARED
        "${_torchtrt_executorch_source_root}/extension/cuda/caller_stream.cpp"
    )
    set(_torchtrt_owns_extension_cuda TRUE)
    target_compile_definitions(extension_cuda
        PRIVATE
            C10_USING_CUSTOM_GENERATED_MACROS
            EXECUTORCH_EXTENSION_CUDA_BUILDING
    )
    target_include_directories(extension_cuda PUBLIC "${_executorch_include_parent}")
    target_link_libraries(extension_cuda PUBLIC CUDA::cudart)
else()
    message(FATAL_ERROR
        "Torch-TensorRT's ExecuTorch backend requires ExecuTorch's shared "
        "extension_cuda library. Add ExecuTorch first with EXECUTORCH_BUILD_CUDA=ON, "
        "or set EXECUTORCH_EXTENSION_CUDA_LIBRARY to a prebuilt libextension_cuda "
        "on Linux/macOS, or set EXECUTORCH_SOURCE_DIR/EXECUTORCH_ROOT to an ExecuTorch "
        "checkout containing extension/cuda/caller_stream.cpp. Do not link a static "
        "copy: all delegates must share one caller-stream TLS."
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
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    )
endif()
