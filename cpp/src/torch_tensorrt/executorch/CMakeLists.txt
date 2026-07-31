cmake_minimum_required(VERSION 3.19)
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
# one caller-stream thread-local. A static copy linked into a second shared object
# would create a second thread-local and silently break the handshake, so every
# branch below must resolve to one shared library. Precedence: an
# ExecuTorch-provided target, then an explicit prebuilt shared library, then a
# source build.
if(TARGET extension_cuda)
    # Provided by ExecuTorch, e.g. add_subdirectory() with EXECUTORCH_BUILD_CUDA=ON.
    get_target_property(_extension_cuda_type extension_cuda TYPE)
    if(NOT _extension_cuda_type STREQUAL "SHARED_LIBRARY")
        message(FATAL_ERROR
            "ExecuTorch's extension_cuda target must be shared; got "
            "${_extension_cuda_type}. Static copies break caller-stream sharing."
        )
    endif()
elseif(EXECUTORCH_EXTENSION_CUDA_LIBRARY)
    # Use a prebuilt shared library, e.g. the one shipped in libtorchtrt.tar.gz.
    # A caller-supplied override is authoritative: never silently fall back to a
    # different source-built library when its path is invalid.
    if(NOT EXISTS "${EXECUTORCH_EXTENSION_CUDA_LIBRARY}")
        message(FATAL_ERROR
            "EXECUTORCH_EXTENSION_CUDA_LIBRARY does not exist: "
            "${EXECUTORCH_EXTENSION_CUDA_LIBRARY}"
        )
    endif()
    # Read the ELF header rather than trusting the file name: the linker accepts
    # a static archive or a linker script named ".so", and either one embeds a
    # private copy of the caller-stream thread-local. e_ident is always 16 bytes,
    # so e_type sits at byte 16 for both ELF32 and ELF64; EI_DATA at byte 5 says
    # whether to read it little- or big-endian.
    file(READ "${EXECUTORCH_EXTENSION_CUDA_LIBRARY}" _extension_cuda_header LIMIT 18 HEX)
    if(_extension_cuda_header MATCHES "^7f454c46..02")
        set(_extension_cuda_et_dyn "0003")
    else()
        set(_extension_cuda_et_dyn "0300")
    endif()
    if(NOT _extension_cuda_header MATCHES "^7f454c46" OR
       NOT _extension_cuda_header MATCHES "${_extension_cuda_et_dyn}$")
        message(FATAL_ERROR
            "EXECUTORCH_EXTENSION_CUDA_LIBRARY is not an ELF shared object: "
            "${EXECUTORCH_EXTENSION_CUDA_LIBRARY}. Static archives, linker scripts "
            "and other platform formats are rejected because each would duplicate "
            "the caller-stream thread-local."
        )
    endif()
    add_library(extension_cuda SHARED IMPORTED)
    set_target_properties(extension_cuda PROPERTIES
        IMPORTED_LOCATION "${EXECUTORCH_EXTENSION_CUDA_LIBRARY}"
    )
elseif(_torchtrt_executorch_source_root AND
       EXISTS "${_torchtrt_executorch_source_root}/extension/cuda/CMakeLists.txt")
    # Build the single shared library from ExecuTorch's own definition rather
    # than a copy of it, so this stays correct if ExecuTorch changes the target
    # and so a duplicate definition is a hard error instead of two libraries.
    if(NOT _common_include_directories)
        set(_common_include_directories "${_executorch_include_parent}")
    endif()
    add_subdirectory(
        "${_torchtrt_executorch_source_root}/extension/cuda"
        "${CMAKE_CURRENT_BINARY_DIR}/executorch_extension_cuda"
    )
else()
    message(FATAL_ERROR
        "Torch-TensorRT's ExecuTorch backend requires ExecuTorch's shared "
        "extension_cuda library. Add ExecuTorch first with EXECUTORCH_BUILD_CUDA=ON, "
        "or set EXECUTORCH_EXTENSION_CUDA_LIBRARY to a prebuilt shared "
        "libextension_cuda, or set EXECUTORCH_SOURCE_DIR/EXECUTORCH_ROOT to an "
        "ExecuTorch checkout containing extension/cuda. Do not link a static copy: "
        "all delegates must share one caller-stream thread-local."
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
