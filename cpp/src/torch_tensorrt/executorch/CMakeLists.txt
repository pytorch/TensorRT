cmake_minimum_required(VERSION 3.19)

# The version comes from the file the rest of the project uses, so the delegate's
# SONAME tracks the release rather than being maintained separately. Only the
# numeric prefix is kept, because a CMake project version cannot carry a
# pre-release suffix.
file(READ "${CMAKE_CURRENT_LIST_DIR}/../../../../version.txt"
     _torchtrt_executorch_version_raw
)
string(STRIP "${_torchtrt_executorch_version_raw}" _torchtrt_executorch_version_raw)
if(_torchtrt_executorch_version_raw MATCHES "^([0-9]+\\.[0-9]+\\.[0-9]+)")
    set(_torchtrt_executorch_version "${CMAKE_MATCH_1}")
else()
    set(_torchtrt_executorch_version "0.0.0")
endif()

project(
    torch_tensorrt_executorch
    VERSION "${_torchtrt_executorch_version}"
    LANGUAGES CXX
)

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

# Optionally build a loadable, coreless delegate plugin
# (libexecutorch_backend_tensorrt.so) that resolves the ExecuTorch runtime from the
# shared libexecutorch.so shipped by the ExecuTorch C++ SDK wheel. When enabled
# we require the SDK's executorch::runtime target, which then also satisfies the
# ExecuTorch dependency of the existing static target below.
option(TORCHTRT_EXECUTORCH_BUILD_SHARED_DELEGATE
       "Build libexecutorch_backend_tensorrt.so as a loadable coreless plugin" OFF
)
if(TORCHTRT_EXECUTORCH_BUILD_SHARED_DELEGATE)
    find_package(executorch CONFIG REQUIRED)
    if(NOT TARGET executorch::runtime)
        message(
            FATAL_ERROR
            "TORCHTRT_EXECUTORCH_BUILD_SHARED_DELEGATE requires the ExecuTorch "
            "C++ SDK to provide the executorch::runtime imported target."
        )
    endif()
endif()

set(_torchtrt_executorch_sources
    "${CMAKE_CURRENT_LIST_DIR}/TensorRTBackend.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/TensorRTBlobHeader.cpp"
)

add_library(executorch_backend_tensorrt STATIC ${_torchtrt_executorch_sources})
set_target_properties(executorch_backend_tensorrt
    PROPERTIES
        OUTPUT_NAME "executorch_backend_tensorrt"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

target_include_directories(executorch_backend_tensorrt
    PUBLIC
        "${CMAKE_CURRENT_LIST_DIR}/../../../include"
)

get_filename_component(_torchtrt_repo_root "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
if(EXISTS "${_torchtrt_repo_root}/core/runtime/TensorRTBindingNames.h")
    target_include_directories(executorch_backend_tensorrt
        PUBLIC
            "${_torchtrt_repo_root}"
    )
endif()

target_compile_definitions(executorch_backend_tensorrt
    PUBLIC
        C10_USING_CUSTOM_GENERATED_MACROS
)

foreach(_executorch_root_candidate IN ITEMS "${EXECUTORCH_SOURCE_DIR}" "${EXECUTORCH_ROOT}")
    if(_executorch_root_candidate AND EXISTS "${_executorch_root_candidate}/runtime/core/portable_type/c10")
        get_filename_component(_executorch_include_parent "${_executorch_root_candidate}" DIRECTORY)
        target_include_directories(executorch_backend_tensorrt
            PUBLIC
                "${_executorch_include_parent}"
                "${_executorch_root_candidate}/runtime/core/portable_type/c10"
        )
    endif()
endforeach()

set(_torchtrt_executorch_link_libraries
    CUDA::cudart
    TensorRT::nvinfer
    Threads::Threads
)

if(TARGET executorch_core)
    list(APPEND _torchtrt_executorch_link_libraries executorch_core)
elseif(TARGET executorch)
    list(APPEND _torchtrt_executorch_link_libraries executorch)
elseif(TARGET executorch::runtime)
    # An installed ExecuTorch package provides the runtime as an imported target,
    # so no source tree is needed. This is the case when building the shared
    # delegate against a wheel.
    list(APPEND _torchtrt_executorch_link_libraries executorch::runtime)
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

target_link_libraries(executorch_backend_tensorrt
    PUBLIC
        ${_torchtrt_executorch_link_libraries}
)

add_library(torchtrt_backend_tensorrt INTERFACE)
add_library(torchtrt::backend_tensorrt ALIAS torchtrt_backend_tensorrt)
add_dependencies(torchtrt_backend_tensorrt executorch_backend_tensorrt)

if(MSVC)
    target_link_libraries(torchtrt_backend_tensorrt
        INTERFACE
            executorch_backend_tensorrt
            ${_torchtrt_executorch_link_libraries}
    )
else()
    target_link_libraries(torchtrt_backend_tensorrt
        INTERFACE
            "-Wl,--whole-archive"
            "$<TARGET_FILE:executorch_backend_tensorrt>"
            "-Wl,--no-whole-archive"
            ${_torchtrt_executorch_link_libraries}
    )
endif()

install(
    TARGETS executorch_backend_tensorrt
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)

# ---------------------------------------------------------------------------
# Loadable, coreless delegate shared library.
#
# The static target above is meant to be whole-archived into a consumer's final
# link (which must also provide the ExecuTorch runtime). That model cannot be
# shipped as a drop-in plugin. This optional target instead builds
# libexecutorch_backend_tensorrt.so: a standalone shared library whose ExecuTorch
# runtime symbols (register_backend, ...) resolve at LOAD time from the shared
# libexecutorch.so shipped by the ExecuTorch C++ SDK wheel
# (executorch::runtime). It stays coreless (no embedded ExecuTorch core) and
# libtorch-free, so its "TensorRTBackend" registration coalesces into the same
# process-global registry as ExecuTorch's own CudaBackend. This mirrors how
# ExecuTorch ships its CUDA delegate as a loadable .so.
#
# Enable with -DTORCHTRT_EXECUTORCH_BUILD_SHARED_DELEGATE=ON and point
# CMAKE_PREFIX_PATH at the ExecuTorch SDK (python -c
# 'import executorch.utils as u; print(u.cmake_prefix_path)').
if(TORCHTRT_EXECUTORCH_BUILD_SHARED_DELEGATE)
    add_library(
        backend_tensorrt_shared SHARED ${_torchtrt_executorch_sources}
    )
    set_target_properties(
        backend_tensorrt_shared
        PROPERTIES OUTPUT_NAME "executorch_backend_tensorrt"
                   POSITION_INDEPENDENT_CODE ON
                   LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
                   # A versioned SONAME so the loader and packaging tools can tell
                   # an incompatible replacement from a compatible one. The major
                   # is this project's own, since that is what the release row
                   # names; the ExecuTorch version it was built against is pinned
                   # separately by the Python requirement.
                   #
                   # Only SOVERSION is set, without VERSION: that makes the
                   # installed file itself carry the SONAME name, so the wheel
                   # ships one library rather than a real file plus two symlinks
                   # that a wheel would store as further full copies.
                   SOVERSION "${PROJECT_VERSION_MAJOR}"
                   # A shipped library has to resolve its own dependencies. A
                   # consumer's DT_RPATH happens to cover them, but DT_RUNPATH,
                   # which is the default on many distributions, does not, so
                   # without this the delegate fails to load there even though it
                   # works where DT_RPATH is used.
                   # $ORIGIN only, with no INSTALL_RPATH_USE_LINK_PATH: that
                   # option appends the absolute link-time directories, which
                   # bakes the builder's filesystem layout into a shipped
                   # library and stops it being relocatable.
                   #
                   # Not DT_RUNPATH, matching the other libraries this project
                   # ships. A consumer that uses DT_RPATH has its search applied
                   # to this library's own dependencies as well, but a DT_RUNPATH
                   # on this library suppresses that inherited search, so the
                   # TensorRT path an application provides would stop working.
                   # The cost is that DT_RPATH is searched before
                   # LD_LIBRARY_PATH, so a user cannot redirect a co-located
                   # library with that variable.
                   INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib;$ORIGIN/../../executorch/lib"
                   INSTALL_RPATH_USE_LINK_PATH FALSE
                   # Carry the install paths at build time. Packaging copies this library out
                   # of the build tree rather than running an install step, so without this the
                   # build-tree search path ships: it names the builder's filesystem and, since
                   # it is the entry that happens to resolve the ExecuTorch runtime during a
                   # local build, its presence hides that the shipped paths do not.
                   BUILD_WITH_INSTALL_RPATH TRUE
                   LINK_FLAGS "-Wl,--disable-new-dtags"
    )
    target_include_directories(
        backend_tensorrt_shared
        PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../../../include"
    )
    if(EXISTS "${_torchtrt_repo_root}/core/runtime/TensorRTBindingNames.h")
        target_include_directories(
            backend_tensorrt_shared PRIVATE "${_torchtrt_repo_root}"
        )
    endif()
    target_compile_definitions(
        backend_tensorrt_shared PRIVATE C10_USING_CUSTOM_GENERATED_MACROS
    )
    # executorch::runtime (shared libexecutorch.so) resolves the ExecuTorch
    # symbols at load; TensorRT/CUDA/Threads are real deps. All PRIVATE so the
    # plugin does not re-export them to anything that links it.
    target_link_libraries(
        backend_tensorrt_shared
        PRIVATE executorch::runtime
                CUDA::cudart
                TensorRT::nvinfer
                Threads::Threads
    )
    install(
        TARGETS backend_tensorrt_shared
        EXPORT torchtrtExecuTorchTargets
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    )

    # ---------------------------------------------------------------------
    # Export a package so an application can consume the delegate from an
    # install tree.
    #
    # Without this the target exists only inside this build, so a C++ app has
    # no way to find the delegate except by adding this directory as a
    # subproject and building it from source. Only the shared delegate is
    # exported: the static target's interface names $<TARGET_FILE:...> to force
    # a whole-archive link, which is a build-tree path and cannot be installed.
    # ---------------------------------------------------------------------
    include(CMakePackageConfigHelpers)

    install(
        EXPORT torchtrtExecuTorchTargets
        FILE torchtrtExecuTorchTargets.cmake
        NAMESPACE torchtrt::
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/torch_tensorrt_executorch"
    )

    install(
        DIRECTORY
            "${CMAKE_CURRENT_LIST_DIR}/../../../include/torch_tensorrt/executorch"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/torch_tensorrt"
        FILES_MATCHING
        PATTERN "*.h"
    )

    # Recorded into the generated config so a consumer resolves a compatible runtime.
    set(TORCHTRT_EXECUTORCH_VERSION "${executorch_VERSION}")

    configure_package_config_file(
        "${_torchtrt_repo_root}/cmake/torch_tensorrt_executorchConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/torch_tensorrt_executorchConfig.cmake"
        INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/torch_tensorrt_executorch"
    )

    # A version file beside the config, so a consumer can pin a version. Without one
    # CMake rejects any versioned find_package because it cannot tell what version the
    # package is. SameMajorVersion matches the delegate's SONAME.
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/torch_tensorrt_executorchConfigVersion.cmake"
        VERSION "${PROJECT_VERSION}"
        COMPATIBILITY SameMajorVersion
    )

    install(
        FILES
            "${CMAKE_CURRENT_BINARY_DIR}/torch_tensorrt_executorchConfig.cmake"
            "${CMAKE_CURRENT_BINARY_DIR}/torch_tensorrt_executorchConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/torch_tensorrt_executorch"
    )
endif()
