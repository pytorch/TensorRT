# TensorRT
find_package(TensorRT QUIET)
if (NOT TensorRT_FOUND)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")
    find_package(TensorRT REQUIRED)
endif()

# If the custom finders are needed at this point, there are good chances that they will be needed when consuming the library as well
install(FILES "${CMAKE_SOURCE_DIR}/cmake/Modules/FindTensorRT.cmake" DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/torchtrt/Modules")

# CUDA
find_package(CUDAToolkit REQUIRED)

# libtorch
find_package(Torch REQUIRED)
find_package(Threads REQUIRED)

add_definitions(-DTORCH_VERSION_MAJOR=${Torch_VERSION_MAJOR})
add_definitions(-DTORCH_VERSION_MINOR=${Torch_VERSION_MINOR})
add_definitions(-DTORCH_VERSION_PATCH=${Torch_VERSION_PATCH})

if(BUILD_TORCHTRT_EXECUTORCH)
    if(NOT DEFINED EXECUTORCH_ROOT OR EXECUTORCH_ROOT STREQUAL "")
        message(FATAL_ERROR "BUILD_TORCHTRT_EXECUTORCH requires EXECUTORCH_ROOT to point to an ExecuTorch source tree")
    endif()

    if(NOT EXISTS "${EXECUTORCH_ROOT}/runtime")
        message(FATAL_ERROR "EXECUTORCH_ROOT='${EXECUTORCH_ROOT}' is missing runtime/")
    endif()

    if(NOT DEFINED EXECUTORCH_CORE_LIBRARY)
        set(
            EXECUTORCH_CORE_LIBRARY
            "${EXECUTORCH_ROOT}/cmake-out/libexecutorch_core.a"
            CACHE FILEPATH "Path to the ExecuTorch static runtime library"
        )
    endif()

    if(NOT EXISTS "${EXECUTORCH_CORE_LIBRARY}")
        message(FATAL_ERROR "EXECUTORCH_CORE_LIBRARY='${EXECUTORCH_CORE_LIBRARY}' does not exist")
    endif()
endif()

if (WITH_TESTS)
	include(FetchContent)
	include(${CMAKE_SOURCE_DIR}/third_party/googletest/googletest.cmake)
endif()
