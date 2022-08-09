# TensorRT
find_package(TensorRT QUIET)
if (NOT TensorRT_FOUND)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")
    find_package(TensorRT REQUIRED)
endif()

# If the custom finders are needed at this point, there are good chances that they will be needed when consuming the library as well
install(FILES "${CMAKE_SOURCE_DIR}/cmake/Modules/FindTensorRT.cmake" DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/torchtrt/Modules")
install(FILES "${CMAKE_SOURCE_DIR}/cmake/Modules/FindcuDNN.cmake" DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/torchtrt/Modules")

# CUDA
find_package(CUDAToolkit REQUIRED)
find_package(cuDNN REQUIRED) # Headers are needed somewhere

# libtorch
find_package(Torch REQUIRED)
find_package(Threads REQUIRED)

add_definitions(-DTORCH_VERSION_MAJOR=${Torch_VERSION_MAJOR})
add_definitions(-DTORCH_VERSION_MINOR=${Torch_VERSION_MINOR})
add_definitions(-DTORCH_VERSION_PATCH=${Torch_VERSION_PATCH})

if (WITH_TESTS)
	include(FetchContent)
	include(${CMAKE_SOURCE_DIR}/third_party/googletest/googletest.cmake)
endif()
