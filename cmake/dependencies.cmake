# TensorRT
find_package(TensorRT REQUIRED)

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
