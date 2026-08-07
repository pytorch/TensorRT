# Wheel-only runner.
#
# The point of this example is that it builds against INSTALLED packages. There is
# no add_subdirectory of an ExecuTorch source tree and no source checkout: both the
# runtime and the TensorRT delegate are found with find_package and linked as
# imported targets, exactly as an application outside this repository would.
#
# Configure with CMAKE_PREFIX_PATH pointing at the installed packages, for example
# the directory reported by
#   python -c "import executorch.utils as u; print(u.cmake_prefix_path)"
cmake_minimum_required(VERSION 3.28)
project(executorch_wheel_runner CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(executorch CONFIG REQUIRED)
if(NOT TARGET executorch::runtime)
    message(
        FATAL_ERROR
        "The installed ExecuTorch package does not provide executorch::runtime, "
        "so there is no prebuilt runtime to link. A wheel new enough to ship the "
        "shared runtime is required."
    )
endif()

add_executable(executorch_wheel_runner main.cpp)
target_link_libraries(executorch_wheel_runner PRIVATE executorch::runtime)

# Component targets come from the ExecuTorch package, so this example does not
# need to know where the wheel puts its libraries or how to keep a
# registration-only library on the link line. Each target is optional, because
# the same runner checks a plain CPU wheel where no delegate is installed.
foreach(_component threadpool kernels_optimized backend_xnnpack)
    if(TARGET executorch::${_component})
        target_link_libraries(
            executorch_wheel_runner PRIVATE executorch::${_component}
        )
        message(STATUS "linking executorch::${_component}")
    endif()
endforeach()

# The TensorRT delegate is optional: the same runner is used to check the plain
# CPU wheel, where no delegate is installed.
find_package(torch_tensorrt_executorch CONFIG QUIET)
if(TARGET torchtrt::backend_tensorrt)
    message(STATUS "TensorRT delegate found, linking it")
    target_link_libraries(
        executorch_wheel_runner PRIVATE torchtrt::backend_tensorrt
    )
    target_compile_definitions(
        executorch_wheel_runner PRIVATE RUNNER_HAS_TENSORRT_DELEGATE
    )
else()
    message(STATUS "TensorRT delegate not installed, running without it")
endif()

# The CUDA delegate ships in the accelerator wheel. Device-side input allocation
# additionally needs the CUDA toolkit headers, so it is only compiled in when both
# the delegate and the toolkit are present.
if(TARGET executorch::backend_cuda)
    target_link_libraries(
        executorch_wheel_runner PRIVATE executorch::backend_cuda
    )
    find_package(CUDAToolkit QUIET)
    if(TARGET CUDA::cudart)
        target_link_libraries(executorch_wheel_runner PRIVATE CUDA::cudart)
        target_compile_definitions(
            executorch_wheel_runner PRIVATE RUNNER_HAS_CUDA_DELEGATE
        )
    else()
        message(
            STATUS
            "CUDA delegate linked, but the toolkit headers are missing so "
            "device-side inputs are unavailable"
        )
    endif()
endif()
