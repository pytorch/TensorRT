cmake_minimum_required(VERSION 3.17)
project(torchtrt_runtime_example LANGUAGES CXX)

# Find dependencies
find_package(Torch REQUIRED)
find_package(torchtrt REQUIRED)

set(SRCS
    main.cpp
)

add_executable(${CMAKE_PROJECT_NAME} ${SRCS})
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE torch "-Wl,--no-as-needed" torchtrt_runtime "-Wl,--as-needed")
