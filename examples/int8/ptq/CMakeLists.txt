cmake_minimum_required(VERSION 3.17)
project(ptq LANGUAGES CXX)

# Find dependencies
find_package(Torch REQUIRED)
find_package(torchtrt REQUIRED)

set(SRCS
    ${CMAKE_SOURCE_DIR}/main.cpp
    ${CMAKE_SOURCE_DIR}/../benchmark/benchmark.cpp
    ${CMAKE_SOURCE_DIR}/../datasets/cifar10.cpp

)

add_executable(${CMAKE_PROJECT_NAME} ${SRCS})
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE "${CMAKE_SOURCE_DIR}/../../..")
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE torch "-Wl,--no-as-needed" torchtrt "-Wl,--as-needed")
