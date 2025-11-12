cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(torchtrt_aoti_example LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No CMAKE_BUILD_TYPE specified; defaulting to Debug for gdb")
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug;Release;RelWithDebInfo;MinSizeRel")
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Torch REQUIRED)
find_package(torchtrt REQUIRED)

add_executable(torchtrt_aoti_example inference.cpp model.pt2)

target_compile_options(torchtrt_aoti_example PRIVATE
  $<$<CONFIG:Debug>:-O0 -g3 -ggdb3 -fno-omit-frame-pointer -fno-optimize-sibling-calls -DDEBUG>
  $<$<CONFIG:RelWithDebInfo>:-g2 -fno-omit-frame-pointer>
)

add_custom_command(
    OUTPUT model.pt2
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/model.py
    DEPENDS model.py
)

target_link_libraries(torchtrt_aoti_example "${TORCH_LIBRARIES}" "-Wl,--no-as-needed" torchtrt_runtime "-Wl,--as-needed")
set_property(TARGET torchtrt_aoti_example PROPERTY CXX_STANDARD 17)
