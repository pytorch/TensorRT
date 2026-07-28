cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(torchtrt_aoti_example LANGUAGES CXX)

find_package(Torch REQUIRED)
find_package(torchtrt REQUIRED)

add_executable(torchtrt_aoti_example inference.cpp model.pt2)

add_custom_command(
    OUTPUT model.pt2
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/model.py
    DEPENDS model.py
)

target_link_libraries(torchtrt_aoti_example "${TORCH_LIBRARIES}" "-Wl,--no-as-needed" torchtrt_runtime "-Wl,--as-needed")
set_property(TARGET torchtrt_aoti_example PROPERTY CXX_STANDARD 17)
