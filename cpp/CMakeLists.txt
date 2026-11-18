# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Keep the library name for Dynamo runtime consumers. The TorchScript compiler
# and torchtrtc CLI have been removed.
set(CMAKE_INSTALL_RPATH "$ORIGIN")

add_library(torchtrt SHARED)
target_sources(torchtrt PRIVATE
    $<TARGET_OBJECTS:core_runtime>
    $<TARGET_OBJECTS:core_plugins>
)
target_link_libraries(torchtrt PUBLIC
    TensorRT::TensorRT
    "${TORCH_LIBRARIES}"
    core_runtime
    core_plugins
)
target_include_directories(torchtrt PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

add_library(torchtrt_runtime SHARED)
target_sources(torchtrt_runtime PRIVATE
    $<TARGET_OBJECTS:core_runtime>
    $<TARGET_OBJECTS:core_plugins>
)
target_link_libraries(torchtrt_runtime PUBLIC
    TensorRT::TensorRT
    "${TORCH_LIBRARIES}"
    core_runtime
    core_plugins
)

add_library(torchtrt_plugins SHARED)
target_sources(torchtrt_plugins PRIVATE $<TARGET_OBJECTS:core_plugins>)
target_link_libraries(torchtrt_plugins PUBLIC TensorRT::TensorRT core_plugins)

if(NOT MSVC)
    foreach(target torchtrt torchtrt_runtime torchtrt_plugins)
        set_target_properties(${target} PROPERTIES
            INSTALL_RPATH_USE_LINK_PATH FALSE
            LINK_FLAGS "-Wl,--disable-new-dtags"
        )
    endforeach()
endif()

install(TARGETS
    torchtrt torchtrt_runtime torchtrt_plugins
    core_runtime core_plugins core_util core_util_logging
    EXPORT torchtrtTargets
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)
install(EXPORT torchtrtTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/torchtrt
    COMPONENT torchtrtTargets
)
