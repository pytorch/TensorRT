set(lib_name "torch_tensorrt")
add_library(${lib_name} OBJECT)

set(CXX_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/compile_spec.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/logging.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ptq.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/torch_tensorrt.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/types.cpp"
)

set(HEADER_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/include/torch_tensorrt/logging.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/torch_tensorrt/macros.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/torch_tensorrt/ptq.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/torch_tensorrt/torch_tensorrt.h"
)

target_sources(${lib_name}
    PRIVATE
        ${CXX_SRCS}
        $<TARGET_OBJECTS:core>
)

target_link_libraries(${lib_name}
    PUBLIC
        torch
        TensorRT::nvinfer
        core
)

target_compile_definitions(${lib_name} PUBLIC "USE_CMAKE_GENERATED_EXPORT_HEADER")

include("GenerateExportHeader")
generate_export_header(${lib_name}
    EXPORT_MACRO_NAME TORCHTRT_API
    NO_EXPORT_MACRO_NAME TORCHTRT_HIDDEN
)

target_include_directories(${lib_name}
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>" # _export.h
    PRIVATE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>"
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${lib_name}_export.h DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

########################
# API library torchtrt #
########################
set(CMAKE_INSTALL_RPATH "$ORIGIN")

set(torchtrt_lib_name "torchtrt")

add_library(${torchtrt_lib_name} SHARED)

if(NOT MSVC)
    set_target_properties(
        ${torchtrt_lib_name}
        PROPERTIES INSTALL_RPATH_USE_LINK_PATH FALSE #
                   LINK_FLAGS "-Wl,--disable-new-dtags")
endif()

target_sources(${torchtrt_lib_name}
    PRIVATE
        $<TARGET_OBJECTS:torch_tensorrt>
)

target_link_libraries(${torchtrt_lib_name}
    PUBLIC
        TensorRT::TensorRT
        torch
    PRIVATE
        torch_tensorrt
        core
)

target_include_directories(${torchtrt_lib_name}
    PUBLIC
        # enable include from core from the tests
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>"
        # API headers
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>" # _export.h
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

install(FILES ${HEADER_FILES} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/torch_tensorrt/")

################################
# API library torchtrt_runtime #
################################
set(runtime_lib_name "torchtrt_runtime")
add_library(${runtime_lib_name} SHARED)

if(NOT MSVC)
    set_target_properties(
        ${runtime_lib_name}
        PROPERTIES INSTALL_RPATH_USE_LINK_PATH FALSE #
                   LINK_FLAGS "-Wl,--disable-new-dtags")
endif()

target_sources(${runtime_lib_name}
    PRIVATE
        $<TARGET_OBJECTS:core_runtime>
        $<TARGET_OBJECTS:core_plugins>
)

target_link_libraries(${runtime_lib_name}
    PUBLIC
        TensorRT::TensorRT
        torch
    PRIVATE
        core_runtime
        core_plugins
)

################################
# API library torchtrt_plugins #
################################
set(plugins_lib_name "torchtrt_plugins")
add_library(${plugins_lib_name} SHARED)

if(NOT MSVC)
    set_target_properties(
        ${plugins_lib_name}
        PROPERTIES INSTALL_RPATH_USE_LINK_PATH FALSE #
                   LINK_FLAGS "-Wl,--disable-new-dtags")
endif()

target_sources(${plugins_lib_name}
    PRIVATE
        $<TARGET_OBJECTS:core_plugins>
)

target_link_libraries(${plugins_lib_name}
    PUBLIC
        TensorRT::TensorRT
    PRIVATE
        core_plugins
)

install(
    TARGETS ${torchtrt_lib_name} ${runtime_lib_name} ${plugins_lib_name} torch_tensorrt core core_runtime core_plugins core_util core_util_logging core_lowering core_partitioning core_ir core_conversion
    EXPORT ${torchtrt_lib_name}Targets
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")

install(
    EXPORT ${torchtrt_lib_name}Targets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/torchtrt
    COMPONENT ${torchtrt_lib_name}Targets)

add_subdirectory(bin)
