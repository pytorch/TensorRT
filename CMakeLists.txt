
cmake_minimum_required(VERSION 3.17)
project(Torch-TensorRT LANGUAGES CXX)

# use c++17
set(CMAKE_CXX_STANDARD 17)

# Build the libraries with -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (DEFINED CMAKE_MODULE_PATH)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} CACHE PATH "Path to the folder containing finders")
endif()

include(cmake/build_options.cmake)
include(cmake/paths.cmake)
include(cmake/dependencies.cmake)
if(MSVC)
	add_compile_options(/wd4624 /wd4067 /permissive-)
  # When using Ninja generator, suppress the warning D9025
  string(REPLACE "/Zi" "/Z7" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
  string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
endif()
# -----------------------------------------
#  compilation
# -----------------------------------------
add_subdirectory(core)
add_subdirectory(cpp)

include(CMakePackageConfigHelpers)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/torchtrtConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/torchtrt
)

install(FILES
          "${CMAKE_CURRENT_BINARY_DIR}/torchtrtConfig.cmake"
          # "${CMAKE_CURRENT_BINARY_DIR}/torchtrtConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/torchtrt
)
