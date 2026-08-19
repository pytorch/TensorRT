cmake_minimum_required(VERSION 3.24)

# rules_foreign_cc initializes these flags with an explicit dynamic -lstdc++.
# CMake consumes the *_INIT variables when project() enables CXX, so removing
# it later from CMAKE_{MODULE,SHARED}_LINKER_FLAGS is too late on some CMake
# versions: the dynamic runtime is already retained in the generated link
# rule. Preserve the remaining Bazel toolchain flags, but remove only that
# dependency before CMake initializes the language.
foreach(_torch_tensorrt_linker_flags_init
    CMAKE_MODULE_LINKER_FLAGS_INIT
    CMAKE_SHARED_LINKER_FLAGS_INIT)
  string(REPLACE "-Wl,--push-state,-as-needed -lstdc++ -Wl,--pop-state" ""
    _torch_tensorrt_sanitized_linker_flags
    "${${_torch_tensorrt_linker_flags_init}}")
  set(${_torch_tensorrt_linker_flags_init}
    "${_torch_tensorrt_sanitized_linker_flags}" CACHE STRING
    "Torch-TensorRT sanitized linker flags" FORCE)
endforeach()

project(torch_tensorrt_executorch_runtime LANGUAGES CXX)

if(NOT EXECUTORCH_SOURCE_DIR AND EXECUTORCH_CMAKE_FILE)
  get_filename_component(EXECUTORCH_SOURCE_DIR "${EXECUTORCH_CMAKE_FILE}" DIRECTORY)
endif()
if(NOT TORCH_PRIMARY_INCLUDE_DIR AND TORCH_HEADER_MARKER)
  get_filename_component(TORCH_PRIMARY_INCLUDE_DIR "${TORCH_HEADER_MARKER}/../../../.." ABSOLUTE)
endif()
if(NOT EXECUTORCH_SOURCE_DIR OR NOT TORCH_TENSORRT_SOURCE_DIR OR NOT TORCH_PRIMARY_INCLUDE_DIR)
  message(FATAL_ERROR
    "EXECUTORCH_SOURCE_DIR, TORCH_TENSORRT_SOURCE_DIR, and TORCH_PRIMARY_INCLUDE_DIR are required")
endif()

list(APPEND CMAKE_MODULE_PATH "${TORCH_TENSORRT_SOURCE_DIR}/cmake/Modules")
find_package(TensorRT REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# ExecuTorch declares every pybind module with pybind11_add_module(<target> SHARED
# ...). pybind11 maps a non-MODULE type onto pybind11::embed, and CMake's
# python_add_library then requires the Python::Python target, so the first module
# ExecuTorch declares (codegen/tools/selective_build) aborts the configure with:
#
#   Python_ADD_LIBRARY: dependent target 'Python::Python' is not defined.
#      Did you miss to request COMPONENT 'Development.Embed'?
#
# Do NOT take that hint literally. Development.Embed needs a libpython, and the
# manylinux CPython in the release image is built without one, so requesting it
# just fails the whole find_package instead:
#
#   Could NOT find Python (missing: Python_INCLUDE_DIRS Python_LIBRARIES ...)
#
# pybind11 hit this and deliberately made the component optional for manylinux
# (pybind11NewTools.cmake: "Development.Module support (required for manylinux)"),
# so match that: Module required, Embed optional. Finding Python before pybind11
# is the override pybind11 documents.
#
# PYTHON_EXECUTABLE is bridged first because the build passes the interpreter
# under that name while FindPython reads Python_EXECUTABLE. ExecuTorch does the
# same bridge, but only after this point, so without it this call could resolve a
# different interpreter than the rest of the build.
if(PYTHON_EXECUTABLE AND NOT Python_EXECUTABLE)
  set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
endif()
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module
             OPTIONAL_COMPONENTS Development.Embed)

# With Embed optional, Python::Python does not exist on manylinux, and the SHARED
# declaration above still demands it. Supply an empty stand-in for that case only.
# It links nothing, which is what an extension module needs: Python symbols
# resolve from the interpreter that loads the module, never from a linked
# libpython. Where the image does provide a usable libpython, the real imported
# target is used instead.
#
# This is a workaround for a defect upstream, not the root fix. Those three
# modules should be declared MODULE rather than SHARED: CMake skips this check
# entirely for MODULE, and MODULE is what a Python extension is. Filed upstream.
if(NOT TARGET Python::Python)
  add_library(Python::Python INTERFACE IMPORTED)
  # The real Python::Python carries INTERFACE_INCLUDE_DIRECTORIES. Forward those
  # through Python::Module so the stand-in is not silently header-less if a
  # consumer ever relies on them. Python::Module links no libpython.
  set_target_properties(Python::Python PROPERTIES
    INTERFACE_LINK_LIBRARIES Python::Module)
endif()

set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_PYBIND ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_MODULE ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_TENSOR ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_KERNELS_OPTIMIZED ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_XNNPACK ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory("${EXECUTORCH_SOURCE_DIR}" executorch)

add_library(torch_tensorrt_executorch_backend STATIC
  "${TORCH_TENSORRT_SOURCE_DIR}/cpp/src/torch_tensorrt/executorch/TensorRTBackend.cpp"
  "${TORCH_TENSORRT_SOURCE_DIR}/cpp/src/torch_tensorrt/executorch/TensorRTBlobHeader.cpp")
target_compile_features(torch_tensorrt_executorch_backend PUBLIC cxx_std_17)
target_compile_definitions(torch_tensorrt_executorch_backend
  PRIVATE C10_USING_CUSTOM_GENERATED_MACROS)
target_include_directories(torch_tensorrt_executorch_backend PRIVATE
  "${TORCH_TENSORRT_SOURCE_DIR}"
  "${TORCH_TENSORRT_SOURCE_DIR}/cpp/include"
  "${EXECUTORCH_SOURCE_DIR}/.."
  "${EXECUTORCH_SOURCE_DIR}/runtime/core/portable_type/c10")
target_link_libraries(torch_tensorrt_executorch_backend PRIVATE
  CUDA::cudart TensorRT::nvinfer Threads::Threads executorch)

if(NOT TARGET portable_lib)
  message(FATAL_ERROR "ExecuTorch did not define portable_lib")
endif()

# The pybind bridge includes both ATen and portable ExecuTorch headers. Keep the
# ATen/c10/torch header set internally consistent by resolving all of it from
# the exact PyTorch wheel that supplies the linked libraries.
if(NOT EXISTS "${TORCH_PRIMARY_INCLUDE_DIR}/torch/headeronly")
  message(FATAL_ERROR
    "torch/headeronly not found under TORCH_PRIMARY_INCLUDE_DIR=${TORCH_PRIMARY_INCLUDE_DIR}")
endif()
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/torch_header_shim/torch")
file(COPY "${TORCH_PRIMARY_INCLUDE_DIR}/c10/"
  DESTINATION "${CMAKE_BINARY_DIR}/torch_header_shim/c10")
file(COPY "${TORCH_PRIMARY_INCLUDE_DIR}/torch/headeronly/"
  DESTINATION "${CMAKE_BINARY_DIR}/torch_header_shim/torch/headeronly")
if(NOT EXISTS "${CMAKE_BINARY_DIR}/torch_header_shim/c10/util/complex.h" OR
   NOT EXISTS "${CMAKE_BINARY_DIR}/torch_header_shim/torch/headeronly/macros/Macros.h")
  message(FATAL_ERROR "Failed to stage the PyTorch header shim")
endif()
target_include_directories(util BEFORE PRIVATE "${CMAKE_BINARY_DIR}/torch_header_shim")
target_include_directories(portable_lib BEFORE PRIVATE "${CMAKE_BINARY_DIR}/torch_header_shim")

# TensorRTBackend registers through a static initializer. Force its complete
# archive into the same pybind module that owns ExecuTorch's backend registry.
target_link_libraries(portable_lib PRIVATE
  "$<LINK_LIBRARY:WHOLE_ARCHIVE,torch_tensorrt_executorch_backend>"
  extension_threadpool
  CUDA::cudart TensorRT::nvinfer Threads::Threads)

# The build and validation containers do not necessarily provide the same
# libstdc++, so both Python extensions link the C++ runtime statically. Setting
# LINKER_LANGUAGE CXX below is what makes -static-libstdc++ take effect: without
# it, data_loader is linked by the C driver and its shared object retained an
# unresolved exception_ptr::_M_addref reference on the older CI runtime. That
# symbol is supplied by libstdc++_nonshared.a, which the toolchain links for us,
# so the driver flags plus the correct linker language are sufficient.
# Probed as a precondition, not as a link input. -static-libstdc++ below makes the
# driver resolve -lstdc++ to exactly this file, so if it is absent the link fails
# with a less obvious error. Nothing names it on the link line: doing that is what
# caused duplicate definitions against libstdc++_nonshared.a.
execute_process(
  COMMAND "${CMAKE_CXX_COMPILER}" -print-file-name=libstdc++.a
  OUTPUT_VARIABLE TORCH_TENSORRT_STATIC_LIBSTDCXX
  OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(
  COMMAND "${CMAKE_CXX_COMPILER}" -print-libgcc-file-name
  OUTPUT_VARIABLE TORCH_TENSORRT_STATIC_LIBGCC
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT EXISTS "${TORCH_TENSORRT_STATIC_LIBSTDCXX}" OR
   NOT EXISTS "${TORCH_TENSORRT_STATIC_LIBGCC}")
  message(FATAL_ERROR "A static C++ runtime is required to build the ExecuTorch runtime wheel")
endif()

# rules_foreign_cc injects an explicit dynamic -lstdc++ from the Bazel toolchain.
# Retaining it makes the wheel depend on the build host's CXXABI version, so drop it
# wherever it lands.
#
# Position on the link line is what decides whether it matters, measured in the
# release image with a shared object that uses std::string and exceptions:
#
#   -static-libstdc++ -static-libgcc                       -> no libstdc++ NEEDED
#   ... plus -lstdc++ BEFORE the objects                    -> no libstdc++ NEEDED
#   ... plus -lstdc++ AFTER the objects                     -> libstdc++.so.6 NEEDED
#
# Before the objects nothing is undefined yet, so --as-needed drops it. After them the
# libstdc++ symbols are still unresolved, because -static-libstdc++ makes the driver
# append libstdc++.a at the very end, so --as-needed keeps it and the NEEDED sticks.
# CMake puts CMAKE_*_LINKER_FLAGS before the objects and CMAKE_*_STANDARD_LIBRARIES
# after them, so the standard-libraries variables are the ones that can actually hurt.
#
# Matched by regex rather than as one exact string: the previous exact-string match
# covered only the flags variables and would also miss a spacing change silently.
foreach(_torch_tensorrt_linker_flags
        CMAKE_MODULE_LINKER_FLAGS
        CMAKE_SHARED_LINKER_FLAGS
        CMAKE_EXE_LINKER_FLAGS
        CMAKE_CXX_STANDARD_LIBRARIES
        CMAKE_C_STANDARD_LIBRARIES)
  string(REGEX REPLACE "(-Wl,--push-state,-as-needed +)?-lstdc\\+\\+( +-Wl,--pop-state)?"
    "" ${_torch_tensorrt_linker_flags} "${${_torch_tensorrt_linker_flags}}")
  # Reported unconditionally: when the check at the end of this file trips, the first
  # question is always what ended up on the link line.
  message(STATUS "torch_tensorrt: ${_torch_tensorrt_linker_flags} = "
    "[${${_torch_tensorrt_linker_flags}}]")
endforeach()

foreach(_torch_tensorrt_executorch_runtime_target portable_lib data_loader)
  # Keep the C++ runtime self-contained. The compiler driver otherwise
  # appends a dynamic -lstdc++ even when libstdc++.a is a direct link input,
  # leaving CXXABI-versioned exception_ptr symbols for the host runtime.
  set_property(TARGET ${_torch_tensorrt_executorch_runtime_target}
    PROPERTY LINKER_LANGUAGE CXX)
  # Do not add libstdc++.a here, and in particular do not whole-archive it. The
  # driver flags below already select the static runtime, and on a Red Hat
  # gcc-toolset the toolchain adds a second archive of its own: libstdc++.so is a
  # text linker script, not an ELF object, and it reads
  #
  #   INPUT ( /usr/lib64/libstdc++.so.6 -lstdc++_nonshared )
  #
  # so libstdc++_nonshared.a arrives as an independent input carrying the
  # newer-ABI symbols the base libstdc++.so.6 lacks. That archive is a strict
  # subset of libstdc++.a: measured in the release image with
  # `nm --defined-only -g`, counting T/W/B/D/R, 1003 definitions and none unique to
  # it. Counting all global types gives 1141 instead; "none unique" holds either
  # way, which is the part that matters. Whole-archiving libstdc++.a
  # forces every member in whether referenced or not, so all 1003 collide and the
  # link fails on whichever the linker reaches first.
  target_link_libraries(${_torch_tensorrt_executorch_runtime_target} PRIVATE
    "${TORCH_TENSORRT_STATIC_LIBGCC}")
  target_link_options(${_torch_tensorrt_executorch_runtime_target} PRIVATE
    -static-libstdc++
    -static-libgcc)

endforeach()

# The runtime wheel intentionally does not bundle PyTorch, TensorRT, or CUDA.
# PyTorch, TensorRT, and CUDA pip packages install their shared libraries in
# directories under site-packages, relative to this extension package.
# Preserve these paths in both the Bazel-collected build output and installed
# extension so importing the wheel does not depend on LD_LIBRARY_PATH.
set(_torch_tensorrt_executorch_runtime_rpath
  "$ORIGIN"
  "$ORIGIN/../torch/lib"
  "$ORIGIN/../tensorrt_libs"
  "$ORIGIN/../nvidia/cuda_runtime/lib"
  "$ORIGIN/../nvidia/cu13/lib")
set_target_properties(portable_lib PROPERTIES
  BUILD_WITH_INSTALL_RPATH ON
  INSTALL_RPATH "${_torch_tensorrt_executorch_runtime_rpath}"
  OUTPUT_NAME "_portable_lib"
  SUFFIX ".so")
set_target_properties(data_loader PROPERTIES SUFFIX ".so")

install(TARGETS portable_lib data_loader LIBRARY DESTINATION lib)

add_custom_target(torch_tensorrt_executorch_portable_lib ALL DEPENDS portable_lib data_loader)

# Guard the two properties the removed whole-archive used to guarantee, on the real
# artifacts: no dynamic dependency on the build host's libstdc++, and no undefined
# exception_ptr::_M_addref. This wheel has no auditwheel step behind it.
#
# ALL on the aggregate target above matters: a custom target without it is excluded
# from the default build, and the default target is what the wheel build runs. The
# check hangs off that aggregate rather than off portable_lib and data_loader
# directly, because add_custom_command(TARGET) only accepts targets created in this
# directory and those two come from ExecuTorch's subdirectory.
#
# The check lives in a script rather than an inline shell string so it can report what
# it saw. A bare "has a dynamic libstdc++ dependency" does not say which input added
# it, and the build logs do not print link lines, so the script prints both.
find_program(TORCH_TENSORRT_READELF NAMES readelf llvm-readelf)
if(NOT TORCH_TENSORRT_READELF AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR
    "readelf is required to verify the Python extensions link the C++ runtime "
    "statically. Install binutils, or set TORCH_TENSORRT_READELF to a readelf.")
endif()
if(TORCH_TENSORRT_READELF)
  foreach(_torch_tensorrt_checked_target portable_lib data_loader)
    add_custom_command(TARGET torch_tensorrt_executorch_portable_lib POST_BUILD
      COMMAND "${CMAKE_COMMAND}" -E echo
        "checking $<TARGET_FILE_NAME:${_torch_tensorrt_checked_target}> links the C++ runtime statically"
      COMMAND sh
        "${CMAKE_CURRENT_LIST_DIR}/check_static_cxx_runtime.sh"
        "${TORCH_TENSORRT_READELF}"
        "$<TARGET_FILE:${_torch_tensorrt_checked_target}>"
        "$<TARGET_PROPERTY:${_torch_tensorrt_checked_target},BINARY_DIR>/CMakeFiles/${_torch_tensorrt_checked_target}.dir/link.txt"
      VERBATIM)
  endforeach()
endif()
