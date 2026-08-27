cmake_minimum_required(VERSION 3.24)

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
# Needed twice over: this backend registers the allocator the exported program's device
# copies look up, and extension/cuda, which owns the shared caller stream, comes with it.
set(EXECUTORCH_BUILD_CUDA ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_KERNELS_OPTIMIZED ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_XNNPACK ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# rules_foreign_cc puts an explicit dynamic -lstdc++ in the Bazel toolchain's linker
# flags, ahead of the object files and wrapped in --as-needed, where it resolves
# nothing. Remove it before anything that links is defined, and put a working one back
# below.
#
# Two things have to be right, and each was got wrong before.
#
# WHERE. Removing it later, after add_subdirectory below, is too late: the extensions
# are created in ExecuTorch's directory and take their copy of these variables at that
# point, so a later edit changes a value nothing reads. Measured by stripping before
# and after add_subdirectory and reading the subdirectory target's own link.txt.
#
# WHY IT IS NOT HARMLESS WHERE IT SITS. It arrives wrapped in --as-needed, before the
# object files, which with the default linker means it gets dropped. This build passes
# -fuse-ld=gold, and gold keeps it anyway. Measured in the release image on a shared
# object using std::string and exceptions:
#
#   bfd,       -lstdc++ present   -> no libstdc++ NEEDED
#   gold,      -lstdc++ present   -> libstdc++.so.6 NEEDED
#   gold+lto,  -lstdc++ present   -> libstdc++.so.6 NEEDED
#   gold+lto,  -lstdc++ removed   -> no libstdc++ NEEDED
#
# So the fragment has to go, wherever it sits. Matched by regex rather than as one
# exact string, so a spacing change cannot make the removal silently stop working.
# The resolved values are printed because the check at the end of this file reports a
# property of the artifact, and the first question is always what was on the link line.
# Only these three can carry it. rules_foreign_cc puts the toolchain's link libraries
# into CMAKE_{SHARED,MODULE,EXE}_LINKER_FLAGS_INIT, and filters the STANDARD_LIBRARIES
# variables down to static-runtime flags, so touching those would strip nothing and
# would additionally shadow the cache entry enable_language(C) creates later.
#
# The whole push/pop group goes at once. Removing only -lstdc++ from inside it would
# leave an unbalanced --push-state, which both linkers reject outright:
#   ld.gold: error: unbalanced --push-state/--pop-state
# The bare form is matched separately, with a boundary, so -lstdc++fs and
# -lstdc++_nonshared are left alone.
foreach(_torch_tensorrt_linker_flags
        CMAKE_MODULE_LINKER_FLAGS
        CMAKE_SHARED_LINKER_FLAGS
        CMAKE_EXE_LINKER_FLAGS)
  string(REGEX REPLACE "-Wl,--push-state,-as-needed +-lstdc\\+\\+ +-Wl,--pop-state"
    "" ${_torch_tensorrt_linker_flags} "${${_torch_tensorrt_linker_flags}}")
  string(REGEX REPLACE "(^| )-lstdc\\+\\+( |$)" "\\2"
    ${_torch_tensorrt_linker_flags} "${${_torch_tensorrt_linker_flags}}")
  message(STATUS "torch_tensorrt: ${_torch_tensorrt_linker_flags} = "
    "[${${_torch_tensorrt_linker_flags}}]")
endforeach()

# Removing that fragment is only half of it: something still has to supply the C++
# runtime. The Bazel toolchain hands CMake the C driver, gcc, as CMAKE_CXX_COMPILER,
# and gcc links no C++ runtime at all, which is exactly why the toolchain injected an
# explicit -lstdc++ in the first place.
#
# So put one back, after the object files. CMake appends CMAKE_CXX_STANDARD_LIBRARIES
# there, which is the position that works, and is the same placement rules_foreign_cc
# documents for exactly this problem.
#
# Measured on a shared object that stores a std::exception_ptr and formats through an
# ostringstream:
#
#                                                UND _M_addref   libstdc++ NEEDED
#   the toolchain's --as-needed group, pre-objects      0                0
#   nothing at all                                      0                0
#   -lstdc++ after the objects                          0                1
#
# It has to be the dynamic library and not libstdc++.a. These artifacts share a C++ ABI
# with libtorch_cpu.so, libc10.so, libtorch_python.so and libnvinfer.so.11, every one of
# which already carries DT_NEEDED libstdc++.so.6 and needs no more than GLIBCXX_3.4.22.
# A second static copy cannot be made safe in that company. Left visible it interposes
# the real runtime for anything loaded alongside it, and hidden it gives this wheel
# private __cxxabiv1 typeinfo and a private unwinder, so exceptions stop crossing the
# boundary and dynamic_cast starts returning null.
string(APPEND CMAKE_CXX_STANDARD_LIBRARIES " -lstdc++")
# try_compile forwards CMAKE_EXE_LINKER_FLAGS but not CMAKE_CXX_STANDARD_LIBRARIES, so
# without this a C++ probe loses the runtime the strip above removed and reports the
# feature absent instead of failing.
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES CMAKE_CXX_STANDARD_LIBRARIES)
message(STATUS "torch_tensorrt: CMAKE_CXX_STANDARD_LIBRARIES now = "
  "[${CMAKE_CXX_STANDARD_LIBRARIES}]")

add_subdirectory("${EXECUTORCH_SOURCE_DIR}" executorch)

add_library(torch_tensorrt_executorch_backend STATIC
  "${TORCH_TENSORRT_SOURCE_DIR}/cpp/src/torch_tensorrt/executorch/TensorRTBackend.cpp"
  "${TORCH_TENSORRT_SOURCE_DIR}/cpp/src/torch_tensorrt/executorch/TensorRTBlobHeader.cpp"
  "${TORCH_TENSORRT_SOURCE_DIR}/cpp/src/torch_tensorrt/executorch/WeightStreamingBudget.cpp")
target_compile_features(torch_tensorrt_executorch_backend PUBLIC cxx_std_17)
target_compile_definitions(torch_tensorrt_executorch_backend
  PRIVATE C10_USING_CUSTOM_GENERATED_MACROS)
target_include_directories(torch_tensorrt_executorch_backend PRIVATE
  "${TORCH_TENSORRT_SOURCE_DIR}"
  "${TORCH_TENSORRT_SOURCE_DIR}/cpp/include"
  "${EXECUTORCH_SOURCE_DIR}/.."
  "${EXECUTORCH_SOURCE_DIR}/runtime/core/portable_type/c10")
if(NOT TARGET extension_cuda)
  message(FATAL_ERROR
    "ExecuTorch did not define extension_cuda; the delegate needs it for the shared "
    "caller-stream selection. Configure ExecuTorch with EXECUTORCH_BUILD_CUDA=ON.")
endif()

target_link_libraries(torch_tensorrt_executorch_backend PRIVATE
  CUDA::cudart TensorRT::nvinfer Threads::Threads executorch extension_cuda)

foreach(_torch_tensorrt_required_cuda_target extension_cuda aoti_cuda_shims)
  if(NOT TARGET ${_torch_tensorrt_required_cuda_target})
    message(FATAL_ERROR
      "ExecuTorch did not define ${_torch_tensorrt_required_cuda_target}; the wheel "
      "ships it and the extensions load it. Check EXECUTORCH_BUILD_CUDA.")
  endif()
endforeach()

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

# What supplies the C++ runtime is the -lstdc++ appended to
# CMAKE_CXX_STANDARD_LIBRARIES near the top of this file. LINKER_LANGUAGE CXX is what
# selects CMAKE_CXX_COMPILER and those standard libraries for these targets, so it has
# to be set on every one of them.
#
# The CUDA backend brings two shared libraries of its own. They get the same treatment,
# because libaoti_cuda_shims.so is the first DT_NEEDED of _portable_lib.so and so leads
# the whole dlopen group in symbol search order.
set(_torch_tensorrt_executorch_runtime_targets portable_lib data_loader)
foreach(_torch_tensorrt_cuda_shared_target extension_cuda aoti_cuda_shims)
  get_target_property(_torch_tensorrt_cuda_shared_type
    ${_torch_tensorrt_cuda_shared_target} TYPE)
  # An IMPORTED prebuilt is already linked and cannot be given link options.
  get_target_property(_torch_tensorrt_cuda_shared_imported
    ${_torch_tensorrt_cuda_shared_target} IMPORTED)
  if(_torch_tensorrt_cuda_shared_type STREQUAL "SHARED_LIBRARY" AND
     NOT _torch_tensorrt_cuda_shared_imported)
    list(APPEND _torch_tensorrt_executorch_runtime_targets
      ${_torch_tensorrt_cuda_shared_target})
  endif()
endforeach()

foreach(_torch_tensorrt_executorch_runtime_target
        IN LISTS _torch_tensorrt_executorch_runtime_targets)
  set_property(TARGET ${_torch_tensorrt_executorch_runtime_target}
    PROPERTY LINKER_LANGUAGE CXX)
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
# data_loader gets $ORIGIN too, so finding those libraries does not depend on which
# module Python imports first.
set_target_properties(data_loader PROPERTIES
  BUILD_WITH_INSTALL_RPATH ON
  INSTALL_RPATH "${_torch_tensorrt_executorch_runtime_rpath}"
  SUFFIX ".so")

install(TARGETS portable_lib data_loader LIBRARY DESTINATION lib)

# ExecuTorch's published wheel ships neither of these and the extensions carry a
# DT_NEEDED on them, so importing the runtime fails without them. ExecuTorch installs
# them to CMAKE_INSTALL_LIBDIR, which is lib64 on some distributions, so put them in
# lib/ as well, beside the extensions.
install(TARGETS extension_cuda aoti_cuda_shims LIBRARY DESTINATION lib)

add_custom_target(torch_tensorrt_executorch_portable_lib ALL
  DEPENDS ${_torch_tensorrt_executorch_runtime_targets})

# Guard both halves of the C++ runtime contract on the real artifacts, because this
# wheel has no auditwheel step behind it: nothing here may define libstdc++'s own
# symbols, and anything referencing the runtime must declare it.
#
# ALL on the aggregate target above matters: a custom target without it is excluded
# from the default build, and the default target is what the wheel build runs. The
# check hangs off that aggregate rather than off portable_lib and data_loader
# directly, because add_custom_command(TARGET) only accepts targets created in this
# directory and those two come from ExecuTorch's subdirectory.
#
# The check lives in a script rather than an inline shell string so it can report what
# it saw. A bare "defines libstdc++ symbols" does not say which input added them, and
# the build logs do not print link lines, so the script prints both.
find_program(TORCH_TENSORRT_READELF NAMES readelf llvm-readelf)
if(NOT TORCH_TENSORRT_READELF AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR
    "readelf is required to verify the Python extensions share the C++ runtime. "
    "Install binutils, or set TORCH_TENSORRT_READELF to a readelf.")
endif()
if(TORCH_TENSORRT_READELF)
  foreach(_torch_tensorrt_checked_target
          IN LISTS _torch_tensorrt_executorch_runtime_targets)
    add_custom_command(TARGET torch_tensorrt_executorch_portable_lib POST_BUILD
      COMMAND "${CMAKE_COMMAND}" -E echo
        "checking $<TARGET_FILE_NAME:${_torch_tensorrt_checked_target}> shares the C++ runtime"
      COMMAND sh
        "${CMAKE_CURRENT_LIST_DIR}/check_shared_cxx_runtime.sh"
        "${TORCH_TENSORRT_READELF}"
        "$<TARGET_FILE:${_torch_tensorrt_checked_target}>"
        "$<TARGET_PROPERTY:${_torch_tensorrt_checked_target},BINARY_DIR>/CMakeFiles/${_torch_tensorrt_checked_target}.dir/link.txt"
      VERBATIM)
  endforeach()
endif()
