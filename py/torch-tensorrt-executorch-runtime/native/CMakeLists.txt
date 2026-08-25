# ExecuTorch's package config withholds every imported target below CMake 3.28, because it
# exports "$ORIGIN"-relative runpaths as link options and CMake writes that token incorrectly
# before then. It withholds them without failing: find_package still reports success and falls
# back to plain path variables, so a lower floor here would configure cleanly and then fail on
# the first executorch:: target with "the target was not found", which reads as a missing
# component rather than a version problem. rules_foreign_cc 0.15.1 supplies 3.31.8.
cmake_minimum_required(VERSION 3.28)

project(torch_tensorrt_executorch_runtime LANGUAGES CXX)

if(NOT TORCH_TENSORRT_SOURCE_DIR)
  message(FATAL_ERROR "TORCH_TENSORRT_SOURCE_DIR is required")
endif()

list(APPEND CMAKE_MODULE_PATH "${TORCH_TENSORRT_SOURCE_DIR}/cmake/Modules")
find_package(TensorRT REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# The prebuilt ExecuTorch runtime, from the installed wheel. Everything this wheel used to
# rebuild from source now comes from here, so the delegate links the same libexecutorch.so the
# user's ExecuTorch already loaded rather than a second private copy of it.
#
# setup.py resolves the prefix from the imported executorch module and passes it through Bazel.
# rules_foreign_cc merges its own dependency prefixes in, so this list is not empty even when
# that value is, and the check below is only a clearer message for the case where CMake is
# driven directly with no prefix at all.
if(NOT CMAKE_PREFIX_PATH)
  message(FATAL_ERROR
    "CMAKE_PREFIX_PATH is empty, so the installed ExecuTorch cannot be located. Build this "
    "wheel through its setup.py, which resolves the path from the imported executorch "
    "module.")
endif()
find_package(executorch REQUIRED)

foreach(_torch_tensorrt_required_target executorch::runtime executorch::extension_cuda)
  if(NOT TARGET ${_torch_tensorrt_required_target})
    message(FATAL_ERROR
      "The installed ExecuTorch does not provide ${_torch_tensorrt_required_target}. A CUDA "
      "wheel from the pinned nightly channel is required; the CPU wheel ships no CUDA "
      "extension, and releases up to 1.4.1 ship no linkable runtime at all.")
  endif()
endforeach()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# One shared library, holding only the TensorRT-specific code. This is the same shape as
# ExecuTorch's own delegates: leave register_backend undefined, carry a DT_NEEDED on
# libexecutorch.so, and let the loader bind the two together. executorch::extension_cuda
# supplies the caller stream, whose thread-local must have exactly one definition in the
# process; linking the shipped shared library is what guarantees that, since a static copy
# would give this library a second, invisibly separate stream.
add_library(executorch_backend_tensorrt SHARED
  "${TORCH_TENSORRT_SOURCE_DIR}/cpp/src/torch_tensorrt/executorch/TensorRTBackend.cpp"
  "${TORCH_TENSORRT_SOURCE_DIR}/cpp/src/torch_tensorrt/executorch/TensorRTBlobHeader.cpp"
  "${TORCH_TENSORRT_SOURCE_DIR}/cpp/src/torch_tensorrt/executorch/WeightStreamingBudget.cpp")

target_include_directories(executorch_backend_tensorrt PRIVATE
  "${TORCH_TENSORRT_SOURCE_DIR}"
  "${TORCH_TENSORRT_SOURCE_DIR}/cpp/include")

# Compile definitions and the C++ standard come from the package rather than being repeated
# here. ET_EVENT_TRACER_ENABLED in particular has to match how the runtime was built: it
# switches the tracer hooks between real bodies and empty ones, so a delegate that guesses
# wrong still links and runs while recording nothing.
target_link_libraries(executorch_backend_tensorrt PRIVATE
  executorch::runtime
  executorch::extension_cuda
  CUDA::cudart
  TensorRT::nvinfer
  Threads::Threads)

# Link the C++ runtime dynamically, the way ExecuTorch's own delegates and every other shared object
# in the process (libtorch, libc10, libnvinfer) already do. The build toolchain is newer than the
# libstdc++.so.6 on a user's machine, so an optimized build emits out-of-line calls into the newer
# runtime (for example std::string::_M_replace_cold). Naming stdc++ as a link library places `-lstdc++`
# after the objects, where the toolchain's own libstdc++.so linker script resolves those references:
# it dynamic-links the old system libstdc++.so.6 for the stable, versioned symbols and pulls only the
# newer helpers statically from its libstdc++_nonshared.a. The result carries a normal DT_NEEDED on
# libstdc++.so.6 and needs no version above what the paired ExecuTorch already requires. A static C++
# runtime is deliberately avoided: this delegate is dlopened beside libtorch and ExecuTorch, and a
# private libstdc++ would give it its own exception type_info and locale state, which breaks exceptions
# and dynamic_cast across the boundary.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_libraries(executorch_backend_tensorrt PRIVATE stdc++)
endif()

# extension_cuda has to be a shared library, or the delegate statically absorbs the CUDA stream
# implementation instead of sharing ExecuTorch's, and a mixed-delegate run ends up on a different
# stream while every registration and import check still passes. The link-time guard requires a
# DT_NEEDED on libexecutorch_extension_cuda.so, which only a shared target leaves; fail configuration
# here too so the cause is named at configure time rather than as a missing DT_NEEDED later.
if(TARGET executorch::extension_cuda)
  get_target_property(_extension_cuda_type executorch::extension_cuda TYPE)
  if(_extension_cuda_type STREQUAL "STATIC_LIBRARY")
    message(FATAL_ERROR
      "executorch::extension_cuda is a STATIC_LIBRARY; the delegate would absorb a private CUDA "
      "stream implementation instead of sharing ExecuTorch's. Use a CUDA ExecuTorch build that "
      "ships libexecutorch_extension_cuda.so.")
  endif()
endif()

# ExecuTorch exports each delegate as executorch::backend_<name>, backing a
# libexecutorch_backend_<name>.so. It reaches that from short internal target names such as
# xnnpack_backend plus OUTPUT_NAME and an alias; naming the target for the file it produces gets
# to the same place with one fewer indirection, so the target, the alias and the shipped file all
# agree and OUTPUT_NAME is unnecessary. What a C++ consumer spells is identical either way.
#
# The runpath reaches sibling site-packages distributions, because this wheel deliberately
# bundles neither ExecuTorch, PyTorch, TensorRT, nor CUDA. The ExecuTorch entry is the new one:
# libexecutorch.so lives in another distribution's lib directory, so $ORIGIN/../lib does not
# reach it. TWO levels, because the wheel ships this library in a lib/ subdirectory of the
# package, matching where ExecuTorch keeps its own backends: $ORIGIN is site-packages/
# torch_tensorrt_executorch_runtime/lib, so site-packages itself is two levels up. Derive the
# depth from that layout, because ExecuTorch's own delegates are laid out differently: measured
# on the shipped wheel, libexecutorch_backend_cuda.so has RUNPATH
# $ORIGIN:$ORIGIN/../../nvidia/cu13/lib:$ORIGIN/../backends/cuda:$ORIGIN/../lib:...,
# where $ORIGIN/../lib reaches the runtime because that delegate sits in a subdirectory of the
# same distribution. This one sits in a different distribution, so the same relative path lands
# somewhere else entirely and the depth has to come from this wheel's layout instead.
#
# The depth and the install location are one decision. setup.py installs the library under
# lib/, __init__.py looks for it there, and the CMake package searches there; if any of those
# moves, every entry below is off by a level and the delegate fails to resolve its own
# dependencies. tests/py/dynamo/executorch/test_api.py asserts the pairing.
#
# One list, two consumers: CMake wants it semicolon-separated, patchelf below wants colons, and
# patchelf runs last so it is the one that decides what ships. Keeping two hand-written copies
# meant the ineffective one could drift without any signal.
#
# Both CUDA layouts are listed because the two major versions package their runtime differently:
# the cu13 wheels install nvidia/cu13/lib, while the cu12 wheels install one directory per
# component, nvidia/cuda_runtime/lib and friends. Measured by unpacking both wheels rather than
# assumed from the version number, since "nvidia/cu12/lib" does not exist. A RUNPATH entry that
# resolves to nothing costs a stat call, so carrying both keeps one delegate correct on either
# channel instead of encoding the CUDA major at build time.
set(TORCH_TENSORRT_DELEGATE_RUNPATH
  "$ORIGIN"
  "$ORIGIN/../../executorch/lib"
  "$ORIGIN/../../tensorrt_libs"
  "$ORIGIN/../../nvidia/cu13/lib"
  "$ORIGIN/../../nvidia/cuda_runtime/lib")
string(JOIN ":" TORCH_TENSORRT_DELEGATE_RUNPATH_COLONS ${TORCH_TENSORRT_DELEGATE_RUNPATH})

# The manylinux platform this row ships under, per architecture: .github/scripts/filter-matrix.py
# sends the aarch64 row to a manylinux_2_39 image rather than 2_28, because TensorRT needs a newer
# glibc there. One fixed tag rejected that row for requiring what its own builder provides.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(TORCH_TENSORRT_MANYLINUX_TAG "manylinux_2_39")
else()
  set(TORCH_TENSORRT_MANYLINUX_TAG "manylinux_2_28")
endif()
set_target_properties(executorch_backend_tensorrt PROPERTIES
  BUILD_WITH_INSTALL_RPATH ON
  INSTALL_RPATH "${TORCH_TENSORRT_DELEGATE_RUNPATH}")

# executorch::runtime carries the build machine's own site-packages path as a raw
# INTERFACE_LINK_OPTIONS -rpath, which BUILD_WITH_INSTALL_RPATH does not suppress and install
# does not rewrite, so without this the published wheel ships a path from the CI builder. It is
# also placed ahead of the entries above, and measuring with LD_DEBUG=libs shows the loader
# resolving libexecutorch.so straight out of it rather than from $ORIGIN/../../executorch/lib --
# which is exactly the blind spot that let an earlier wrong RUNPATH depth go unnoticed, because
# on the build machine the absolute path works no matter what the relative entries say.
#
# Rewritten rather than deleted: the relative entries are the ones that have to do the work, and
# a run with no absolute fallback is the only run that can prove they do.
#
# Plain --set-rpath, never --force-rpath: patchelf writes DT_RUNPATH without it and the older
# DT_RPATH with it. The pinned ExecuTorch passes --enable-new-dtags for exactly this reason and
# writes down why -- DT_RPATH is searched before LD_LIBRARY_PATH and applies transitively to a
# dependency's dependencies, so a consumer could not point an instrumented or locally built
# runtime at their application. Forcing the old tag here would reverse that for the delegate.
find_program(TORCH_TENSORRT_PATCHELF NAMES patchelf)
if(TORCH_TENSORRT_PATCHELF)
  add_custom_command(TARGET executorch_backend_tensorrt POST_BUILD
    COMMAND "${TORCH_TENSORRT_PATCHELF}" --remove-rpath
      "$<TARGET_FILE:executorch_backend_tensorrt>"
    COMMAND "${TORCH_TENSORRT_PATCHELF}" --set-rpath
      "${TORCH_TENSORRT_DELEGATE_RUNPATH_COLONS}"
      "$<TARGET_FILE:executorch_backend_tensorrt>"
    COMMENT "Removing absolute build-machine RUNPATH entries from the delegate"
    VERBATIM)
else()
  message(FATAL_ERROR
    "patchelf is required to strip the absolute build-machine RUNPATH that ExecuTorch's "
    "imported targets add. Install patchelf, or set TORCH_TENSORRT_PATCHELF.")
endif()

# The same spelling consumers use for ExecuTorch's own delegates, so a project that builds this
# in-tree and links executorch::backend_cuda can link executorch::backend_tensorrt without
# learning a second convention.
#
# In-tree only, deliberately. Exporting an installed tree would need a generated package config
# file for find_package to resolve, and the wheel would have to ship it: the wheel carries only
# the .so, and the Bazel rule declares no output directory for anything else, so an
# install(EXPORT) here would generate targets files that reach no consumer on any path this
# repository ships.
add_library(executorch::backend_tensorrt ALIAS executorch_backend_tensorrt)

# Registration happens in a static initializer, so a consumer references no symbol from this
# library and --as-needed drops the DT_NEEDED entirely -- measured: the entry disappears and the
# initializer never runs, with no diagnostic. ExecuTorch wraps its own registration-only component
# libraries the same way, one option per library because CMake dedupes identical push-state text,
# and only on Linux because that is where the linker has the flag.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_options(executorch_backend_tensorrt INTERFACE
    "LINKER:--push-state,--no-as-needed,$<TARGET_FILE:executorch_backend_tensorrt>,--pop-state")
endif()

install(TARGETS executorch_backend_tensorrt LIBRARY DESTINATION lib)

# The delegate has to resolve register_backend from the shipped runtime rather than carry its
# own copy, because a private copy would register into a registry nothing queries. Undefined is
# the correct state for that symbol here, so assert it on the real artifact.
#
# The same check compares the delegate's C++ symbol-version floor against libexecutorch.so's, and
# confirms libstdc++ is a dynamic dependency whose version ceiling the paired runtime already
# satisfies. The newer C++ helpers come from libstdc++_nonshared.a (see the link libraries above),
# so the delegate needs only the old, stable symbols the system libstdc++ provides.
# eu-readelf too: the guard parses all three dialects, so it should be able to find all
# three. Its output differs (bare RPATH, UNDEF for UND), which is why the guard accepts both.
find_program(TORCH_TENSORRT_READELF NAMES readelf llvm-readelf eu-readelf)
if(NOT TORCH_TENSORRT_READELF AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR
    "readelf is required to verify the delegate imports the ExecuTorch runtime. Install "
    "binutils, or set TORCH_TENSORRT_READELF to a readelf.")
endif()
if(TORCH_TENSORRT_READELF)
  add_custom_command(TARGET executorch_backend_tensorrt POST_BUILD
    # Invocation asserted by test_the_guard_is_wired_into_the_build: a POST_BUILD command that
    # runs `true` instead of the guard leaves every source-text assertion green.
    COMMAND sh
      "${CMAKE_CURRENT_LIST_DIR}/check_imports_executorch_runtime.sh"
      "${TORCH_TENSORRT_READELF}"
      "$<TARGET_FILE:executorch_backend_tensorrt>"
      "$<TARGET_FILE:executorch::runtime>"
      # The same string patchelf applies above, so the guard compares the whole set against what
      # the build asked for rather than spot-checking one entry it restates. Checking only
      # $ORIGIN/../../executorch/lib accepted a delegate with the tensorrt_libs or the CUDA entry
      # missing, and the loader cannot substitute for this check: any runner with a CUDA toolkit
      # installed resolves libcudart out of ld.so.cache whatever the RUNPATH says.
      "${TORCH_TENSORRT_DELEGATE_RUNPATH_COLONS}"
      "${TORCH_TENSORRT_MANYLINUX_TAG}"
    VERBATIM)
endif()
