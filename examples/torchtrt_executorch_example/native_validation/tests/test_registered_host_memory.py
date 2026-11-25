# SPDX-License-Identifier: BSD-3-Clause
"""Compile the registration helper with a fake CUDA API to test cleanup on CPU."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(shutil.which("g++"), "Requires g++")
class RegistrationTests(unittest.TestCase):
    def test_page_overlap_cleanup_failure_and_unsupported_device(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cuda_runtime.h").write_text(r"""
#pragma once
#include <cstddef>
using cudaError_t = int;
constexpr int cudaSuccess = 0, cudaDevAttrHostRegisterSupported = 0, cudaHostRegisterDefault = 0;
inline int calls=0, releases=0, fences=0, fail_on=0, supported=1;
inline size_t bytes=0;
inline int cudaGetDevice(int* d) { *d=0; return 0; }
inline int cudaDeviceGetAttribute(int* v,int,int) { *v=supported; return 0; }
inline int cudaHostRegister(void*,size_t n,int) { ++calls; bytes=n; return calls==fail_on ? 1:0; }
inline int cudaHostUnregister(void*) { ++releases; return 0; }
inline int cudaDeviceSynchronize() { ++fences; return 0; }
inline const char* cudaGetErrorString(int) { return "test error"; }
""")
            (root / "test.cpp").write_text(r"""
#include "registered_host_memory.h"
#include <cassert>
int main() {
  const uintptr_t page=sysconf(_SC_PAGESIZE);
  auto ptr=[&](uintptr_t p){return reinterpret_cast<void*>(p);};
  {
    RegisteredHostMemory memory({{ptr(10*page+1), 8}, {ptr(10*page+20), 8}, {ptr(11*page+1), 8}});
    assert(calls==1 && bytes==2*page && releases==0);
  }
  assert(releases==1 && fences==1);
  calls=releases=0; fail_on=2;
  try {
    RegisteredHostMemory memory({{ptr(10*page), 8},{ptr(20*page),8}});
    assert(false);
  } catch (const std::runtime_error&) {}
  assert(calls==2 && releases==1);
  calls=releases=0; supported=0;
  try { RegisteredHostMemory memory({{ptr(10*page),8}}); assert(false); }
  catch (const std::runtime_error&) {}
  assert(calls==0 && releases==0);
}
""")
            binary = root / "test"
            subprocess.run(
                [
                    "g++",
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-I",
                    str(root),
                    "-I",
                    str(Path(__file__).resolve().parents[1] / "optimizations"),
                    str(root / "test.cpp"),
                    "-o",
                    str(binary),
                ],
                check=True,
            )
            subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main()
