// Verify that //toolchains/torch_nccl:defs.bzl correctly detects USE_C10D_NCCL.
//
// This test is compiled with copts = if_torch_nccl(["-DUSE_C10D_NCCL"]).
// When USE_C10D_NCCL is defined, it includes ProcessGroupNCCL.hpp and
// references NCCL_BACKEND_NAME, which forces the linker to resolve symbols
// from libtorch_cuda.so.  If torch_nccl_detect gave a false positive
// (defined USE_C10D_NCCL when NCCL wasn't compiled in), the link fails.

#include <gtest/gtest.h>

#ifdef USE_C10D_NCCL
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#endif

TEST(TorchNcclDetect, MacroMatchesLibrary) {
#ifdef USE_C10D_NCCL
  // NCCL_BACKEND_NAME is defined inside the #ifdef USE_C10D_NCCL guard in
  // ProcessGroupNCCL.hpp.  If USE_C10D_NCCL was wrongly set, this header
  // would either fail to compile or the link would fail on NCCL symbols.
  EXPECT_STREQ(c10d::NCCL_BACKEND_NAME, "nccl");
#else
  GTEST_SKIP() << "Built without USE_C10D_NCCL; nothing to check";
#endif
}
