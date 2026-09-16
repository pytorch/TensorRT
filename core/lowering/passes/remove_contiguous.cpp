/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <torch/csrc/jit/passes/subgraph_rewrite.h>

#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace lowering {
namespace passes {

void RemoveContiguous(std::shared_ptr<torch::jit::Graph>& graph) {
  std::string contiguous_pattern = R"IR(
        graph(%input, %1):
            %2 = aten::contiguous(%input, %1)
            return (%2))IR";
  std::string no_contiguous_pattern = R"IR(
        graph(%input, %1):
            return (%input))IR";

  // remove contiguous
  torch::jit::SubgraphRewriter remove_contiguous;
  remove_contiguous.RegisterRewritePattern(contiguous_pattern, no_contiguous_pattern);
  remove_contiguous.runOnGraph(graph);
  LOG_GRAPH("Post remove contiguous: " << *graph);
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace torch_tensorrt
