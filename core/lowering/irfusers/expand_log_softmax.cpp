#include "torch/csrc/jit/passes/fuse_linear.h"
#include "torch/csrc/jit/passes/subgraph_rewrite.h"

namespace trtorch {
namespace core {
namespace lowering {
namespace irfusers {

void ExpandLogSoftmax(std::shared_ptr<torch::jit::Graph>& graph) {
    // Its easier for TensorRT if we seperate softmax and log
    // There might need to be a reshape inserted see:
    // https://github.com/onnx/onnx-tensorrt/blob/5dca8737851118f6ab8a33ea1f7bcb7c9f06caf5/builtin_op_importers.cpp#L1593
    // Should the reshapes be added here or in the converter?
    
    // TODO: In the future this should be removed for a deicated log_softmax converter (more efficent)
    // But its easier to stand up a working system if the number of op converters is lower
    std::string logsoftmax_pattern = R"IR(
        graph(%input, %dim, %dtype):
            %log_softmax = aten::log_softmax(%input, %dim, %dtype)
            return (%log_softmax))IR";
    std::string softmax_log_pattern = R"IR(
        graph(%input, %dim, %dtype):
            %softmax = aten::softmax(%input, %dim, %dtype)
            %log_softmax = aten::log(%softmax)
            return (%log_softmax))IR";
    std::string logsoftmax_none_pattern = R"IR(
        graph(%input, %dim):
            %dtype : int? = prim::Constant()
            %log_softmax = aten::log_softmax(%input, %dim, %dtype)
            return (%log_softmax))IR";
    std::string softmax_log_none_pattern = R"IR(
        graph(%input, %dim):
            %dtype : int? = prim::Constant()
            %softmax = aten::softmax(%input, %dim, %dtype)
            %log_softmax = aten::log(%softmax)
            return (%log_softmax))IR";   

    torch::jit::SubgraphRewriter logsoftmax_to_softmax_log;
    logsoftmax_to_softmax_log.RegisterRewritePattern(logsoftmax_pattern, softmax_log_pattern);
    logsoftmax_to_softmax_log.runOnGraph(graph);
    
    torch::jit::SubgraphRewriter logsoftmax_none_to_softmax_log_none;
    logsoftmax_none_to_softmax_log_none.RegisterRewritePattern(
        logsoftmax_none_pattern, softmax_log_none_pattern);
    logsoftmax_none_to_softmax_log_none.runOnGraph(graph);
}

} // namespace irfusers
} // namespace lowering
} // namespace core
} // namespace trtorch
