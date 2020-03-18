#include <utility>
#include "torch/script.h"
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "trtorch/trtorch.h"
#include "c10/cuda/CUDACachingAllocator.h"

using PathAndInSize = std::pair<std::string, std::vector<std::vector<int64_t>>>;

class ModuleTests
    : public testing::TestWithParam<PathAndInSize> {
public:
    void SetUp() override {
        auto params = GetParam();
        auto path = params.first;
        try {
            // Deserialize the ScriptModule from a file using torch::jit::load().
            mod = torch::jit::load(path);
        }
        catch (const c10::Error& e) {
            std::cerr << "error loading the model\n";
            return;
        }
        input_shapes = params.second;
    }

    void TearDown() {
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
protected:
    torch::jit::script::Module mod;
    std::vector<std::vector<int64_t>> input_shapes;
};
