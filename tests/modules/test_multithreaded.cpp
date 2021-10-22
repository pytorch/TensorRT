#include <torch/script.h>
#include <trtorch/trtorch.h>
using namespace std;

void compile(bool fp16) {
  constexpr int64_t INPUT_CHANNEL_NUM = 256;
  constexpr int64_t WIDTH = 32;
  torch::jit::Module module = torch::jit::load("model.ts");
  if (fp16) {
    module.to(torch::kCUDA, torch::kHalf);
  } else {
    module.to(torch::kCUDA);
  }
  module.eval();

  std::vector<int64_t> in_sizes = {1, INPUT_CHANNEL_NUM, WIDTH, WIDTH};
  trtorch::CompileSpec::InputRange range(in_sizes);
  trtorch::CompileSpec info({range});
  if (fp16) {
    info.op_precision = torch::kHalf;
  }
  module = trtorch::CompileGraph(module, info);
}

int main() {
  while (true) {
    // fp32, this thread -> OK
    compile(false);
    cout << "fp32, this thread -> finish" << endl;

    // fp32, another thread -> OK
    std::thread thread0([]() { compile(false); });
    thread0.join();
    cout << "fp32, another thread -> finish" << endl;

    // fp16, this thread -> OK
    compile(true);
    cout << "fp16, this thread -> finish" << endl;

    // fp16, another thread -> NG
    std::thread thread1([]() { compile(true); });
    thread1.join();
    cout << "fp16, another thread -> finish" << endl;
  }
}