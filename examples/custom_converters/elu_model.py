import torch
import trtorch

# After "python3 setup install", you should find this .so file under generated "build" directory
torch.ops.load_library('./elu_converter/build/lib.linux-x86_64-3.6/elu_converter.cpython-36m-x86_64-linux-gnu.so')


class Elu(torch.nn.Module):

    def __init__(self):
        super(Elu, self).__init__()
        self.elu = torch.nn.ELU()

    def forward(self, x):
        return self.elu(x)


def MaxDiff(pytorch_out, trtorch_out):
    diff = torch.sub(pytorch_out, trtorch_out)
    abs_diff = torch.abs(diff)
    max_diff = torch.max(abs_diff)
    print("Maximum differnce between TRTorch and PyTorch: \n", max_diff)


def main():
    model = Elu().eval()  #.cuda()

    scripted_model = torch.jit.script(model)
    compile_settings = {
        "input_shapes": [{
            "min": [1024, 1, 32, 32],
            "opt": [1024, 1, 33, 33],
            "max": [1024, 1, 34, 34],
        }],
        "op_precision":
            torch.half  # Run with FP16
    }
    trt_ts_module = trtorch.compile(scripted_model, compile_settings)
    input_data = torch.randn((1024, 1, 32, 32))
    input_data = input_data.half().to("cuda")
    pytorch_out = model.forward(input_data)

    trtorch_out = trt_ts_module(input_data)
    print('PyTorch output: \n', pytorch_out[0, :, :, 0])
    print('TRTorch output: \n', trtorch_out[0, :, :, 0])
    MaxDiff(pytorch_out, trtorch_out)


if __name__ == "__main__":
    main()
