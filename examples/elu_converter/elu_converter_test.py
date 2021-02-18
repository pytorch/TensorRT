import torch
import trtorch

torch.ops.load_library('./build/lib.linux-x86_64-3.6/elu_converter.cpython-36m-x86_64-linux-gnu.so')


class Elu(torch.nn.Module):

    def __init__(self):
        super(Elu, self).__init__()
        self.elu = torch.nn.ELU()

    def forward(self, x):
        return self.elu(x)


def main():
    data = torch.randn((1, 1, 2, 2)).to("cuda")
    model = Elu().eval()  #.cuda()

    # traced_model = torch.jit.trace(model, [data])
    scripted_model = torch.jit.script(model)
    print(scripted_model.graph)
    # torch.jit.save(scripted_model, 'elu.jit')
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
    print(input_data[0, :, :, 0])
    input_data = input_data.half().to("cuda")
    result = trt_ts_module(input_data)
    print(result[0, :, :, 0])
    # torch.jit.save(trt_ts_module, "trt_ts_module.ts")


if __name__ == "__main__":
    main()
