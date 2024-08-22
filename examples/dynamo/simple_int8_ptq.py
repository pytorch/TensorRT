import modelopt.torch.quantization as mtq
import torch
import torch_tensorrt as torchtrt
from modelopt.torch.quantization.utils import export_torch_mode


class SimpleNetwork(torch.nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()
        self.linear1 = torch.nn.Linear(in_features=6, out_features=3)
        self.linear2 = torch.nn.Linear(in_features=3, out_features=1)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.nn.ReLU()(x)
        x = self.linear2(x)
        return x


def calibrate_loop(model):
    """Simple calibration function for testing."""
    model(input_tensor)


input_tensor = torch.randn(1, 6).cuda()
model = SimpleNetwork().eval().cuda()
print(f"model before quantize: {model}")

quant_cfg = mtq.INT8_DEFAULT_CFG
mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
print(f"model after quantize: {model}")

# int8 torchtrt output
with torch.no_grad():
    with export_torch_mode():
        # torch.export.export() failed due to RuntimeError: Attempting to use FunctionalTensor on its own. Instead, please use it with a corresponding FunctionalTensorMode()
        exp_program = torch.export._trace._export(model, (input_tensor,))
        print(f"{exp_program=}")
        trt_model = torchtrt.dynamo.compile(
            exp_program,
            inputs=[input_tensor],
            enabled_precisions={torch.int8},
            min_block_size=1,
            debug=True,
        )
        output_int8_trt = trt_model(input_tensor)
