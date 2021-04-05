import torch
import torch.nn as nn

# Create a sample network with a conv and gelu node.
# Gelu layer in TRTorch is converted to CustomGeluPluginDynamic from TensorRT plugin registry.
class ConvGelu(torch.nn.Module):
    def __init__(self):
        super(ConvGelu, self).__init__()
        self.conv = nn.Conv2d(3, 32, 3, 1)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.gelu(x)
        return x

def main():

    model = ConvGelu().eval().cuda()
    scripted_model = torch.jit.script(model)
    # Save the torchscript model
    torch.jit.save(scripted_model, 'conv_gelu.jit')
    print("Generated conv_gelu.jit model.")

if __name__ == "__main__":
    main()
