import torch
from torch import nn

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(1, 5),
            nn.Linear(5, 5),
            nn.Linear(5, 5),
            nn.Linear(5, 5))
    def forward(self, x):
        return self.backbone(x)

m = SimpleNet().eval().to("cuda")
m = torch.jit.script(m)
torch.jit.save(m, "simplenet.jit.pt")
