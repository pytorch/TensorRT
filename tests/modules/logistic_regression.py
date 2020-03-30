import torch
from torch.nn import functional as F

class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
    def forward(self, x):
        y_pred = torch.tanh(self.linear(x))
        return y_pred

model = LogisticRegression()
jit_module = torch.jit.trace(model, torch.tensor([1.0]))

jit_module.save("logistic_regression.jit.pt")
