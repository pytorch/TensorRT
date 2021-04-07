import torch
import pdb
import trtorch

class FallbackBase(torch.nn.Module):
    def __init__(self):
        super(FallbackBase, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(32, 16, 3, 1, 1)
        self.relu1 = torch.nn.ReLU()
        self.log_sig = torch.nn.LogSigmoid()
        self.conv3 = torch.nn.Conv2d(16, 8, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.log_sig(x)
        x = self.conv3(x)
        return x

class FallbackEdge(torch.nn.Module):
    def __init__(self):
        super(FallbackEdge, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1)
        self.log_sig = torch.nn.LogSigmoid()
        self.conv2 = torch.nn.Conv2d(32, 16, 3, 1, 1)
        self.relu = torch.nn.ReLU()
        self.pooling = torch.nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.log_sig(x)
        x1 = self.conv2(x1)
        x2 = self.conv2(x)
        x2 = self.relu(x2)
        x = x1 + x2
        x = self.pooling(x)
        return x

def main():
    model1 = FallbackBase().eval().cuda()

    scripted_model1 = torch.jit.script(model1)
    torch.jit.save(scripted_model1, 'test_base_model.jit')

    model2 = FallbackEdge().eval().cuda()
    scripted_model2 = torch.jit.script(model2)
    torch.jit.save(scripted_model2, 'test_edge_model.jit')

if __name__ == "__main__":
    main()
