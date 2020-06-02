# import torch.nn as nn
# import torch

# class FeatureExtractor(nn.Module):
#     def __init__(self):
#         super(FeatureExtractor, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, 3)
#         self.conv2 = nn.Conv2d(6, 16, 3)

#     def forward(self, x):
#         x = torch.max_pool2d(torch.relu(self.conv1(x)), (2, 2))
#         x = torch.max_pool2d(torch.relu(self.conv2(x)), 2)

#         return x

# class Classifier(nn.Module):
#     def __init__(self):
#         super(Classifier, self).__init__()

#         self.fc1 = nn.Linear(16*6*6, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = torch.flatten(x, 1)
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)

#         return x

# class LeNet(nn.Module):
#     def __init__(self):
#         super(LeNet, self).__init__()
#         self.feat = FeatureExtractor()
#         self.classifier = Classifier()
    
#     def forward(self, x):
#         x = self.feat(x)
#         x = self.classifier(x)

#         return x

# model = LeNet()
# model.eval()
# traced_model = torch.jit.trace(model, torch.empty([1, 1, 32, 32]))
# torch.jit.save(traced_model, 'traced_model.ts')
# torch.jit.save(torch.jit.script(model), 'script_model.ts')


import torch.nn as nn
import torch
import torch.nn.functional as F
#import trtorch

class Interp(nn.Module):
    def __init__(self):
        super(Interp, self).__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=(5,5), mode='nearest')

model = Interp()
model.eval()
trace = torch.jit.trace(model, torch.empty([1, 1, 2, 2]))
torch.jit.save(trace, 'trace.ts')

#trtorch.check_method_op_support(trace, "forward")


