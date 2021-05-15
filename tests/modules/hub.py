import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

models = {
    "alexnet": {
        "model": models.alexnet(pretrained=True),
        "path": "both"
    },
    "vgg16": {
        "model": models.vgg16(pretrained=True),
        "path": "both"
    },
    "squeezenet": {
        "model": models.squeezenet1_0(pretrained=True),
        "path": "both"
    },
    "densenet": {
        "model": models.densenet161(pretrained=True),
        "path": "both"
    },
    "inception_v3": {
        "model": models.inception_v3(pretrained=True),
        "path": "both"
    },
    #"googlenet": models.googlenet(pretrained=True),
    "shufflenet": {
        "model": models.shufflenet_v2_x1_0(pretrained=True),
        "path": "both"
    },
    "mobilenet_v2": {
        "model": models.mobilenet_v2(pretrained=True),
        "path": "both"
    },
    "resnext50_32x4d": {
        "model": models.resnext50_32x4d(pretrained=True),
        "path": "both"
    },
    "wideresnet50_2": {
        "model": models.wide_resnet50_2(pretrained=True),
        "path": "both"
    },
    "mnasnet": {
        "model": models.mnasnet1_0(pretrained=True),
        "path": "both"
    },
    "resnet18": {
        "model": torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True),
        "path": "both"
    },
    "resnet50": {
        "model": torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True),
        "path": "both"
    },
    "fcn_resnet101": {
        "model": torch.hub.load('pytorch/vision:v0.9.0', 'fcn_resnet101', pretrained=True),
        "path": "script"
    },
    "ssd": {
        "model": torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math="fp32"),
        "path": "trace"
    },
    "faster_rcnn": {
        "model": models.detection.fasterrcnn_resnet50_fpn(pretrained=True),
        "path": "script"
    }
}

# Download sample models
for n, m in models.items():
    print("Downloading {}".format(n))
    m["model"] = m["model"].eval().cuda()
    x = torch.ones((1, 3, 300, 300)).cuda()
    if m["path"] == "both" or m["path"] == "trace":
        trace_model = torch.jit.trace(m["model"], [x])
        torch.jit.save(trace_model, n + '_traced.jit.pt')
    if m["path"] == "both" or m["path"] == "script":
        script_model = torch.jit.script(m["model"])
        torch.jit.save(script_model, n + '_scripted.jit.pt')


# Sample Pool Model (for testing plugin serialization)
class Pool(nn.Module):

    def __init__(self):
        super(Pool, self).__init__()

    def forward(self, x):
        return F.adaptive_avg_pool2d(x, (5, 5))


model = Pool().eval().cuda()
x = torch.ones([1, 3, 10, 10]).cuda()

trace_model = torch.jit.trace(model, x)
torch.jit.save(trace_model, "pooling_traced.jit.pt")


# Sample Conditional Model (for testing partitioning and fallback in conditionals)
class FallbackIf(torch.nn.Module):
    def __init__(self):
        super(FallbackIf, self).__init__()
        self.relu1 = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1)
        self.log_sig = torch.nn.LogSigmoid()
        self.conv2 = torch.nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(32, 3, 3, 1, 1)

    def forward(self, x):
        x = self.relu1(x)
        x_first = x[0][0][0][0].item()
        if x_first > 0:
            x = self.conv1(x)
            x1 = self.log_sig(x)
            x2 = self.conv2(x)
            x = self.conv3(x1 + x2)
        else:
            x = self.log_sig(x)
        x = self.conv1(x)
        return x

conditional_model = FallbackIf().eval().cuda()
conditional_script_model = torch.jit.script(conditional_model)
torch.jit.save(conditional_script_model, "conditional_scripted.jit.pt")