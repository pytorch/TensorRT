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
        "model": torch.hub.load('pytorch/vision:v0.8.2', 'resnet18', pretrained=True),
        "path": "both"
    },
    "resnet50": {
        "model": torch.hub.load('pytorch/vision:v0.8.2', 'resnet50', pretrained=True),
        "path": "both"
    },
    "fcn_resnet101": {
        "model": torch.hub.load('pytorch/vision:v0.8.2', 'fcn_resnet101', pretrained=True),
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
