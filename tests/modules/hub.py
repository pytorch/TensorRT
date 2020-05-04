import torch
import torchvision.models as models

models = {
          "alexnet": models.alexnet(pretrained=True),
          "vgg16": models.vgg16(pretrained=True),
          "squeezenet": models.squeezenet1_0(pretrained=True),
          "densenet": models.densenet161(pretrained=True),
          "inception_v3": models.inception_v3(pretrained=True),
          #"googlenet": models.googlenet(pretrained=True),
          "shufflenet": models.shufflenet_v2_x1_0(pretrained=True),
          "mobilenet_v2": models.mobilenet_v2(pretrained=True),
          "resnext50_32x4d": models.resnext50_32x4d(pretrained=True),
          "wideresnet50_2": models.wide_resnet50_2(pretrained=True),
          "mnasnet": models.mnasnet1_0(pretrained=True),
          "resnet18": torch.hub.load('pytorch/vision:v0.5.0', 'resnet18', pretrained=True),
          "resnet50": torch.hub.load('pytorch/vision:v0.5.0', 'resnet50', pretrained=True)}

for n, m in models.items():
    print("Downloading {}".format(n))
    m = m.eval().cuda()
    x = torch.ones((1, 3, 224, 224)).cuda()
    trace_model = torch.jit.trace(m, x)
    torch.jit.save(trace_model, n + '_traced.jit.pt')
    script_model = torch.jit.script(m)
    torch.jit.save(script_model, n + '_scripted.jit.pt')