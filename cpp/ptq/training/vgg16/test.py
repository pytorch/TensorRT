import argparse
import os
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.tensorboard import SummaryWriter

from vgg16 import vgg16

model = vgg16(num_classes=10, init_weights=False)

model.forward(torch.rand([1,3,224,224]))