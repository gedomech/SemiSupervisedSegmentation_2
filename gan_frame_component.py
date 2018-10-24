# coding=utf-8
import os
import sys
import warnings

import torch.nn as nn
import torchvision.models as models

sys.path.insert(-1, os.getcwd())
warnings.filterwarnings('ignore')

# here we use resnet for a discriminator from torchvision.
Cat_Discriminator = models.resnet18(pretrained=True)
Cat_Discriminator.fc = nn.Linear(512, 4)
criterion = nn.CrossEntropyLoss()


def cat_adverserial_training(input, cat):
    result = Cat_Discriminator(input)
    loss = criterion(result, cat)
    return loss


'''
attention based fusion technology.
'''
from myutils.myNetworks import UNet


class Attention_based_fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet(2)

    def forward(self, input):
        return self.unet(input)


class binary_discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.vgg = models.vgg19(True)
        self.vgg.classifier[6] = nn.Linear(4096, 2)


d2 = binary_discriminator()
