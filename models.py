import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
# from dataload import CustomData
from torch.utils.data import DataLoader
from torchvision.models import vgg16,resnet50
import cv2


class BasicNetwork(nn.Module):
    def __init__(self):
        super(BasicNetwork,self).__init__()
        #(W-F+2P)/S + 1
        self.conv1 = nn.Conv2d(1,64,2)
        self.maxpool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(64,128,5)
        self.conv3 = nn.Conv2d(128,256,5)
        self.conv4 = nn.Conv2d(256,256,5)
        self.lin1 = nn.Linear(256*10*10,128)
        self.lin2 = nn.Linear(128,64)
        self.lin3 = nn.Linear(64,4)
        self.lin4 = nn.Linear(128,128)
    def forward(self,x):
        x = self.maxpool1(F.relu(self.conv1(x)))
        x = self.maxpool1(F.relu(self.conv2(x)))
        x = self.maxpool1(F.relu(self.conv3(x)))
        x = x.view(-1,256*10*10)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin4(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x

class TransferNetworkVGG(nn.Module):
    def __init__(self):
        super(TransferNetworkVGG,self).__init__()
        self.model = vgg16(True)
        self.model.fc = torch.nn.Linear(2048,4)
    def forward(self,x):
        return self.model(x)



class ResnetNetwork(nn.Module):
    def __init__(self):
        super(ResnetNetwork,self).__init__()
        self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        self.model.fc = torch.nn.Linear(2048,4)
    def forward(self,x):
        return self.model(x)



