import torch
import torchvision
import torchvision.transforms as TF
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
from torchvision.transforms import ToTensor

class CustomData(Dataset):
    def __init__(self,folder_dir):
        super(CustomData,self).__init__()
        self.folder_dir = folder_dir
        self.images = os.listdir(folder_dir)

    def __len__(self):
        return len(self.images)

    def load_data(self,folder, name, size =(112, 112)):
        '''
        Loads the image and returns the label
        '''
        img = Image.open(str(folder)+"/"+str(name))
        img = np.array(img)
        img = Image.fromarray(img).resize(size)
        img = np.array(img)
        label = int(name[-5])
        return img, label

    def transforms(self,image):
        '''
        Augmentation operations/preprocessing if necessary
        '''
        img = torch.tensor(image, dtype=torch.float32)
        img = torch.reshape(img, (1, 112, 112))
        img = torch.repeat_interleave(img, repeats=3, dim=0)
        return img

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image,label = self.load_data(name = str(img_path), folder= self.folder_dir)
        image = self.transforms(image)
        return image, label





