import torch
import torchvision
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tqdm 
from datetime import datetime


from models import BasicNetwork, TransferNetworkVGG, ResnetNetwork
from dataload import CustomData

#paths
TRAIN_PATH = '//Users/utkucicek/Desktop/hand_gesture_based_remote_car_controller/train'
TEST_PATH = '/Users/utkucicek/Desktop/hand_gesture_based_remote_car_controller/test'

#Hyper Paramters
NUM_EPOCHS =10
LR = 0.001
BATCH_SIZE = 4

device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')

transfer_net = TransferNetworkVGG().to(device)
net = BasicNetwork().to(device)
network = ResnetNetwork().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LR)
step = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

train_data = CustomData(TRAIN_PATH)
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)

test_data = CustomData(TEST_PATH)
test_loader = DataLoader(test_data, batch_size=4, shuffle=False)


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.0
    last_loss = 0.0
    acc = 0.0

    for i,data in enumerate(train_loader):

        inputs,labels = data
        
        # Zero your gradients for every batch
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = net(inputs)
        # print('outputs:',outputs)
        # Compute the loss and its gradients
        loss = criterion(outputs,labels)
        # print("labels",labels)
        loss.backward()
        
        optimizer.step()

        running_loss+=loss.item()

        if i %25 ==24:
            last_loss = running_loss / 25 # loss per batch
            print('batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(test_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter()
epoch_number = 0
running_corrects = 0
best_vloss = 1_000_000.
for epoch in range(5):
    print('Epoch {}'.format(epoch_number+1))
    
    net.train(True)
    avg_loss = train_one_epoch(epoch_number,writer)
    net.train(False)

    running_vloss=0.0
    running_corrects = 0.0
    for i, vdata in enumerate(test_loader):
        vinputs, vlabels = vdata
        voutputs = net(vinputs)
        _,preds = torch.max(voutputs,1)
        running_corrects += torch.sum(preds == vlabels)
        vloss = criterion(voutputs, vlabels)
        running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    epoch_acc = (running_corrects.double() / (4*(i + 1)))*100
    print('LOSS: train {} valid {}'.format(avg_loss, avg_vloss))
    print(f"Valdation accuracy: {epoch_acc}")
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(net.state_dict(), model_path)

    epoch_number += 1

