from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.models as models
import shutil 
import warnings
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import argparse
import torchvision
from torch.autograd import Variable
from tqdm import tqdm
import array
import matplotlib.pyplot as plt
from PIL import Image
# Ignore warnings
warnings.filterwarnings("ignore")
plt.ion()   # interactive mode

path_to_ddsm = "/home/harsh/project/dog/"

# model_conv = torch.load('/home/harsh/project/code/model.pt')
model_conv = models.resnet18(pretrained=True)
print("My model - ", model_conv)

data_dir = path_to_ddsm
input_shape = 224
batch_size = 1
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
scale = (224,224)
use_parallel = True
use_gpu = False
epochs = 100
data_transforms = {
        'train': transforms.Compose([
        transforms.Resize(scale),
        transforms.RandomResizedCrop(input_shape),
        #transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        #transforms.RandomRotation(degrees=90),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]),
        'val': transforms.Compose([
        transforms.Resize(scale),
        transforms.CenterCrop(input_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]),}
print(data_transforms)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    img = img.cpu()
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

print(os.path.join(data_dir, 'val'))    
    
image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x),
                                      data_transforms[x]) for x in ['val']}
print(image_datasets)
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                         shuffle=True, num_workers=3) for x in [ 'val']}
print("Hello")
dataset_sizes = {x: len(image_datasets[x]) for x in ['val']}
print(dataset_sizes)
dataiter = iter(dataloaders['val'])
images, labels= next(dataiter)
# images = torch.zeros([1,3,224,224])
# show images
print(images.shape)
imshow(torchvision.utils.make_grid(images))
print(labels)
criterion = nn.CrossEntropyLoss()
#optimizer_conv = optim.SGD(list(filter(lambda p: p.requires_grad, model_conv.parameters())), lr=0.001, momentum=0.9)
for i in range(40):
    images, labels = Variable(images.cuda(), requires_grad = True), Variable(labels.cuda())
    outputs = model_conv(images)
    if type(outputs) == tuple:
        outputs, _ = outputs
    _, preds = torch.max(outputs.data, 1)
    
#     for i in range(outputs.shape[1]):
#         if i == preds.data:
#             outputs[0][i] = 1
#         else:
#             outputs[0][i] = 0
            
    print(outputs)
#     c = torch.zeros([3])
#     c[preds.data] = 1
#     print(c)
#     #optimizer_conv.zero_grad()
#     loss = criterion(outputs, labels)
#     print(loss)
    outputs[0][preds.data].backward()
    print("Before")
    images.grad = F.normalize(images.grad, p=2, dim=1)
    print(images) 
    images = images +  (images.grad)
#     print(images)
    #inputs = inputs.detach().numpy()
    #print("The backward prop is",images.shape)
    imshow(torchvision.utils.make_grid(images))