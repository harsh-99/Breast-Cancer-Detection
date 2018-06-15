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

path_to_ddsm = "/home/harsh/project/DDSM/"
no_image_val = 3

#It is used to delete file with different different extension present in the directory once you convert the LJPEG jpg format 
def delete_file_with_ext(cur_dir, old_ext, sub_dirs=False):                     
    if sub_dirs:
        for root, dirs, files in os.walk(cur_dir):
            for filename in files:
                file_ext = os.path.splitext(filename)[1]
                if old_ext == file_ext:
                    oldname = os.path.join(root, filename)
                    os.remove(oldname)
    else:
        os.remove(cur_dir)



for root, subFolders, file_names in os.walk(path_to_ddsm):
    for file_name in file_names:
        if ".LJPEG" in file_name:
            ljpeg_path = os.path.join(root, file_name)
            delete_path = os.path.join(root, file_name)
            #print(delete_path)
            out_path = delete_path.split('.LJPEG')[0] + ".jpg"
            
            cmd = './ljpeg.py "{0}" "{1}" --visual --scale 1.0'.format(ljpeg_path, out_path)
            os.system(cmd)
            delete_file_with_ext(delete_path, '.LJPEG', False)

print('done')

delete_file_with_ext(path_to_ddsm, '.OVERLAY', True)
delete_file_with_ext(path_to_ddsm, '.ics', True)
delete_file_with_ext(path_to_ddsm, '.16_PGM', True)

# These functions i.e check_word_type, flip_image, flip_cut_paste cut the images from the seqential folder of benigns, cancers and 
#normal and paste it to the parent directory i.e. benign, cancers and normal and then delete the sequential folder and then rename the images
#given in sequence, also it flips the LEFT image, so basically all are right after implementing this function

def check_word_type(filename):
    words = "LEFT"
    if words in filename: 
        return True
    
def flip_image(image_path, saved_location):
    """
    image_path: The path to the image to edit
    saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    rotated_image = image_obj.transpose(Image.FLIP_LEFT_RIGHT)
    rotated_image.save(saved_location)

def flip_cut_paste(paste_dir):
    for root, dirs, files in os.walk(paste_dir):
        for filename in files:
            #print(os.path.basename(filename))
            a = check_word_type(os.path.basename(filename))
            if a: 
                image = (os.path.join(root,os.path.basename(filename)))
                img = flip_image(image, image);
            shutil.move(os.path.join(root, os.path.basename(filename)), paste_dir)
    for root, dirs, files in os.walk(paste_dir):
        for directory in dirs:
            #print(directory)
            shutil.rmtree(os.path.join(root, directory))
    files = os.listdir(paste_dir)
    i = 1
    for file in files:
        os.rename(os.path.join(paste_dir, file), os.path.join(paste_dir, str(i)+'.jpg'))
        i = i+1

flip_cut_paste(os.path.join(path_to_ddsm, 'benigns'))
flip_cut_paste(os.path.join(path_to_ddsm, 'cancers'))
flip_cut_paste(os.path.join(path_to_ddsm, 'normal'))

train_dir = os.path.join(path_to_ddsm,'train')
train_benigns_dir = os.path.join(train_dir,'benigns')
train_cancers_dir = os.path.join(train_dir,'cancers')
train_normal_dir = os.path.join(train_dir,'normal')
val_dir = os.path.join(path_to_ddsm,'val')
val_benigns_dir = os.path.join(val_dir,'benigns') 
val_cancers_dir = os.path.join(val_dir,'cancers') 
val_normal_dir =  os.path.join(val_dir,'normal') 
os.makedirs(train_dir)
os.makedirs(train_benigns_dir)
os.makedirs(train_cancers_dir)
os.makedirs(train_normal_dir)
os.makedirs(val_dir)
os.makedirs(val_benigns_dir)
os.makedirs(val_cancers_dir)
os.makedirs(val_normal_dir)
cut_paste = os.path.join(path_to_ddsm, 'benigns')
for root, dirs, files in os.walk(cut_paste):
    i = 0
    for filename in files:
        if i < no_image_val:
            shutil.move(os.path.join(cut_paste,os.path.basename(filename)),val_benigns_dir)
        else:
            shutil.move(os.path.join(cut_paste,os.path.basename(filename)),train_benigns_dir)
        i = i+1
shutil.rmtree(cut_paste)
cut_paste = os.path.join(path_to_ddsm, 'cancers')
for root, dirs, files in os.walk(cut_paste):
    i = 0
    for filename in files:
        if i < no_image_val:
            shutil.move(os.path.join(cut_paste,os.path.basename(filename)),val_cancers_dir)
        else:
            shutil.move(os.path.join(cut_paste,os.path.basename(filename)),train_cancers_dir)
        i = i+1
shutil.rmtree(cut_paste)
cut_paste = os.path.join(path_to_ddsm, 'normal')
for root, dirs, files in os.walk(cut_paste):
    i = 0
    for filename in files:
        if i < no_image_val:
            shutil.move(os.path.join(cut_paste,os.path.basename(filename)),val_normal_dir)
        else:
            shutil.move(os.path.join(cut_paste,os.path.basename(filename)),train_normal_dir)
        i = i+1
shutil.rmtree(cut_paste)


# to change the out feature of the model to 3
num_ftrs = model_conv.fc.in_features   #for resnet18
# num_ftrs = model_conv.classifier._modules['6'].in_features    #for VGG16_BN
print (num_ftrs)
# model_conv.classifier._modules['6'] = nn.Linear(num_ftrs, 3)    #for VGG16_BN
model_conv.fc = nn.Linear(num_ftrs, 3)     #for resnet18
print ("the value of output feature is", model_conv.fc.out_features)       #for resnet18   
# print ("The value of output feature is ",model_conv.classifier._modules['6'].out_features)        #for VGG16_BN

data_dir = path_to_ddsm
input_shape = 224
batch_size = 64
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
scale = (224,224)
use_parallel = True
use_gpu = True
epochs = 50
data_transforms = {
        'train': transforms.Compose([
        transforms.Resize(scale),
        transforms.RandomResizedCrop(input_shape),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=90),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]),
        'val': transforms.Compose([
        transforms.Resize(scale),
        transforms.CenterCrop(input_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]),}

print("The transformation of data is as followed ",data_transforms)

# the function to train the model 
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, use_gpu, num_epochs=25, mixup = False, alpha = 0.1):
    print("MIXUP".format(mixup))
    since = time.time()
    global l
    global e 
    global a 
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.
            for data in tqdm(dataloaders[phase]):
                # get the inputs
                inputs, labels = data

                #augementation using mixup
                if phase == 'train' and mixup:
                    inputs = mixup_batch(inputs, alpha)
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                if type(outputs) == tuple:
                    outputs, _ = outputs
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
            #print("the running loss is")
            #print(running_loss)
            #print("the running corrects is")
            #print(running_corrects)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = float(running_corrects) / float(dataset_sizes[phase])
            #print(epoch_acc)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            lo = '{:4f}'.format(epoch_loss)
            ac = '{:4f}'.format(epoch_acc)
            if phase == 'train':
                #print("the value of a is")
                #print(a)
                l.append(float(lo))
            if phase == 'val':
                a.append(float(ac))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                
                
        print()

    time_elapsed = time.time() - since
    i = 0
    for i in range(num_epochs):
        e.append(float(i))
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x),
                                      data_transforms[x]) for x in ['train', 'val']}
# print (image_datasets)
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                         shuffle=True, num_workers=3) for x in ['train', 'val']}
print("Hello")
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
print("Using CrossEntropyLoss")
criterion = nn.CrossEntropyLoss()

print("Using small learning rate with momentum")
optimizer_conv = optim.SGD(list(filter(lambda p: p.requires_grad, model_conv.parameters())), lr=0.001, momentum=0.9)

print("Creating Learning rate scheduler")
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

print("Training the model begun")
model_conv = model_conv.cuda()
l = array.array('f',[])
e = array.array('f',[])
a = array.array('f',[])
model_ft = train_model(model_conv, dataloaders, dataset_sizes, criterion, optimizer_conv, exp_lr_scheduler, True,
                     num_epochs=epochs)
plt.plot(e,l)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(a,l)
plt.xlabel('Epoch')
plt.ylabel('accuracy')

plt.show()