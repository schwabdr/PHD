'''
Purpose of this file is to split the Mal_image dataset into test and train, then save both sets in numpy configuration
to match the same data format for the CIFAR10 dataset.
'''

import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader


import torch.backends.cudnn as cudnn
import numpy as np

from utils import utils
from utils import config
from utils import data 
from models.resnet_new import ResNet18

import matplotlib
#matplotlib.use('tkagg') #need this line on Lambda
#matplotlib.use('gtk3')
import matplotlib.pyplot as plt

import random

#from torchinfo import summary

args = config.Configuration().getArgs()

args.batch_size = 2048 #16 was from the small GPU on paperspace
#https://stackoverflow.com/questions/10712002/create-an-empty-list-with-certain-size-in-python
classes = [None] * 25 # 25 spaces for classes

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def show_grid(x, y, name):
    print("Displaying Images ...")
    classes = ['Adialer.C', 'Agent.FYI', 'Allaple.A', 'Allaple.L', 'Alueron.gen!J', 'Autorun.K', 'C2LOP.P', 'C2LOP.gen!g', 'Dialplatform.B', 'Dontovo.A', 'Fakerean', 
                'Instantaccess', 'Lolyda.AA1', 'Lolyda.AA2', 'Lolyda.AA3', 'Lolyda.AT', 'Malex.gen!J', 'Obfuscator.AD', 'Rbot!gen', 'Skintrim.N', 'Swizzor.gen!E', 'Swizzor.gen!I', 
                'VB.AT', 'Wintrim.BX', 'Yuner.A']
    rows = 5
    cols = 5
    fig,axes1 = plt.subplots(rows,cols,figsize=(10,10))
        
    lst = list(range(0, len(x)))
    random.shuffle(lst)
    print(f"len(lst):{len(lst)}")
    #print("min/max of numpy arrays: ", np.min(x), np.max(x)) #this was very close to 0 and 1
    for j in range(rows):
        for k in range(rows):
            #get a random index
            i = lst.pop()
            axes1[j][k].set_axis_off()
            #axes1[j][k+1].set_axis_off()
            axes1[j][k].imshow(x[i],interpolation='nearest')
            axes1[j][k].text(0,0,classes[y[i]]) # this gets the point accross but needs fixing.
            #axes1[j][k+1].imshow(x_adv[i], interpolation='nearest')
            #pred_ind = pred_adv[i]
            #axes1[j][k+1].text(0,0,classes[pred_ind])
    plt.show()
    plt.savefig(name, format='png')
    print("Display Complete!")

def open_samples():
    print(f"Opening samples and printing random")
    #i'm not normalizing here since I want to display
    trans_train = transforms.Compose([
        transforms.ToTensor()
    ])

    trans_test = transforms.Compose([
        transforms.ToTensor()
    ])
    
    trainset = data.data_dataset(img_path=args.nat_img_train_mal, clean_label_path=args.nat_label_train_mal, transform=trans_train)
    testset = data.data_dataset(img_path=args.nat_img_test_mal, clean_label_path=args.nat_label_test_mal, transform=trans_test)
    
    #create data loaders
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, drop_last=False,
                                               shuffle=False, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, drop_last=False,
                                               shuffle=False, num_workers=4, pin_memory=True)
    first = True
    for dat, label in train_loader:
        x = dat.detach().cpu().numpy().transpose(0,2,3,1)
        y = label.detach().cpu().numpy()

        x = (x*255).astype(int)
        if first:
            first = False
            show_grid(x, y, "./imgs/train_imgs.png")
    
    first = True
    for dat, label in test_loader:
        x = dat.detach().cpu().numpy().transpose(0,2,3,1)
        y = label.detach().cpu().numpy()

        x = (x*255).astype(int)
        if first:
            first = False
            show_grid(x, y, "./imgs/test_imgs.png")

def main():
    print(f"hello MIAT - we are converting img data from MalImage and saving on disk as np arrays")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"device: {device}") #sanity check - using GPU
    
    # setup data loader
    # don't augment the images here!!! But do resize all to (224,224) to match original work
    trans = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    full_dataset = datasets.ImageFolder('./data/malimg_paper_dataset_imgs', transform=trans)
    #extract class names https://stackoverflow.com/questions/51906144/pytorch-image-label
    for name in full_dataset.classes:
        print(f"name: {name}")
        print(f"id_x: {full_dataset.class_to_idx[name]}")
        classes[full_dataset.class_to_idx[name]] = name
    print(f"classes: {classes}")

    #split dataset 90% train, 10% test
    subsets_dataset = torch.utils.data.random_split(full_dataset, [8405, 934], generator=torch.Generator().manual_seed(42)) #still the best seed ever
    
    #data loaders [0] is train, [1] is test
    train_loader = torch.utils.data.DataLoader(subsets_dataset[0], batch_size=args.batch_size, drop_last=False,
                                               shuffle=False, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(subsets_dataset[1], batch_size=args.batch_size, drop_last=False, shuffle=False,
                                              num_workers=8, pin_memory=True)

    cudnn.benchmark = True # added to see if it speeds up.
    
    first = True
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        if first:
            x = data.detach().cpu().numpy().transpose(0,2,3,1)
            y_true = target.detach().cpu().numpy()
            first = False
        else:
            #note here we put the two arrays in as a tuple (extra parenthesis)
            x = np.concatenate((x, data.detach().cpu().numpy().transpose(0,2,3,1)))
            y_true = np.concatenate((y_true, target.detach().cpu().numpy()))
            #break #for now just do two batches
    #print(f"x type: {type(x)}")
    print(f"x.shape: {x.shape}")
    print(f"len(x): {len(x)}")
    print(f"y_true.shape:{y_true.shape}")
    print(f"len(y_true):{len(y_true)}")
    #print(y_true)
    x = (x*255).astype(np.uint8)
    #x is now [b][w][h][c] range of [0,255]
    y_true = y_true.astype(np.uint8)
    
    first = True
    #now do it again for the test set
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        if first:
            x_test = data.detach().cpu().numpy().transpose(0,2,3,1)
            y_test_true = target.detach().cpu().numpy()
            first = False
        else:
            #note here we put the two arrays in as a tuple (extra parenthesis)
            x_test = np.concatenate((x_test, data.detach().cpu().numpy().transpose(0,2,3,1)))
            y_test_true = np.concatenate((y_test_true, target.detach().cpu().numpy()))
            
    print(f"x_test.shape: {x_test.shape}")
    print(f"len(x_test): {len(x_test)}")
    print(f"y_test_true.shape:{y_test_true.shape}")
    print(f"len(y_test_true):{len(y_test_true)}")
    
    #TODO check this for rounding errors
    #need to denormalize:
    x_test = (x_test*255).astype(np.uint8)
    #x_test is now [b][w][h][c] range of [0,255]
    y_test_true = y_test_true.astype(np.uint8)
    
    show_grid(x, y_true) #display a random 5x5 grid of the imgs for sanity check
    show_grid(x_test, y_test_true) #display a random 5x5 grid of the imgs for sanity check

    print("Saving numpy arrays to file ...")
    with open(args.nat_img_train_mal, 'wb') as f:
        np.save(f, x)
    with open(args.nat_label_train_mal, 'wb') as f:
        np.save(f, y_true)
    with open(args.nat_img_test_mal, 'wb') as f:
        np.save(f, x_test)
    with open(args.nat_label_test_mal, 'wb') as f:
        np.save(f, y_test_true)
    print("Save complete!")
    
    #now to reopen them and make sure they work

        
if __name__ == '__main__':
    #main()
    open_samples() #verify images can be read back from disk

    
