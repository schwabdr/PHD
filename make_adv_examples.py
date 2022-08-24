'''
The point of this file is to create an adversarial example dataset that can be read from disk simliar to the training / test set
'''

import torch
from torchvision import transforms

import numpy as np
import os
import argparse
import projected_gradient_descent as pgd

from models.resnet_new import ResNet18

from utils import utils
from utils import config
from utils import data 

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

c = config.Configuration()
args = c.getArgs()
stats = c.getNormStats()

def make_examples(model, device, train_loader):
    eps = 8/255.
    eps_iter = .007
    nb_iter = 40
    print(f"Using PGD with eps: {eps}, eps_iter: {eps_iter}, nb_iter: {nb_iter}")

    first = True
    for data, target in train_loader:
        x = data.detach().cpu().numpy().transpose(0,2,3,1)
        data, target = data.to(device), target.to(device)

        data_adv = pgd.projected_gradient_descent(model, data, eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, norm=np.inf,y=target, targeted=False)
        if first:
            x_adv = data_adv.detach().cpu().numpy().transpose(0,2,3,1)
            first = False
        else:
            #note here we put the two arrays in as a tuple (extra parenthesis)
            x_adv = np.concatenate((x_adv, data_adv.detach().cpu().numpy().transpose(0,2,3,1)))
            break #for now just do two batches
    #print(f"x_adv type: {type(x_adv)}")
    print(f"x_adv.shape(): {x_adv.shape}")
    #need to denormalize:
    x_adv = np.clip(((x_adv * stats[1]) + stats[0]),0,1.)
    x_adv = (x_adv*255).astype(int)
    print(x_adv)



def main():
    print(f"hello MIAT - we are creating adversarial example data on disk as np arrays")
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    name = 'resnet-new-100' #input("Name of model to load: ") #for now I'll hard code the only model I have trained
    #model = models.resnet18()
    model = ResNet18(10)
    path = str(os.path.join(args.SAVE_MODEL_PATH, name))

    model.load_state_dict(torch.load(os.path.join(args.SAVE_MODEL_PATH, name)))
    model.to(device)
    model.eval()
    print(f"Model loaded: {name}. Will generate adversarial examples from this model.")

    # setup data loader
    trans_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats, inplace=False) #original code was True here from MIAT - not sure why, just making a note
    ])

    trans_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    #now I need to pull in the data and create the dataloaders
    # this link states that we do indeed move the data out of the range [0,1] - so I guess this is correct.
    # https://www.kaggle.com/code/fanbyprinciple/cifar10-explanation-with-pytorch
    # min: -1.9259666204452515
    # max: 2.130864143371582

    trainset = data.data_dataset(img_path=args.nat_img_train, clean_label_path=args.nat_label_train, transform=trans_train)
    testset = data.data_dataset(img_path=args.nat_img_test, clean_label_path=args.nat_label_test, transform=trans_test)

    #create data loaders
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, drop_last=False,
                                               shuffle=False, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, drop_last=False, shuffle=False,
                                              num_workers=4, pin_memory=True)

    make_examples(model, device, train_loader)
        
if __name__ == '__main__':
    main()
