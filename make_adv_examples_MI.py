'''
NOT COMPLETE
The point of this file is to create an adversarial example dataset using MI loss that can be read from disk simliar to the training / test set
This will be used once I fine tune the MI-Craft method for crafting adversarial examples.
'''

import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn

import numpy as np
import os

from pgd_mi import projected_gradient_descent as pgd

import random

from models.resnet_new import ResNet18

from utils import config
from utils import data 

from models.estimator import Estimator
from models.discriminators import MI1x1ConvNet, MIInternalConvNet, MIInternallastConvNet

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

c = config.Configuration()
args = c.getArgs()
stats = c.getNormStats()

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

args.batch_size=1024 #may need to reduce. 

def show_grid(x, y):
    rows = 5
    cols = 5
    fig,axes1 = plt.subplots(rows,cols,figsize=(5,5))
        
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

def open_adv_examples():
    print(f"Opening adv examples and printing random")
    #i'm not normalizing here since I want to display
    trans_train = transforms.Compose([
        transforms.ToTensor()
    ])

    trans_test = transforms.Compose([
        transforms.ToTensor()
    ])
    
    #now I need to pull in the data and create the dataloaders
    # this link states that we do indeed move the data out of the range [0,1] - so I guess this is correct.
    # https://www.kaggle.com/code/fanbyprinciple/cifar10-explanation-with-pytorch
    # min: -1.9259666204452515
    # max: 2.130864143371582

    trainset = data.data_dataset(img_path=args.adv_img_train, clean_label_path=args.adv_label_train, transform=trans_train)
    testset = data.data_dataset(img_path=args.adv_img_test, clean_label_path=args.adv_label_test, transform=trans_test)
    
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
            show_grid(x, y)
    
    first = True
    for dat, label in test_loader:
        x = dat.detach().cpu().numpy().transpose(0,2,3,1)
        y = label.detach().cpu().numpy()

        x = (x*255).astype(int)
        if first:
            first = False
            show_grid(x, y)
    



def make_examples(models, device, train_loader, test_loader):
    #I'll pick these values once I see the data.
    eps = 8/255.
    eps_iter = .007
    nb_iter = 40
    print(f"Using PGD with eps: {eps}, eps_iter: {eps_iter}, nb_iter: {nb_iter}")

    first = True
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        #data_adv = pgd.projected_gradient_descent(model, data, eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, norm=np.inf,y=None, targeted=False)
        data_adv = pgd(models, data, eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, norm=np.inf, y=None, targeted=False)
            
        if first:
            x_adv = data_adv.detach().cpu().numpy().transpose(0,2,3,1)
            y_true = target.detach().cpu().numpy()
            first = False
        else:
            #note here we put the two arrays in as a tuple (extra parenthesis)
            x_adv = np.concatenate((x_adv, data_adv.detach().cpu().numpy().transpose(0,2,3,1)))
            y_true = np.concatenate((y_true, target.detach().cpu().numpy()))
            #break #for now just do two batches
    #print(f"x_adv type: {type(x_adv)}")
    print(f"x_adv.shape: {x_adv.shape}")
    print(f"len(x_adv): {len(x_adv)}")
    print(f"y_true.shape:{y_true.shape}")
    print(f"len(y_true):{len(y_true)}")
    #print(y_true)
    #need to denormalize:
    x_adv = np.clip(((x_adv * stats[1]) + stats[0]),0,1.)
    x_adv = (x_adv*255).astype(np.uint8)
    #x_test_adv is now [b][w][h][c] range of [0,255]
    y_true = y_true.astype(np.uint8)
    
    #c = 0
    first = True
    #now do it again for the test set
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        #c = c + 1
        #data_adv = pgd.projected_gradient_descent(model, data, eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, norm=np.inf,y=None, targeted=False)
        data_adv = pgd(models, data, eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, norm=np.inf, y=None, targeted=False)
            
        if first:
            x_test_adv = data_adv.detach().cpu().numpy().transpose(0,2,3,1)
            y_test_true = target.detach().cpu().numpy()
            first = False
        else:
            #note here we put the two arrays in as a tuple (extra parenthesis)
            x_test_adv = np.concatenate((x_test_adv, data_adv.detach().cpu().numpy().transpose(0,2,3,1)))
            y_test_true = np.concatenate((y_test_true, target.detach().cpu().numpy()))
            #if c == 10:
            #    break #use this to do just two batches 
    print(f"x_test_adv.shape: {x_test_adv.shape}")
    print(f"len(x_test_adv): {len(x_test_adv)}")
    print(f"y_test_true.shape:{y_test_true.shape}")
    print(f"len(y_test_true):{len(y_test_true)}")
    
    #TODO check this for rounding errors
    #need to denormalize:
    x_test_adv = np.clip(((x_test_adv * stats[1]) + stats[0]),0,1.)
    x_test_adv = (x_test_adv*255).astype(np.uint8)
    #x_test_adv is now [b][w][h][c] range of [0,255]
    y_test_true = y_test_true.astype(np.uint8)
    


    show_grid(x_adv, y_true) #display a random 5x5 grid of the imgs for sanity check
    show_grid(x_test_adv, y_test_true) #display a random 5x5 grid of the imgs for sanity check



    print("Saving numpy arrays to file ...")
    with open(args.adv_img_train, 'wb') as f:
        np.save(f, x_adv)
    with open(args.adv_label_train, 'wb') as f:
        np.save(f, y_true)
    with open(args.adv_img_test, 'wb') as f:
        np.save(f, x_test_adv)
    with open(args.adv_label_test, 'wb') as f:
        np.save(f, y_test_true)
    print("Save complete!")
    
    #now to reopen them and make sure they work
    #will make the call from __main__
    

def main():
    print(f"hello MIAT - we are creating adversarial (using MI as loss) example data on disk as np arrays")
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print(f"device: {device}") #sanity check - using GPU
    
    # setup data loader
    # don't flip the images here!!!
    trans_train = transforms.Compose([
        #transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        #transforms.RandomHorizontalFlip(),
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

   
    # need to read up on this next section ...
    # Estimator part 1: X or layer3 to H space
    local_n = Estimator(args.va_hsize)
    local_a = Estimator(args.va_hsize)

    # estimator part 2: Z to H space
    if args.is_internal == True:
        if args.is_internal_last == True:
            z_size = 512
            global_n = MIInternallastConvNet(z_size, args.va_hsize)
            global_a = MIInternallastConvNet(z_size, args.va_hsize)
        else:
            z_size = 256
            global_n = MIInternalConvNet(z_size, args.va_hsize)
            global_a = MIInternalConvNet(z_size, args.va_hsize)
    else: #it's this case based on 'args' in config.py
        z_size = 10
        global_n = MI1x1ConvNet(z_size, args.va_hsize)
        global_a = MI1x1ConvNet(z_size, args.va_hsize)

    local_n.load_state_dict(torch.load(os.path.join(args.SAVE_MODEL_PATH, 'local_n')))
    global_n.load_state_dict(torch.load(os.path.join(args.SAVE_MODEL_PATH, 'global_n')))
    local_a.load_state_dict(torch.load(os.path.join(args.SAVE_MODEL_PATH, 'local_a')))
    global_a.load_state_dict(torch.load(os.path.join(args.SAVE_MODEL_PATH, 'global_a')))

    
    local_n = torch.nn.DataParallel(local_n).cuda()
    global_n = torch.nn.DataParallel(global_n).cuda()
    local_a = torch.nn.DataParallel(local_a).cuda()
    global_a = torch.nn.DataParallel(global_a).cuda()

    #load models
    name = 'resnet-new-100-MIAT-from-scratch'
    model = ResNet18(10)
    path = str(os.path.join(args.SAVE_MODEL_PATH, name))
    
    model.load_state_dict(torch.load(os.path.join(args.SAVE_MODEL_PATH, name)))
    model.to(device)
    model = torch.nn.DataParallel(model).cuda() 
    model.eval()

    cudnn.benchmark = True #trying this

    print(f"Model loaded: {name}. Will craft examples from here.")
    

    #           0      1       2        3        4         5
    models = [model, model, local_n, local_a, global_n, global_a]

    make_examples(models, device, train_loader, test_loader)

        
if __name__ == '__main__':
    main()
    open_adv_examples() #verify images can be read back from disk
