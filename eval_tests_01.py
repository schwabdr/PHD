'''
Purpose of this file is to evaluate any given model with adversarial examples given by any model.
Both models can be the same for a white box evaluation.
We will test all three MIAT/NAMID models using a range of epsilon values and nb_iter = 100 in all cases.
'''
import os
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import torchvision.models as models

import numpy as np

from utils import config
from utils import data
from projected_gradient_descent import projected_gradient_descent as pgd
from models.resnet_new import ResNet18

#next is only needed to visualize samples
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

import random

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

args = config.Configuration().getArgs()
stats = config.Configuration().getNormStats()

#args.batch_size=512 #trying this
args.batch_size=2048 #trying this - this works with the data parallel - GPU util ~95%, memory 10800/11019 MB each GPU

'''
There's plenty of CPU RAM (I think) so I'm going to stack the NumPy data on the CPU, and send it all here
number of rows and columns you want: pick 2x as many columns as you want images b/c you'll have clean / adv example alternating
'''
def show_img_grid(rows, cols, x, x_adv, y, y_adv):
    fig, axes1 = plt.subplots(rows,cols,figsize=(5,5))
    lst = list(range(0, len(x)))
    random.shuffle(lst)
    #print("min/max of numpy arrays: ", np.min(x), np.max(x)) #this was very close to 0 and 1
    for j in range(rows):
        for k in range(0,cols,2):
            #get a random index
            i = lst.pop()
            axes1[j][k].set_axis_off()
            axes1[j][k+1].set_axis_off()
            axes1[j][k].imshow(x[i],interpolation='nearest')
            axes1[j][k].text(0,0,classes[y[i]]) # this gets the point accross but needs fixing.
            axes1[j][k+1].imshow(x_adv[i], interpolation='nearest')
            axes1[j][k+1].text(0,0,classes[y_adv[i]])
    plt.show()

'''
param: model: The DNN model to evaluate
param: device: cuda or cpu
param: model_adv: The DNN model for creating adversarial examples if None, then model will be used.

'''
def eval_test_w_adv(model, device, test_loader, model_adv=None):
    if model_adv is None:
        model_adv = model

    loss = []
    acc = []
    acc_adv = []

    model.eval()
    model_adv.eval()
    '''these values will cause large perturbations, and nearly 100% misclassifications.
    eps = 1.27
    eps_iter = .05
    nb_iter = 50
    '''
    
    #this is our L_infty constraint - added 1.5+ 
    eps_lst = [.025, .05, .075, .1, .125, .15, .175, .2, .25, .3, .4, .5, .75, 1.] #, 1.5, 2., 2.5]
    #eps_lst = [.025, .05] # for quick test

    for eps in eps_lst:
        print(25*'=')
        test_loss = 0
        correct = 0
        correct_adv = 0

        eps_iter = .007
        #nb_iter = round(eps/eps_iter) + 10
        nb_iter = 100
        print(f"Using PGD with eps: {eps}, eps_iter: {eps_iter}, nb_iter: {nb_iter}")
        #with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data_adv = pgd(model_adv, data, eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, norm=np.inf, y=None, targeted=False)
            
            #x_adv = data_adv.detach().cpu().numpy().transpose(0,2,3,1) #I'll use this later - gonna paste all the images together.
            output = model(data)
            output_adv = model(data_adv)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            pred_adv = output_adv.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            correct_adv += pred_adv.eq(target.view_as(pred_adv)).sum().item()
            
            #break # do one batch for quick test
        test_loss /= len(test_loader.dataset)
        print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Robust Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset), correct_adv, len(test_loader.dataset),
            100. * correct_adv / len(test_loader.dataset)))

        test_accuracy = correct / len(test_loader.dataset)
        adv_accuracy = correct_adv / len(test_loader.dataset)
        print(25*'=')
        loss.append(test_loss)
        acc.append(test_accuracy)
        acc_adv.append(adv_accuracy)

        
    return loss, acc, acc_adv


def main():
    print("Beginning Evaluation ...")

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    #TODO add choice to specify two models, one to generate examples, and one to evaluate, i.e. blackbox evaluation
    # current task - load MIAT model and model below, compare them.
    # for now we just use one model.
    name = 'resnet-new-100-MIAT-0.25-from-scratch' #input("Name of model to load: ") #for now I'll hard code the only model I have trained
    
    model = ResNet18(10)
   
    model.load_state_dict(torch.load(os.path.join(args.SAVE_MODEL_PATH, name)))
    model.to(device)
    model = torch.nn.DataParallel(model).cuda() 
    model.eval()

    print(f"Model loaded: {name}")
    
    trans_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    testset = data.data_dataset(img_path=args.nat_img_test, clean_label_path=args.nat_label_test, transform=trans_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, drop_last=False, shuffle=False,
                                              num_workers=4, pin_memory=True)
    t_acc = []
    t_adv = []
    t_loss = []
    

    print(64*'=')
    test_loss, test_accuracy, adv_accuracy = eval_test_w_adv(model, device, test_loader, model_adv=model)
    t_acc.append(test_accuracy)
    t_loss.append(test_loss)
    t_adv.append(adv_accuracy)
    print("total acc:")
    print(test_accuracy)
    print("total adv acc:")
    print(adv_accuracy)
    print("total loss:")
    print(test_loss)
    
    
    
    print(64*'=')

    print(t_acc)
    print(t_adv)
    print(t_loss)

    print("Clean Accuracy:")
    for x in t_acc:
        for y in x:
            print(y)
    print("Adversarial Accuracy:")
    for x in t_adv:
        for y in x:
            print(y)
    print("Loss:")
    for x in t_loss:
        for y in x:
            print(y)

if __name__ == '__main__':
    main()