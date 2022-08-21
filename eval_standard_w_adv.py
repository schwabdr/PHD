'''
Purpose of this file is to evaluate any given model with adversarial examples given by any model.
Both models can be the same for a white box evaluation.
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
import projected_gradient_descent as pgd
from models.resnet_new import ResNet18

#next is only needed to visualize samples
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

import random

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

args = config.Configuration().getArgs()

def eval_test_w_adv(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    correct_adv = 0
    '''these values will cause large perturbations, and nearly 100% misclassifications.
    eps = 1.27
    eps_iter = .05
    nb_iter = 50
    '''
    eps = 8/255.
    eps_iter = .007
    nb_iter = 40
    print(f"Using PGD with eps: {eps}, eps_iter: {eps_iter}, nb_iter: {nb_iter}")
    first = True
    #with torch.no_grad():
    for data, target in test_loader:
        #print("data: ", data)
        #print("target: ", target)
        #print("min: ", torch.min(data)) #I can't figure why the min = -2.4291 and max = 2.7537
        #print("max: ", torch.max(data))
        x = data.detach().cpu().numpy().transpose(0,2,3,1)
        data, target = data.to(device), target.to(device)
        #epsilon was 8/255
        #perturb_steps was 40
        #data_adv = craft_adversarial_example(model=model, x_natural=data, y=target,
        #                                     step_size=0.007, epsilon=25/255,
        #                                     perturb_steps=100, distance='l_inf')
        #eps_iter is the steps size I think
        #eps is the clamp on the max change in l_inf norm?
        #data_adv = pgd.projected_gradient_descent(model, data, eps=.1, eps_iter=.007, nb_iter=100, norm=np.inf,y=None, targeted=False)
        #the following line parameters reach a 0% robust accuracy.
        #data_adv = pgd.projected_gradient_descent(model, data, eps=5.2, eps_iter=.25, nb_iter=500, norm=np.inf,y=target, targeted=False)
        #changed to targeted seems to be the key to get 0% acc.
        #eps=1.5 was 0%
        
        #next line achieves *almost* 0% accuracy - 3/10000.
        data_adv = pgd.projected_gradient_descent(model, data, eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, norm=np.inf,y=target, targeted=False)
        #next line is to use original samples to verify actual test accuracy is correct
        #data_adv = data

        #print("data_adv", data_adv)
        x_adv = data_adv.detach().cpu().numpy().transpose(0,2,3,1)
        output = model(data)
        output_adv = model(data_adv)
        test_loss += F.cross_entropy(output, target, reduction='sum').item()
        pred = output.max(1, keepdim=True)[1]
        pred_adv = output_adv.max(1, keepdim=True)[1]
        #print(f"pred: {pred}, pred_adv{pred_adv}")
        correct += pred.eq(target.view_as(pred)).sum().item()
        correct_adv += pred_adv.eq(target.view_as(pred_adv)).sum().item()
        

        #randomly show a grid of the adversarial examples:
        if first: #False to block out this section of code for now.
            first = False
            #goal here is to randomly display an image and it's adverarial example
            stats = ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)) # mean and stdev
            x = np.clip(((x * stats[1]) + stats[0]),0,1.)
            x_adv = np.clip(((x_adv * stats[1]) + stats[0]),0,1.)
            rows = 5
            cols = 10
            fig, axes1 = plt.subplots(rows,cols,figsize=(5,5))
            lst = list(range(0, len(x)))
            random.shuffle(lst)
            #print("min/max of numpy arrays: ", np.min(x), np.max(x)) #this was very close to 0 and 1
            for j in range(5):
                for k in range(0,10,2):
                    #get a random index
                    i = lst.pop()
                    axes1[j][k].set_axis_off()
                    axes1[j][k+1].set_axis_off()
                    axes1[j][k].imshow(x[i],interpolation='nearest')
                    axes1[j][k].text(0,0,classes[target[i]]) # this gets the point accross but needs fixing.
                    axes1[j][k+1].imshow(x_adv[i], interpolation='nearest')
                    pred_ind = pred_adv[i]
                    axes1[j][k+1].text(0,0,classes[pred_ind])
            plt.show()    
            
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Robust Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), correct_adv, len(test_loader.dataset),
        100. * correct_adv / len(test_loader.dataset)))

    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def main():
    print("Beginning Evaluation ...")

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    #TODO add choice to specify two models, one to generate examples, and one to evaluate, i.e. blackbox evaluation
    # for now we just use one model.
    name = 'resnet-new-100' #input("Name of model to load: ") #for now I'll hard code the only model I have trained
    #model = models.resnet18()
    model = ResNet18(10)
    path = str(os.path.join(args.SAVE_MODEL_PATH, name))
    
    '''
    #this block of code is good to know - if I want to load the state_dict and make any changes
    #note than an optimizer can also have it's state_dict saved
    state_dic = torch.load(path)
    new_state = model.state_dict()

    for k in state_dic.keys():
        if k in new_state.keys():
            new_state[k] = state_dic[k]
            # print(k)
        else:
            break

    model.load_state_dict(new_state)
    '''
    #model=model.to(device)
    
    #print(f"Loading from {path}")
    model.load_state_dict(torch.load(os.path.join(args.SAVE_MODEL_PATH, name)))
    model.to(device)
    model.eval()
    print(f"Model loaded: {name}")

    #https://www.kaggle.com/code/kmldas/cifar10-resnet-90-accuracy-less-than-5-min/notebook
    #stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    #https://github.com/kuangliu/pytorch-cifar/issues/19
    stats = ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))

    trans_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    testset = data.data_dataset(img_path=args.nat_img_test, clean_label_path=args.nat_label_test, transform=trans_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, drop_last=False, shuffle=False,
                                              num_workers=4, pin_memory=True)

    print(64*'=')
    _, test_accuracy = eval_test_w_adv(model, device, test_loader)
    print(64*'=')


if __name__ == '__main__':
    main()