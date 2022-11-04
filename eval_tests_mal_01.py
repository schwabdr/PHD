'''
10/30/22 - First - display a grid of images
    a) pick image from each class
    b) should be classified correctly
    c) and adversarial example classified incorrectly
    d) use these indices for the grid.
    e) we'll do 6 classes.
        i) VB.AT [22]
        ii) Fakerean [10]
        iii) Allaple.A [2]
        iv) C2LOP.P [6]
        v) Lolyda.AA3 [14]
        vi) Alueron.gen!J [4]

Purpose of this file is to evaluate any given model with adversarial examples given by any model.
Both models can be the same for a white box evaluation.
We will test all three MIAT/NAMID models using a range of epsilon values and nb_iter = 100 in all cases.


'''
import os
import argparse
from symbol import continue_stmt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import torchvision.models as models

import numpy as np

from utils import config
from utils.data import data_dataset
from projected_gradient_descent import projected_gradient_descent as pgd
from models.resnet_new import ResNet18

#next is only needed to visualize samples
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

import random

classes = ['Adialer.C', 'Agent.FYI', 'Allaple.A', 'Allaple.L', 'Alueron.gen!J', 'Autorun.K', 'C2LOP.P', 'C2LOP.gen!g', 'Dialplatform.B', 'Dontovo.A', 'Fakerean', 
                'Instantaccess', 'Lolyda.AA1', 'Lolyda.AA2', 'Lolyda.AA3', 'Lolyda.AT', 'Malex.gen!J', 'Obfuscator.AD', 'Rbot!gen', 'Skintrim.N', 'Swizzor.gen!E', 'Swizzor.gen!I', 
                'VB.AT', 'Wintrim.BX', 'Yuner.A']
    
args = config.Configuration().getArgs()
stats = ((0.4454, 0.4454, 0.4454), (0.3122, 0.3122, 0.3122)) #mean and stdev

args.batch_size= 40


def show_img_grid(rows, cols, x, x_adv, y, y_adv,inds=None,fname=None):
    fig, axes1 = plt.subplots(rows,cols,figsize=(25,25))
    if inds is None: #pick random indices
        a = list(range(0, len(x)))
        random.shuffle(a)
        #print(f"a: {a}")
    else: #use the given indices
        a = inds
    #print("min/max of numpy arrays: ", np.min(x), np.max(x)) #for MalImg 0-254
    if rows == 1:
        for k in range(0,cols,2):
            #get a random index
            i = a.pop(0) #removes from front of list
            axes1[k].set_axis_off()
            axes1[k+1].set_axis_off()
            axes1[k].imshow(x[i],interpolation='nearest')
            axes1[k].text(0,0,classes[y[i]]) # this gets the point accross but needs fixing.
            axes1[k+1].imshow(x_adv[i], interpolation='nearest')
            axes1[k+1].text(0,0,classes[y_adv[i]])
    else: 
        for j in range(rows):
            for k in range(0,cols,2):
                #get a random index
                i = a.pop(0) #removes from front of list
                axes1[j][k].set_axis_off()
                axes1[j][k+1].set_axis_off()
                axes1[j][k].imshow(x[i],interpolation='nearest')
                axes1[j][k].text(0,0,classes[y[i]]) # this gets the point accross but needs fixing.
                axes1[j][k+1].imshow(x_adv[i], interpolation='nearest')
                axes1[j][k+1].text(0,0,classes[y_adv[i]])
    if fname is None:
        plt.show()
        plt.close()
    else:
        plt.savefig(fname, format='png')
        plt.close()

'''
There's plenty of CPU RAM (I think) so I'm going to stack the NumPy data on the CPU, and send it all here
number of rows and columns you want: pick 2x as many columns as you want images b/c you'll have clean / adv example alternating
'''
def show_img_grid_old(rows, cols, x, x_adv, y, y_adv):
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
    '''these values will cause large perturbations, and nearly 100% misclassifications on CIFAR10
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
    print(f"device: {device}")
    #all the models named
    names = ['resnet-mal-std-100', 'resnet-mal-std-aug-100', 'resnet-mal-MIAT.25', 'resnet-mal-MIAT-AT.25.40']
    name = names[3]
    
    model = ResNet18(25)
   
    model.load_state_dict(torch.load(os.path.join(args.SAVE_MODEL_PATH, name)))
    model.to(device)
    model = torch.nn.DataParallel(model).cuda() 
    model.eval()

    print(f"Model loaded: {name}")
    
    trans_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    testset = data_dataset(img_path=args.nat_img_test_mal, clean_label_path=args.nat_label_test_mal, transform=trans_test)
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

#purpose of this function is simply to display adversarial examples
def display_examples():
    print("Beginning Image Display ...")
    #for i, c_name in enumerate(classes):
    #    print(f"{i}: {c_name}")
    
    device = torch.device("cuda")
    


    #model_names = ['resnet-new-100', 'resnet-new-100-MIAT-from-scratch', 'resnet-new-100-MIAT-0.1-from-scratch', 'resnet-new-100-MIAT-0.25-from-scratch']
    model_names = ['resnet-mal-std-100']
    trans_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    testset = data_dataset(img_path=args.nat_img_test_mal, clean_label_path=args.nat_label_test_mal, transform=trans_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, drop_last=False, shuffle=False,
                                              num_workers=4, pin_memory=True)
    
    for name in model_names:
        #load model parameters for this test
        model = ResNet18(25)
        model.load_state_dict(torch.load(os.path.join(args.SAVE_MODEL_PATH, name)))
        model.to(device)
        model = torch.nn.DataParallel(model).cuda() 
        model.eval()

        print(f"Model loaded: {name}")
    
        
        #this is our L_infty constraint 
        eps_lst = [.025, .05, .075, .1, .125, .15, .175, .2, .25, .3, .4, .5, .75, 1.] #, 1.5, 2., 2.5]
        #eps_lst = [.3] # for quick test

        for eps in eps_lst:
            print(25*'=')
            
            eps_iter = .007
            #nb_iter = round(eps/eps_iter) + 10
            nb_iter = 100 #100
            print(f"Using PGD with eps: {eps}, eps_iter: {eps_iter}, nb_iter: {nb_iter}")
            #with torch.no_grad():
            batch_num = 0
            for data, target in test_loader:
                batch_num += 1
                if batch_num != 7:
                    continue
                print(f"batch_num: {batch_num}")
                data, target = data.to(device), target.to(device)
                output = model(data)
                top2 = torch.topk(output, 2) # this is for a targeted attack
                y_target = torch.select(top2.indices, 1, 1) #y_target is second most likely class
                data_adv = pgd(model, data, eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, norm=np.inf, y=None, targeted=False) #NON_targeted attack
                #data_adv = pgd(model, data, eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, norm=np.inf, y=y_target, targeted=True) #targeted attack

                # here I'll have to look back at some old code for stacking np images together to show in a grid
                #x_adv = data_adv.detach().cpu().numpy().transpose(0,2,3,1) #I'll use this later - gonna paste all the images together.
                #output = model(data)
                output_adv = model(data_adv)
                pred = output.max(1, keepdim=True)[1] # we'll use this instead of true label -make sure it is misclassified
                pred_adv = output_adv.max(1, keepdim=True)[1]
                
                x_test_clean = data.detach().cpu().numpy().transpose(0,2,3,1)
                #y_test_pred = pred.detach().cpu().numpy()
                y_test = target.detach().cpu().numpy()
                x_test_adv = data_adv.detach().cpu().numpy().transpose(0,2,3,1)
                y_test_adv = pred_adv.detach().cpu().numpy()
                # de normalize
                x_test_clean = np.clip(((x_test_clean * stats[1]) + stats[0]),0,1.)
                x_test_clean = (x_test_clean*255).astype(np.uint8)
                x_test_adv = np.clip(((x_test_adv * stats[1]) + stats[0]),0,1.)
                x_test_adv = (x_test_adv*255).astype(np.uint8)
                
                #x_test_adv is now [b][w][h][c] range of [0,255]
                y_test = y_test.astype(np.uint8)
                y_test = y_test.reshape((args.batch_size))
                y_test_adv = y_test_adv.astype(np.uint8)
                y_test_adv = y_test_adv.reshape((args.batch_size))

                print(f"y_test: {y_test}")
                fname = os.path.join('./results/imgs/', str(eps) + '-batch-' + str(batch_num) + '.PNG') #T is for targeted attack, remove for non-targeted
                inds = [5, 2, 36, 1, 38, 14] # hate hard coding magic numbers - these are the indicies of the classes I want in batch 7.
                show_img_grid(1,12, x_test_clean, x_test_adv, y_test, y_test_adv, inds=inds, fname=fname)

                break
            
            print(25*'=')
        

if __name__ == '__main__':
    main() # use for PGD-CE Mal data in tables in Ch5
    #display_examples() # use to display the grid used in ch 4