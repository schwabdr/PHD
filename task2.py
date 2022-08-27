'''
Purpose of this file is to begin work on task 2 of my planned contribution
2. Use the Mutual Information (MI) estimation networks as a metric to aid in the crafting of adversarial examples.
'''
from locale import locale_encoding_alias
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler, Adam

import numpy as np

import projected_gradient_descent as pgd
from models.resnet_new import ResNet18
#from models.wideresnet_new import WideResNet #might use eventually

from models.estimator import Estimator
from models.discriminators import MI1x1ConvNet, MIInternalConvNet, MIInternallastConvNet
from compute_MI import compute_loss

#import projected_gradient_descent as pgd #cleverhans PGD
from pgd_mi import projected_gradient_descent as pgd

from utils import config
from utils import utils
from utils.data import data_dataset #doing it this way avoids clash with variable named "data"

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

args = config.Configuration().getArgs()
stats = config.Configuration().getNormStats()

args.batch_size=512 #may need to reduce. 2048 was too high 1024 too high

#PGD Parameters
eps = args.eps
eps_iter = args.eps_iter
nb_iter = args.nb_iter

def craft_and_eval(models, device, test_loader):
    if len(models) == 1:
        model = models[0]
        model_adv = models[0]
    elif len(models) == 2:
        model = models[0]
        model_adv = models[1]
    elif len(models)==6:
        model = models[0]
        model_adv = models[1]
        local_n = models[2]
        local_a = models[3]
        global_n = models[4]
        global_a = models[5]

    loss = []
    acc = []
    acc_adv = []

    #set all models to eval mode
    for m in models:
        m.eval()
    

    '''these values will cause large perturbations, and nearly 100% misclassifications.
    eps = 1.27
    eps_iter = .05
    nb_iter = 50
    '''
    
    #this is our L_infty constraint - added 1.5+ 
    eps_lst = [.025, .05, .075, .1, .125, .15, .175, .2, .25, .3, .4, .5, .75, 1.] #, 1.5, 2., 2.5] #stopping at 1
    #eps_lst = [.025, .05] # for quick test

    for eps in eps_lst:
        print(25*'=')
        test_loss = 0
        correct = 0
        correct_adv = 0

        eps_iter = .007
        nb_iter = round(eps/eps_iter) + 10
        #nb_iter = 100 #trying this since I can do it in parallel now
        print(f"Using PGD with eps: {eps}, eps_iter: {eps_iter}, nb_iter: {nb_iter}")
        #with torch.no_grad():
        i = 0
        for data, target in test_loader:
            i = i+1
            print(f"batch number {i}, {i*args.batch_size} / {len(test_loader.dataset)}")
            data, target = data.to(device), target.to(device)
            #pass all the models to the pgd function
            data_adv = pgd(models, data, eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, norm=np.inf, y=None, targeted=False)
            
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
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    print(f"device: {device}") #sanity check - using GPU

    trans_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats, inplace=False)
    ])

    testset = data_dataset(img_path=args.nat_img_test, clean_label_path=args.nat_label_test, transform=trans_test)
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

    #load both models 
    name = 'resnet-new-100' 
    name2 = 'resnet-new-100-MIAT-from-scratch'
    model = ResNet18(10)
    model2 = ResNet18(10)
    path = str(os.path.join(args.SAVE_MODEL_PATH, name))
    
    model.load_state_dict(torch.load(os.path.join(args.SAVE_MODEL_PATH, name)))
    model.to(device)
    model = torch.nn.DataParallel(model).cuda() 
    model.eval()

    model2.load_state_dict(torch.load(os.path.join(args.SAVE_MODEL_PATH, name2)))
    model2.to(device)
    model2 = torch.nn.DataParallel(model2).cuda() 
    model2.eval()
    
    cudnn.benchmark = False

    print(f"Model loaded: {name}")
    print(f"Model loaded: {name2}")
    


    t_acc = []
    t_adv = []
    t_loss = []
    

    print(64*'=')
    #           0      1       2        3        4         5
    models = [model, model, local_n, local_a, global_n, global_a]

    print("Test Target: STD, Oracle: STD")
    test_loss, test_accuracy, adv_accuracy = craft_and_eval(models, device, test_loader)
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
    #           0      1       2        3        4         5
    models = [model2, model2, local_n, local_a, global_n, global_a]

    print("Test Target: STD, Oracle: STD")
    test_loss, test_accuracy, adv_accuracy = craft_and_eval(models, device, test_loader)
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