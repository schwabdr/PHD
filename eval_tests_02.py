'''
https://pytorch.org/docs/stable/generated/torch.cdist.html
Computes batched the p-norm distance between each pair of the two collections of row vectors.

The purpose of this file is to study the loss metrics used for MIAT/NAMID

Currently the only loss metric is cosine_loss = | 1 - cos_sim(a,b) |
One issue is that this only considers the angle between the two vectors, a,b. It does not consider the magnitude.
So it seems we are leaving some room on the table so to speak for adversarial examples to grow.
More specifically, the maximum cosine loss is 2, this occurs when 2 vectors are 180 degress apart.
Therefore, in this file I also add euclidean loss as a metric:

torch.sqrt(sum((loss_n - loss_a)**2))

I've learned from this file:
1) cosine loss has a range of [0,2]
2) can use euclidean distance as an additional metric
3) maximizing the loss_mi does not necessarily decrease the loss_mi for the other class labels
4) So i'm saving this file as it is - I've got some good data here I can reproduce.
5) moving on to loss_study_targeted.py


'''
import os

import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import numpy as np

from models.resnet_new import ResNet18
from models.estimator import Estimator
from models.discriminators import MI1x1ConvNet, MIInternalConvNet, MIInternallastConvNet
from compute_MI import compute_loss

from utils import config
from utils.data import data_adv_dataset #not sure why I thought I should us this one ...
from utils.data import data_dataset
from utils.utils import optimize_linear
from utils.utils import clip_eta

#from pgd_loss_test import projected_gradient_descent as pgd


#next is only needed to visualize samples
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

import random


args = config.Configuration().getArgs()
stats = config.Configuration().getNormStats() #
classes = config.Configuration().getClasses()

args.batch_size = 512

#####################################################
#THIS CODE IS FROM train_MIAT_alpha.py
#####################################################
'''
This should be wrapped up in a utils file or something ... but for now I'll just duplicate some code and get it working ...
'''

'''
Robust Accuracy: 24.950%
Test: Average loss: 0.0034, Accuracy: 9997/10000 (100%), Robust Accuracy: 2495/10000 (25%)
Using PGD with eps: 0.15, eps_iter: 0.007, nb_iter: 50

Name of this function is perhaps misleading as it gives TOTAL loss, not just the MI loss.
This will be my first test - a baseline. Once I tune it I'll work on other losses.
Uses the same MI_loss that is used for MIAT/NAMID training (CE+3MI-Terms)
:param model_fns: models for calculating loss
:param alpha: (optional) float. Hyper parameter for tuning the MI portion of loss.
'''
def MI_loss_test01(model_fns, x_natural, x_adv, y_true, iter=iter):
    #this setting 
    # typical layout for model_fns
    #               0        1         2        3         4        5
    #model_fns = {std_res, miat_res, local_n, global_n, local_a, global_a}
    target_model = model_fns[1] #target model to calculate loss
    encoder = model_fns[0] # model to use as encoder
    local_n = model_fns[2]
    global_n = model_fns[3]
    local_a = model_fns[4]
    global_a = model_fns[5]
    #all models should already be in eval() mode
    alpha = 1.
    lambd = .1
    #print(f"alpha: {alpha}, lambda: {lambd}")

    logits_adv = target_model(x_adv) # current pred of the model we are attacking
    loss_ce = F.cross_entropy(logits_adv, y_true)

    #code snippet to select the indices of adv examples that are misclassified and clean samples that
    # are correctly classified
    pseudo_label = F.softmax(target_model(x_natural), dim=0).max(1, keepdim=True)[1].squeeze()
    index = (pseudo_label == y_true)
    pseudo_label = F.softmax(target_model(x_adv), dim=0).max(1, keepdim=True)[1].squeeze()
    index = index * (pseudo_label != y_true)

    #init vars for print in case we hit the "else"
    loss_mea_n = 0
    loss_mea_a = 0
    loss_a_all = 0
    
    if torch.nonzero(index).size(0) != 0:
        loss_n = compute_loss(args=args, former_input=x_natural, latter_input=x_natural, encoder=encoder,
                dim_local=local_n, dim_global=global_n, v_out=True)* index

        loss_a = compute_loss(args=args, former_input=x_natural, latter_input=x_adv, encoder=encoder,
                               dim_local=local_n, dim_global=global_n, v_out=True) * index

        loss_a_all = loss_a # for the 3rd term
        loss_mea_n = torch.abs(torch.tensor(1.0).cuda() - torch.cosine_similarity(loss_n, loss_a, dim=0))


        loss_a = compute_loss(args=args, former_input=x_adv - x_natural, latter_input=x_adv, encoder=encoder,
                              dim_local=local_a, dim_global=global_a, v_out=True) * index

        loss_n = compute_loss(args=args, former_input=x_adv - x_natural, latter_input=x_natural, encoder=encoder,
                              dim_local=local_a, dim_global=global_a, v_out=True) * index
        
        # I changed the order of subtraction here to match the paper as far as I can tell eqn 8
        loss_a_all = loss_a - loss_a_all
        loss_a_all = loss_a_all.sum()/(torch.nonzero(index).size(0))
        loss_a_all = torch.abs(torch.tensor(lambd).cuda() * loss_a_all)
        
        loss_mea_a = torch.abs(torch.tensor(1.0).cuda() - torch.cosine_similarity(loss_n, loss_a, dim=0))

        loss_mi = loss_mea_n + loss_mea_a + loss_a_all #see eqn 8
    else:
        loss_mi = 0.0
        
        
    loss_all = loss_ce + torch.tensor(alpha).cuda() * loss_mi
    print(f"loss_all: {loss_all}, loss_ce: {loss_ce}, loss_mea_n: {loss_mea_n}, loss_mea_a: {loss_mea_a}, loss_a_all: {loss_a_all}")
    return loss_all

'''
Robust Accuracy: 25.20000%
Test: Average loss: 0.0034, Accuracy: 9997/10000 (100%), Robust Accuracy: 2520/10000 (25%)

For a single batch:
Robust Accuracy: 24.60938% 
Test: Average loss: 0.0001, Accuracy: 512/10000 (5%), Robust Accuracy: 126/10000 (1%)
Using PGD with eps: 0.15, eps_iter: 0.007, nb_iter: 50

Name of this function is perhaps misleading as it gives TOTAL loss, not just the MI loss.
This is the first version of MI_loss_testXX that incorporates a euclidean distance metric
between the MI vectors - I'm not convinced it helps - but it doesn't hurt. See stats above.

:param model_fns: models for calculating loss
:param alpha: (optional) float. Hyper parameter for tuning the MI portion of loss.
'''

def MI_loss_test02(model_fns, x_natural, x_adv, y_true):
    #this setting 
    # typical layout for model_fns
    #               0        1         2        3         4        5
    #model_fns = [std_res, miat_res, local_n, global_n, local_a, global_a]
    target_model = model_fns[1] #target model to calculate loss
    encoder = model_fns[0] # model to use as encoder
    local_n = model_fns[2]
    global_n = model_fns[3]
    local_a = model_fns[4]
    global_a = model_fns[5]
    #all models should already be in eval() mode
    alpha = 1.
    lambd = 1.
    #print(f"alpha: {alpha}, lambda: {lambd}")
    global euc_flag

    logits_adv = target_model(x_adv) # current pred of the model we are attacking
    loss_ce = F.cross_entropy(logits_adv, y_true)

    #code snippet to select the indices of adv examples that are misclassified and clean samples that
    # are correctly classified
    pseudo_label = F.softmax(target_model(x_natural), dim=0).max(1, keepdim=True)[1].squeeze()
    index = (pseudo_label == y_true)
    pseudo_label = F.softmax(target_model(x_adv), dim=0).max(1, keepdim=True)[1].squeeze()
    index = index * (pseudo_label != y_true)

    #init vars for print in case we hit the "else"
    loss_mea_n = 0
    loss_mea_a = 0
    loss_a_all = 0
    loss_euclid_mea_n = 0
    loss_euclid_mea_a = 0
    
    if torch.nonzero(index).size(0) != 0:
        loss_n = compute_loss(args=args, former_input=x_natural, latter_input=x_natural, encoder=encoder,
                dim_local=local_n, dim_global=global_n, v_out=True)* index

        loss_a = compute_loss(args=args, former_input=x_natural, latter_input=x_adv, encoder=encoder,
                               dim_local=local_n, dim_global=global_n, v_out=True) * index

        loss_euclid_mea_n = torch.sqrt(sum((loss_n - loss_a)**2))
        
        #print(f"loss_n.size(): {loss_n.size()}, loss_a.size() {loss_a.size()}") #loss_n.size(): torch.Size([512]), loss_a.size() torch.Size([512])

        loss_a_all = loss_a # for the 3rd term
        loss_mea_n = torch.abs(torch.tensor(1.0).cuda() - torch.cosine_similarity(loss_n, loss_a, dim=0))


        loss_a = compute_loss(args=args, former_input=x_adv - x_natural, latter_input=x_adv, encoder=encoder,
                              dim_local=local_a, dim_global=global_a, v_out=True) * index

        loss_n = compute_loss(args=args, former_input=x_adv - x_natural, latter_input=x_natural, encoder=encoder,
                              dim_local=local_a, dim_global=global_a, v_out=True) * index

        loss_euclid_mea_a = torch.sqrt(sum((loss_n - loss_a)**2))
        #print(f"loss_euclid_mea_n: {loss_euclid_mea_n}, loss_euclid_mea_a: {loss_euclid_mea_a}")
        
        #print(f"loss_n.size(): {loss_n.size()}, loss_a.size() {loss_a.size()}") #loss_n.size(): torch.Size([512]), loss_a.size() torch.Size([512])

        # I changed the order of subtraction here to match the paper as far as I can tell eqn 8
        loss_a_all = loss_a - loss_a_all
        loss_a_all = loss_a_all.sum()/(torch.nonzero(index).size(0))
        loss_a_all = torch.abs(torch.tensor(lambd).cuda() * loss_a_all)
        
        loss_mea_a = torch.abs(torch.tensor(1.0).cuda() - torch.cosine_similarity(loss_n, loss_a, dim=0))

        '''
        if euc_flag: #switch to euclidean distance
            loss_mi = .01*loss_euclid_mea_n + .1*loss_euclid_mea_a
            euc_flag = True # each batch will need to reset this to False
        else:
            loss_mi = loss_mea_n + loss_mea_a + loss_a_all #see eqn 8
            if loss_mea_a > 1.6 and loss_mea_n > 1.6:
                euc_flag = True
        '''
        loss_mi = loss_mea_n + loss_mea_a + loss_a_all + .01*loss_euclid_mea_n + .1*loss_euclid_mea_a
        #loss_mi = loss_euclid_mea_a + loss_euclid_mea_n

    else:
        loss_mi = 0.0
        
        
    loss_all = loss_ce + torch.tensor(alpha).cuda() * loss_mi
    print(f"loss_all: {loss_all}, loss_ce: {loss_ce}, loss_mea_n: {loss_mea_n}, loss_mea_a: {loss_mea_a}, loss_a_all: {loss_a_all}, euc_n: {loss_euclid_mea_n}, euc_a: {loss_euclid_mea_a}")
    return loss_all

'''
test03
Note here are we using the MIAT for encoder (I tried using STD for encoder and it performed worse)
1st Run:
Robust Accuracy: 21.90000%
Test: Average loss: 0.0034, Accuracy: 9997/10000 (100%), Robust Accuracy: 2190/10000 (22%)
2nd Run:
Robust Accuracy: 21.94000%
Test: Average loss: 0.0034, Accuracy: 9997/10000 (100%), Robust Accuracy: 2194/10000 (22%)


For a single batch:
Robust Accuracy: 23.04688%
Test: Average loss: 0.0001, Accuracy: 512/10000 (5%), Robust Accuracy: 118/10000 (1%)

Using PGD with eps: 0.15, eps_iter: 0.007, nb_iter: 50

Name of this function is perhaps misleading as it gives TOTAL loss, not just the MI loss.
This is the first version of MI_loss_testXX that incorporates a euclidean distance metric
between the MI vectors - I'm not convinced it helps - but it doesn't hurt. See stats above.

:param model_fns: models for calculating loss
:param alpha: (optional) float. Hyper parameter for tuning the MI portion of loss.
'''

def MI_loss_test03(model_fns, x_natural, x_adv, y_true, iter=0):
    #this setting 
    # typical layout for model_fns
    #               0        1         2        3         4        5
    #model_fns = [std_res, miat_res, local_n, global_n, local_a, global_a]
    target_model = model_fns[1] #target model to calculate loss
    encoder = model_fns[1] # model to use as encoder
    local_n = model_fns[2]
    global_n = model_fns[3]
    local_a = model_fns[4]
    global_a = model_fns[5]
    #all models should already be in eval() mode
    alpha = 1.
    lambd = .1
    #print(f"alpha: {alpha}, lambda: {lambd}")
    global euc_flag

    logits_adv = target_model(x_adv) # current pred of the model we are attacking
    loss_ce = F.cross_entropy(logits_adv, y_true)

    #code snippet to select the indices of adv examples that are misclassified and clean samples that
    # are correctly classified
    pseudo_label = F.softmax(target_model(x_natural), dim=0).max(1, keepdim=True)[1].squeeze()
    index = (pseudo_label == y_true)
    pseudo_label = F.softmax(target_model(x_adv), dim=0).max(1, keepdim=True)[1].squeeze()
    index = index * (pseudo_label != y_true)

    #init vars for print in case we hit the "else"
    loss_mea_n = 0
    loss_mea_a = 0
    loss_a_all = 0
    loss_euclid_mea_n = 0
    loss_euclid_mea_a = 0
    
    if torch.nonzero(index).size(0) != 0:
        loss_n = compute_loss(args=args, former_input=x_natural, latter_input=x_natural, encoder=encoder,
                dim_local=local_n, dim_global=global_n, v_out=True)* index

        loss_a = compute_loss(args=args, former_input=x_natural, latter_input=x_adv, encoder=encoder,
                               dim_local=local_n, dim_global=global_n, v_out=True) * index

        loss_euclid_mea_n = torch.sqrt(sum((loss_n - loss_a)**2))
        
        loss_a_all = loss_a # for the 3rd term
        loss_mea_n = torch.abs(torch.tensor(1.0).cuda() - torch.cosine_similarity(loss_n, loss_a, dim=0))


        loss_a = compute_loss(args=args, former_input=x_adv - x_natural, latter_input=x_adv, encoder=encoder,
                              dim_local=local_a, dim_global=global_a, v_out=True) * index

        loss_n = compute_loss(args=args, former_input=x_adv - x_natural, latter_input=x_natural, encoder=encoder,
                              dim_local=local_a, dim_global=global_a, v_out=True) * index

        loss_euclid_mea_a = torch.sqrt(sum((loss_n - loss_a)**2))
        
        # I changed the order of subtraction here to match the paper as far as I can tell eqn 8
        loss_a_all = loss_a - loss_a_all
        loss_a_all = loss_a_all.sum()/(torch.nonzero(index).size(0))
        loss_a_all = torch.abs(torch.tensor(lambd).cuda() * loss_a_all)
        
        loss_mea_a = torch.abs(torch.tensor(1.0).cuda() - torch.cosine_similarity(loss_n, loss_a, dim=0))

        #here we do different loss depending on how far along we are.
        if iter < 10:
            loss_all = loss_ce
        elif iter < 25: #add in cosine based MI_loss 
            loss_all = loss_ce + loss_mea_n + loss_mea_a + loss_a_all
        else: #switch cosine based for euclidean based MI_loss
            loss_all = loss_ce + .01*loss_euclid_mea_n + .1*loss_euclid_mea_a + loss_a_all
        #loss_mi = loss_mea_n + loss_mea_a + loss_a_all + .01*loss_euclid_mea_n + .1*loss_euclid_mea_a
        #loss_mi = loss_euclid_mea_a + loss_euclid_mea_n
    else:
        loss_all = loss_ce
        
        
    #loss_all = loss_ce + torch.tensor(alpha).cuda() * loss_mi
    print(f"loss_all: {loss_all}, loss_ce: {loss_ce}, loss_mea_n: {loss_mea_n}, loss_mea_a: {loss_mea_a}, loss_a_all: {loss_a_all}, euc_n: {loss_euclid_mea_n}, euc_a: {loss_euclid_mea_a}")
    return loss_all


'''
test04

Baseline: MIAT model attacked with PGD-CE
Robust Accuracy: 21.83000%
Test: Average loss: 0.0034, Accuracy: 9997/10000 (100%), Robust Accuracy: 2183/10000 (22%)
Robust Accuracy: 21.92000%
Test: Average loss: 0.0034, Accuracy: 9997/10000 (100%), Robust Accuracy: 2192/10000 (22%)

Baseline: STD model attacked with PGD-CE
Robust Accuracy: 0.07000%
Test: Average loss: 0.0015, Accuracy: 9998/10000 (100%), Robust Accuracy: 7/10000 (0%)
Robust Accuracy: 0.06000%
Test: Average loss: 0.0015, Accuracy: 9998/10000 (100%), Robust Accuracy: 6/10000 (0%)


Note here are we using the MIAT for encoder 
in the PGD and FGSM functions - I used the STD model for model_fn
It is only used for ground truth - so it's ok.

5 runs whole test dataset
Robust Accuracy: 21.93000%
Test: Average loss: 0.0034, Accuracy: 9997/10000 (100%), Robust Accuracy: 2193/10000 (22%)
Robust Accuracy: 21.90000%
Test: Average loss: 0.0034, Accuracy: 9997/10000 (100%), Robust Accuracy: 2190/10000 (22%)
Robust Accuracy: 21.87000%
Test: Average loss: 0.0034, Accuracy: 9997/10000 (100%), Robust Accuracy: 2187/10000 (22%)
Robust Accuracy: 22.01000%
Test: Average loss: 0.0034, Accuracy: 9997/10000 (100%), Robust Accuracy: 2201/10000 (22%)
Robust Accuracy: 22.02000%
Test: Average loss: 0.0034, Accuracy: 9997/10000 (100%), Robust Accuracy: 2202/10000 (22%)

For a single batch:
Robust Accuracy: 23.04688%
Test: Average loss: 0.0001, Accuracy: 512/10000 (5%), Robust Accuracy: 118/10000 (1%)

Using PGD with eps: 0.15, eps_iter: 0.007, nb_iter: 50

Name of this function is perhaps misleading as it gives TOTAL loss, not just the MI loss.
This is the first version of MI_loss_testXX that incorporates a euclidean distance metric
between the MI vectors - I'm not convinced it helps - but it doesn't hurt. See stats above.

:param model_fns: models for calculating loss
:param alpha: (optional) float. Hyper parameter for tuning the MI portion of loss.
'''

def MI_loss_test04(model_fns, x_natural, x_adv, y_true, iter=0):
    #this setting 
    # typical layout for model_fns
    #               0        1         2        3         4        5
    #model_fns = [std_res, miat_res, local_n, global_n, local_a, global_a]
    target_model = model_fns[1] #target model to calculate loss
    encoder = model_fns[1] # model to use as encoder
    local_n = model_fns[2]
    global_n = model_fns[3]
    local_a = model_fns[4]
    global_a = model_fns[5]
    #all models should already be in eval() mode
    alpha = 1.
    lambd = .1
    #print(f"alpha: {alpha}, lambda: {lambd}")
    global euc_flag

    logits_adv = target_model(x_adv) # current pred of the model we are attacking
    loss_ce = F.cross_entropy(logits_adv, y_true)

    #code snippet to select the indices of adv examples that are misclassified and clean samples that
    # are correctly classified
    pseudo_label = F.softmax(target_model(x_natural), dim=0).max(1, keepdim=True)[1].squeeze()
    index = (pseudo_label == y_true)
    pseudo_label = F.softmax(target_model(x_adv), dim=0).max(1, keepdim=True)[1].squeeze()
    index = index * (pseudo_label != y_true)

    #init vars for print in case we hit the "else"
    loss_mea_n = 0
    loss_mea_a = 0
    loss_a_all = 0
    loss_euclid_mea_n = 0
    loss_euclid_mea_a = 0
    
    if torch.nonzero(index).size(0) != 0:
        loss_n = compute_loss(args=args, former_input=x_natural, latter_input=x_natural, encoder=encoder,
                dim_local=local_n, dim_global=global_n, v_out=True)* index

        loss_a = compute_loss(args=args, former_input=x_natural, latter_input=x_adv, encoder=encoder,
                               dim_local=local_n, dim_global=global_n, v_out=True) * index

        loss_euclid_mea_n = torch.sqrt(sum((loss_n - loss_a)**2))
        
        loss_a_all = loss_a # for the 3rd term
        loss_mea_n = torch.abs(torch.tensor(1.0).cuda() - torch.cosine_similarity(loss_n, loss_a, dim=0))


        loss_a = compute_loss(args=args, former_input=x_adv - x_natural, latter_input=x_adv, encoder=encoder,
                              dim_local=local_a, dim_global=global_a, v_out=True) * index

        loss_n = compute_loss(args=args, former_input=x_adv - x_natural, latter_input=x_natural, encoder=encoder,
                              dim_local=local_a, dim_global=global_a, v_out=True) * index

        loss_euclid_mea_a = torch.sqrt(sum((loss_n - loss_a)**2))
        
        # I changed the order of subtraction here to match the paper as far as I can tell eqn 8
        loss_a_all = loss_a - loss_a_all
        loss_a_all = loss_a_all.sum()/(torch.nonzero(index).size(0))
        loss_a_all = torch.abs(torch.tensor(lambd).cuda() * loss_a_all)
        
        loss_mea_a = torch.abs(torch.tensor(1.0).cuda() - torch.cosine_similarity(loss_n, loss_a, dim=0))

        #here we do different loss depending on how far along we are.
        if iter < 5:
            loss_all = loss_ce
        elif iter < 25: #add in cosine based MI_loss 
            loss_all = loss_ce + loss_mea_n + loss_mea_a + loss_a_all
        else: #switch cosine based for euclidean based MI_loss
            loss_all = loss_ce + .01*loss_euclid_mea_n + .1*loss_euclid_mea_a + loss_a_all
        #loss_mi = loss_mea_n + loss_mea_a + loss_a_all + .01*loss_euclid_mea_n + .1*loss_euclid_mea_a
        #loss_mi = loss_euclid_mea_a + loss_euclid_mea_n
    else:
        loss_all = loss_ce
        
        
    #loss_all = loss_ce + torch.tensor(alpha).cuda() * loss_mi
    print(f"loss_all: {loss_all}, loss_ce: {loss_ce}, loss_mea_n: {loss_mea_n}, loss_mea_a: {loss_mea_a}, loss_a_all: {loss_a_all}, euc_n: {loss_euclid_mea_n}, euc_a: {loss_euclid_mea_a}")
    return loss_all


'''
test04

Baseline: MIAT model attacked with PGD-CE
Robust Accuracy: 21.83000%
Test: Average loss: 0.0034, Accuracy: 9997/10000 (100%), Robust Accuracy: 2183/10000 (22%)
Robust Accuracy: 21.92000%
Test: Average loss: 0.0034, Accuracy: 9997/10000 (100%), Robust Accuracy: 2192/10000 (22%)

Baseline: STD model attacked with PGD-CE
Robust Accuracy: 0.07000%
Test: Average loss: 0.0015, Accuracy: 9998/10000 (100%), Robust Accuracy: 7/10000 (0%)
Robust Accuracy: 0.06000%
Test: Average loss: 0.0015, Accuracy: 9998/10000 (100%), Robust Accuracy: 6/10000 (0%)


Note here are we using the MIAT for encoder 
in the PGD and FGSM functions - I used the STD model for model_fn
It is only used for ground truth - so it's ok.

5 runs whole test dataset
Robust Accuracy: 21.93000%
Test: Average loss: 0.0034, Accuracy: 9997/10000 (100%), Robust Accuracy: 2193/10000 (22%)
Robust Accuracy: 21.90000%
Test: Average loss: 0.0034, Accuracy: 9997/10000 (100%), Robust Accuracy: 2190/10000 (22%)
Robust Accuracy: 21.87000%
Test: Average loss: 0.0034, Accuracy: 9997/10000 (100%), Robust Accuracy: 2187/10000 (22%)
Robust Accuracy: 22.01000%
Test: Average loss: 0.0034, Accuracy: 9997/10000 (100%), Robust Accuracy: 2201/10000 (22%)
Robust Accuracy: 22.02000%
Test: Average loss: 0.0034, Accuracy: 9997/10000 (100%), Robust Accuracy: 2202/10000 (22%)

For a single batch:
Robust Accuracy: 23.04688%
Test: Average loss: 0.0001, Accuracy: 512/10000 (5%), Robust Accuracy: 118/10000 (1%)

Using PGD with eps: 0.15, eps_iter: 0.007, nb_iter: 50

Name of this function is perhaps misleading as it gives TOTAL loss, not just the MI loss.
This is the first version of MI_loss_testXX that incorporates a euclidean distance metric
between the MI vectors - I'm not convinced it helps - but it doesn't hurt. See stats above.

:param model_fns: models for calculating loss
:param alpha: (optional) float. Hyper parameter for tuning the MI portion of loss.
'''


def MI_loss_test05(model_fns, x_natural, x_adv, y_true, iter=0):
    #this setting 
    # typical layout for model_fns
    #               0        1         2        3         4        5
    #model_fns = [std_res, miat_res, local_n, global_n, local_a, global_a]
    target_model = model_fns[0] #target model to calculate loss
    encoder = model_fns[0] # model to use as encoder
    local_n = model_fns[2]
    global_n = model_fns[3]
    local_a = model_fns[4]
    global_a = model_fns[5]
    #all models should already be in eval() mode
    alpha = 1.
    lambd = .1
    #print(f"alpha: {alpha}, lambda: {lambd}")
    global euc_flag

    logits_adv = target_model(x_adv) # current pred of the model we are attacking
    loss_ce = F.cross_entropy(logits_adv, y_true)

    #code snippet to select the indices of adv examples that are misclassified and clean samples that
    # are correctly classified
    pseudo_label = F.softmax(target_model(x_natural), dim=0).max(1, keepdim=True)[1].squeeze()
    index = (pseudo_label == y_true)
    pseudo_label = F.softmax(target_model(x_adv), dim=0).max(1, keepdim=True)[1].squeeze()
    index = index * (pseudo_label != y_true)

    #init vars for print in case we hit the "else"
    loss_mea_n = 0
    loss_mea_a = 0
    loss_a_all = 0
    loss_euclid_mea_n = 0
    loss_euclid_mea_a = 0
    
    if torch.nonzero(index).size(0) != 0:
        loss_n = compute_loss(args=args, former_input=x_natural, latter_input=x_natural, encoder=encoder,
                dim_local=local_n, dim_global=global_n, v_out=True)* index

        loss_a = compute_loss(args=args, former_input=x_natural, latter_input=x_adv, encoder=encoder,
                               dim_local=local_n, dim_global=global_n, v_out=True) * index

        loss_euclid_mea_n = torch.sqrt(sum((loss_n - loss_a)**2))
        
        loss_a_all = loss_a # for the 3rd term
        loss_mea_n = torch.abs(torch.tensor(1.0).cuda() - torch.cosine_similarity(loss_n, loss_a, dim=0))


        loss_a = compute_loss(args=args, former_input=x_adv - x_natural, latter_input=x_adv, encoder=encoder,
                              dim_local=local_a, dim_global=global_a, v_out=True) * index

        loss_n = compute_loss(args=args, former_input=x_adv - x_natural, latter_input=x_natural, encoder=encoder,
                              dim_local=local_a, dim_global=global_a, v_out=True) * index

        loss_euclid_mea_a = torch.sqrt(sum((loss_n - loss_a)**2))
        
        # I changed the order of subtraction here to match the paper as far as I can tell eqn 8
        loss_a_all = loss_a - loss_a_all
        loss_a_all = loss_a_all.sum()/(torch.nonzero(index).size(0))
        loss_a_all = torch.abs(torch.tensor(lambd).cuda() * loss_a_all)
        
        loss_mea_a = torch.abs(torch.tensor(1.0).cuda() - torch.cosine_similarity(loss_n, loss_a, dim=0))

        #here we do different loss depending on how far along we are.
        '''
        if iter < 10:
            loss_all = loss_ce
        elif iter < 25: #add in cosine based MI_loss 
            loss_all = loss_mea_n + loss_mea_a# + loss_a_all
        else: #switch cosine based for euclidean based MI_loss
            loss_all = .05*loss_euclid_mea_n + .1*loss_euclid_mea_a# + loss_a_all
        #loss_mi = loss_mea_n + loss_mea_a + loss_a_all + .01*loss_euclid_mea_n + .1*loss_euclid_mea_a
        #loss_mi = loss_euclid_mea_a + loss_euclid_mea_n
        '''
        beta = 0.5 #0. -> 99.41406%; .5 -> 19.14062% nb_iter = 250;  .75 -> 17.96875%; 1. -> 16.79688%
        #loss_all = loss_ce + .01*loss_euclid_mea_n + .1*loss_euclid_mea_a #23.82812%
        #loss_all = loss_ce + loss_a_all + .01*loss_euclid_mea_n + .1*loss_euclid_mea_a #24.80469%
        #loss_all = loss_ce + loss_a_all + .05*loss_euclid_mea_n + .1*loss_euclid_mea_a #27.92969
        #loss_all = loss_ce + alpha* (loss_mea_n + loss_mea_a + loss_a_all) + .01*loss_euclid_mea_n + .1*loss_euclid_mea_a # 27.53906%
        #loss_all = beta * loss_ce + (1-beta) * (alpha* (loss_mea_n + loss_mea_a + loss_a_all) + .01*loss_euclid_mea_n + .1*loss_euclid_mea_a) # Robust Accuracy: 26.75781%
        loss_all = beta * loss_ce + (1-beta) * (alpha* (loss_mea_n + loss_mea_a + loss_a_all) + .01*loss_euclid_mea_n + .1*loss_euclid_mea_a) # Robust Accuracy: 26.75781%
    else:
        loss_all = loss_ce
        
        
    #loss_all = loss_ce + torch.tensor(alpha).cuda() * loss_mi
    print(f"loss_all: {loss_all}, loss_ce: {loss_ce}, loss_mea_n: {loss_mea_n}, loss_mea_a: {loss_mea_a}, loss_a_all: {loss_a_all}, euc_n: {loss_euclid_mea_n}, euc_a: {loss_euclid_mea_a}")
    return loss_all


'''
Name of this function is perhaps misleading as it gives TOTAL loss, not just the MI loss.
:param model
:param alpha: (optional) float. Hyper parameter for tuning the MI portion of loss.
'''
def MI_loss(models, x_natural, y, y_true ,x_adv, local_n, global_n, local_a, global_a, alpha=5.0, lambd=0.1, iter=0):
    model = models[0] #changed to eval not train()
    model.eval()
    model2 = models[1]
    local_n.eval()
    global_n.eval()
    local_a.eval()
    global_a.eval()

    #if y_true is None:
    #    y_true = y

    # logits_nat = model(x_natural)
    logits_adv = model(x_adv)

    #loss_ce = F.cross_entropy(F.softmax(logits_adv, dim=1), y)#, reduction='mean')
    loss_ce = F.cross_entropy(logits_adv, y)#, reduction='mean')
    # loss_ce = 0.2 * F.cross_entropy(logits_nat, y) + 0.8 * F.cross_entropy(logits_adv, y)

    # I believe this little block of code is looking for the indices of the samples that are misclassified
    pseudo_label = F.softmax(model(x_natural), dim=0).max(1, keepdim=True)[1].squeeze()
    #print(f"pseudo_label: {pseudo_label}")
    index = (pseudo_label == y_true)
    pseudo_label = F.softmax(model(x_adv), dim=0).max(1, keepdim=True)[1].squeeze()
    index = index * (pseudo_label != y_true)
    #print(f"index: {index}")

    #if torch.nonzero(index).size(0) != 0:
    
    if True:
        
        #see equation 8, 9 - it looks like in the actual code implmentation they leave off the lambda term E_a(h(x)) - E_n(h(x))
        loss_n = compute_loss(args=args, former_input=x_natural, latter_input=x_natural, encoder=model,
                dim_local=local_n, dim_global=global_n, v_out=True)#* index

        loss_a = compute_loss(args=args, former_input=x_natural, latter_input=x_adv, encoder=model,
                               dim_local=local_n, dim_global=global_n, v_out=True)# * index

        loss_a_all = loss_a # added this back in it was commented out
        loss_mea_n = torch.abs(torch.tensor(1.0).cuda() - torch.cosine_similarity(loss_n, loss_a, dim=0))


        loss_a = compute_loss(args=args, former_input=x_adv - x_natural, latter_input=x_adv, encoder=model,
                              dim_local=local_a, dim_global=global_a, v_out=True) #* index

        loss_n = compute_loss(args=args, former_input=x_adv - x_natural, latter_input=x_natural, encoder=model,
                              dim_local=local_a, dim_global=global_a, v_out=True) #* index

        #loss_a_all = torch.tensor(0.1).cuda() * (loss_a_all - loss_a) #added back in - it was commented out
        lambd = .1
        alpha = 5.
        loss_a_all = torch.tensor(lambd).cuda() * torch.max((loss_a_all - loss_a)) #see line above, this is 3rd term but we take the MAX from the output
        #loss_a_all = torch.tensor(lambd).cuda() * torch.mean((loss_a_all - loss_a)) 
        loss_mea_a = torch.abs(torch.tensor(1.0).cuda() - torch.cosine_similarity(loss_n, loss_a, dim=0))

        #loss_mi = 100. * loss_mea_n + 10. * loss_mea_a  #+ loss_a_all 
        #loss_mi = loss_mea_n + loss_mea_a + loss_a_all
        loss_mi = loss_mea_a #+ loss_mea_n#+ loss_a_all
        #loss_mi = loss_a_all
        #loss_all = alpha * (loss_mi + loss_a_all)
        
        beta = .5
        if loss_mi < 1e-8:
            loss_all = loss_ce
        elif iter < 10:
            loss_all = loss_ce
        else:
            loss_all = loss_mi
        print(f"iter: {iter}, loss_ce: {loss_ce}, loss_mea_n: {loss_mea_n}, loss_mea_a: {loss_mea_a}, loss_a_all: {loss_a_all}, loss_all: {loss_all}")
        
        loss_all = alpha * loss_mi + loss_ce
    else:
        #print("we're in the else where loss_mi is set to 0!")
        loss_mi = 0.0
        loss_all = loss_ce
        #print(f"loss_all=loss_ce={loss_all}")
        
    # default is alpha = 5
    #loss_all = loss_ce + alpha * loss_mi # original line
    #loss_all = alpha * loss_mi
    loss_all = loss_ce
    return loss_all


#####################################################
#THIS CODE IS FROM CLEVERHANS
#####################################################

"""The Fast Gradient Method attack."""

def fast_gradient_method(
    model_fns,
    x,
    x_natural,
    eps,
    norm,
    clip_min=None,
    clip_max=None,
    y=None,
    y_true=None,
    targeted=False,
    sanity_checks=False,
    alpha=5.,
    iter=0,
):
    """
    PyTorch implementation of the Fast Gradient Method.
    :param model_fns: a list of callables that takes an input tensor and returns the model logits.
    :param x: input tensor. with PGD this is the current iteration of adversarial example.
    :param x_natural: the clean sample - needed for LOSS_MI
    :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
    :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
    :param clip_min: (optional) float. Minimum float value for adversarial example components.
    :param clip_max: (optional) float. Maximum float value for adversarial example components.
    :param y: (optional) Tensor with true labels. If targeted is true, then provide the
              target label. Otherwise, only provide this parameter if you'd like to use true
              labels when crafting adversarial samples. Otherwise, model predictions are used
              as labels to avoid the "label leaking" effect (explained in this paper:
              https://arxiv.org/abs/1611.01236). Default is None.
    :param y_true: Passed alont to LOSS_MI
    :param targeted: (optional) bool. Is the attack targeted or untargeted?
              Untargeted, the default, will try to make the label incorrect.
              Targeted will instead try to move in the direction of being more like y.
    :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
              memory or for unit tests that intentionally pass strange input)
    :param alpha: (optional) float. Hyper parameter for tuning the MI portion of loss.
    :param iter: (optional) int. Current iteration number for PGD.
    :return: a tensor for the adversarial example
    """
    if norm not in [np.inf, 1, 2]:
        raise ValueError(
            "Norm order must be either np.inf, 1, or 2, got {} instead.".format(norm)
        )
    if eps < 0:
        raise ValueError(
            "eps must be greater than or equal to 0, got {} instead".format(eps)
        )

    model_fn = model_fns[0] #[1] is miat - should have used a dictionary with a named key - this is python after all.
    
    if eps == 0:
        return x
    if clip_min is not None and clip_max is not None:
        if clip_min > clip_max:
            raise ValueError(
                "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                    clip_min, clip_max
                )
            )

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        assert_ge = torch.all(
            torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_ge)

    if clip_max is not None:
        assert_le = torch.all(
            torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_le)

    # x needs to be a leaf variable, of floating point type and have requires_grad being True for
    # its grad to be computed and stored properly in a backward call
    x = x.clone().detach().to(torch.float).requires_grad_(True)
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        _, y = torch.max(model_fn(x), 1)

    # Compute loss
    #loss_fn = torch.nn.CrossEntropyLoss()
    #loss = loss_fn(model_fn(x), y)
    #print(f"loss_ce: {loss}")
    loss = MI_loss_test05(model_fns, x_natural=x_natural, x_adv=x, y_true=y_true,iter=iter)
    # If attack is targeted, minimize loss of target label rather than maximize loss of correct label
    if targeted:
        loss = -loss

    # Define gradient of loss wrt input
    loss.backward()
    optimal_perturbation = optimize_linear(x.grad, eps, norm)

    # Add perturbation to original example to obtain adversarial example
    adv_x = x + optimal_perturbation

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) or (clip_max is not None):
        if clip_min is None or clip_max is None:
            raise ValueError(
                "One of clip_min and clip_max is None but we don't currently support one-sided clipping"
            )
        adv_x = torch.clamp(adv_x, clip_min, clip_max)

    if sanity_checks:
        assert np.all(asserts)
    return adv_x




"""The Projected Gradient Descent attack."""


def pgd(
    model_fns,
    x,
    eps,
    eps_iter,
    nb_iter,
    norm,
    clip_min=None,
    clip_max=None,
    y=None,
    y_true=None,
    targeted=False,
    rand_init=True,
    rand_minmax=None,
    sanity_checks=True,
):
    """
    This class implements either the Basic Iterative Method
    (Kurakin et al. 2016) when rand_init is set to False. or the
    Madry et al. (2017) method if rand_init is set to True.
    Paper link (Kurakin et al. 2016): https://arxiv.org/pdf/1607.02533.pdf
    Paper link (Madry et al. 2017): https://arxiv.org/pdf/1706.06083.pdf
    :param model_fn: a list of callables that takes an input tensor and returns the model logits.
    :param x: input tensor.
    :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572. schwab: Used for clipping when norm=np.inf
    :param eps_iter: step size for each attack iteration Schwab: I think this is multiplied by the sign of the gradient to add to the image.
    :param nb_iter: Number of attack iterations.
    :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
    :param clip_min: (optional) float. Minimum float value for adversarial example components.
    :param clip_max: (optional) float. Maximum float value for adversarial example components.
    :param y: (optional) Tensor with true labels. If targeted is true, then provide the
              target label. Otherwise, only provide this parameter if you'd like to use true
              labels when crafting adversarial samples. Otherwise, model predictions are used
              as labels to avoid the "label leaking" effect (explained in this paper:
              https://arxiv.org/abs/1611.01236). Default is None.
    :param y_true: The true label - passed on to LOSS_MI
    :param targeted: (optional) bool. Is the attack targeted or untargeted?
              Untargeted, the default, will try to make the label incorrect.
              Targeted will instead try to move in the direction of being more like y.
    :param rand_init: (optional) bool. Whether to start the attack from a randomly perturbed x.
    :param rand_minmax: (optional) bool. Support of the continuous uniform distribution from
              which the random perturbation on x was drawn. Effective only when rand_init is
              True. Default equals to eps.
    :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
              memory or for unit tests that intentionally pass strange input)
    :param alpha: (optional) float. Hyper parameter for tuning the MI portion of loss.
    :return: a tensor for the adversarial example
    """
    #model_fns[0] is the target model
    #model_fns[1] is the model to construct adv samples with with (same as model_fns[0] for white-box attack)
    model_fn = model_fns[0] #should have used a named key in a dictionary for this = back to miat.
    if norm == 1:
        raise NotImplementedError(
            "It's not clear that FGM is a good inner loop"
            " step for PGD when norm=1, because norm=1 FGM "
            " changes only one pixel at a time. We need "
            " to rigorously test a strong norm=1 PGD "
            "before enabling this feature."
        )
    if norm not in [np.inf, 2]:
        raise ValueError("Norm order must be either np.inf or 2.")
    if eps < 0:
        raise ValueError(
            "eps must be greater than or equal to 0, got {} instead".format(eps)
        )
    if eps == 0:
        return x
    if eps_iter < 0:
        raise ValueError(
            "eps_iter must be greater than or equal to 0, got {} instead".format(
                eps_iter
            )
        )
    if eps_iter == 0:
        return x

    assert eps_iter <= eps, (eps_iter, eps)
    if clip_min is not None and clip_max is not None:
        if clip_min > clip_max:
            raise ValueError(
                "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                    clip_min, clip_max
                )
            )

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        assert_ge = torch.all(
            torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_ge)

    if clip_max is not None:
        assert_le = torch.all(
            torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_le)

    # Initialize loop variables
    if rand_init:
        if rand_minmax is None:
            rand_minmax = eps
        #schwab: I think this goes from -rand_minmax to rand_minmax b/c you can lower or raise any single pixel by this value.
        eta = torch.zeros_like(x).uniform_(-rand_minmax, rand_minmax) 
    else:
        eta = torch.zeros_like(x)

    # Clip eta
    eta = clip_eta(eta, norm, eps)
    adv_x = x + eta
    if clip_min is not None or clip_max is not None:
        adv_x = torch.clamp(adv_x, clip_min, clip_max)

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        _, y = torch.max(model_fn(x), 1)

    i = 0
    while i < nb_iter:
        adv_x = fast_gradient_method(
            model_fns,
            adv_x,
            x, #x_natural
            eps_iter,
            norm,
            clip_min=clip_min,
            clip_max=clip_max,
            y=y,
            y_true=y_true,
            targeted=targeted,
            iter=i
        )

        # Clipping perturbation eta to norm norm ball #schwab: eta is the perturbation. Need to clip to Norm ball.
        eta = adv_x - x
        eta = clip_eta(eta, norm, eps)
        adv_x = x + eta

        # Redo the clipping.
        # FGM already did it, but subtracting and re-adding eta can add some
        # small numerical error.
        if clip_min is not None or clip_max is not None:
            adv_x = torch.clamp(adv_x, clip_min, clip_max)
        i += 1

    asserts.append(eps_iter <= eps)
    if norm == np.inf and clip_min is not None:
        # TODO necessary to cast clip_min and clip_max to x.dtype?
        asserts.append(eps + clip_min <= clip_max)

    if sanity_checks:
        assert np.all(asserts)
    return adv_x

#               0        1         2        3         4        5
#model_fns = [std_res, miat_res, local_n, global_n, local_a, global_a]
def eval_loss(model_fns, device, test_loader):
    std_res_fn = model_fns[0]
    miat_res_fn = model_fns[1]
    local_n = model_fns[2]
    global_n = model_fns[3]
    local_a = model_fns[4]
    global_a = model_fns[5]

    loss = []
    acc = []
    acc_adv = []

    #this is our L_infty constraint - added 1.5+ 
    #eps_lst = [.025, .05, .075, .1, .125, .15, .175, .2, .25, .3, .4, .5, .75, 1.] #, 1.5, 2., 2.5] #stopping at 1
    eps_lst = [.025, .05, .075, .1]

    for eps in eps_lst:
        #eps = .025
        print(25*'=')
        test_loss = 0
        correct = 0
        correct_adv = 0

        eps_iter = .007
        #eps_iter = .005
        nb_iter = round(eps/eps_iter) + 10
        #nb_iter = 40 
        print(f"Using PGD with eps: {eps}, eps_iter: {eps_iter}, nb_iter: {nb_iter}")
        #with torch.no_grad():
        i = 0
        for x_natural, y_true in test_loader:
            i = i+1
            print(f"batch number {i}, {i*args.batch_size} / {len(test_loader.dataset)}")
            x_natural, y_true = x_natural.to(device), y_true.to(device)
            
            x_adv = pgd(model_fns, x_natural, eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, norm=np.inf, y=None, y_true=y_true, targeted=False)
            
            #x_adv = data_adv.detach().cpu().numpy().transpose(0,2,3,1) #I'll use this later - gonna paste all the images together.
            #y_nat_pred = miat_res_fn(x_natural)
            #y_adv_pred = miat_res_fn(x_adv)
            #next two lines are in case you want to eval the standard resnet
            y_nat_pred = std_res_fn(x_natural)
            y_adv_pred = std_res_fn(x_adv)
            
            
            test_loss += F.cross_entropy(y_nat_pred, y_true, reduction='sum').item()
            pred = y_nat_pred.max(1, keepdim=True)[1]
            pred_adv = y_adv_pred.max(1, keepdim=True)[1]
            correct += pred.eq(y_true.view_as(pred)).sum().item()
            correct_adv += pred_adv.eq(y_true.view_as(pred_adv)).sum().item()
            #if i==1:
            #     break
            
        l = i*args.batch_size 
        if l > len(test_loader.dataset):
            l = len(test_loader.dataset)
        
        print("Robust Accuracy: {:.5f}%".format(100*correct_adv / (l)))
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
    #first lets load the data - clean and adversarial examples
    
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    print(f"device: {device}") #sanity check - using GPU

    trans_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats, inplace=False)
    ])
    print(f"Loading data ...")
    #not going to augment the data for now - don't think I need to
    testset=data_dataset(img_path=args.nat_img_test, clean_label_path=args.nat_label_test, transform=trans_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, drop_last=False, shuffle=False, num_workers=4, pin_memory=True)
    print("Data loaded!")

    print(f"Loading Models to {device} ...")
    #load resnet models
    std_res = ResNet18(10)
    miat_res = ResNet18(10)

    # Estimator part 1: X or layer3 to H space
    local_n = Estimator(args.va_hsize)
    local_a = Estimator(args.va_hsize)

    # estimator part 2: Z to H space
    z_size = 10
    global_n = MI1x1ConvNet(z_size, args.va_hsize)
    global_a = MI1x1ConvNet(z_size, args.va_hsize)

    std_res_name = 'resnet-new-100' 
    miat_res_name = 'resnet-new-100-MIAT-from-scratch'
    l_n = 'local_n.5'
    g_n = 'global_n.5'
    l_a = 'local_a.5'
    g_a = 'global_a.5'

    std_res.load_state_dict(torch.load(os.path.join(args.SAVE_MODEL_PATH, std_res_name)))
    miat_res.load_state_dict(torch.load(os.path.join(args.SAVE_MODEL_PATH, miat_res_name)))
    local_n.load_state_dict(torch.load(os.path.join(args.SAVE_MODEL_PATH, l_n)))
    global_n.load_state_dict(torch.load(os.path.join(args.SAVE_MODEL_PATH, g_n)))
    local_a.load_state_dict(torch.load(os.path.join(args.SAVE_MODEL_PATH, l_a)))
    global_a.load_state_dict(torch.load(os.path.join(args.SAVE_MODEL_PATH, g_a)))
    
    print(f"Resnet Models Loaded: {std_res_name} {miat_res_name}")
    print(f"Estimator Models Loaded: {l_n} {g_n} {l_a} {g_a}")
    
    std_res = torch.nn.DataParallel(std_res).cuda()
    miat_res = torch.nn.DataParallel(miat_res).cuda()
    local_n = torch.nn.DataParallel(local_n).cuda()
    global_n = torch.nn.DataParallel(global_n).cuda()
    local_a = torch.nn.DataParallel(local_a).cuda()
    global_a = torch.nn.DataParallel(global_a).cuda()

    std_res.eval()
    miat_res.eval()
    local_n.eval()
    global_n.eval()
    local_a.eval()
    global_a.eval()

    cudnn.benchmark = True
    
    print(f"Models Loaded!")

    print(f"Evaluating Loss Functions...")
    #               0        1         2        3         4        5
    model_fns = [std_res, miat_res, local_n, global_n, local_a, global_a]
    eval_loss(model_fns, device, test_loader)

if __name__ == '__main__':
    main()
