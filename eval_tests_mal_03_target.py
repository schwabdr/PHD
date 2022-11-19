'''
I didn't take very good notes.
DON'T USE THIS FILE.
FOR eval_tests_mal_03 instead use
********************loss_study_targeted_mal.py***********************


The purpose of this new file: eval_tests_03_target.py
5) Two phases - targeted attack phase (CE loss only) then non-targeted attack phase (MI-Craft)


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
from utils.data import data_mal_sample_dataset
from utils.utils import optimize_linear
from utils.utils import clip_eta

#next is only needed to visualize samples
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

import random

args = config.Configuration().getArgs()

stats = ((0.4454, 0.4454, 0.4454), (0.3122, 0.3122, 0.3122)) #mean and stdev

args.batch_size = 48 #10 is max on Lambda workstations.

classes = ['Adialer.C', 'Agent.FYI', 'Allaple.A', 'Allaple.L', 'Alueron.gen!J', 'Autorun.K', 'C2LOP.P', 'C2LOP.gen!g', 'Dialplatform.B', 'Dontovo.A', 'Fakerean', 
                'Instantaccess', 'Lolyda.AA1', 'Lolyda.AA2', 'Lolyda.AA3', 'Lolyda.AT', 'Malex.gen!J', 'Obfuscator.AD', 'Rbot!gen', 'Skintrim.N', 'Swizzor.gen!E', 'Swizzor.gen!I', 
                'VB.AT', 'Wintrim.BX', 'Yuner.A']
#####################################################
#THIS CODE IS FROM train_MIAT_alpha.py
#####################################################
'''

Name of this function is perhaps misleading as it gives TOTAL loss, not just the MI loss.
This will be my first test - a baseline. Once I tune it I'll work on other losses.
Uses the same MI_loss that is used for MIAT/NAMID training (CE+3MI-Terms)
:param model_fns: models for calculating loss
:param alpha: (optional) float. Hyper parameter for tuning the MI portion of loss.
'''

'''
MI_loss2 - 
This was done as a tuning for the MI-Craft Method.
1) calculate the MI based loss terms for ALL images, not just the misclassified ones.
2) add 'alpha' parameters to remove pieces as we go along
3) 


Following settings on single batch achieved 31.83594% robust accuracy at eps = 0.1
#default lambdas for now - use cosine loss only
        l1 = 1
        l2 = 1
        l3 = 0
        l4 = 0
        l5 = 0
        #if above threshold - switch to Euclidean loss
        if loss_mea_n > 1.5:
            l1 = 0
            l3 = .01
        if loss_mea_a > 1.5:
            l2 = 0
            l4 = .1
        beta = .5
        loss_all = beta * loss_ce + (1-beta) * 5.0 * (l1*loss_mea_n + l2*loss_mea_a + l3*loss_euclid_mea_n + l4*loss_euclid_mea_a + l5*loss_a_all)

But the Euclidean loss was never used - the cosine loss never got high enough.

Name of this function is perhaps misleading as it gives TOTAL loss, not just the MI loss.
:param model
:param alpha: (optional) float. Hyper parameter for tuning the MI portion of loss.
'''
#               0        1         2        3         4        5
#model_fns = [std_res, miat_res, local_n, global_n, local_a, global_a]
def MI_loss2(model_fns, x_natural, y_true ,x_adv, alpha=5.0, lambd=0.1, iter=0, x_target=None):
    model = model_fns[1] #[0] is std [1] is MIAT
    encoder = model_fns[0]
    local_n = model_fns[2]
    global_n = model_fns[3]
    local_a = model_fns[4]
    global_a = model_fns[5]

    logits_adv = model(x_adv)
    
    loss_ce = F.cross_entropy(logits_adv, y_true)#, reduction='mean')
    
    # I believe this little block of code is looking for the indices of the samples that are misclassified
    pseudo_label = F.softmax(model(x_natural), dim=0).max(1, keepdim=True)[1].squeeze()
    index = (pseudo_label == y_true)
    pseudo_label = F.softmax(model(x_adv), dim=0).max(1, keepdim=True)[1].squeeze()
    index = index * (pseudo_label != y_true)
    
    #init vars 
    loss_mea_n = 0.0 #torch.tensor(0.0).cuda()  #lambda1
    loss_mea_a = 0.0 #torch.tensor(0.0).cuda() #lambda2
    loss_euclid_mea_n = 0.0 #torch.tensor(0.0).cuda() #lambda3 default .01
    loss_euclid_mea_a = 0.0 #torch.tensor(0.0).cuda() #lambda4 default .1
    loss_a_all = 0.0 #torch.tensor(0.0).cuda() #lambda5 default 1
    
    if torch.nonzero(index).size(0) != 0:
        #see equation 8, 9 - it looks like in the actual code implmentation they leave off the lambda term E_a(h(x)) - E_n(h(x))
        loss_n = compute_loss(args=args, former_input=x_natural, latter_input=x_target, encoder=encoder,
                dim_local=local_n, dim_global=global_n, v_out=True) * index
        loss_a = compute_loss(args=args, former_input=x_natural, latter_input=x_adv, encoder=encoder,
                               dim_local=local_n, dim_global=global_n, v_out=True) * index
        
        loss_euclid_mea_n = torch.sqrt(sum((loss_n - loss_a)**2))

        loss_a_all = loss_a # added this back in it was commented out
        loss_mea_n = torch.abs(torch.tensor(1.0).cuda() - torch.cosine_similarity(loss_n, loss_a, dim=0))

        loss_a = compute_loss(args=args, former_input=x_adv - x_natural, latter_input=x_adv, encoder=encoder,
                              dim_local=local_a, dim_global=global_a, v_out=True) * index
        loss_n = compute_loss(args=args, former_input=x_adv - x_natural, latter_input=x_target, encoder=encoder,
                              dim_local=local_a, dim_global=global_a, v_out=True) * index
        
        loss_euclid_mea_a = torch.sqrt(sum((loss_n - loss_a)**2))

        alpha = 5.
        loss_a_all = (loss_a - loss_a_all)
        loss_a_all = loss_a_all.sum()/(torch.nonzero(index).size(0)) #so this is an avg
        #loss_a_all = torch.abs(torch.tensor(.1).cuda() * loss_a_all)
        loss_a_all = torch.abs(loss_a_all)
        
        loss_mea_a = torch.abs(torch.tensor(1.0).cuda() - torch.cosine_similarity(loss_n, loss_a, dim=0))
        
        #lambdas
        l1 = 1
        l2 = 1
        l3 = .01
        l4 = .1
        l5 = .1

        #if iter < 10:
        #    loss_all = loss_ce
        #elif iter < 25: #add in cosine based MI_loss 
        loss_all =  loss_ce + 5 * (loss_mea_n + loss_mea_a + .1 * loss_a_all) + l3 * loss_euclid_mea_n + l4 * loss_euclid_mea_a
        #else: #switch cosine based for euclidean based MI_loss
        #    loss_all = loss_ce +  (.01*loss_euclid_mea_n + .1*loss_euclid_mea_a + .1 * loss_a_all)
        #loss_all = torch.tensor(loss_all).cuda()

        '''
        #if above threshold - switch to Euclidean loss
        if loss_mea_n > 1.5:
            l1 = 0
            l3 = .01
        if loss_mea_a > 1.5:
            l2 = 0
            l4 = .1
        
        beta = .5

        loss_all = beta * loss_ce + (1-beta) * 5.0 * (l1*loss_mea_n + l2*loss_mea_a + l3*loss_euclid_mea_n + l4*loss_euclid_mea_a + l5*loss_a_all)
        #loss_all = 5.0 * (l1*loss_mea_n + l2*loss_mea_a + l3*loss_euclid_mea_n + l4*loss_euclid_mea_a + l5*loss_a_all)
        '''
    else:
        loss_all = loss_ce
        
    print(f"iter: {iter}, loss_all: {loss_all}, loss_ce: {loss_ce}, loss_mea_n: {loss_mea_n}, loss_mea_a: {loss_mea_a}, euc_n: {loss_euclid_mea_n}, euc_a: {loss_euclid_mea_a}, loss_a_all: {loss_a_all}")
    
    return loss_all




'''
MI_loss - use original MIAT training
This was done as a baseline for the MI-Craft Method.

Name of this function is perhaps misleading as it gives TOTAL loss, not just the MI loss.
:param model
:param alpha: (optional) float. Hyper parameter for tuning the MI portion of loss.
'''
#               0        1         2        3         4        5
#model_fns = [std_res, miat_res, local_n, global_n, local_a, global_a]
def MI_loss(model_fns, x_natural, y_true ,x_adv, alpha=5.0, lambd=0.1, iter=0):
    model = model_fns[1] #[0] is std
    
    local_n = model_fns[2]
    global_n = model_fns[3]
    local_a = model_fns[4]
    global_a = model_fns[5]

    logits_adv = model(x_adv)
    
    loss_ce = F.cross_entropy(logits_adv, y_true)#, reduction='mean')
    
    # I believe this little block of code is looking for the indices of the samples that are misclassified
    pseudo_label = F.softmax(model(x_natural), dim=0).max(1, keepdim=True)[1].squeeze()
    index = (pseudo_label == y_true)
    pseudo_label = F.softmax(model(x_adv), dim=0).max(1, keepdim=True)[1].squeeze()
    index = index * (pseudo_label != y_true)
    
    loss_mea_n = 0
    loss_mea_a = 0
    loss_a_all = 0 
    
    
    if torch.nonzero(index).size(0) != 0:
        #see equation 8, 9 - it looks like in the actual code implmentation they leave off the lambda term E_a(h(x)) - E_n(h(x))
        loss_n = compute_loss(args=args, former_input=x_natural, latter_input=x_natural, encoder=model,
                dim_local=local_n, dim_global=global_n, v_out=True) * index

        loss_a = compute_loss(args=args, former_input=x_natural, latter_input=x_adv, encoder=model,
                               dim_local=local_n, dim_global=global_n, v_out=True) * index

        loss_a_all = loss_a # added this back in it was commented out
        loss_mea_n = torch.abs(torch.tensor(1.0).cuda() - torch.cosine_similarity(loss_n, loss_a, dim=0))


        loss_a = compute_loss(args=args, former_input=x_adv - x_natural, latter_input=x_adv, encoder=model,
                              dim_local=local_a, dim_global=global_a, v_out=True) * index

        loss_n = compute_loss(args=args, former_input=x_adv - x_natural, latter_input=x_natural, encoder=model,
                              dim_local=local_a, dim_global=global_a, v_out=True) * index

        lambd = .1
        alpha = 5.
        loss_a_all = (loss_a_all - loss_a)
        loss_a_all = loss_a_all.sum()/(torch.nonzero(index).size(0)) #so this is an avg
        loss_a_all = torch.abs(torch.tensor(.1).cuda() * loss_a_all)
        #loss_a_all = torch.tensor(lambd).cuda() * torch.max((loss_a_all - loss_a)) #see line above, this is 3rd term but we take the MAX from the output
        
        loss_mea_a = torch.abs(torch.tensor(1.0).cuda() - torch.cosine_similarity(loss_n, loss_a, dim=0))
        loss_mi = loss_mea_a + loss_mea_n + loss_a_all
        
        loss_all = loss_ce + alpha * loss_mi
    else:
        loss_all = loss_ce
        
    print(f"iter: {iter}, loss_all: {loss_all}, loss_ce: {loss_ce}, loss_mea_n: {loss_mea_n}, loss_mea_a: {loss_mea_a}, loss_a_all: {loss_a_all}")
    
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
    loss_fn=None,
    x_target=None,
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
    :param y_true: Passed along to LOSS_MI
    :param targeted: (optional) bool. Is the attack targeted or untargeted?
              Untargeted, the default, will try to make the label incorrect.
              Targeted will instead try to move in the direction of being more like y.
    :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
              memory or for unit tests that intentionally pass strange input)
    :param alpha: (optional) float. Hyper parameter for tuning the MI portion of loss.
    :param iter: (optional) int. Current iteration number for PGD.
    :param loss_fn: (optional) the loss function to use. Default will be cross-entropy.
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
    
    model_fn = model_fns[1] #[1] is miat [0] is std
    
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
    loss = loss_fn(model_fns, x_natural=x_natural, x_adv=x, y_true=y_true,iter=iter,x_target=x_target)
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
    loss_fn=None,
    x_target=None,
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
    model_fn = model_fns[1] # 0 is STD, 1 is MIAT
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
            iter=i,
            loss_fn=loss_fn,
            x_target=x_target
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
#model_fns = [std_res, miat_res, local_n, global_n, local_a, global_a] #now a dictionary but indexing should work.
def eval_loss(model_fns, device, test_loader, loss_fn, sample_loader):
    target_model_fn = model_fns[1]
    
    loss = []
    acc = []
    acc_adv = []

    eps_lst = [.025, .05, .075, .1, .125, .15, .175, .2, .25, .3, .4, .5, .75, 1.]
    #eps_lst = [.025, .05, .075, .1]
    #eps_lst = [.1]

    #loop to find one sample from each class in the test set.
    #this will be used for MI estimation
    x_base = []
    y_base = []
    for x_samples, y_samples in sample_loader:
        for x,y in zip(x_samples, y_samples):
            x.to(device)
            y.to(device)
            x_base.append(x)
            y_base.append(y)
        break #only need one batch that has all 25 class examples
    print(f"x_base len: {len(x_base)}")
    for eps in eps_lst:
        #eps = .025
        print(25*'=')
        test_loss = 0
        correct = 0
        correct_adv = 0

        eps_iter = .005
        nb_iter = 150
        print(f"Using PGD with eps: {eps}, eps_iter: {eps_iter}, nb_iter: {nb_iter}")
        #with torch.no_grad():
        i = 0
        for x_natural, y_true in test_loader:
            #print(f"batch length: {x_natural.shape[0]}")
            batch_len = x_natural.shape[0]
            i = i+1
            print(f"batch number {i}, {i*args.batch_size} / {len(test_loader.dataset)}")
            x_natural, y_true = x_natural.to(device), y_true.to(device)
            output = target_model_fn(x_natural) #fwd pass
            top2 = torch.topk(output, 2) # get values / indices for top 2 classes
            y_target = torch.select(top2.indices, 1, 1) #y_target is second most likely class
            #y_target = torch.select(top2.indices, 1, 9) #y_target is least likely class
            #y_target2 = torch.argmin(output, 0, keepdim=True)
            #print(f"y_true[0]: {y_true[0]}")
            #print(f"y_target[0]: {y_target[0]}")
            first = True
            for y_tar in y_target:
                #print(y.item())
                #set loop variable j and count while iterating through y_true
                #find a y_true.item() == y_target.item()
                #then stack or concat the corresponding tensor for x to create x_target, j will be the index of the image to take.
                #goal is to find the index of a label in y_true that is equal to y_target label
                # we have that label now because we created a list of ordered samples one from each class.
                #use that index to stack X
                if first:
                    #print(f"y_tar.item(): {y_tar.item()}")
                    x_target = x_base[y_tar.item()] #shape is [1,3,224,224]
                    #x_target = x_target.unsqueeze(0) #[1,3,32,32]
                    first = False
                else: #https://stackoverflow.com/questions/54307225/whats-the-difference-between-torch-stack-and-torch-cat
                    x_target = torch.cat([x_target, x_base[y_tar.item()]], dim=0)  
                
                #print(f"x_target.shape: {x_target.shape}") #last iteration is [batch_size, 3, 224, 224] as desired
            
            #two rounds of adv crafting 
            # 1) maximize loss for correct label
            # 2) minimize loss for second most likely class
            #x_adv = pgd(model_fns, x_natural, eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, norm=np.inf, y=None, y_true=y_true, targeted=False, x_target=x_natural)
            #x_adv = pgd(model_fns, x_adv, eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, norm=np.inf, y=y_target, y_true=y_true, targeted=True, x_target=x_target)
            
            # round 1
            x_adv = pgd(model_fns, x_natural, eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, norm=np.inf, y=None, y_true=y_true, targeted=False, rand_init=True, loss_fn=loss_fn, x_target=x_natural)
            #round 2
            x_adv = x_adv.clone().detach().to(torch.float).requires_grad_(True) #had to do this - it said I was calling backward twice or something.
            x_adv = pgd(model_fns, x_adv, eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, norm=np.inf, y=y_target, y_true=y_true, targeted=True, rand_init=False, loss_fn=loss_fn, x_target=x_target)
            
            #x_adv = data_adv.detach().cpu().numpy().transpose(0,2,3,1) #I'll use this later - gonna paste all the images together.
            #y_nat_pred = miat_res_fn(x_natural)
            #y_adv_pred = miat_res_fn(x_adv)
            #next two lines are in case you want to eval the standard resnet
            y_nat_pred = target_model_fn(x_natural)
            y_adv_pred = target_model_fn(x_adv)
            
            
            test_loss += F.cross_entropy(y_nat_pred, y_true, reduction='sum').item()
            pred = y_nat_pred.max(1, keepdim=True)[1]
            pred_adv = y_adv_pred.max(1, keepdim=True)[1]
            correct += pred.eq(y_true.view_as(pred)).sum().item()
            correct_adv += pred_adv.eq(y_true.view_as(pred_adv)).sum().item()
            #if i==1:
            #    break
            
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
    testset = data_dataset(img_path=args.nat_img_test_mal, clean_label_path=args.nat_label_test_mal, transform=trans_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, drop_last=False, shuffle=False, num_workers=4, pin_memory=True)
    print("Data loaded!")

    sample_set=data_mal_sample_dataset(img_path=args.nat_img_test_mal, clean_label_path=args.nat_label_test_mal, transform=trans_test)
    #note batchsize = 1 below because we get back a list of 25 images at once - we only need 1 batch.
    sample_loader = torch.utils.data.DataLoader(sample_set, batch_size=1, drop_last=False, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Loading Models to {device} ...")
    #load resnet models
    std_res = ResNet18(25)
    miat_res = ResNet18(25)

    # Estimator part 1: X or layer3 to H space
    local_n = Estimator(args.va_hsize)
    local_a = Estimator(args.va_hsize)

    # estimator part 2: Z to H space
    z_size = 25
    global_n = MI1x1ConvNet(z_size, args.va_hsize)
    global_a = MI1x1ConvNet(z_size, args.va_hsize)

    #           [0]                     [1]                         [2]                     [3]
    names = ['resnet-mal-std-100', 'resnet-mal-std-aug-100', 'resnet-mal-MIAT.25', 'resnet-mal-MIAT-AT.25.40']

    std_res_name = names[0]
    miat_res_name = names[3]
    l_n = 'local_n_mal.25'
    g_n = 'global_n_mal.25'
    l_a = 'local_a_mal.25'
    g_a = 'global_a_mal.25'

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
    #model_fns = {'std': std_res, 'miat': miat_res, 'local_n':local_n, 'global_n': global_n, 'local_a': local_a, 'global_a': global_a}
    loss_fn = MI_loss2

    loss, acc, acc_adv = eval_loss(model_fns, device, test_loader, loss_fn, sample_loader)

    ## TEST THIS BEFORE RUNNING THE WHOLE THING
    
    print("Clean Accuracy:")
    for x in acc:
        print(x)
    print("Adversarial Accuracy:")
    for x in acc_adv:
        print(x)
    

if __name__ == '__main__':
    main()
