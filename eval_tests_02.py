'''
The purpose of this new file: eval_tests_02.py
1) Evaluate MI-Craft under same LOSS function as NAMID/MIAT
2) Evalute MI-Craft using ALL samples for LOSS_MI not just the samples that are misclassified by the target.
3) Adaptively switch between Euclidean loss and cosine loss.
4) Perform targeted attacks.
5) Two phases - non-targeted attack phase then targeted attack phase.




#################### original notes from loss_study.py BELOW THIS LINE ###############################
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

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

args = config.Configuration().getArgs()
stats = config.Configuration().getNormStats() #
classes = config.Configuration().getClasses()

args.batch_size = 512 #512 worked, 704 crashed
f = False
# these are the indices randomly selected from the first batch - I printed them to save for all future tests.
a =  [365, 234, 109, 63, 90, 94, 265, 238, 198, 129, 35, 56, 372, 478, 427, 329, 165, 27, 300, 22, 462, 486, 31, 187, 43, 330, 179, 263, 6, 328, 384, 97, 285, 295, 92, 409, 272, 68, 139, 503, 354, 289, 65, 36, 117, 78, 442, 80, 406, 13, 348, 7, 11, 506, 459, 414, 436, 168, 284, 41, 370, 45, 293, 418, 480, 212, 341, 99, 135, 258, 89, 46, 323, 146, 134, 182, 339, 73, 487, 507, 26, 269, 143, 297, 349, 131, 227, 38, 237, 359, 426, 305, 77, 428, 24, 229, 358, 108, 244, 448, 9, 16, 120, 247, 152, 18, 140, 350, 389, 302, 55, 193, 124, 484, 88, 458, 403, 113, 47, 322, 125, 388, 489, 438, 421, 127, 312, 211, 412, 2, 277, 386, 53, 132, 192, 493, 225, 67, 275, 144, 268, 206, 173, 282, 317, 156, 153, 253, 242, 402, 391, 502, 315, 176, 326, 304, 171, 4, 149, 501, 380, 71, 385, 376, 208, 114, 452, 86, 419, 207, 76, 505, 290, 93, 495, 408, 130, 226, 202, 334, 169, 62, 363, 357, 101, 344, 210, 463, 316, 278, 460, 307, 72, 510, 228, 59, 195, 106, 451, 82, 17, 340, 85, 61, 471, 343, 183, 355, 447, 252, 240, 485, 469, 383, 443, 54, 369, 121, 267, 103, 218, 8, 74, 296, 175, 201, 281, 141, 496, 405, 167, 416, 338, 221, 83, 394, 236, 21, 461, 498, 137, 70, 230, 324, 214, 186, 194, 352, 320, 327, 353, 366, 450, 147, 399, 470, 142, 190, 29, 116, 245, 368, 177, 262, 455, 239, 204, 104, 488, 456, 429, 174, 356, 48, 360, 170, 500, 410, 128, 475, 199, 504, 69, 294, 508, 292, 28, 32, 115, 44, 306, 331, 361, 243, 422, 261, 390, 318, 260, 308, 345, 271, 248, 232, 23, 479, 122, 465, 332, 298, 446, 203, 259, 467, 424, 110, 241, 155, 33, 220, 401, 301, 51, 10, 437, 215, 466, 181, 362, 374, 273, 197, 396, 347, 415, 0, 377, 255, 160, 417, 439, 472, 381, 25, 433, 178, 509, 196, 373, 162, 79, 1, 34, 407, 145, 303, 299, 404, 57, 477, 159, 158, 42, 157, 336, 335, 432, 52, 185, 30, 393, 474, 314, 473, 440, 280, 20, 287, 12, 430, 367, 172, 453, 256, 497, 333, 161, 180, 398, 309, 392, 310, 37, 150, 75, 209, 337, 375, 188, 270, 191, 222, 274, 163, 400, 291, 420, 371, 283, 264, 148, 464, 387, 482, 279, 319, 233, 288, 257, 250, 235, 423, 483, 444, 254, 223, 395, 96, 39, 231, 166, 311, 313, 219, 382, 164, 249, 95, 205, 100, 492, 511, 251, 351, 91, 266, 87, 217, 84, 490, 449, 499, 107, 397, 14, 118, 123, 126, 216, 151, 58, 5, 105, 184, 468, 431, 138, 60, 378, 98, 64, 102, 481, 19, 454, 189, 154, 491, 411, 346, 434, 81, 476, 49, 286, 136, 325, 413, 379, 342, 15, 276, 112, 321, 111, 246, 224, 364, 66, 435, 213, 494, 441, 40, 50, 119, 445, 133, 3, 457, 425, 200]
b =  [365, 234, 109, 63, 90, 94, 265, 238, 198, 129, 35, 56, 372, 478, 427, 329, 165, 27, 300, 22, 462, 486, 31, 187, 43, 330, 179, 263, 6, 328, 384, 97, 285, 295, 92, 409, 272, 68, 139, 503, 354, 289, 65, 36, 117, 78, 442, 80, 406, 13, 348, 7, 11, 506, 459, 414, 436, 168, 284, 41, 370, 45, 293, 418, 480, 212, 341, 99, 135, 258, 89, 46, 323, 146, 134, 182, 339, 73, 487, 507, 26, 269, 143, 297, 349, 131, 227, 38, 237, 359, 426, 305, 77, 428, 24, 229, 358, 108, 244, 448, 9, 16, 120, 247, 152, 18, 140, 350, 389, 302, 55, 193, 124, 484, 88, 458, 403, 113, 47, 322, 125, 388, 489, 438, 421, 127, 312, 211, 412, 2, 277, 386, 53, 132, 192, 493, 225, 67, 275, 144, 268, 206, 173, 282, 317, 156, 153, 253, 242, 402, 391, 502, 315, 176, 326, 304, 171, 4, 149, 501, 380, 71, 385, 376, 208, 114, 452, 86, 419, 207, 76, 505, 290, 93, 495, 408, 130, 226, 202, 334, 169, 62, 363, 357, 101, 344, 210, 463, 316, 278, 460, 307, 72, 510, 228, 59, 195, 106, 451, 82, 17, 340, 85, 61, 471, 343, 183, 355, 447, 252, 240, 485, 469, 383, 443, 54, 369, 121, 267, 103, 218, 8, 74, 296, 175, 201, 281, 141, 496, 405, 167, 416, 338, 221, 83, 394, 236, 21, 461, 498, 137, 70, 230, 324, 214, 186, 194, 352, 320, 327, 353, 366, 450, 147, 399, 470, 142, 190, 29, 116, 245, 368, 177, 262, 455, 239, 204, 104, 488, 456, 429, 174, 356, 48, 360, 170, 500, 410, 128, 475, 199, 504, 69, 294, 508, 292, 28, 32, 115, 44, 306, 331, 361, 243, 422, 261, 390, 318, 260, 308, 345, 271, 248, 232, 23, 479, 122, 465, 332, 298, 446, 203, 259, 467, 424, 110, 241, 155, 33, 220, 401, 301, 51, 10, 437, 215, 466, 181, 362, 374, 273, 197, 396, 347, 415, 0, 377, 255, 160, 417, 439, 472, 381, 25, 433, 178, 509, 196, 373, 162, 79, 1, 34, 407, 145, 303, 299, 404, 57, 477, 159, 158, 42, 157, 336, 335, 432, 52, 185, 30, 393, 474, 314, 473, 440, 280, 20, 287, 12, 430, 367, 172, 453, 256, 497, 333, 161, 180, 398, 309, 392, 310, 37, 150, 75, 209, 337, 375, 188, 270, 191, 222, 274, 163, 400, 291, 420, 371, 283, 264, 148, 464, 387, 482, 279, 319, 233, 288, 257, 250, 235, 423, 483, 444, 254, 223, 395, 96, 39, 231, 166, 311, 313, 219, 382, 164, 249, 95, 205, 100, 492, 511, 251, 351, 91, 266, 87, 217, 84, 490, 449, 499, 107, 397, 14, 118, 123, 126, 216, 151, 58, 5, 105, 184, 468, 431, 138, 60, 378, 98, 64, 102, 481, 19, 454, 189, 154, 491, 411, 346, 434, 81, 476, 49, 286, 136, 325, 413, 379, 342, 15, 276, 112, 321, 111, 246, 224, 364, 66, 435, 213, 494, 441, 40, 50, 119, 445, 133, 3, 457, 425, 200]

def show_img_grid(rows, cols, x, x_adv, y, y_adv,fname=None):
    fig, axes1 = plt.subplots(rows,cols,figsize=(25,25))
    global f
    global a
    global b
    if f:
        lst = list(range(0, len(x)))
        random.shuffle(lst)
        a = lst.copy()
        b = lst.copy()
        print(f"Using these indices from first batch: {a}")
        f = False
    #print("min/max of numpy arrays: ", np.min(x), np.max(x)) #this was very close to 0 and 1
    for j in range(rows):
        for k in range(0,cols,2):
            #get a random index
            i = a.pop()
            axes1[j][k].set_axis_off()
            axes1[j][k+1].set_axis_off()
            axes1[j][k].imshow(x[i],interpolation='nearest')
            axes1[j][k].text(0,0,classes[y[i]]) # this gets the point accross but needs fixing.
            axes1[j][k+1].imshow(x_adv[i], interpolation='nearest')
            axes1[j][k+1].text(0,0,classes[y_adv[i]])
    a = b.copy()

    if fname is None:
        plt.show()
    else:
        plt.savefig(fname, format='png')
        plt.close()


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
def MI_loss2(model_fns, x_natural, y_true ,x_adv, alpha=5.0, lambd=0.1, iter=0):
    model = model_fns[1] #[0] is std
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
    loss_mea_n = 0  #lambda1
    loss_mea_a = 0 #lambda2
    loss_euclid_mea_n = 0 #lambda3 default .01
    loss_euclid_mea_a = 0 #lambda4 default .1
    loss_a_all = 0 #lambda5 default 1
    
    
    
    if torch.nonzero(index).size(0) != 0:
        #see equation 8, 9 - it looks like in the actual code implmentation they leave off the lambda term E_a(h(x)) - E_n(h(x))
        loss_n = compute_loss(args=args, former_input=x_natural, latter_input=x_natural, encoder=encoder,
                dim_local=local_n, dim_global=global_n, v_out=True) * index
        loss_a = compute_loss(args=args, former_input=x_natural, latter_input=x_adv, encoder=encoder,
                               dim_local=local_n, dim_global=global_n, v_out=True) * index
        
        loss_euclid_mea_n = torch.sqrt(sum((loss_n - loss_a)**2))

        loss_a_all = loss_a # added this back in it was commented out
        loss_mea_n = torch.abs(torch.tensor(1.0).cuda() - torch.cosine_similarity(loss_n, loss_a, dim=0))

        loss_a = compute_loss(args=args, former_input=x_adv - x_natural, latter_input=x_adv, encoder=encoder,
                              dim_local=local_a, dim_global=global_a, v_out=True) * index
        loss_n = compute_loss(args=args, former_input=x_adv - x_natural, latter_input=x_natural, encoder=encoder,
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
    model = model_fns[1] #[0] is std [1] is MIAT
    
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
    loss = loss_fn(model_fns, x_natural=x_natural, x_adv=x, y_true=y_true,iter=iter)
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

def display_examples():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    print(f"device: {device}") #sanity check - using GPU

    model_names = ['resnet-new-100', 'resnet-new-100-MIAT-from-scratch', 'resnet-new-100-MIAT-0.1-from-scratch', 'resnet-new-100-MIAT-0.25-from-scratch']
    estimator_suffixes = ['', '.1', '.25']

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
    miat_res_name = 'resnet-new-100-MIAT-0.25-from-scratch'
    #miat_res_name = 'resnet-new-100-MIAT-from-scratch'
    l_n = 'local_n.25'
    g_n = 'global_n.25'
    l_a = 'local_a.25'
    g_a = 'global_a.25'

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
    #               0        1         2        3         4        5
    model_fns = [std_res, miat_res, local_n, global_n, local_a, global_a]
    #model_fns = {'std': std_res, 'miat': miat_res, 'local_n':local_n, 'global_n': global_n, 'local_a': local_a, 'global_a': global_a}
    loss_fn = MI_loss #MI_loss - MI-Craft, MI_loss2 - MI-Craft-Euc

    target_model_fn = model_fns[1]
    
    eps_lst = [.025, .05, .075, .1, .125, .15, .175, .2, .25, .3, .4, .5, .75, 1.]
    #eps_lst = [.025, .05, .075, .1]
    #eps_lst = [.1,.15]

    for eps in eps_lst:
        
        print(25*'=')
        
        eps_iter = .007
        #eps_iter = .005
        #nb_iter = round(eps/eps_iter) + 10
        nb_iter = 100
        print(f"Using PGD-MI-Craft with eps: {eps}, eps_iter: {eps_iter}, nb_iter: {nb_iter}")
        #with torch.no_grad():
        for x_natural, y_true in test_loader:
            x_natural, y_true = x_natural.to(device), y_true.to(device)
            
            x_adv = pgd(model_fns, x_natural, eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, norm=np.inf, y=None, y_true=y_true, targeted=False, rand_init=True, loss_fn=loss_fn)
            
            y_nat_pred = target_model_fn(x_natural)
            y_adv_pred = target_model_fn(x_adv)
            
            pred = y_nat_pred.max(1, keepdim=True)[1]
            pred_adv = y_adv_pred.max(1, keepdim=True)[1]

            x_test_clean = x_natural.detach().cpu().numpy().transpose(0,2,3,1)
            y_test_pred = pred.detach().cpu().numpy()
            x_test_adv = x_adv.detach().cpu().numpy().transpose(0,2,3,1)
            y_test_adv = pred_adv.detach().cpu().numpy()
            # de normalize
            x_test_clean = np.clip(((x_test_clean * stats[1]) + stats[0]),0,1.)
            x_test_clean = (x_test_clean*255).astype(np.uint8)
            x_test_adv = np.clip(((x_test_adv * stats[1]) + stats[0]),0,1.)
            x_test_adv = (x_test_adv*255).astype(np.uint8)
            
            #x_test_adv is now [b][w][h][c] range of [0,255]
            y_test_pred = y_test_pred.astype(np.uint8)
            y_test_pred = y_test_pred.reshape((args.batch_size))
            y_test_adv = y_test_adv.astype(np.uint8)
            y_test_adv = y_test_adv.reshape((args.batch_size))
            
            fname = os.path.join('./results/imgs/', str(eps) + "-" + miat_res_name + "-MI-COS.PNG") #T is for targeted attack, remove for non-targeted
            show_img_grid(10,20, x_test_clean, x_test_adv, y_test_pred, y_test_adv, fname=fname)       
            
            break # only printing from 1st batch of images    
            
                
        print(25*'=')
        
#               0        1         2        3         4        5
#model_fns = [std_res, miat_res, local_n, global_n, local_a, global_a] #now a dictionary but indexing should work.
def eval_loss(model_fns, device, test_loader, loss_fn):
    target_model_fn = model_fns[0]
    
    loss = []
    acc = []
    acc_adv = []

    eps_lst = [.025, .05, .075, .1, .125, .15, .175, .2, .25, .3, .4, .5, .75, 1.]
    #eps_lst = [.025, .05, .075, .1]
    #eps_lst = [.1]

    for eps in eps_lst:
        #eps = .025
        print(25*'=')
        test_loss = 0
        correct = 0
        correct_adv = 0

        eps_iter = .007
        #eps_iter = .005
        #nb_iter = round(eps/eps_iter) + 10
        nb_iter = 100
        print(f"Using PGD with eps: {eps}, eps_iter: {eps_iter}, nb_iter: {nb_iter}")
        #with torch.no_grad():
        i = 0
        for x_natural, y_true in test_loader:
            i = i+1
            print(f"batch number {i}, {i*args.batch_size} / {len(test_loader.dataset)}")
            x_natural, y_true = x_natural.to(device), y_true.to(device)
            
            x_adv = pgd(model_fns, x_natural, eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, norm=np.inf, y=None, y_true=y_true, targeted=False, rand_init=True, loss_fn=loss_fn)
            
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
    miat_res_name = 'resnet-new-100-MIAT-0.25-from-scratch'
    #miat_res_name = 'resnet-new-100-MIAT-from-scratch'
    l_n = 'local_n'
    g_n = 'global_n'
    l_a = 'local_a'
    g_a = 'global_a'

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

    loss, acc, acc_adv = eval_loss(model_fns, device, test_loader, loss_fn)

    ## TEST THIS BEFORE RUNNING THE WHOLE THING
    
    print("Clean Accuracy:")
    for x in acc:
        print(x)
    print("Adversarial Accuracy:")
    for x in acc_adv:
        print(x)
    

if __name__ == '__main__':
    #main()
    display_examples()
