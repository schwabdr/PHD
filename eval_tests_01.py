'''
Purpose of this file is to evaluate any given model with adversarial examples given by any model.
Both models can be the same for a white box evaluation.
We will test all three MIAT/NAMID models using a range of epsilon values and nb_iter = 100 in all cases.

10/13/22 Update - Going to add a function that will print some samples for me to choose from 
so I can make a grid of all the different attacks - I think they will look the same but I don't know!
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
from utils.data import data_dataset
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
#using 512 here for generating and displaying adv examples.
args.batch_size=512#2048-2048 was original for the test  #trying this - this works with the data parallel - GPU util ~95%, memory 10800/11019 MB each GPU
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
    name = 'resnet-new-100' #input("Name of model to load: ") #for now I'll hard code the only model I have trained
    
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

    testset = data_dataset(img_path=args.nat_img_test, clean_label_path=args.nat_label_test, transform=trans_test)
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

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    model_names = ['resnet-new-100', 'resnet-new-100-MIAT-from-scratch', 'resnet-new-100-MIAT-0.1-from-scratch', 'resnet-new-100-MIAT-0.25-from-scratch']
    
    trans_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    testset = data_dataset(img_path=args.nat_img_test, clean_label_path=args.nat_label_test, transform=trans_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, drop_last=False, shuffle=False,
                                              num_workers=4, pin_memory=True)
    
    for name in model_names:
        #load model parameters for this test
        model = ResNet18(10)
        model.load_state_dict(torch.load(os.path.join(args.SAVE_MODEL_PATH, name)))
        model.to(device)
        model = torch.nn.DataParallel(model).cuda() 
        model.eval()

        print(f"Model loaded: {name}")
    
        '''these values will cause large perturbations, and nearly 100% misclassifications.
        eps = 1.27
        eps_iter = .05
        nb_iter = 50
        '''
        
        #this is our L_infty constraint 
        eps_lst = [.025, .05, .075, .1, .125, .15, .175, .2, .25, .3, .4, .5, .75, 1.] #, 1.5, 2., 2.5]
        #eps_lst = [.025, .05] # for quick test

        for eps in eps_lst:
            print(25*'=')
            
            eps_iter = .007
            #nb_iter = round(eps/eps_iter) + 10
            nb_iter = 100
            print(f"Using PGD with eps: {eps}, eps_iter: {eps_iter}, nb_iter: {nb_iter}")
            #with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                top2 = torch.topk(output, 2) # this is for a targeted attack
                y_target = torch.select(top2.indices, 1, 1) #y_target is second most likely class
                #data_adv = pgd(model, data, eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, norm=np.inf, y=None, targeted=False) #NON_targeted attack
                data_adv = pgd(model, data, eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, norm=np.inf, y=y_target, targeted=True) #targeted attack

                # here I'll have to look back at some old code for stacking np images together to show in a grid
                #x_adv = data_adv.detach().cpu().numpy().transpose(0,2,3,1) #I'll use this later - gonna paste all the images together.
                #output = model(data)
                output_adv = model(data_adv)
                pred = output.max(1, keepdim=True)[1] # we'll use this instead of true label -make sure it is misclassified
                pred_adv = output_adv.max(1, keepdim=True)[1]
                
                x_test_clean = data.detach().cpu().numpy().transpose(0,2,3,1)
                y_test_pred = pred.detach().cpu().numpy()
                x_test_adv = data_adv.detach().cpu().numpy().transpose(0,2,3,1)
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

                fname = os.path.join('./results/imgs/', str(eps) + "-" + name + "-PGD-CE-T.PNG") #T is for targeted attack, remove for non-targeted
                show_img_grid(10,20, x_test_clean, x_test_adv, y_test_pred, y_test_adv, fname=fname)

                break # do one batch for quick test
            
            
            print(25*'=')

if __name__ == '__main__':
    #main()
    display_examples()