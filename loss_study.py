'''
https://pytorch.org/docs/stable/generated/torch.cdist.html
Computes batched the p-norm distance between each pair of the two collections of row vectors.

The purpose of this file is to study the loss metrics used for MIAT/NAMID

Currently the only loss metric is cosine_loss = | 1 - cos_sim(a,b) |
One issue is that this only considers the angle between the two vectors, a,b. It does not consider the magnitude.
So it seems we are leaving some room on the table so to speak for adversarial examples to grow.

'''
import os

import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn

from models.resnet_new import ResNet18
from models.estimator import Estimator
from models.discriminators import MI1x1ConvNet, MIInternalConvNet, MIInternallastConvNet
from compute_MI import compute_loss

from utils import config
from utils.data import data_adv_dataset

from pgd_loss_test import projected_gradient_descent as pgd


#next is only needed to visualize samples
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

import random


args = config.Configuration().getArgs()
stats = config.Configuration().getNormStats() 
classes = config.Configuration().getClasses()

#               0        1         2        3         4        5
#model_fns = {std_res, miat_res, local_n, global_n, local_a, global_a}
def eval_loss(model_fns, device, test_loader):
    std_res_fn = model_fns[0]
    miat_res_fn = model_fns[1]


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
    testset = data_adv_dataset(img_path=args.nat_img_test, adv_img_path=args.adv_img_test, clean_label_path=args.nat_label_test, transform=trans_test, augment=False)
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
    model_fns = {std_res, miat_res, local_n, global_n, local_a, global_a}
    eval_loss(model_fns, device, test_loader)

if __name__ == '__main__':
    main()
