'''
Purpose of this file is to begin work on task 2 of my planned contribution
2. Use the Mutual Information (MI) estimation networks as a metric to aid in the crafting of adversarial examples.
'''
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

import projected_gradient_descent as pgd #cleverhans PGD

from utils import config
from utils import utils
from utils.data import data_dataset #doing it this way avoids clash with variable named "data"

#next is only needed to visualize samples
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt




classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

args = config.Configuration().getArgs()
stats = config.Configuration().getNormStats()

args.batch_size=2048 #may need to reduce.

#PGD Parameters
eps = args.eps
eps_iter = args.eps_iter
nb_iter = args.nb_iter


def MI_loss_nat(i, model, x_natural, y, x_adv, local_n, global_n, epoch):
    model.train()
    local_n.eval()
    global_n.eval()


    # logits_nat = model(x_natural)
    logits_adv = model(x_adv)

    loss_ce = F.cross_entropy(logits_adv, y)
    # loss_ce = 0.2 * F.cross_entropy(logits_nat, y) + 0.8 * F.cross_entropy(logits_adv, y)

    pesudo_label = F.softmax(model(x_natural), dim=0).max(1, keepdim=True)[1].squeeze()
    index = (pesudo_label == y)
    pesudo_label = F.softmax(model(x_adv), dim=0).max(1, keepdim=True)[1].squeeze()
    index = index * (pesudo_label != y)

    if torch.nonzero(index).size(0) != 0:


        loss_n = compute_loss(args=args, former_input=x_natural, latter_input=x_natural, encoder=model,
                dim_local=local_n, dim_global=global_n, v_out=True) * index

        loss_a = compute_loss(args=args, former_input=x_natural, latter_input=x_adv, encoder=model,
                               dim_local=local_n, dim_global=global_n, v_out=True) * index

        loss_mea = torch.abs(torch.tensor(1.0).cuda() - torch.cosine_similarity(loss_n, loss_a, dim=0))

        loss_a = loss_a.sum()/torch.nonzero(index).size(0)

        loss_mi = loss_mea + 0.1 * loss_a

    else:
        loss_mi = 0.0

    loss_all = loss_ce + loss_mi

    if (i + 1) % args.print_freq == 0:
        print('select samples:' + str(torch.nonzero(index).size(0)))
        print('Epoch [%d], Iter [%d/%d] Train target model. Natural MI: %.4f; Loss_ce: %.4f; Loss_all: %.4f'
              % (epoch, i + 1, 50000 // args.batch_size, loss_mi.item(), loss_ce.item(), loss_all.item()))

    return loss_all


def MI_loss(i, model, x_natural, y, x_adv, local_n, global_n, local_a, global_a, epoch):
    model.train()
    local_n.eval()
    global_n.eval()
    local_a.eval()
    global_a.eval()

    # logits_nat = model(x_natural)
    logits_adv = model(x_adv)

    loss_ce = F.cross_entropy(logits_adv, y)
    # loss_ce = 0.2 * F.cross_entropy(logits_nat, y) + 0.8 * F.cross_entropy(logits_adv, y)

    pesudo_label = F.softmax(model(x_natural), dim=0).max(1, keepdim=True)[1].squeeze()
    index = (pesudo_label == y)
    pesudo_label = F.softmax(model(x_adv), dim=0).max(1, keepdim=True)[1].squeeze()
    index = index * (pesudo_label != y)

    if torch.nonzero(index).size(0) != 0:


        loss_n = compute_loss(args=args, former_input=x_natural, latter_input=x_natural, encoder=model,
                dim_local=local_n, dim_global=global_n, v_out=True) * index

        loss_a = compute_loss(args=args, former_input=x_natural, latter_input=x_adv, encoder=model,
                               dim_local=local_n, dim_global=global_n, v_out=True) * index

        # loss_a_all = loss_a
        loss_mea_n = torch.abs(torch.tensor(1.0).cuda() - torch.cosine_similarity(loss_n, loss_a, dim=0))


        loss_a = compute_loss(args=args, former_input=x_adv - x_natural, latter_input=x_adv, encoder=model,
                              dim_local=local_a, dim_global=global_a, v_out=True) * index

        loss_n = compute_loss(args=args, former_input=x_adv - x_natural, latter_input=x_natural, encoder=model,
                              dim_local=local_a, dim_global=global_a, v_out=True) * index

        # loss_a_all = torch.tensor(0.1).cuda() * (loss_a_all - loss_a)
        loss_mea_a = torch.abs(torch.tensor(1.0).cuda() - torch.cosine_similarity(loss_n, loss_a, dim=0))


        loss_mi = loss_mea_n + loss_mea_a # + loss_a_all

    else:
        loss_mi = 0.0

    loss_all = loss_ce + 5.0 * loss_mi

    if (i + 1) % args.print_freq == 0:
        print('select samples:' + str(torch.nonzero(index).size(0)))
        print('Epoch [%d], Iter [%d/%d] Train target model. Natural MI: %.4f; Loss_ce: %.4f; Loss_all: %.4f'
              % (epoch, i + 1, 50000 // args.batch_size, loss_mi.item(), loss_ce.item(), loss_all.item()))

    return loss_all


def evaluate_mi_nat(encoder, x_natural, y, x_adv, local_n, global_n):

    encoder.eval()
    local_n.eval()
    global_n.eval()

    pesudo_label = F.softmax(encoder(x_natural), dim=0).max(1, keepdim=True)[1].squeeze()
    index = (pesudo_label == y)
    pesudo_label = F.softmax(encoder(x_adv), dim=0).max(1, keepdim=True)[1].squeeze()
    index = index * (pesudo_label == y)

    loss_r_n = (compute_loss(args=args, former_input=x_natural, latter_input=x_natural, encoder=encoder,
                        dim_local=local_n, dim_global=global_n, v_out=True) * index).sum()/torch.nonzero(index).size(0)

    loss_r_a = (compute_loss(args=args, former_input=x_natural, latter_input=x_adv, encoder=encoder,
                            dim_local=local_n, dim_global=global_n, v_out=True) * index).sum()/torch.nonzero(index).size(0)

    pesudo_label = F.softmax(encoder(x_natural), dim=0).max(1, keepdim=True)[1].squeeze()
    index = (pesudo_label == y)
    pesudo_label = F.softmax(encoder(x_adv), dim=0).max(1, keepdim=True)[1].squeeze()
    index = index * (pesudo_label != y)

    loss_w_n = (compute_loss(args=args, former_input=x_natural, latter_input=x_natural, encoder=encoder,
                        dim_local=local_n, dim_global=global_n, v_out=True) * index).sum()/torch.nonzero(index).size(0)

    loss_w_a = (compute_loss(args=args, former_input=x_natural, latter_input=x_adv, encoder=encoder,
                            dim_local=local_n, dim_global=global_n, v_out=True) * index).sum()/torch.nonzero(index).size(0)

    return loss_r_n, loss_r_a, loss_w_n, loss_w_a


def evaluate_mi_adv(encoder, x_natural, y, x_adv, local_n, global_n):

    encoder.eval()
    local_n.eval()
    global_n.eval()

    pesudo_label = F.softmax(encoder(x_natural), dim=0).max(1, keepdim=True)[1].squeeze()
    index = (pesudo_label == y)
    pesudo_label = F.softmax(encoder(x_adv), dim=0).max(1, keepdim=True)[1].squeeze()
    index = index * (pesudo_label == y)

    loss_r_n = (compute_loss(args=args, former_input=x_adv - x_natural, latter_input=x_natural, encoder=encoder,
                        dim_local=local_n, dim_global=global_n, v_out=True) * index).sum()/torch.nonzero(index).size(0)

    loss_r_a = (compute_loss(args=args, former_input=x_adv - x_natural, latter_input=x_adv, encoder=encoder,
                            dim_local=local_n, dim_global=global_n, v_out=True) * index).sum()/torch.nonzero(index).size(0)

    pesudo_label = F.softmax(encoder(x_natural), dim=0).max(1, keepdim=True)[1].squeeze()
    index = (pesudo_label == y)
    pesudo_label = F.softmax(encoder(x_adv), dim=0).max(1, keepdim=True)[1].squeeze()
    index = index * (pesudo_label != y)

    loss_w_n = (compute_loss(args=args, former_input=x_adv - x_natural, latter_input=x_natural, encoder=encoder,
                        dim_local=local_n, dim_global=global_n, v_out=True) * index).sum()/torch.nonzero(index).size(0)

    loss_w_a = (compute_loss(args=args, former_input=x_adv - x_natural, latter_input=x_adv, encoder=encoder,
                            dim_local=local_n, dim_global=global_n, v_out=True) * index).sum()/torch.nonzero(index).size(0)

    return loss_r_n, loss_r_a, loss_w_n, loss_w_a


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

    print(f"Model loaded: {name}")
    print(f"Model loaded: {name2}")


if __name__ == '__main__':
    main()