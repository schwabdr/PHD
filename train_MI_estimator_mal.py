# This version max Natural MI of x and max Adversarial MI of x_adv

#TODO
'''
1) remove Accuracy and Robust Accuracy between every Epoch
2) Create the adversarial examples ONCE and save them for usage in all EPOCHS
3) This will greatly speed up the execution time.
4) explore meaning of "wrong" and "right" samples - printed every 50 iters
5) What do the MI values mean?
6) if this training ever finishes - save the models on disk with a new name
'''

import os
import argparse
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler, Adam

import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from utils.data import data_dataset
from utils.data import data_adv_dataset

from models.resnet_new import ResNet18

from models.estimator import Estimator
from models.discriminators import MI1x1ConvNet, MIInternalConvNet, MIInternallastConvNet
from compute_MI import compute_loss

from utils import config
from utils import utils

#import projected_gradient_descent as pgd # for attacks

#next is only needed to visualize samples
import matplotlib
#matplotlib.use('tkagg')
import matplotlib.pyplot as plt

import random


args = config.Configuration().getArgs()
#normalization stats - calculated for MalImg
#mean: tensor([0.4454, 0.4454, 0.4454])
#std: tensor([0.3122, 0.3122, 0.3122])

stats = ((0.4454, 0.4454, 0.4454), (0.3122, 0.3122, 0.3122)) #mean and stdev

classes = ['Adialer.C', 'Agent.FYI', 'Allaple.A', 'Allaple.L', 'Alueron.gen!J', 'Autorun.K', 'C2LOP.P', 'C2LOP.gen!g', 'Dialplatform.B', 'Dontovo.A', 'Fakerean', 
                'Instantaccess', 'Lolyda.AA1', 'Lolyda.AA2', 'Lolyda.AA3', 'Lolyda.AT', 'Malex.gen!J', 'Obfuscator.AD', 'Rbot!gen', 'Skintrim.N', 'Swizzor.gen!E', 'Swizzor.gen!I', 
                'VB.AT', 'Wintrim.BX', 'Yuner.A']

#override the default values from config file
args.batch_size = 64 #200
args.epochs = 100 #for CIFAR10 we used 50
args.save_freq = 1
'''
parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr-mi', type=float, default=1e-2, metavar='LR',
                    help='learning rate')
'''

print(f"Total Epochs: {args.epochs}")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def make_optimizer_and_schedule(model, lr):
    optimizer = Adam(model.parameters(), lr)
    schedule = lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30, 40], gamma=0.5)
    return optimizer, schedule

def MI_loss_nat(i, model, x_natural, y, x_adv, local_n, global_n, epoch):
    model.eval()
    local_n.train()
    global_n.train()

    pesudo_label = F.softmax(model(x_natural), dim=0).max(1, keepdim=True)[1].squeeze()
    index = (pesudo_label == y)

    if torch.nonzero(index).size(0) != 0:

        loss_n = (compute_loss(args=args, former_input=x_natural, latter_input=x_natural, encoder=model,
                dim_local=local_n, dim_global=global_n, v_out=True) * index).sum()/torch.nonzero(index).size(0)

    else:
        loss_n = compute_loss(args=args, former_input=x_natural, latter_input=x_natural, encoder=model,
                                dim_local=local_n, dim_global=global_n, v_out=True).mean()

    if (i + 1) % args.print_freq == 0:
        print('select right samples:' + str(torch.nonzero(index).size(0)))
        print('Epoch [%d], Iter [%d/%d] Train MI estimator. Natural MI: -n %.4f'
              % (epoch, i + 1, 50000 // args.batch_size, loss_n.item()))

    return loss_n


def MI_loss_adv(i, model, x_natural, y, x_adv, local_n, global_n, epoch):
    model.eval()
    local_n.train()
    global_n.train()

    pesudo_label = F.softmax(model(x_adv), dim=0).max(1, keepdim=True)[1].squeeze()
    index = (pesudo_label != y)

    if torch.nonzero(index).size(0) != 0:

        loss_a = (compute_loss(args=args, former_input=x_adv-x_natural, latter_input=x_adv, encoder=model,
                dim_local=local_n, dim_global=global_n, v_out=True) * index).sum()/torch.nonzero(index).size(0)
    else:
        loss_a = compute_loss(args=args, former_input=x_adv-x_natural, latter_input=x_adv, encoder=model,
                                dim_local=local_n, dim_global=global_n, v_out=True).mean()

    if (i + 1) % args.print_freq == 0:
        print('select wrong samples:' + str(torch.nonzero(index).size(0)))
        print('Epoch [%d], Iter [%d/%d] Train MI estimator. Adversasrial MI: -n %.4f'
              % (epoch, i + 1, 50000 // args.batch_size, loss_a.item()))

    return loss_a


def evaluate_mi_nat(encoder, x_natural, y, x_adv, local_n, global_n):

    encoder.eval()
    local_n.eval() #from DIM?
    global_n.eval() #from DIM?

    pesudo_label = F.softmax(encoder(x_natural), dim=0).max(1, keepdim=True)[1].squeeze()
    index = (pesudo_label == y)
    pesudo_label = F.softmax(encoder(x_adv), dim=0).max(1, keepdim=True)[1].squeeze()
    index = index * (pesudo_label != y)

    #torch.nonzero() gives a Tensor containing indices of all non-zero elements
    loss_n = (compute_loss(args=args, former_input=x_natural, latter_input=x_natural, encoder=encoder,
                        dim_local=local_n, dim_global=global_n, v_out=True) * index).sum()/torch.nonzero(index).size(0)

    loss_a = (compute_loss(args=args, former_input=x_natural, latter_input=x_adv, encoder=encoder,
                            dim_local=local_n, dim_global=global_n, v_out=True) * index).sum()/torch.nonzero(index).size(0)

    return loss_n, loss_a


def evaluate_mi_adv(encoder, x_natural, y, x_adv, local_n, global_n):

    encoder.eval()
    local_n.eval()
    global_n.eval()

    pesudo_label = F.softmax(encoder(x_natural), dim=0).max(1, keepdim=True)[1].squeeze()
    index = (pesudo_label == y)
    pesudo_label = F.softmax(encoder(x_adv), dim=0).max(1, keepdim=True)[1].squeeze()
    index = index * (pesudo_label != y)

    loss_n = (compute_loss(args=args, former_input=x_adv-x_natural, latter_input=x_natural, encoder=encoder,
                        dim_local=local_n, dim_global=global_n, v_out=True) * index).sum()/torch.nonzero(index).size(0)

    loss_a = (compute_loss(args=args, former_input=x_adv-x_natural, latter_input=x_adv, encoder=encoder,
                            dim_local=local_n, dim_global=global_n, v_out=True) * index).sum()/torch.nonzero(index).size(0)

    return loss_n, loss_a


def eval_test(model, device, test_loader, local_n, global_n, local_a, global_a):
    model.eval()
    local_n.eval()
    global_n.eval()
    local_a.eval()
    global_a.eval()

    cnt = 0
    correct = 0
    correct_adv = 0
    losses_n_n = 0
    losses_n_a = 0
    losses_a_n = 0
    losses_a_a = 0

    for data, data_adv, target in test_loader:
        cnt += 1
        data, data_adv, target = data.to(device), data_adv.to(device), target.to(device)
              
        with torch.no_grad():
            output = model(data)
            output_adv = model(data_adv)
            pred = output.max(1, keepdim=True)[1]
            pred_adv = output_adv.max(1, keepdim=True)[1]

            test_loss_n_n, test_loss_n_a = evaluate_mi_nat(encoder=model, x_natural=data, y=target, x_adv=data_adv,
                                                           local_n=local_n, global_n=global_n)

            test_loss_a_n, test_loss_a_a = evaluate_mi_adv(encoder=model, x_natural=data, y=target, x_adv=data_adv,
                                                           local_n=local_a, global_n=global_a)


        correct += pred.eq(target.view_as(pred)).sum().item()
        correct_adv += pred_adv.eq(target.view_as(pred_adv)).sum().item()
        losses_n_n += test_loss_n_n.item()
        losses_n_a += test_loss_n_a.item()
        losses_a_n += test_loss_a_n.item()
        losses_a_a += test_loss_a_a.item()

    test_accuracy = correct_adv / len(test_loader.dataset)
    print('Test:  Accuracy: {}/{} ({:.2f}%), Robust Accuracy: {}/{} ({:.2f}%)'.format(correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), correct_adv, len(test_loader.dataset),
        100. * correct_adv / len(test_loader.dataset)))
    print('Test: Natural MI: -n: {:.4f}, -a: {:.4f}'.format(
        losses_n_n/cnt, losses_n_a/cnt))
    print('Test: Adversarial MI: -n: {:.4f}, -a: {:.4f}'.format(
        losses_a_n / cnt, losses_a_a / cnt))

    return test_accuracy


def main():
    # settings
    setup_seed(42) #best seed ever
    
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print(f"device: {device}") #sanity check - using GPU

    # setup data loader - should we normalize again? I think so for consistency.
    trans_train = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(*stats, inplace=True)
    ])

    trans_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(*stats, inplace=False)
    ])
    
    trainset = data_adv_dataset(img_path=args.nat_img_train_mal, adv_img_path=args.adv_img_train_mal,clean_label_path=args.nat_label_train_mal, transform=trans_train, augment=True)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, drop_last=False,shuffle=True, num_workers=8, pin_memory=True)
    testset = data_adv_dataset(img_path=args.nat_img_test_mal, adv_img_path=args.adv_img_test_mal, clean_label_path=args.nat_label_test_mal, transform=trans_test, augment=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, drop_last=False, shuffle=False, num_workers=8, pin_memory=True)

    # load MI estimation model

    # Estimator part 1: X or layer3 to H space
    local_n = Estimator(args.va_hsize) #default 2048
    local_a = Estimator(args.va_hsize)

    #args.is_internal = True # 2nd attempt save with _is_internal in file name
    #args.is_internal_last = True # see line above
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
    else:
        #z_size = 10
        z_size = 25 #changing to 25 for MalImg
        global_n = MI1x1ConvNet(z_size, args.va_hsize)
        global_a = MI1x1ConvNet(z_size, args.va_hsize)

    print('----------------Start training-------------')
    target_model = ResNet18(25)
    name = 'resnet-mal-std-aug-100' #input("Name of model to load: ") #for now I'll hard code so I don't have to retype the name while prototyping
    print(f"Target Model Loaded (encoder): {name}")
    target_model.load_state_dict(torch.load(os.path.join(args.SAVE_MODEL_PATH, name)))
    target_model.to(device)
    
    #I'm going to leave it in - hope it works (next two lines were commented out by me ...)
    target_model = torch.nn.DataParallel(target_model).cuda() #don't think I need this for eval
    target_model.eval()

    local_n.to(device)
    global_n.to(device)
    local_a.to(device)
    global_a.to(device)

    local_n = torch.nn.DataParallel(local_n).cuda()
    global_n = torch.nn.DataParallel(global_n).cuda()
    local_a = torch.nn.DataParallel(local_a).cuda()
    global_a = torch.nn.DataParallel(global_a).cuda()

    cudnn.benchmark = True

    #_,_ = make_optimizer_and_schedule(target_model, lr=args.lr_mi)
    opt_local_n, schedule_local_n = make_optimizer_and_schedule(local_n, lr=args.lr_mi)
    opt_global_n, schedule_global_n = make_optimizer_and_schedule(global_n, lr=args.lr_mi)
    opt_local_a, schedule_local_a = make_optimizer_and_schedule(local_a, lr=args.lr_mi)
    opt_global_a, schedule_global_a = make_optimizer_and_schedule(global_a, lr=args.lr_mi)
   
    first = False
    # Train
    for epoch in range(1, args.epochs + 1):
        loss_n_all = 0
        loss_a_all = 0

        for batch_idx, (data, data_adv, target) in enumerate(train_loader):
            data, data_adv, target = data.to(device), data_adv.to(device), target.to(device)
            # now adv is taken from the train_loader
            #SANITY CHECK
            #make sure images line up:
            #TODO move this block of code into a function in utils. I've copied it so many times now.
            if first: #False to block out this section of code for now.
                first = False
                x = data.detach().cpu().numpy().transpose(0,2,3,1)
                x_adv = data_adv.detach().cpu().numpy().transpose(0,2,3,1)
                pred = target.detach().cpu().numpy()
                #goal here is to randomly display an image and it's adverarial example
                x = np.clip(((x * stats[1]) + stats[0]),0,1.)
                x_adv = np.clip(((x_adv * stats[1]) + stats[0]),0,1.)
                rows = 5
                cols = 10
                fig, axes1 = plt.subplots(rows,cols,figsize=(10,10))
                lst = list(range(0, len(x)))
                random.shuffle(lst)
                for j in range(5):
                    for k in range(0,10,2):
                        #get a random index
                        i = lst.pop()
                        axes1[j][k].set_axis_off()
                        axes1[j][k+1].set_axis_off()
                        axes1[j][k].imshow(x[i],interpolation='nearest')
                        axes1[j][k].text(0,0,classes[target[i]]) # this gets the point accross but needs fixing.
                        axes1[j][k+1].imshow(x_adv[i], interpolation='nearest')
                        pred_ind = pred[i]
                        axes1[j][k+1].text(0,0,classes[pred_ind])
                plt.show()
                plt.savefig("./imgs/test_grid_MINE.png", format='png')



            # Train MI estimator
            loss_n = MI_loss_nat(i=batch_idx, model=target_model, x_natural=data, y=target, x_adv=data_adv,
                           local_n=local_n, global_n=global_n, epoch=epoch)

            loss_n_all += loss_n

            opt_local_n.zero_grad()
            opt_global_n.zero_grad()
            loss_n.backward()
            opt_local_n.step()
            opt_global_n.step()

            loss_a = MI_loss_adv(i=batch_idx, model=target_model, x_natural=data, y=target, x_adv=data_adv,
                                 local_n=local_a, global_n=global_a, epoch=epoch)
            loss_a_all += loss_a

            opt_local_a.zero_grad()
            opt_global_a.zero_grad()
            loss_a.backward()
            opt_local_a.step()
            opt_global_a.step()


        schedule_local_n.step()
        schedule_global_n.step()
        schedule_local_a.step()
        schedule_global_a.step()

        loss_n_all = loss_n_all / (batch_idx +1)
        loss_a_all = loss_a_all / (batch_idx + 1)

        # evaluation
        print('================================================================')
        # _ = eval_train(model=target_model, device=device, test_loader=train_loader, local_n=local_n,
        #              global_n=global_n)

        #test_accuracy = eval_test(model=target_model, device=device, test_loader=test_loader, local_n=local_n,
        #                          global_n=global_n, local_a=local_a,
        #                          global_a=global_a)

        # save checkpoint
        '''
        if epoch % args.save_freq == 0:

            
            #torch.save(model.module.state_dict(),
            #           os.path.join(model_dir, 'model-epoch{}.pt'.format(epoch)))
            

            torch.save(local_n.module.state_dict(),
                       os.path.join(args.model_dir, 'local_n_model.pth'))
            torch.save(global_n.module.state_dict(),
                       os.path.join(args.model_dir, 'global_n_model.pth'))
            torch.save(local_a.module.state_dict(),
                       os.path.join(args.model_dir, 'local_a_model.pth'))
            torch.save(global_a.module.state_dict(),
                       os.path.join(args.model_dir, 'global_a_model.pth'))
            print('save the model')
        '''
        print('================================================================')
    # no need to save target - it is not changed by this process.
    print("Saving estimator models ...")
    utils.save_model(local_n, 'local_n_mal.25')
    utils.save_model(global_n, 'global_n_mal.25')
    utils.save_model(local_a, 'local_a_mal.25')
    utils.save_model(global_a, 'global_a_mal.25')
    print("Save Complete. Exiting ...")

if __name__ == '__main__':
    main()
