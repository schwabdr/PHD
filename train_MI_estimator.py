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

from models.resnet_new import ResNet18

from models.estimator import Estimator
from models.discriminators import MI1x1ConvNet, MIInternalConvNet, MIInternallastConvNet
from compute_MI import compute_loss

from utils import config
from utils import utils

import projected_gradient_descent as pgd # for attacks


args = config.Configuration().getArgs()
stats = config.Configuration().getNormStats() 

#override the default values from config file
args.batch_size = 200
args.epochs = 50
args.save_freq = 1
'''
parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr-mi', type=float, default=1e-2, metavar='LR',
                    help='learning rate')
'''
#for PGD
eps = 8/255. #approx .0314
eps_iter = .007
nb_iter = 40

print(f"Using PGD with eps: {eps}, eps_iter: {eps_iter}, nb_iter: {nb_iter}")

print(f"Total Epochs: {args.epochs}")

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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


    for data, target in test_loader:
        cnt += 1
        data, target = data.to(device), target.to(device)
        #data_adv = craft_adversarial_example_pgd(model=model, x_natural=data, y=target,
        #                                     step_size=0.007, epsilon=8/255,
        #                                     perturb_steps=40, distance='l_inf')
        data_adv = pgd.projected_gradient_descent(model, data, eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, norm=np.inf,y=target, targeted=False)
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

    
    # load MI estimation model

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
    else:
        z_size = 10
        global_n = MI1x1ConvNet(z_size, args.va_hsize)
        global_a = MI1x1ConvNet(z_size, args.va_hsize)

    print('----------------Start training-------------')
    target_model = ResNet18(10)
    name = 'resnet-new-100' #input("Name of model to load: ") #for now I'll hard code so I don't have to retype the name while prototyping
    #target_model = target_model.load_statetorch.load(os.path.join(args.SAVE_MODEL_PATH, name))
    #target_model = model = models.resnet18()
    target_model.load_state_dict(torch.load(os.path.join(args.SAVE_MODEL_PATH, name)))
    target_model.to(device)
    target_model.eval()


    #target_model = torch.nn.DataParallel(target_model).cuda() #don't think I need this for eval
    #target_model.eval()

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

    #TODO move this back above all the model set up.
    # setup data loader - should we normalize again? I think so for consistency.
    trans_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats, inplace=False)
    ])

    trans_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats, inplace=False)
    ])

    #set up data loaders for the original clean CIFAR images
    trainset = data_dataset(img_path=args.nat_img_train, clean_label_path=args.nat_label_train, transform=trans_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, drop_last=False,shuffle=True, num_workers=4, pin_memory=True)
    testset = data_dataset(img_path=args.nat_img_test, clean_label_path=args.nat_label_test, transform=trans_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, drop_last=False, shuffle=False, num_workers=4, pin_memory=True)

    # Train
    for epoch in range(1, args.epochs + 1):
        loss_n_all = 0
        loss_a_all = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # craft adversarial examples - using PGD from CleverHans
            adv = pgd.projected_gradient_descent(target_model, data, eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, norm=np.inf,y=target, targeted=False)
            # Train MI estimator
            loss_n = MI_loss_nat(i=batch_idx, model=target_model, x_natural=data, y=target, x_adv=adv,
                           local_n=local_n, global_n=global_n, epoch=epoch)

            loss_n_all += loss_n

            opt_local_n.zero_grad()
            opt_global_n.zero_grad()
            loss_n.backward()
            opt_local_n.step()
            opt_global_n.step()

            loss_a = MI_loss_adv(i=batch_idx, model=target_model, x_natural=data, y=target, x_adv=adv,
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

        test_accuracy = eval_test(model=target_model, device=device, test_loader=test_loader, local_n=local_n,
                                  global_n=global_n, local_a=local_a,
                                  global_a=global_a)

        # save checkpoint
        if epoch % args.save_freq == 0:

            '''
            torch.save(model.module.state_dict(),
                       os.path.join(model_dir, 'model-epoch{}.pt'.format(epoch)))
            '''

            torch.save(local_n.module.state_dict(),
                       os.path.join(args.model_dir, 'local_n_model.pth'))
            torch.save(global_n.module.state_dict(),
                       os.path.join(args.model_dir, 'global_n_model.pth'))
            torch.save(local_a.module.state_dict(),
                       os.path.join(args.model_dir, 'local_a_model.pth'))
            torch.save(global_a.module.state_dict(),
                       os.path.join(args.model_dir, 'global_a_model.pth'))
            print('save the model')

        print('================================================================')

    print("Saving estimator models ...")
    utils.save_model(local_n, 'local_n')
    utils.save_model(global_n, 'global_n')
    utils.save_model(local_a, 'local_a')
    utils.save_model(global_a, 'global_a')
    print("Save Complete. Exiting ...")

if __name__ == '__main__':
    main()
