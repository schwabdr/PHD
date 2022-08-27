'''
This is the code to train a network using the MIAT method.
-replaced default adv ex crafting with Cleverhans PGD implementation 
    original parms: step_size=0.007, epsilon=8/255, perturb_steps=40, distance='l_inf'
'''
import os
import argparse
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler, Adam

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from models.resnet_new import ResNet18
#from models.wideresnet_new import WideResNet #might use eventually

from models.estimator import Estimator
from models.discriminators import MI1x1ConvNet, MIInternalConvNet, MIInternallastConvNet
from compute_MI import compute_loss

import projected_gradient_descent as pgd #cleverhans PGD

from utils import config
from utils import utils
from utils.data import data_dataset

stats = config.Configuration().getNormStats() 

args = config.Configuration().getArgs()

args.model_dir = './checkpoint/wideresnet/MIAT_standard'

#PGD Parameters
eps = args.eps
eps_iter = args.eps_iter
nb_iter = args.nb_iter

print(f"Using PGD with eps: {eps}, eps_iter: {eps_iter}, nb_iter: {nb_iter}")

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
    schedule = lr_scheduler.MultiStepLR(optimizer, milestones=[75, 90], gamma=0.1)
    return optimizer, schedule


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr

    if epoch >= 90:
        lr = args.lr * 0.01
    elif epoch >= 75:
        lr = args.lr * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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


def eval_test(model, device, test_loader, local_n, global_n, local_a, global_a):
    model.eval()
    local_n.eval()
    global_n.eval()
    local_a.eval()
    global_a.eval()

    cnt = 0
    correct = 0
    correct_adv = 0
    losses_r_n = 0
    losses_r_a = 0
    losses_w_n = 0
    losses_w_a = 0
    losses_r_n_1 = 0
    losses_r_a_1 = 0
    losses_w_n_1 = 0
    losses_w_a_1 = 0

    for data, target in test_loader:
        cnt += 1
        data, target = data.to(device), target.to(device)
        data_adv = pgd.projected_gradient_descent(model, data, eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, norm=np.inf,y=None, targeted=False)
        #data_adv = craft_adversarial_example_pgd(model=model, x_natural=data, y=target, step_size=0.007, epsilon=8/255, perturb_steps=40, distance='l_inf')

        with torch.no_grad():
            output = model(data)
            output_adv = model(data_adv)
            pred = output.max(1, keepdim=True)[1]
            pred_adv = output_adv.max(1, keepdim=True)[1]

            test_loss_r_n, test_loss_r_a, test_loss_w_n, test_loss_w_a = evaluate_mi_nat(encoder=model, x_natural=data,
                                                    y=target, x_adv=data_adv, local_n=local_n, global_n=global_n)

            test_loss_r_n_1, test_loss_r_a_1, test_loss_w_n_1, test_loss_w_a_1 = evaluate_mi_nat(encoder=model, x_natural=data,
                                                                                     y=target, x_adv=data_adv,
                                                                                     local_n=local_a, global_n=global_a)

        correct += pred.eq(target.view_as(pred)).sum().item()
        correct_adv += pred_adv.eq(target.view_as(pred_adv)).sum().item()
        losses_r_n += test_loss_r_n.item()
        losses_r_a += test_loss_r_a.item()
        losses_w_n += test_loss_w_n.item()
        losses_w_a += test_loss_w_a.item()

        losses_r_n_1 += test_loss_r_n_1.item()
        losses_r_a_1 += test_loss_r_a_1.item()
        losses_w_n_1 += test_loss_w_n_1.item()
        losses_w_a_1 += test_loss_w_a_1.item()

    test_accuracy = (correct_adv + correct) / (2.0 * len(test_loader.dataset))
    print('Test:  Accuracy: {}/{} ({:.2f}%), Robust Accuracy: {}/{} ({:.2f}%)'.format(correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), correct_adv, len(test_loader.dataset),
        100. * correct_adv / len(test_loader.dataset)))
    print('Test: Natural MI Right: -n: {:.4f}, -a: {:.4f}'.format(
        losses_r_n/cnt, losses_r_a/cnt))
    print('Test: Natural MI Wrong: -n: {:.4f}, -a: {:.4f}'.format(
        losses_w_n / cnt, losses_w_a / cnt))
    print('Test: Adv MI Right: -n: {:.4f}, -a: {:.4f}'.format(
        losses_r_n_1/cnt, losses_r_a_1/cnt))
    print('Test: Adv MI Wrong: -n: {:.4f}, -a: {:.4f}'.format(
        losses_w_n_1 / cnt, losses_w_a_1 / cnt))

    return test_accuracy



def main():
    # settings
    setup_seed(42) #still the best seed ever
    
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    print(f"device: {device}") #sanity check - using GPU

    # setup data loader - I added the consistent normalization I've been using.
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

    trainset = data_dataset(img_path=args.nat_img_train, clean_label_path=args.nat_label_train,
                            transform=trans_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, drop_last=False,
                                               shuffle=True, num_workers=4, pin_memory=True)
    testset = data_dataset(img_path=args.nat_img_test, clean_label_path=args.nat_label_test, transform=trans_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, drop_last=False, shuffle=False,
                                              num_workers=4, pin_memory=True)

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

    #TODO start here when I get back to this file ... sigh
    target_model = ResNet18(10)
    #target_model = WideResNet(34, 10, 10)
    target_model = torch.nn.DataParallel(target_model).cuda()

    local_n.load_state_dict(torch.load(os.path.join(args.SAVE_MODEL_PATH, 'local_n')))
    global_n.load_state_dict(torch.load(os.path.join(args.SAVE_MODEL_PATH, 'global_n')))
    local_a.load_state_dict(torch.load(os.path.join(args.SAVE_MODEL_PATH, 'local_a')))
    global_a.load_state_dict(torch.load(os.path.join(args.SAVE_MODEL_PATH, 'global_a')))

    ''' #original lines to load - didn't work
    local_n.load_state_dict(torch.load(args.pre_local_n))
    global_n.load_state_dict(torch.load(args.pre_global_n))
    local_a.load_state_dict(torch.load(args.pre_local_a))
    global_a.load_state_dict(torch.load(args.pre_global_a))
    '''
    local_n = torch.nn.DataParallel(local_n).cuda()
    global_n = torch.nn.DataParallel(global_n).cuda()
    local_a = torch.nn.DataParallel(local_a).cuda()
    global_a = torch.nn.DataParallel(global_a).cuda()

    cudnn.benchmark = True

    #TODO try adam optimizer here
    optimizer = optim.SGD(target_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # opt_local_n, schedule_local_n = make_optimizer_and_schedule(local_n, lr=args.lr_mi)
    # opt_global_n, schedule_global_n = make_optimizer_and_schedule(global_n, lr=args.lr_mi)
    # opt_local_a, schedule_local_a = make_optimizer_and_schedule(local_a, lr=args.lr_mi)
    # opt_global_a, schedule_global_a = make_optimizer_and_schedule(global_a, lr=args.lr_mi)

    # warm up
    print('--------Warm up--------')
    for epocah in range(0, 2):
        for batch_idx, (data, target) in enumerate(train_loader):
            target_model.train()

            data, target = data.to(device), target.to(device)

            logits_nat = target_model(data)

            loss = F.cross_entropy(logits_nat, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    # Train
    best_accuracy = 0
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch)

        print('--------Train the target model--------')

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # craft adversarial examples
            adv = pgd.projected_gradient_descent(target_model, data, eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, norm=np.inf,y=None, targeted=False)
            #adv = craft_adversarial_example_pgd(model=target_model, x_natural=data, y=target, step_size=0.007, epsilon=8/255, perturb_steps=40, distance='l_inf')

            # Train MI estimator
            loss = MI_loss(i=batch_idx, model=target_model, x_natural=data, y=target, x_adv=adv, local_n=local_n,
                           global_n=global_n, local_a=local_a, global_a=global_a, epoch=epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # evaluation
        print('--------Evaluate the target model--------')

        test_accuracy = eval_test(model=target_model, device=device, test_loader=test_loader, local_n=local_n,
                                  global_n=global_n, local_a=local_a, global_a=global_a)

        # save checkpoint
        if test_accuracy >= best_accuracy:  # epoch % args.save_freq == 0:
            best_accuracy = test_accuracy
            '''
            torch.save(model.module.state_dict(),
                       os.path.join(model_dir, 'model-epoch{}.pt'.format(epoch)))
            '''

            torch.save(target_model.module.state_dict(),
                       os.path.join(args.model_dir, 'target_model.pth'))
            '''
            torch.save(local_n.module.state_dict(),
                       os.path.join(args.model_dir, 'local_model.pth'))
            torch.save(global_n.module.state_dict(),
                       os.path.join(args.model_dir, 'global_model.pth'))
            '''
            print('save the model')

        print('================================================================')
    name = 'resnet-new-100-MIAT-from-scratch'
    print(40*'=')
    print(f"Saving model as {name}.")
    utils.save_model(target_model, name)
    print(40*'=')

if __name__ == '__main__':
    main()
