'''
Purpose of this file is to train a standard resnet18 model on the CIFAR10 data set.
The following will be used for normalization.
https://github.com/ernoult/scalingDTP/pull/36

and also see:

https://github.com/kuangliu/pytorch-cifar/issues/19

'''
# I removed a bunch of imports that I *think* were not being used. Need to double check this.
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

import torchvision.models as models

import torch.backends.cudnn as cudnn
import numpy as np

from utils import utils
from utils import config
from utils import data 

from torchinfo import summary

args = config.Configuration().getArgs()

resnet = models.resnet18(weights=None) #use pretrained=False for older version of PyTorch

print(f"Preparing to train a ResNet18 Model on CIFAR10 dataset ...")

print(64*'=')
utils.print_cuda_info()
print(64*'=')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr

    if epoch >= 90:
        lr = args.lr * 0.01
    elif epoch >= 75:
        lr = args.lr * 0.1
    #note how this is done - I've not seen that before.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            #test_loss += F.cross_entropy(output, target, size_average=False).item() #old, size_average is deprecated
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
        
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad() # zero the gradients or they accumulate
        
        loss = F.cross_entropy(model(data), target)
        loss.backward() #backprop the gradients
        optimizer.step() #update the weights

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx+1) / len(train_loader), loss.item()))


def main():
    setup_seed(42) #best seed ever
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print(f"device: {device}") #sanity check - using GPU

    #normalization stats
    stats = ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)) #mean and stdev
    #https://www.kaggle.com/code/kmldas/cifar10-resnet-90-accuracy-less-than-5-min/notebook
    #stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    
    # setup data loader
    trans_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats, inplace=False) #original code was True here from MIAT - not sure why, just making a note
    ])

    trans_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    #now I need to pull in the data and create the dataloaders
    # min: -1.9259666204452515
    # max: 2.130864143371582

    trainset = data.data_dataset(img_path=args.nat_img_train, clean_label_path=args.nat_label_train, transform=trans_train)
    testset = data.data_dataset(img_path=args.nat_img_test, clean_label_path=args.nat_label_test, transform=trans_test)

    #create data loaders
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, drop_last=False,
                                               shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, drop_last=False, shuffle=False,
                                              num_workers=4, pin_memory=True)

    model = resnet.to(device)
    model = torch.nn.DataParallel(model)

    summary(model, input_size=(args.batch_size,3,32,32), verbose=2)
    
    cudnn.benchmark = True
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)

    #now we begin the actual training
    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch) #inefficient to call this every epoch?

        # train step
        train(args, model, device, train_loader, optimizer, epoch)

        print(64*'=')
        # eval_train(model, device, train_loader)
        test_loss, test_accuracy = eval_test(model, device, test_loader)
        print(64*'=')
    
    print("Saving resnet model 'resnet18-120' ...")
    utils.save_model(model, 'resnet18-120')
    print("Save Complete. Exiting ...")

    print(f"Training Complete. Exiting ...")

if __name__ == '__main__':
    main()