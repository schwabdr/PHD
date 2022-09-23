'''
Purpose of this file is to train a standard resnet18 model on the MalImg data set.


The following were used for normalization on C10
https://github.com/ernoult/scalingDTP/pull/36

and also see:

https://github.com/kuangliu/pytorch-cifar/issues/19

I will use this code to find values for the MalImg dataset.
https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
and:
https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949
and:
https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/5?u=kuzand


'''
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader


import torch.backends.cudnn as cudnn
import numpy as np

from utils import utils
from utils import config
from utils import data 
from models.resnet_new import ResNet18

#from torchinfo import summary

args = config.Configuration().getArgs()

args.epochs = 25
args.batch_size = 128 #16 was from the small GPU on paperspace

'''
Function to find some information about our dataset.
'''
def calcStdMean():
    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.ToTensor()
    ])

    full_dataset = datasets.ImageFolder('./data/malimg_paper_dataset_imgs', transform=transform)
    print(full_dataset)

    loader = DataLoader(full_dataset,batch_size=1, num_workers=0, shuffle=False)
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, _ in loader:
        print(data.shape)
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
        break

    mean /= nb_samples
    std /= nb_samples

    #mean: tensor([0.4454, 0.4454, 0.4454])
    #std: tensor([0.3122, 0.3122, 0.3122])





print(f"Preparing to train a ResNet18 Model on MalImg dataset ...")

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

def eval_model(model, device, loader, type="Test"):
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
        
    loss /= len(loader.dataset)
    print('{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(type,
        loss, correct, len(loader.dataset),
        100. * correct / len(loader.dataset)))

    accuracy = correct / len(loader.dataset)
    return loss, accuracy


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

    #normalization stats - calculated above
    #mean: tensor([0.4454, 0.4454, 0.4454])
    #std: tensor([0.3122, 0.3122, 0.3122])

    stats = ((0.4454, 0.4454, 0.4454), (0.3122, 0.3122, 0.3122)) #mean and stdev
    
    
    # setup data loader 224x224 is the original work so we stick with that.
    trans_train = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomCrop(224, padding=10, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats, inplace=False) #original code was True here from MIAT - not sure why, just making a note
    ])

    trans_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    full_dataset = datasets.ImageFolder('./data/malimg_paper_dataset_imgs', transform=trans_train)
    print(f"len(full_dataset): {len(full_dataset)}")
    #split dataset 90% train, 10% test
    subsets_dataset = torch.utils.data.random_split(full_dataset, [8405, 934], generator=torch.Generator().manual_seed(42)) #still the best seed ever

    #data loaders [0] is train, [1] is test
    train_loader = DataLoader(subsets_dataset[0],batch_size=args.batch_size, num_workers=8, shuffle=True)
    test_loader = DataLoader(subsets_dataset[1],batch_size=args.batch_size, num_workers=8, shuffle=False)

    resnet = ResNet18(25) #25 classes

    model = resnet.to(device)
    model = torch.nn.DataParallel(model)

    #summary(model, input_size=(args.batch_size,3,32,32), verbose=2)
    
    cudnn.benchmark = True
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)

    #now we begin the actual training
    model_name = 'resnet-mal-std'
    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch) #inefficient to call this every epoch?

        # train step
        train(args, model, device, train_loader, optimizer, epoch)

        print(64*'=')
        train_loss, train_acc = eval_model(model, device, train_loader, type="Train")
        test_loss, test_accuracy = eval_model(model, device, test_loader, type="Test")
        print(64*'=')
    
        name = model_name + str(epoch)
        print(f"Saving resnet model '{name}' ...")
        utils.save_model(model, name)
        print("Save Complete. Next Epoch ...")

    print(f"Training Complete. Exiting ...")

if __name__ == '__main__':
    main()

