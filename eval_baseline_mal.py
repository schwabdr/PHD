'''
Same but for MalImg dataset.
Use for table in Chapter 4 
Purpose of this file is to evaluate the 4 models baseline. 
We want 
1) Train Acc
2) Test Acc
3) Robust Acc
The Robust Acc will be computed based on the same adv samples used for MIAT training.
The Models (all ResNet18):
1) STD (no adversarial training) - eval against the 0.03137 examples
2) MIAT-0.03137
3) MIAT-0.1
4) MIAT-0.25
'''
import os
import torch
from torchvision import transforms

from utils import config
from utils.data import data_adv_dataset 
from models.resnet_new import ResNet18


args = config.Configuration().getArgs()
stats = ((0.4454, 0.4454, 0.4454), (0.3122, 0.3122, 0.3122)) #mean and stdev

args.batch_size=64


'''
param: model: The DNN model to evaluate
param: device: cuda or cpu
param: model_adv: The DNN model for creating adversarial examples if None, then model will be used.

'''
def eval_test_w_adv(model, device, test_loader):
    model.eval()
    correct_clean = 0
    correct_adv = 0
    
    for data_clean, data_adv, target in test_loader:
        data_clean, data_adv, target = data_clean.to(device), data_adv.to(device), target.to(device)
        
        output_clean = model(data_clean)
        output_adv = model(data_adv)
        #loss_clean += F.cross_entropy(output_clean, target, reduction='sum').item() #need this line if we want to plot loss
        pred_clean = output_clean.max(1, keepdim=True)[1]
        pred_adv = output_adv.max(1, keepdim=True)[1]
        correct_clean += pred_clean.eq(target.view_as(pred_clean)).sum().item()
        correct_adv += pred_adv.eq(target.view_as(pred_adv)).sum().item()
        
    print('Clean Accuracy: {}/{} ({:.5f}%), Robust Accuracy: {}/{} ({:.5f}%)'.format(
        correct_clean, len(test_loader.dataset),
        100. * correct_clean / len(test_loader.dataset), correct_adv, len(test_loader.dataset),
        100. * correct_adv / len(test_loader.dataset)))


def main():
    print("Beginning Evaluation ...")

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"device: {device}")

    #set the model name here you want to evaluate
    #name = 'resnet-mal-std-aug-100' #input("Name of model to load: ") #for now I'll hard code 
    name = 'resnet-mal-MIAT-AT.25.40' #input("Name of model to load: ") #for now I'll hard code 
    model = ResNet18(25)
    path = str(os.path.join(args.SAVE_MODEL_PATH, name))
    
    model.load_state_dict(torch.load(os.path.join(args.SAVE_MODEL_PATH, name)))
    model.to(device)
    model = torch.nn.DataParallel(model).cuda() 
    model.eval()

    print(f"Model loaded: {name}")
    
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    trainset = data_adv_dataset(img_path=args.nat_img_train_mal, adv_img_path=args.adv_img_train_mal, clean_label_path=args.nat_label_train_mal, transform=trans)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, drop_last=False, shuffle=False,
                                              num_workers=4, pin_memory=True)

    testset = data_adv_dataset(img_path=args.nat_img_test_mal, adv_img_path=args.adv_img_test_mal, clean_label_path=args.nat_label_test_mal, transform=trans)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, drop_last=False, shuffle=False,
                                              num_workers=4, pin_memory=True)
    
    print(64*'=')
    print(f"TRAIN DATA")
    eval_test_w_adv(model, device, train_loader)

    print(64*'=')
    print(f"TEST DATA")
    
    eval_test_w_adv(model, device, test_loader)
    
    print(64*'=')

    
if __name__ == '__main__':
    main()