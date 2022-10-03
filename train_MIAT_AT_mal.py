'''
This is the code to train a network using the MIAT method.
-replaced default adv ex crafting with Cleverhans PGD implementation 
    original parms: step_size=0.007, epsilon=8/255, perturb_steps=40, distance='l_inf'

See paper Adversarial Machine Learning at Scale
http://arxiv.org/abs/1611.01236

Making two changes to MIAT here:
1) Randomize epsilon 
2) Apply the minibatch technique to train on a mixture of clean / adv examples (see pg 3-4 of paper)

Plan - during training for each batch I'll randomly choose for each image if I should create an adversarial example or not.
This will be the batch for training.
loss_ce will be used for non-adversarial samples.
loss_mi will be used for adversarial examples.
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


from models.resnet_new import ResNet18
#from models.wideresnet_new import WideResNet #might use eventually

from models.estimator import Estimator
from models.discriminators import MI1x1ConvNet, MIInternalConvNet, MIInternallastConvNet
from compute_MI import compute_loss

import projected_gradient_descent as pgd #cleverhans PGD

from utils import config
from utils import utils
from utils.data import data_dataset

#normalization stats - calculated above
#mean: tensor([0.4454, 0.4454, 0.4454])
#std: tensor([0.3122, 0.3122, 0.3122])

stats = ((0.4454, 0.4454, 0.4454), (0.3122, 0.3122, 0.3122)) #mean and stdev

args = config.Configuration().getArgs()

args.model_dir = './checkpoint/wideresnet/MIAT_standard'

#PGD Parameters
#eps = .1
eps_iter = .007
nb_iter = 50

args.batch_size = 48 #48 uses almost 80GB of VRAM with the MalImg dataset

print(f"Using PGD with eps: random range .1-.5, eps_iter: {eps_iter}, nb_iter: {nb_iter}")
print("This is the .1 Version - Normalized")


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
        loss_mi = torch.tensor(0.0).cuda() #changed to tensor or else print below may crash - loss_mi.item()

    loss_all = loss_ce + loss_mi
    
    if (i + 1) % 10 == 0:
        print('select samples:' + str(torch.nonzero(index).size(0)))
        print('Epoch [%d], Iter [%d/%d] Train target model. Natural MI: %.4f; Loss_ce: %.4f; Loss_all: %.4f'
              % (epoch, i + 1, 8405 // args.batch_size, loss_mi.item(), loss_ce.item(), loss_all.item()))

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

        loss_a_all = loss_a
        loss_mea_n = torch.abs(torch.tensor(1.0).cuda() - torch.cosine_similarity(loss_n, loss_a, dim=0))


        loss_a = compute_loss(args=args, former_input=x_adv - x_natural, latter_input=x_adv, encoder=model,
                              dim_local=local_a, dim_global=global_a, v_out=True) * index

        loss_n = compute_loss(args=args, former_input=x_adv - x_natural, latter_input=x_natural, encoder=model,
                              dim_local=local_a, dim_global=global_a, v_out=True) * index

        loss_a_all = (loss_a_all - loss_a)
        loss_a_all = loss_a_all.sum()/(torch.nonzero(index).size(0))
        loss_a_all = torch.abs(torch.tensor(.1).cuda() * loss_a_all)

        loss_mea_a = torch.abs(torch.tensor(1.0).cuda() - torch.cosine_similarity(loss_n, loss_a, dim=0))


        loss_mi = loss_mea_n + loss_mea_a  + loss_a_all

    else:
        loss_mi = torch.tensor(0.0).cuda() #changed to tensor or else print below may crash - loss_mi.item()

    loss_all = loss_ce + 5.0 * loss_mi

    if (i + 1) % 10 == 0:
        print('select samples:' + str(torch.nonzero(index).size(0)))
        print('Epoch [%d], Iter [%d/%d] Train target model. Loss MI: %.4f; Loss_ce: %.4f; Loss_all: %.4f'
              % (epoch, i + 1, 8405 // args.batch_size, loss_mi.item(), loss_ce.item(), loss_all.item()))
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
        eps = np.random.rand(1)[0] * .4 + .1 #put eps in range [.1,.5]
        data_adv = pgd.projected_gradient_descent(model, data, eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, norm=np.inf,y=None, targeted=False)
        
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
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    print(f"device: {device}") #sanity check - using GPU

    # setup data loader - I added the consistent normalization I've been using.
    ############### LEFT AS A REMINDER ! FORGOT TO REMOVE THE RANDOM CROP WHICH WAS CUTTING MY IMAGES AND RUINING TRAINING!!! ##################
    '''
    trans_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats, inplace=False)
    ])
    '''
    # images are already resized on disk as npy arrays
    trans = transforms.Compose([
        transforms.Resize((224,224)), #don't think I need this?
        transforms.ToTensor(),
        #transforms.Normalize(*stats, inplace=True) #this standardizes mean 0, stddev 1 - so leave the data just normalized, not standardized
    ])

    trainset = data_dataset(img_path=args.nat_img_train_mal, clean_label_path=args.nat_label_train_mal, transform=trans)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, drop_last=False, shuffle=True, num_workers=8, pin_memory=True)
    testset = data_dataset(img_path=args.nat_img_test_mal, clean_label_path=args.nat_label_test_mal, transform=trans)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, drop_last=False, shuffle=False, num_workers=8, pin_memory=True)

    # Estimator part 1: X or layer3 to H space
    local_n = Estimator(args.va_hsize)
    local_a = Estimator(args.va_hsize)
    #args.is_internal = True
    #args.is_internal_last = True

    # estimator part 2: Z to H space
    if args.is_internal == True:
        if args.is_internal_last == True:
            z_size = 512
            global_n = MIInternallastConvNet(z_size, args.va_hsize)
            global_a = MIInternallastConvNet(z_size, args.va_hsize)
        else:
            z_size = 256
            global_n = MIInternalConvNet(z_size, args.va_hsize) #2048 is default
            global_a = MIInternalConvNet(z_size, args.va_hsize)
    else:
        z_size = 25
        global_n = MI1x1ConvNet(z_size, args.va_hsize)
        global_a = MI1x1ConvNet(z_size, args.va_hsize)

    target_model = ResNet18(25)
    #next three lines if you want to start with the pretrained standard model.
    #name = 'resnet-mal-std-aug-100' #input("Name of model to load: ") #for now I'll hard code so I don't have to retype the name while prototyping
    #print(f"Target Model Loaded (encoder): {name}")
    #target_model.load_state_dict(torch.load(os.path.join(args.SAVE_MODEL_PATH, name)))
    
    target_model.to(device)
    target_model = torch.nn.DataParallel(target_model).cuda()

    local_n.load_state_dict(torch.load(os.path.join(args.SAVE_MODEL_PATH, 'local_n_mal.1')))
    global_n.load_state_dict(torch.load(os.path.join(args.SAVE_MODEL_PATH, 'global_n_mal.1')))
    local_a.load_state_dict(torch.load(os.path.join(args.SAVE_MODEL_PATH, 'local_a_mal.1')))
    global_a.load_state_dict(torch.load(os.path.join(args.SAVE_MODEL_PATH, 'global_a_mal.1')))

    local_n = local_n.to(device)
    global_n = global_n.to(device)
    local_a = local_a.to(device)
    global_a = global_a.to(device)

    local_n = torch.nn.DataParallel(local_n).cuda()
    global_n = torch.nn.DataParallel(global_n).cuda()
    local_a = torch.nn.DataParallel(local_a).cuda()
    global_a = torch.nn.DataParallel(global_a).cuda()

    cudnn.benchmark = True

    #optimizer = optim.SGD(target_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.Adam(target_model.parameters(), lr=args.lr)

    #print("Pre-Warm Up Acc:")
    #_ = eval_test(model=target_model, device=device, test_loader=test_loader, local_n=local_n, global_n=global_n, local_a=local_a, global_a=global_a)

    # warm up
    
    print('--------Warm up--------')
    for epocah in range(0, 2):
        for batch_idx, (data, target) in enumerate(train_loader):
            target_model.train()
            data, target = data.to(device), target.to(device)
            
            #print(f"correct dim{data.shape[0]}")
            logits_nat = target_model(data)
            
            optimizer.zero_grad()
            loss = F.cross_entropy(logits_nat, target)
            loss.backward()
            optimizer.step()

    #print("Post-Warm Up Acc:")
    #_ = eval_test(model=target_model, device=device, test_loader=test_loader, local_n=local_n, global_n=global_n, local_a=local_a, global_a=global_a)
    
    # Train
    target_model.train()
    best_accuracy = 0
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch)
        print(f"epoch: {epoch}")

        print('--------Train the target model--------')

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            #data.shape torch.Size([48, 3, 224, 224])
            #print(f"data.shape {data.shape}")
            #print(data[0].shape) #[3, 224, 224]
            #print(data[0]) #a single image sample at index 0
            
            first = True
            ac = 0 #adversarial count -> count of adversarial examples
            for i in range(data.shape[0]): # should be batch size
                if np.random.rand(1)[0] < .3: #using .3 as threshold per original paper
                    
                    # craft adversarial examples
                    eps = np.random.rand(1)[0] * .4 + .1 #put eps in range [.1,.5]
                    #print(f"eps: {eps}")
                    x_nat = torch.unsqueeze(data[i-ac],0)
                    #print(f"torch.unsqueeze(data[i],0).shape: {adv.shape}") #[1,3,224,224]
                    adv = pgd.projected_gradient_descent(target_model, x_nat, eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, norm=np.inf,y=None, targeted=False)
                    adv_label = torch.unsqueeze(target[i-ac],0) #correct label for adv example
                    #remove the adversarial example row from the clean samples
                    data = torch.cat((data[:i-ac],data[i+1-ac:]))
                    # and labels
                    target = torch.cat((target[:i-ac],target[i+1-ac:]))

                    if first:
                        first = False
                        data_adv = adv
                        target_adv = adv_label
                        x_natural = x_nat
                    else:
                        data_adv = torch.cat((data_adv, adv))
                        target_adv = torch.cat((target_adv, adv_label))
                        x_natural = torch.cat((x_natural, x_nat)) #MI_loss needs the natural sample in addition to the adv ex
                    ac += 1
            '''
            #print shapes make sure it looks right
            print(f"data.shape: {data.shape}")
            print(f"target.shape: {target.shape}")
            print(f"data_adv.shape: {data_adv.shape}")
            print(f"target_adv.shape: {target_adv.shape}")
            ''' 
            # Train MI estimator
            if ac == 0: #randomly didn't set any adversarial examples this epoch
                loss_mi = torch.tensor(0.0).cuda()
            else:
                loss_mi = MI_loss(i=batch_idx, model=target_model, x_natural=x_natural, y=target_adv, x_adv=data_adv, local_n=local_n,
                           global_n=global_n, local_a=local_a, global_a=global_a, epoch=epoch)
            
            '''
            I got the following error:
            Traceback (most recent call last):
                File "train_MIAT_AT_mal.py", line 502, in <module>
                    main()
                File "train_MIAT_AT_mal.py", line 483, in main
                    loss_total.backward()
                File "/home/dschwab/.conda/envs/phd02/lib/python3.8/site-packages/torch/_tensor.py", line 396, in backward
                    torch.autograd.backward(self, gradient, retain_graph, create_graph, inputs=inputs)
                File "/home/dschwab/.conda/envs/phd02/lib/python3.8/site-packages/torch/autograd/__init__.py", line 173, in backward
                    Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
                RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward.
            The if statements below are my best guess as to what happened - maybe the adversarial set or non-adversarial set accidentally didn't have any samples in it (it could happen)
            So ... hopefully the below will protect. If that was indeed the problem.
            '''
            #protect against empty natural set
            if data.size()[0] == 0:
                loss_ce = torch.tensor(0.0).cuda()
            else: 
                logits_nat = target_model(data)
                loss_ce = F.cross_entropy(logits_nat, target)
            
            loss_total = loss_mi + loss_ce
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

        # evaluation
        print('--------Evaluate the target model--------')

        #we don't need to evaluate every epoch - this will speed it up
        if epoch % 5 == 0:
            test_accuracy = eval_test(model=target_model, device=device, test_loader=test_loader, local_n=local_n,
                                  global_n=global_n, local_a=local_a, global_a=global_a)
            name = 'resnet-mal-MIAT-AT-Norm.1.' + str(epoch) #not pretrained resnet used to start
            print(40*'=')
            print(f"Saving model as {name}.")
            utils.save_model(target_model, name)
            print(40*'=')

        print('================================================================')
    name = 'resnet-mal-MIAT-AT-Norm.1' #not pretrained resnet used to start
    print(40*'=')
    print(f"Saving model as {name}.")
    utils.save_model(target_model, name)
    print(40*'=')

if __name__ == '__main__':
    main()
