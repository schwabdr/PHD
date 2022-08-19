import argparse

'''
A class to centralize all the parameters that we can set.
I *think* these can be set from the comman line if you don't want to use the default
'''
class Configuration:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
        #schwab note
        #you will access these by replacing "-" with "_"
        # i.e. args.nat_img_train
        self.parser.add_argument('--nat-img-train', type=str, help='natural training data', default='./data/train_images.npy')
        self.parser.add_argument('--nat-label-train', type=str, help='natural training label', default='./data/train_labels.npy')
        self.parser.add_argument('--nat-img-test', type=str, help='natural test data', default='./data/test_images.npy')
        self.parser.add_argument('--nat-label-test', type=str, help='natural test label', default='./data/test_labels.npy')

        self.parser.add_argument('--SAVE-MODEL-PATH', type=str, help='path to save trained models to', default='./models/saved/')

        self.parser.add_argument('--batch-size', type=int, default=256, metavar='N', help='input batch size for training (default: 128)')
                    #as always, I'm changing the epochs for getting the code up and running
                    #default was 100, I'm making it 5 just to get everything running.
        self.parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
        self.parser.add_argument('--weight-decay', '--wd', default=2e-4, type=float, metavar='W')
        self.parser.add_argument('--lr', type=float, default=1e-1, metavar='LR', help='learning rate')
        self.parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
        self.parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')

        self.parser.add_argument('--epsilon', default=8/255, help='perturbation')
        self.parser.add_argument('--num-steps', default=10, help='perturb number of steps')
        self.parser.add_argument('--step-size', default=0.007, help='perturb step size')

        self.parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
        self.parser.add_argument('--log-interval', type=int, default=50, metavar='N', help='how many batches to wait before logging training status')
        self.parser.add_argument('--model-dir', default='./checkpoint/wideresnet/standard_AT', help='directory of model for saving checkpoint')
        self.parser.add_argument('--save-freq', '-s', default=2, type=int, metavar='N', help='save frequency')

    

    def getArgs(self):
        return self.parser.parse_args()