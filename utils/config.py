'''
A class to centralize all the parameters that we can set.
I *think* these can be set from the command line if you don't want to use the default
'''
import argparse

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
        self.parser.add_argument('--adv-img-train', type=str, help='adv train data', default='./data/adv_train_images.npy')
        self.parser.add_argument('--adv-label-train', type=str, help='adv train label', default='./data/adv_train_labels.npy')
        self.parser.add_argument('--adv-img-test', type=str, help='adv test data', default='./data/adv_test_images.npy')
        self.parser.add_argument('--adv-label-test', type=str, help='adv test label', default='./data/adv_test_labels.npy')
        self.parser.add_argument('--SAVE-MODEL-PATH', type=str, help='path to save trained models to', default='./models/saved/')

        self.parser.add_argument('--batch-size', type=int, default=256, metavar='N', help='input batch size for training (default: 128)')
        self.parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
        self.parser.add_argument('--weight-decay', '--wd', default=2e-4, type=float, metavar='W')
        self.parser.add_argument('--lr', type=float, default=1e-1, metavar='LR', help='learning rate')
        self.parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
        self.parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
        
        #MI specific values
        self.parser.add_argument('--lr-mi', type=float, default=1e-2, metavar='LR', help='learning rate')


        self.parser.add_argument('--epsilon', default=8/255, help='perturbation')
        self.parser.add_argument('--num-steps', default=10, help='perturb number of steps')
        self.parser.add_argument('--step-size', default=0.007, help='perturb step size')

        self.parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
        self.parser.add_argument('--log-interval', type=int, default=50, metavar='N', help='how many batches to wait before logging training status')
        self.parser.add_argument('--model-dir', default='./checkpoint/wideresnet/standard_AT', help='directory of model for saving checkpoint')
        self.parser.add_argument('--save-freq', '-s', default=2, type=int, metavar='N', help='save frequency')
        self.parser.add_argument('--print_freq', default=50, type=int)
        #2 more paths from train_MI_estimator
        #self.parser.add_argument('--pre-target', default='./checkpoint/resnet_18/standard_AT/best_model.pth', help='directory of model for saving checkpoint')
        #self.parser.add_argument('--model-dir', default='./checkpoint/resnet_18/MI_estimator/alpha', help='directory of model for saving checkpoint')



        #more from train_MI_estimator
        self.parser.add_argument('--va-mode', choices=['nce', 'fd', 'dv'], default='dv')
        self.parser.add_argument('--va-fd-measure', default='JSD')
        self.parser.add_argument('--va-hsize', type=int, default=2048)
        self.parser.add_argument('--is_internal', type=bool, default=False)
        self.parser.add_argument('--is_internal_last', type=bool, default=False)

        self.stats = ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)) #mean and stdev

    def getArgs(self):
        return self.parser.parse_args()

    def getNormStats(self):
        return self.stats