''' MISC utils in this file 
    - displaying images
    - CleverHans support functions
    - saving/loading a model
'''
import os

import numpy as np
import torch

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from utils import config

args = config.Configuration().getArgs()
#print(args.SAVE_MODEL_PATH)

#displays a grid of images with labels.
#XXXXXwe assume input is in shape B, C, H, W (img count, channels, height, width)
# assume input is in shape B, H, W, C
#assume X is NOT normalized - caller must ensure this for now.
def displayRandomImgGrid(X, Y, classes, rows=5, cols=5, Y_hat=None):
    #Now lets show some images.
    # thank you stack exchange
    #https://stackoverflow.com/questions/35995999/why-cifar-10-images-are-not-displayed-properly-using-matplotlib
    #https://stackoverflow.com/questions/2659312/how-do-i-convert-a-numpy-array-to-and-display-an-image
    #X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    #For adding labels see this: https://stackoverflow.com/questions/42435446/how-to-put-text-outside-python-plots
    #https://stackoverflow.com/questions/61341119/write-a-text-inside-a-subplot
    #X_v = X.transpose(0,2,3,1).astype("uint8") #shouldn't need this line since I now transpose the data for the ML model to learn
    X_v = X
    #Visualizing CIFAR 10
    fig, axes1 = plt.subplots(rows,cols,figsize=(5,5))
    for j in range(rows):
        for k in range(cols):
            i = np.random.choice(range(len(X_v)))
            axes1[j][k].set_axis_off()
            axes1[j][k].imshow(X_v[i:i+1][0],interpolation='nearest')
            axes1[j][k].text(0,0,classes[Y[i]]) # this gets the point accross but needs fixing.
            if Y_hat is not None:
                print("not implemented yet") #add the predicted label
    plt.show()




#this is from the old blackbox CIFAR10 code utils.py file.
#save a model to disk to reuse it after it's been trained
#note that a model can simply be loaded with
# model = torch.load(os.path.join(args.SAVE_MODEL_PATH, bb))
# where bb is the name of the saved model to load
# it may be a bit more complex than that ...
# see here: https://pytorch.org/tutorials/beginner/saving_loading_models.html
def save_model(model, name=None):
    if not os.path.exists(args.SAVE_MODEL_PATH):
        os.mkdir(args.SAVE_MODEL_PATH)
    files = os.listdir(args.SAVE_MODEL_PATH)
    if name is None:
        while True:
            filename = input("Enter filename for saved model: ")
            if filename in files:
                response = input("Warning! File already exists. Overwrite? (y/n) : ")
                if response.strip() in ("Y", "y"):
                    break
                else:
                    continue
            break
    else:
        filename = name #dangerous - will overwrite model - but this prevents losing a trained model when I DC
    # had to use model.module instead of model b/c it's a parallel trained model.
    torch.save(model.module.state_dict(), os.path.join(args.SAVE_MODEL_PATH, filename))

#utility to verify cuda information for current hardware environment
def print_cuda_info():
    cuda_dev_count = torch.cuda.device_count()    
    print("cuda devices: " + str(cuda_dev_count))

    for i in range(cuda_dev_count):
        print("Device " + str(i) + ": " + str(torch.cuda.get_device_name(i)) + ": Capability: " + str(torch.cuda.get_device_capability(i)))


def print_tensor_details(name, data):
    print("Printing details for: " + name)
    print("len(data): " + str(len(data)))
    #print(data)
    print('\n*****************' + name + '*******************\n\n')
    print("data.ndim: " + str(data.ndim) + "\n")
    print("data.shape: " + str(data.shape) + "\n")
    print("data.size: " + str(data.size) + "\n")
    print("data.dtype: " + str(data.dtype) + "\n")


#####################################################
#BEGIN CLEVERHANS UTILS FILE
#####################################################
"""Utils for PyTorch"""

#schwab: eta is the perturbation added each iteration of BIM/PGD
#must be clipped so that we don't stray outside of L2 or Linf norm constraint
def clip_eta(eta, norm, eps):
    """
    PyTorch implementation of the clip_eta in utils_tf.

    :param eta: Tensor
    :param norm: np.inf, 1, or 2
    :param eps: float
    """
    if norm not in [np.inf, 1, 2]:
        raise ValueError("norm must be np.inf, 1, or 2.")

    avoid_zero_div = torch.tensor(1e-12, dtype=eta.dtype, device=eta.device)
    reduc_ind = list(range(1, len(eta.size())))
    if norm == np.inf:
        eta = torch.clamp(eta, -eps, eps)
    else:
        if norm == 1:
            raise NotImplementedError("L1 clip is not implemented.")
            norm = torch.max(
                avoid_zero_div, torch.sum(torch.abs(eta), dim=reduc_ind, keepdim=True)
            )
        elif norm == 2:
            norm = torch.sqrt(
                torch.max(
                    avoid_zero_div, torch.sum(eta ** 2, dim=reduc_ind, keepdim=True)
                )
            )
        factor = torch.min(
            torch.tensor(1.0, dtype=eta.dtype, device=eta.device), eps / norm
        )
        eta *= factor
    return eta


def get_or_guess_labels(model, x, **kwargs):
    """
    Get the label to use in generating an adversarial example for x.
    The kwargs are fed directly from the kwargs of the attack.
    If 'y' is in kwargs, then assume it's an untargeted attack and
    use that as the label.
    If 'y_target' is in kwargs and is not none, then assume it's a
    targeted attack and use that as the label.
    Otherwise, use the model's prediction as the label and perform an
    untargeted attack.

    :param model: PyTorch model. Do not add a softmax gate to the output.
    :param x: Tensor, shape (N, d_1, ...).
    :param y: (optional) Tensor, shape (N).
    :param y_target: (optional) Tensor, shape (N).
    """
    if "y" in kwargs and "y_target" in kwargs:
        raise ValueError("Can not set both 'y' and 'y_target'.")
    if "y" in kwargs:
        labels = kwargs["y"]
    elif "y_target" in kwargs and kwargs["y_target"] is not None:
        labels = kwargs["y_target"]
    else:
        _, labels = torch.max(model(x), 1)
    return labels


def optimize_linear(grad, eps, norm=np.inf):
    """
    Solves for the optimal input to a linear function under a norm constraint.

    Optimal_perturbation = argmax_{eta, ||eta||_{norm} < eps} dot(eta, grad)

    :param grad: Tensor, shape (N, d_1, ...). Batch of gradients
    :param eps: float. Scalar specifying size of constraint region
    :param norm: np.inf, 1, or 2. Order of norm constraint.
    :returns: Tensor, shape (N, d_1, ...). Optimal perturbation
    """

    red_ind = list(range(1, len(grad.size())))
    avoid_zero_div = torch.tensor(1e-12, dtype=grad.dtype, device=grad.device)
    if norm == np.inf:
        # Take sign of gradient
        optimal_perturbation = torch.sign(grad)
    elif norm == 1:
        abs_grad = torch.abs(grad)
        sign = torch.sign(grad)
        red_ind = list(range(1, len(grad.size())))
        abs_grad = torch.abs(grad)
        ori_shape = [1] * len(grad.size())
        ori_shape[0] = grad.size(0)

        max_abs_grad, _ = torch.max(abs_grad.view(grad.size(0), -1), 1)
        max_mask = abs_grad.eq(max_abs_grad.view(ori_shape)).to(torch.float)
        num_ties = max_mask
        for red_scalar in red_ind:
            num_ties = torch.sum(num_ties, red_scalar, keepdim=True)
        optimal_perturbation = sign * max_mask / num_ties
        # TODO integrate below to a test file
        # check that the optimal perturbations have been correctly computed
        opt_pert_norm = optimal_perturbation.abs().sum(dim=red_ind)
        assert torch.all(opt_pert_norm == torch.ones_like(opt_pert_norm))
    elif norm == 2:
        square = torch.max(avoid_zero_div, torch.sum(grad ** 2, red_ind, keepdim=True))
        optimal_perturbation = grad / torch.sqrt(square)
        # TODO integrate below to a test file
        # check that the optimal perturbations have been correctly computed
        opt_pert_norm = (
            optimal_perturbation.pow(2).sum(dim=red_ind, keepdim=True).sqrt()
        )
        one_mask = (square <= avoid_zero_div).to(torch.float) * opt_pert_norm + (
            square > avoid_zero_div
        ).to(torch.float)
        assert torch.allclose(opt_pert_norm, one_mask, rtol=1e-05, atol=1e-08)
    else:
        raise NotImplementedError(
            "Only L-inf, L1 and L2 norms are " "currently implemented."
        )

    # Scale perturbation to be the solution for the norm=eps rather than
    # norm=1 problem
    scaled_perturbation = eps * optimal_perturbation
    return scaled_perturbation


def zero_out_clipped_grads(grad, x, clip_min, clip_max):
    """
    Helper function to erase entries in the gradient where the update would be
    clipped.
    :param grad: The gradient
    :param x: The current input
    :param clip_min: Minimum input component value
    :param clip_max: Maximum input component value
    """
    signed_grad = torch.sign(grad)

    # Find input components that lie at the boundary of the input range, and
    # where the gradient points in the wrong direction.
    clip_low = torch.le(x, clip_min) & torch.lt(signed_grad, 0)
    clip_high = torch.ge(x, clip_max) & torch.gt(signed_grad, 0)
    clip = clip_low | clip_high
    grad = torch.where(clip, torch.zeros_like(grad), grad)

    return grad

#####################################################
#END CLEVERHANS UTILS FILE
#####################################################