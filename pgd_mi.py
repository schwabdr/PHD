'''
This is my attempt to use the MI loss instead of cross-entropy alone to construct adversarial examples.
The 'challenge' is that the MI loss requires multiple more paramaters ... so I'll have to change these functions a bit.
The other challenge is that to compute this loss you require an actual adversarial example. Thus I propose the following.
For now I will use my precomputed adversarial examples. Then I may tune this later on.
It even seems feasible to set X_adv_0 = 1 step of FGSM to start, then use the new loss function

'''



import numpy as np
import torch
import torch.nn.functional as F

from utils.utils import clip_eta
from utils.utils import optimize_linear
from utils import config

from compute_MI import compute_loss

args = config.Configuration().getArgs()

#####################################################
#THIS CODE IS FROM train_MIAT_alpha.py
#####################################################
'''
This should be wrapped up in a utils file or something ... but for now I'll just duplicate some code and get it working ...
'''
# one issue is that model is set to "train" - don't need that for crafting adv examples
# not sure why the original code chose to make that change here in this function.


'''
Name of this function is perhaps misleading as it gives TOTAL loss, not just the MI loss.
:param alpha: (optional) float. Hyper parameter for tuning the MI portion of loss.
'''
def MI_loss(model, x_natural, y, x_adv, local_n, global_n, local_a, global_a, alpha=5.0):
    model.eval() #changed to eval not train()
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

        #see equation 8, 9 - it looks like in the actual code implmentation they leave off the lambda term E_a(h(x)) - E_n(h(x))
        loss_n = compute_loss(args=args, former_input=x_natural, latter_input=x_natural, encoder=model,
                dim_local=local_n, dim_global=global_n, v_out=True) * index

        loss_a = compute_loss(args=args, former_input=x_natural, latter_input=x_adv, encoder=model,
                               dim_local=local_n, dim_global=global_n, v_out=True) * index

        loss_a_all = loss_a # added this back in it was commented out
        loss_mea_n = torch.abs(torch.tensor(1.0).cuda() - torch.cosine_similarity(loss_n, loss_a, dim=0))


        loss_a = compute_loss(args=args, former_input=x_adv - x_natural, latter_input=x_adv, encoder=model,
                              dim_local=local_a, dim_global=global_a, v_out=True) * index

        loss_n = compute_loss(args=args, former_input=x_adv - x_natural, latter_input=x_natural, encoder=model,
                              dim_local=local_a, dim_global=global_a, v_out=True) * index

        #loss_a_all = torch.tensor(0.1).cuda() * (loss_a_all - loss_a) #added back in - it was commented out
        loss_a_all = (loss_a_all - loss_a) #added back in - it was commented out
        loss_mea_a = torch.abs(torch.tensor(1.0).cuda() - torch.cosine_similarity(loss_n, loss_a, dim=0))

        loss_mi = loss_mea_n + loss_mea_a # + loss_a_all

        print(f"loss_ce: {loss_ce}, loss_mi: {loss_mi}, 5*loss_mi {5*loss_mi}, loss_a_all: {loss_a_all}")

    else:
        loss_mi = 0.0
    # default is alpha = 5
    loss_all = loss_ce + alpha * loss_mi

    return loss_all


#####################################################
#THIS CODE IS FROM CLEVERHANS
#####################################################

"""The Fast Gradient Method attack."""

def fast_gradient_method(
    model_fns,
    x,
    x_clean,
    eps,
    norm,
    clip_min=None,
    clip_max=None,
    y=None,
    targeted=False,
    sanity_checks=False,
    alpha=5,
):
    """
    PyTorch implementation of the Fast Gradient Method.
    :param model_fns: a list of callables that takes an input tensor and returns the model logits.
    :param x: input tensor.
    :param x_clean: the clean sample - non-adv
    :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
    :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
    :param clip_min: (optional) float. Minimum float value for adversarial example components.
    :param clip_max: (optional) float. Maximum float value for adversarial example components.
    :param y: (optional) Tensor with true labels. If targeted is true, then provide the
              target label. Otherwise, only provide this parameter if you'd like to use true
              labels when crafting adversarial samples. Otherwise, model predictions are used
              as labels to avoid the "label leaking" effect (explained in this paper:
              https://arxiv.org/abs/1611.01236). Default is None.
    :param targeted: (optional) bool. Is the attack targeted or untargeted?
              Untargeted, the default, will try to make the label incorrect.
              Targeted will instead try to move in the direction of being more like y.
    :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
              memory or for unit tests that intentionally pass strange input)
    :param alpha: (optional) float. Hyper parameter for tuning the MI portion of loss.
    :return: a tensor for the adversarial example
    """
    if norm not in [np.inf, 1, 2]:
        raise ValueError(
            "Norm order must be either np.inf, 1, or 2, got {} instead.".format(norm)
        )
    if eps < 0:
        raise ValueError(
            "eps must be greater than or equal to 0, got {} instead".format(eps)
        )
    #model_fns[0] is target
    #model_fns[1] is used to craft examples - for white box they are the same
    model_fn = model_fns[0]
    if eps == 0:
        return x
    if clip_min is not None and clip_max is not None:
        if clip_min > clip_max:
            raise ValueError(
                "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                    clip_min, clip_max
                )
            )

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        assert_ge = torch.all(
            torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_ge)

    if clip_max is not None:
        assert_le = torch.all(
            torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_le)

    # x needs to be a leaf variable, of floating point type and have requires_grad being True for
    # its grad to be computed and stored properly in a backward call
    x = x.clone().detach().to(torch.float).requires_grad_(True)
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        _, y = torch.max(model_fn(x), 1)

    # Compute loss
    #loss_fn = torch.nn.CrossEntropyLoss()
    #loss = loss_fn(model_fn(x), y)
    loss = MI_loss(model_fns[1], x_clean, y, x, model_fns[2], model_fns[4], model_fns[3], model_fns[5],alpha=alpha)
    #def: loss = MI_loss(model, x_natural, y, x_adv, local_n, global_n, local_a, global_a)
    # If attack is targeted, minimize loss of target label rather than maximize loss of correct label
    if targeted:
        loss = -loss

    # Define gradient of loss wrt input
    loss.backward()
    optimal_perturbation = optimize_linear(x.grad, eps, norm)

    # Add perturbation to original example to obtain adversarial example
    adv_x = x + optimal_perturbation

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) or (clip_max is not None):
        if clip_min is None or clip_max is None:
            raise ValueError(
                "One of clip_min and clip_max is None but we don't currently support one-sided clipping"
            )
        adv_x = torch.clamp(adv_x, clip_min, clip_max)

    if sanity_checks:
        assert np.all(asserts)
    return adv_x




"""The Projected Gradient Descent attack."""


def projected_gradient_descent(
    model_fns,
    x,
    eps,
    eps_iter,
    nb_iter,
    norm,
    clip_min=None,
    clip_max=None,
    y=None,
    targeted=False,
    rand_init=True,
    rand_minmax=None,
    sanity_checks=True,
    alpha=5,
):
    """
    This class implements either the Basic Iterative Method
    (Kurakin et al. 2016) when rand_init is set to False. or the
    Madry et al. (2017) method if rand_init is set to True.
    Paper link (Kurakin et al. 2016): https://arxiv.org/pdf/1607.02533.pdf
    Paper link (Madry et al. 2017): https://arxiv.org/pdf/1706.06083.pdf
    :param model_fn: a list of callables that takes an input tensor and returns the model logits.
    :param x: input tensor.
    :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572. schwab: Used for clipping when norm=np.inf
    :param eps_iter: step size for each attack iteration Schwab: I think this is multiplied by the sign of the gradient to add to the image.
    :param nb_iter: Number of attack iterations.
    :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
    :param clip_min: (optional) float. Minimum float value for adversarial example components.
    :param clip_max: (optional) float. Maximum float value for adversarial example components.
    :param y: (optional) Tensor with true labels. If targeted is true, then provide the
              target label. Otherwise, only provide this parameter if you'd like to use true
              labels when crafting adversarial samples. Otherwise, model predictions are used
              as labels to avoid the "label leaking" effect (explained in this paper:
              https://arxiv.org/abs/1611.01236). Default is None.
    :param targeted: (optional) bool. Is the attack targeted or untargeted?
              Untargeted, the default, will try to make the label incorrect.
              Targeted will instead try to move in the direction of being more like y.
    :param rand_init: (optional) bool. Whether to start the attack from a randomly perturbed x.
    :param rand_minmax: (optional) bool. Support of the continuous uniform distribution from
              which the random perturbation on x was drawn. Effective only when rand_init is
              True. Default equals to eps.
    :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
              memory or for unit tests that intentionally pass strange input)
    :param alpha: (optional) float. Hyper parameter for tuning the MI portion of loss.
    :return: a tensor for the adversarial example
    """
    #model_fns[0] is the target model
    #model_fns[1] is the model to construct adv samples with with (same as model_fns[0] for white-box attack)
    model_fn = model_fns[0]
    if norm == 1:
        raise NotImplementedError(
            "It's not clear that FGM is a good inner loop"
            " step for PGD when norm=1, because norm=1 FGM "
            " changes only one pixel at a time. We need "
            " to rigorously test a strong norm=1 PGD "
            "before enabling this feature."
        )
    if norm not in [np.inf, 2]:
        raise ValueError("Norm order must be either np.inf or 2.")
    if eps < 0:
        raise ValueError(
            "eps must be greater than or equal to 0, got {} instead".format(eps)
        )
    if eps == 0:
        return x
    if eps_iter < 0:
        raise ValueError(
            "eps_iter must be greater than or equal to 0, got {} instead".format(
                eps_iter
            )
        )
    if eps_iter == 0:
        return x

    assert eps_iter <= eps, (eps_iter, eps)
    if clip_min is not None and clip_max is not None:
        if clip_min > clip_max:
            raise ValueError(
                "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                    clip_min, clip_max
                )
            )

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        assert_ge = torch.all(
            torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_ge)

    if clip_max is not None:
        assert_le = torch.all(
            torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_le)

    # Initialize loop variables
    if rand_init:
        if rand_minmax is None:
            rand_minmax = eps
        #schwab: I think this goes from -rand_minmax to rand_minmax b/c you can lower or raise any single pixel by this value.
        eta = torch.zeros_like(x).uniform_(-rand_minmax, rand_minmax) 
    else:
        eta = torch.zeros_like(x)

    # Clip eta
    eta = clip_eta(eta, norm, eps)
    adv_x = x + eta
    if clip_min is not None or clip_max is not None:
        adv_x = torch.clamp(adv_x, clip_min, clip_max)

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        _, y = torch.max(model_fn(x), 1)

    i = 0
    while i < nb_iter:
        adv_x = fast_gradient_method(
            model_fns,
            adv_x,
            x, #x_clean
            eps_iter,
            norm,
            clip_min=clip_min,
            clip_max=clip_max,
            y=y,
            targeted=targeted,
            alpha=alpha,
        )

        # Clipping perturbation eta to norm norm ball #schwab: eta is the perturbation. Need to clip to Norm ball.
        eta = adv_x - x
        eta = clip_eta(eta, norm, eps)
        adv_x = x + eta

        # Redo the clipping.
        # FGM already did it, but subtracting and re-adding eta can add some
        # small numerical error.
        if clip_min is not None or clip_max is not None:
            adv_x = torch.clamp(adv_x, clip_min, clip_max)
        i += 1

    asserts.append(eps_iter <= eps)
    if norm == np.inf and clip_min is not None:
        # TODO necessary to cast clip_min and clip_max to x.dtype?
        asserts.append(eps + clip_min <= clip_max)

    if sanity_checks:
        assert np.all(asserts)
    return adv_x