"""Optimizer."""

import torch
from .adamw import AdamW

import compaction.utils.lr_policy as lr_policy
import compaction.utils.logging as logging

logger = logging.get_logger(__name__)

def construct_optimizer(model, cfg):
    """
    Construct a stochastic gradient descent or ADAM optimizer with momentum.
    Details can be found in:
    Herbert Robbins, and Sutton Monro. "A stochastic approximation method."
    and
    Diederik P.Kingma, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."

    Args:
        model (model): model to perform stochastic gradient descent
        optimization or ADAM optimization.
        cfg (config): configs of hyper-parameters of SGD or ADAM, includes base
        learning rate,  momentum, weight_decay, dampening, and etc.
    """
    # Batchnorm parameters.
    bn_params = []
    # Non-batchnorm parameters.
    non_bn_parameters = []
    for m in model.modules():
        is_bn = isinstance(m, torch.nn.modules.batchnorm._NormBase)
        for p in m.parameters(recurse=False):
            if is_bn:
                bn_params.append(p)
            else:
                non_bn_parameters.append(p)

    # Apply different weight decay to Batchnorm and non-batchnorm parameters.
    # In Caffe2 classification codebase the weight decay for batchnorm is 0.0.
    # Having a different weight decay on batchnorm might cause a performance
    # drop.
    optim_params = [
        {"params": bn_params, "weight_decay": cfg.BN.WEIGHT_DECAY},
        {"params": non_bn_parameters, "weight_decay": cfg.SOLVER.WEIGHT_DECAY},
    ]
    # Check all parameters will be passed into optimizer.
    assert len(list(model.parameters())) == len(non_bn_parameters) + len(
        bn_params
    ), "parameter size does not match: {} + {} != {}".format(
        len(non_bn_parameters), len(bn_params), len(list(model.parameters()))
    )

    if cfg.SOLVER.OPTIMIZING_METHOD == "sgd":
        return torch.optim.SGD(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            dampening=cfg.SOLVER.DAMPENING,
            nesterov=cfg.SOLVER.NESTEROV,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adam":
        return torch.optim.Adam(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adamw":
        return AdamW(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            amsgrad=False
        )
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(cfg.SOLVER.OPTIMIZING_METHOD)
        )

def custom_optimizer(model, cfg):
    """
    修改的上面的 construct_optimizer 函数。给 box 和 vis 两个网络设置两个 optimizer

    Args:
        model (model): model to perform stochastic gradient descent
        optimization or ADAM optimization.
        cfg (config): configs of hyper-parameters of SGD or ADAM, includes base
        learning rate,  momentum, weight_decay, dampening, and etc.
    """
    vis_params = []
    box_params = []
    box_params_names = []
    for name, param in model.named_parameters():
        if 'box' in name and param.requires_grad:
            box_params.append(param)
            box_params_names.append(name)
        elif param.requires_grad:
            vis_params.append(param)
        else: pass
    logger.info(f'box paprams: {box_params_names}')
    # Apply different weight decay to Batchnorm and non-batchnorm parameters.
    # In Caffe2 classification codebase the weight decay for batchnorm is 0.0.
    # Having a different weight decay on batchnorm might cause a performance
    # drop.

    box_opt = AdamW(
        params=box_params,
        lr=cfg.SOLVER.BOX_LR,
        betas=(0.9, 0.999),
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        amsgrad=False)

    vis_opt = AdamW(
        params=vis_params,
        lr=cfg.SOLVER.BASE_LR,
        betas=(0.9, 0.999),
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        amsgrad=False)

    return [box_opt, vis_opt]

def get_epoch_lr(cur_epoch, cfg):
    """
    Retrieves the lr for the given epoch (as specified by the lr policy).
    Args:
        cfg (config): configs of hyper-parameters of ADAM, includes base
        learning rate, betas, and weight decays.
        cur_epoch (float): the number of epoch of the current training stage.
    """
    return lr_policy.get_lr_at_epoch(cfg, cur_epoch)

def set_lr(optimizer, new_lr):
    """
    Sets the optimizer lr to the specified value.
    Args:
        optimizer (optim): the optimizer using to optimize the current network.
        new_lr (float): the new learning rate to set.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

