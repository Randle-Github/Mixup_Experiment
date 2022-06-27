"""Loss functions."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import compaction.utils.metrics as metrics

class SoftCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(SoftCrossEntropyLoss, self).__init__()
        
    def forward(self, input, target):
        """
        target: (b, num_classes), mix and sum the one hot label with weights
        """
        loss = -1 * F.log_softmax(input, dim=-1) * target
        return loss.sum(-1).mean(0)

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "label_smoothing_cross_entropy": LabelSmoothingCrossEntropy,
    "mse_loss": nn.MSELoss,
    "soft_target_cross_entropy": SoftCrossEntropyLoss # 改成自己的 soft loss
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]
