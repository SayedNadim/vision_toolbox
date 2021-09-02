# =================================================================================================
#                            Tensor Helper functions                                              #
# =================================================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# for tensor operations
import torch


# tensor functions
def normalize_tensor(tensor_image):
    return tensor_image.mul_(2).add_(-1)


def denormalize_tensor(tensor_image):
    return tensor_image.add_(1).div_(2)


def reduce_mean(tensor, axis=None, keepdim=False):
    if not axis:
        axis = range(len(tensor.shape))
    for i in sorted(axis, reverse=True):
        tensor = torch.mean(tensor, dim=i, keepdim=keepdim)
    return tensor


def reduce_sum(tensor, axis=None, keepdim=False):
    if not axis:
        axis = range(len(tensor.shape))
    for i in sorted(axis, reverse=True):
        tensor = torch.sum(tensor, dim=i, keepdim=keepdim)
    return tensor


def reduce_std(tensor, axis=None, keepdim=False):
    if not axis:
        axis = range(len(tensor.shape))
    for i in sorted(axis, reverse=True):
        tensor = torch.std(tensor, dim=i, keepdim=keepdim)
    return tensor
