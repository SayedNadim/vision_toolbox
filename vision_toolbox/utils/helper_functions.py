# =================================================================================================
#                            Helper functions                                                     #
# =================================================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# for paths
import os
import sys
import natsort

# for tensor operations
import torch

import ntpath

# for GPU acceleration
from numba import njit


# file helpers
def return_file_name_tail(path, image=True, cut_off_value=-4):
    head, tail = ntpath.split(path)
    if image:
        return tail[:cut_off_value] or ntpath.basename(tail)
    else:
        return tail or ntpath.basename(tail)


def flist_reader(flist):
    """
    flist format: image1_path\nimage2_path\n ...
    """
    image_path_list = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            image_path = line.strip().split()
            image_path_list.append(image_path)
    return natsort.natsorted(image_path_list)


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
