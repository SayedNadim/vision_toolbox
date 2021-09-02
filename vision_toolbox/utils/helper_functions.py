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
