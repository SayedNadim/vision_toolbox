# =================================================================================================
# The tests are not correctly written. I will update the test files soon.                         #
# =================================================================================================


# =================================================================================================
#                             Imports                                                             #
# =================================================================================================
import pytest
import torch
from pytorch_vision_toolbox.ops.layers import *


# =================================================================================================
#                             PyTest - test input values                                          #
# =================================================================================================

@pytest.fixture
def input_for_convolution_testing():
    batch_size = 4
    input_channels = [3, 16, 32, 64]
    output_channels = [16, 32, 64, 128]
    kernels = [1, 3, 5, 7]
    stride = [1, 2, 3, 4]
    padding = [0, 1, 2, 3]
    dilation = [1, 1, 1, 1]
    return None, None, batch_size, \
           input_channels, \
           output_channels, \
           kernels, \
           stride, \
           padding, \
           dilation


# =================================================================================================
#                             PyTest - convolution function                                       #
# =================================================================================================

# ================================ test 0 =========================================================
def test_convolution_0(input_for_convolution_testing):
    batch_size = input_for_convolution_testing[2]
    input_channels = input_for_convolution_testing[3]
    output_channels = input_for_convolution_testing[4]
    kernels = input_for_convolution_testing[5]
    stride = input_for_convolution_testing[6]
    padding = input_for_convolution_testing[7]
    dilation = input_for_convolution_testing[8]
    input_tensor_0 = torch.rand(batch_size,
                                input_channels[0],
                                64,
                                64)
    conv_function = conv2d(input_channels=input_channels[0],
                           output_channels=output_channels[0],
                           kernel_size=kernels[0],
                           stride=stride[0],
                           padding=padding[0],
                           dilation=dilation[0])
    out_0 = conv_function(input_tensor_0)
    assert out_0.shape == torch.Size([batch_size,
                                      output_channels[0],
                                      64,
                                      64])


# ================================ test 1 =========================================================
def test_convolution_1(input_for_convolution_testing):
    batch_size = input_for_convolution_testing[2]
    input_channels = input_for_convolution_testing[3]
    output_channels = input_for_convolution_testing[4]
    kernels = input_for_convolution_testing[5]
    stride = input_for_convolution_testing[6]
    padding = input_for_convolution_testing[7]
    dilation = input_for_convolution_testing[8]
    input_tensor_0 = torch.rand(batch_size,
                                input_channels[1],
                                64,
                                64)
    conv_function = conv2d(input_channels=input_channels[1],
                           output_channels=output_channels[1],
                           kernel_size=kernels[1],
                           stride=stride[1],
                           padding=padding[1],
                           dilation=dilation[1])
    out_0 = conv_function(input_tensor_0)
    assert out_0.shape == torch.Size([batch_size,
                                      output_channels[1],
                                      32,
                                      32])


# ================================ test 2 =========================================================
def test_convolution_2(input_for_convolution_testing):
    batch_size = input_for_convolution_testing[2]
    input_channels = input_for_convolution_testing[3]
    output_channels = input_for_convolution_testing[4]
    kernels = input_for_convolution_testing[5]
    stride = input_for_convolution_testing[6]
    padding = input_for_convolution_testing[7]
    dilation = input_for_convolution_testing[8]
    input_tensor_0 = torch.rand(batch_size,
                                input_channels[2],
                                64,
                                64)
    conv_function = conv2d(input_channels=input_channels[2],
                           output_channels=output_channels[2],
                           kernel_size=kernels[2],
                           stride=stride[2],
                           padding=padding[2],
                           dilation=dilation[2])
    out_0 = conv_function(input_tensor_0)
    assert out_0.shape == torch.Size([batch_size,
                                      output_channels[2],
                                      22,
                                      22])


# ================================ test 3 =========================================================
def test_convolution_3(input_for_convolution_testing):
    batch_size = input_for_convolution_testing[2]
    input_channels = input_for_convolution_testing[3]
    output_channels = input_for_convolution_testing[4]
    kernels = input_for_convolution_testing[5]
    stride = input_for_convolution_testing[6]
    padding = input_for_convolution_testing[7]
    dilation = input_for_convolution_testing[8]
    input_tensor_0 = torch.rand(batch_size,
                                input_channels[3],
                                64,
                                64)
    conv_function = conv2d(input_channels=input_channels[3],
                           output_channels=output_channels[3],
                           kernel_size=kernels[3],
                           stride=stride[3],
                           padding=padding[3],
                           dilation=dilation[3])
    out_0 = conv_function(input_tensor_0)
    assert out_0.shape == torch.Size([batch_size,
                                      output_channels[3],
                                      16,
                                      16])
