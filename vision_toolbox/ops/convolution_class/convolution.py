"""
Class for convolution class
"""
# =================================================================================================
#                             Imports                                                             #
# =================================================================================================
from torch import nn
from torch.nn.utils import spectral_norm as spectral_norm_fn
from torch.nn.utils import weight_norm as weight_norm_fn
from torch.nn.modules.utils import _pair


# =================================================================================================
#                             Convolution Class                                                   #
# =================================================================================================
class Conv2dBlock(nn.Module):
    def __init__(self, input_channels,
                 output_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 weight_norm='none',
                 norm='bn',
                 activation='relu',
                 padding_mode='zeros',
                 use_bias=False,
                 ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super().__init__()

        # initialize padding
        if padding_mode == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif padding_mode == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif padding_mode == 'zeros':
            self.pad = nn.ZeroPad2d(padding)
        elif padding_mode == 'none':
            self.pad = None
        else:
            assert 0, "Unsupported padding type: {}".format(padding_mode)

        # initialize normalization
        norm_dim = output_channels
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        if weight_norm == 'sn':
            self.weight_norm = spectral_norm_fn
        elif weight_norm == 'wn':
            self.weight_norm = weight_norm_fn
        elif weight_norm == 'none':
            self.weight_norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(weight_norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'swish':
            self.activation = nn.SiLU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(in_channels=input_channels,
                              out_channels=output_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=use_bias,
                              padding_mode=padding_mode)

        if self.weight_norm:
            self.conv = self.weight_norm(self.conv)

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


# =================================================================================================
#                             EOF                                                                 #
# =================================================================================================
