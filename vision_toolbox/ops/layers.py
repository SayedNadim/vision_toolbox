from pytorch_vision_toolbox.ops.convolution_class.convolution import Conv2dBlock
from pytorch_vision_toolbox.ops.convolution_class.transposed_convolution import TransposedConv2dBlock
from pytorch_vision_toolbox.ops.convolution_class.depth_wise_separable_convolution import DepthwiseSeparableConv2dBlock
from pytorch_vision_toolbox.ops.convolution_class.upsample_convolution import UpsampleConv2dBlock


# =================================================================================================
#                             Convolution function                                                #
# =================================================================================================

def conv2d(input_channels,
           output_channels,
           kernel_size,
           stride,
           padding,
           dilation,
           weight_norm='none',
           norm='bn',
           activation='relu',
           padding_mode='zeros',
           use_bias=False):
    return Conv2dBlock(input_channels=input_channels,
                       output_channels=output_channels,
                       kernel_size=kernel_size,
                       stride=stride,
                       padding=padding,
                       dilation=dilation,
                       weight_norm=weight_norm,
                       norm=norm,
                       activation=activation,
                       padding_mode=padding_mode,
                       use_bias=use_bias)


# =================================================================================================
#                             Transposed Convolution function                                     #
# =================================================================================================

def conv2d_transposed(input_channels,
                      output_channels,
                      kernel_size,
                      stride,
                      padding,
                      output_padding=1,
                      dilation=1,
                      weight_norm='none',
                      norm='bn',
                      activation='relu',
                      padding_mode='zeros',
                      use_bias=False):
    return TransposedConv2dBlock(input_channels=input_channels,
                                 output_channels=output_channels,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 output_padding=output_padding,
                                 dilation=dilation,
                                 weight_norm=weight_norm,
                                 norm=norm,
                                 activation=activation,
                                 padding_mode=padding_mode,
                                 use_bias=use_bias)


# =================================================================================================
#                             Depthwise Separable Convolution function                            #
# =================================================================================================

def conv2d_depthwise_separable(input_channels,
                               output_channels,
                               kernel_size,
                               stride,
                               padding,
                               dilation,
                               weight_norm='none',
                               norm='bn',
                               activation='relu',
                               padding_mode='zeros',
                               use_bias=False):
    return DepthwiseSeparableConv2dBlock(input_channels=input_channels,
                                         output_channels=output_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         weight_norm=weight_norm,
                                         norm=norm,
                                         activation=activation,
                                         padding_mode=padding_mode,
                                         use_bias=use_bias)


# =================================================================================================
#                             Upsample Convolution function                                                #
# =================================================================================================

def conv2d_transposed_upsample(input_channels,
                               output_channels,
                               kernel_size,
                               stride,
                               padding,
                               dilation,
                               weight_norm='none',
                               norm='bn',
                               activation='relu',
                               padding_mode='zeros',
                               use_bias=False):
    return UpsampleConv2dBlock(input_channels=input_channels,
                               output_channels=output_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               weight_norm=weight_norm,
                               norm=norm,
                               activation=activation,
                               padding_mode=padding_mode,
                               use_bias=use_bias)
