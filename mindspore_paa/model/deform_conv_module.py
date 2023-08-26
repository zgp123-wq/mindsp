import math
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.initializer import initializer
from mindspore.common import Tensor
from model.deform_conv_func import deform_conv, modulated_deform_conv

def _pair(val):
    if isinstance(val, (list, tuple)):
        return val
    return (val, val)

class DeformConv(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, deformable_groups=1, bias=False):
        super(DeformConv, self).__init__()
        self.with_bias = bias

        assert in_channels % groups == 0, f'in_channels {in_channels} cannot be divisible by groups {groups}'
        assert out_channels % groups == 0, f'out_channels {out_channels} cannot be divisible by groups {groups}'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.weight = nn.Parameter(initializer('uniform', [out_channels, in_channels // self.groups, *self.kernel_size]))

        if self.with_bias:
            self.bias = nn.Parameter(initializer('zeros', [out_channels]))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.set_data(initializer('uniform', [self.out_channels, self.in_channels // self.groups, *self.kernel_size], -stdv, stdv))
        if self.with_bias:
            self.bias.set_data(initializer('zeros', [self.out_channels]))

    def construct(self, input, offset):
        y = deform_conv(input, offset, self.weight, self.stride, self.padding, self.dilation,
                        self.groups, self.deformable_groups)
        if self.with_bias:
            assert y.shape[1] == 4
            y = y + self.bias.reshape(1, -1, 1, 1)
        return y
    
    def __repr__(self):
        return "".join(["{}(".format(self.__class__.__name__),
                        "in_channels={}, ".format(self.in_channels),
                        "out_channels={}, ".format(self.out_channels),
                        "kernel_size={}, ".format(self.kernel_size),
                        "stride={}, ".format(self.stride),
                        "dilation={}, ".format(self.dilation),
                        "padding={}, ".format(self.padding),
                        "groups={}, ".format(self.groups),
                        "deformable_groups={}, ".format(self.deformable_groups),
                        "bias={})".format(self.with_bias)])
        
class ModulatedDeformConv(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, deformable_groups=1, bias=True):
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias
        self.weight = nn.Parameter(initializer('uniform', [out_channels, in_channels // groups, *self.kernel_size]))

        if bias:
            self.bias = nn.Parameter(initializer('zeros', [out_channels]))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.set_data(initializer('uniform', [self.out_channels, self.in_channels // self.groups, *self.kernel_size], -stdv, stdv))
        if self.bias is not None:
            self.bias.set_data(initializer('zeros', [self.out_channels]))

    def construct(self, input, offset, mask):
        return modulated_deform_conv(input, offset, mask, self.weight, self.bias, self.stride,
                                     self.padding, self.dilation, self.groups, self.deformable_groups)
    
    
    def __repr__(self):
        return "".join(["{}(".format(self.__class__.__name__),
                        "in_channels={}, ".format(self.in_channels),
                        "out_channels={}, ".format(self.out_channels),
                        "kernel_size={}, ".format(self.kernel_size),
                        "stride={}, ".format(self.stride),
                        "dilation={}, ".format(self.dilation),
                        "padding={}, ".format(self.padding),
                        "groups={}, ".format(self.groups),
                        "deformable_groups={}, ".format(self.deformable_groups),
                        "bias={})".format(self.with_bias)])
        
class ModulatedDeformConvPack(ModulatedDeformConv):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, deformable_groups=1, bias=True):
        super(ModulatedDeformConvPack, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                                      groups, deformable_groups, bias)

        self.conv_offset_mask = nn.Conv2d(self.in_channels // self.groups,
                                          self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
                                          kernel_size=self.kernel_size,
                                          stride=_pair(self.stride),
                                          padding=_pair(self.padding),
                                          has_bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.set_data(initializer('zeros', self.conv_offset_mask.weight.shape()))
        self.conv_offset_mask.bias.set_data(initializer('zeros', self.conv_offset_mask.bias.shape()))

    def construct(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = P.Split(3)(out)
        offset = P.Concat(axis=1)((o1, o2))
        mask = P.Sigmoid()(mask)

        return modulated_deform_conv(input, offset, mask, self.weight, self.bias, self.stride,
                                     self.padding, self.dilation, self.groups, self.deformable_groups)
