import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from mindspore.ops import operations as P
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter

class _NewEmptyTensorOp(nn.Cell):
    def __init__(self):
        super(_NewEmptyTensorOp, self).__init__()

    def construct(self, x, new_shape):
        return ops.NewEmptyTensor(x.dtype, new_shape)


class Conv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(*args, **kwargs)

    def construct(self, x):
        if x.numel() > 0:
            return super(Conv2d, self).construct(x)

        # get output shape
        output_shape = [(i + 2 * p - (di * (k - 1) + 1)) // d + 1
                        for i, p, di, k, d in zip(x.shape[-2:], self.padding, self.dilation,
                                                  self.kernel_size, self.stride)]

        output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
        return _NewEmptyTensorOp().construct(x, output_shape)




class ConvTranspose2d(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0):
        super(ConvTranspose2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        self.weight = Parameter(initializer('normal', [out_channels, in_channels, *kernel_size]))
        self.bias = Parameter(initializer('zeros', [out_channels]))

        self.conv_transpose = P.Conv2dTranspose(out_channel=out_channels,
                                                kernel_size=kernel_size,
                                                mode=1,
                                                pad_mode='pad',
                                                padding=padding,
                                                stride=stride,
                                                output_padding=output_padding)

    def construct(self, x):
        return self.conv_transpose(x, self.weight, self.bias)





class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super(BatchNorm2d, self).__init__(*args, **kwargs)

    def construct(self, x):
        if x.numel() > 0:
            return super(BatchNorm2d, self).construct(x)
        # get output shape
        output_shape = x.shape
        return _NewEmptyTensorOp().construct(x, output_shape)


class DFConv2d(nn.Cell):
    def __init__(self, in_channels, out_channels, with_modulated_dcn=True, kernel_size=3, stride=1,
                 groups=1, padding=1, dilation=1, deformable_groups=1, bias=False):
        super(DFConv2d, self).__init__()

        if isinstance(kernel_size, (list, tuple)):
            assert len(kernel_size) == 2
            offset_base_channels = kernel_size[0] * kernel_size[1]
        else:
            offset_base_channels = kernel_size * kernel_size
        if with_modulated_dcn:
            from mindspore.ops import operations as P
            from mindspore import Tensor
            from model.deform_conv_module import ModulatedDeformConv
            offset_channels = offset_base_channels * 3  # default: 27
            self.offset = Conv2d(in_channels,
                                 deformable_groups * offset_channels,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 pad_mode='pad',
                                 group=1,
                                 dilation=dilation)
            self.conv = ModulatedDeformConv(in_channels,
                                            out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            pad_mode='pad',
                                            padding=padding,
                                            dilation=dilation,
                                            group=groups,
                                            deformable_groups=deformable_groups,
                                            has_bias=bias)
        else:
            from mindspore.ops import operations as P
            from mindspore import Tensor
            from model.deform_conv_module import DeformConv
            offset_channels = offset_base_channels * 2  # default: 18
            self.offset = Conv2d(in_channels,
                                 deformable_groups * offset_channels,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 pad_mode='pad',
                                 group=1,
                                 dilation=dilation)
            self.conv = DeformConv(in_channels,
                                   out_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   pad_mode='pad',
                                   padding=padding,
                                   dilation=dilation,
                                   group=groups,
                                   deformable_groups=deformable_groups,
                                   has_bias=bias)
        self.with_modulated_dcn = with_modulated_dcn
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.offset_base_channels = offset_base_channels

    def construct(self, x):
        assert x.numel() > 0, "only non-empty tensors are supported"
        if x.numel() > 0:
            if not self.with_modulated_dcn:
                offset = self.offset(x)
                x = self.conv(x, offset)
            else:
                offset_mask = self.offset(x)
                split_point = self.offset_base_channels * 2
                offset = offset_mask[:, :split_point, :, :]
                mask = offset_mask[:, split_point:, :, :].sigmoid()
                x = self.conv(x, offset, mask)
            return x


class FrozenBatchNorm2d(nn.Cell):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.weight = mindspore.Parameter(initializer("ones", [n], mindspore.float32))
        self.bias = mindspore.Parameter(initializer("zeros", [n], mindspore.float32))
        self.running_mean = mindspore.Parameter(initializer("zeros", [n], mindspore.float32))
        self.running_var = mindspore.Parameter(initializer("ones", [n], mindspore.float32))

    def construct(self, x):
        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias
