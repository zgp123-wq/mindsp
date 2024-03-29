   # 使用默认的随机初始化
import mindspore.nn as nn
import mindspore.ops.operations as P
from model.layers import Conv2d  
from mindspore.common.initializer import XavierUniform,Normal, Constant, Zero

from mindspore.common.initializer import initializer

class FPN(nn.Cell):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        self.inner_blocks = []
        self.layer_blocks = []
        for idx, in_channels in enumerate(in_channels_list, 1):
            inner_block = "fpn_inner{}".format(idx)
            layer_block = "fpn_layer{}".format(idx)

            if in_channels == 0:
                continue
            inner_block_module = self.make_conv(in_channels, out_channels, 1)
            layer_block_module = self.make_conv(out_channels, out_channels, 3, 1)
            setattr(self, inner_block, inner_block_module)
            setattr(self, layer_block, layer_block_module)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)

        self.top_blocks = LastLevelP6P7(256, out_channels)

    def make_conv(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        conv = Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=dilation * (kernel_size - 1) // 2, dilation=dilation, has_bias=True,
                      pad_mode='pad', weight_init=XavierUniform(), bias_init=Normal())
        return conv



    def construct(self, x):
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        results = [getattr(self, self.layer_blocks[-1])(last_inner)]
        for feature, inner_block, layer_block in zip(
                x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]):
            if not inner_block:
                continue
            inner_lateral = getattr(self, inner_block)(feature)
            inner_top_down = P.ResizeNearestNeighbor((int(inner_lateral.shape[-2]), int(inner_lateral.shape[-1])))(last_inner)
            last_inner = inner_lateral + inner_top_down
            results.insert(0, getattr(self, layer_block)(last_inner))

        assert isinstance(self.top_blocks, LastLevelP6P7), 'fpn error'
        last_results = self.top_blocks(x[-1], results[-1])
        print(len(results))
        print(len(last_results))
        results.extend(last_results)

        return tuple(results)


class LastLevelP6P7(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super(LastLevelP6P7, self).__init__()
        self.p6 = nn.Conv2d(in_channels=in_channels,
                       out_channels=out_channels,
                       kernel_size=3,
                       stride=2,
                       pad_mode="pad",
                       padding=1)
        self.p7 = nn.Conv2d(in_channels=in_channels,
                       out_channels=out_channels,
                       kernel_size=3,
                       stride=2,
                       pad_mode="pad",
                       padding=1)
      
        weight_init = initializer(Normal(), self.p6.weight.data.shape)
        self.p6.weight.set_data(weight_init)
        if self.p6.bias is not None:
            bias_init = initializer(Zero(), self.p6.bias.shape)
            self.p6.bias.set_data(bias_init(self.p6.bias.data.shape))
        weight_init = initializer(Normal(), self.p7.weight.data.shape)
        self.p7.weight.set_data(weight_init)
        if self.p7.bias is not None:
            bias_init = initializer(Zero(), self.p7.bias.shape)
            self.p7.bias.set_data(bias_init(self.p7.bias.data.shape))
      
        
        self.use_P5 = in_channels == out_channels

    def construct(self, c5, p5):
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(nn.ReLU()(p6))
        return [p6, p7]
