from model.layers import Conv2d, DFConv2d, FrozenBatchNorm2d
import numpy as np
from mindspore import nn
import mindspore
from collections import namedtuple


StageSpec = namedtuple("StageSpec", ["index",  # Index of the stage, eg 1, 2, ..,. 5
                                     "block_count",  # Number of residual blocks in the stage
                                     "return_features",  # True => return the last feature map from this stage
                                     ])

res50_fpn = tuple(StageSpec(index=i, block_count=c, return_features=r)
                  for (i, c, r) in ((1, 3, True), (2, 4, True), (3, 6, True), (4, 3, True)))
res101_fpn = tuple(StageSpec(index=i, block_count=c, return_features=r)
                   for (i, c, r) in ((1, 3, True), (2, 4, True), (3, 23, True), (4, 3, True)))


class ResNet(nn.Cell):
    def __init__(self, cfg):
        super(ResNet, self).__init__()
        backbone_dict = {'res50': res50_fpn, 'res101': res101_fpn}

        self.stem = BaseStem()

        num_groups = 1
        width_per_group = 64
        in_channels = 64
        stage2_bottleneck_channels = num_groups * width_per_group
        stage2_out_channels = 256
        self.stages = []
        self.return_features = {}
        stage_specs = backbone_dict[cfg.backbone]

        for stage in stage_specs:
            name = "layer" + str(stage.index)
            stage2_relative_factor = 2 ** (stage.index - 1)
            bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
            out_channels = stage2_out_channels * stage2_relative_factor
            stage_with_dcn = cfg.stage_with_dcn[stage.index - 1]
            blocks = []
            stride = int(stage.index > 1) + 1

            for _ in range(stage.block_count):
                blocks.append(Bottleneck(in_channels,
                                         bottleneck_channels,
                                         out_channels,
                                         num_groups,
                                         stride,
                                         with_dcn=stage_with_dcn))
                stride = 1
                in_channels = out_channels

            module = nn.SequentialCell(blocks)
            in_channels = out_channels
            setattr(self, name, module)
            self.stages.append(name)
            self.return_features[name] = stage.return_features


    def construct(self, x):
        outputs = []
        x = self.stem(x)

        for stage_name in self.stages:
            x = getattr(self, stage_name)(x)

            if self.return_features[stage_name]:
                outputs.append(x)

        return tuple(outputs)


class Bottleneck(nn.Cell):
    def __init__(self, in_channels, bottleneck_channels, out_channels, num_groups=1, stride=1,
                 stride_in_1x1=True, dilation=1, with_dcn=False):
        super().__init__()
        self.downsample = None

        if in_channels != out_channels:
            down_stride = stride if dilation == 1 else 1
            self.downsample = nn.SequentialCell([Conv2d(in_channels, out_channels, kernel_size=1,
                                                        stride=down_stride, pad_mode='valid', has_bias=False),
                                                 FrozenBatchNorm2d(out_channels)])

        if dilation > 1:
            stride = 1  # reset to be 1

        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=stride_1x1, pad_mode='valid', has_bias=False)
        self.bn1 = FrozenBatchNorm2d(bottleneck_channels)

        if with_dcn:
            self.conv2 = DFConv2d(bottleneck_channels, bottleneck_channels, with_modulated_dcn=True,
                                  kernel_size=3, stride=stride_3x3, groups=num_groups, dilation=dilation,
                                  deformable_groups=1, pad_mode='same', has_bias=False)
        else:
           self.conv2 = Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride_3x3,
                    padding=0, group=num_groups, dilation=dilation, pad_mode='same', has_bias=False)


        self.bn2 = FrozenBatchNorm2d(bottleneck_channels)
        self.conv3 = Conv2d(bottleneck_channels, out_channels, kernel_size=1, pad_mode='valid', has_bias=False)
        self.bn3 = FrozenBatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out



class BaseStem(nn.Cell):
    def __init__(self):
        super(BaseStem, self).__init__()
        # 7x7 stem is used in all models

        self.conv1 = Conv2d(3, 64, kernel_size=7, stride=2, padding=3, pad_mode='pad', has_bias=False)

        self.bn1 = FrozenBatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        
        zero_tensor = mindspore.Tensor(np.zeros(self.conv1.weight.shape), dtype=mindspore.float32)
        self.conv1.weight = zero_tensor


    
    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x
