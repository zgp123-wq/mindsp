
import mindspore.nn as nn
import mindspore
from mindspore import Parameter
from mindspore.common.initializer import Normal, Constant, Zero
from mindspore.common.initializer import initializer
from mindspore import Parameter
import numpy as np
from collections import OrderedDict
from model import fpn as fpn_module
from .fpn2 import resnet50_fpn
from model import resnet
from model.loss import PAALoss
from utils.anchor_generator import AnchorGenerator
from model.layers import DFConv2d  # Make sure this is correctly implemented in MindSpore

class Scale(nn.Cell):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = Parameter(initializer('normal', [1], dtype=mindspore.float32))

    def construct(self, input):
        return input * self.scale

class PAAHead(nn.Cell):
    def __init__(self, cfg):
        super(PAAHead, self).__init__()
        self.cfg = cfg
        num_classes = cfg.num_classes - 1
        num_anchors = len(cfg.aspect_ratios)

        cls_tower, bbox_tower = [], []
        for i in range(4):
            if cfg.dcn_tower and i == 3:
                conv_func = DFConv2d  # Make sure this is correctly implemented in MindSpore
            else:
                conv_func = nn.Conv2d

             # For cls_tower
            cls_tower.append(conv_func(256, 256, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True))
            cls_tower.append(nn.GroupNorm(num_groups=32, num_channels=256, affine=True))
            cls_tower.append(nn.ReLU())

            # For bbox_tower
            bbox_tower.append(conv_func(256, 256, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True))
            bbox_tower.append(nn.GroupNorm(num_groups=32, num_channels=256, affine=True))
            bbox_tower.append(nn.ReLU())

        self.cls_tower = nn.SequentialCell(cls_tower)
        self.bbox_tower = nn.SequentialCell(bbox_tower)

        self.cls_logits = nn.Conv2d(256, num_anchors * num_classes, kernel_size=3, stride=1, pad_mode='pad', padding=1)
        self.bbox_pred = nn.Conv2d(256, num_anchors * 4, kernel_size=3, stride=1, pad_mode='pad', padding=1)
        self.iou_pred = nn.Conv2d(256, num_anchors * 1, kernel_size=3, stride=1, pad_mode='pad', padding=1)

        self.scales = nn.CellList([Scale(init_value=1.0) for _ in range(5)])

        self.cls_tower = nn.SequentialCell(cls_tower)
        self.bbox_tower = nn.SequentialCell(bbox_tower)

        self.cls_logits = nn.Conv2d(256, num_anchors * num_classes, kernel_size=3, stride=1, pad_mode='pad', padding=1)
        self.bbox_pred = nn.Conv2d(256, num_anchors * 4, kernel_size=3, stride=1, pad_mode='pad', padding=1)
        self.iou_pred = nn.Conv2d(256, num_anchors * 1, kernel_size=3, stride=1, pad_mode='pad', padding=1)
        self.scales = nn.CellList([Scale(init_value=1.0) for _ in range(5)])

        all_modules = [self.cls_tower, self.bbox_tower, self.cls_logits, self.bbox_pred, self.iou_pred]
        
        for module in [self.cls_tower, self.bbox_tower, self.cls_logits, self.bbox_pred, self.iou_pred]:
            for m in module.cells():
                if isinstance(m, nn.Conv2d):
                    weight_init = initializer(Normal(), m.weight.shape)
                    bias_init = initializer(Zero(), m.bias.shape)
                    m.weight.set_data(weight_init)
                    m.bias.set_data(bias_init)
                    
        # Initialize cls_logits bias with Focal Loss initializer
        prior_prob = 0.01
        bias_value = -np.log((1 - prior_prob) / prior_prob)
        if self.cls_logits.bias is not None:  # Check if bias is not None
            cls_logits_bias_init = initializer(Constant(bias_value), self.cls_logits.bias.shape)
            self.cls_logits.bias.set_data(cls_logits_bias_init)
        
        
    def construct(self, x):
        logits, bbox_reg, iou_pred = [], [], []

        for i, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)

            logits.append(self.cls_logits(cls_tower))
            bbox_reg.append(self.scales[i](self.bbox_pred(box_tower)))
            iou_pred.append(self.iou_pred(box_tower))

        return logits, bbox_reg, iou_pred

class PAA(nn.Cell):
    def __init__(self, cfg):
        super(PAA, self).__init__()
        # body = resnet.ResNet(cfg)
        # fpn = fpn_module.FPN(in_channels_list=[0, 512, 1024, 2048], out_channels=256)
        # self.backbone = nn.SequentialCell(OrderedDict([("body", body), ("fpn", fpn)]))
        self.feature_extractor = resnet50_fpn()
        self.head = PAAHead(cfg)
        self.paa_loss = PAALoss(cfg)  # Make sure this is correctly implemented in MindSpore
        self.anchor_generator = AnchorGenerator(cfg)  # Make sure this is correctly implemented in MindSpore

    def construct(self, img_tensor_batch, box_list_batch=None):
        features = self.feature_extractor(img_tensor_batch)
        c_pred, box_pred, iou_pred = self.head(features)
        anchors = self.anchor_generator(features)
        self.paa_loss.anchors = anchors

        if self.training:  # Make sure the training flag is set correctly in MindSpore
            return self.paa_loss(c_pred, box_pred, iou_pred, box_list_batch)
        else:
            return c_pred, box_pred, iou_pred, anchors
