import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from mindspore.common.parameter import Parameter

from mindspore.nn.optim import SGD
from mindspore.nn.optim import Optimizer

def cat(tensors, dim=0):

    assert isinstance(tensors, (list, tuple))
    return ops.concat(dim)(tensors)

def permute_and_flatten(layer, N, A, C, H, W):
    layer = ops.reshape(layer, (N, -1, C, H, W))
    layer = ops.transpose(layer, (0, 3, 4, 1, 2))
    layer = ops.reshape(layer, (N, -1, C))
    return layer

def concat_fpn_pred(c_pred, box_pred, iou_pred, anchor_cat):
    bs = c_pred[0].shape[0]
    c_all_level, box_all_level = [], []

    for c_per_level, box_per_level in zip(c_pred, box_pred):
        N, AxC, H, W = c_per_level.shape
        Ax4 = box_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        c_per_level = permute_and_flatten(c_per_level, N, A, C, H, W)
        box_per_level = permute_and_flatten(box_per_level, N, A, 4, H, W)

        c_all_level.append(c_per_level)
        box_all_level.append(box_per_level)

    c_flatten = cat(c_all_level, dim=1).reshape(-1, C)
    box_flatten = cat(box_all_level, dim=1).reshape(-1, 4)

    iou_pred_flatten = [ops.transpose(aa, (0, 2, 3, 1)).reshape(bs, -1, 1) for aa in iou_pred]
    iou_pred_flatten = cat(iou_pred_flatten, dim=1).reshape(-1)
    anchor_flatten = ops.tile(anchor_cat.box, (bs, 1))

    return c_flatten, box_flatten, iou_pred_flatten, anchor_flatten


def encode(gt_boxes, anchors):
    TO_REMOVE = 1
    ex_widths = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
    ex_heights = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
    ex_ctr_x = (anchors[:, 2] + anchors[:, 0]) / 2
    ex_ctr_y = (anchors[:, 3] + anchors[:, 1]) / 2

    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0] + TO_REMOVE
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + TO_REMOVE
    gt_ctr_x = (gt_boxes[:, 2] + gt_boxes[:, 0]) / 2
    gt_ctr_y = (gt_boxes[:, 3] + gt_boxes[:, 1]) / 2

    wx, wy, ww, wh = (10., 10., 5., 5.)
    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * ops.log(gt_widths / ex_widths)
    targets_dh = wh * ops.log(gt_heights / ex_heights)

    return ops.stack((targets_dx, targets_dy, targets_dw, targets_dh), 1)

def decode(preds, anchors):
    anchors = ops.cast(anchors, preds.dtype)

    TO_REMOVE = 1
    widths = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
    heights = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
    ctr_x = (anchors[:, 2] + anchors[:, 0]) / 2
    ctr_y = (anchors[:, 3] + anchors[:, 1]) / 2

    wx, wy, ww, wh = (10., 10., 5., 5.)
    dx = preds[:, 0::4] / wx
    dy = preds[:, 1::4] / wy
    dw = preds[:, 2::4] / ww
    dh = preds[:, 3::4] / wh

    dw = ops.clip_by_value(dw, -np.inf, np.log(1000. / 16))
    dh = ops.clip_by_value(dh, -np.inf, np.log(1000. / 16))

    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = ops.exp(dw) * widths[:, None]
    pred_h = ops.exp(dh) * heights[:, None]

    pred_boxes = ops.ZerosLike()(preds)
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * (pred_w - 1)
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * (pred_h - 1)
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * (pred_w - 1)
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * (pred_h - 1)
    
    return pred_boxes

def match(iou_matrix, high_thre, low_thre):
    if iou_matrix.size() == 0:
        if iou_matrix.shape[0] == 0:
            raise ValueError('No ground-truth boxes available for one of the images')
        else:
            raise ValueError('No proposal boxes available for one of the images')

    matched_vals, match_i = ops.argmax(iou_matrix, 0)
    match_i_clone = ops.assign(match_i)

    below_low_thre = matched_vals < low_thre
    between_thre = (matched_vals >= low_thre) & (matched_vals < high_thre)
    match_i[below_low_thre] = -1
    match_i[between_thre] = -2

    max_dt_per_gt, _ = ops.argmax(iou_matrix, 1)
    dt_index_per_gt = ops.nonzero(ops.equal(iou_matrix, max_dt_per_gt[:, None]))

    index_to_update = dt_index_per_gt[:, 1]
    match_i[index_to_update] = match_i_clone[index_to_update]

    return match_i

class ProgressBar:
    def __init__(self, length, max_val):
        self.max_val = max_val
        self.length = length
        self.cur_val = 0

        self.cur_num_bars = -1
        self.update_str()

    def update_str(self):
        num_bars = int(self.length * (self.cur_val / self.max_val))

        if num_bars != self.cur_num_bars:
            self.cur_num_bars = num_bars
            self.string = '█' * num_bars + '░' * (self.length - num_bars)

    def get_bar(self, new_val):
        self.cur_val = new_val

        if self.cur_val > self.max_val:
            self.cur_val = self.max_val
        self.update_str()
        return self.string


class Optimizer:
    def __init__(self, model, cfg):
        self.cfg = cfg
        self.init_lr_group = []
        self.lr_milestones = None
        self.bias_lr_factor = 2
        self.bias_weight_decay = 0
        self.lr_miles = [cfg.warmup_iters] + list(cfg.decay_steps)
    
        params = []
        added_param_names = set()
        for _, cell in model.cells_and_names():
            for name, param in cell.parameters_and_names():
                if not isinstance(param, Parameter) or not param.requires_grad:
                    continue

                lr, weight_decay = cfg.base_lr, cfg.weight_decay

                if "bias" in name:
                    lr *= self.bias_lr_factor
                    weight_decay = self.bias_weight_decay
                if param.name in added_param_names:
                    continue
                added_param_names.add(param.name)  # 添加到已添加的参数集合中
                params.append({"params": [param], "lr": lr, "weight_decay": weight_decay})

        self.optimizer = SGD(params=params, learning_rate=cfg.base_lr, momentum=cfg.momentum)
      

        for param_group in params:
            self.init_lr_group.append(param_group['lr'])

    def update_lr(self, step):
        if (0 <= step < self.lr_miles[0]) and (self.lr_milestones is None):
            for param_group, lr in zip(self.optimizer.param_groups, self.init_lr_group):
                param_group['lr'] = self.cfg.warmup_factor * lr

            self.lr_milestones = 'warmup'

        if (self.lr_miles[0] <= step < self.lr_miles[1]) and (self.lr_milestones == 'warmup'):
            for param_group, lr in zip(self.optimizer.param_groups, self.init_lr_group):
                param_group['lr'] = lr

            self.lr_milestones = 'beginning'

        if (self.lr_miles[1] <= step < self.lr_miles[2]) and (self.lr_milestones == 'beginning'):
            for param_group, lr in zip(self.optimizer.param_groups, self.init_lr_group):
                param_group['lr'] = lr * 0.1

            self.lr_milestones = 'decay_0'

        if (self.lr_miles[2] <= step) and (self.lr_milestones == 'decay_0'):
            for param_group, lr in zip(self.optimizer.param_groups, self.init_lr_group):
                param_group['lr'] = lr * 0.1 * 0.1

            self.lr_milestones = 'decay_1'
            


def build_optimizer(model, cfg):
    params = []
    added_param_names = set()
    for _, cell in model.cells_and_names():
        for name, param in cell.parameters_and_names():
            if not isinstance(param, Parameter) or not param.requires_grad:
                continue
            lr, weight_decay = cfg.base_lr, cfg.weight_decay
            if "bias" in name:
                lr *= 2  # using the hardcoded value here for bias_lr_factor
                weight_decay = 0  # using the hardcoded value here for bias_weight_decay
            if param.name in added_param_names:
                continue
            added_param_names.add(param.name)
            params.append({"params": [param], "lr": lr, "weight_decay": weight_decay})

    optimizer = SGD(params=params, learning_rate=cfg.base_lr, momentum=cfg.momentum)
    return optimizer
