import numpy as np
from mindspore import nn, Tensor
from mindspore.ops import operations as ops
from utils.utils import concat_fpn_pred, encode, decode, match
from utils.box_list import boxlist_iou, cat_boxlist
import copy
from mindspore import ops
from mindspore.common import dtype as mstype
from sklearn.mixture import GaussianMixture
from model.sigmoid_focal_loss import SigmoidFocalLossFunction


class PAALoss(nn.Cell):
    def __init__(self, cfg):
        super(PAALoss, self).__init__()
        self.cfg = cfg
        self.anchors = None
        self.iou_bce_loss = nn.BCEWithLogitsLoss(reduction="sum")
        
    @staticmethod
    def GIoULoss(self, pred, target, anchor, weight=None):
        # MindSpore 中并不直接支持张量切片, 所以这里我们使用StridedSlice
        pred_l = ops.StridedSlice()(pred, (0, 0), (pred.shape[0], 2), (1, 1))
        pred_r = ops.StridedSlice()(pred, (0, 2), (pred.shape[0], 4), (1, 1))
        
        target_l = ops.StridedSlice()(target, (0, 0), (target.shape[0], 2), (1, 1))
        target_r = ops.StridedSlice()(target, (0, 2), (target.shape[0], 4), (1, 1))
        
        anchor_l = ops.StridedSlice()(anchor, (0, 0), (anchor.shape[0], 2), (1, 1))
        anchor_r = ops.StridedSlice()(anchor, (0, 2), (anchor.shape[0], 4), (1, 1))
        
        pred_area = (pred_r[..., 0] - pred_l[..., 0]) * (pred_r[..., 1] - pred_l[..., 1])
        target_area = (target_r[..., 0] - target_l[..., 0]) * (target_r[..., 1] - target_l[..., 1])
        # 等价于 torch.min 操作
        inter_l = ops.Maximum()(pred_l, target_l)
        inter_r = ops.Minimum()(pred_r, target_r)
        # MindSpore 的 clamp 操作需要指定 min 和 max
        inter_area = ops.Maximum()(0.0, inter_r[..., 0] - inter_l[..., 0]) * ops.Maximum()(0.0, inter_r[..., 1] - inter_l[..., 1])
        
        enclose_l = ops.Minimum()(pred_l, target_l)
        enclose_r = ops.Maximum()(pred_r, target_r)
        enclose_area = (enclose_r[..., 0] - enclose_l[..., 0]) * (enclose_r[..., 1] - enclose_l[..., 1])
        
        # 以下替代 torch.log 和其他基础操作
        GIoU = inter_area / (pred_area + target_area - inter_area) - (enclose_area - (pred_area + target_area - inter_area)) / enclose_area
        iou = inter_area / (pred_area + target_area - inter_area)
        
        # 这里我们可以用 ops.TensorAdd 和 ops.TensorSub 等进行操作，但为了简化，我们直接用常规操作
        
        loss = 1 - iou + GIoU
        if weight is not None:
            loss = loss * weight
        return loss

    def initial_preparation(self, box_list_batch):
        category_all, offset_all, index_all = [], [], []

        for box_list in box_list_batch:
    
            if box_list.mode != 'x1y1x2y2':
                raise ValueError("The mode should be 'x1y1x2y2'")

            iou_matrix = boxlist_iou(box_list, self.anchor_cat)  # You might need to convert or adjust this function.
            matched_index = match(iou_matrix, self.cfg.match_iou_thre, self.cfg.match_iou_thre)

            box_list = Tensor(box_list.asnumpy().copy(), box_list.dtype)

            box_list_matched = ops.Gather()(box_list, ops.Clamp()(matched_index, 0, ops.Shape()(box_list)[0]-1))

            offset = encode(box_list_matched.box, self.anchor_cat.box)

            category = box_list_matched.label

            category = ops.AssignAdd()(category, ops.Where()(ops.Equal()(matched_index, -1), -category, 0))
            category = ops.AssignAdd()(category, ops.Where()(ops.Equal()(matched_index, -2), 1, 0))

            offset_all.append(offset)
            category_all.append(category)
            index_all.append(matched_index)

        category_all = ops.Concat(0)(category_all)
        category_all = ops.Cast()(category_all, mstype.int32)  # equivalent of '.int()'
        offset_all = ops.Concat(0)(offset_all)
        index_all = ops.Concat(0)(index_all)

        return category_all, offset_all, index_all


    def compute_paa(self, box_list_batch, c_init_batch, score_batch, index_init_batch):
        bs = len(box_list_batch)
        c_init_batch = c_init_batch.reshape(bs, -1)
        score_batch = score_batch.reshape(bs, -1)
        index_init_batch = index_init_batch.reshape(bs, -1)
        num_anchor_per_fpn = [len(anchor_per_fpn.box) for anchor_per_fpn in self.anchors]

        final_c_batch, final_offset_batch = [], []
        for i in range(len(box_list_batch)):
            box_list = box_list_batch[i]
            if box_list.mode != "x1y1x2y2":
                raise ValueError("The mode should be 'x1y1x2y2'")

            c_gt, box_gt = box_list.label, box_list.box

            c_init = c_init_batch[i]
            score = score_batch[i]
            index_init = index_init_batch[i]
            if c_init.shape != index_init.shape:
                raise ValueError("Shapes of c_init and index_init should be the same")

            final_c = ops.fill(mstype.int32, self.anchor_cat.box.shape, 0)
            final_box_gt = ops.fill(mstype.float32, self.anchor_cat.box.shape, 0.0)

            for gt_i in range(box_gt.shape[0]):
                candi_i_per_gt = []
                start_i = 0

                for j in range(len(num_anchor_per_fpn)):
                    end_i = start_i + num_anchor_per_fpn[j]
                    score_per_fpn = score[start_i:end_i]
                    index_init_per_fpn = index_init[start_i:end_i]

                    matched_i = ops.where(ops.equal(index_init_per_fpn, gt_i))
                    matched_num = matched_i.shape[0]

                    if matched_num > 0:
                        topk_values, topk_i = ops.top_k(score_per_fpn[matched_i], min(matched_num, self.cfg.fpn_topk))
                        topk_i_per_fpn = matched_i[topk_i]
                        candi_i_per_gt.append(ops.concat(0, [topk_i_per_fpn, start_i]))

                    start_i = end_i

                if candi_i_per_gt:
                    candi_i_per_gt = ops.concat(0, candi_i_per_gt)

                    candi_score = score[candi_i_per_gt]
                    sorted_values, sorted_indices = ops.sort(candi_score)
                    sorted_values_np = sorted_values.asnumpy().reshape(-1, 1)

                    gmm = GaussianMixture(n_components=2, 
                                        weights_init=[0.5, 0.5],
                                        means_init=[[sorted_values_np.min()], [sorted_values_np.max()]],
                                        precisions_init=[[[1.0]], [[1.0]]])
                    gmm.fit(sorted_values_np)

                    gmm_component = gmm.predict(sorted_values_np)
                    gmm_score = gmm.score_samples(sorted_values_np)

                    gmm_component_tensor = Tensor(gmm_component, mstype.int32)
                    gmm_score_tensor = Tensor(gmm_score, mstype.float32)

                    fg = ops.equal(gmm_component_tensor, 0)
                    if fg.any():
                        _, fg_max_i = ops.reduce_max(gmm_score_tensor[fg])
                        is_pos = sorted_indices[:fg_max_i + 1]
                    else:
                        is_pos = sorted_indices

                    pos_i = candi_i_per_gt[is_pos]
                    final_c[pos_i] = ops.reshape(c_gt[gt_i], [-1, 1])
                    final_box_gt[pos_i] = ops.reshape(box_gt[gt_i], [-1, 4])

            final_offset = encode(final_box_gt, self.anchor_cat.box)

            final_c_batch.append(final_c)
            final_offset_batch.append(final_offset)

        final_c_batch = ops.concat(0, final_c_batch).astype(mstype.int32)
        final_offset_batch = ops.concat(0, final_offset_batch)

        return final_c_batch, final_offset_batch

    @staticmethod
    def compute_ious(boxes1, boxes2):
        area1 = (boxes1[:, 2] - boxes1[:, 0] + 1) * (boxes1[:, 3] - boxes1[:, 1] + 1)
        area2 = (boxes2[:, 2] - boxes2[:, 0] + 1) * (boxes2[:, 3] - boxes2[:, 1] + 1)
        
        max_op = ops.Maximum()
        min_op = ops.Minimum()
        sub_op = ops.Sub()
        add_op = ops.Add()
        mul_op = ops.Mul()
        
        lt = max_op(boxes1[:, :2], boxes2[:, :2])
        rb = min_op(boxes1[:, 2:], boxes2[:, 2:])
        wh = sub_op(rb, lt) + 1
        wh_clamped = ops.ClipByValue()(wh, 0, float('inf'))
        inter = mul_op(wh_clamped[:, 0], wh_clamped[:, 1])
        denom = add_op(area1 + area2, sub_op(inter, mul_op(wh[:, 0], wh[:, 1])))

        return inter / denom

    def __call__(self, c_pred, box_pred, iou_pred, box_list_batch):
        # c_init_batch: (bs * num_anchor,), 0 for background, -1 for ignored
        # offset_init_batch: (bs * num_anchor, 4)
        # index_init_batch: (bs * num_anchor,), -1 for background, -2 for ignored
        self.anchor_cat = cat_boxlist(self.anchors)

        c_init_batch, offset_init_batch, index_init_batch = self.initial_preparation(box_list_batch)
        pos_i_init = ops.nonzero(c_init_batch > 0).reshape(-1)

        c_pred_f, box_pred_f, iou_pred_f, anchor_f = concat_fpn_pred(c_pred, box_pred, iou_pred, self.anchor_cat)

        if pos_i_init.size > 0:
            sigmoid_focal_loss = SigmoidFocalLossFunction(self.cfg.fl_gamma, self.cfg.fl_alpha)
            c_loss = sigmoid_focal_loss(c_pred_f, c_init_batch)
            box_loss = self.GIoULoss(box_pred_f, offset_init_batch, anchor_f, weight=None)
            box_loss = box_loss[c_init_batch > 0].reshape(-1)

            box_loss_full = np.full((c_loss.shape[0],), fill_value=10000, dtype=np.float32)
            assert box_loss.max() < 10000, 'box_loss_full initial value error'
            box_loss_full[pos_i_init] = box_loss

            sum_op = ops.ReduceSum()
            score_batch = sum_op(c_loss, axis=1) + box_loss_full
            assert not np.isnan(score_batch).any()  # all the elements should not be nan

            # compute labels and targets using PAA
            final_c_batch, final_offset_batch = self.compute_paa(box_list_batch, c_init_batch, score_batch,
                                                                 index_init_batch)

            pos_i_final = ops.nonzero(final_c_batch > 0).reshape(-1)
            num_pos = pos_i_final.size

            box_pred_f = box_pred_f[pos_i_final]
            final_offset_batch = final_offset_batch[pos_i_final]
            anchor_f = anchor_f[pos_i_final]
            iou_pred_f = iou_pred_f[pos_i_final]

            gt_boxes = decode(final_offset_batch, anchor_f)
            box_pred_decoded = decode(box_pred_f, anchor_f).detach()
            iou_gt = self.compute_ious(gt_boxes, box_pred_decoded)
            
            cls_loss = sigmoid_focal_loss(c_pred_f, final_c_batch.astype(np.int32))
            box_loss = self.GIoULoss(box_pred_f, final_offset_batch, anchor_f, weight=iou_gt)
            box_loss = box_loss[final_c_batch[pos_i_final] > 0].reshape(-1)
            iou_pred_loss = self.iou_bce_loss(iou_pred_f, iou_gt)
            iou_gt_sum = np.sum(iou_gt).item()
        else:
            box_loss = box_pred_f.sum()

        sum_op = ops.ReduceSum()
        category_loss = sum_op(cls_loss) / num_pos
        box_loss = sum_op(box_loss) / iou_gt_sum * self.cfg.box_loss_w
        iou_pred_loss = iou_pred_loss / num_pos * self.cfg.iou_loss_w

        return category_loss, box_loss, iou_pred_loss
