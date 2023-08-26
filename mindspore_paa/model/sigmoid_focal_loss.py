import mindspore.nn as nn
import mindspore.ops as ops


class SigmoidFocalLoss(nn.Cell):
    def __init__(self, gamma, alpha):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.sigmoid = nn.Sigmoid()
        self.reduce_sum = ops.ReduceSum()

    def construct(self, logits, targets):
        num_classes = logits.shape[1]
        dtype = targets.dtype
        device = targets.device

        class_range = ops.arange(1, num_classes + 1, dtype=dtype, device=device).unsqueeze(0)
        t = ops.unsqueeze(targets, 1)
        p = self.sigmoid(logits)
        
        term1 = (1 - p) ** self.gamma * ops.log(p)
        term2 = p ** self.gamma * ops.log(1 - p)
        
        is_class_range = ops.equal(t, class_range)
        is_not_class_range = ops.logical_not(is_class_range) & ops.greater_equal(t, 0)
        
        loss = -is_class_range * term1 * self.alpha - is_not_class_range * term2 * (1 - self.alpha)
        loss = self.reduce_sum(loss)
        return loss

class SigmoidFocalLossFunction(SigmoidFocalLoss):
    def __init__(self, gamma, alpha):
        super(SigmoidFocalLossFunction, self).__init__(gamma, alpha)
    
    def construct(self, logits, targets):
        return super(SigmoidFocalLossFunction, self).construct(logits, targets)

