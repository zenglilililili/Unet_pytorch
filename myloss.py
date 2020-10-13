import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputx, targetx):
        smooth = 0.00001

        input_flat = inputx.flatten()

        target_flat = targetx.flatten()

        intersection = input_flat * target_flat

        loss = (2.0 * intersection.sum() + smooth) / (input_flat.sum() + target_flat.sum() + smooth)

        loss = 1.0 - loss.sum()
        return loss


class softmax_mse_loss(nn.Module):
    def __init__(self):
        super(softmax_mse_loss, self).__init__()

    def forward(self, input_logits, target_logits):

        assert input_logits.size() == target_logits.size()
        input_flat = input_logits.flatten()
        target_flat = target_logits.flatten()
        mse_loss = (input_flat - target_flat) ** 2
        return mse_loss
