import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self,mode='mix'):
        """Choices: ['ce' or 'focal' or ' or 'Dice''mix]"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'mix':
            print('DiceLoss+FocalLoss ！')
            print('Calling Attention to Modify High and Low Dimensional Features！')
            return self.MixLoss
        elif mode == 'Dice':
            print('DiceLoss！')
            print('Calling Attention to Modify High and Low Dimensional Features！')
            return self.MixLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

    def MixLoss(self, logit, target, gamma=2, alpha=0.5):

        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss_1 = -((1 - pt) ** gamma) * logpt
        ################################################################################################################
        # Apply softmax to the model's output to get probabilities
        input_probs = F.softmax(logit, dim=1)

        # Flatten the predicted and target tensors
        input_flat = input_probs[:, 1].reshape(-1) # Assuming binary classification
        target_flat = target.reshape(-1)

        # Calculate intersection and union
        intersection = torch.sum(input_flat * target_flat)
        union = torch.sum(input_flat) + torch.sum(target_flat)

        # Calculate Dice coefficient
        dice_coeff = (2.0 * intersection + 1e-5) / (union + 1e-5)

        # Calculate Dice loss (complement of Dice coefficient)
        loss_2 = 1.0 - dice_coeff
        ################################################################################################################
        w1 = 0.25
        w2 = 1 - w1
        loss_mix=loss_1*w1+loss_2*w2
        ################################################################################################################
        if self.batch_average:
            loss_mix /= n

        return loss_mix
    def DiceLoss(self, logit, target, gamma=2, alpha=0.5):
        ################################################################################################################
        # Apply softmax to the model's output to get probabilities
        n, c, h, w = logit.size()
        input_probs = F.softmax(logit, dim=1)

        # Flatten the predicted and target tensors
        input_flat = input_probs[:, 1].reshape(-1) # Assuming binary classification
        target_flat = target.reshape(-1)

        # Calculate intersection and union
        intersection = torch.sum(input_flat * target_flat)
        union = torch.sum(input_flat) + torch.sum(target_flat)

        # Calculate Dice coefficient
        dice_coeff = (2.0 * intersection + 1e-5) / (union + 1e-5)

        # Calculate Dice loss (complement of Dice coefficient)
        loss = 1.0 - dice_coeff
        ################################################################################################################
        if self.batch_average:
            loss /= n

        return loss

if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 4, 7, 7).cuda()
    b = torch.rand(1, 4, 7).cuda()
    print(loss.DiceLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




