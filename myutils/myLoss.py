import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None, reduce=True, size_average=True):
        super().__init__()
        weight = torch.Tensor(weight)

        self.loss = nn.NLLLoss(weight, reduce=reduce, size_average=size_average)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs, dim=1), targets)


class JensenShannonDivergence(nn.Module):

    def __init__(self, reduce=True, size_average=False):
        super().__init__()

        self.loss = nn.KLDivLoss(reduce=reduce, size_average=size_average)

    def forward(self, ensemble_probs):
        n, c, h, w = ensemble_probs.shape  # number of distributions
        mixture_dist = ensemble_probs.mean(0, keepdim=True).expand(n, c, h, w)
        return self.loss(torch.log(ensemble_probs), mixture_dist) / (h * w)
