import torch.nn as nn
from torch.autograd import Function


def flatten(x):
    N = list(x.size())[0]
    # print('dim 0', N, 1024*19*37)
    return x.view(N, -1)


def grad_reverse(x, beta=1.0):
    return GradReverse.apply(x, beta)


class GradReverse(Function):
    # def __init__(self, beta):
    #     super(GradReverse, self).__init__()
    #     self.beta = beta

    # def set_beta(self, beta):
    #     self.beta = beta

    @staticmethod
    def forward(ctx, x, beta):
        ctx.beta = beta
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.beta, None


# pool_feat dim: N x 2048, where N may be 300.

class d_cls_inst(nn.Module):
    def __init__(self, beta=1, fc_size=2048):
        super(d_cls_inst, self).__init__()
        self.fc_1_inst = nn.Linear(fc_size, 100)
        self.fc_2_inst = nn.Linear(100, 2)
        self.relu = nn.ReLU(inplace=True)
        self.beta = beta
        # self.softmax = nn.Softmax()
        # self.logsoftmax = nn.LogSoftmax()
        self.bn = nn.BatchNorm1d(2)

    def forward(self, x):
        x = grad_reverse(x, self.beta)
        x = self.relu(self.fc_1_inst(x))
        x = self.relu(self.bn(self.fc_2_inst(x)))
        # y = self.softmax(x)
        # x = self.logsoftmax(x)
        # return x, y
        return x

    def set_beta(self, beta):
        self.beta = beta