import os,sys
sys.path.append('.')
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


def activation(act_type, inplace=False, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU()
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


class WIRW1(nn.Module):
    # WIRW module (SA input n_feats = n_feats)
    """
    SRBW1 --> WIRW1
    Code for Wide Identical Residual Weighting (WIRW) unit
    """

    def __init__(
            self, n_feats, wn=lambda x: torch.nn.utils.weight_norm(x), act=nn.ReLU(True), groups_sa=8):
        super(WIRW1, self).__init__()
        self.res_scale = Scale(1)
        self.x_scale = Scale(1)

        # #additional by Jinni
        self.SAlayer = sa_layer(n_feats, groups=groups_sa)

    def forward(self, x):
        x_sa = self.SAlayer(x)
        y = self.res_scale(x_sa)  + self.x_scale(x)
        return y
class sa_layer(nn.Module):
    """Constructs a Channel Spatial Group module.
    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, n_feats, groups=6):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, n_feats // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, n_feats // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, n_feats // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, n_feats // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(n_feats // (2 * groups), n_feats // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape
        # print("debug here", x.shape)
        # print("print all", b, c, h, w)

        x = x.reshape(b * self.groups, -1, h, w)
        # print(x.shape)
        x_0, x_1 = x.chunk(2, dim=1)
        # print("x_0.shape", x_0.shape)
        # print("x_1.shape", x_1.shape)

        # channel attention
        # print("self.cbias.shape", self.cbias.shape)
        # print("self.cweight.shape", self.cweight.shape)
        xn = self.avg_pool(x_0)
        # print("xn.shape", xn.shape)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        # print("xs.shape", xs.shape)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out


if __name__ == "__main__":
    import torch
    batch_size = 64
    width = 7
    height = 7
    c = 2048

    X = torch.rand((batch_size, c, width, height))
    print(X.shape)

    wirw1 = WIRW1(n_feats=2048)

    x_out = wirw1(X)
    print(x_out.shape)
    
