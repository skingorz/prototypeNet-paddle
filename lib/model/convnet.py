# import torch.nn as nn
from os import strerror
import paddle
import paddle.nn as nn


def conv_block(in_channels, out_channels):
    # weight_attr = paddle.framework.ParamAttr(initializer=paddle.nn.initializer.Uniform())
    # weight_attr = paddle.framework.ParamAttr(initializer=paddle.nn.initializer.Uniform(low=-0.5, high=0.5))
    weight_attr = paddle.framework.ParamAttr(initializer=nn.initializer.KaimingUniform())
    bn = nn.BatchNorm2D(out_channels, weight_attr=weight_attr)
    # nn.initializer.Uniform()
    return nn.Sequential(
        nn.Conv2D(in_channels, out_channels, 3, padding=1, data_format="NCHW",),
        bn,
        nn.ReLU(),
        nn.MaxPool2D(2)
    )


class Convnet(nn.Layer):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        self.out_channels = 1600

    def forward(self, x):
        x = self.encoder(x)
        return x.reshape([x.shape[0], -1])
        # return x.view(x.size(0), -1)

