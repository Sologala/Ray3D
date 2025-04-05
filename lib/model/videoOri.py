# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import time
import torch
import torch.nn as nn


class TemporalModelBase(nn.Module):
    """
    Do not instantiate this class.
    """

    def __init__(self, num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels):
        super().__init__()

        # Validate input
        for fw in filter_widths:
            assert fw % 2 != 0, "Only odd filter widths are supported"

        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.num_joints_out = num_joints_out
        self.filter_widths = filter_widths

        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        self.pad = [filter_widths[0] // 2]
        self.expand_bn = nn.BatchNorm2d(channels, momentum=0.1)
        # self.shrink = nn.Conv2d(channels, num_joints_out * 3, 1)
        self.num_classes = 8

        self.coarse_classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, self.num_classes)
        )
        self.fine_regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 + self.num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def cls_and_reg(self, x):
        coarse_logits = self.coarse_classifier(x)
        coarse_probs = torch.softmax(coarse_logits, dim=1)

        # 将粗分类的概率与输入特征拼接
        combined_features = torch.cat([x.view(x.size(0), -1), coarse_probs], dim=1)

        # 细回归
        fine_angle = self.fine_regressor(combined_features)

        # 将细回归的输出映射到具体的角度范围
        class_width = 360 / self.num_classes
        coarse_angle = torch.argmax(coarse_probs, dim=1, keepdim=True).float() * class_width
        final_angle = coarse_angle + fine_angle * (class_width / 2)

        return final_angle

    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.layers_bn:
            bn.momentum = momentum

    def receptive_field(self):
        """
        Return the total receptive field of this model as # of frames.
        """
        frames = 0
        for f in self.pad:
            frames += f
        return 1 + 2 * frames

    def total_causal_shift(self):
        """
        Return the asymmetric offset for sequence padding.
        The returned value is typically 0 if causal convolutions are disabled,
        otherwise it is half the receptive field.
        """
        frames = self.causal_shift[0]
        next_dilation = self.filter_widths[0]
        for i in range(1, len(self.filter_widths)):
            frames += self.causal_shift[i] * next_dilation
            next_dilation *= self.filter_widths[i]
        return frames

    def forward(self, x):
        print(x.shape)
        print(self.in_features)
        print(self.num_joints_in)

        assert len(x.shape) == 4
        assert x.shape[-2] == self.num_joints_in
        assert x.shape[-1] == self.in_features

        # 调整输入形状以匹配卷积层期望的通道数
        sz = x.shape[:3]
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.size(0), -1, x.size(2))
        x = x.unsqueeze(-1)

        x = self._forward_blocks(x)
        ang = self.cls_and_reg(x)
        return ang


class TemporalModel(TemporalModelBase):
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.
    """

    def __init__(
        self,
        num_joints_in,
        in_features,
        num_joints_out,
        filter_widths,
        causal=False,
        dropout=0.25,
        channels=1024,
        dense=False,
    ):
        """
        Initialize this model.

        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        dense -- use regular dense convolutions instead of dilated convolutions (ablation experiment)
        """
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels)

        self.expand_conv = nn.Conv2d(num_joints_in * in_features, channels, (filter_widths[0], 1), bias=False)

        layers_conv = []
        layers_bn = []

        self.causal_shift = [(filter_widths[0]) // 2 if causal else 0]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
            self.causal_shift.append((filter_widths[i] // 2 * next_dilation) if causal else 0)

            layers_conv.append(
                nn.Conv2d(
                    channels,
                    channels,
                    (filter_widths[i], 1) if not dense else (2 * self.pad[-1] + 1, 1),
                    dilation=(next_dilation, 1) if not dense else (1, 1),
                    bias=False,
                )
            )
            layers_bn.append(nn.BatchNorm2d(channels, momentum=0.1))
            layers_conv.append(nn.Conv2d(channels, channels, (1, 1), dilation=(1, 1), bias=False))
            layers_bn.append(nn.BatchNorm2d(channels, momentum=0.1))

            next_dilation *= filter_widths[i]

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)

    def _forward_blocks(self, x):
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))

        for i in range(len(self.pad) - 1):
            pad = self.pad[i + 1]
            shift = self.causal_shift[i + 1]
            res = x[:, :, pad + shift : x.shape[2] - pad + shift, :]

            x = self.drop(self.relu(self.layers_bn[2 * i](self.layers_conv[2 * i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2 * i + 1](self.layers_conv[2 * i + 1](x))))

        return x


class TemporalModelOptimized1f(TemporalModelBase):
    """
    3D pose estimation model optimized for single-frame batching, i.e.
    where batches have input length = receptive field, and output length = 1.
    This scenario is only used for training when stride == 1.

    This implementation replaces dilated convolutions with strided convolutions
    to avoid generating unused intermediate results. The weights are interchangeable
    with the reference implementation.
    """

    def __init__(
        self, num_joints_in, in_features, num_joints_out, filter_widths, causal=False, dropout=0.25, channels=1024
    ):
        """
        Initialize this model.

        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        """
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels)

        self.expand_conv = nn.Conv2d(
            num_joints_in * in_features, channels, (filter_widths[0], 1), stride=(filter_widths[0], 1), bias=False
        )

        layers_conv = []
        layers_bn = []

        self.causal_shift = [(filter_widths[0] // 2) if causal else 0]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
            self.causal_shift.append((filter_widths[i] // 2) if causal else 0)

            layers_conv.append(
                nn.Conv2d(channels, channels, (filter_widths[i], 1), stride=(filter_widths[i], 1), bias=False)
            )
            layers_bn.append(nn.BatchNorm2d(channels, momentum=0.1))
            layers_conv.append(nn.Conv2d(channels, channels, (1, 1), dilation=(1, 1), bias=False))
            layers_bn.append(nn.BatchNorm2d(channels, momentum=0.1))
            next_dilation *= filter_widths[i]

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)

    def _forward_blocks(self, x):

        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))

        for i in range(len(self.pad) - 1):
            res = x[:, :, self.causal_shift[i + 1] + self.filter_widths[i + 1] // 2 :: self.filter_widths[i + 1], :]

            x = self.drop(self.relu(self.layers_bn[2 * i](self.layers_conv[2 * i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2 * i + 1](self.layers_conv[2 * i + 1](x))))

        return x


if __name__ == "__main__":
    start = time.time()
    receptive_field = 81
    num_joints = 17
    filter_widths = [3, 3, 3, 3]

    model = TemporalModel(17, 2, num_joints, filter_widths=filter_widths)

    input = torch.randn([512, receptive_field, 17, 2])
    y = model(input)
    end = time.time()
    print(end - start)
    print(y.shape)
