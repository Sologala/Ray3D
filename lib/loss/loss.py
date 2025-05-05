# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))


def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape) - 1))


def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1))


def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape

    norm_predicted = torch.mean(torch.sum(predicted**2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target * predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return mpjpe(scale * predicted, target)


def weighted_bonelen_loss(predict_3d_length, gt_3d_length):
    loss_length = 0.001 * torch.pow(predict_3d_length - gt_3d_length, 2).mean()
    return loss_length


def weighted_boneratio_loss(predict_3d_length, gt_3d_length):
    loss_length = 0.1 * torch.pow((predict_3d_length - gt_3d_length) / gt_3d_length, 2).mean()
    return loss_length


def mean_velocity_error(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape

    velocity_predicted = np.diff(predicted, axis=0)
    velocity_target = np.diff(target, axis=0)

    return np.mean(np.linalg.norm(velocity_predicted - velocity_target, axis=len(target.shape) - 1))


class AngleLoss(nn.Module):
    def __init__(self, num_bins=8, alpha=1.0):
        super(AngleLoss, self).__init__()
        self.num_bins = num_bins
        self.alpha = alpha
        self.cls_criterion = nn.CrossEntropyLoss()

    def _compute_bin_and_target(self, gt_angle):
        """
        将真实角度转换为 bin 索引和归一化偏移量
        :param gt_angle: (B,) 张量，表示真实角度，范围 [0, 360)
        :return: bin_indices (B,) 整型张量，reg_targets (B,) 浮点张量
        """
        B = gt_angle.size(0)
        bin_width = 360.0 / self.num_bins
        bin_width_half = bin_width / 2.0

        # 计算 bin 索引
        shifted_angle = (gt_angle + bin_width_half) % 360.0
        bin_indices = (shifted_angle // bin_width).long()
        bin_indices = bin_indices % self.num_bins  # 确保索引在有效范围内
        
        # 确保 bin_indices 是一维张量 [B]
        bin_indices = bin_indices.squeeze()
        if bin_indices.dim() != 1:
            bin_indices = bin_indices.view(-1)

        # 计算 bin 中心
        bin_centers = bin_indices.float() * bin_width

        # 计算最小有符号角度差
        delta_angle = (gt_angle - bin_centers + 180.0) % 360.0 - 180.0

        # 归一化偏移量
        reg_targets = delta_angle / bin_width_half
        
        # 确保 reg_targets 是一维张量 [B]
        reg_targets = reg_targets.squeeze()
        if reg_targets.dim() != 1:
            reg_targets = reg_targets.view(-1)

        return bin_indices, reg_targets

    def forward(self, logits, reg_output, gt_angle):
        """
        :param logits: (B, num_bins) 分类输出
        :param reg_output: (B,) 回归输出
        :param gt_angle: (B,) 真实角度，范围 [0, 360)
        :return: 总损失
        """
        # 确保输入的 gt_angle 是一维张量
        if gt_angle.dim() > 1:
            gt_angle = gt_angle.squeeze()
            if gt_angle.dim() != 1:
                raise ValueError(f"gt_angle should be 1D tensor, got shape {gt_angle.shape}")
        
        # 获取 bin 索引和归一化偏移量
        bin_indices, reg_targets = self._compute_bin_and_target(gt_angle)

        # 验证维度
        assert bin_indices.dim() == 1, f"bin_indices should be 1D, got {bin_indices.dim()}D"
        assert logits.size(0) == bin_indices.size(0), f"Batch size mismatch: {logits.size(0)} vs {bin_indices.size(0)}"
        
        # 分类损失
        loss_cls = self.cls_criterion(logits, bin_indices)

        # 回归损失
        reg_output = reg_output.view(-1)
        reg_targets = reg_targets.detach()  # 不需要梯度
        loss_reg = F.smooth_l1_loss(reg_output, reg_targets)

        # 总损失
        total_loss = loss_cls + self.alpha * loss_reg

        return total_loss