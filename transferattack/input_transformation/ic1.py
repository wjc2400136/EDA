import torch
import numpy as np
import torchvision.transforms as T
from torch.nn import functional as F
import math

from ..utils import *
from ..gradient.mifgsm import MIFGSM


def s_component(x, s_range):
    """
    Spatial random rescale-shift-pad (S-Component)
    Simulates multi-position activation via random translation and scaling
    Supports both down-scaling and up-scaling

    :param x: input tensor (N, C, H, W)
    :param s_range: scaling range parameter (scale will be between 1/s_range and s_range)
    :return: spatially transformed tensor
    """
    batch_size, channels, height, width = x.shape

    # Generate random scaling factor using the reference pattern:
    # r_sample = ((r - 1/r) * torch.rand(1) + 1/r)
    # This generates a value in [1/r, r] uniformly
    scale_factor = ((s_range - 1.0 / s_range) * torch.rand(1, device=x.device) + 1.0 / s_range).item()

    if scale_factor < 1.0:
        # Down-scaling: resize down then reflect-pad
        new_h, new_w = int(math.floor(height * scale_factor)), int(math.floor(width * scale_factor))

        # Resize down using functional API (preserves gradient)
        x_resized = F.interpolate(x, size=(new_h, new_w), mode='bilinear',
                                  align_corners=False, antialias=True)

        # Random shift position
        w_rem = width - new_w
        h_rem = height - new_h
        pad_top = torch.randint(0, h_rem + 1, (1,), device=x.device).item()
        pad_left = torch.randint(0, w_rem + 1, (1,), device=x.device).item()
        pad_bottom = h_rem - pad_top
        pad_right = w_rem - pad_left

        # Reflect padding using functional API (as requested by user)
        x_padded = F.pad(x_resized, [pad_left, pad_right, pad_top, pad_bottom],
                         mode='reflect')

        return x_padded
    else:
        # Up-scaling: resize up then random crop
        new_h, new_w = int(math.floor(height * scale_factor)), int(math.floor(width * scale_factor))

        # Resize up using functional API (preserves gradient)
        x_resized = F.interpolate(x, size=(new_h, new_w), mode='bilinear',
                                  align_corners=False, antialias=True)

        # Random crop position
        crop_h = new_h - height
        crop_w = new_w - width
        crop_top = torch.randint(0, crop_h + 1, (1,), device=x.device).item()
        crop_left = torch.randint(0, crop_w + 1, (1,), device=x.device).item()

        # Crop back to original size
        x_cropped = x_resized[:, :, crop_top:crop_top + height, crop_left:crop_left + width]

        return x_cropped


import torch


def c_component_random_neighbor(x, w_b, k=4):
    """
    跨像素块融合（C组件）——随机8邻域版本（向量化实现）

    将图像分为k×k个块，每个块从其8邻域中随机选择一个块进行加权融合
    边界块会自动选择可用的邻居（少于8个）

    :param x: 输入张量 (N, C, H, W)
    :param w_b: 邻居信息的融合权重，范围[0,1]
    :param k: 网格维度，k≥3时可实现完整的8邻域效果
    :return: 块融合后的张量 (N, C, H, W)
    """
    N, C, H, W = x.shape

    # 确保维度可被k整除
    assert H % k == 0 and W % k == 0, f"图像尺寸({H},{W})必须能被k={k}整除"

    block_h, block_w = H // k, W // k

    # Reshape为 (N, C, k, block_h, k, block_w)
    blocks = x.view(N, C, k, block_h, k, block_w)

    # Permute为 (k, k, N, C, block_h, block_w) 用于块级操作
    blocks = blocks.permute(2, 4, 0, 1, 3, 5)  # shape: (k, k, N, C, block_h, block_w)

    # 创建8邻域偏移表 (8, 2)
    offsets = torch.tensor([[-1, -1], [-1, 0], [-1, 1],
                            [0, -1], [0, 1],
                            [1, -1], [1, 0], [1, 1]], device=x.device)

    # 为每个块生成随机邻居坐标 (k, k, 2)
    # 先随机选择偏移方向
    rand_offset_idx = torch.randint(8, (k, k), device=x.device)
    selected_offsets = offsets[rand_offset_idx]  # (k, k, 2)

    # 计算邻居坐标（需处理边界）
    coords_i = torch.arange(k, device=x.device).view(-1, 1, 1)
    coords_j = torch.arange(k, device=x.device).view(1, -1, 1)

    neighbor_i = (coords_i + selected_offsets[:, :, 0:1]).clamp(0, k - 1).squeeze(-1)
    neighbor_j = (coords_j + selected_offsets[:, :, 1:2]).clamp(0, k - 1).squeeze(-1)

    # 使用索引获取邻居块
    neighbor_blocks = blocks[neighbor_i, neighbor_j]

    # 加权融合：当前块*(1-w_b) + 邻居块*w_b
    fused_blocks = blocks * (1 - w_b) + neighbor_blocks * w_b

    # 恢复形状: (N, C, H, W)
    # 先permute回 (N, C, k, k, block_h, block_w)
    fused_blocks = fused_blocks.permute(2, 3, 0, 1, 4, 5)

    # 合并块
    # 沿k维度合并块
    fused = fused_blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
    return fused.view(N, C, H, W)


# 使用示例（k=2时的简化版本）
def c_component(x, w_b):
    """
    2x2网格的简化版本（每个块最多2个邻居）
    """
    return c_component_random_neighbor(x, w_b, k=2)

def FSCA_transformation(x,  s_range, w_b):
    """
    Full Frequency-Space Cooperative Augmentation transformation

    :param x: input tensor (N, C, H, W)
    :param s_range: spatial scaling range parameter (scale will be between 1/s_range and s_range)
    :param w_b: block fusion weight
    :return: transformed tensor
    """

    # S-Component: spatial rescale-shift-pad (preserves gradient)
    x_s = s_component(x, s_range)

    # C-Component: cross-pixel block fusion (preserves gradient)
    x_c = c_component(x_s, w_b)

    return x_c


class FSCA(MIFGSM):
    """
    FSCA Attack (Frequency-Space Cooperative Augmentation)

    Arguments:
        model_name (str): the name of surrogate model for attack
        epsilon (float): the perturbation budget (default: 16/255)
        alpha (float): the step size (default: 1.6/255)
        epoch (int): the number of iterations (default: 10)
        decay (float): the decay factor for momentum calculation
        num_scale (int): the number of image transformations N (default: 20)
        lambda_f (float): frequency enhancement strength (default: 0.5)
        s_range (float): spatial scaling range parameter (scale between 1/s_range and s_range, default: 1/0.7)
        w_b (float): block fusion weight (default: 0.3)
        targeted (bool): targeted/untargeted attack
        random_start (bool): whether using random initialization for delta
        norm (str): the norm of perturbation, l2/linfty
        loss (str): the loss function
        device (torch.device): the device for data. If it is None, the device would be same as model
    """

    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1.0,
                 num_scale=25,  s_range=1.5, w_b=0.4,
                 targeted=False, random_start=False,
                 norm='linfty', loss='crossentropy', device=None, attack='FSCA', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)

        self.num_scale = num_scale  # N = 20
        self.s_range = s_range  # spatial scaling range parameter
        self.w_b = w_b  # block fusion weight

    def forward(self, data, label, **kwargs):
        """
        The main attack procedure
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1]  # the second element is the targeted label tensor

        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        momentum = 0
        for _ in range(self.epoch):  # T = 10
            # Initialize gradients for this iteration
            grads = 0

            # Apply multiple FSCA transformations (N = num_scale)
            for n in range(self.num_scale):
                # Apply FSCA transformation (all operations preserve gradient)
                adv_x = data + delta
                x_trans = FSCA_transformation(
                    adv_x,
                    s_range=self.s_range,
                    w_b=self.w_b
                )

                # Calculate loss (now x_trans depends on delta through entire graph)
                logits = self.get_logits(x_trans)
                loss = self.get_loss(logits, label)

                # Calculate gradients (this will now work)
                grad = self.get_grad(loss, delta)
                grads += grad

            # Average gradients across transformations
            grads = grads / self.num_scale

            # Calculate momentum
            momentum = self.get_momentum(grads, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()