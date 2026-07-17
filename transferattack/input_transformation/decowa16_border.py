import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import *
from ..gradient.mifgsm import MIFGSM
import random
import torchvision.transforms.functional as TF


# TPS相关函数（保持与原始代码一致）
def K_matrix(X, Y):
    eps = 1e-9
    D2 = torch.pow(X[:, :, None, :] - Y[:, None, :, :], 2).sum(-1)
    K = D2 * torch.log(D2 + eps)
    return K


def P_matrix(X):
    n, k = X.shape[:2]
    device = X.device
    P = torch.ones(n, k, 3, device=device)
    P[:, :, 1:] = X
    return P


class TPS_coeffs(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, Y):
        n, k = X.shape[:2]
        device = X.device
        Z = torch.zeros(1, k + 3, 2, device=device)
        P = torch.ones(n, k, 3, device=device)
        L = torch.zeros(n, k + 3, k + 3, device=device)
        K = K_matrix(X, X)
        P[:, :, 1:] = X
        Z[:, :k, :] = Y
        L[:, :k, :k] = K
        L[:, :k, k:] = P
        L[:, k:, :k] = P.permute(0, 2, 1)
        Q = torch.linalg.solve(L, Z)
        return Q[:, :k], Q[:, k:]


class TPS(torch.nn.Module):
    def __init__(self, size: tuple = (256, 256), device=None):
        super().__init__()
        h, w = size
        self.size = size
        self.device = device
        self.tps = TPS_coeffs()
        grid = torch.ones(1, h, w, 2, device=device)
        grid[:, :, :, 0] = torch.linspace(-1, 1, w)
        grid[:, :, :, 1] = torch.linspace(-1, 1, h)[..., None]
        self.grid = grid.view(-1, h * w, 2)

    def forward(self, X, Y):
        h, w = self.size
        W, A = self.tps(X, Y)
        U = K_matrix(self.grid, X)
        P = P_matrix(self.grid)
        grid = P @ A + U @ W
        return grid.view(-1, h, w, 2)


def grid_points_2d(width, height, device):
    xx, yy = torch.meshgrid(
        [torch.linspace(-1.0, 1.0, height, device=device),
         torch.linspace(-1.0, 1.0, width, device=device)], indexing='ij')
    return torch.stack([yy, xx], dim=-1).contiguous().view(-1, 2)


def generate_center_grid(width, height, device):
    """生成由原始控制点中心组成的新网格"""
    if width <= 1 or height <= 1:
        return torch.empty(0, 2, device=device)

    x_step = 2.0 / (width - 1)
    y_step = 2.0 / (height - 1)

    x_centers = torch.linspace(-1 + x_step / 2, 1 - x_step / 2, width - 1, device=device)
    y_centers = torch.linspace(-1 + y_step / 2, 1 - y_step / 2, height - 1, device=device)

    xx, yy = torch.meshgrid([x_centers, y_centers], indexing='ij')
    return torch.stack([yy, xx], dim=-1).contiguous().view(-1, 2)


def noisy_grid(width, height, noise_map, device):
    grid = grid_points_2d(width, height, device)
    mod = torch.zeros([height, width, 2], device=device)
    mod[1:height - 1, 1:width - 1, :] = noise_map
    return grid + mod.reshape(-1, 2)

class DeCowABM(MIFGSM):
    """
    弹性形变攻击方法（支持双控制点布局和丰富的颜色/内容变换）
    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        mesh_width: the number of the control points (width)
        mesh_height: the number of the control points (height)
        noise_scale: random noise strength (控制点偏移强度)
        num_warping: the number of warping transformation samples
        num_block: number of blocks for splicing
        retain_prob: probability of retaining the warped block
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        use_dual_grid (bool): whether to use dual grid layout (default: True)
        exclude_transform (str): which transform to exclude in ablation study (default: None)

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/elastic_warping/resnet18
                       --attack elastic_warping --model=resnet18
        python main.py --input_dir ./path/to/data --output_dir adv_data/elastic_warping/resnet18 --eval
    """

    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1.,
                 mesh_width=3, mesh_height=3, noise_scale=0.55, num_warping=25,
                 targeted=False, random_start=False, norm='linfty',
                 loss='crossentropy', device=None, attack='DeCowABM', seed=42,
                 use_dual_grid=True, exclude_transform=None, **kwargs):
        super().__init__(model_name=model_name, epsilon=epsilon, alpha=alpha, epoch=epoch, decay=decay,
                         targeted=targeted, random_start=random_start, norm=norm, loss=loss,
                         device=device, attack=attack, seed=seed)
        self.num_warping = num_warping
        self.noise_scale = noise_scale
        self.mesh_width = mesh_width
        self.mesh_height = mesh_height

        self.epsilon = epsilon
        self.seed = seed
        self.use_dual_grid = use_dual_grid
        self.exclude_transform = exclude_transform  # 新增参数

        # 生成原始控制点网格
        self.original_grid = grid_points_2d(mesh_width, mesh_height, self.device)

        # 生成中心点网格（新控制点布局）
        if use_dual_grid:
            self.center_grid = generate_center_grid(mesh_width, mesh_height, self.device)
            self.center_mesh_width = max(mesh_width - 1, 0)
            self.center_mesh_height = max(mesh_height - 1, 0)
        else:
            self.center_grid = None
            self.center_mesh_width = 0
            self.center_mesh_height = 0

        # 设置随机种子以确保TPS变换的确定性
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)

        # 添加 Python random 模块的种子设置
        random.seed(self.seed)

    def contrast(self, x):
        r = float((torch.rand(1, ) * 2 - 1) * 1 + 1)
        return TF.adjust_contrast(x, r)

    def saturation(self, x):
        r = float((torch.rand(1, ) * 2 - 1) * 1 + 1)
        return TF.adjust_saturation(x, r)

    def brightness(self, x):
        r = float((torch.rand(1, ) * 2 - 1) * 1 + 1)
        return TF.adjust_brightness(x, r)

    # --- 高级颜色/内容变换 ---
    def color_noise(self, x, noise_std=0.05):
        """添加彩色噪声，每个通道独立"""
        noise = torch.randn_like(x) * noise_std
        return torch.clamp(x + noise, 0.0, 1.0)

    def grayscale_noise(self, x, noise_std=0.05):
        """添加灰度噪声，所有通道共享相同的噪声"""
        noise_map = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) * noise_std
        return torch.clamp(x + noise_map, 0.0, 1.0)

    # =================================================================

    def elastic_warp(self, x, use_center_grid=False):
        """
        对图像进行弹性形变（使用replicate扩展技术,填充量基于控制点的最大偏移值）
        :param x: 输入图像 (B, C, H, W)
        :param use_center_grid: 是否使用中心点网格
        :return: 形变后的图像
        """
        n, c, h, w = x.size()

        # 根据选择的网格类型设置参数
        if use_center_grid and self.center_grid.numel() > 0:
            mesh_width = self.center_mesh_width
            mesh_height = self.center_mesh_height
            X = self.center_grid
        else:
            mesh_width = self.mesh_width
            mesh_height = self.mesh_height
            X = self.original_grid

        # 【修改点】根据控制点的最大偏移值计算填充量
        max_offset_norm = 0.5 * self.noise_scale

        # 【修复 TypeError】将 float 结果显式转换为 Tensor
        pad_w_float = max_offset_norm * (w - 1) / 2
        pad_h_float = max_offset_norm * (h - 1) / 2

        pad_w = int(torch.ceil(torch.tensor(pad_w_float, device=self.device)).item())
        pad_h = int(torch.ceil(torch.tensor(pad_h_float, device=self.device)).item())

        # 确保填充量至少为1，以避免边界问题
        pad_w = max(1, pad_w)
        pad_h = max(1, pad_h)

        # 【修复】使用replicate填充扩展图像边界（border扩展在PyTorch中对应replicate模式）
        padded_img = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode='replicate')
        new_h, new_w = h + 2 * pad_h, w + 2 * pad_w

        # 生成覆盖所有控制点的随机偏移
        noise_map = (torch.rand([mesh_height, mesh_width, 2],
                                device=self.device) - 0.5) * self.noise_scale

        # 直接将噪声加到所有控制点上
        Y = X + noise_map.reshape(-1, 2)

        # 将控制点坐标从原始图像坐标系映射到扩展图像坐标系
        X_pixel = (X + 1) * (w - 1) / 2
        Y_pixel = (Y + 1) * (h - 1) / 2

        X_padded_pixel = X_pixel + pad_w
        Y_padded_pixel = Y_pixel + pad_h

        X_padded_norm = X_padded_pixel * 2 / (new_w - 1) - 1
        Y_padded_norm = Y_padded_pixel * 2 / (new_h - 1) - 1

        # 创建针对扩展图像的TPS变换器
        tps_padded = TPS(size=(new_h, new_w), device=self.device)

        # 应用TPS变换，使用调整后的控制点
        warped_grid = tps_padded(X_padded_norm[None, ...], Y_padded_norm[None, ...])
        warped_grid = warped_grid.repeat(x.shape[0], 1, 1, 1)
        warped_grid = warped_grid.view(n, new_h, new_w, 2)

        # 在扩展图像上应用形变
        warped_padded_img = torch.grid_sampler_2d(padded_img, warped_grid, 0, 0, False)

        # 裁剪回原始尺寸
        warped_img = warped_padded_img[:, :, pad_h:pad_h + h, pad_w:pad_w + w]

        return warped_img

    def forward(self, data, label, **kwargs):
        if self.targeted:
            assert len(label) == 2
            label = label[1]  # the second element is the targeted label tensor

        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)
        momentum = 0

        # 【核心修改】创建一个包含所有颜色/内容变换的列表
        all_transforms = [
            self.brightness,
            # 高级颜色/内容变换
            self.color_noise,
        ]

        # 根据exclude_transform参数排除特定变换
        if self.exclude_transform is not None:
            transform_names = [  'brightness', 'color_noise',]
            if self.exclude_transform in transform_names:
                idx = transform_names.index(self.exclude_transform)
                all_transforms.pop(idx)

        for epoch_idx in range(self.epoch):
            # 设置当前epoch的随机种子，确保噪声生成可复现
            current_seed = self.seed + epoch_idx
            torch.manual_seed(current_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(current_seed)

            # 添加 Python random 模块的种子设置
            random.seed(current_seed)

            # 随机选择控制点布局（仅在启用双网格时）
            use_center_grid = False
            if self.use_dual_grid and self.center_grid.numel() > 0:
                use_center_grid = random.random() < 0.5

            grads = torch.zeros_like(data)

            for warp_idx in range(self.num_warping):
                # 1. 对当前对抗样本进行变换
                adv = data + delta  # 保持梯度连接

                # 【核心修改】从所有变换中随机选择一个
                transform = random.choice(all_transforms)
                adv = transform(adv)

                # 应用弹性形变（使用选择的控制点布局）
                warped_img = self.elastic_warp(adv, use_center_grid)

                # 直接使用混合图像计算梯度
                logits = self.get_logits(warped_img)
                loss = self.get_loss(logits, label)

                # 计算mixed_img对adv的梯度
                grad_adv = torch.autograd.grad(loss, adv, retain_graph=True, create_graph=False)[0]
                # 因为adv = data + delta，所以grad_delta = grad_adv
                grad_delta = grad_adv

                grads += grad_delta

            # 平均梯度
            grads /= self.num_warping

            # 计算动量
            momentum = self.get_momentum(grads, momentum)

            # 更新对抗扰动
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()