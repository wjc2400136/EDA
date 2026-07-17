import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import *
from ..gradient.mifgsm import MIFGSM
import random
import torchvision.transforms.functional as TF


# =================================================================
# TPS 相关函数（所有变体通用）
# =================================================================
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


# =================================================================
# 基类：包含所有通用逻辑
# =================================================================
class DeCowABMBase(MIFGSM):
    """
    弹性形变攻击方法的基类，用于消融实验。
    包含所有通用功能，子类只需实现独特的 elastic_warp 方法。
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
        self.exclude_transform = exclude_transform

        # 生成原始控制点网格
        self.original_grid = grid_points_2d(mesh_width, mesh_height, self.device)

        # 生成中心点网格
        if use_dual_grid:
            self.center_grid = generate_center_grid(mesh_width, mesh_height, self.device)
            self.center_mesh_width = max(mesh_width - 1, 0)
            self.center_mesh_height = max(mesh_height - 1, 0)
        else:
            self.center_grid = None
            self.center_mesh_width = 0
            self.center_mesh_height = 0

        # 设置随机种子
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)

    def brightness(self, x):
        r = float((torch.rand(1, ) * 2 - 1) * 1 + 1)
        return TF.adjust_brightness(x, r)

    def channel_noise(self, x, noise_std=0.05):
        """添加通道噪声，每个通道独立"""
        noise = torch.randn_like(x) * noise_std
        return torch.clamp(x + noise, 0.0, 1.0)

    def elastic_warp(self, x, use_center_grid=False):
        """
        子类必须实现此方法以定义特定的弹性形变逻辑。
        """
        raise NotImplementedError("Subclasses must implement the elastic_warp method.")

    def forward(self, data, label, **kwargs):
        if self.targeted:
            assert len(label) == 2
            label = label[1]

        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        delta = self.init_delta(data)
        momentum = 0

        all_transforms = [self.brightness, self.channel_noise]
        transform_names = ['brightness', 'channel_noise']
        if self.exclude_transform is not None:
            if self.exclude_transform in transform_names:
                idx = transform_names.index(self.exclude_transform)
                all_transforms.pop(idx)

        for epoch_idx in range(self.epoch):
            current_seed = self.seed + epoch_idx
            torch.manual_seed(current_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(current_seed)
            random.seed(current_seed)

            use_center_grid = False
            if self.use_dual_grid and self.center_grid.numel() > 0:
                use_center_grid = random.random() < 0.5

            grads = torch.zeros_like(data)

            for warp_idx in range(self.num_warping):
                adv = data + delta
                transform = random.choice(all_transforms)
                adv = transform(adv)
                warped_img = self.elastic_warp(adv, use_center_grid)

                logits = self.get_logits(warped_img)
                loss = self.get_loss(logits, label)
                grad_adv = torch.autograd.grad(loss, adv, retain_graph=True, create_graph=False)[0]
                grads += grad_adv

            grads /= self.num_warping
            momentum = self.get_momentum(grads, momentum)
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()


# =================================================================
# 子类：四种不同的控制点布局实现
# =================================================================

class DeCowABM_A1(DeCowABMBase):
    """
    配置 A1: 单网格-带边缘
    - 只使用原始网格
    - 对所有控制点（包括边缘）施加噪声
    """

    def elastic_warp(self, x, use_center_grid=False):
        n, c, h, w = x.size()
        mesh_width, mesh_height = self.mesh_width, self.mesh_height
        X = self.original_grid

        max_offset_norm = 0.5 * self.noise_scale
        pad_w_float = max_offset_norm * (w - 1) / 2
        pad_h_float = max_offset_norm * (h - 1) / 2
        pad_w = int(torch.ceil(torch.tensor(pad_w_float, device=self.device)).item())
        pad_h = int(torch.ceil(torch.tensor(pad_h_float, device=self.device)).item())
        pad_w, pad_h = max(1, pad_w), max(1, pad_h)

        padded_img = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
        new_h, new_w = h + 2 * pad_h, w + 2 * pad_w

        # 【A1 特定】对所有控制点施加噪声
        noise_map = (torch.rand([mesh_height, mesh_width, 2], device=self.device) - 0.5) * self.noise_scale
        Y = X + noise_map.reshape(-1, 2)

        X_pixel = (X + 1) * (w - 1) / 2
        Y_pixel = (Y + 1) * (h - 1) / 2
        X_padded_pixel = X_pixel + pad_w
        Y_padded_pixel = Y_pixel + pad_h
        X_padded_norm = X_padded_pixel * 2 / (new_w - 1) - 1
        Y_padded_norm = Y_padded_pixel * 2 / (new_h - 1) - 1

        tps_padded = TPS(size=(new_h, new_w), device=self.device)
        warped_grid = tps_padded(X_padded_norm[None, ...], Y_padded_norm[None, ...])
        warped_grid = warped_grid.repeat(x.shape[0], 1, 1, 1).view(n, new_h, new_w, 2)
        warped_padded_img = torch.grid_sampler_2d(padded_img, warped_grid, 0, 0, False)
        warped_img = warped_padded_img[:, :, pad_h:pad_h + h, pad_w:pad_w + w]

        return warped_img


class DeCowABM_A2(DeCowABMBase):
    """
    配置 A2: 单网格-不带边缘
    - 只使用原始网格
    - 只对内部控制点施加噪声，边缘点保持不动
    """

    def elastic_warp(self, x, use_center_grid=False):
        n, c, h, w = x.size()
        mesh_width, mesh_height = self.mesh_width, self.mesh_height
        X = self.original_grid

        max_offset_norm = 0.5 * self.noise_scale
        pad_w_float = max_offset_norm * (w - 1) / 2
        pad_h_float = max_offset_norm * (h - 1) / 2
        pad_w = int(torch.ceil(torch.tensor(pad_w_float, device=self.device)).item())
        pad_h = int(torch.ceil(torch.tensor(pad_h_float, device=self.device)).item())
        pad_w, pad_h = max(1, pad_w), max(1, pad_h)

        padded_img = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
        new_h, new_w = h + 2 * pad_h, w + 2 * pad_w

        # 【A2 特定】只对内部控制点施加噪声
        noise_map = torch.zeros([mesh_height, mesh_width, 2], device=self.device)
        if mesh_height > 2 and mesh_width > 2:
            noise_map[1:mesh_height - 1, 1:mesh_width - 1, :] = (torch.rand([mesh_height - 2, mesh_width - 2, 2],
                                                                            device=self.device) - 0.5) * self.noise_scale
        Y = X + noise_map.reshape(-1, 2)

        X_pixel = (X + 1) * (w - 1) / 2
        Y_pixel = (Y + 1) * (h - 1) / 2
        X_padded_pixel = X_pixel + pad_w
        Y_padded_pixel = Y_pixel + pad_h
        X_padded_norm = X_padded_pixel * 2 / (new_w - 1) - 1
        Y_padded_norm = Y_padded_pixel * 2 / (new_h - 1) - 1

        tps_padded = TPS(size=(new_h, new_w), device=self.device)
        warped_grid = tps_padded(X_padded_norm[None, ...], Y_padded_norm[None, ...])
        warped_grid = warped_grid.repeat(x.shape[0], 1, 1, 1).view(n, new_h, new_w, 2)
        warped_padded_img = torch.grid_sampler_2d(padded_img, warped_grid, 0, 0, False)
        warped_img = warped_padded_img[:, :, pad_h:pad_h + h, pad_w:pad_w + w]

        return warped_img


class DeCowABM_B1(DeCowABMBase):
    """
    配置 B1: 双网格-原始带边缘 (原始实现)
    - 随机选择使用原始网格或中心网格
    - 原始网格的所有点都施加噪声
    - 中心网格的所有点都施加噪声
    """

    def elastic_warp(self, x, use_center_grid=False):
        n, c, h, w = x.size()

        if use_center_grid and self.center_grid.numel() > 0:
            mesh_width, mesh_height = self.center_mesh_width, self.center_mesh_height
            X = self.center_grid
        else:
            mesh_width, mesh_height = self.mesh_width, self.mesh_height
            X = self.original_grid

        max_offset_norm = 0.5 * self.noise_scale
        pad_w_float = max_offset_norm * (w - 1) / 2
        pad_h_float = max_offset_norm * (h - 1) / 2
        pad_w = int(torch.ceil(torch.tensor(pad_w_float, device=self.device)).item())
        pad_h = int(torch.ceil(torch.tensor(pad_h_float, device=self.device)).item())
        pad_w, pad_h = max(1, pad_w), max(1, pad_h)

        padded_img = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
        new_h, new_w = h + 2 * pad_h, w + 2 * pad_w

        # 【B1 特定】对所选网格的所有控制点施加噪声
        noise_map = (torch.rand([mesh_height, mesh_width, 2], device=self.device) - 0.5) * self.noise_scale
        Y = X + noise_map.reshape(-1, 2)

        X_pixel = (X + 1) * (w - 1) / 2
        Y_pixel = (Y + 1) * (h - 1) / 2
        X_padded_pixel = X_pixel + pad_w
        Y_padded_pixel = Y_pixel + pad_h
        X_padded_norm = X_padded_pixel * 2 / (new_w - 1) - 1
        Y_padded_norm = Y_padded_pixel * 2 / (new_h - 1) - 1

        tps_padded = TPS(size=(new_h, new_w), device=self.device)
        warped_grid = tps_padded(X_padded_norm[None, ...], Y_padded_norm[None, ...])
        warped_grid = warped_grid.repeat(x.shape[0], 1, 1, 1).view(n, new_h, new_w, 2)
        warped_padded_img = torch.grid_sampler_2d(padded_img, warped_grid, 0, 0, False)
        warped_img = warped_padded_img[:, :, pad_h:pad_h + h, pad_w:pad_w + w]

        return warped_img


class DeCowABM_B2(DeCowABMBase):
    """
    配置 B2: 双网格-原始不带边缘
    - 随机选择使用原始网格或中心网格
    - 原始网格只对内部控制点施加噪声
    - 中心网格的所有点都施加噪声
    """

    def elastic_warp(self, x, use_center_grid=False):
        n, c, h, w = x.size()

        if use_center_grid and self.center_grid.numel() > 0:
            mesh_width, mesh_height = self.center_mesh_width, self.center_mesh_height
            X = self.center_grid
            # 【B2 特定】中心网格：对所有点施加噪声
            noise_map = (torch.rand([mesh_height, mesh_width, 2], device=self.device) - 0.5) * self.noise_scale
        else:
            mesh_width, mesh_height = self.mesh_width, self.mesh_height
            X = self.original_grid
            # 【B2 特定】原始网格：只对内部控制点施加噪声
            noise_map = torch.zeros([mesh_height, mesh_width, 2], device=self.device)
            if mesh_height > 2 and mesh_width > 2:
                noise_map[1:mesh_height - 1, 1:mesh_width - 1, :] = (torch.rand([mesh_height - 2, mesh_width - 2, 2],
                                                                                device=self.device) - 0.5) * self.noise_scale

        Y = X + noise_map.reshape(-1, 2)

        max_offset_norm = 0.5 * self.noise_scale
        pad_w_float = max_offset_norm * (w - 1) / 2
        pad_h_float = max_offset_norm * (h - 1) / 2
        pad_w = int(torch.ceil(torch.tensor(pad_w_float, device=self.device)).item())
        pad_h = int(torch.ceil(torch.tensor(pad_h_float, device=self.device)).item())
        pad_w, pad_h = max(1, pad_w), max(1, pad_h)

        padded_img = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
        new_h, new_w = h + 2 * pad_h, w + 2 * pad_w

        X_pixel = (X + 1) * (w - 1) / 2
        Y_pixel = (Y + 1) * (h - 1) / 2
        X_padded_pixel = X_pixel + pad_w
        Y_padded_pixel = Y_pixel + pad_h
        X_padded_norm = X_padded_pixel * 2 / (new_w - 1) - 1
        Y_padded_norm = Y_padded_pixel * 2 / (new_h - 1) - 1

        tps_padded = TPS(size=(new_h, new_w), device=self.device)
        warped_grid = tps_padded(X_padded_norm[None, ...], Y_padded_norm[None, ...])
        warped_grid = warped_grid.repeat(x.shape[0], 1, 1, 1).view(n, new_h, new_w, 2)
        warped_padded_img = torch.grid_sampler_2d(padded_img, warped_grid, 0, 0, False)
        warped_img = warped_padded_img[:, :, pad_h:pad_h + h, pad_w:pad_w + w]

        return warped_img
