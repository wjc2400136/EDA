import torch
import numpy as np
import torchvision.transforms as T

from ..utils import *
from ..gradient.mifgsm import MIFGSM


def generate_noise(x: torch.Tensor, scale: float):
    """
    Generate random noise tensor

    :param x: input tensor for shape reference
    :param scale: noise scale
    :return: noise tensor
    """
    noise = (torch.rand_like(x) - 0.5) * 2 * scale
    noise.requires_grad_(True)
    return noise


def get_length(length: int, num_block: int):
    """
    Calculate random block lengths that sum to the total length

    For the meaning of the parameters, see the original implementation:
    https://github.com/Zhijin-Ge/TransferAttack/blob/main/transferattack/input_transformation/bsr.py

    :param length: total length to split
    :param num_block: number of blocks
    :return: tuple of block lengths
    """
    length = int(length)
    rand = np.random.uniform(size=num_block, low=0.1, high=0.9)
    rand_norm = np.round(rand * length / rand.sum()).astype(np.int32)
    rand_norm[rand_norm.argmax()] += length - rand_norm.sum()
    return tuple(rand_norm)


def shuffle_single_dim(x: torch.Tensor, dim: int, num_block: int):
    """
    Shuffle tensor strips along a single dimension

    :param x: input tensor
    :param dim: dimension to shuffle
    :param num_block: number of blocks
    :return: list of shuffled tensor strips
    """
    lengths = get_length(x.size(dim), num_block)
    x_strips = list(x.split(lengths, dim=dim))
    np.random.shuffle(x_strips)
    return x_strips


def shuffle(x: torch.Tensor, num_block: int):
    """
    Apply block shuffling in random order across height and width dimensions

    The block shuffle implementation is based on:
    https://github.com/Zhijin-Ge/TransferAttack/blob/main/transferattack/input_transformation/bsr.py

    :param x: input tensor of shape (batch_size, channels, height, width)
    :param num_block: number of blocks for shuffling
    :return: shuffled tensor
    """
    dims = [2, 3]
    np.random.shuffle(dims)
    x_strips = shuffle_single_dim(x, dims[0], num_block)

    # Apply second dimension shuffling to each strip
    shuffled_strips = []
    for x_strip in x_strips:
        second_dim_shuffled = shuffle_single_dim(x_strip, dims[1], num_block)
        shuffled_strips.append(torch.cat(second_dim_shuffled, dim=dims[1]))

    return torch.cat(shuffled_strips, dim=dims[0])
# grid_shuffle_transformation.py
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as T


# ---------- 核心 Grid-Shuffle 坐标变换 ----------
def grid_shuffle_map(x: torch.Tensor, grid_size: int):
    """
    生成相邻交换的坐标映射并直接重采样
    x: (B,C,H,W)  Tensor  [0,1]
    return: (B,C,H,W)  Grid-Shuffle 后的图像
    """
    B, C, H, W = x.shape
    device = x.device

    # 1. 规则坐标网格  [-1,1]  顺序 (x,y)
    base_x = torch.linspace(-1, 1, grid_size, device=device)
    base_y = torch.linspace(-1, 1, grid_size, device=device)
    grid = torch.stack(torch.meshgrid(base_x, base_y, indexing='xy'), dim=-1)  # g×g×2

    # 2. 仅相邻交换（左右 or 上下）
    grid = grid.clone()
    for i in range(grid_size):
        for j in range(grid_size):
            if j < grid_size - 1 and np.random.rand() < 0.5:
                grid[i, j], grid[i, j + 1] = grid[i, j + 1].clone(), grid[i, j].clone()
            elif i < grid_size - 1 and np.random.rand() < 0.5:
                grid[i, j], grid[i + 1, j] = grid[i + 1, j].clone(), grid[i, j].clone()

    # 3. 插值到图像分辨率
    grid = grid.permute(2, 0, 1).unsqueeze(0)          # 1,2,g,g
    grid = F.interpolate(grid, size=(H, W), mode='bilinear', align_corners=True)
    grid = grid.squeeze(0).permute(1, 2, 0)            # H,W,2

    # 4. 按新坐标重采样
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)     # B,H,W,2
    x_shuf = F.grid_sample(x, grid, align_corners=True)
    return x_shuf


# ---------- 供框架调用的统一接口 ----------
def grid_shuffle_transformation(x: torch.Tensor, grid_size: int = 10):
    """
    纯 Grid-Shuffle 输入变换，供攻击框架直接调用
    x: (B,C,H,W)
    return: (B,C,H,W)  已混合图像  0.3*shuffled + 0.7*original
    """
    x_shuf = grid_shuffle_map(x, grid_size=grid_size)
    return 0.1 * x_shuf + 0.3 * x          # 与 I-C 原代码保持一致混合比例

def I_C_transformation(x: torch.Tensor, noise_scale: float, num_block: int, resize_factor: float):
    """
    I-C transformation for attack augmentation

    This transformation can replace the transformation in a transformation-based
    attack framework to obtain the full I-C attack.

    Recommended framework:
    https://github.com/Zhijin-Ge/TransferAttack/blob/main/transferattack/input_transformation

    :param x: input tensor of shape (batch_size, channels, height, width)
    :param noise_scale: scale factor for noise generation (parameter a)
    :param num_block: number of blocks for shuffling (parameter b)
    :param resize_factor: resizing factor (parameter r)
    :return: transformed tensor
    """
    _, _, height, width = x.shape

    # Generate noise
    noise = generate_noise(x, noise_scale)

    # Apply block shuffling
    x_shuffled = shuffle(x=x, num_block=num_block)

    # Add noise
    x_noisy = x_shuffled + noise

    # Resize and crop
    resize_transform = T.Compose([
        T.Resize((int(resize_factor * height), int(resize_factor * width))),
        T.CenterCrop((height, width))
    ])

    return resize_transform(x_noisy)


class GS(MIFGSM):
    """
    I-C Attack (Input-Congruent Attack)

    Arguments:
        model_name (str): the name of surrogate model for attack
        epsilon (float): the perturbation budget (default: 16/255)
        alpha (float): the step size (default: 1.6/255)
        epoch (int): the number of iterations (default: 10)
        decay (float): the decay factor for momentum calculation
        num_scale (int): the number of image transformations N (default: 20)
        num_block (int): the number of blocks for shuffling b (default: 3)
        noise_scale (float): scale factor for noise generation a
        resize_factor_range (tuple): range (min, max) for random resize factor r
        mode (str): 'undefended' or 'defended' target mode (for default parameter setting)
        targeted (bool): targeted/untargeted attack
        random_start (bool): whether using random initialization for delta
        norm (str): the norm of perturbation, l2/linfty
        loss (str): the loss function
        device (torch.device): the device for data. If it is None, the device would be same as model
    """

    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1.0,
                 num_scale=25, num_block=3, noise_scale=None, resize_factor_range=None,
                 mode='undefended', targeted=False, random_start=False,
                 norm='linfty', loss='crossentropy', device=None, attack='IC', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)

        self.num_scale = num_scale  # N = 20
        self.num_block = num_block  # b = 3
        self.mode = mode

        # Set default parameters based on mode
        if mode == 'undefended':
            self.noise_scale = noise_scale if noise_scale is not None else 0.07  # a = 0.07
            self.resize_factor_range = resize_factor_range if resize_factor_range is not None else (
            1.0, 1.40)  # r = 1.40
        elif mode == 'defended':
            self.noise_scale = noise_scale if noise_scale is not None else 0.15  # a = 0.15
            self.resize_factor_range = resize_factor_range if resize_factor_range is not None else (
            1.0, 1.10)  # r = 1.10
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose 'undefended' or 'defended'")

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

            # Apply multiple I-C transformations (N = num_scale)
            for n in range(self.num_scale):
                # Sample random resize factor for this transformation
                resize_factor = np.random.uniform(
                    self.resize_factor_range[0],
                    self.resize_factor_range[1]
                )

                # Apply I-C transformation
                adv_x = data + delta
                x_trans = grid_shuffle_transformation(adv_x, grid_size=10)

                # Calculate loss
                logits = self.get_logits(x_trans)
                loss = self.get_loss(logits, label)

                # Calculate gradients
                grad = self.get_grad(loss, delta)
                grads += grad

            # Average gradients across transformations (combine with MI-FGSM)
            grads = grads / self.num_scale

            # Calculate momentum
            momentum = self.get_momentum(grads, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()