import torch
import numpy as np
import scipy.stats as st
import torch.nn.functional as F
from ..utils import *
from .._momentum import MomentumIterativeAttack
import torchvision.transforms as T
from torchvision import transforms as T


# DCT related functions
def dct1(x):
    """
    Discrete Cosine Transform, Type I

    :param x: the input signal
    :return: the DCT-I of the signal over the last dimension
    """
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])

    return torch.fft.fft(torch.cat([x, x.flip([1])[:, 1:-1]], dim=1), 1).real.view(*x_shape)


def idct1(X):
    """
    The inverse of DCT-I, which is just a scaled DCT-I

    Our definition if idct1 is such that idct1(dct1(x)) == x

    :param X: the input signal
    :return: the inverse DCT-I of the signal over the last dimension
    """
    n = X.shape[-1]
    return dct1(X) / (2 * (n - 1))


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.fft.fft(v)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc.real * W_r - Vc.imag * W_i
    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    tmp = torch.complex(real=V[:, :, 0], imag=V[:, :, 1])
    v = torch.fft.ifft(tmp)

    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape).real


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_2d(dct_2d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def dct_3d(x, norm=None):
    """
    3-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    X3 = dct(X2.transpose(-1, -3), norm=norm)
    return X3.transpose(-1, -3).transpose(-1, -2)


def idct_3d(X, norm=None):
    """
    The inverse to 3D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_3d(dct_3d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    x3 = idct(x2.transpose(-1, -3), norm=norm)
    return x3.transpose(-1, -3).transpose(-1, -2)


# Transformation functions
def get_length(length, num_block):
    length = int(length)
    rand = np.random.uniform(size=num_block, low=0.1, high=0.9)
    rand_norm = np.round(rand * length / rand.sum()).astype(np.int32)
    rand_norm[rand_norm.argmax()] += length - rand_norm.sum()
    return tuple(rand_norm)


def random_flip(x):
    ret = x.clone()
    if torch.rand(1) < 0.5:
        ret = torch.flip(ret, dims=(3,))
    return ret


def frequency_fusion(patch, x):
    org_x = x.clone()
    _, _, patch_w, patch_h = patch.shape
    rescale_x = F.interpolate(org_x, size=[patch_w, patch_h], mode='bilinear', align_corners=False)
    rescale_flip_x = random_flip(rescale_x)
    dctx = dct_2d(rescale_flip_x)
    dctp = dct_2d(patch)
    _, _, w, h = dctx.shape
    low_ratio = 0.4
    low_w = int(w * low_ratio)
    low_h = int(h * low_ratio)
    dctx[:, :, 0:low_w, 0:low_h] = dctp[:, :, 0:low_w, 0:low_h]
    idctx = idct_2d(dctx)
    return idctx


def linear_fusion(patch, x, omega=0.5):
    org_x = x.clone()
    _, _, patch_w, patch_h = patch.shape
    rescale_x = F.interpolate(org_x, size=[patch_w, patch_h], mode='bilinear', align_corners=False)
    rescale_flip_x = random_flip(rescale_x)
    ret = rescale_flip_x * omega + patch * (1 - omega)
    return ret


def block_fusion(patch, x, probabilities=0.5, omega=0.5):
    if torch.rand(1) < probabilities:
        return patch
    else:
        if torch.rand(1) < 0.5:
            return frequency_fusion(patch, x)
        else:
            return linear_fusion(patch, x, omega)


def local_fusion(x, num_block=2, probabilities=0.5, omega=0.5):
    batch_size, _, w, h = x.shape
    width_length, height_length = get_length(w, num_block), get_length(h, num_block)
    x_split_w = torch.split(x, width_length, dim=2)
    x_split_h_l = [torch.split(x_split_w[i], height_length, dim=3) for i in range(num_block)]

    ret_list = []
    for strip in x_split_h_l:
        temp_list = []
        for i in range(num_block):
            x_enh = block_fusion(strip[i], x, probabilities, omega)
            x_enh_flip = random_flip(x_enh)
            temp_list.append(x_enh_flip)
        temp = torch.cat(temp_list, dim=3)
        ret_list.append(temp)
    x_h_perm = torch.cat(ret_list, dim=2)
    return x_h_perm


def multi_scale(x, resize_ratio):
    img_size = x.shape[-1]
    if resize_ratio == 1:
        ret = x
    else:
        img_resize = int(img_size * resize_ratio)
        rescaled = F.interpolate(x, size=[img_resize, img_resize], mode='bilinear', align_corners=False)
        h_rem = img_size - img_resize
        w_rem = img_size - img_resize
        pad_top = torch.randint(low=0, high=h_rem, size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem, size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left
        ret = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)
    ret = random_flip(ret)
    return ret


# Main attack class
class SID(MomentumIterativeAttack):
    """
    SID Attack (Spatial Inconsistency and Diversity Attack)
    Based on the implementation from the provided documents

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        num_scale (int): the number of scale transformations
        num_block (int): the number of blocks for local fusion
        beta (float): the downsampling factor
        p (float): the probabilities of image block fusion
        omega (float): the weight of linear fusion
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
    """

    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1.0,
                 num_scale=20, num_block=2, beta=0.1, p=0.5, omega=0.5,
                 targeted=False, random_start=False, norm='linfty',
                 loss='crossentropy', device=None, attack='SID', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.num_scale = num_scale
        self.num_block = num_block
        self.beta = beta
        self.p = p
        self.omega = omega

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
        for _ in range(self.epoch):
            # Initialize gradients
            grads = 0

            # Apply multiple scale transformations
            for n in range(self.num_scale):
                # Obtain the output after transformation
                adv_x = data + delta
                x_emb = local_fusion(adv_x, self.num_block, self.p, self.omega)
                resize_ratio = 1 - (n * self.beta / self.num_scale)
                x_enh = multi_scale(x_emb, resize_ratio)

                # Calculate loss
                logits = self.get_logits(x_enh)
                loss = self.get_loss(logits, label)

                # Calculate gradients
                grad = self.get_grad(loss, delta)
                grads += grad

            # Average the gradients
            grads = grads / self.num_scale

            # Calculate momentum
            momentum = self.get_momentum(grads, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()

