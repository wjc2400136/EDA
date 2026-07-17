from functools import partial
import torch
import torch.fft
import numpy as np
from ..gradient.mifgsm import MIFGSM
from ..utils import *


class FER(MIFGSM):
    """
    FER Attack
    'Improving Adversarial Transferability via Frequency Energy Redistribution (CVPR 2025)'

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data.
        alpha_fer (float): the redistribution strength for commonality enhancement (α).
        beta_fer (float): the redistribution strength for individuality suppression (β).

    Official arguments:
        epsilon=16/255, alpha=1.6/255, epoch=10, decay=1, alpha_fer=0.5, beta_fer=0.5.

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/fer/resnet --attack fer --model=resnet50 --batchsize=8
    """

    def __init__(self, model_name='vit_base_patch16_224', epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1.,
                 targeted=False, random_start=False, norm='linfty', loss='crossentropy', device=None,
                 attack='fer', alpha_fer=0.5, beta_fer=0.5, **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.alpha_fer = alpha_fer
        self.beta_fer = beta_fer

    def forward(self, data, label, **kwargs):
        """
        The general attack procedure

        Arguments:
            data (N, C, H, W): tensor for input images
            labels (N,): tensor for ground-truth labels if untargetd
            labels (2,N): tensor for [ground-truth, targeted labels] if targeted
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1]

        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        momentum = 0
        for _ in range(self.epoch):
            # Apply frequency energy redistribution
            perturbed_data = self.frequency_redistribution(data + delta)

            # Obtain the output
            logits = self.get_logits(perturbed_data)

            # Calculate the loss
            loss = self.get_loss(logits, label)

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()

    def frequency_redistribution(self, x):
        """
        Frequency Energy Redistribution (FER) module - Fixed Version

        Arguments:
            x (N, C, H, W): input tensor

        Returns:
            x_redistributed (N, C, H, W): redistributed tensor
        """
        N, C, H, W = x.shape

        # Pre-compute frequency grid (static)
        freq_y = torch.fft.fftfreq(H, device=x.device, dtype=torch.float32)
        freq_x = torch.fft.fftfreq(W, device=x.device, dtype=torch.float32)
        freq_grid = torch.sqrt(freq_y[:, None] ** 2 + freq_x[None, :] ** 2)
        freq_grid = freq_grid / freq_grid.max()

        x_redistributed = torch.zeros_like(x)

        for c in range(C):
            # Get channel data
            x_channel = x[:, c, :, :]

            # Apply FFT
            x_fft = torch.fft.fft2(x_channel)
            x_fft_shifted = torch.fft.fftshift(x_fft)

            # Compute energy MANUALLY: |z|^2 = Re(z)^2 + Im(z)^2
            # This avoids torch.abs() on complex tensors which triggers NVRTC
            energy = x_fft_shifted.real.pow(2) + x_fft_shifted.imag.pow(2)

            # Average across batch for stable statistics
            avg_energy = energy.mean(dim=0)

            # Sort by frequency position (not by energy magnitude)
            flat_freq = freq_grid.flatten()
            sorted_indices = torch.argsort(flat_freq)

            # Compute CDF of energy sorted by frequency
            sorted_energy = avg_energy.flatten()[sorted_indices]
            cdf = torch.cumsum(sorted_energy, dim=0)
            cdf = cdf / (cdf[-1] + 1e-8)  # Add epsilon to avoid division by zero

            # Map back to 2D
            cdf_map = torch.zeros_like(avg_energy.flatten())
            cdf_map[sorted_indices] = cdf
            cdf_2d = cdf_map.view(H, W)

            # Compute redistribution matrix R
            R = 1.0 + self.alpha_fer * (1.0 - cdf_2d) - self.beta_fer * cdf_2d

            # Apply redistribution
            x_fft_redistributed = x_fft_shifted * R.unsqueeze(0)

            # Inverse shift and IFFT
            x_fft_ishifted = torch.fft.ifftshift(x_fft_redistributed)
            x_ifft = torch.fft.ifft2(x_fft_ishifted)

            # Store real part
            x_redistributed[:, c, :, :] = x_ifft.real

        return x_redistributed