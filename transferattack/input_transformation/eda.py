import random

import torch
import torch.nn.functional as F

from ..gradient.mifgsm import MIFGSM
from ..utils import *


def tps_kernel_matrix(x, y):
    """Thin-plate-spline radial basis kernel U(r)=r^2 log(r^2)."""
    eps = 1e-9
    dist_sq = torch.pow(x[:, :, None, :] - y[:, None, :, :], 2).sum(-1)
    return dist_sq * torch.log(dist_sq + eps)


def tps_affine_matrix(x):
    batch_size, num_points = x.shape[:2]
    p = torch.ones(batch_size, num_points, 3, device=x.device, dtype=x.dtype)
    p[:, :, 1:] = x
    return p


class TPSCoefficients(torch.nn.Module):
    """Solve TPS coefficients for source points x and target points y."""

    def forward(self, x, y):
        batch_size, num_points = x.shape[:2]
        device = x.device
        dtype = x.dtype

        k = tps_kernel_matrix(x, x)
        p = tps_affine_matrix(x)

        lhs = torch.zeros(
            batch_size,
            num_points + 3,
            num_points + 3,
            device=device,
            dtype=dtype,
        )
        rhs = torch.zeros(batch_size, num_points + 3, 2, device=device, dtype=dtype)

        lhs[:, :num_points, :num_points] = k
        lhs[:, :num_points, num_points:] = p
        lhs[:, num_points:, :num_points] = p.transpose(1, 2)
        rhs[:, :num_points, :] = y

        coeffs = torch.linalg.solve(lhs, rhs)
        return coeffs[:, :num_points], coeffs[:, num_points:]


class TPSGrid(torch.nn.Module):
    """Build a dense TPS sampling grid for an image canvas."""

    def __init__(self, size, device=None, dtype=torch.float32):
        super().__init__()
        height, width = size
        self.size = size
        self.coeffs = TPSCoefficients()

        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack([xx, yy], dim=-1).view(1, height * width, 2)
        self.register_buffer("grid", grid, persistent=False)

    def forward(self, source_points, target_points):
        height, width = self.size
        weights, affine = self.coeffs(source_points, target_points)
        kernel = tps_kernel_matrix(self.grid, source_points)
        p = tps_affine_matrix(self.grid)
        grid = p @ affine + kernel @ weights
        return grid.view(-1, height, width, 2)


def regular_grid(width, height, device, dtype=torch.float32):
    """Return full-grid control points in normalized [x,y] coordinates."""
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx, yy], dim=-1).contiguous().view(-1, 2)


def center_grid(width, height, device, dtype=torch.float32):
    """Return the cell-center grid used by the dual control point layout."""
    if width <= 1 or height <= 1:
        return torch.empty(0, 2, device=device, dtype=dtype)

    x_step = 2.0 / (width - 1)
    y_step = 2.0 / (height - 1)
    x = torch.linspace(
        -1.0 + x_step / 2.0,
        1.0 - x_step / 2.0,
        width - 1,
        device=device,
        dtype=dtype,
    )
    y = torch.linspace(
        -1.0 + y_step / 2.0,
        1.0 - y_step / 2.0,
        height - 1,
        device=device,
        dtype=dtype,
    )
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx, yy], dim=-1).contiguous().view(-1, 2)


class EDA(MIFGSM):
    """
    Enhanced Deformation Attack aligned with the paper description.

    The attack estimates the gradient of an expected transformed loss by
    averaging gradients over multiple independently sampled transformations:
    dual control point layout, TPS offsets, reflection canvas expansion, and
    randomized appearance augmentation.
    """

    def __init__(
        self,
        model_name,
        epsilon=16 / 255,
        alpha=1.6 / 255,
        epoch=10,
        decay=1.0,
        mesh_width=3,
        mesh_height=3,
        noise_scale=0.45,
        num_warping=25,
        targeted=False,
        random_start=False,
        norm="linfty",
        loss="crossentropy",
        device=None,
        attack="EDA",
        seed=42,
        use_dual_grid=True,
        move_edge=True,
        use_appearance=True,
        exclude_transform=None,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            epsilon=epsilon,
            alpha=alpha,
            epoch=epoch,
            decay=decay,
            targeted=targeted,
            random_start=random_start,
            norm=norm,
            loss=loss,
            device=device,
            attack=attack,
        )
        self.num_warping = num_warping
        self.noise_scale = noise_scale
        self.mesh_width = mesh_width
        self.mesh_height = mesh_height
        self.seed = seed
        self.use_dual_grid = use_dual_grid
        self.move_edge = move_edge
        self.use_appearance = use_appearance
        self.exclude_transform = exclude_transform

        self.full_grid = regular_grid(mesh_width, mesh_height, self.device)
        self.center_grid = (
            center_grid(mesh_width, mesh_height, self.device)
            if use_dual_grid
            else None
        )
        self.center_mesh_width = max(mesh_width - 1, 0)
        self.center_mesh_height = max(mesh_height - 1, 0)
        self._tps_cache = {}

        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            random.seed(self.seed)

    def _select_layout(self):
        if self.use_dual_grid and self.center_grid is not None and self.center_grid.numel() > 0:
            if random.random() < 0.5:
                return self.center_grid, self.center_mesh_width, self.center_mesh_height
        return self.full_grid, self.mesh_width, self.mesh_height

    def _sample_offsets(self, points, mesh_width, mesh_height):
        offsets = (torch.rand_like(points) - 0.5) * self.noise_scale
        if self.move_edge:
            return offsets

        # Fixed-edge ablation: do not move points on the full-grid boundary.
        if points.shape[0] == mesh_width * mesh_height:
            boundary = (
                torch.isclose(points[:, 0], points[:, 0].new_tensor(-1.0))
                | torch.isclose(points[:, 0], points[:, 0].new_tensor(1.0))
                | torch.isclose(points[:, 1], points[:, 1].new_tensor(-1.0))
                | torch.isclose(points[:, 1], points[:, 1].new_tensor(1.0))
            )
            offsets = offsets.clone()
            offsets[boundary] = 0.0
        return offsets

    def _padding_amounts(self, height, width):
        max_offset_norm = 0.5 * self.noise_scale
        pad_w = int(torch.ceil(torch.tensor(max_offset_norm * (width - 1) / 2)).item())
        pad_h = int(torch.ceil(torch.tensor(max_offset_norm * (height - 1) / 2)).item())
        return pad_h, pad_w

    def _to_pixel_coords(self, points, height, width):
        scale = points.new_tensor([(width - 1) / 2.0, (height - 1) / 2.0])
        return (points + 1.0) * scale

    def _to_normalized_coords(self, points, height, width):
        scale = points.new_tensor([2.0 / (width - 1), 2.0 / (height - 1)])
        return points * scale - 1.0

    def _get_tps_grid(self, height, width, dtype):
        cache_key = (height, width, str(self.device), str(dtype))
        if cache_key not in self._tps_cache:
            self._tps_cache[cache_key] = TPSGrid(
                size=(height, width),
                device=self.device,
                dtype=dtype,
            )
        return self._tps_cache[cache_key]

    def elastic_warp(self, x):
        batch_size, _, height, width = x.size()
        source_points, mesh_width, mesh_height = self._select_layout()
        source_points = source_points.to(device=x.device, dtype=x.dtype)
        target_points = source_points + self._sample_offsets(
            source_points,
            mesh_width,
            mesh_height,
        )

        pad_h, pad_w = self._padding_amounts(height, width)
        padded = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
        canvas_h = height + 2 * pad_h
        canvas_w = width + 2 * pad_w

        source_pixel = self._to_pixel_coords(source_points, height, width)
        target_pixel = self._to_pixel_coords(target_points, height, width)
        pad_vector = source_points.new_tensor([pad_w, pad_h])

        source_canvas = self._to_normalized_coords(
            source_pixel + pad_vector,
            canvas_h,
            canvas_w,
        )
        target_canvas = self._to_normalized_coords(
            target_pixel + pad_vector,
            canvas_h,
            canvas_w,
        )

        tps = self._get_tps_grid(canvas_h, canvas_w, x.dtype)
        sampling_grid = tps(source_canvas[None, ...], target_canvas[None, ...])
        sampling_grid = sampling_grid.repeat(batch_size, 1, 1, 1)

        warped_canvas = F.grid_sample(
            padded,
            sampling_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return warped_canvas[:, :, pad_h : pad_h + height, pad_w : pad_w + width]

    def brightness(self, x):
        factor = torch.rand(1, device=x.device, dtype=x.dtype) * 2.0
        return torch.clamp(x * factor, 0.0, 1.0)

    def channel_noise(self, x, noise_std=0.05):
        noise = torch.randn_like(x) * noise_std
        return torch.clamp(x + noise, 0.0, 1.0)

    def _appearance_transforms(self):
        if not self.use_appearance:
            return []

        transforms = [
            ("brightness", self.brightness),
            ("channel_noise", self.channel_noise),
        ]

        if self.exclude_transform is None:
            return [fn for _, fn in transforms]

        excluded = {name.strip() for name in str(self.exclude_transform).split(",")}
        if "all" in excluded or "appearance" in excluded:
            return []
        if "noise" in excluded:
            excluded.add("channel_noise")

        return [fn for name, fn in transforms if name not in excluded]

    def apply_appearance(self, x):
        transforms = self._appearance_transforms()
        if not transforms:
            return x
        return random.choice(transforms)(x)

    def forward(self, data, label, **kwargs):
        if self.targeted:
            assert len(label) == 2
            label = label[1]

        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        delta = self.init_delta(data)
        momentum = 0

        for epoch_idx in range(self.epoch):
            if self.seed is not None:
                current_seed = self.seed + epoch_idx
                torch.manual_seed(current_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(current_seed)
                random.seed(current_seed)

            grads = torch.zeros_like(delta)

            for _ in range(self.num_warping):
                adv = data + delta
                transformed = self.apply_appearance(self.elastic_warp(adv))
                logits = self.get_logits(transformed)
                loss = self.get_loss(logits, label)
                grads += torch.autograd.grad(
                    loss,
                    delta,
                    retain_graph=False,
                    create_graph=False,
                )[0]

            grads = grads / float(self.num_warping)
            momentum = self.get_momentum(grads, momentum)
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()