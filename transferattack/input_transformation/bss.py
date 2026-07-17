import torch
import torch.nn.functional as F
import random
from ..utils import *
from ..gradient.mifgsm import MIFGSM


class BSS(MIFGSM):
    """
    Block Stretch and Shrink (BSS) Attack
    'Enhancing Adversarial Transferability through Block Stretch and Shrink'

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        dp: minimum distance between segmentation points
        db: minimum distance from points to image boundaries
        r: range-to-center ratio for stretch/shrink factors
        M: number of segmentation points
        num_transforms: number of transformed inputs (number scale)
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/bss/resnet50 --attack bss --model=resnet50
    """

    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1.,
                 dp=40, db=35, r=1, M=2, num_transforms=25,
                 targeted=False, random_start=False, norm='linfty',
                 loss='crossentropy', device=None, attack='BSS', seed=42, **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.dp = dp  # minimum distance between points
        self.db = db  # minimum distance from points to boundaries
        self.r = r  # range-to-center ratio
        self.M = M  # number of segmentation points
        self.num_transforms = num_transforms  # number of transformed inputs
        self.seed = seed

    def constrained_random_points(self, h, w):
        """Sample M points with constraints on distance to boundaries and between points"""
        points = []
        max_attempts = 1000  # prevent infinite loop

        for _ in range(self.M):
            attempts = 0
            while attempts < max_attempts:
                # Sample a point within the valid region
                x = random.uniform(self.db, w - self.db)
                y = random.uniform(self.db, h - self.db)

                # Check constraints
                valid = True
                for px, py in points:
                    # 🔥 FIX: 独立检查两个维度，符合论文描述
                    if abs(x - px) < self.dp and abs(y - py) < self.dp:
                        valid = False
                        break

                if valid:
                    points.append((x, y))
                    break
                attempts += 1

            # If we couldn't find a valid point after max_attempts, use a random point
            if attempts == max_attempts:
                x = random.uniform(self.db, w - self.db)
                y = random.uniform(self.db, h - self.db)
                points.append((x, y))

        return points

    def transform_one_dim(self, x, points, dim):
        """Apply block stretch and shrink along one dimension (height or width)"""
        # Get the size of the dimension
        if dim == 'height':
            length = x.size(2)
            coords = [p[1] for p in points]  # y-coordinates
        else:  # width
            length = x.size(3)
            coords = [p[0] for p in points]  # x-coordinates

        # Sort coordinates and add boundaries
        coords.sort()
        segments = [0] + coords + [length]

        # Create blocks
        blocks = []
        original_lengths = []
        for i in range(len(segments) - 1):
            start = int(segments[i])
            end = int(segments[i + 1])

            # 🔥 FIX: 确保块的最小长度为1，防止空块
            if end <= start:
                end = start + 1

            if dim == 'height':
                block = x[:, :, start:end, :]
            else:  # width
                block = x[:, :, :, start:end]
            blocks.append(block)
            original_lengths.append(end - start)

        # Sample stretch/shrink factors
        s = [random.uniform((1 - self.r / 2) / 2, (1 + self.r / 2) / 2) for _ in range(len(blocks))]
        total = sum(s)
        weights = [si / total for si in s]

        # Calculate new lengths
        new_lengths = [round(l * w) for l, w in zip(original_lengths, weights)]

        # 🔥 FIX: 将新长度截断到最小值1，符合论文描述
        new_lengths = [max(1, nl) for nl in new_lengths]

        # Adjust lengths to match the original dimension
        diff = length - sum(new_lengths)
        if diff > 0:  # Need to add pixels
            for _ in range(diff):
                # Find the block with the largest length
                idx = new_lengths.index(max(new_lengths))
                new_lengths[idx] += 1
        elif diff < 0:  # Need to remove pixels
            for _ in range(-diff):
                # Find the block with the smallest length > 1
                min_idx = None
                for i, l in enumerate(new_lengths):
                    if l > 1:
                        if min_idx is None or l < new_lengths[min_idx]:
                            min_idx = i
                if min_idx is not None:
                    new_lengths[min_idx] -= 1

        # Apply stretch/shrink to each block
        transformed_blocks = []
        for block, new_len in zip(blocks, new_lengths):
            if dim == 'height':
                # Interpolate along height
                transformed_block = F.interpolate(
                    block, size=(new_len, block.size(3)), mode='bilinear', align_corners=False)
            else:  # width
                # Interpolate along width
                transformed_block = F.interpolate(
                    block, size=(block.size(2), new_len), mode='bilinear', align_corners=False)
            transformed_blocks.append(transformed_block)

        # Concatenate blocks
        if dim == 'height':
            transformed_x = torch.cat(transformed_blocks, dim=2)
        else:  # width
            transformed_x = torch.cat(transformed_blocks, dim=3)

        return transformed_x

    def bss_transform(self, x):
        """Apply the complete BSS transformation to an image"""
        h, w = x.size(2), x.size(3)

        # Sample constrained random points
        points = self.constrained_random_points(h, w)

        # Randomly choose the order of dimensions
        if random.random() < 0.5:
            # First height then width
            transformed_x = self.transform_one_dim(x, points, 'height')
            transformed_x = self.transform_one_dim(transformed_x, points, 'width')
        else:
            # First width then height
            transformed_x = self.transform_one_dim(x, points, 'width')
            transformed_x = self.transform_one_dim(transformed_x, points, 'height')

        return transformed_x

    def forward(self, data, label, **kwargs):
        # 🔥 FIX: 设置随机种子以确保实验可复现
        random.seed(self.seed)

        if self.targeted:
            assert len(label) == 2
            label = label[1]  # the second element is the targeted label tensor

        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        momentum = 0
        for _ in range(self.epoch):
            grads = 0
            for _ in range(self.num_transforms):
                # Apply BSS transformation
                transformed_data = self.bss_transform(data + delta)

                # Get the logits
                logits = self.get_logits(transformed_data)

                # Calculate the loss
                loss = self.get_loss(logits, label)

                # Calculate the gradients
                grad = self.get_grad(loss, delta)
                grads += grad

            grads /= self.num_transforms

            # Calculate the momentum
            momentum = self.get_momentum(grads, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()