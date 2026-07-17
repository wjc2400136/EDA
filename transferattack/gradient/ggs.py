import torch
import torch.nn as nn
from ..utils import *
from ..attack import Attack


class GGS(Attack):
    """
    GGS (Gradient Guidance Sampling) Attack

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        num_neighbor (int): the number of neighbors.
        decay (float): the decay factor for momentum calculation.
        zeta (float): the noise scale multiplier.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model.

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, num_neighbor=20, decay=1., zeta=2

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/ggs/resnet50 --attack ggs --model=resnet50
        python main.py --input_dir ./path/to/data --output_dir adv_data/ggs/resnet50 --eval
    """

    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, num_neighbor=20, decay=1.,
                 zeta=2, targeted=False, random_start=False, norm='linfty', loss='crossentropy', device=None,
                 attack='GGS', **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.zeta = epsilon * zeta
        self.num_neighbor = num_neighbor

    def loss_function(self, loss):
        """
        Get the loss function
        """
        if loss == 'crossentropy':
            return nn.CrossEntropyLoss()
        else:
            raise Exception("Unsupported loss {}".format(loss))

    def forward(self, data, label, **kwargs):
        """
        The attack procedure for GGS

        Arguments:
            data: (N, C, H, W) tensor for input images
            labels: (N,) tensor for ground-truth labels if untargetd, otherwise targeted labels
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1]  # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)
        momentum = torch.zeros_like(delta)

        for _ in range(self.epoch):
            av_grad = torch.zeros_like(delta)
            prev_grad = None  # Initialize previous gradient for noise guidance

            for i in range(1, self.num_neighbor + 1):
                # Generate noise for neighbor sampling
                noise = torch.zeros_like(delta).uniform_(-self.zeta, self.zeta)

                # For subsequent neighbors, use gradient-guided noise
                if i > 1 and prev_grad is not None:
                    noise = noise.abs() * prev_grad.sign()
                noise = noise.clamp(-self.zeta, self.zeta)

                # Get neighbor point
                x_near = self.transform(data + delta + noise)

                # Calculate the output
                logits = self.get_logits(x_near)

                # Calculate the loss
                loss = self.get_loss(logits, label)

                # Calculate the gradient
                grad = self.get_grad(loss, x_near)

                # Store gradient for next iteration's noise guidance
                prev_grad = grad.detach()

                # Accumulate gradients
                av_grad += grad

            # Average gradients over neighbors
            av_grad = av_grad / self.num_neighbor

            # Calculate the momentum
            momentum = self.get_momentum(av_grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()