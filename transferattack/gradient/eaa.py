import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import *
from ..attack import Attack


class EAA(Attack):
    """
    EAA (Evidence-Aware Attack)

    This attack reverses the logic of Evidential Deep Learning (EDL) used for defense.
    Instead of minimizing uncertainty, it aims to maximize the uncertainty of the true class
    and minimize the evidence for the true label, thereby crafting stronger adversarial examples.

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation (MI-FGSM style).
        lambd (float): the weight for balancing uncertainty maximization.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function (using 'evidence' logic).
        device (torch.device): the device for data. If it is None, the device would be same as model.

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/eaa/resnet50 --attack eaa --model=resnet50
    """

    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1.0,
                 lambd=1.0, targeted=False, random_start=False, norm='linfty', loss='evidence', device=None,
                 attack='EAA', **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.lambd = lambd

    def loss_function(self, loss):
        """
        Overwrite the loss function getter to support 'evidence'.
        Since we calculate the loss manually in forward(), we return None here
        to avoid errors in the base class initialization.
        """
        if loss == 'evidence':
            return None
        elif loss == 'crossentropy':
            return nn.CrossEntropyLoss()
        else:
            raise Exception("Unsupported loss {}".format(loss))

    def forward(self, data, label, **kwargs):
        """
        The attack procedure for EAA

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
            # CRITICAL FIX:
            # We must explicitly enable gradients for the input tensor (data + delta)
            # so that the computational graph is built during the forward pass.
            adv_images = data + delta
            adv_images.requires_grad_()

            # 1. Get logits from the model
            # The graph now flows: adv_images -> logits -> evidence -> loss
            logits = self.get_logits(adv_images)

            # 2. Convert Logits to Evidence using SoftPlus (Non-negative)
            evidence = F.softplus(logits)

            # 3. Calculate Dirichlet parameters (Alpha)
            alpha = evidence + 1

            # 4. Calculate Strength (S) - Sum of Alphas for each sample
            strength = torch.sum(alpha, dim=1, keepdim=True)

            # 5. Calculate the Evidence Loss

            # Ensure y_one_hot does not require gradient
            y_one_hot = torch.zeros(logits.size(), device=self.device, dtype=logits.dtype)
            y_one_hot.scatter_(1, label.unsqueeze(1), 1)

            # Using log approximation for digamma
            digamma_S = torch.log(strength)
            digamma_alpha = torch.log(alpha)

            # Attack Loss: Minimize evidence for true label
            loss_edl = - torch.sum(y_one_hot * (digamma_S - digamma_alpha), dim=1)
            loss = torch.mean(loss_edl)

            # 6. Calculate the gradient
            # We compute the gradient of loss w.r.t adv_images
            grad = self.get_grad(loss, adv_images)

            # 7. Calculate the momentum (MI-FGSM style)
            momentum = self.get_momentum(grad, momentum)

            # 8. Update adversarial perturbation
            # Note: update_delta usually returns a detached tensor.
            # We will re-enable requires_grad in the next loop iteration.
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()