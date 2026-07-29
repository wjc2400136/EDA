"""Shared momentum-iterative optimizer used by four retained attacks."""

from .attack import Attack


class MomentumIterativeAttack(Attack):
    """Internal base class; it is not exposed as a standalone attack method."""

    def __init__(
        self,
        model_name,
        epsilon=16 / 255,
        alpha=1.6 / 255,
        epoch=10,
        decay=1.0,
        targeted=False,
        random_start=False,
        norm="linfty",
        loss="crossentropy",
        device=None,
        attack="momentum-iterative",
        **kwargs,
    ):
        super().__init__(
            attack,
            model_name,
            epsilon,
            targeted,
            random_start,
            norm,
            loss,
            device,
        )
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
