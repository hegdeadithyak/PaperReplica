from typing import Any, Dict
import torch
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT


class Adam(Optimizer):
    def __init__(
        self,
        params,
        stepsize,
        bias_m1=0.01,
        bias_m2=0.999,
        epsilon=10e-8,
        bias_correction=True,
    ) -> None:
        if stepsize < 0:
            raise ValueError("Invalid Step Size {[]}").format(stepsize)
        if bias_m1 < 0 or bias_m2 < 0 and bias_correction:
            raise ValueError(
                "Invalid Bias Parameters {[]}{[]}.Check them again"
            ).format(bias_m1, bias_m2)

        DEFAULTS = dict(
            stepsize=stepsize,
            bias_m1=bias_m1,
            bias_m2=bias_m2,
            bias_correction=bias_correction,
        )
        super(Adam).__init__(params, DEFAULTS)

        def Step(self, closure=None):
            loss = None

            loss = closure() if closure != None else loss
