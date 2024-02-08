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

        if not self.state["step"]:
            self.state["step"] = 0
        else:
            self.state["step"] += 1

        for param_group in self.param_group:
            for param in param_group["params"]:
                if param.grad.data == None:
                    continue
                else:
                    gradients = param.grad.data

                if self.state["step"] == 1:
                    self.state["first_moment_estimate"] = torch.zeros_like(param.data)
                    self.state["second_moment_estimate"] = torch.zeros_like(param.data)
                first_moment_estimate = self.state["first_moment_estimate"]
                second_moment_estimate = self.state["second_moment_estimate"]

                first_moment_estimate.mul_(param_group["bias_m1"]).add(
                    gradients * (1 - param_group["bias_m1"])
                )
                second_moment_estimate.mul_(param_group["bias_m2"]).add(
                    gradients.pow(2) * (1.0 - param_group["bias_m2"])
                )
                param.data.add_(
                    (-param_group["stepsize"])
                    * first_moment_estimate.divide_(
                        second_moment_estimate.sqrt_() + param_group["epsilon"]
                    )
                )

        return loss
