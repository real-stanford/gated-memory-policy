import copy

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm


class EMAModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        inv_gamma: float,
        power: float,
        min_value: float,
        max_value: float,
    ):
        super().__init__()
        self.averaged_model: nn.Module = copy.deepcopy(model)
        self.averaged_model.requires_grad_(False)
        self.averaged_model.eval()

        self.inv_gamma: float = inv_gamma
        self.power: float = power
        self.min_value: float = min_value
        self.max_value: float = max_value

        self.optimization_step: nn.Parameter = nn.Parameter(
            torch.tensor(0), requires_grad=False
        )

    def _get_decay(self) -> float:
        """
        Compute the decay factor for the exponential moving average.
        """
        step = max(0, int(self.optimization_step.item()) - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power

        if step <= 0:
            return 0.0

        return min(self.max_value, max(self.min_value, value))

    @torch.no_grad()
    def step(self, new_model: nn.Module):
        decay = self._get_decay()

        for module, ema_module in zip(
            new_model.modules(), self.averaged_model.modules()
        ):
            for param, ema_param in zip(
                module.parameters(recurse=False), ema_module.parameters(recurse=False)
            ):
                if isinstance(param, _BatchNorm):
                    ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)
                elif isinstance(param, nn.Parameter):
                    ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)
                else:
                    ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

        self.optimization_step.add_(1)
