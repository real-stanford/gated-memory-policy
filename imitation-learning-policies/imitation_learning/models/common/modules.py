import re

import torch
import torch.nn as nn


class Projector(nn.Module):
    def __init__(self, projector_type: str, in_dim: int, out_dim: int):
        super().__init__()

        self.projector_type: str = projector_type
        self.in_dim: int = in_dim
        self.out_dim: int = out_dim
        self.model: nn.Module

        if projector_type == "linear":
            self.model = nn.Linear(in_dim, out_dim)

        else:
            mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules: list[nn.Module] = [nn.Linear(in_dim, out_dim)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU(approximate="tanh"))
                    modules.append(nn.Linear(out_dim, out_dim))
                self.model = nn.Sequential(*modules)

            else:
                raise ValueError(f"Unknown projector type: {projector_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, ..., in_dim)
        """
        return self.model(x)  # (batch_size, ..., out_dim)
