import torch
import torch.nn as nn


class BaseDenoisingNetwork(nn.Module):
    def __init__(
        self,
        name: str,
        action_dim: int,
        action_token_num: int,
        global_cond_dim: int,
        global_cond_token_num: int,
        seed: int,
        local_cond_dim: int = 0,
        local_cond_token_num: int = 0,
        **unused_kwargs,
    ):
        if unused_kwargs:
            print(f"BaseDenoisingNetwork unused kwargs: {unused_kwargs}")
        super().__init__()
        self.seed: int = seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

        self.name: str = name
        self.action_dim: int = action_dim
        self.action_token_num: int = action_token_num
        self.global_cond_dim: int = global_cond_dim
        self.global_cond_token_num: int = global_cond_token_num
        self.local_cond_dim: int = local_cond_dim
        self.local_cond_token_num: int = local_cond_token_num
        print(
            f"BaseDenoisingNetwork: {self.name=}, \
                {self.action_dim=}, {self.action_token_num=}, {self.global_cond_dim=}, {self.global_cond_token_num=}, \
                {self.local_cond_dim=}, {self.local_cond_token_num=}"
        )

    def forward(
        self,
        data_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError()

    def reset(self):
        pass
