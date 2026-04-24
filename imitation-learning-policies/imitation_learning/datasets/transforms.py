from typing import Any, Union

from imitation_learning.utils.cv_util import draw_predefined_mask
import kornia.augmentation as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from imitation_learning.common.dataclasses import DataMeta


def _build_mask(
    h: int, w: int, mirror: bool = False, gripper: bool = False
) -> torch.Tensor:
    """Build a boolean mask (H, W) using UMI's draw_predefined_mask."""
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    draw_predefined_mask(img, color=(0, 0, 0), mirror=mirror, gripper=gripper, finger=False)
    mask = torch.from_numpy(img[:, :, 0] > 0)
    return mask


class MirrorMask(nn.Module):
    """Masks out the side mirror regions with black, applied with probability p."""

    def __init__(self, p: float = 1.0):
        super().__init__()
        self.p = p
        self._mask_cache: dict[tuple[int, int], torch.Tensor] = {}

    def _get_mask(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        key = (h, w)
        if key not in self._mask_cache or self._mask_cache[key].device != device:
            self._mask_cache[key] = _build_mask(h, w, mirror=True).to(device)
        return self._mask_cache[key]

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """data: (..., C, H, W)"""
        if torch.rand(1).item() > self.p:
            return data
        h, w = data.shape[-2], data.shape[-1]
        mask = self._get_mask(h, w, data.device)
        return data * mask


class GripperMask(nn.Module):
    """Masks out the gripper and finger regions with black, applied with probability p."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        self._mask_cache: dict[tuple[int, int], torch.Tensor] = {}

    def _get_mask(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        key = (h, w)
        if key not in self._mask_cache or self._mask_cache[key].device != device:
            self._mask_cache[key] = _build_mask(h, w, gripper=True).to(device)
        return self._mask_cache[key]

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """data: (..., C, H, W)"""
        if torch.rand(1).item() > self.p:
            return data
        h, w = data.shape[-2], data.shape[-1]
        mask = self._get_mask(h, w, data.device)
        return data * mask


class ResizeWithPadding(nn.Module):
    """Resize image preserving aspect ratio, then center-pad to target size.

    Input:  (..., C, H, W)
    Output: (..., C, new_H, new_W)
    """

    def __init__(self, size: list[int] | tuple[int, int], pad_value: float = 0.0):
        super().__init__()
        self.new_H = size[0]
        self.new_W = size[1]
        self.pad_value = pad_value

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        batch_shape = data.shape[:-3]
        C, H, W = data.shape[-3:]

        original_aspect = W / H
        new_aspect = self.new_W / self.new_H

        if original_aspect > new_aspect:
            # Wider than target: fit width, pad top/bottom
            resize_W = self.new_W
            resize_H = int(self.new_W / original_aspect)
        else:
            # Taller than target: fit height, pad left/right
            resize_H = self.new_H
            resize_W = int(self.new_H * original_aspect)

        flat = data.reshape(-1, C, H, W)
        resized = torch.nn.functional.interpolate(
            flat.float(), size=(resize_H, resize_W), mode="bilinear", align_corners=False
        ).to(data.dtype)
        resized = resized.reshape(*batch_shape, C, resize_H, resize_W)

        pad_top = (self.new_H - resize_H) // 2
        pad_bottom = self.new_H - resize_H - pad_top
        pad_left = (self.new_W - resize_W) // 2
        pad_right = self.new_W - resize_W - pad_left

        # F.pad takes (left, right, top, bottom) for 4D-last-two-dims
        return torch.nn.functional.pad(
            resized, (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant", value=self.pad_value,
        )


class RandomGaussianNoise(nn.Module):
    def __init__(self, std: float | list[float], p: float):
        super().__init__()
        self.std: float | list[float] = std
        self.p: float = p

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.p:
            if isinstance(self.std, float):
                return data + torch.randn_like(data) * self.std
            elif isinstance(self.std, list):
                return data + torch.randn_like(data) * torch.tensor(self.std)
            else:
                raise ValueError(f"Invalid std type: {type(self.std)}")
        else:
            return data


_CUSTOM_IMAGE_TRANSFORMS = {
    "MirrorMask": MirrorMask,
    "GripperMask": GripperMask,
    "ResizeWithPadding": ResizeWithPadding,
}


class BaseTransforms:
    def __init__(
        self, data_meta: dict[str, DataMeta], apply_image_augmentation_in_cpu: bool, seed: int
    ):
        self.transforms: dict[str, Union[K.VideoSequential, nn.Sequential]] = {}
        self.pre_transforms: dict[str, nn.Sequential] = {}
        self.data_meta: dict[str, DataMeta] = data_meta
        self.apply_image_augmentation_in_cpu: bool = apply_image_augmentation_in_cpu
        self.seed: int = seed
        torch.manual_seed(self.seed)
        for entry_meta in data_meta.values():
            kornia_list = []
            custom_list = []
            for aug_cfg in entry_meta.augmentation:
                aug_name = aug_cfg["name"]
                aug_cfg.pop("name")
                if entry_meta.data_type == "image":
                    if aug_name in _CUSTOM_IMAGE_TRANSFORMS:
                        custom_list.append(
                            _CUSTOM_IMAGE_TRANSFORMS[aug_name](**aug_cfg)
                        )
                    elif aug_name in K.__dict__:
                        kornia_list.append(K.__dict__[aug_name](**aug_cfg))
                    else:
                        raise ValueError(
                            f"Augmentation {aug_name} not found in kornia.augmentation or custom transforms."
                        )
                elif entry_meta.data_type == "low_dim":
                    if aug_name == "RandomGaussianNoise":
                        kornia_list.append(RandomGaussianNoise(**aug_cfg))
                    else:
                        raise ValueError(
                            f"Augmentation {aug_name} not found in low dim transforms. Please implement your own augmentation method."
                        )

            if len(custom_list) > 0:
                self.pre_transforms[entry_meta.name] = nn.Sequential(*custom_list)
            if len(kornia_list) > 0:
                if entry_meta.data_type == "image":
                    self.transforms[entry_meta.name] = K.VideoSequential(
                        *kornia_list
                    )
                else:
                    self.transforms[entry_meta.name] = nn.Sequential(*kornia_list)

    def to(self, device: Union[torch.device, str]):
        for transform in self.transforms.values():
            transform.to(device)
        for transform in self.pre_transforms.values():
            transform.to(device)

    def apply(
        self, data_dict: dict[str, Any], consistent_on_batch: bool = False
    ) -> dict[str, torch.Tensor]:
        for name, data in data_dict.items():
            if (
                not self.apply_image_augmentation_in_cpu
                and self.data_meta[name].data_type == "image"
            ):
                continue
            if isinstance(data, dict):
                data_dict[name] = self.apply(data, consistent_on_batch)
            elif isinstance(data, torch.Tensor):
                if name in self.pre_transforms:
                    data = self.pre_transforms[name](data)
                    data_dict[name] = data
                if name in self.transforms:
                    batch_size, traj_len, *shape = data.shape
                    if consistent_on_batch:
                        data = data.reshape(1, batch_size * traj_len, *shape)

                    data_dim_num = len(self.data_meta[name].shape)
                    new_data_dim_num = len(data.shape)
                    squeeze_data = False
                    if new_data_dim_num - data_dim_num == 1:
                        data = data.unsqueeze(0)
                        squeeze_data = True
                    elif new_data_dim_num - data_dim_num != 2:
                        raise ValueError(
                            f"Data {name} has more than 2 additional dimensions: {data.shape}. Currently only support (traj_len, *shape) or (batch_size, traj_len, *shape)."
                        )
                    try:
                        data = self.transforms[name](data)
                    except Exception as e:
                        print(
                            f"Error applying transform {name} to data {data.shape}: {e}"
                        )
                        raise e
                    if squeeze_data:
                        data = data.squeeze(0)
                    if consistent_on_batch:
                        data = rearrange(data, "1 (b t) ... -> b t ...", b=batch_size, t=traj_len)
                        # data = data.reshape(batch_size, traj_len, *shape)
                    data_dict[name] = data

            else:
                raise ValueError(f"Unknown data type {type(data)} for {name}")
        return data_dict
