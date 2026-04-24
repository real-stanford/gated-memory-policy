from dataclasses import asdict
import json
from typing import Any

import numpy as np
import torch
from torch import nn

from imitation_learning.common.dataclasses import DataMeta


class SingleFieldLinearNormalizer(nn.Module):
    def __init__(self, meta: DataMeta):
        super().__init__()
        self.meta: DataMeta = meta
        self.scale: nn.Parameter
        self.offset: nn.Parameter
        self.normalizer_type: str = meta.normalizer

    def fit(self, x: torch.Tensor):
        raise NotImplementedError()

    def from_dict(
        self,
        state_dict: dict[str, torch.Tensor],
    ):
        if self.normalizer_type == "identity":
            return

        state_dict_tensor: dict[str, torch.Tensor] = {}
        for key, val in state_dict.items():
            if key not in ["scale", "offset", "min", "max"]:
                continue

            if isinstance(val, torch.Tensor):
                state_dict_tensor[key] = val
            elif isinstance(val, np.ndarray):
                state_dict_tensor[key] = torch.from_numpy(val)
            elif isinstance(val, list):
                state_dict_tensor[key] = torch.Tensor(val)
            else:
                raise ValueError(f"Unknown type {type(val)} for {key}")


        if "min" in state_dict_tensor and "max" in state_dict_tensor:
            # Map [min, max] to [-1, 1]
            # normalize: (x-offset) / scale
            state_dict_tensor["scale"] = (state_dict_tensor["max"] - state_dict_tensor["min"]) / 2 + 1e-7
            state_dict_tensor["offset"] = (state_dict_tensor["max"] + state_dict_tensor["min"]) / 2
            del state_dict_tensor["min"]
            del state_dict_tensor["max"]

        
        # Set scale to 1 if the original scale is too small
        if hasattr(self, "skip_threshold"):
            scale_too_small_mask = state_dict_tensor["scale"] < self.skip_threshold
            state_dict_tensor["scale"][scale_too_small_mask] = 1.0

        keys = ["scale", "offset"]

        for key in keys:
            val = state_dict_tensor[key]
            assert (
                key in state_dict_tensor
            ), f"State dict must contain '{key}' key for {self.meta.name}"

            if self.normalizer_type in ["range", "normal", "range_clip"]:
                assert (
                    val.shape == self.meta.shape
                ), f"{key} must have the same shape as the data {self.meta.shape} for range normalizer {self.meta.name}"
            else:
                raise ValueError(
                    f"Unknown normalizer {self.normalizer_type} for {self.meta.name}. Valid normalizers are 'identity', 'range', 'normal', 'range_clip'."
                )

            setattr(self, key, nn.Parameter(val))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def as_dict(self, data_class: str) -> dict[str, Any]:
        assert (
            self.scale is not None and self.offset is not None
        ), f"Normalizer for {self.meta.name} is not initialized."
        if data_class == "numpy":
            return {
                "type": self.normalizer_type,
                "meta": asdict(self.meta),
                "scale": self.scale.detach().cpu().numpy(),
                "offset": self.offset.detach().cpu().numpy(),
            }
        elif data_class == "torch":
            return {
                "type": self.normalizer_type,
                "meta": asdict(self.meta),
                "scale": self.scale.detach().cpu(),
                "offset": self.offset.detach().cpu(),
            }
        elif data_class == "list":
            return {
                "type": self.normalizer_type,
                "meta": asdict(self.meta),
                "scale": self.scale.detach().cpu().tolist(),
                "offset": self.offset.detach().cpu().tolist(),
            }
        else:
            raise ValueError(
                f"Unknown data type {data_class} for normalizer {self.meta.name}. Valid types are 'numpy', 'torch', and 'list'."
            )

    def _check_input_shape(self, x: torch.Tensor):
        data_dim = len(self.meta.shape)
        assert (
            x.shape[-data_dim:] == self.meta.shape
        ), f"The last {data_dim} dimensions of {self.meta.name} (shape {x.shape}) must match {self.meta.shape} from meta data"


class IdentityNormalizer(SingleFieldLinearNormalizer):
    def __init__(self, meta: DataMeta):
        super().__init__(meta)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.offset = nn.Parameter(torch.tensor(0.0))

    def fit(self, x: torch.Tensor):
        pass

    def load(self, state_dict: dict[str, torch.Tensor]):
        pass

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x


class RangeNormalizer(SingleFieldLinearNormalizer):
    """
    Normalize data to be between -1 and 1.
    """

    def __init__(self, meta: DataMeta, skip_threshold: float = 1e-2):
        super().__init__(meta)
        self.scale = nn.Parameter(torch.nan * torch.ones(meta.shape))
        self.offset = nn.Parameter(torch.nan * torch.ones(meta.shape))
        self.skip_threshold = skip_threshold

    def fit(self, x: torch.Tensor):
        """
        x: (traj_num, *shape)
        """
        self._check_input_shape(x)
        x = x.clone().detach().reshape(-1, *self.meta.shape)
        min_val = x.min(dim=0).values
        max_val = x.max(dim=0).values
        scale = nn.Parameter((max_val - min_val) / 2 + 1e-7)
        self.scale[:] = 1.0
        max_abs_val = torch.max(torch.abs(x), dim=0).values
        ratio = scale / max_abs_val
        normalize_mask = (max_abs_val > self.skip_threshold) & (ratio > self.skip_threshold)
        print(f"{self.meta.name}: Normalize mask {normalize_mask}")
        self.scale[normalize_mask] = scale[normalize_mask]
        self.offset = nn.Parameter((max_val + min_val) / 2)
        assert tuple(self.scale.shape) == tuple(self.offset.shape) == self.meta.shape

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            not self.scale.isnan().any() and not self.offset.isnan().any()
        ), f"Normalizer for {self.meta.name} is not initialized"
        self._check_input_shape(x)
        return (x - self.offset) / self.scale

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            not self.scale.isnan().any() and not self.offset.isnan().any()
        ), f"Normalizer for {self.meta.name} is not initialized"
        self._check_input_shape(x)
        return x * self.scale + self.offset


class ClampedRangeNormalizer(SingleFieldLinearNormalizer):
    """
    Normalize data to be between -1 and 1, but clip the data if the normalized value is out of range.
    """

    def __init__(self, meta: DataMeta):
        super().__init__(meta)
        self.scale = nn.Parameter(torch.nan * torch.ones(meta.shape))
        self.offset = nn.Parameter(torch.nan * torch.ones(meta.shape))

    def fit(self, x: torch.Tensor):
        raise NotImplementedError("Please manually assign the scale and offset parameters after calculating the stats")
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            not self.scale.isnan().any() and not self.offset.isnan().any()
        ), f"Normalizer for {self.meta.name} is not initialized"
        self._check_input_shape(x)
        return torch.clamp((x - self.offset) / self.scale, -1, 1)

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            not self.scale.isnan().any() and not self.offset.isnan().any()
        ), f"Normalizer for {self.meta.name} is not initialized"
        self._check_input_shape(x)
        return x * self.scale + self.offset


class NormalNormalizer(RangeNormalizer):
    """
    Normalize data distribution to be N(0, 1).
    """

    def fit(self, x: torch.Tensor):
        print(f"Fitting normalizer for {self.meta.name}")
        self._check_input_shape(x)
        x = x.clone().detach().reshape(-1, *self.meta.shape)
        mean = x.mean(dim=0)
        std = x.std(dim=0)
        self.scale = nn.Parameter(std)
        self.offset = nn.Parameter(mean)


class FixedNormalizer(nn.Module):
    """
    Normalizer that is fixed after fitting. Not trainable.
    """

    @torch.no_grad()
    def __init__(
        self,
        data_meta: dict[str, DataMeta] | None = None,
        state_dict_with_meta: dict[str, dict[str, Any]] | None = None,
    ):
        super().__init__()
        if data_meta is not None:
            assert state_dict_with_meta is None, "Should not provide both data_meta and state_dict_with_meta"
            self.data_meta: dict[str, DataMeta] = data_meta
        elif state_dict_with_meta is not None:
            self.data_meta: dict[str, DataMeta] = {}
            # self.data_meta = {name: DataMeta(**data["meta"]) for name, data in state_dict_with_meta.items()}
            for name, data in state_dict_with_meta.items():
                if data["length"] == 0:
                    continue
                self.data_meta[name] = DataMeta(**data["meta"])
        else:
            raise ValueError("Either data_meta or state_dict_with_meta must be provided")

        self.normalizers: nn.ModuleDict = nn.ModuleDict()
        for meta in self.data_meta.values():
            if meta.normalizer == "identity":
                self.normalizers[meta.name] = IdentityNormalizer(meta)
            elif meta.normalizer == "range":
                self.normalizers[meta.name] = RangeNormalizer(meta)
            elif meta.normalizer == "normal":
                self.normalizers[meta.name] = NormalNormalizer(meta)
            elif meta.normalizer == "range_clip":
                self.normalizers[meta.name] = ClampedRangeNormalizer(meta)
            else:
                raise ValueError(
                    f"Unknown normalizer {meta.normalizer} for {meta.name}"
                )
        
        if state_dict_with_meta is not None:
            self.from_dict(state_dict_with_meta)

    @torch.no_grad()
    def fit_normalizer(self, data_dict: dict[str, torch.Tensor]):
        for meta in self.data_meta.values():
            if meta.normalizer not in ["range", "normal", "range_clip"]:
                continue
            normalizer = self.normalizers[meta.name]
            assert isinstance(normalizer, SingleFieldLinearNormalizer)
            normalizer.fit(data_dict[meta.name])

    @torch.no_grad()
    def from_dict(
        self,
        state_dict: dict[str, dict[str, torch.Tensor]],
    ):
        for meta in self.data_meta.values():
            if meta.normalizer == "identity":
                continue
            assert (
                meta.name in state_dict
            ), f"State dict for {meta.name} not found when loading normalizer"
            normalizer = self.normalizers[meta.name]
            assert isinstance(normalizer, SingleFieldLinearNormalizer)
            normalizer.from_dict(state_dict[meta.name])

    @torch.no_grad()
    def normalize(self, data_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for name, data in data_dict.items():
            if name not in self.normalizers:
                continue
            normalizer = self.normalizers[name]
            assert isinstance(normalizer, SingleFieldLinearNormalizer)
            data_dict[name] = normalizer.normalize(data)

        return data_dict

    @torch.no_grad()
    def unnormalize(
        self, data_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        for name, data in data_dict.items():
            if name not in self.normalizers:
                continue
            normalizer = self.normalizers[name]
            assert isinstance(normalizer, SingleFieldLinearNormalizer)
            data_dict[name] = normalizer.unnormalize(data)
        return data_dict

    @torch.no_grad()
    def as_dict(self, data_class: str) -> dict[str, dict[str, Any]]:
        """
        data_class: "numpy", "torch", "list"
        """
        state_dict = {}
        for name, normalizer in self.normalizers.items():
            if name not in state_dict:
                state_dict[name] = {}
            assert isinstance(normalizer, SingleFieldLinearNormalizer)
            state_dict[name] = normalizer.as_dict(data_class)
        return state_dict



def get_multi_task_normalizer(normalizer_json_paths: list[str], output_normalizer_json_path: str):
    """
    Will find the normalizer in each dataset and take the maximum range to get the final normalizer
    """

    # Collect all normalizer dictionaries
    normalizer_dicts = []
    for normalizer_json_path in normalizer_json_paths:
        with open(normalizer_json_path, "r") as f:
            normalizer_dicts.append(json.load(f))

    # Get field names from the first dataset
    field_names = list(normalizer_dicts[0].keys())

    # For each field, compute the global min and max across all datasets
    global_normalizer_dict = {}
    for field_name in field_names:
        # Get normalizer type and meta from first dataset
        first_field_dict = normalizer_dicts[0][field_name]
        normalizer_type = first_field_dict["type"]
        meta = first_field_dict["meta"]

        # Skip identity normalizers
        if normalizer_type == "identity":
            global_normalizer_dict[field_name] = first_field_dict
            continue

        # Collect scale and offset from all datasets to compute min/max
        # Recall: normalized_x = (x - offset) / scale
        # So: x = normalized_x * scale + offset
        # For normalized_x = -1: min_x = -scale + offset
        # For normalized_x = +1: max_x = scale + offset

        scales = []
        offsets = []
        for norm_dict in normalizer_dicts:
            scales.append(torch.tensor(norm_dict[field_name]["scale"]))
            offsets.append(torch.tensor(norm_dict[field_name]["offset"]))

        scales_tensor = torch.stack(scales)  # (num_datasets, *shape)
        offsets_tensor = torch.stack(offsets)  # (num_datasets, *shape)

        # Compute min and max for each dataset
        min_vals = offsets_tensor - scales_tensor  # When normalized value is -1
        max_vals = offsets_tensor + scales_tensor  # When normalized value is +1

        # Take global min and max across all datasets
        global_min = min_vals.min(dim=0).values
        global_max = max_vals.max(dim=0).values

        # Compute global scale and offset
        global_scale = (global_max - global_min) / 2 + 1e-7
        global_offset = (global_max + global_min) / 2

        global_normalizer_dict[field_name] = {
            "type": normalizer_type,
            "meta": meta,
            "scale": global_scale.tolist(),
            "offset": global_offset.tolist(),
        }

    # Save the final normalizer to a json file
    with open(output_normalizer_json_path, "w") as f:
        json.dump(global_normalizer_dict, f)
        print(f"Merged normalizer dict saved to {output_normalizer_json_path}")
            
