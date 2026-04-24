
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from imitation_learning.common.datatypes import batch_type
from imitation_learning.models.encoders.image_encoders import SharedModelManager
from imitation_learning.policies.base_policy import BasePolicy
from robot_utils.data_utils import dict_apply
from robot_utils.torch_utils import aggregate_batch, split_batch


class BCRNNPolicy(BasePolicy):
    """
    Behavioral Cloning with an RNN policy for multi-trajectory data.

    Encodes observations via global/local condition encoders, processes the
    sequence through an RNN (LSTM or GRU), and decodes actions via MLP + action_decoder.

    Trajectories are processed sequentially so that the RNN hidden state from
    trajectory i feeds into trajectory i+1.

    Expected input shapes:
        Training:  (batch_size, traj_num, seq_len, ...)
        Inference: (batch_size, traj_num, 1, ...)
    """

    def __init__(
        self,
        shared_model_manager: SharedModelManager,
        detach_rnn_state: bool,
        rnn_hidden_dim: int,
        rnn_num_layers: int,
        rnn_type: str,
        mlp_layer_dims: list[int],
        open_loop: bool,
        action_no_error_range: tuple[int, int],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.shared_model_manager = shared_model_manager
        self.action_no_error_range: tuple[int, ...] = tuple(action_no_error_range)
        self.detach_rnn_state = detach_rnn_state
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_num_layers = rnn_num_layers
        self.rnn_type = rnn_type
        self._rnn_is_open_loop = open_loop

        # Compute input dimension from encoders
        # Both global (image) and local (proprio) are pooled to single vectors
        input_dim = self.global_cond_encoder.feature_dim
        if self.local_cond_encoder is not None:
            input_dim += self.local_cond_encoder.feature_dim

        # RNN
        rnn_cls = nn.LSTM if rnn_type == "LSTM" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            batch_first=True,
        )

        # MLP decoder head: RNN output -> full action chunk
        self._action_dim = self.action_decoder.latent_dim
        self._action_length = self.action_decoder.traj_len
        output_dim = self._action_length * self._action_dim
        layers: list[nn.Module] = []
        prev_dim = rnn_hidden_dim
        for dim in mlp_layer_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp_head = nn.Sequential(*layers)

        # Inference state — per-episode dictionaries keyed by episode_idx
        self._rnn_hidden_state_dict: dict[int, tuple[torch.Tensor, torch.Tensor] | torch.Tensor] = {}
        """episode_idx -> RNN hidden state for that episode"""
        self._rnn_counter_dict: dict[int, int] = {}
        """episode_idx -> step counter for that episode"""
        self._open_loop_obs_dict: dict[int, batch_type] = {}
        """episode_idx -> cached observation for open-loop inference"""

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(
            f"BCRNNPolicy trainable parameters: {trainable_params}, "
            f"total parameters: {total_params}"
        )

    # ================================ Helpers =================================

    def _encode_obs(self, normalized_batch: batch_type, load_multi_traj: bool) -> torch.Tensor:
        """
        Encode observations into a single feature vector (B, traj_len, input_dim).
        Image is pooled to traj_len token; proprio is pooled via mean across timesteps.
        This produces the same representation for both training and inference.

        Args:
            normalized_batch: keys with shape 
            (batch_size, traj_num, seq_len, ...) # Training with multi-trajectory loading
            or (batch_size, seq_len, ...) # Inference with single-trajectory loading

        Returns:
            obs_features: (batch_size, traj_len, 1, input_dim)
            or (batch_size, 1, input_dim)
        """
        batch_size = normalized_batch[self.global_cond_encoder.data_entry_names[0]].shape[0]
        global_obs = {
            k: normalized_batch[k] for k in self.global_cond_encoder.data_entry_names
        }
        if load_multi_traj:
            global_obs = dict_apply(global_obs, lambda x: einops.rearrange(x, "b t ... -> (b t) ..."))
        
        global_cond = self.global_cond_encoder.forward(global_obs)
        features = global_cond  # (B, 1, feature_dim)

        # Local condition encoding (proprio) — pool to single vector
        if self.local_cond_encoder is not None:
            local_obs = {
                k: normalized_batch[k]
                for k in self.local_cond_encoder.data_entry_names
            }
            if load_multi_traj:
                local_obs = dict_apply(local_obs, lambda x: einops.rearrange(x, "b t ... -> (b t) ..."))
            local_cond = self.local_cond_encoder.forward(local_obs)
            # print(f"features: {features.shape}, local_cond: {local_cond.shape}")
            features = torch.cat([features, local_cond], dim=-1)

        # assert features.shape[1] == 1, f"features.shape: {features.shape}, expected: (B, 1, input_dim)"
        if load_multi_traj:
            features = einops.rearrange(features, "(b t) ... -> b t ...", b=batch_size)
        return features  # (B, 1, input_dim) or (B, traj_num, 1, input_dim)

    def _get_rnn_init_state(self, batch_size: int) -> torch.Tensor:
        """Return zero-initialized hidden state for the RNN."""
        num_directions = 1
        shape = (self.rnn_num_layers * num_directions, batch_size, self.rnn_hidden_dim)
        if self.rnn_type == "LSTM":
            return (
                torch.zeros(*shape, device=self.device),
                torch.zeros(*shape, device=self.device),
            )
        else:
            return torch.zeros(*shape, device=self.device)

    def _forward_rnn(
        self,
        obs_features: torch.Tensor,
        rnn_init_state=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run the RNN on a single observation and predict a full action chunk.

        Args:
            obs_features: (B, 1, input_dim)
            rnn_init_state: optional initial hidden state

        Returns:
            actions: (B, action_length, action_dim)
            rnn_state: final hidden state
        """
        batch_size = obs_features.shape[0]
        if rnn_init_state is None:
            rnn_init_state = self._get_rnn_init_state(batch_size)

        rnn_out, rnn_state = self.rnn(obs_features, rnn_init_state)
        # rnn_out: (B, 1, rnn_hidden_dim)
        flat_actions = self.mlp_head(rnn_out)  # (B, 1, action_length * action_dim)
        flat_actions = torch.tanh(flat_actions)
        # Reshape to (B, action_length, action_dim)
        actions = flat_actions.reshape(batch_size, self._action_length, self._action_dim)
        return actions, rnn_state

    # ================================ Training =================================

    def compute_loss(self, normalized_batch: batch_type) -> batch_type:
        """
        Compute BC loss (MSE between predicted and ground-truth actions).

        Processes trajectories sequentially through the RNN so that the hidden
        state from trajectory i is the initial state for trajectory i+1.

        normalized_batch:
            image keys: (batch_size, traj_num, seq_len, 3, H, W)
            proprio keys: (batch_size, traj_num, seq_len, proprio_dim)
            action keys: (batch_size, traj_num, seq_len, action_dim)
            "entire_traj_is_padding": (batch_size, traj_num)
        """
        self.shared_model_manager.clear_cache()

        action_key_names = self.action_decoder.data_entry_names
        global_cond_key_names = self.global_cond_encoder.data_entry_names
        batch_size = normalized_batch[global_cond_key_names[0]].shape[0]
        traj_num = normalized_batch[global_cond_key_names[0]].shape[1]

        # Encode GT actions: flatten (B, traj_num) -> (B*traj_num) for action_decoder,
        # matching diffusion policy's encode pattern
        gt_trajectory = einops.rearrange(
            self.action_decoder.encode(
                dict_apply(
                    {k: normalized_batch[k] for k in action_key_names},
                    lambda x: einops.rearrange(x, "b t ... -> (b t) ..."),
                )
            ),
            "(b t) ... -> b t ...",
            b=batch_size,
        )  # (batch_size, traj_num, seq_len, action_dim)

        rnn_state = self._get_rnn_init_state(batch_size)
        all_pred_actions = []

        obs_features = self._encode_obs(normalized_batch, load_multi_traj=True)  # (B, traj_num, 1, input_dim)

        for traj_idx in range(traj_num):
            # Extract trajectory traj_idx: (batch_size, seq_len, ...)
            traj_batch = {}
            for k, v in normalized_batch.items():
                if isinstance(v, torch.Tensor) and v.dim() >= 3 and k not in ("entire_traj_is_padding", "action_is_error", "action_is_critical"):
                    traj_batch[k] = v[:, traj_idx]
                elif k not in ("entire_traj_is_padding", "action_is_error", "action_is_critical"):
                    traj_batch[k] = v

            # obs_features = self._encode_obs(traj_batch)  # (B, 1, input_dim)


            if self.detach_rnn_state:
                # RNN hidden state carries from previous trajectory (detached to avoid
                # backprop through the entire trajectory history)
                if self.rnn_type == "LSTM":
                    detached_state = (rnn_state[0].detach(), rnn_state[1].detach())
                else:
                    detached_state = rnn_state.detach()
                # Single RNN call: 1 obs → action_length actions
                pred_actions, rnn_state = self._forward_rnn(
                    obs_features[:, traj_idx], rnn_init_state=detached_state
                )  # (B, action_length, action_dim)
            else:
                pred_actions, rnn_state = self._forward_rnn(
                    obs_features[:, traj_idx], rnn_init_state=rnn_state
                )  # (B, action_length, action_dim)

            all_pred_actions.append(pred_actions)

        # Stack: (batch_size, traj_num, seq_len, action_dim)
        all_pred_actions = torch.stack(all_pred_actions, dim=1)

        action_loss = F.mse_loss(all_pred_actions, gt_trajectory, reduction="none")

        # Reduce over all dims except batch and traj_num
        action_loss = einops.reduce(
            action_loss, "b t ... -> b t", "mean"
        )  # (batch_size, traj_num)

        # Mask out padding trajectories
        if "entire_traj_is_padding" in normalized_batch:
            action_loss = action_loss * (~normalized_batch["entire_traj_is_padding"])

        # Error trajectory is only for debugging
        # Mask out error trajectories (matching diffusion policy)
        if "action_is_error" in normalized_batch:
            action_traj_length = gt_trajectory.shape[2]
            traj_error_mask = torch.zeros(action_traj_length, device=self.device)
            traj_error_mask[
                self.action_no_error_range[0] : self.action_no_error_range[1]
            ] = 1
            traj_is_error = (
                normalized_batch["action_is_error"] * traj_error_mask[None, None, :]
            )  # (batch_size, traj_num, traj_length)
            traj_is_error = einops.reduce(
                traj_is_error, "b t ... -> b t", "any"
            )  # (batch_size, traj_num)
            action_loss = action_loss * (~traj_is_error)

        # Track critical action loss separately (matching diffusion policy)
        loss: batch_type = {}
        critical_action_loss = None
        if "action_is_critical" in normalized_batch:
            single_action_is_critical = normalized_batch["action_is_critical"]
            traj_action_is_critical = torch.any(
                single_action_is_critical, dim=2
            ).squeeze(-1)  # (batch_size, traj_num)
            critical_action_loss = action_loss * traj_action_is_critical

            if critical_action_loss.sum() > 0:
                loss["critical_action"] = (
                    critical_action_loss.sum() / (critical_action_loss != 0).sum()
                )
            else:
                loss["critical_action"] = critical_action_loss.sum()

        # Average over valid trajectories
        if (action_loss != 0).sum() == 0:
            loss["action"] = action_loss.sum() / action_loss.numel()
        else:
            loss["action"] = action_loss.sum() / (action_loss != 0).sum()

        return loss

    # ================================ Inference =================================

    def predict_action(self, normalized_batch: batch_type) -> batch_type:
        """
        Single-trajectory input (real inference, one observation at a time):
            image keys: (batch_size, 1, 3, H, W)
            proprio keys: (batch_size, 1, proprio_dim)
            "episode_idx": (batch_size,)
            return: action dict (batch_size, action_length, action_dim)

        Multi-trajectory input (validation with multiple GT trajectories):
            image keys: (batch_size, traj_num, 1, 3, H, W)
            proprio keys: (batch_size, traj_num, 1, proprio_dim)
            "episode_idx": (batch_size, traj_num)
            return: action dict (batch_size, traj_num, action_length, action_dim)
        """
        meta = next(iter(self.global_cond_encoder.cond_meta.values()))
        input_shape = normalized_batch[meta.name].shape
        expected_shape = meta.shape

        if len(input_shape) - len(expected_shape) == 2:
            return self._predict_single_traj(normalized_batch)
        elif len(input_shape) - len(expected_shape) == 3:
            return self._predict_multi_traj(normalized_batch)
        else:
            raise ValueError(
                f"Unexpected input shape: {input_shape}, expected shape: {expected_shape}"
            )

    def _aggregate_rnn_hidden_states(
        self, episode_indices: torch.Tensor, batch_size: int
    ):
        """
        Assemble a batched RNN hidden state from per-episode dictionaries.
        Episodes not yet in the dict get zero-initialized states.
        """
        init_state = self._get_rnn_init_state(batch_size)

        if self.rnn_type == "LSTM":
            h, c = init_state
            for idx, ep_idx in enumerate(episode_indices):
                ep_key = int(ep_idx)
                if ep_key in self._rnn_hidden_state_dict:
                    ep_h, ep_c = self._rnn_hidden_state_dict[ep_key]
                    h[:, idx, :] = ep_h[:, 0, :]
                    c[:, idx, :] = ep_c[:, 0, :]
            return (h, c)
        else:
            h = init_state
            for idx, ep_idx in enumerate(episode_indices):
                ep_key = int(ep_idx)
                if ep_key in self._rnn_hidden_state_dict:
                    ep_h = self._rnn_hidden_state_dict[ep_key]
                    h[:, idx, :] = ep_h[:, 0, :]
            return h

    def _scatter_rnn_hidden_states(
        self, episode_indices: torch.Tensor, rnn_state
    ):
        """
        Store the batched RNN hidden state back into per-episode dictionaries.
        Each episode gets its own slice (with batch dim = 1).
        """
        for idx, ep_idx in enumerate(episode_indices):
            ep_key = int(ep_idx)
            if self.rnn_type == "LSTM":
                h, c = rnn_state
                self._rnn_hidden_state_dict[ep_key] = (
                    h[:, idx:idx+1, :].detach(),
                    c[:, idx:idx+1, :].detach(),
                )
            else:
                self._rnn_hidden_state_dict[ep_key] = rnn_state[:, idx:idx+1, :].detach()

    def _predict_single_traj(self, normalized_batch: batch_type) -> batch_type:
        """
        Single-observation inference. One RNN call produces the full action chunk.
        Matches training exactly. RNN hidden state is maintained per episode_idx.
        """
        batch_size = next(iter(normalized_batch.values())).shape[0]
        episode_indices = normalized_batch["episode_idx"]  # (batch_size,)

        # # Reset hidden state every rnn_horizon steps (robomimic BC-RNN pattern)
        # for idx, ep_idx in enumerate(episode_indices):
        #     ep_key = int(ep_idx)
        #     counter = self._rnn_counter_dict.get(ep_key, 0)
        #     if counter % self._rnn_horizon == 0:
        #         # Clear hidden state so it gets zero-initialized in _aggregate
        #         self._rnn_hidden_state_dict.pop(ep_key, None)
        #         if self._rnn_is_open_loop:
        #             # Cache observation for open-loop inference
        #             self._open_loop_obs_dict[ep_key] = {
        #                 k: v[idx:idx+1].clone().detach()
        #                 for k, v in normalized_batch.items()
        #                 if k != "episode_idx"
        #             }

        # Aggregate per-episode hidden states into a batched hidden state
        rnn_init_state = self._aggregate_rnn_hidden_states(episode_indices, batch_size)

        obs_to_use = normalized_batch
        if self._rnn_is_open_loop:
            # Build per-element obs: use cached obs if available, else current obs
            obs_to_use = {}
            for k in normalized_batch:
                if k == "episode_idx":
                    continue
                parts = []
                for idx, ep_idx in enumerate(episode_indices):
                    ep_key = int(ep_idx)
                    if ep_key in self._open_loop_obs_dict:
                        parts.append(self._open_loop_obs_dict[ep_key][k][0])
                    else:
                        parts.append(normalized_batch[k][idx])
                obs_to_use[k] = torch.stack(parts)

        obs_features = self._encode_obs(obs_to_use, load_multi_traj=False)  # (B, 1, input_dim)

        # Single RNN call: 1 obs → action_length actions
        pred_actions, new_rnn_state = self._forward_rnn(
            obs_features, rnn_init_state=rnn_init_state
        )  # (B, action_length, action_dim)

        # Store per-episode hidden states back into the dictionary
        self._scatter_rnn_hidden_states(episode_indices, new_rnn_state)

        # Update per-episode counters
        for ep_idx in episode_indices:
            ep_key = int(ep_idx)
            self._rnn_counter_dict[ep_key] = self._rnn_counter_dict.get(ep_key, 0) + 1

        return self.action_decoder.forward(pred_actions)

    def _predict_multi_traj(self, normalized_batch: batch_type) -> batch_type:
        """
        Multi-trajectory validation: unbind along traj_num, process each
        sequentially carrying RNN hidden state from one to the next.
        """
        traj_num_dim_idx = 1

        self.reset()
        actions: list[batch_type] = []

        batch_size = normalized_batch["episode_idx"].shape[0]

        for batch in split_batch(
            normalized_batch,
            partial(torch.unbind, dim=traj_num_dim_idx),
        ):
            batch["episode_idx"] = torch.arange(batch_size, device=self.device)
            actions.append(self._predict_single_traj(batch))

        return aggregate_batch(
            actions, partial(torch.stack, dim=traj_num_dim_idx)
        )  # (batch_size, traj_num, 1, action_dim)

    def reset(self):
        """Reset RNN hidden state for a new episode."""
        self.shared_model_manager.clear_cache()
        self._rnn_hidden_state_dict = {}
        self._rnn_counter_dict = {}
        self._open_loop_obs_dict = {}

    def __del__(self):
        SharedModelManager.reset()
