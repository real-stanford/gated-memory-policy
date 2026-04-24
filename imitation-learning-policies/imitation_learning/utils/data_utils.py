import numpy as np
import numpy.typing as npt
import torch
from scipy.signal import convolve
from torch.nn.functional import conv1d


def get_shakiness_score_torch(traj: torch.Tensor) -> torch.Tensor:
    """
    traj: (batch_size, traj_num, traj_length, action_dim) or (batch_size, traj_length, action_dim) or (traj_length, action_dim)
    output: (action_dim,)
    """
    init_shape = traj.shape
    if len(traj.shape) == 4:
        batch_size, traj_num, traj_length, action_dim = traj.shape
        traj = traj.reshape(batch_size * traj_num, traj_length, action_dim)
        batch_size = batch_size * traj_num
    elif len(traj.shape) == 3:
        batch_size, traj_length, action_dim = traj.shape
    elif len(traj.shape) == 2:
        traj_length, action_dim = traj.shape
        batch_size = 1
        traj = traj.unsqueeze(0)
    else:
        raise ValueError(f"Invalid trajectory shape: {traj.shape}")

    traj = traj.permute(0, 2, 1)  # (batch_size, action_dim, traj_length)
    traj = traj.reshape(batch_size * action_dim, 1, traj_length)  # in_channels = 1

    if traj_length < 4:
        return torch.zeros(action_dim, device=traj.device)

    kernel_1 = torch.tensor([-0.5, 1.0, -0.5], device=traj.device)[
        None, None, :
    ]  # (1, 1, 3); out_channels = 1
    kernel_2 = torch.tensor([-0.5, 0.5, 0.5, -0.5], device=traj.device)[
        None, None, :
    ]  # (1, 1, 4); out_channels = 1

    score_1 = conv1d(traj, kernel_1, padding=0)
    score_2 = conv1d(traj, kernel_2, padding=0)
    assert score_1.shape == (batch_size * action_dim, 1, traj_length - 2)
    assert score_2.shape == (batch_size * action_dim, 1, traj_length - 3)
    score_1 = score_1.reshape(batch_size, action_dim, traj_length - 2)
    score_2 = score_2.reshape(batch_size, action_dim, traj_length - 3)
    traj = traj.reshape(batch_size, action_dim, traj_length)
    # max_vals = torch.max(traj, axis=0)
    # min_vals = torch.min(traj, axis=0)
    std_vals = torch.std(traj, dim=2)

    score_1 = torch.sum(torch.abs(score_1), dim=2) / (traj_length - 2) / (std_vals + 1e-5)
    score_2 = torch.sum(torch.abs(score_2), dim=2) / (traj_length - 3) / (std_vals + 1e-5)

    score = (score_1 + score_2) / 2

    mean_score = torch.mean(score, dim=0)

    # print(f"{os.environ.get('RANK')}: {mean_score=}, {std_vals=}, {traj_length=}")

    return mean_score


def get_shakiness_score_numpy(traj: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    traj: (T, D)
    """
    assert len(traj.shape) == 2, "Trajectory must be a 2D array"
    T, D = traj.shape
    if T < 4:
        return np.zeros(D)

    kernel_1 = np.array([-0.5, 1.0, -0.5])
    kernel_2 = np.array([-0.5, 0.5, 0.5, -0.5])

    score_1 = convolve(traj, kernel_1[:, None], mode="valid")
    score_2 = convolve(traj, kernel_2[:, None], mode="valid")
    assert score_1.shape == (T - 2, D)
    assert score_2.shape == (T - 3, D)
    # max_vals = np.max(traj, axis=0)
    # min_vals = np.min(traj, axis=0)
    std_vals = np.std(traj, axis=0)
    score_1 = np.sum(np.abs(score_1), axis=0) / (T - 2) / std_vals
    score_2 = np.sum(np.abs(score_2), axis=0) / (T - 3) / std_vals

    return (score_1 + score_2) / 2



