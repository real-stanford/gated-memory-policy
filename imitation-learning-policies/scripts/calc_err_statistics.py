import os
import tqdm
import dill
import matplotlib.pyplot as plt
import torch
from einops import rearrange
from robot_utils.data_utils import dict_apply
from robot_utils.torch_utils import torch_load, torch_save

def plot_err_distribution(
    err_results: dict[str, torch.Tensor], plot_filename: str, multi_traj: bool
):
    normalized_errs = {}
    if multi_traj:
        err_results = dict_apply(
            err_results,
            lambda x: torch.reshape(x, (x.shape[0] * x.shape[1], *x.shape[2:])),
        )
    for key, err in err_results.items():
        if key == "traj_idx" or key == "episode_idx":
            continue
        if key.endswith("_mse"):
            # normalized_errs[key] = err / torch.std(err, dim=0)
            normalized_errs[key] = err

    max_episode_length = int(torch.max(err_results["traj_idx"]))
    sum_err_at_each_frame = torch.zeros(max_episode_length + 1)
    count_at_each_frame = torch.zeros(max_episode_length + 1)
    traj_num = len(err_results["traj_idx"])
    for i in range(traj_num):
        entry_count = 0
        current_sum = 0
        for k, normalized_err in normalized_errs.items():
            current_sum += torch.sum(normalized_err[i])
            entry_count += len(normalized_err[i])
        traj_idx = int(err_results["traj_idx"][i])
        sum_err_at_each_frame[traj_idx] += current_sum / entry_count
        count_at_each_frame[traj_idx] += 1

    mean_err_at_each_frame = sum_err_at_each_frame / count_at_each_frame
    print(count_at_each_frame)
    plt.plot(mean_err_at_each_frame)
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    plt.close()


def calc_statistics(pred_results_path: str, gripper_scale: float = 0.05, action_execution_steps: int = -1, debug: bool = False, eval_episode_num: int = -1):

    pred_results = torch_load(pred_results_path, pickle_module=dill, weights_only=False)

    if pred_results["episode_idx"].ndim == 3:
        """
        Multi Traj Policies:
        keys:
            - "gt_action0_10d": (N, traj_num, T, 10)
            - "ema_action0_10d": (N, traj_num, T, 10)
            - "ema_action0_10d_mse": (N, traj_num, 10)
            - "traj_idx": (N, traj_num, 1)
            - "episode_idx": (N, traj_num, 1)
            - "entire_traj_is_padding": (N, traj_num)
        """
        for key, value in pred_results.items():
            pred_results[key] = rearrange(value, "N traj_num ... -> (N traj_num) ...")

    if "entire_traj_is_padding" in pred_results and pred_results["entire_traj_is_padding"].ndim == 1:
        pred_results["entire_traj_is_padding"] = pred_results["entire_traj_is_padding"][:, None]

    """
    keys:
        - "episode_idx": (N, 1)
        - "traj_idx": (N, 1)
        - "entire_traj_is_padding": (N, 1) # Can be omitted
        - "gt_action0_10d": (N, T, 10)
        - "policy_action0_10d": (N, T, 10) # Can be omitted
        - "policy_action0_10d_mse": (N, 10) # Can be omitted
        - "ema_action0_10d": (N, T, 10)
        - "ema_action0_10d_mse": (N, 10)
    """

    episode_frame_nums = {}
    episode_indices = list(torch.unique(pred_results["episode_idx"]))
    if debug:
        episode_indices = episode_indices[:5]
        
    for episode_idx in episode_indices:
        i = int(episode_idx)
        traj_indices = pred_results["traj_idx"][pred_results["episode_idx"] == i]
        # print(episode_idx, traj_indices.shape, torch.max(traj_indices))
        if "entire_traj_is_padding" in pred_results:
            is_padding = pred_results["entire_traj_is_padding"][pred_results["episode_idx"] == i]
            traj_indices = traj_indices[~is_padding]
        # print(episode_idx, traj_indices.shape, torch.max(traj_indices))
        episode_frame_nums[i] = (
            int(torch.max(traj_indices))
            + 1
        )
    # print(episode_indices)
    # print(episode_frame_nums)

    # Normalize the mse
    # normalized_pred_results: dict[str, torch.Tensor] = {}
    # for key in pred_results.keys():
    #     if key.endswith("_mse"):
    #         # normalized_pred_results[f"normalized_{key}"] = (pred_results[key] - pred_results[key].mean(dim=0)) / pred_results[key].std(dim=0)
    #         print(f"{key}:")
    #         print(f"    mean: {pred_results[key].mean(dim=0)}, std: {pred_results[key].std(dim=0)}")
    #         normalized_pred_results[f"normalized_{key}"] = pred_results[key]
    # Will use the ema results to calculate the variance

    # for key in pred_results.keys():
    #     if key.startswith("ema") and key.endswith("mse"):
    #         print(f"Using {key} to calculate the variance")
    #         mse = pred_results[key]
    #         variance = torch.mean(mse, dim=1) # (N, 8) -> (N,)
    #         percentile_90 = torch.quantile(variance, 0.9, dim=0)
    #         print(f"    percentile_90: {percentile_90}")
    #         normalized_variance = variance / percentile_90
    #         normalized_variance = torch.clip(normalized_variance, 0, 1) # (N,) -> (N, )
    #         break
    # else:
    #     raise ValueError("No normalized ema mse found")

    variance_key = "ema_action0_10d"
    gt_key = "gt_action0_10d"
    mse_key = "ema_action0_10d_mse"
    normalizer_key = "action0_10d"

    # unused_indices = [0]  # the 0-th frame is not used because of padding

    print(f"Using {variance_key} to calculate the variance")

    temp_results: dict[int, dict[int, torch.Tensor]] = {}
    """
    temp_results: episode_idx -> traj_idx -> (N, T, 10), N is the number of eval rounds
    """
    gt_results: dict[int, dict[int, torch.Tensor]] = {}
    """
    gt_results: episode_idx -> traj_idx -> (T, 10)
    """
    raw_mse_results: dict[int, torch.Tensor] = {}
    """
    raw_mse_results: episode_idx -> (traj_num, N)
    """

    variances: dict[int, torch.Tensor] = {}
    """
    variances: episode_idx -> (F, ), F is the episode frame number
    """
    errors: dict[int, torch.Tensor] = {}
    """
    episode_idx -> (F, )
    """

    for episode_idx in tqdm.tqdm(episode_indices, desc="Aggregating results"):
        i = int(episode_idx)

        temp_results[i] = {}
        gt_results[i] = {}
        mean_mse = []
        variances[i] = torch.nan * torch.ones(episode_frame_nums[i])
        errors[i] = torch.nan * torch.ones(episode_frame_nums[i])
        for traj_idx in range(int(episode_frame_nums[i])):
            data_indices = (pred_results["episode_idx"] == i) & (pred_results["traj_idx"] == traj_idx)
            data_indices = data_indices.squeeze(1)
            if "entire_traj_is_padding" in pred_results and any(pred_results["entire_traj_is_padding"][data_indices]):
                continue
            if data_indices.sum() == 0:
                continue
            if action_execution_steps > 0:
                temp_results[i][traj_idx] = pred_results[variance_key][data_indices][:, :action_execution_steps]
                gt_results[i][traj_idx] = pred_results[gt_key][data_indices][0, :action_execution_steps]
            else:
                temp_results[i][traj_idx] = pred_results[variance_key][data_indices]
                gt_results[i][traj_idx] = pred_results[gt_key][data_indices][0]
            
            # if not torch.allclose(mse, calculated_mse):
            #     print(f"i: {i}, traj_idx: {traj_idx}, mse: {mse}, calculated_mse: {calculated_mse}")
            # assert torch.allclose(mse, calculated_mse), \

            mean_mse.append(torch.mean(pred_results[mse_key][data_indices]))
            variances[i][traj_idx] = torch.mean(torch.var(temp_results[i][traj_idx], dim=0))
            errors[i][traj_idx] = torch.nn.functional.mse_loss(gt_results[i][traj_idx][None, ...].repeat(temp_results[i][traj_idx].shape[0], 1, 1), temp_results[i][traj_idx])

        raw_mse_results[i] = torch.stack(mean_mse)

    # for i in range(len(pred_results["episode_idx"])):
    #     episode_idx = int(pred_results["episode_idx"][i, 0])
    #     traj_idx = int(pred_results["traj_idx"][i, 0])
    #     if torch.any(torch.isnan(pred_results[mse_key][i])):
    #         continue
    #     if "entire_traj_is_padding" in pred_results and pred_results["entire_traj_is_padding"][i, 0]:
    #         continue
    #     # print(f"episode_idx: {episode_idx}, traj_idx: {traj_idx}")
    #     if episode_idx not in temp_results:
    #         temp_results[episode_idx] = {}
    #         gt_results[episode_idx] = {}
    #     if traj_idx not in temp_results[episode_idx]:
    #         temp_results[episode_idx][traj_idx] = []
    #         gt_results[episode_idx][traj_idx] = pred_results[gt_key][i, :action_execution_steps]
    #     else:
    #         assert torch.allclose(gt_results[episode_idx][traj_idx], pred_results[gt_key][i, :action_execution_steps]), \
    #             f"episode_idx: {episode_idx}, traj_idx: {traj_idx}, i: {i}, gt_results[episode_idx][traj_idx]: {gt_results[episode_idx][traj_idx]}, pred_results[gt_key][i]: {pred_results[gt_key][i]}"

    #     temp_results[episode_idx][traj_idx].append(pred_results[variance_key][i, :action_execution_steps])

    # Debug: Plot the mse results
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.set_ylabel("MSE")
    # ax.set_xlabel("Frame Index")
    # ax.set_title("MSE")
    # ax.plot(errors[5])
    # plt.savefig("data/pick_and_place_back/2025-12-01/mse_episode_5.png")
    # plt.close()
    # action_mask = torch.ones(10)
    action_mask = torch.zeros(10)

    for episode_idx in tqdm.tqdm(temp_results.keys(), desc="Calculating statistics"):
        frame_num = episode_frame_nums[episode_idx]
        variances[episode_idx] = torch.nan * torch.ones(frame_num)
        # variances[episode_idx][unused_indices] = 0

        errors[episode_idx] = torch.nan * torch.ones(frame_num)
        # errors[episode_idx][unused_indices] = 0

        for traj_idx in temp_results[episode_idx].keys():
            if len(temp_results[episode_idx][traj_idx]) == 0:
                print(f"No valid results found for episode {episode_idx} and traj {traj_idx}")
                continue
            # outputs = torch.stack(temp_results[episode_idx][traj_idx])  # (N, T, 8)
            outputs = temp_results[episode_idx][traj_idx]
            outputs[:, :, -1] *= gripper_scale
            temp_gt = gt_results[episode_idx][traj_idx]
            temp_gt[:, -1] *= gripper_scale

            # outputs = single_field_normalizer.unnormalize(outputs).detach()
            # temp_gt = single_field_normalizer.unnormalize(temp_gt[None, ...])[0].detach()
            # print(f"unnormalized_gt: {unnormalized_gt[..., -1, 2]}")
            # print(f"unnormalized_outputs: {unnormalized_outputs[..., -1, 2]}")

            outputs *= action_mask[None, None, :]
            temp_gt *= action_mask[None, :]



            traj_num = outputs.shape[0]
            if traj_num <= 5:
                # Skip the episode if there are too few trajectories
                continue

            variance = torch.var(outputs, dim=0)  # (T, 8)
            variances[episode_idx][traj_idx] = torch.mean(variance)
            if torch.isnan(variances[episode_idx][traj_idx]):
                print(f"NaN values found in episode {episode_idx} and traj {traj_idx}")
                print(f"    outputs: {outputs}")
                print(f"    variance: {variance}")
                print(f"    mean: {torch.mean(variance)}")
                raise ValueError(
                    f"NaN values found in episode {episode_idx} and traj {traj_idx}"
                )


            errors[episode_idx][traj_idx] = torch.nn.functional.mse_loss(temp_gt[None, ...].repeat(traj_num, 1, 1), outputs)
            
        # assert (
        #     torch.isnan(variances[episode_idx]).sum() == 0
        # ), f"NaN values found in episode {episode_idx}, frame num: {frame_num}"

    # torch_save(variances, pred_results_path.replace(".pt", "_variance.pt"))
    # torch_save(errors, pred_results_path.replace(".pt", "_error.pt"))
    
    statistics_path = pred_results_path.replace(".pt", "_statistics.pt")
    if eval_episode_num > 0:
        statistics_path = statistics_path.replace(".pt", f"_{eval_episode_num}.pt")
    if debug:
        statistics_path = statistics_path.replace(".pt", "_debug.pt")

    torch_save(
        {
            "variances": variances,
            "errors": errors,
        }, 
        statistics_path
    )
    # normalized_results: dict[int, torch.Tensor] = {}
    # all_results = torch.cat([results[i] for i in results.keys()])
    # percentile_95 = torch.quantile(all_results, 0.95, dim=0)
    # percentile_90 = torch.quantile(all_results, 0.90, dim=0)
    # percentile_80 = torch.quantile(all_results, 0.80, dim=0)
    # percentile_5 = torch.quantile(all_results, 0.05, dim=0)
    # percentile_10 = torch.quantile(all_results, 0.10, dim=0)
    # percentile_20 = torch.quantile(all_results, 0.20, dim=0)
    # for episode_idx in results.keys():
    #     rescaled_variance = (results[episode_idx] - percentile_20) / (
    #         percentile_80 - percentile_20
    #     )
    #     normalized_results[episode_idx] = torch.clip(rescaled_variance, 0, 1)

    # torch_save(
    #     normalized_results, pred_results_path.replace(".pt", "_normalized_variance.pt")
    # )


def plot_statistics(statistics_paths: dict[str, str], plot_dir: str):
    statistics: dict[str, dict[str, dict[int, torch.Tensor]]] = {}
    for key, statistics_path in statistics_paths.items():
        print("loading statistics from", statistics_path)
        statistics[key] = torch_load(
            statistics_path, pickle_module=dill, weights_only=False
        )
    episode_indices = set(next(iter(statistics.values()))["variances"].keys())
    for key in statistics.keys():
        print(key, statistics[key].keys())
        assert set(statistics[key]["variances"].keys()) == episode_indices, \
            f"Episode indices mismatch for {key}"

    os.makedirs(plot_dir, exist_ok=True)
    for episode_idx in episode_indices:
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        ax1.set_ylabel("Variance")
        # ax1.set_xlabel("Frame Index")
        ax2.set_ylabel("Error")
        ax2.set_xlabel("Frame Index")


        for key in statistics.keys():
            variance = statistics[key]["variances"][episode_idx]
            error = statistics[key]["errors"][episode_idx]

            traj_idx = torch.arange(len(variance))
            valid_traj_idx = traj_idx[~torch.isnan(variance)]
            # fig, ax = plt.subplots(2, 1, figsize=(10, 10))

            ax1.plot(valid_traj_idx, variance[valid_traj_idx], label=key)
            ax2.plot(valid_traj_idx, error[valid_traj_idx], label=key)

        ax1.set_yscale("log")
        ax2.set_yscale("log")
        ax1.legend()
        ax2.legend()

        plt.savefig(os.path.join(plot_dir, f"episode_{episode_idx}_statistics.pdf"))
        print(f"Plot saved to {os.path.join(plot_dir, f'episode_{episode_idx}_statistics.pdf')}")
        plt.close(fig)

    # each_frame_num_variance_list: dict[int, list[float]] = {}
    # for episode_idx in statistics["variances"].keys():
    #     for traj_idx, variance in enumerate(statistics["variances"][episode_idx]):
    #         if traj_idx not in each_frame_num_variance_list:
    #             each_frame_num_variance_list[traj_idx] = []
    #         each_frame_num_variance_list[traj_idx].append(float(variance))
    # max_frame_num = max(each_frame_num_variance_list.keys())
    # each_frame_num_variance: torch.Tensor = torch.zeros(max_frame_num)
    # for traj_idx in range(max_frame_num):
    #     each_frame_num_variance[traj_idx] = torch.mean(
    #         torch.tensor(each_frame_num_variance_list[traj_idx])
    #     )

    # plt.plot(each_frame_num_variance)

    # fig_path = statistics_path.replace(".pt", "_distribution.png")
    # plt.savefig(fig_path)
    # print(f"Plot saved to {fig_path}")
    # plt.close()
def plot_statistics_with_gate(merged_statistics_path: str, plot_dir: str, memory_gate_vals_path: str = ""):
    merged_statistics = torch_load(merged_statistics_path, pickle_module=dill, weights_only=False)
    if memory_gate_vals_path != "":
        memory_gate_vals = torch_load(memory_gate_vals_path, pickle_module=dill, weights_only=False)
    else:
        memory_gate_vals = None

    # plot_dir = merged_statistics_path.replace(".pt", "_with_gate")
    os.makedirs(plot_dir, exist_ok=True)

    """
    merged_statistics: {
        episode_idx: {
            "with_mem_variances": torch.Tensor(frame_num),
            "no_mem_variances": torch.Tensor(frame_num),
            "with_mem_errors": torch.Tensor(frame_num),
            "no_mem_errors": torch.Tensor(frame_num),
        }
    }
    memory_gate_vals: {
        "memory_gate_vals": torch.Tensor(traj_num),
        "traj_indices": torch.Tensor(traj_num),
        "episode_indices": torch.Tensor(traj_num),
    }
    """

    for episode_idx in merged_statistics.keys():
        if memory_gate_vals is not None and episode_idx not in memory_gate_vals["episode_indices"]:
            continue
        print(f"episode_idx: {episode_idx}")
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        ax1.set_ylabel("Variance")
        ax2.set_ylabel("Error")
        ax2.set_xlabel("Frame Index")

        for key in ["with_mem", "no_mem"]:
            variance = merged_statistics[episode_idx][f"{key}_variances"]
            error = merged_statistics[episode_idx][f"{key}_errors"]

            traj_idx = torch.arange(len(variance))
            valid_traj_idx = traj_idx[~torch.isnan(variance)]
            # fig, ax = plt.subplots(2, 1, figsize=(10, 10))

            ax1.plot(valid_traj_idx, variance[valid_traj_idx], label=key)
            ax2.plot(valid_traj_idx, error[valid_traj_idx], label=key)

        ax1.set_yscale("log")
        ax2.set_yscale("log")
        ylim1 = ax1.get_ylim()
        ylim2 = ax2.get_ylim()
        if memory_gate_vals is not None:
            array_indices = (memory_gate_vals["episode_indices"] == episode_idx).squeeze().to("cpu")
            print(f"{array_indices.shape=}, {memory_gate_vals['traj_indices'].shape=}, {memory_gate_vals['memory_gate_vals'].shape=}")
            valid_traj_idx = (memory_gate_vals["traj_indices"][array_indices]).squeeze().to("cpu")
            valid_memory_gate_val = memory_gate_vals["memory_gate_vals"][array_indices].to("cpu")
            ax1.plot(valid_traj_idx, valid_memory_gate_val * ylim1[1], label="memory_gate_val")
            ax2.plot(valid_traj_idx, valid_memory_gate_val * ylim2[1], label="memory_gate_val")

        ax1.set_ylim(ylim1)
        ax2.set_ylim(ylim2)

        ax1.legend()
        ax2.legend()

        plt.savefig(os.path.join(plot_dir, f"episode_{episode_idx}_statistics_with_gate.pdf"))
        print(f"Plot saved to {os.path.join(plot_dir, f'episode_{episode_idx}_statistics_with_gate.pdf')}")
        plt.close(fig)



def merge_statistics(with_mem_statistics_path: str, no_mem_statistics_path: str, output_path: str):
    with_mem_statistics = torch_load(with_mem_statistics_path, pickle_module=dill, weights_only=False)
    no_mem_statistics = torch_load(no_mem_statistics_path, pickle_module=dill, weights_only=False)

    """
    statistics:
        "variances": {episode_idx: torch.Tensor(frame_num)}
        "errors": {episode_idx: torch.Tensor(frame_num)}
    """

    episode_indices = set(with_mem_statistics["variances"].keys())
    assert episode_indices == set(no_mem_statistics["variances"].keys()), \
        f"Episode indices mismatch"
    assert episode_indices == set(with_mem_statistics["errors"].keys()), \
        f"Episode indices mismatch"
    assert episode_indices == set(no_mem_statistics["errors"].keys()), \
        f"Episode indices mismatch"

    merged_statistics: dict[int, dict[str, torch.Tensor]] = {}
    """
    merged_statistics: dict[int, dict[str, torch.Tensor]] = {
        episode_idx: {
            "with_mem_variances": torch.Tensor(frame_num),
            "no_mem_variances": torch.Tensor(frame_num),
            "with_mem_errors": torch.Tensor(frame_num),
            "no_mem_errors": torch.Tensor(frame_num),
        }
    """

    for idx in episode_indices:
        merged_statistics[idx] = {
            "with_mem_variances": with_mem_statistics["variances"][idx],
            "no_mem_variances": no_mem_statistics["variances"][idx],
            "with_mem_errors": with_mem_statistics["errors"][idx],
            "no_mem_errors": no_mem_statistics["errors"][idx],
        }
    torch_save(merged_statistics, output_path)
    print(f"Merged statistics saved to {output_path}")


def statistics_sliding_window(merged_statistics_path: str, window_size: int):
    merged_statistics = torch_load(merged_statistics_path, pickle_module=dill, weights_only=False)
    """
    merged_statistics: {
        episode_idx: {
            "with_mem_variances": torch.Tensor(frame_num),
            "no_mem_variances": torch.Tensor(frame_num),
            "with_mem_errors": torch.Tensor(frame_num),
            "no_mem_errors": torch.Tensor(frame_num),
        }
    """

    for episode_idx in merged_statistics.keys():
        for keys in ["with_mem_variances", "no_mem_variances", "with_mem_errors", "no_mem_errors"]:
            tensor = merged_statistics[episode_idx][keys]
            averaged_tensor = torch.zeros(tensor.shape[0])
            for i in range(tensor.shape[0]):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(tensor.shape[0], i + window_size // 2 + 1)
                not_nan = ~torch.isnan(tensor[start_idx:end_idx])
                if not_nan.sum() > 0:
                    averaged_tensor[i] = torch.mean(tensor[start_idx:end_idx][not_nan])
                else:
                    averaged_tensor[i] = torch.nan
            merged_statistics[episode_idx][keys] = averaged_tensor
    torch_save(merged_statistics, merged_statistics_path.replace(".pt", f"_window_{window_size}.pt"))
    print(f"Merged statistics saved to {merged_statistics_path.replace('.pt', f'_window_{window_size}.pt')}")



if __name__ == "__main__":

    plot_statistics_with_gate(
        merged_statistics_path = f"{s3_prefix}/real_world_iterative_casting/2026-01-28/20-59-27_val_comparison/val_results_statistics_window_20.pt",
        plot_dir = "data/real_world_iterative_casting/2026-01-28/20-59-27_val_comparison_with_gate"
    )
