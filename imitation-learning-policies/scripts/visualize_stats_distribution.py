import click 
import os
import matplotlib.pyplot as plt
from robot_utils.torch_utils import torch_load, torch_save
import dill
import torch
@click.command()
@click.argument("stats_path", type=str)
@click.argument("quantile", type=float, default=0.99)
def visualize_stats_distribution(stats_path: str, quantile: float):
    assert 0.90 <= quantile <= 1.0, "Quantile must be between 0.90 and 1.0"
    stats = torch_load(stats_path, pickle_module=dill)
    dataset_dir = os.path.dirname(stats_path)
    dataset_name = os.path.basename(stats_path).replace("_stats.pth", "")
    for entry_name, entry_stats in stats.items():
        entry_dim = entry_stats["min"].shape[1]
        
        fig = plt.figure(figsize=(16, 10))
        axes = fig.subplots(entry_dim//2, 2)
        quantile_min = torch.quantile(entry_stats["min"], (1 - quantile)/2, dim=0)
        quantile_max = torch.quantile(entry_stats["max"], (1 + quantile)/2, dim=0)
        print(f"Quantile {quantile_min=}, {quantile_max=}, diff: {quantile_max - quantile_min}")
        for i in range(entry_dim):
            ax = axes[i//2, i%2]
            ax.hist(entry_stats["min"][:, i].numpy(), bins=100, label=f"{entry_name}_{i}_min", alpha=0.5)
            ax.hist(entry_stats["max"][:, i].numpy(), bins=100, label=f"{entry_name}_{i}_max", alpha=0.5)
            ax.legend()
            ax.axvline(quantile_min[i], color="red", linestyle="--")
            ax.axvline(quantile_max[i], color="red", linestyle="--")
        fig_path = os.path.join(dataset_dir, f"{dataset_name}_{entry_name}_distribution.pdf")
        plt.savefig(fig_path)
        print(f"Distribution plot saved to {fig_path}")

def aggregate_stats(stats_paths: list[str]):
    stats = {}
    for stats_path in stats_paths:
        stats_dict = torch_load(stats_path, pickle_module=dill)
        for entry_name, entry_stats in stats_dict.items():
            if entry_name not in stats:
                stats[entry_name] = entry_stats
            else:
                stats[entry_name]["min"] = torch.cat((stats[entry_name]["min"], entry_stats["min"]), dim=0)
                stats[entry_name]["max"] = torch.cat((stats[entry_name]["max"], entry_stats["max"]), dim=0)
    
    stats_dir = os.path.dirname(stats_paths[0])
    multi_task_stats_path = os.path.join(stats_dir, "multi_task_stats.pth")
    torch_save(stats, multi_task_stats_path, pickle_module=dill)
    print(f"Aggregated stats saved to {multi_task_stats_path}")



if __name__ == "__main__":
    visualize_stats_distribution()
