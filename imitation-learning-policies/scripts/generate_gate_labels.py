import os
import click
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.calc_err_statistics import calc_statistics, merge_statistics, plot_statistics, statistics_sliding_window
from scripts.eval_model import eval_model



def get_gate_labels(
    with_mem_ckpt_path: str,
    no_mem_ckpt_path: str,
    with_mem_idx_pool_size: int,
    no_mem_idx_pool_size: int,
    dataset_type: str,
    dataset_dir: str = "",
    dataset_name: str = "",
    eval_episode_num: int = -1,
    date_str: str = datetime.now().strftime("%Y-%m-%d"),
    time_str: str = datetime.now().strftime("%H-%M-%S"),
    window_size: int = 5,
):
    assert dataset_type in ["train", "val"]

    with_mem_ckpt_dir = "/".join(with_mem_ckpt_path.split("/")[:-1])
    with_mem_epoch = int(with_mem_ckpt_path.split("/")[-1].split("_")[1])
    with_mem_task_name = with_mem_ckpt_path.split("/")[-5]
    with_mem_date_str = with_mem_ckpt_path.split("/")[-4]
    with_mem_eval_results_path = f"{with_mem_ckpt_dir}/epoch_{with_mem_epoch}_eval/{dataset_type}_results.pt"

    no_mem_ckpt_dir = "/".join(no_mem_ckpt_path.split("/")[:-1])
    no_mem_task_name = no_mem_ckpt_path.split("/")[-5]
    assert with_mem_task_name == no_mem_task_name
    no_mem_date_str = no_mem_ckpt_path.split("/")[-4]
    no_mem_epoch = int(no_mem_ckpt_path.split("/")[-1].split("_")[1])
    no_mem_eval_results_path = f"{no_mem_ckpt_dir}/epoch_{no_mem_epoch}_eval/{dataset_type}_results.pt"

    comparison_dir = f"data/{with_mem_task_name}/{date_str}/{time_str}_{dataset_type}_comparison"
    if int(os.environ.get("RANK", 0)) == 0:
        os.makedirs(comparison_dir, exist_ok=True)
        slurm_job_id = os.environ.get("SLURM_JOB_ID", "unknown")
        with open(f"{comparison_dir}/gate_job_{slurm_job_id}.txt", "w") as f:
            f.write(f"with_mem_ckpt_path: {with_mem_ckpt_path}\n")
            f.write(f"no_mem_ckpt_path: {no_mem_ckpt_path}\n")
            f.write(f"with_mem_idx_pool_size: {with_mem_idx_pool_size}\n")
            f.write(f"no_mem_idx_pool_size: {no_mem_idx_pool_size}\n")
            f.write(f"dataset_type: {dataset_type}\n")
            f.write(f"dataset_dir: {dataset_dir}\n")
    eval_model(
        ckpt_path=with_mem_ckpt_path,
        index_pool_size_per_episode=with_mem_idx_pool_size,
        rounds=1,
        dataset_type=dataset_type,
        dataset_dir=dataset_dir,
        eval_episode_num=eval_episode_num,
    )

    eval_model(
        ckpt_path=no_mem_ckpt_path,
        index_pool_size_per_episode=no_mem_idx_pool_size,
        rounds=1,
        dataset_type=dataset_type,
        dataset_dir=dataset_dir,
        eval_episode_num=eval_episode_num,
    )

    if int(os.environ.get("RANK", 0)) > 0:
        # Only keep one process running
        exit()


    assert with_mem_task_name == no_mem_task_name

    calc_statistics(no_mem_eval_results_path, eval_episode_num=eval_episode_num)
    calc_statistics(with_mem_eval_results_path, eval_episode_num=eval_episode_num)

    if eval_episode_num > 0:
        with_mem_statistics_path = f"{with_mem_ckpt_dir}/epoch_{with_mem_epoch}_eval/{dataset_type}_results_statistics_{eval_episode_num}.pt"
        no_mem_statistics_path = f"{no_mem_ckpt_dir}/epoch_{no_mem_epoch}_eval/{dataset_type}_results_statistics_{eval_episode_num}.pt"
        merged_statistics_path = f"{comparison_dir}/val_results_statistics_{eval_episode_num}.pt"
    else:
        with_mem_statistics_path = f"{with_mem_ckpt_dir}/epoch_{with_mem_epoch}_eval/{dataset_type}_results_statistics.pt"
        no_mem_statistics_path = f"{no_mem_ckpt_dir}/epoch_{no_mem_epoch}_eval/{dataset_type}_results_statistics.pt"
        merged_statistics_path = f"{comparison_dir}/val_results_statistics.pt"


    plot_statistics(
        {
            "no_mem": no_mem_statistics_path,
            "with_mem": with_mem_statistics_path,
        },
        comparison_dir
    )

    merge_statistics(with_mem_statistics_path, no_mem_statistics_path, merged_statistics_path)

    if window_size > 1:
        statistics_sliding_window(merged_statistics_path, window_size)

@click.command()
@click.option("--with_mem_ckpt_path", type=str, required=True)
@click.option("--no_mem_ckpt_path", type=str, required=True)
@click.option("--with_mem_idx_pool_size", type=int, default=200)
@click.option("--no_mem_idx_pool_size", type=int, default=50000)
@click.option("--dataset_type", type=str, required=True)
@click.option("--dataset_dir", type=str, default="")
@click.option("--eval_episode_num", type=int, required=True)
@click.option("--dataset_name", type=str, default="")
@click.option("--date_str", type=str, default=datetime.now().strftime("%Y-%m-%d"))
@click.option("--time_str", type=str, default=datetime.now().strftime("%H-%M-%S"))
@click.option("--window_size", type=int, default=5)
def main(
        with_mem_ckpt_path: str, 
        no_mem_ckpt_path: str, 
        with_mem_idx_pool_size: int, 
        no_mem_idx_pool_size: int, 
        dataset_type: str, 
        dataset_dir: str,
        eval_episode_num: int,
        dataset_name: str,
        date_str: str,
        time_str: str,
        window_size: int
    ):
    get_gate_labels(
        with_mem_ckpt_path, 
        no_mem_ckpt_path, 
        with_mem_idx_pool_size, 
        no_mem_idx_pool_size, 
        dataset_type, 
        dataset_dir,
        dataset_name,
        eval_episode_num,
        date_str,
        time_str,
        window_size
    )


if __name__ == "__main__":
    main()