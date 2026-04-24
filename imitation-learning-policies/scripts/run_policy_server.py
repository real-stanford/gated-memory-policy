import click
import numpy as np
import hydra
from imitation_learning.envs.policy_server import PolicyServer
from imitation_learning.utils.config_utils import compose_hydra_config


@click.command()
@click.option("--server_endpoint", type=str, default="tcp://0.0.0.0:18765")
@click.option("--device", type=str, default="cuda")
@click.option("--ckpt_path", type=str, default="")
@click.option("--task_name", type=str, default="")
@click.option("--policy_name", type=str, default="")
@click.option("--server_name", type=str, default="localhost")
@click.option("--record_history_attention", type=bool, default=False)
def main(server_endpoint: str, device: str, ckpt_path: str, task_name: str, policy_name: str, server_name: str, record_history_attention: bool):
    np.set_printoptions(threshold=np.inf, precision=3)
    server = PolicyServer(server_endpoint, device, wait_ckpt_writing_time_s=20)
    if ckpt_path:
        if task_name and policy_name:
            print(f"Creating config from {task_name} and {policy_name}")
            with hydra.initialize('../imitation_learning/configs'):
                cfg = compose_hydra_config(task_name, policy_name)
            server._load_weights_only(cfg, ckpt_path)
            print(f"Loaded legacy weights from {ckpt_path}")
        else:
            ckpt_info_dict = {
                "ckpt_path": ckpt_path,
                "train_server_name": server_name,
                "record_history_attention": record_history_attention,
            }
            server._load_ckpt(ckpt_info_dict)
    else:
        print("No checkpoint path provided, will wait for the checkpoint to be written")
    server.run()


if __name__ == "__main__":
    main()
