from typing import Any


class BaseEnv:
    def __init__(
        self,
        run_name: str,
        time_tag: str,
    ):
        self.run_name: str = run_name
        self.time_tag: str = time_tag

    def start_rollout(self, epoch: int, ckpt_path: str) -> None:
        raise NotImplementedError

    def fetch_results(self) -> list[dict[str, Any]]:
        """
        Return: list[dict[str, Any]]
            Each dict contains:
                - "epoch": int
                - "success_rate": float
                - "rollout_data_path": str
            Will return an empty list if no results are available
            "run_name" and "time_tag" will match the values passed to the constructor
        """
        raise NotImplementedError
