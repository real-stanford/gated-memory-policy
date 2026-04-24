import os
import shutil
import threading
from typing import Any

import dill
from robot_utils.torch_utils import torch_save

class CheckpointManager:
    def __init__(
        self,
        output_dir: str,
        keep_latest: bool,
        max_ckpt_num: int,
        sort_key: str,
        keep_min: bool,
        save_workspace: bool,
    ):
        self.output_dir: str = output_dir
        self.saved_ckpts: dict[str, dict[str, Any]] = {}  # tag -> ckpt_meta
        self.max_ckpt_num: int = max_ckpt_num
        self.sort_key: str = sort_key
        self.keep_min: bool = keep_min
        self.keep_latest: bool = keep_latest
        self.save_workspace: bool = save_workspace
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        if not os.path.exists(os.path.join(self.output_dir, "checkpoints")):
            os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        self.last_ckpt_path: str = ""

    def get_ckpt_tag(self, ckpt_meta: dict[str, Any]) -> str:
        assert self.sort_key in ckpt_meta.keys()
        assert "info/epoch" in ckpt_meta.keys()
        tag = (
            f"epoch_{ckpt_meta['info/epoch']}_{self.sort_key.replace('/', '_')}_{ckpt_meta[self.sort_key]:.3f}"
        )
        tag = tag.replace(".", "_")
        return tag

    def get_ckpt_path(self, tag: str) -> str:
        return os.path.join(self.output_dir, "checkpoints", f"{tag}.ckpt")

    def get_last_ckpt_path(self) -> str:
        assert self.last_ckpt_path != "", "Last ckpt path is not set"
        return self.last_ckpt_path

    def clear_old_ckpts(self):
        sorted_ckpts = sorted(
            self.saved_ckpts.keys(),
            key=lambda x: self.saved_ckpts[x][self.sort_key],
            reverse=not self.keep_min,
        )
        # by default in the ascending order (keep the min)
        # if keep_min is False, then the descending order (keep the max)
        for tag in sorted_ckpts[self.max_ckpt_num :]:
            self.saved_ckpts.pop(tag)
            if os.path.exists(self.get_ckpt_path(tag)):
                os.remove(self.get_ckpt_path(tag))

    def save_ckpt(
        self,
        ckpt_meta: dict[str, Any],
        ckpt: dict[str, Any],
        workspace: dict[str, Any] | None = None,
    ):
        """
        ckpt_meta should include meta information for this checkpoint:
            - epoch
            - global_step
            - train_loss (sort key)
        """
        tag = self.get_ckpt_tag(ckpt_meta)
        ckpt_path = self.get_ckpt_path(tag)
        self.saved_ckpts[tag] = ckpt_meta

        def save_and_copy():
            torch_save(ckpt, ckpt_path, pickle_module=dill)
            if self.keep_latest:
                latest_ckpt_path = self.get_ckpt_path("latest")
                shutil.copy(ckpt_path, latest_ckpt_path)
            if self.save_workspace and workspace is not None:
                workspace_path = self.get_ckpt_path(f"workspace")
                try:
                    torch_save(workspace, workspace_path, pickle_module=dill)
                except RuntimeError:
                    print(f"Workspace is currently being saved by the last epoch. Skipping...")
            self.clear_old_ckpts()

        self._saving_thread = threading.Thread(target=save_and_copy)
        self._saving_thread.start()
        self.last_ckpt_path = ckpt_path
        print(f"\nCheckpoint saved to {ckpt_path}")
