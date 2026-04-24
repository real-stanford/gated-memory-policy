import os
from typing import Any

import hydra
from omegaconf import DictConfig

from env.modules.remote_env_server import RemoteEnvServer


@hydra.main(config_path="../config", config_name="serve_remote_env", version_base=None)
def main(cfg: DictConfig) -> None:
    instanciated_cfg: dict[str, Any] = hydra.utils.instantiate(cfg)
    server: RemoteEnvServer = instanciated_cfg["server"]
    server.run()


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
