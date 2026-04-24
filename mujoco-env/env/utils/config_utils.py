
from omegaconf import DictConfig, OmegaConf



def convert_task_to_parallel(task_cfg: DictConfig):
    OmegaConf.set_struct(task_cfg, False)
    task_cfg.env_num = "???"
    # Currently hardcode that policy only use the last image
    task_cfg.render_image_indices = [-1]
    task_cfg.data_storage_dir = "???"

    if task_cfg.name.startswith("robomimic"):
        task_cfg._target_ = "env.modules.tasks.robomimic_task.RobomimicParallel"

    elif task_cfg.name in [
        "pick_and_match_color",
        "pick_and_match_color_rand_delay",
        "pick_and_place_back",
    ]:
        task_cfg._target_ = (
            "env.modules.tasks.pick_and_place_back.PickAndPlaceBackParallel"
        )
    elif task_cfg.name == "push_cube":
        task_cfg._target_ = "env.modules.tasks.push_cube.PushCubeParallel"
    elif task_cfg.name == "fling_cloth":
        task_cfg._target_ = "env.modules.tasks.fling_cloth.FlingClothParallel"
    else:
        raise ValueError(f"Task {task_cfg.name} is not supported")

    return task_cfg
