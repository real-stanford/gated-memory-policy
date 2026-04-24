"""
Extract 768-dim image features from third-person camera images using a pretrained ViT (SigLIP-2)
loaded from a training checkpoint.

Usage:
    python scripts/extract_image_features.py \
        --data_dir ~/robotics/repositories/mujoco/mujoco-env/data/collect_heuristic_data/pick_and_match_color_rand_delay_5_120 \
        --ckpt_path data/pick_and_match_color/2026-02-13/20-16-21_match_color_diffusion_memory_lr3e-4/checkpoints/epoch_2_train_mean_loss_0_001.ckpt \
        --batch_size 64
"""

import copy
import os

import click
import numpy as np
import torch
import zarr
from tqdm import tqdm
from transformers import SiglipVisionModel


def load_vit_from_checkpoint(ckpt_path: str, device: torch.device):
    """Load the SigLIP-2 ViT and MAP head from a training checkpoint."""
    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]

    # Extract ViT backbone weights
    vit_prefix = "shared_model_manager.vit_models.google/siglip2-base-patch16-256."
    vit_state_dict = {
        k[len(vit_prefix):]: v
        for k, v in state_dict.items()
        if k.startswith(vit_prefix)
    }
    nested_vit_state_dict = {"model_state_dict": {k: v for k, v in state_dict.items() if k.startswith(vit_prefix)}}

    vit_ckpt_path = ckpt_path.replace(".ckpt", "_vit.ckpt")
    if not os.path.exists(vit_ckpt_path):
        torch.save(nested_vit_state_dict, vit_ckpt_path)
        print(f"Saved ViT state dict to {vit_ckpt_path}")
    else:
        print(f"ViT state dict already exists at {vit_ckpt_path}")

    # Extract MAP (Multihead Attention Pooling) head weights
    map_prefix = "shared_model_manager.map_models.third_person_camera."
    map_state_dict = {
        k[len(map_prefix):]: v
        for k, v in state_dict.items()
        if k.startswith(map_prefix)
    }

    # Load ViT model
    vit_model = SiglipVisionModel.from_pretrained("google/siglip2-base-patch16-256")
    vit_model.load_state_dict(vit_state_dict, strict=True)
    vit_model.eval()
    vit_model.to(device)

    # Load MAP head (reuse the architecture from the pretrained model's head)
    temp_model = SiglipVisionModel.from_pretrained("google/siglip2-base-patch16-256")
    map_head = copy.deepcopy(temp_model.vision_model.head)
    del temp_model
    map_head.load_state_dict(map_state_dict, strict=True)
    map_head.eval()
    map_head.to(device)

    print(f"Loaded ViT and MAP head from checkpoint (epoch {ckpt.get('epoch', '?')})")
    return vit_model, map_head


def preprocess_images(images: np.ndarray) -> torch.Tensor:
    """
    Preprocess uint8 images (N, H, W, C) -> normalized float tensor (N, C, H, W).
    Uses SigLIP normalization: (x / 255 - 0.5) / 0.5
    """
    x = torch.from_numpy(images).float() / 255.0
    x = x.permute(0, 3, 1, 2)  # (N, C, H, W)
    x = (x - 0.5) / 0.5
    return x


@torch.no_grad()
def extract_features(
    images: np.ndarray,
    vit_model: SiglipVisionModel,
    map_head: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Extract 768-dim features from images using ViT + MAP pooling."""
    n_images = len(images)
    all_features = []

    for start in range(0, n_images, batch_size):
        end = min(start + batch_size, n_images)
        batch = preprocess_images(images[start:end]).to(device)

        # ViT forward -> patch features
        patch_features = vit_model(batch).last_hidden_state  # (B, 256, 768)
        # MAP pooling -> single 768-dim token
        pooled = map_head(patch_features)  # (B, 768)

        all_features.append(pooled.cpu().numpy())

    return np.concatenate(all_features, axis=0)  # (N, 768)


def override_vit_to_ckpt(vit_model_path: str, ckpt_path: str):
    """Override the ViT model in the checkpoint with the ViT model from the path."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    vit_state_dict = torch.load(vit_model_path, map_location="cpu", weights_only=False)["model_state_dict"]
    dict_keys = list(vit_state_dict.keys())
    override_key_num = len([k for k in dict_keys if k in ckpt["model_state_dict"]])
    print(f"Overridden {override_key_num} keys out of {len(dict_keys)}")
    
    ckpt["model_state_dict"].update(vit_state_dict)
    torch.save(ckpt, ckpt_path)
    print(f"Overridden ViT model to {ckpt_path}")

@click.command()
@click.option(
    "--data_dir",
    type=str,
    default=os.path.expanduser(
        "~/robotics/repositories/mujoco/mujoco-env/data/collect_heuristic_data/"
        "pick_and_match_color_rand_delay_5_600"
    ),
    help="Path to the dataset directory containing episode_data.zarr",
)
@click.option(
    "--ckpt_path",
    type=str,
    default="data/pick_and_match_color/2026-03-23/18-17-33_match_color_diffusion_memory_lr3e-4/checkpoints/epoch_10_train_mean_loss_0_000.ckpt",
    help="Path to the model checkpoint",
)
@click.option("--batch_size", type=int, default=64, help="Batch size for inference")
@click.option("--device", type=str, default=None, help="Device (default: auto-detect)")
def main(data_dir: str, ckpt_path: str, batch_size: int, device: str):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Using device: {device}")

    # Load model
    vit_model, map_head = load_vit_from_checkpoint(ckpt_path, device)

    # Open source zarr (read-write to save features back)
    zarr_path = os.path.join(data_dir, "episode_data.zarr")
    store = zarr.open(zarr_path, mode="r+")
    episode_keys = sorted(
        [k for k in store.keys() if k.startswith("episode_")],
        key=lambda x: int(x.split("_")[1]),
    )
    print(f"Found {len(episode_keys)} episodes in {zarr_path}")

    # episode_keys = episode_keys[:250]
    episode_keys = episode_keys[250:500]

    for ep_key in tqdm(episode_keys, desc="Extracting features"):
        if "third_person_camera_feature" in store[ep_key]:
            # Remove the existing feature dataset
            del store[ep_key]["third_person_camera_feature"]
        images = store[ep_key]["third_person_camera"][:]  # (T, 256, 256, 3)
        features = extract_features(images, vit_model, map_head, device, batch_size)
        store[ep_key].create_dataset(
            "third_person_camera_feature",
            data=features,
            dtype="float32",
            chunks=(min(len(features), 128), 768),
        )

    print(f"Features saved to {zarr_path}")
    print(f"Feature shape per episode: (T, 768)")


if __name__ == "__main__":
    main()
    # load_vit_from_checkpoint("data/pick_and_match_color/2026-02-13/20-16-21_match_color_diffusion_memory_lr3e-4/checkpoints/epoch_2_train_mean_loss_0_001.ckpt", torch.device("cpu"))
    # load_vit_from_checkpoint("data/pick_and_match_color/2026-03-23/18-17-33_match_color_diffusion_memory_lr3e-4/checkpoints/epoch_10_train_mean_loss_0_000.ckpt", torch.device("cpu"))
    # override_vit_to_ckpt("data/pick_and_match_color/2026-03-23/18-17-33_match_color_diffusion_memory_lr3e-4/checkpoints/epoch_10_train_mean_loss_0_000_vit.ckpt", "data/pick_and_match_color_rand_delay/2026-03-24/09-53-47_match_color_rand_delay_diffusion_memory_lr3e-4_feature_only/checkpoints/epoch_10_train_mean_loss_0_000.ckpt")
