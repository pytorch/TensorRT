"""LeRobot frame loading used by PI05 / GR00T ``prepare_sample_inputs``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME, HF_LEROBOT_HUB_CACHE, OBS_STATE
from PIL import Image

IMAGE_KEYS = ("observation.images.image", "observation.images.image2")
DEFAULT_DATASET_ID = "lerobot/libero"
DEFAULT_DATASET_REVISION = "v3.0"


def _lerobot_dataset_has_metadata(root: Path) -> bool:
    return (root / "meta" / "info.json").is_file()


def _resolve_lerobot_dataset_root(
    dataset_id: str,
    revision: str = DEFAULT_DATASET_REVISION,
) -> Path | None:
    materialized = HF_LEROBOT_HOME / dataset_id
    if _lerobot_dataset_has_metadata(materialized):
        return materialized

    hub_dir = HF_LEROBOT_HUB_CACHE / f"datasets--{dataset_id.replace('/', '--')}"
    ref_file = hub_dir / "refs" / revision
    if ref_file.is_file():
        snapshot = hub_dir / "snapshots" / ref_file.read_text().strip()
        if _lerobot_dataset_has_metadata(snapshot):
            return snapshot

    snapshots_dir = hub_dir / "snapshots"
    if snapshots_dir.is_dir():
        for snapshot in sorted(snapshots_dir.iterdir(), reverse=True):
            if snapshot.is_dir() and _lerobot_dataset_has_metadata(snapshot):
                return snapshot
    return None


def frame_from_test_data(
    data: dict[str, Any],
    policy: Any,
    *,
    fill_missing: bool = False,
) -> dict[str, Any]:
    frame = dict(data["images"])
    frame[OBS_STATE] = data["state"]
    frame["task"] = data.get("task", "")
    if fill_missing:
        for key, feature in policy.config.input_features.items():
            if key.startswith("observation.images.") and key not in frame:
                frame[key] = torch.zeros(feature.shape, dtype=torch.float32)
    return frame


def load_test_data(
    dataset_id: str = DEFAULT_DATASET_ID,
    *,
    episode_index: int = 0,
    frame_index: int = 0,
) -> dict[str, Any]:
    local_root = _resolve_lerobot_dataset_root(dataset_id)
    dataset_kwargs: dict[str, Any] = {
        "episodes": [episode_index],
        "video_backend": "pyav",
        "revision": DEFAULT_DATASET_REVISION,
    }
    if local_root is not None:
        dataset_kwargs["root"] = local_root
    dataset = LeRobotDataset(dataset_id, **dataset_kwargs)
    frame = dataset[frame_index]
    images = {key: frame[key] for key in IMAGE_KEYS if key in frame}
    return {
        "images": images,
        "state": frame[OBS_STATE],
        "task": frame.get("task", "") or "Perform the task.",
    }


def create_pil_messages(data: dict[str, Any]) -> list[dict[str, Any]]:
    images = data["images"]
    task = str(data.get("task", "") or "Perform the task.")
    image_content = [
        {"type": "image", "image": _tensor_image_to_pil(img)}
        for _, img in sorted(images.items())
    ]
    return [
        {
            "role": "user",
            "content": image_content + [{"type": "text", "text": str([task])}],
        }
    ]


def pack_state(
    state: torch.Tensor,
    max_state_dim: int,
    device: str | torch.device,
) -> torch.Tensor:
    state = torch.as_tensor(state, dtype=torch.float32, device=device)
    if state.ndim == 1:
        state = state.unsqueeze(0)
    if state.ndim == 2:
        state = state.unsqueeze(1)
    bsz, _, state_dim = state.shape
    if state_dim > max_state_dim:
        state = state[:, :, :max_state_dim]
    elif state_dim < max_state_dim:
        pad = torch.zeros(
            bsz,
            1,
            max_state_dim - state_dim,
            dtype=state.dtype,
            device=device,
        )
        state = torch.cat([state, pad], dim=-1)
    return state


def _tensor_image_to_pil(img: torch.Tensor) -> Image.Image:
    img = img.detach().cpu()
    if img.dtype.is_floating_point:
        img = (img.clamp(0, 1) * 255).to(torch.uint8)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = img.permute(1, 2, 0)
    return Image.fromarray(img.numpy())
