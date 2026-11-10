# SPDX-License-Identifier: Apache-2.0
"""Convert locally trusted upstream nuScenes temporal infos to a portable JSON manifest.

Uses the supplied camera extrinsics and full CAN-bus vector; does not invent
calibration/motion. Pickle files can execute code: the explicit trust flag is
required and is appropriate only for locally generated or trusted upstream files.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from recorded_scene import CAMERAS, sha256


def image_path(raw, root):
    path = Path(raw)
    candidates = [path] if path.is_absolute() else [root / path]
    # Official infos can contain a previous workspace's data/nuscenes prefix.
    for part in ("samples", "sweeps"):
        if part in path.parts:
            candidates.append(root.joinpath(*path.parts[path.parts.index(part) :]))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    raise FileNotFoundError(f"Image absent under dataset root: {raw}")


def convert(info, root):
    cameras = {}
    for name in CAMERAS:
        cam = info["cams"][name]
        # sensor2lidar maps camera coordinates to the keyframe LiDAR frame.
        sensor2lidar = np.eye(4)
        sensor2lidar[:3, :3] = np.asarray(cam["sensor2lidar_rotation"])
        sensor2lidar[:3, 3] = np.asarray(cam["sensor2lidar_translation"])
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = np.asarray(cam["cam_intrinsic"])
        cameras[name] = dict(
            image=image_path(cam["data_path"], root),
            lidar2img=(intrinsic @ np.linalg.inv(sensor2lidar)).tolist(),
            timestamp_us=int(cam["timestamp"]),
        )
    return dict(
        sample_token=info["token"],
        scene_token=info["scene_token"],
        prev_token=info["prev"],
        timestamp_us=int(info["timestamp"]),
        ego_translation=np.asarray(info["ego2global_translation"]).tolist(),
        ego_rotation_wxyz=np.asarray(info["ego2global_rotation"]).tolist(),
        can_bus=np.asarray(info["can_bus"]).tolist(),
        cameras=cameras,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--infos", type=Path, required=True)
    p.add_argument("--trust-info-pickle", action="store_true")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--scene-token", required=True)
    p.add_argument("--frames", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not args.trust_info_pickle:
        p.error(
            "Use --trust-info-pickle only after verifying the provenance of the local infos file"
        )
    if args.frames < 1:
        p.error("--frames must be positive")
    with args.infos.open("rb") as f:
        payload = pickle.load(f)
    infos = sorted(
        [i for i in payload["infos"] if i["scene_token"] == args.scene_token],
        key=lambda i: i["timestamp"],
    )
    if len(infos) < args.frames:
        p.error(f"Scene contains only {len(infos)} frames")
    if infos[0]["prev"]:
        p.error(
            "Infos omit the scene start; provide the full scene for correct recurrence"
        )
    manifest = dict(
        schema="bevformer-recorded-v1",
        input_kind="recorded",
        source_infos_sha256=sha256(args.infos),
        frames=[convert(i, args.data_root.resolve()) for i in infos[: args.frames]],
    )
    from recorded_scene import load_manifest

    # Write beside the requested output so validation resolves relative names identically.
    with args.output.open("x") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    load_manifest(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
