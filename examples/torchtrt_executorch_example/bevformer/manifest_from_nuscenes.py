# SPDX-License-Identifier: Apache-2.0
"""Read nuScenes JSON tables directly, without the upstream training framework.

Projection and CAN-bus conventions follow BEVFormer commit 66b65f3's
tools/data_converter/nuscenes_converter.py. Missing CAN-bus data is an error.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from recorded_scene import CAMERAS, load_manifest, sha256


def transform(record):
    """Homogeneous local-to-parent transform from a w,x,y,z quaternion."""
    w, x, y, z = np.asarray(record["rotation"], dtype=np.float64)
    if not np.isclose(w * w + x * x + y * y + z * z, 1, atol=1e-5):
        raise ValueError("Non-unit calibration/pose quaternion")
    out = np.eye(4)
    out[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    out[:3, 3] = record["translation"]
    return out


def upstream_bus(poses, timestamp):
    """Preserve the pinned converter's exact timestamp-selection convention.

    It takes position/orientation from the last message at/before the sample
    (or the first available message), but accel/rotation_rate/vel from the
    loop's final message, which is the first later message when one exists.
    The dataset layer subsequently replaces position/orientation with ego pose.
    """
    if not poses or any(b["utime"] < a["utime"] for a, b in zip(poses, poses[1:])):
        raise ValueError("CAN-bus poses must be present and timestamp ordered")
    last = poses[0]
    for pose in poses:
        if pose["utime"] > timestamp:
            break
        last = pose
    return (
        list(last["pos"])
        + list(last["orientation"])
        + [value for key in ("accel", "rotation_rate", "vel") for value in pose[key]]
        + [0.0, 0.0]
    )


def build_manifest(root, version, scene_name, count):
    root = Path(root).resolve()
    tables, hashes = {}, {}
    for name in (
        "scene",
        "sample",
        "sample_data",
        "calibrated_sensor",
        "ego_pose",
        "sensor",
    ):
        path = root / version / (name + ".json")
        records = json.loads(path.read_text())
        tables[name] = {r["token"]: r for r in records}
        hashes[name] = sha256(path)
    scenes = [s for s in tables["scene"].values() if s["name"] == scene_name]
    if len(scenes) != 1:
        raise ValueError(f"Scene not found uniquely: {scene_name}")
    scene = scenes[0]
    if not 1 <= count <= scene["nbr_samples"]:
        raise ValueError("Frame count exceeds scene length or is not positive")
    bus_path = root / "can_bus" / (scene_name + "_pose.json")
    poses = json.loads(bus_path.read_text())
    by_sample = {}
    for sd in tables["sample_data"].values():
        if sd["is_key_frame"]:
            calibration = tables["calibrated_sensor"][sd["calibrated_sensor_token"]]
            channel = tables["sensor"][calibration["sensor_token"]]["channel"]
            channels = by_sample.setdefault(sd["sample_token"], {})
            if channel in channels:
                raise ValueError("Duplicate keyframe sensor record")
            channels[channel] = sd
    frames = []
    token = scene["first_sample_token"]
    for _ in range(count):
        sample = tables["sample"][token]
        sensors = by_sample[token]
        lidar = sensors["LIDAR_TOP"]
        lidar_pose = tables["ego_pose"][lidar["ego_pose_token"]]
        lidar_calib = tables["calibrated_sensor"][lidar["calibrated_sensor_token"]]
        lidar_to_global = transform(lidar_pose) @ transform(lidar_calib)
        cameras = {}
        for name in CAMERAS:
            sd = sensors[name]
            calib = tables["calibrated_sensor"][sd["calibrated_sensor_token"]]
            pose = tables["ego_pose"][sd["ego_pose_token"]]
            camera_to_global = transform(pose) @ transform(calib)
            k = np.eye(4)
            k[:3, :3] = calib["camera_intrinsic"]
            image = (root / sd["filename"]).resolve(strict=True)
            cameras[name] = dict(
                image=str(image),
                timestamp_us=sd["timestamp"],
                lidar2img=(
                    k @ np.linalg.inv(camera_to_global) @ lidar_to_global
                ).tolist(),
            )
        frames.append(
            dict(
                sample_token=token,
                scene_token=scene["token"],
                prev_token=sample["prev"],
                timestamp_us=sample["timestamp"],
                ego_translation=lidar_pose["translation"],
                ego_rotation_wxyz=lidar_pose["rotation"],
                can_bus=upstream_bus(poses, sample["timestamp"]),
                cameras=cameras,
            )
        )
        token = sample["next"]
    return dict(
        schema="bevformer-recorded-v1",
        input_kind="recorded",
        version=version,
        scene_name=scene_name,
        table_sha256=hashes,
        can_bus_sha256=sha256(bus_path),
        convention="BEVFormer 66b65f3 converter, including CAN-bus timestamp selection",
        frames=frames,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--scene", default="scene-0061")
    p.add_argument("--frames", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    manifest = build_manifest(args.data_root, args.version, args.scene, args.frames)
    with args.output.open("x") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    load_manifest(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
