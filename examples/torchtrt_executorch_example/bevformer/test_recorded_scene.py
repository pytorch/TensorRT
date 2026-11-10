# SPDX-License-Identifier: Apache-2.0
"""Geometry, preprocessing and evidence checks independent of GPU availability."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from manifest_from_infos import convert
from manifest_from_nuscenes import transform, upstream_bus
from recorded_scene import (
    CAMERAS,
    corners,
    decode,
    load_manifest,
    motion_bus,
    preprocess,
    project_edge,
    sha256,
)


def frame(yaw=0, token="a", prev="", timestamp=1, scene="scene", position=(0, 0, 0)):
    return dict(
        sample_token=token,
        prev_token=prev,
        timestamp_us=timestamp,
        scene_token=scene,
        ego_translation=list(position),
        ego_rotation_wxyz=[
            math.cos(math.radians(yaw) / 2),
            0,
            0,
            math.sin(math.radians(yaw) / 2),
        ],
        can_bus=list(range(18)),
        cameras={},
    )


class RecordedTests(unittest.TestCase):
    def test_motion_reset_and_upstream_yaw_wrap(self):
        a = frame(359, position=(10, 20, 0))
        b = frame(1, "b", "a", 2, position=(11, 18, 1))
        bus, history = motion_bus(a, None)
        self.assertFalse(history)
        np.testing.assert_equal(bus[0, :3], 0)
        self.assertEqual(bus[0, -1], 0)
        bus, history = motion_bus(b, a)
        self.assertTrue(history)
        np.testing.assert_equal(bus[0, :3], [1, -2, 1])
        self.assertAlmostEqual(float(bus[0, -1]), -358)
        self.assertAlmostEqual(float(bus[0, -2]), math.pi / 180)
        np.testing.assert_equal(bus[0, 7:16], np.arange(7, 16))
        b["scene_token"] = "new"
        bus, history = motion_bus(b, a)
        self.assertFalse(history)
        np.testing.assert_equal(bus[0, :3], 0)

    def test_reject_skipped_or_reversed_frames(self):
        a = frame()
        for b in (
            frame(token="b", prev="other", timestamp=2),
            frame(token="b", prev="a", timestamp=0),
        ):
            with self.assertRaises(ValueError):
                motion_bus(b, a)

    def test_raw_pose_transform_and_can_timestamps(self):
        pose = dict(
            rotation=[math.sqrt(0.5), 0, 0, math.sqrt(0.5)], translation=[10, 20, 30]
        )
        np.testing.assert_allclose(
            transform(pose) @ [1, 0, 0, 1], [10, 21, 30, 1], atol=1e-12
        )
        poses = [
            dict(
                utime=t,
                pos=[t] * 3,
                orientation=[1, 0, 0, 0],
                accel=[t + 1] * 3,
                rotation_rate=[t + 2] * 3,
                vel=[t + 3] * 3,
            )
            for t in (10, 20, 30)
        ]
        for timestamp, prior, final in [
            (0, 10, 10),
            (10, 10, 20),
            (15, 10, 20),
            (30, 30, 30),
        ]:
            bus = upstream_bus(poses, timestamp)
            self.assertEqual(bus[:3], [prior] * 3)
            self.assertEqual(bus[7:10], [final + 1] * 3)
            self.assertEqual(len(bus), 18)
        self.assertIn("utime", poses[0])
        with self.assertRaises(ValueError):
            upstream_bus([], 10)
        with self.assertRaises(ValueError):
            upstream_bus(list(reversed(poses)), 10)

    def test_projection_clip(self):
        matrix = np.array(
            [[100, 0, 50, 0], [0, 100, 50, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        out = project_edge([0, 0, 2], [1, 0, 2], matrix)
        np.testing.assert_allclose(out, [[50, 50], [100, 50]])
        self.assertIsNone(project_edge([0, 0, -2], [1, 0, -1], matrix))
        self.assertTrue(np.isfinite(project_edge([0, 0, -1], [1, 0, 2], matrix)).all())

    def test_box_dimensions_yaw_and_bottom(self):
        box = dict(cx=10, cy=20, w=2, l=4, h=3, yaw=math.pi / 2, z_bottom=-1)
        pts = corners(box)
        np.testing.assert_allclose(np.ptp(pts, axis=0), [4, 2, 3])
        np.testing.assert_allclose(pts[:4, 2], -1)
        np.testing.assert_allclose(pts[4:, 2], 2)

    def test_decoder_range_before_bottom_and_topk(self):
        logits = np.full((900, 10), -100, dtype=np.float32)
        boxes = np.zeros((900, 10), dtype=np.float32)
        boxes[:, 7] = 1
        logits[0, 3] = 5
        boxes[0, 4] = -9
        boxes[0, 5] = np.log(16)
        logits[1, 0] = 4
        boxes[1, 0] = 62
        result = decode(logits, boxes, 0.3)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["label"], 3)
        self.assertAlmostEqual(result[0]["z_bottom"], -17, places=5)
        logits.fill(0)
        indices = decode(logits, boxes, 0)
        self.assertLessEqual(len(indices), 300)
        self.assertEqual(indices[0]["query"], 0)
        self.assertEqual(decode(np.full((900, 10), -100), boxes, 0.3), [])

    def test_preprocess_color_padding_and_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.png"
            image = np.full((900, 1600, 3), [10, 20, 30], dtype=np.uint8)
            cv2.imwrite(str(path), image)
            f = frame()
            f["cameras"] = {
                n: dict(image=str(path), lidar2img=np.eye(4).tolist()) for n in CAMERAS
            }
            images, matrices = preprocess(f)
            self.assertEqual(images.shape, (1, 6, 3, 480, 800))
            expected = (np.array([30, 20, 10]) - [123.675, 116.28, 103.53]) / [
                58.395,
                57.12,
                57.375,
            ]
            np.testing.assert_allclose(images[0, 0, :, 0, 0], expected, rtol=1e-6)
            np.testing.assert_equal(images[..., 450:, :], 0)
            np.testing.assert_equal(matrices[0, 0], np.diag([0.5, 0.5, 1, 1]))
            manifest = dict(
                schema="bevformer-recorded-v1", input_kind="synthetic-test", frames=[f]
            )
            m = Path(tmp) / "scene.json"
            m.write_text(json.dumps(manifest))
            self.assertEqual(load_manifest(m)["frames"][0]["sample_token"], "a")
            del manifest["frames"][0]["cameras"]["CAM_BACK"]
            m.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_manifest(m)

    def test_infos_projection_matches_upstream_formula(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "samples" / "image.jpg"
            path.parent.mkdir()
            path.touch()
            rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
            trans = np.array([2.0, 3, 4])
            intrinsic = np.diag([800.0, 810, 1])
            cam = dict(
                data_path="old/data/nuscenes/samples/image.jpg",
                sensor2lidar_rotation=rot,
                sensor2lidar_translation=trans,
                cam_intrinsic=intrinsic,
                timestamp=1,
            )
            info = dict(
                token="a",
                scene_token="s",
                prev="",
                timestamp=1,
                ego2global_translation=[0, 0, 0],
                ego2global_rotation=[1, 0, 0, 0],
                can_bus=np.arange(18),
                cams=dict.fromkeys(CAMERAS, cam),
            )
            result = convert(info, Path(tmp))
            r = np.linalg.inv(rot)
            t = trans @ r.T
            rt = np.eye(4)
            rt[:3, :3] = r.T
            rt[3, :3] = -t
            k = np.eye(4)
            k[:3, :3] = intrinsic
            np.testing.assert_allclose(
                result["cameras"]["CAM_FRONT"]["lidar2img"], k @ rt.T
            )

    def test_renderer_provenance_and_native_mismatch(self):
        from render_recorded import render

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = root / "case"
            case.mkdir()
            native = root / "native"
            native.mkdir()
            image = root / "camera.png"
            cv2.imwrite(str(image), np.full((900, 1600, 3), 120, np.uint8))
            f = frame()
            f["cameras"] = {
                n: dict(image=str(image), lidar2img=np.eye(4).tolist()) for n in CAMERAS
            }
            record = dict(
                frame=f,
                input_kind="synthetic-test",
                history=False,
                image_sha256={n: sha256(image) for n in CAMERAS},
                outputs=[],
            )
            for i, shape in enumerate(((1, 900, 10), (1, 900, 10), (1, 2500, 256))):
                a = np.zeros(shape, dtype="<f4")
                if i == 0:
                    a.fill(-100)
                if i == 1:
                    a[..., 7] = 1
                a.tofile(case / f"output_{i}.bin")
                a.tofile(native / f"output_{i}.bin")
                record["outputs"].append(dict(sha256=sha256(case / f"output_{i}.bin")))
            (case / "frame.json").write_text(json.dumps(record))
            result = render(case, native, root / "figure.png", "native")
            self.assertEqual(result["input_kind"], "synthetic-test")
            self.assertEqual(result["detections"], 0)
            self.assertGreater((root / "figure.png").stat().st_size, 10000)
            self.assertTrue((root / "figure.json").is_file())
            with self.assertRaises(ValueError):
                render(case, case, root / "invalid.png", "native")
            with (native / "output_0.bin").open("r+b") as out:
                out.write(np.float32(100).tobytes())
            with self.assertRaises(ValueError):
                render(case, native, root / "mismatch.png", "native")
            cv2.imwrite(str(image), np.full((900, 1600, 3), 50, np.uint8))
            with self.assertRaises(ValueError):
                render(case, case, root / "wrong-image.png", "pytorch")


if __name__ == "__main__":
    unittest.main()
