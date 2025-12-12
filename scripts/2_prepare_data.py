#!/usr/bin/env python3
"""Prepare calibration data with SE(3) interpolation for RWHEC solver."""

import numpy as np
import yaml
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from scipy.spatial.transform import Rotation

from preprocessing.optitrack_parser_se3 import OptiTrackParserSE3


def invert_quat_trans(quat_wxyz: np.ndarray, trans: np.ndarray):
    """Invert a rigid body transformation."""
    r = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    R = r.as_matrix()
    R_inv = R.T
    t_inv = -R_inv @ trans
    r_inv = Rotation.from_matrix(R_inv)
    q_inv = r_inv.as_quat()
    return np.array([q_inv[3], q_inv[0], q_inv[1], q_inv[2]]), t_inv


# OptiTrack Y-up to OpenCV Y-down
R_OT_TO_CV = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])


def transform_optitrack_to_opencv(quat_wxyz: np.ndarray, trans: np.ndarray):
    """Convert OptiTrack coordinates to OpenCV frame."""
    r = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    R = r.as_matrix()
    R_new = R_OT_TO_CV @ R @ R_OT_TO_CV.T
    t_new = R_OT_TO_CV @ trans
    r_new = Rotation.from_matrix(R_new)
    q_new = r_new.as_quat()
    return np.array([q_new[3], q_new[0], q_new[1], q_new[2]]), t_new


@dataclass
class SyncConfig:
    video_file: str
    flip_frame: int
    flip_time: float
    fps: float
    offset: float


@dataclass 
class PosePair:
    video_frame: int
    video_time: float
    optitrack_time: float
    tag_id: int
    A_pose: np.ndarray
    B_pose: np.ndarray
    A_quat: np.ndarray
    A_trans: np.ndarray
    B_quat: np.ndarray
    B_trans: np.ndarray


def load_sync_config(yaml_path: str) -> Dict[str, SyncConfig]:
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    configs = {}
    for cam in ['left_camera', 'right_camera']:
        if cam in data:
            d = data[cam]
            configs[cam.replace('_camera', '')] = SyncConfig(
                video_file=d.get('file', ''),
                flip_frame=d.get('flip_frame', 0),
                flip_time=d.get('flip_time', 0),
                fps=d.get('fps', 30.0),
                offset=d.get('offset', 0)
            )
    return configs


def load_apriltag_detections(json_path: str) -> Dict[int, List[dict]]:
    with open(json_path, 'r') as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


def create_matched_pairs(ot_parser, detections, cfg, cam_name) -> Dict[int, List[PosePair]]:
    """Match AprilTag detections with interpolated OptiTrack poses."""
    pairs_by_tag = {}
    total, skipped = 0, 0

    for frame_str, dets in detections.items():
        frame = int(frame_str)
        video_time = frame / cfg.fps
        ot_time = video_time + cfg.offset

        ot_pose = ot_parser.get_pose_interpolated(ot_time)
        if ot_pose is None:
            skipped += 1
            continue

        # A matrix: OptiTrack pose (coord-transformed)
        A_quat_raw = ot_pose.get_quaternion_wxyz()
        A_trans_raw = ot_pose.get_position_meters()
        A_quat, A_trans = transform_optitrack_to_opencv(A_quat_raw, A_trans_raw)

        A = np.eye(4)
        r_a = Rotation.from_quat([A_quat[1], A_quat[2], A_quat[3], A_quat[0]])
        A[:3, :3] = r_a.as_matrix()
        A[:3, 3] = A_trans

        for det in dets:
            tag_id = det['tag_id']

            # B matrix: inverted AprilTag pose (camera->tag becomes tag->camera)
            B_quat_raw = np.array(det['quaternion_wxyz'])
            B_trans_raw = np.array(det['translation'])
            B_quat, B_trans = invert_quat_trans(B_quat_raw, B_trans_raw)

            B = np.eye(4)
            r_b = Rotation.from_quat([B_quat[1], B_quat[2], B_quat[3], B_quat[0]])
            B[:3, :3] = r_b.as_matrix()
            B[:3, 3] = B_trans

            pair = PosePair(
                video_frame=frame, video_time=video_time, optitrack_time=ot_time,
                tag_id=tag_id, A_pose=A, B_pose=B,
                A_quat=A_quat, A_trans=A_trans, B_quat=B_quat, B_trans=B_trans
            )

            if tag_id not in pairs_by_tag:
                pairs_by_tag[tag_id] = []
            pairs_by_tag[tag_id].append(pair)
            total += 1

    print(f"  {cam_name}: {total} matches, {skipped} skipped")
    return pairs_by_tag


def export_julia_format(pairs_by_tag, output_dir: Path, cam_name: str):
    """Export pose pairs in format expected by Julia solver."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cam_idx = 0 if cam_name == 'left' else 1
    
    for tag_id, pairs in pairs_by_tag.items():
        if len(pairs) < 1:
            continue
            
        a_path = output_dir / f"tag_{tag_id}_cam_{cam_idx}_A.csv"
        b_path = output_dir / f"tag_{tag_id}_cam_{cam_idx}_B.csv"
        
        with open(a_path, 'w', newline='') as fa, open(b_path, 'w', newline='') as fb:
            wa, wb = csv.writer(fa), csv.writer(fb)
            for p in pairs:
                wa.writerow([p.A_quat[0], p.A_quat[1], p.A_quat[2], p.A_quat[3],
                            p.A_trans[0], p.A_trans[1], p.A_trans[2]])
                wb.writerow([p.B_quat[0], p.B_quat[1], p.B_quat[2], p.B_quat[3],
                            p.B_trans[0], p.B_trans[1], p.B_trans[2]])
        
        print(f"    Tag {tag_id}: {len(pairs)} poses")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--optitrack', default='data/session_008/5550 Take 12-08 03.08.47 PM.csv')
    parser.add_argument('--sync', default='data/session_008/sync_offsets.yaml')
    parser.add_argument('--detections', default='output/detections')
    parser.add_argument('--output', default='output/julia_data')
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    
    print("Loading OptiTrack data...")
    ot = OptiTrackParserSE3(str(root / args.optitrack))
    
    sync_cfgs = load_sync_config(str(root / args.sync))
    output_path = root / args.output
    det_path = root / args.detections

    for cam in ['left', 'right']:
        print(f"Processing {cam} camera...")
        det_json = det_path / f"{cam}_detections.json"
        if not det_json.exists():
            print(f"  {det_json} not found, skipping")
            continue

        detections = load_apriltag_detections(str(det_json))
        if cam not in sync_cfgs:
            continue

        pairs = create_matched_pairs(ot, detections, sync_cfgs[cam], cam)
        export_julia_format(pairs, output_path, cam)

    print(f"Done. Output: {output_path}")


if __name__ == '__main__':
    main()