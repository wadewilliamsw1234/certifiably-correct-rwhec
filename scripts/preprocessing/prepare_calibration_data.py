#!/usr/bin/env python3
"""
RWHEC Calibration Data Preparation

Synchronizes OptiTrack motion capture data with AprilTag detections
and exports matched A,B pose pairs for the Julia RWHEC solver.

A = Robot base to hand transformation (OptiTrack rigid body pose)
B = Camera to target transformation (AprilTag detection)

The RWHEC problem solves: AX = YB
- X = hand-to-camera transformation (hand-eye calibration)
- Y = robot-base-to-target transformation (robot-world calibration)
"""

import numpy as np
import yaml
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial.transform import Rotation

from optitrack_parser import OptiTrackParser, OptiTrackPose


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 homogeneous transformation matrix."""
    T_inv = np.eye(4)
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def invert_quat_trans(quat_wxyz: np.ndarray, trans: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Invert a transformation given as quaternion (wxyz) and translation."""
    # Build 4x4 matrix
    r = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    R = r.as_matrix()

    # Invert: R_inv = R^T, t_inv = -R^T @ t
    R_inv = R.T
    t_inv = -R_inv @ trans

    # Convert back to quaternion
    r_inv = Rotation.from_matrix(R_inv)
    q_inv_xyzw = r_inv.as_quat()
    q_inv_wxyz = np.array([q_inv_xyzw[3], q_inv_xyzw[0], q_inv_xyzw[1], q_inv_xyzw[2]])

    return q_inv_wxyz, t_inv


# Coordinate system transformation: OptiTrack (Y-up, Z-forward) to OpenCV (Y-down, Z-forward)
# This is a 180-degree rotation around the X-axis
R_OPTITRACK_TO_OPENCV = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1]
])


def transform_optitrack_to_opencv(quat_wxyz: np.ndarray, trans: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Transform OptiTrack coordinates to OpenCV coordinates."""
    # Convert quaternion to rotation matrix
    r = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    R = r.as_matrix()

    # Transform: R' = R_conv @ R @ R_conv^T, t' = R_conv @ t
    R_new = R_OPTITRACK_TO_OPENCV @ R @ R_OPTITRACK_TO_OPENCV.T
    t_new = R_OPTITRACK_TO_OPENCV @ trans

    # Convert back to quaternion
    r_new = Rotation.from_matrix(R_new)
    q_new_xyzw = r_new.as_quat()
    q_new_wxyz = np.array([q_new_xyzw[3], q_new_xyzw[0], q_new_xyzw[1], q_new_xyzw[2]])

    return q_new_wxyz, t_new


@dataclass
class SyncConfig:
    """Synchronization configuration for a camera."""
    video_file: str
    flip_frame: int
    flip_time: float
    fps: float
    offset: float  # optitrack_time = video_time + offset


@dataclass
class MatchedPosePair:
    """A synchronized A,B pose pair."""
    video_frame: int
    video_time: float
    optitrack_time: float
    tag_id: int
    A_pose: np.ndarray  # 4x4 transformation (OptiTrack)
    B_pose: np.ndarray  # 4x4 transformation (AprilTag)
    A_quat_wxyz: np.ndarray
    A_trans: np.ndarray
    B_quat_wxyz: np.ndarray
    B_trans: np.ndarray


def load_sync_config(yaml_path: str) -> Dict[str, SyncConfig]:
    """Load synchronization configuration from YAML."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    configs = {}
    for cam in ['left_camera', 'right_camera']:
        if cam in data:
            cam_data = data[cam]
            configs[cam.replace('_camera', '')] = SyncConfig(
                video_file=cam_data.get('file', ''),
                flip_frame=cam_data.get('flip_frame', 0),
                flip_time=cam_data.get('flip_time', 0),
                fps=cam_data.get('fps', 30.0),
                offset=cam_data.get('offset', 0)
            )

    return configs


def load_apriltag_detections(json_path: str) -> Dict[int, List[dict]]:
    """Load AprilTag detections from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Convert string keys to int
    return {int(k): v for k, v in data.items()}


def video_time_to_optitrack_time(video_time: float, sync_config: SyncConfig) -> float:
    """Convert video timestamp to OptiTrack timestamp."""
    return video_time + sync_config.offset


def create_matched_pairs(
    optitrack_parser: OptiTrackParser,
    apriltag_detections: Dict[int, List[dict]],
    sync_config: SyncConfig,
    camera_name: str
) -> Dict[int, List[MatchedPosePair]]:
    """
    Create matched A,B pose pairs by synchronizing timestamps.

    Args:
        optitrack_parser: Parsed OptiTrack data
        apriltag_detections: AprilTag detections keyed by video frame
        sync_config: Synchronization configuration
        camera_name: Name of camera for logging

    Returns:
        Dictionary mapping tag_id to list of MatchedPosePair
    """
    pairs_by_tag: Dict[int, List[MatchedPosePair]] = {}

    total_matches = 0
    skipped = 0

    for video_frame, detections in apriltag_detections.items():
        video_frame = int(video_frame)
        video_time = video_frame / sync_config.fps

        # Convert to OptiTrack time
        ot_time = video_time_to_optitrack_time(video_time, sync_config)

        # Get interpolated OptiTrack pose
        ot_pose = optitrack_parser.get_pose_interpolated(ot_time)
        if ot_pose is None:
            skipped += 1
            continue

        # Get A matrix (OptiTrack pose)
        # OptiTrack gives T_world_rig (world to rig) = T_base_hand
        # Convert from OptiTrack coords (Y-up) to OpenCV coords (Y-down)
        # Then use as-is (NOT inverted) per reference implementation
        A_quat_raw = ot_pose.get_quaternion_wxyz()
        A_trans_raw = ot_pose.get_position_meters()

        # Convert OptiTrack -> OpenCV coordinate system
        A_quat, A_trans = transform_optitrack_to_opencv(A_quat_raw, A_trans_raw)

        # Build A transformation matrix (coord-transformed, but NOT inverted)
        A = np.eye(4)
        r_a = Rotation.from_quat([A_quat[1], A_quat[2], A_quat[3], A_quat[0]])
        A[:3, :3] = r_a.as_matrix()
        A[:3, 3] = A_trans

        for det in detections:
            tag_id = det['tag_id']

            # Get B matrix (AprilTag pose)
            # AprilTag gives T_camera_tag (camera to tag)
            # Reference implementation expects T_target_camera (tag to camera)
            # So invert the AprilTag output
            B_quat_raw = np.array(det['quaternion_wxyz'])
            B_trans_raw = np.array(det['translation'])

            # Invert B (camera->tag becomes tag->camera)
            B_quat_wxyz, B_trans = invert_quat_trans(B_quat_raw, B_trans_raw)

            # Build B transformation matrix
            B = np.eye(4)
            r = Rotation.from_quat([B_quat_wxyz[1], B_quat_wxyz[2],
                                    B_quat_wxyz[3], B_quat_wxyz[0]])  # wxyz -> xyzw
            B[:3, :3] = r.as_matrix()
            B[:3, 3] = B_trans

            pair = MatchedPosePair(
                video_frame=video_frame,
                video_time=video_time,
                optitrack_time=ot_time,
                tag_id=tag_id,
                A_pose=A,
                B_pose=B,
                A_quat_wxyz=A_quat,
                A_trans=A_trans,
                B_quat_wxyz=B_quat_wxyz,
                B_trans=B_trans
            )

            if tag_id not in pairs_by_tag:
                pairs_by_tag[tag_id] = []
            pairs_by_tag[tag_id].append(pair)
            total_matches += 1

    print(f"  {camera_name}: Created {total_matches} matched pairs across {len(pairs_by_tag)} tags")
    if skipped > 0:
        print(f"  Warning: Skipped {skipped} frames (out of OptiTrack time range)")

    return pairs_by_tag


def export_julia_format(
    pairs_by_tag: Dict[int, List[MatchedPosePair]],
    output_dir: Path,
    camera_name: str
):
    """
    Export matched pairs to Julia solver format.

    Creates tag_X_cam_Y_A.csv and tag_X_cam_Y_B.csv files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Map camera name to index
    cam_idx = 0 if camera_name == 'left' else 1

    for tag_id, pairs in pairs_by_tag.items():
        # Sort by video frame
        pairs.sort(key=lambda p: p.video_frame)

        # Export A matrices
        a_path = output_dir / f"tag_{tag_id}_cam_{cam_idx}_A.csv"
        with open(a_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for pair in pairs:
                q = pair.A_quat_wxyz
                t = pair.A_trans
                writer.writerow([q[0], q[1], q[2], q[3], t[0], t[1], t[2]])

        # Export B matrices
        b_path = output_dir / f"tag_{tag_id}_cam_{cam_idx}_B.csv"
        with open(b_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for pair in pairs:
                q = pair.B_quat_wxyz
                t = pair.B_trans
                writer.writerow([q[0], q[1], q[2], q[3], t[0], t[1], t[2]])

        print(f"  Exported tag {tag_id} cam {cam_idx}: {len(pairs)} pairs")

    # Export frame index mapping for debugging
    index_path = output_dir / f"{camera_name}_matched_frames.csv"
    with open(index_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['video_frame', 'video_time', 'optitrack_time', 'tag_ids'])
        all_frames = {}
        for tag_id, pairs in pairs_by_tag.items():
            for pair in pairs:
                if pair.video_frame not in all_frames:
                    all_frames[pair.video_frame] = {
                        'video_time': pair.video_time,
                        'optitrack_time': pair.optitrack_time,
                        'tags': []
                    }
                all_frames[pair.video_frame]['tags'].append(tag_id)

        for frame in sorted(all_frames.keys()):
            data = all_frames[frame]
            writer.writerow([
                frame,
                data['video_time'],
                data['optitrack_time'],
                ','.join(map(str, data['tags']))
            ])


def prepare_calibration_data(
    project_root: Path,
    optitrack_csv: str,
    sync_yaml: str,
    detections_dir: str,
    output_dir: str
):
    """
    Main function to prepare calibration data.

    Args:
        project_root: Project root directory
        optitrack_csv: Path to OptiTrack CSV (relative to project root)
        sync_yaml: Path to sync config YAML (relative to project root)
        detections_dir: Directory with AprilTag detections
        output_dir: Output directory for Julia-format data
    """
    print("=" * 60)
    print("RWHEC Calibration Data Preparation")
    print("=" * 60)

    # Load OptiTrack data
    print("\n1. Loading OptiTrack data...")
    ot_path = project_root / optitrack_csv
    optitrack = OptiTrackParser(str(ot_path))

    # Load sync configuration
    print("\n2. Loading sync configuration...")
    sync_path = project_root / sync_yaml
    sync_configs = load_sync_config(str(sync_path))
    for cam, cfg in sync_configs.items():
        print(f"  {cam}: offset={cfg.offset:.4f}s, fps={cfg.fps}")

    # Process each camera
    output_path = project_root / output_dir
    det_path = project_root / detections_dir

    for camera in ['left', 'right']:
        print(f"\n3. Processing {camera} camera...")

        # Load AprilTag detections
        det_json = det_path / f"{camera}_detections.json"
        if not det_json.exists():
            print(f"  Warning: {det_json} not found, skipping")
            continue

        detections = load_apriltag_detections(str(det_json))
        print(f"  Loaded {sum(len(v) for v in detections.values())} detections from {len(detections)} frames")

        # Create matched pairs
        if camera not in sync_configs:
            print(f"  Warning: No sync config for {camera}, skipping")
            continue

        pairs = create_matched_pairs(
            optitrack,
            detections,
            sync_configs[camera],
            camera
        )

        # Export to Julia format
        print(f"\n4. Exporting {camera} camera data to Julia format...")
        export_julia_format(pairs, output_path, camera)

    # Summary
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print(f"Output directory: {output_path}")
    print("=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Prepare RWHEC calibration data')
    parser.add_argument('--optitrack', type=str,
                       default='data/session_008/5550 Take 12-08 03.08.47 PM.csv',
                       help='Path to OptiTrack CSV')
    parser.add_argument('--sync', type=str,
                       default='data/session_008/sync_offsets.yaml',
                       help='Path to sync config YAML')
    parser.add_argument('--detections', type=str,
                       default='output/detections',
                       help='Directory with AprilTag detections')
    parser.add_argument('--output', type=str,
                       default='output/julia_data',
                       help='Output directory')

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    prepare_calibration_data(
        project_root=project_root,
        optitrack_csv=args.optitrack,
        sync_yaml=args.sync,
        detections_dir=args.detections,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
