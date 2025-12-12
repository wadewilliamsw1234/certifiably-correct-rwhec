#!/usr/bin/env python3
"""Evaluate RWHEC calibration results."""

import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path
import glob
import argparse
import json


def load_pose(path):
    """Load pose from CSV (qw, qx, qy, qz, tx, ty, tz)."""
    data = np.loadtxt(path, delimiter=',')
    qw, qx, qy, qz = data[:4]
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = data[4:7]
    return T


def load_poses(path):
    """Load multiple poses from CSV."""
    data = np.loadtxt(path, delimiter=',')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    poses = []
    for row in data:
        qw, qx, qy, qz = row[:4]
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = row[4:7]
        poses.append(T)
    return poses


def compute_consistency_error(A, X, Y, B):
    """Compute AX = YB consistency error."""
    AX = A @ X
    YB = Y @ B
    trans_err = np.linalg.norm(AX[:3, 3] - YB[:3, 3])
    R_diff = AX[:3, :3].T @ YB[:3, :3]
    trace = np.clip((np.trace(R_diff) - 1) / 2, -1, 1)
    rot_err = np.degrees(np.abs(np.arccos(trace)))
    return trans_err, rot_err


def evaluate(calib_dir, data_dir):
    calib_dir, data_dir = Path(calib_dir), Path(data_dir)
    
    if not (calib_dir / "left_X.csv").exists():
        return None
    
    X_left = load_pose(calib_dir / "left_X.csv")
    X_right = load_pose(calib_dir / "right_X.csv")
    Xs = {0: X_left, 1: X_right}
    
    # Baseline between cameras
    baseline = np.linalg.norm(X_left[:3, 3] - X_right[:3, 3])
    
    # Angle between optical axes
    z_left = X_left[:3, :3][:, 2]
    z_right = X_right[:3, :3][:, 2]
    angle = np.degrees(np.arccos(np.clip(np.dot(z_left, z_right), -1, 1)))
    
    # Load Y transforms
    Ys = {}
    for yf in glob.glob(str(calib_dir / "Y_tag_*.csv")):
        tag_id = int(Path(yf).stem.split('_')[-1])
        Ys[tag_id] = load_pose(yf)
    
    # Compute consistency errors
    errors = []
    for cam_idx in [0, 1]:
        X = Xs[cam_idx]
        for tag_id in Ys:
            a_file = data_dir / f"tag_{tag_id}_cam_{cam_idx}_A.csv"
            if not a_file.exists():
                continue
            A_poses = load_poses(a_file)
            B_poses = load_poses(data_dir / f"tag_{tag_id}_cam_{cam_idx}_B.csv")
            Y = Ys[tag_id]
            for A, B in zip(A_poses, B_poses):
                err, _ = compute_consistency_error(A, X, Y, B)
                errors.append(err)
    
    if not errors:
        return None
    
    return {
        'baseline_cm': baseline * 100,
        'angle_deg': angle,
        'mean_error_cm': np.mean(errors) * 100,
        'std_error_cm': np.std(errors) * 100,
        'max_error_cm': np.max(errors) * 100,
        'n_poses': len(errors),
        'X_left': X_left,
        'X_right': X_right
    }


def print_results(r):
    if r is None:
        print("No valid calibration found")
        return
    
    baseline_ok = "[OK]" if 10 <= r['baseline_cm'] <= 13 else "[!]"
    angle_ok = "[OK]" if 45 <= r['angle_deg'] <= 90 else "[!]"
    
    print("-" * 50)
    print("RWHEC Calibration Results")
    print("-" * 50)
    print(f"Baseline:     {r['baseline_cm']:.2f} cm {baseline_ok} (expected 10-13)")
    print(f"Axis angle:   {r['angle_deg']:.1f} deg {angle_ok} (expected 45-90)")
    print(f"Mean error:   {r['mean_error_cm']:.2f} cm")
    print(f"Max error:    {r['max_error_cm']:.2f} cm")
    print(f"Poses:        {r['n_poses']}")
    print()
    
    for name, X in [('Left', r['X_left']), ('Right', r['X_right'])]:
        t = X[:3, 3]
        euler = Rotation.from_matrix(X[:3, :3]).as_euler('xyz', degrees=True)
        print(f"{name} camera:")
        print(f"  t = [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m")
        print(f"  R = [{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}] deg")
    
    print("-" * 50)
    if baseline_ok == "[OK]" and angle_ok == "[OK]":
        print("Calibration valid")
    else:
        print("Warning: some metrics outside expected range")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calib', default='output/final')
    parser.add_argument('--data', default='output/julia_data')
    parser.add_argument('--json', default=None, help='Save results to JSON')
    args = parser.parse_args()
    
    root = Path(__file__).parent.parent
    results = evaluate(root / args.calib, root / args.data)
    print_results(results)
    
    if args.json and results:
        out = {k: v for k, v in results.items() if not isinstance(v, np.ndarray)}
        with open(root / args.json, 'w') as f:
            json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()