#!/usr/bin/env python3
"""
RANSAC-based outlier rejection following the paper's methodology.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path
import csv
import glob
from typing import Dict, List, Tuple
import random

def load_pose_csv(path):
    """Load poses from CSV file."""
    data = np.loadtxt(path, delimiter=',')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    poses = []
    raw_rows = []
    for row in data:
        qw, qx, qy, qz = row[:4]
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = row[4:7]
        poses.append(T)
        raw_rows.append(row)
    return poses, raw_rows

def load_single_pose(path):
    """Load single pose from CSV."""
    data = np.loadtxt(path, delimiter=',')
    qw, qx, qy, qz = data[:4]
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = data[4:7]
    return T

def save_poses(rows, path):
    """Save poses to CSV."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

def compute_pose_error(A, X, Y, B):
    """
    Compute translation and rotation error for AX = YB.
    
    Returns:
        trans_error: Translation error in meters
        rot_error: Rotation error in degrees
    """
    AX = A @ X
    YB = Y @ B
    
    # Translation error
    trans_error = np.linalg.norm(AX[:3, 3] - YB[:3, 3])
    
    # Rotation error (geodesic distance on SO(3))
    R_diff = AX[:3, :3].T @ YB[:3, :3]
    rot_error = np.abs(np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1)))
    rot_error_deg = np.degrees(rot_error)
    
    return trans_error, rot_error_deg

def count_inliers(all_data, X, Y_estimates, trans_thresh=0.6, rot_thresh=60.0):
    """
    Count inliers given estimated X and Y transformations.
    
    Paper thresholds: 0.6m translation, 60° rotation
    """
    inlier_count = 0
    inlier_indices = {}
    
    for (tag_id, cam_idx), (A_poses, B_poses, _) in all_data.items():
        if tag_id not in Y_estimates:
            continue
        Y = Y_estimates[tag_id]
        
        inlier_indices[(tag_id, cam_idx)] = []
        for i, (A, B) in enumerate(zip(A_poses, B_poses)):
            trans_err, rot_err = compute_pose_error(A, X, Y, B)
            if trans_err < trans_thresh and rot_err < rot_thresh:
                inlier_count += 1
                inlier_indices[(tag_id, cam_idx)].append(i)
    
    return inlier_count, inlier_indices

def estimate_Y_from_AXB(A_poses, B_poses, X):
    """
    Estimate Y given A, B poses and X.
    Y = A @ X @ B^(-1)
    
    Returns average Y estimate.
    """
    Y_estimates = []
    for A, B in zip(A_poses, B_poses):
        B_inv = np.linalg.inv(B)
        Y_est = A @ X @ B_inv
        Y_estimates.append(Y_est)
    
    # Average rotation using quaternion averaging
    quats = [Rotation.from_matrix(Y[:3, :3]).as_quat() for Y in Y_estimates]
    # Simple averaging (works for small spread)
    avg_quat = np.mean(quats, axis=0)
    avg_quat /= np.linalg.norm(avg_quat)
    
    # Average translation
    avg_trans = np.mean([Y[:3, 3] for Y in Y_estimates], axis=0)
    
    Y_avg = np.eye(4)
    Y_avg[:3, :3] = Rotation.from_quat(avg_quat).as_matrix()
    Y_avg[:3, 3] = avg_trans
    
    return Y_avg

def ransac_calibration(all_data, n_iterations=100, sample_fraction=0.3,
                       trans_thresh=0.6, rot_thresh=60.0, seed=42):
    """
    RANSAC-based calibration following paper methodology.
    
    1. Randomly sample a subset of pose pairs
    2. Estimate X (using simple averaging or SVD)
    3. Estimate Y for each tag
    4. Count inliers
    5. Keep best model
    """
    random.seed(seed)
    np.random.seed(seed)
    
    best_inlier_count = 0
    best_inlier_indices = None
    best_X = None
    best_Ys = None
    
    # Collect all poses for sampling
    all_poses = []
    for (tag_id, cam_idx), (A_poses, B_poses, _) in all_data.items():
        for i, (A, B) in enumerate(zip(A_poses, B_poses)):
            all_poses.append((tag_id, cam_idx, i, A, B))
    
    print(f"RANSAC: {n_iterations} iterations, {len(all_poses)} total poses")
    print(f"Thresholds: {trans_thresh}m translation, {rot_thresh}° rotation")
    
    for iteration in range(n_iterations):
        # Sample subset
        n_sample = max(100, int(len(all_poses) * sample_fraction))
        sample = random.sample(all_poses, n_sample)
        
        # Group by tag for Y estimation
        by_tag = {}
        for tag_id, cam_idx, i, A, B in sample:
            if tag_id not in by_tag:
                by_tag[tag_id] = {'A': [], 'B': []}
            by_tag[tag_id]['A'].append(A)
            by_tag[tag_id]['B'].append(B)
        
        # Simple X estimation: average of A^{-1} @ Y @ B for each pose
        # But we don't know Y yet... use iterative approach
        
        # Initialize X as identity (or use average from sample)
        # For simplicity, we'll estimate X by assuming tags are roughly at their
        # expected positions and solving
        
        # Better approach: use closed-form solution for hand-eye calibration
        # on the sampled data. For now, use simpler method.
        
        # Estimate X using the relationship: if we have multiple (A,B) pairs
        # for the same tag, then A1 @ X @ B1^{-1} ≈ A2 @ X @ B2^{-1}
        # This gives us: A1^{-1} @ A2 @ X ≈ X @ B1^{-1} @ B2
        # Which is the classic AX = XB hand-eye calibration
        
        # Collect relative motions for hand-eye calibration
        A_rel_list = []
        B_rel_list = []
        
        for tag_id, data in by_tag.items():
            A_list = data['A']
            B_list = data['B']
            if len(A_list) < 2:
                continue
            # Create relative motion pairs
            for j in range(len(A_list) - 1):
                A_rel = np.linalg.inv(A_list[j]) @ A_list[j+1]
                B_rel = np.linalg.inv(B_list[j]) @ B_list[j+1]
                A_rel_list.append(A_rel)
                B_rel_list.append(B_rel)
        
        if len(A_rel_list) < 10:
            continue
        
        # Solve AX = XB using Tsai-Lenz or similar
        # For simplicity, use rotation averaging approach
        X_est = solve_hand_eye_simple(A_rel_list, B_rel_list)
        
        if X_est is None:
            continue
        
        # Estimate Y for each tag using X
        Y_estimates = {}
        for tag_id, data in by_tag.items():
            Y_estimates[tag_id] = estimate_Y_from_AXB(data['A'], data['B'], X_est)
        
        # Also estimate Y for tags not in sample using all data
        for (tag_id, cam_idx), (A_poses, B_poses, _) in all_data.items():
            if tag_id not in Y_estimates and len(A_poses) > 0:
                Y_estimates[tag_id] = estimate_Y_from_AXB(A_poses, B_poses, X_est)
        
        # Count inliers
        inlier_count, inlier_indices = count_inliers(
            all_data, X_est, Y_estimates, trans_thresh, rot_thresh
        )
        
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inlier_indices = inlier_indices
            best_X = X_est
            best_Ys = Y_estimates
            print(f"  Iteration {iteration}: {inlier_count} inliers ({100*inlier_count/len(all_poses):.1f}%)")
    
    print(f"\nBest result: {best_inlier_count} inliers ({100*best_inlier_count/len(all_poses):.1f}%)")
    
    return best_X, best_Ys, best_inlier_indices

def solve_hand_eye_simple(A_rel_list, B_rel_list):
    """
    Simple hand-eye calibration solver.
    Solves AX = XB for rotation using quaternion method.
    """
    if len(A_rel_list) < 3:
        return None
    
    # Extract rotations
    R_A = [A[:3, :3] for A in A_rel_list]
    R_B = [B[:3, :3] for B in B_rel_list]
    
    # Use quaternion method (Park & Martin)
    # Build matrix M where M @ q_x = 0
    M = []
    for Ra, Rb in zip(R_A, R_B):
        qa = Rotation.from_matrix(Ra).as_quat()  # xyzw
        qb = Rotation.from_matrix(Rb).as_quat()
        
        # Convert to wxyz for the math
        qa = np.array([qa[3], qa[0], qa[1], qa[2]])
        qb = np.array([qb[3], qb[0], qb[1], qb[2]])
        
        # Build skew-symmetric matrices
        # (qa - qb) and (qa + qb) cross product matrices
        diff = qa - qb
        summ = qa + qb
        
        row = np.array([
            [diff[0], -diff[1], -diff[2], -diff[3]],
            [diff[1], diff[0], -summ[3], summ[2]],
            [diff[2], summ[3], diff[0], -summ[1]],
            [diff[3], -summ[2], summ[1], diff[0]]
        ])
        M.append(row)
    
    M = np.vstack(M)
    
    # Solve using SVD
    try:
        _, S, Vh = np.linalg.svd(M)
        q_x = Vh[-1]  # Last row of V^T (smallest singular value)
        q_x = q_x / np.linalg.norm(q_x)
        
        # Convert back to xyzw
        q_x_xyzw = np.array([q_x[1], q_x[2], q_x[3], q_x[0]])
        R_x = Rotation.from_quat(q_x_xyzw).as_matrix()
    except:
        return None
    
    # Solve for translation using least squares
    # t_A + R_A @ t_X = R_X @ t_B + t_X
    # (R_A - I) @ t_X = R_X @ t_B - t_A
    
    A_mat = []
    b_vec = []
    for A, B in zip(A_rel_list, B_rel_list):
        R_A_i = A[:3, :3]
        t_A_i = A[:3, 3]
        t_B_i = B[:3, 3]
        
        A_mat.append(R_A_i - np.eye(3))
        b_vec.append(R_x @ t_B_i - t_A_i)
    
    A_mat = np.vstack(A_mat)
    b_vec = np.hstack(b_vec)
    
    try:
        t_x, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    except:
        t_x = np.zeros(3)
    
    X = np.eye(4)
    X[:3, :3] = R_x
    X[:3, 3] = t_x
    
    return X

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='output/julia_data_se3')
    parser.add_argument('--output', default='output/julia_data_ransac')
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--trans-thresh', type=float, default=0.6)
    parser.add_argument('--rot-thresh', type=float, default=60.0)
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    input_dir = project_root / args.input
    output_dir = project_root / args.output
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("RANSAC-based Outlier Rejection (Paper Method)")
    print("=" * 60)
    
    # Load all data
    all_data = {}
    excluded_tags = {9, 15, 20}  # Known problematic tags
    
    for a_file in input_dir.glob("tag_*_cam_*_A.csv"):
        parts = a_file.stem.split('_')
        tag_id = int(parts[1])
        cam_idx = int(parts[3])
        
        if tag_id in excluded_tags:
            continue
        
        b_file = input_dir / f"tag_{tag_id}_cam_{cam_idx}_B.csv"
        if not b_file.exists():
            continue
        
        A_poses, A_rows = load_pose_csv(a_file)
        B_poses, B_rows = load_pose_csv(b_file)
        
        if len(A_poses) < 10:
            continue
        
        all_data[(tag_id, cam_idx)] = (A_poses, B_poses, (A_rows, B_rows))
    
    print(f"\nLoaded {sum(len(v[0]) for v in all_data.values())} poses from {len(all_data)} tag-camera pairs")
    
    # Run RANSAC
    best_X, best_Ys, inlier_indices = ransac_calibration(
        all_data,
        n_iterations=args.iterations,
        trans_thresh=args.trans_thresh,
        rot_thresh=args.rot_thresh
    )
    
    # Save inlier data
    print(f"\nSaving inlier data to {output_dir}")
    total_saved = 0
    
    for (tag_id, cam_idx), indices in inlier_indices.items():
        if len(indices) < 20:
            print(f"  Skipping tag {tag_id} cam {cam_idx}: only {len(indices)} inliers")
            continue
        
        _, _, (A_rows, B_rows) = all_data[(tag_id, cam_idx)]
        
        inlier_A = [A_rows[i] for i in indices]
        inlier_B = [B_rows[i] for i in indices]
        
        save_poses(inlier_A, output_dir / f"tag_{tag_id}_cam_{cam_idx}_A.csv")
        save_poses(inlier_B, output_dir / f"tag_{tag_id}_cam_{cam_idx}_B.csv")
        
        print(f"  Tag {tag_id}, Cam {cam_idx}: {len(indices)} inliers saved")
        total_saved += len(indices)
    
    print(f"\nTotal inliers saved: {total_saved}")
    print("=" * 60)

if __name__ == '__main__':
    main()
