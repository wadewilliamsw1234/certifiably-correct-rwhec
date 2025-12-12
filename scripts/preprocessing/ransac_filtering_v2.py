#!/usr/bin/env python3
"""
RANSAC v2: Self-Consistency with Appropriate Thresholds
"""

import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path
import csv
import random
from typing import Dict, List, Tuple, Optional
import argparse

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

def save_poses(rows, path):
    """Save poses to CSV."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

def rotation_error_deg(R1, R2):
    """Geodesic distance between two rotations in degrees."""
    R_diff = R1.T @ R2
    trace = np.clip((np.trace(R_diff) - 1) / 2, -1, 1)
    return np.degrees(np.abs(np.arccos(trace)))

def solve_hand_eye_quaternion(A_rel_list, B_rel_list):
    """
    Solve AX = XB using quaternion method (Park & Martin).
    
    This solves for X given relative motions A_rel and B_rel where:
    A_rel[i] = A[i]^{-1} @ A[i+1]  (relative robot motion)
    B_rel[i] = B[i]^{-1} @ B[i+1]  (relative camera motion)
    
    Returns X (hand-eye transformation) or None if failed.
    """
    if len(A_rel_list) < 3:
        return None
    
    # Build quaternion constraint matrix
    M = []
    for Ra, Rb in zip([A[:3,:3] for A in A_rel_list], [B[:3,:3] for B in B_rel_list]):
        qa = Rotation.from_matrix(Ra).as_quat()  # xyzw format
        qb = Rotation.from_matrix(Rb).as_quat()
        
        # Convert to wxyz for the math
        qa = np.array([qa[3], qa[0], qa[1], qa[2]])
        qb = np.array([qb[3], qb[0], qb[1], qb[2]])
        
        diff = qa - qb
        summ = qa + qb
        
        # Build constraint rows
        row = np.array([
            [diff[0], -diff[1], -diff[2], -diff[3]],
            [diff[1], diff[0], -summ[3], summ[2]],
            [diff[2], summ[3], diff[0], -summ[1]],
            [diff[3], -summ[2], summ[1], diff[0]]
        ])
        M.append(row)
    
    M = np.vstack(M)
    
    try:
        _, S, Vh = np.linalg.svd(M)
        q_x = Vh[-1]  # Eigenvector for smallest singular value
        q_x = q_x / np.linalg.norm(q_x)
        
        # Convert back to xyzw and get rotation matrix
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
        R_A = A[:3, :3]
        t_A = A[:3, 3]
        t_B = B[:3, 3]
        A_mat.append(R_A - np.eye(3))
        b_vec.append(R_x @ t_B - t_A)
    
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

def estimate_Y_robust(A_poses, B_poses, X):
    """
    Estimate Y given A, B poses and X.
    Y = A @ X @ B^{-1}, robustly averaged over all poses.
    
    Uses median for translation (robust to outliers) and
    quaternion averaging for rotation.
    """
    Y_estimates = []
    for A, B in zip(A_poses, B_poses):
        B_inv = np.linalg.inv(B)
        Y_est = A @ X @ B_inv
        Y_estimates.append(Y_est)
    
    # Robust averaging: median for translation
    trans_list = [Y[:3, 3] for Y in Y_estimates]
    median_trans = np.median(trans_list, axis=0)
    
    # Quaternion averaging for rotation (handle sign ambiguity)
    quats = [Rotation.from_matrix(Y[:3, :3]).as_quat() for Y in Y_estimates]
    ref_quat = quats[0]
    aligned_quats = []
    for q in quats:
        if np.dot(q, ref_quat) < 0:
            aligned_quats.append(-q)
        else:
            aligned_quats.append(q)
    
    avg_quat = np.mean(aligned_quats, axis=0)
    avg_quat /= np.linalg.norm(avg_quat)
    
    Y = np.eye(4)
    Y[:3, :3] = Rotation.from_quat(avg_quat).as_matrix()
    Y[:3, 3] = median_trans
    return Y

def ransac_self_consistency(all_data: Dict,
                            n_iterations: int = 500,
                            sample_size: int = 300,
                            trans_thresh: float = 0.20,
                            rot_thresh: float = 15.0,
                            min_inliers_per_pair: int = 20,
                            seed: int = 42):
    """
    RANSAC using self-consistency criterion (no ground truth needed).
    
    Algorithm:
    1. Sample random poses across all tag-camera pairs
    2. Build relative motion constraints for hand-eye calibration
    3. Solve for X using sampled data
    4. Estimate Y for each tag using X
    5. Count inliers: poses where ||AX - YB|| < threshold
    6. Keep model with most inliers
    7. Return inliers from best model
    
    Key insight: Without ground truth, we find the largest subset
    of poses that are mutually consistent with each other.
    
    Args:
        all_data: Dict mapping (tag_id, cam_idx) to (A_poses, B_poses, raw_rows)
        n_iterations: Number of RANSAC iterations (more = better but slower)
        sample_size: Number of poses to sample per iteration
        trans_thresh: Translation error threshold in METERS
                     (use ~0.15-0.25 for self-consistency, NOT paper's 0.6)
        rot_thresh: Rotation error threshold in DEGREES
                   (use ~10-20 for self-consistency, NOT paper's 60)
        min_inliers_per_pair: Minimum inliers needed to keep a tag-camera pair
        seed: Random seed for reproducibility
    
    Returns:
        best_inlier_data: Dict of inlier poses grouped by (tag_id, cam_idx)
        best_X: Best estimated X transformation
        n_inliers: Total number of inliers
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Flatten all poses with metadata for sampling
    all_poses = []
    for (tag_id, cam_idx), (A_poses, B_poses, raw) in all_data.items():
        A_rows, B_rows = raw
        for i, (A, B) in enumerate(zip(A_poses, B_poses)):
            all_poses.append({
                'tag_id': tag_id,
                'cam_idx': cam_idx,
                'idx': i,
                'A': A,
                'B': B,
                'A_row': A_rows[i],
                'B_row': B_rows[i]
            })
    
    n_total = len(all_poses)
    
    print(f"\n{'='*60}")
    print("RANSAC Self-Consistency")
    print(f"{'='*60}")
    print(f"Total poses: {n_total}")
    print(f"Iterations: {n_iterations}")
    print(f"Sample size: {sample_size}")
    print(f"Thresholds: {trans_thresh*100:.0f}cm translation, {rot_thresh:.0f}° rotation")
    print(f"\nNote: Paper used 60cm/60° for ground truth comparison.")
    print(f"      We use {trans_thresh*100:.0f}cm/{rot_thresh:.0f}° for self-consistency.\n")
    
    best_inlier_count = 0
    best_X = None
    best_Ys = None
    best_inlier_mask = None
    
    for iteration in range(n_iterations):
        # Sample poses
        if sample_size >= n_total:
            sample = all_poses
        else:
            sample = random.sample(all_poses, sample_size)
        
        # Group by tag and camera for building relative motions
        by_tag_cam = {}
        for p in sample:
            key = (p['tag_id'], p['cam_idx'])
            if key not in by_tag_cam:
                by_tag_cam[key] = []
            by_tag_cam[key].append(p)
        
        # Build relative motion constraints for hand-eye calibration
        A_rel_list = []
        B_rel_list = []
        
        for (tag_id, cam_idx), poses in by_tag_cam.items():
            if len(poses) < 2:
                continue
            
            # Create consecutive relative motions
            for j in range(len(poses) - 1):
                A1, A2 = poses[j]['A'], poses[j+1]['A']
                B1, B2 = poses[j]['B'], poses[j+1]['B']
                
                A_rel = np.linalg.inv(A1) @ A2
                B_rel = np.linalg.inv(B1) @ B2
                
                # Only use if there's significant motion (avoids numerical issues)
                trans_motion = np.linalg.norm(A_rel[:3, 3])
                rot_motion = rotation_error_deg(np.eye(3), A_rel[:3, :3])
                
                if trans_motion > 0.01 or rot_motion > 1.0:
                    A_rel_list.append(A_rel)
                    B_rel_list.append(B_rel)
        
        if len(A_rel_list) < 20:
            continue
        
        # Solve for X
        X_est = solve_hand_eye_quaternion(A_rel_list, B_rel_list)
        if X_est is None:
            continue
        
        # Estimate Y for each tag using ALL data (not just sample)
        Y_estimates = {}
        for (tag_id, cam_idx), (A_poses, B_poses, _) in all_data.items():
            if len(A_poses) >= 5:
                Y_est = estimate_Y_robust(A_poses, B_poses, X_est)
                if tag_id not in Y_estimates:
                    Y_estimates[tag_id] = Y_est
        
        # Count inliers across ALL poses
        inlier_mask = np.zeros(n_total, dtype=bool)
        
        for i, p in enumerate(all_poses):
            tag_id = p['tag_id']
            if tag_id not in Y_estimates:
                continue
            
            Y = Y_estimates[tag_id]
            AX = p['A'] @ X_est
            YB = Y @ p['B']
            
            trans_err = np.linalg.norm(AX[:3, 3] - YB[:3, 3])
            rot_err = rotation_error_deg(AX[:3, :3], YB[:3, :3])
            
            if trans_err < trans_thresh and rot_err < rot_thresh:
                inlier_mask[i] = True
        
        inlier_count = np.sum(inlier_mask)
        
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_X = X_est
            best_Ys = Y_estimates
            best_inlier_mask = inlier_mask.copy()
            
            pct = 100 * inlier_count / n_total
            print(f"  Iter {iteration:3d}: {inlier_count} inliers ({pct:.1f}%)")
    
    print(f"\nBest result: {best_inlier_count} inliers ({100*best_inlier_count/n_total:.1f}%)")
    
    # Extract inlier data grouped by tag-camera pair
    inlier_data = {}
    for i, p in enumerate(all_poses):
        if best_inlier_mask[i]:
            key = (p['tag_id'], p['cam_idx'])
            if key not in inlier_data:
                inlier_data[key] = {'A_rows': [], 'B_rows': []}
            inlier_data[key]['A_rows'].append(p['A_row'])
            inlier_data[key]['B_rows'].append(p['B_row'])
    
    return inlier_data, best_X, best_inlier_count

def main():
    parser = argparse.ArgumentParser(
        description="RANSAC outlier rejection with self-consistency (proper thresholds)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default (20cm/15° thresholds - good starting point)
  python ransac_filtering_v2.py
  
  # Tighter thresholds (more aggressive filtering)
  python ransac_filtering_v2.py --trans-thresh 0.15 --rot-thresh 10
  
  # Looser thresholds (keep more data)
  python ransac_filtering_v2.py --trans-thresh 0.25 --rot-thresh 20

Note: Paper used 60cm/60° for comparison to Kalibr ground truth.
      We use tighter thresholds for SELF-CONSISTENCY since we don't have GT.
        """
    )
    parser.add_argument('--input', default='output/julia_data_se3',
                        help='Input directory with pose data')
    parser.add_argument('--output', default='output/julia_data_ransac_v2',
                        help='Output directory for filtered data')
    parser.add_argument('--iterations', type=int, default=500,
                        help='Number of RANSAC iterations (default: 500)')
    parser.add_argument('--trans-thresh', type=float, default=0.20,
                        help='Translation threshold in meters (default: 0.20 = 20cm)')
    parser.add_argument('--rot-thresh', type=float, default=15.0,
                        help='Rotation threshold in degrees (default: 15)')
    parser.add_argument('--min-poses', type=int, default=20,
                        help='Minimum poses per tag-camera pair (default: 20)')
    parser.add_argument('--exclude-tags', type=str, default='9,15,20',
                        help='Comma-separated list of tags to exclude (default: 9,15,20)')
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    input_dir = project_root / args.input
    output_dir = project_root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse excluded tags
    excluded_tags = set()
    if args.exclude_tags:
        excluded_tags = {int(t.strip()) for t in args.exclude_tags.split(',')}
    
    print("=" * 70)
    print("RANSAC v2: Self-Consistency with Appropriate Thresholds")
    print("=" * 70)
    print(f"\nInput:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Excluded tags: {excluded_tags}")
    print(f"\nThresholds: {args.trans_thresh*100:.0f}cm translation, {args.rot_thresh:.0f}° rotation")
    print("(Paper used 60cm/60° for ground truth comparison - too loose for self-consistency)")
    
    # Load all data
    all_data = {}
    
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
    
    total_poses = sum(len(v[0]) for v in all_data.values())
    print(f"\nLoaded {total_poses} poses from {len(all_data)} tag-camera pairs")
    
    if total_poses == 0:
        print("ERROR: No data found! Check input directory.")
        return
    
    # Run RANSAC
    inlier_data, best_X, n_inliers = ransac_self_consistency(
        all_data,
        n_iterations=args.iterations,
        trans_thresh=args.trans_thresh,
        rot_thresh=args.rot_thresh,
        min_inliers_per_pair=args.min_poses
    )
    
    # Save inlier data
    print(f"\n{'='*60}")
    print(f"Saving filtered data to {output_dir}")
    print(f"{'='*60}")
    
    total_saved = 0
    pairs_saved = 0
    pairs_skipped = 0
    
    for (tag_id, cam_idx), data in sorted(inlier_data.items()):
        n_inliers_pair = len(data['A_rows'])
        
        if n_inliers_pair < args.min_poses:
            print(f"  Tag {tag_id:2d}, Cam {cam_idx}: SKIP ({n_inliers_pair} < {args.min_poses} min)")
            pairs_skipped += 1
            continue
        
        save_poses(data['A_rows'], output_dir / f"tag_{tag_id}_cam_{cam_idx}_A.csv")
        save_poses(data['B_rows'], output_dir / f"tag_{tag_id}_cam_{cam_idx}_B.csv")
        
        print(f"  Tag {tag_id:2d}, Cam {cam_idx}: {n_inliers_pair} inliers saved")
        total_saved += n_inliers_pair
        pairs_saved += 1
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Total poses saved: {total_saved} / {total_poses} ({100*total_saved/total_poses:.1f}%)")
    print(f"  Tag-camera pairs: {pairs_saved} saved, {pairs_skipped} skipped")
    print(f"\nNext step: Run calibration on filtered data:")
    print(f"  julia scripts/run_ransac_v2_calibration.jl")

if __name__ == '__main__':
    main()
