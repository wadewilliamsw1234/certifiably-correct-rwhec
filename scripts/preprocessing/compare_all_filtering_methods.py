#!/usr/bin/env python3
"""
Comparison of filtering methods.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path
import glob

def load_single_pose(path):
    """Load a single pose from CSV (qw, qx, qy, qz, tx, ty, tz)."""
    data = np.loadtxt(path, delimiter=',')
    qw, qx, qy, qz = data[:4]
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = data[4:7]
    return T

def load_pose_csv(path):
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

def evaluate_calibration(name, julia_data_dir, calib_dir, project_root):
    """Evaluate a calibration result."""
    julia_data = project_root / julia_data_dir
    calib_data = project_root / calib_dir
    
    if not calib_data.exists():
        return None
    
    left_x_path = calib_data / "left_X.csv"
    right_x_path = calib_data / "right_X.csv"
    
    if not left_x_path.exists() or not right_x_path.exists():
        return None
    
    X_left = load_single_pose(left_x_path)
    X_right = load_single_pose(right_x_path)
    Xs = {0: X_left, 1: X_right}
    
    # Baseline
    baseline = np.linalg.norm(X_left[:3, 3] - X_right[:3, 3])
    
    # Angle between optical axes
    z_left = X_left[:3, :3][:, 2]
    z_right = X_right[:3, :3][:, 2]
    angle = np.arccos(np.clip(np.dot(z_left, z_right), -1, 1)) * 180 / np.pi
    
    # Load Ys
    Ys = {}
    for yf in glob.glob(str(calib_data / "Y_tag_*.csv")):
        tag_id = int(Path(yf).stem.split('_')[-1])
        Ys[tag_id] = load_single_pose(yf)
    
    if not julia_data.exists():
        # Can't compute errors without pose data
        return {
            'name': name,
            'baseline': baseline * 100,
            'angle': angle,
            'mean_error': None,
            'std_error': None,
            'max_error': None,
            'n_poses': None
        }
    
    # Compute errors
    all_errors = []
    for cam_idx in [0, 1]:
        X = Xs[cam_idx]
        for tag_id in Ys.keys():
            a_file = julia_data / f"tag_{tag_id}_cam_{cam_idx}_A.csv"
            if not a_file.exists():
                continue
            
            A_poses = load_pose_csv(a_file)
            B_poses = load_pose_csv(julia_data / f"tag_{tag_id}_cam_{cam_idx}_B.csv")
            Y = Ys[tag_id]
            
            for A, B in zip(A_poses, B_poses):
                AX = A @ X
                YB = Y @ B
                all_errors.append(np.linalg.norm(AX[:3, 3] - YB[:3, 3]))
    
    if len(all_errors) == 0:
        return None
    
    return {
        'name': name,
        'baseline': baseline * 100,
        'angle': angle,
        'mean_error': np.mean(all_errors) * 100,
        'std_error': np.std(all_errors) * 100,
        'max_error': np.max(all_errors) * 100,
        'n_poses': len(all_errors)
    }

def main():
    project_root = Path("/home/wadewilliams/Dev/rwhec")
    
    print("=" * 95)
    print("COMPREHENSIVE COMPARISON: All Filtering Methods")
    print("=" * 95)
    
    # Define all configurations to compare
    configs = [
        # (Name, julia_data_dir, calib_dir)
        
        # Original approaches (nearest-neighbor interpolation)
        ("Original (nearest-neighbor)", "output/julia_data", "output/calibration"),
        ("Filtered tags only (NN)", "output/julia_data_filtered", "output/calibration_filtered"),
        ("Iterative 25cm (NN)", "output/julia_data_clean", "output/calibration_clean"),
        
        # SE(3) interpolation approaches  
        ("SE(3) interpolated", "output/julia_data_se3", "output/calibration_se3"),
        ("SE(3) + Iterative 25cm", "output/julia_data_se3_clean", "output/calibration_se3_clean"),
        
        # RANSAC approaches
        ("RANSAC v1 (60cm/60° paper)", "output/julia_data_ransac", "output/calibration_ransac"),
        ("RANSAC v2 (20cm/15° proper)", "output/julia_data_ransac_v2", "output/calibration_ransac_v2"),
    ]
    
    results = []
    for name, data_dir, calib_dir in configs:
        r = evaluate_calibration(name, data_dir, calib_dir, project_root)
        if r:
            results.append(r)
    
    if not results:
        print("\nNo calibration results found!")
        print("Make sure you have run the calibration scripts.")
        return
    
    # Print comparison table
    print(f"\n{'Method':<32} {'Baseline':>10} {'Angle':>8} {'Mean':>10} {'Std':>10} {'Max':>10} {'N':>8}")
    print("-" * 100)
    
    for r in results:
        # Baseline check
        if 10 <= r['baseline'] <= 13:
            baseline_ok = "✓"
        elif 8 <= r['baseline'] <= 15:
            baseline_ok = "~"
        else:
            baseline_ok = "✗"
        
        # Angle check
        angle_ok = "✓" if 45 <= r['angle'] <= 90 else "✗"
        
        # Format error values
        if r['mean_error'] is not None:
            mean_str = f"{r['mean_error']:>8.2f}cm"
            std_str = f"{r['std_error']:>8.2f}cm"
            max_str = f"{r['max_error']:>8.2f}cm"
            n_str = f"{r['n_poses']:>8d}"
        else:
            mean_str = std_str = max_str = n_str = "    N/A"
        
        print(f"{r['name']:<32} {r['baseline']:>8.2f}cm{baseline_ok} {r['angle']:>6.1f}°{angle_ok} "
              f"{mean_str} {std_str} {max_str} {n_str}")
    
    # Analysis section
    print("\n" + "=" * 95)
    print("ANALYSIS")
    print("=" * 95)
    
    # Find results by category
    nn_results = [r for r in results if 'nearest-neighbor' in r['name'].lower() or 'NN' in r['name']]
    se3_results = [r for r in results if 'SE(3)' in r['name'] and 'RANSAC' not in r['name']]
    ransac_v1 = next((r for r in results if 'v1' in r['name'].lower() or '60cm' in r['name']), None)
    ransac_v2 = next((r for r in results if 'v2' in r['name'].lower() or '20cm' in r['name']), None)
    
    # 1. SE(3) vs Nearest-Neighbor
    print("\n1. INTERPOLATION METHOD (SE(3) vs Nearest-Neighbor):")
    nn_clean = next((r for r in nn_results if 'clean' in r['name'].lower() or '25cm' in r['name']), None)
    se3_clean = next((r for r in se3_results if 'clean' in r['name'].lower() or '25cm' in r['name']), None)
    
    if nn_clean and se3_clean and nn_clean['mean_error'] and se3_clean['mean_error']:
        print(f"   Nearest-neighbor + 25cm: baseline={nn_clean['baseline']:.2f}cm, mean={nn_clean['mean_error']:.2f}cm")
        print(f"   SE(3) + 25cm:            baseline={se3_clean['baseline']:.2f}cm, mean={se3_clean['mean_error']:.2f}cm")
        print(f"   → Baseline change: {se3_clean['baseline'] - nn_clean['baseline']:+.2f}cm")
        print(f"   → SE(3) interpolation {'improved' if se3_clean['baseline'] > nn_clean['baseline'] else 'worsened'} baseline accuracy")
    
    # 2. RANSAC comparison
    print("\n2. RANSAC THRESHOLD COMPARISON:")
    if ransac_v1 and ransac_v2:
        if ransac_v1['mean_error'] and ransac_v2['mean_error']:
            print(f"   RANSAC v1 (60cm/60°): baseline={ransac_v1['baseline']:.2f}cm, mean={ransac_v1['mean_error']:.2f}cm, n={ransac_v1['n_poses']}")
            print(f"   RANSAC v2 (20cm/15°): baseline={ransac_v2['baseline']:.2f}cm, mean={ransac_v2['mean_error']:.2f}cm, n={ransac_v2['n_poses']}")
            print(f"   → v1 kept {ransac_v1['n_poses']} poses, v2 kept {ransac_v2['n_poses']} poses")
            print(f"   → v2 is {'stricter' if ransac_v2['n_poses'] < ransac_v1['n_poses'] else 'looser'} filtering")
    elif ransac_v1:
        print(f"   RANSAC v1 (60cm/60°): baseline={ransac_v1['baseline']:.2f}cm - thresholds too loose!")
        print("   RANSAC v2 not yet run - execute: python scripts/ransac_filtering_v2.py")
    else:
        print("   No RANSAC results found")
    
    # 3. Best method recommendation
    print("\n3. RECOMMENDATION:")
    
    # Filter for valid baselines
    valid_results = [r for r in results if 8 <= r['baseline'] <= 15 and r['mean_error'] is not None]
    
    if valid_results:
        # Sort by multiple criteria
        def score(r):
            # Prefer baseline closer to 11.5cm (middle of 10-13 range)
            baseline_score = abs(r['baseline'] - 11.5)
            # Prefer lower mean error
            error_score = r['mean_error']
            # Combined score (weight baseline more heavily)
            return baseline_score * 2 + error_score
        
        best = min(valid_results, key=score)
        
        # Also find lowest error
        lowest_error = min(valid_results, key=lambda x: x['mean_error'])
        
        print(f"\n   🏆 BEST OVERALL: {best['name']}")
        print(f"      Baseline: {best['baseline']:.2f} cm (expected: 10-13cm)")
        print(f"      Angle: {best['angle']:.1f}° (expected: 45-90°)")
        print(f"      Mean error: {best['mean_error']:.2f} cm")
        print(f"      Max error: {best['max_error']:.2f} cm")
        
        if lowest_error != best:
            print(f"\n   📉 LOWEST ERROR: {lowest_error['name']}")
            print(f"      Baseline: {lowest_error['baseline']:.2f} cm")
            print(f"      Mean error: {lowest_error['mean_error']:.2f} cm")
            if lowest_error['baseline'] < 8 or lowest_error['baseline'] > 15:
                print("      ⚠️  Baseline outside expected range - possible overfitting!")
    
    print("\n" + "=" * 95)
    print("EXPECTED VALUES: Baseline 10-13cm, Angle 45-90°")
    print("Legend: ✓ = within expected range, ~ = close, ✗ = outside range")
    print("=" * 95)

if __name__ == '__main__':
    main()
