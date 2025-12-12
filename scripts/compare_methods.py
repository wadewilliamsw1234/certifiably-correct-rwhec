#!/usr/bin/env python3
"""
compare_methods.py

Compare different preprocessing methods and generate tables/figures for the paper.

Usage:
    python scripts/compare_methods.py
"""

import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path
import glob
import matplotlib.pyplot as plt

def load_single_pose(path):
    """Load single pose from CSV."""
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

def evaluate_calibration(calib_dir, data_dir):
    """Evaluate a single calibration."""
    calib_dir = Path(calib_dir)
    data_dir = Path(data_dir)
    
    if not calib_dir.exists() or not (calib_dir / "left_X.csv").exists():
        return None
    
    X_left = load_single_pose(calib_dir / "left_X.csv")
    X_right = load_single_pose(calib_dir / "right_X.csv")
    Xs = {0: X_left, 1: X_right}
    
    baseline = np.linalg.norm(X_left[:3, 3] - X_right[:3, 3])
    
    z_left = X_left[:3, :3][:, 2]
    z_right = X_right[:3, :3][:, 2]
    angle = np.degrees(np.arccos(np.clip(np.dot(z_left, z_right), -1, 1)))
    
    Ys = {}
    for yf in glob.glob(str(calib_dir / "Y_tag_*.csv")):
        tag_id = int(Path(yf).stem.split('_')[-1])
        Ys[tag_id] = load_single_pose(yf)
    
    all_errors = []
    for cam_idx in [0, 1]:
        X = Xs[cam_idx]
        for tag_id in Ys.keys():
            a_file = data_dir / f"tag_{tag_id}_cam_{cam_idx}_A.csv"
            if not a_file.exists():
                continue
            A_poses = load_pose_csv(a_file)
            B_poses = load_pose_csv(data_dir / f"tag_{tag_id}_cam_{cam_idx}_B.csv")
            Y = Ys[tag_id]
            for A, B in zip(A_poses, B_poses):
                AX = A @ X
                YB = Y @ B
                all_errors.append(np.linalg.norm(AX[:3, 3] - YB[:3, 3]))
    
    if not all_errors:
        return None
    
    return {
        'baseline_cm': baseline * 100,
        'angle_deg': angle,
        'mean_error_cm': np.mean(all_errors) * 100,
        'std_error_cm': np.std(all_errors) * 100,
        'max_error_cm': np.max(all_errors) * 100,
        'n_poses': len(all_errors)
    }

def main():
    project_root = Path(__file__).parent.parent
    
    # Define methods to compare
    # Format: (name, data_dir, calib_dir)
    methods = [
        ("Nearest-Neighbor", "output/comparison/julia_data_filtered", "output/comparison/calibration_filtered"),
        ("NN + Iterative 25cm", "output/comparison/julia_data_clean", "output/comparison/calibration_clean"),
        ("SE(3) Interpolated", "output/julia_data", "output/final"),
        ("SE(3) + Iterative 25cm", "output/comparison/julia_data_se3_clean", "output/comparison/calibration_se3_clean"),
        ("RANSAC (60cm/60°)", "output/comparison/julia_data_ransac", "output/comparison/calibration_ransac"),
    ]
    
    print("=" * 100)
    print("PREPROCESSING METHOD COMPARISON")
    print("=" * 100)
    print()
    
    results = []
    for name, data_dir, calib_dir in methods:
        r = evaluate_calibration(project_root / calib_dir, project_root / data_dir)
        if r:
            r['name'] = name
            results.append(r)
    
    if not results:
        print("No calibration results found!")
        print("\nExpected directories:")
        for name, data_dir, calib_dir in methods:
            exists = (project_root / calib_dir).exists()
            status = "✓" if exists else "✗"
            print(f"  {status} {calib_dir}")
        return
    
    # Print table
    header = f"{'Method':<25} {'Baseline':>10} {'Angle':>8} {'Mean':>10} {'Std':>10} {'Max':>10} {'N':>8}"
    print(header)
    print("-" * len(header))
    
    for r in results:
        baseline_ok = "✓" if 10 <= r['baseline_cm'] <= 13 else ("~" if 8 <= r['baseline_cm'] <= 15 else "✗")
        angle_ok = "✓" if 45 <= r['angle_deg'] <= 90 else "✗"
        print(f"{r['name']:<25} {r['baseline_cm']:>8.2f}cm{baseline_ok} {r['angle_deg']:>6.1f}°{angle_ok} "
              f"{r['mean_error_cm']:>9.2f}cm {r['std_error_cm']:>9.2f}cm {r['max_error_cm']:>9.2f}cm {r['n_poses']:>8d}")
    
    print()
    print("=" * 100)
    print("EXPECTED VALUES: Baseline 10-13cm, Angle 45-90°")
    print("Legend: ✓ = within expected, ~ = close, ✗ = outside")
    print("=" * 100)
    
    # Generate LaTeX table
    print("\n\n=== LaTeX Table ===\n")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Comparison of preprocessing methods for RWHEC calibration}")
    print(r"\label{tab:preprocessing}")
    print(r"\begin{tabular}{lccccc}")
    print(r"\toprule")
    print(r"Method & Baseline (cm) & Angle ($^\circ$) & Mean Error (cm) & Max Error (cm) & N \\")
    print(r"\midrule")
    
    for r in results:
        name = r['name'].replace("_", r"\_")
        print(f"{name} & {r['baseline_cm']:.2f} & {r['angle_deg']:.1f} & "
              f"{r['mean_error_cm']:.2f} & {r['max_error_cm']:.2f} & {r['n_poses']} \\\\")
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    
    # Generate figure if matplotlib available
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        names = [r['name'] for r in results]
        baselines = [r['baseline_cm'] for r in results]
        mean_errors = [r['mean_error_cm'] for r in results]
        
        # Baseline comparison
        ax = axes[0]
        bars = ax.bar(range(len(names)), baselines, color='steelblue', alpha=0.8)
        ax.axhline(y=10, color='green', linestyle='--', label='Expected range')
        ax.axhline(y=13, color='green', linestyle='--')
        ax.axhspan(10, 13, alpha=0.2, color='green')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('Baseline (cm)')
        ax.set_title('Camera Baseline by Method')
        ax.legend()
        
        # Mean error comparison
        ax = axes[1]
        ax.bar(range(len(names)), mean_errors, color='coral', alpha=0.8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('Mean Error (cm)')
        ax.set_title('Mean Consistency Error by Method')
        
        plt.tight_layout()
        
        fig_path = project_root / 'figures' / 'method_comparison.png'
        fig_path.parent.mkdir(exist_ok=True)
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"\n\nFigure saved to: {fig_path}")
        plt.close()
        
    except Exception as e:
        print(f"\nCould not generate figure: {e}")

if __name__ == '__main__':
    main()
