# Certifiable Robot-World Hand-Eye Calibration (RWHEC)

A dual-camera hand-eye calibration system using AprilTag fiducials and OptiTrack motion capture, implementing the certifiable semidefinite relaxation from [Rosen et al. 2025](https://arxiv.org/abs/2501.00683).

## Overview

This project solves the **Robot-World Hand-Eye Calibration** problem for a dual-camera platform mounted on a mobile robot tracked by OptiTrack:

```
AX = YB

Where:
  A = OptiTrack pose (world → end-effector)
  B = AprilTag detection (tag → camera)
  X = Hand-eye transform (end-effector → camera) [SOLVE FOR]
  Y = Tag pose in world frame [SOLVE FOR]
```

### Key Features
- **SE(3) interpolation** for OptiTrack-camera synchronization (Barfoot Ch. 8)
- **Certifiable global optimum** via semidefinite relaxation
- **Multi-tag, multi-camera** joint optimization
- **Dual-camera baseline validation** (expected 10-13cm)

## Results

| Metric | Value | Expected |
|--------|-------|----------|
| **Camera Baseline** | 10.06 cm | 10-13 cm ✓ |
| **Optical Axis Angle** | 49.7° | 45-90° ✓ |
| **Mean Consistency Error** | 12.84 cm | - |

## Quick Start

### Prerequisites
```bash
# Python 3.10+
pip install numpy scipy opencv-python pupil-apriltags pandas pyyaml

# Julia 1.9+
julia -e 'using Pkg; Pkg.add(["LinearAlgebra", "Rotations", "DelimitedFiles", "SparseArrays", "JuMP", "COSMO"])'
```

### Run Calibration

```bash
# 1. Detect AprilTags in camera videos
python scripts/1_detect_apriltags.py \
    --left-video data/session_008/left_synced.mp4 \
    --right-video data/session_008/right_synced.mp4 \
    --output output/detections

# 2. Prepare calibration data with SE(3) interpolation
python scripts/2_prepare_data.py \
    --detections output/detections \
    --optitrack data/session_008/optitrack.csv \
    --output output/julia_data

# 3. Run certifiable solver
julia scripts/3_run_calibration.jl

# 4. Evaluate results
python scripts/4_evaluate_results.py
```

## Method: SE(3) Interpolation

### The Synchronization Problem

OptiTrack and cameras run at different rates:
- **OptiTrack**: ~120 Hz (pose every ~8ms)
- **Cameras**: 30 fps (frame every ~33ms)

When a camera captures a frame at time `t`, OptiTrack may not have a pose at exactly `t`.

### Solution: SE(3) Interpolation

Instead of nearest-neighbor matching (which introduces up to ±4ms timing error), we interpolate OptiTrack poses using proper **SE(3) geometry**:

1. Find bracketing OptiTrack poses: `T(t₁)` and `T(t₂)` where `t₁ < t < t₂`
2. Compute interpolation parameter: `α = (t - t₁) / (t₂ - t₁)`
3. Interpolate on SE(3):
   - **Translation**: Linear interpolation
   - **Rotation**: SLERP (Spherical Linear Interpolation)

```python
# SE(3) interpolation (simplified)
def interpolate_se3(T1, T2, alpha):
    # Translation: linear
    t = (1 - alpha) * T1[:3, 3] + alpha * T2[:3, 3]
    
    # Rotation: SLERP via quaternions
    q1 = Rotation.from_matrix(T1[:3, :3]).as_quat()
    q2 = Rotation.from_matrix(T2[:3, :3]).as_quat()
    q = slerp(q1, q2, alpha)
    
    return compose_se3(q, t)
```

This approach follows Barfoot's "State Estimation for Robotics" Chapter 8.

## Project Structure

```
rwhec/
├── README.md
├── requirements.txt
├── config/
│   └── camera_intrinsics/      # Camera calibration YAML files
├── data/                        # Raw data (not tracked in git)
│   ├── session_008/
│   │   ├── left_synced.mp4     # Left camera video (time-synced)
│   │   ├── right_synced.mp4    # Right camera video (time-synced)
│   │   └── optitrack.csv       # OptiTrack poses
│   └── *_calibration_images/   # Intrinsic calibration images
├── scripts/
│   ├── 1_detect_apriltags.py   # AprilTag detection
│   ├── 2_prepare_data.py       # SE(3) interpolation + data formatting
│   ├── 3_run_calibration.jl    # Certifiable SDP solver
│   ├── 4_evaluate_results.py   # Result evaluation and metrics
│   └── preprocessing/          # Alternative preprocessing methods
│       ├── iterative_filtering.py
│       └── ransac_filtering.py
├── third_party/
│   └── certifiable-rwhe-calibration/  # Rosen et al. solver
├── output/
│   ├── final/                  # Final calibration results
│   │   ├── left_X.csv          # Left camera hand-eye transform
│   │   ├── right_X.csv         # Right camera hand-eye transform
│   │   └── Y_tag_*.csv         # Tag poses in world frame
│   └── comparison/             # Results from different preprocessing methods
└── figures/                    # Generated figures for report
```

## Alternative Preprocessing Methods

While SE(3) interpolation is our primary method, we also implemented:

| Method | Baseline | Mean Error | Notes |
|--------|----------|------------|-------|
| **SE(3) interpolated** | 10.06cm ✓ | 12.84cm | **Primary method** |
| SE(3) + Iterative 25cm | 9.19cm | 11.90cm | Outlier filtering |
| Nearest-neighbor + 25cm | 8.65cm | 11.68cm | No interpolation |
| RANSAC (paper thresholds) | 9.08cm | 12.12cm | 60cm/60° thresholds |

See `scripts/preprocessing/` for alternative implementations.

## References

1. **Rosen, D. M. (2025)**. "A Certifiably Correct Algorithm for Generalized Robot-World and Hand-Eye Calibration." [[PDF]](third_party/certifiable-rwhe-calibration/)

2. **Barfoot, T. D. (2024)**. "State Estimation for Robotics." Chapter 8: SE(3) Interpolation.

3. **Park, F. C., & Martin, B. J. (1994)**. "Robot sensor calibration: solving AX=XB on the Euclidean group."

## License

MIT License - See LICENSE file for details.

## Authors

- Wade Williams (Northeastern University, MS Robotics)
- Luke Jansen, Theo McArn (collaborators)
