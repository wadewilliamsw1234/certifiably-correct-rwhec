# Certifiable Robot-World Hand-Eye Calibration

Dual-camera hand-eye calibration using AprilTags and OptiTrack, with a certifiable SDP solver from [Rosen et al. 2025](https://arxiv.org/abs/2501.00683).

## Problem

Solve `AX = YB` for a dual-camera rig tracked by OptiTrack:
- `A` = OptiTrack pose (world → rig)
- `B` = AprilTag pose (tag → camera)
- `X` = hand-eye transform (rig → camera) — what we want
- `Y` = tag pose in world frame

## Results

| Metric | Value | Expected |
|--------|-------|----------|
| Camera baseline | 10.06 cm | 10-13 cm |
| Optical axis angle | 49.7° | 45-90° |
| Duality gap | 2.14×10⁻⁹ | < 10⁻⁴ |

The duality gap confirms the solution is certifiably globally optimal.

## Setup

```bash
# Python
pip install numpy scipy opencv-python pupil-apriltags pyyaml

# Julia
julia -e 'using Pkg; Pkg.add(["Rotations", "JuMP", "COSMO"])'
```

## Usage

```bash
# 1. Detect AprilTags
python scripts/1_detect_apriltags.py --camera left
python scripts/1_detect_apriltags.py --camera right

# 2. Sync with OptiTrack (SE(3) interpolation)
python scripts/2_prepare_data.py

# 3. Run SDP solver
julia scripts/3_run_calibration.jl

# 4. Evaluate
python scripts/4_evaluate_results.py
```

## SE(3) Interpolation

OptiTrack runs at 120Hz, cameras at 30fps. Instead of nearest-neighbor matching (±4ms error), we interpolate poses properly on SE(3):
- Translation: linear interpolation
- Rotation: SLERP

See Barfoot's "State Estimation for Robotics" Ch. 8.

## Structure

```
scripts/
├── 1_detect_apriltags.py    # AprilTag detection + pose
├── 2_prepare_data.py        # SE(3) sync, coordinate transforms
├── 3_run_calibration.jl     # Certifiable SDP solver
├── 4_evaluate_results.py    # Metrics
└── preprocessing/           # Alternative methods (RANSAC, etc.)

output/
├── julia_data/              # A,B pose pairs for solver
└── final/                   # X,Y calibration results
```

## References

- Rosen et al. (2025) - Certifiable RWHEC algorithm
- Barfoot (2024) - State Estimation for Robotics
- Park & Martin (1994) - AX=XB on Euclidean group