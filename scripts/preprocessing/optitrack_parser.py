#!/usr/bin/env python3
"""
OptiTrack Motion Capture Data Parser

Parses OptiTrack CSV exports and extracts rigid body poses.
Converts to the format expected by the Julia RWHEC solver.

I.e. Constructs A matrices (robot/mocap poses)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import csv


@dataclass
class OptiTrackPose:
    """Single OptiTrack pose measurement."""
    frame_idx: int
    timestamp: float  # seconds
    quaternion_xyzw: np.ndarray  # OptiTrack native format
    position_mm: np.ndarray  # millimeters

    def get_quaternion_wxyz(self) -> np.ndarray:
        """Get quaternion in W, X, Y, Z order (Julia format)."""
        q = self.quaternion_xyzw
        return np.array([q[3], q[0], q[1], q[2]])

    def get_position_meters(self) -> np.ndarray:
        """Get position in meters."""
        return self.position_mm / 1000.0

    def get_transform_matrix(self) -> np.ndarray:
        """Get 4x4 homogeneous transformation matrix."""
        r = Rotation.from_quat(self.quaternion_xyzw)  # scipy uses xyzw
        T = np.eye(4)
        T[:3, :3] = r.as_matrix()
        T[:3, 3] = self.get_position_meters()
        return T


class OptiTrackParser:
    """Parser for OptiTrack CSV export files."""

    def __init__(self, csv_path: str):
        """
        Initialize parser with path to OptiTrack CSV.

        Args:
            csv_path: Path to OptiTrack CSV export file
        """
        self.csv_path = Path(csv_path)
        self.metadata = {}
        self.rigid_body_name = None
        self.frame_rate = 120.0
        self.poses: List[OptiTrackPose] = []

        self._parse_file()

    def _parse_file(self):
        """Parse the OptiTrack CSV file."""
        with open(self.csv_path, 'r') as f:
            lines = f.readlines()

        # Parse metadata from first line
        meta_line = lines[0].strip().split(',')
        for i in range(0, len(meta_line) - 1, 2):
            key = meta_line[i].strip()
            val = meta_line[i + 1].strip() if i + 1 < len(meta_line) else ''
            if key:
                self.metadata[key] = val

        # Extract frame rate
        if 'Capture Frame Rate' in self.metadata:
            self.frame_rate = float(self.metadata['Capture Frame Rate'])

        # Parse header rows to find rigid body columns
        type_row = lines[1].strip().split(',')
        name_row = lines[2].strip().split(',')
        data_type_row = lines[4].strip().split(',')

        # Find rigid body rotation and position columns
        # Looking for: Rotation X, Y, Z, W and Position X, Y, Z
        rot_cols = []
        pos_cols = []

        for i, (dtype, name) in enumerate(zip(data_type_row, name_row)):
            if 'Rigid Body' in type_row[i] if i < len(type_row) else '':
                if 'Rotation' in dtype:
                    rot_cols.append(i)
                    if self.rigid_body_name is None and name:
                        self.rigid_body_name = name.split(':')[0]
                elif 'Position' in dtype:
                    pos_cols.append(i)

        print(f"Found rigid body: {self.rigid_body_name}")
        print(f"Rotation columns: {rot_cols[:4]}")
        print(f"Position columns: {pos_cols[:3]}")

        # Rotation columns should be X, Y, Z, W (indices 2-5 in header)
        # Position columns should be X, Y, Z (indices 6-8 in header)
        # But column 0 is Frame, column 1 is Time

        # Parse data rows (starting from row 7, index 6)
        for line in lines[7:]:
            parts = line.strip().split(',')
            if len(parts) < 9:
                continue

            try:
                frame_idx = int(parts[0])
                timestamp = float(parts[1])

                # Quaternion: X, Y, Z, W (columns 2, 3, 4, 5)
                qx = float(parts[2])
                qy = float(parts[3])
                qz = float(parts[4])
                qw = float(parts[5])

                # Position: X, Y, Z (columns 6, 7, 8) in mm
                px = float(parts[6])
                py = float(parts[7])
                pz = float(parts[8])

                pose = OptiTrackPose(
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    quaternion_xyzw=np.array([qx, qy, qz, qw]),
                    position_mm=np.array([px, py, pz])
                )
                self.poses.append(pose)

            except (ValueError, IndexError) as e:
                continue  # Skip malformed rows

        print(f"Parsed {len(self.poses)} poses from OptiTrack data")
        print(f"Time range: {self.poses[0].timestamp:.3f}s to {self.poses[-1].timestamp:.3f}s")

    def get_pose_at_time(self, target_time: float) -> Optional[OptiTrackPose]:
        """
        Get pose closest to target time.

        Args:
            target_time: Time in seconds

        Returns:
            OptiTrackPose closest to target time, or None if out of range
        """
        if not self.poses:
            return None

        # Binary search for closest time
        times = np.array([p.timestamp for p in self.poses])
        idx = np.searchsorted(times, target_time)

        if idx == 0:
            return self.poses[0]
        if idx >= len(self.poses):
            return self.poses[-1]

        # Check which neighbor is closer
        if abs(times[idx] - target_time) < abs(times[idx - 1] - target_time):
            return self.poses[idx]
        return self.poses[idx - 1]

    def get_pose_interpolated(self, target_time: float) -> Optional[OptiTrackPose]:
        """
        Get interpolated pose at target time using SLERP for rotation.

        Args:
            target_time: Time in seconds

        Returns:
            Interpolated OptiTrackPose
        """
        if not self.poses:
            return None

        times = np.array([p.timestamp for p in self.poses])
        idx = np.searchsorted(times, target_time)

        if idx == 0:
            return self.poses[0]
        if idx >= len(self.poses):
            return self.poses[-1]

        # Interpolation factor
        t0, t1 = times[idx - 1], times[idx]
        alpha = (target_time - t0) / (t1 - t0) if t1 != t0 else 0

        p0, p1 = self.poses[idx - 1], self.poses[idx]

        # Interpolate position linearly
        pos_interp = (1 - alpha) * p0.position_mm + alpha * p1.position_mm

        # Interpolate rotation using SLERP
        r0 = Rotation.from_quat(p0.quaternion_xyzw)
        r1 = Rotation.from_quat(p1.quaternion_xyzw)

        # Use scipy's slerp
        from scipy.spatial.transform import Slerp
        slerp = Slerp([0, 1], Rotation.concatenate([r0, r1]))
        r_interp = slerp(alpha)

        return OptiTrackPose(
            frame_idx=p0.frame_idx,
            timestamp=target_time,
            quaternion_xyzw=r_interp.as_quat(),
            position_mm=pos_interp
        )

    def export_to_julia_format(self, output_path: str,
                               start_time: Optional[float] = None,
                               end_time: Optional[float] = None):
        """
        Export poses to Julia solver CSV format.

        Args:
            output_path: Path to output CSV file
            start_time: Optional start time filter
            end_time: Optional end time filter
        """
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            for pose in self.poses:
                if start_time and pose.timestamp < start_time:
                    continue
                if end_time and pose.timestamp > end_time:
                    continue

                q = pose.get_quaternion_wxyz()
                t = pose.get_position_meters()
                writer.writerow([q[0], q[1], q[2], q[3], t[0], t[1], t[2]])

        print(f"Exported poses to {output_path}")


def main():
    """Test the parser."""
    import argparse

    parser = argparse.ArgumentParser(description='Parse OptiTrack CSV')
    parser.add_argument('--csv', type=str,
                       default='data/session_008/5550 Take 12-08 03.08.47 PM.csv',
                       help='Path to OptiTrack CSV')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for Julia format CSV')

    args = parser.parse_args()

    # Get project root
    project_root = Path(__file__).parent.parent
    csv_path = project_root / args.csv

    print(f"Parsing {csv_path}")
    ot_parser = OptiTrackParser(str(csv_path))

    # Print some sample poses
    print("\nSample poses:")
    for i in [0, 100, 500, 1000]:
        if i < len(ot_parser.poses):
            p = ot_parser.poses[i]
            print(f"  Frame {p.frame_idx}, t={p.timestamp:.3f}s:")
            print(f"    Quat (wxyz): {p.get_quaternion_wxyz()}")
            print(f"    Pos (m): {p.get_position_meters()}")

    if args.output:
        output_path = project_root / args.output
        ot_parser.export_to_julia_format(str(output_path))


if __name__ == '__main__':
    main()
