#!/usr/bin/env python3
"""OptiTrack CSV parser with SE(3) interpolation (SLERP + linear)."""

import numpy as np
import csv
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d


@dataclass
class OptiTrackPose:
    timestamp: float
    position: np.ndarray      # [x, y, z] in mm
    quaternion: np.ndarray    # [qx, qy, qz, qw]
    
    def get_position_meters(self):
        return self.position / 1000.0
    
    def get_quaternion_wxyz(self):
        return np.array([self.quaternion[3], self.quaternion[0], 
                        self.quaternion[1], self.quaternion[2]])
    
    def get_rotation_matrix(self):
        return Rotation.from_quat(self.quaternion).as_matrix()


class OptiTrackParserSE3:
    """Parse OptiTrack CSV with SE(3) interpolation."""
    
    def __init__(self, csv_path: str):
        self.poses: List[OptiTrackPose] = []
        self.timestamps = None
        self._rot_interp = None
        self._trans_interp = None
        self._parse(csv_path)
        self._build_interpolators()
    
    def _parse(self, csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i < 7 or len(row) < 9:
                    continue
                try:
                    time = float(row[1])
                    qx, qy, qz, qw = float(row[2]), float(row[3]), float(row[4]), float(row[5])
                    x, y, z = float(row[6]), float(row[7]), float(row[8])
                    
                    if np.isnan([qx, qy, qz, qw, x, y, z]).any():
                        continue
                    if np.allclose([qx, qy, qz, qw], 0):
                        continue
                    
                    self.poses.append(OptiTrackPose(
                        timestamp=time,
                        position=np.array([x, y, z]),
                        quaternion=np.array([qx, qy, qz, qw])
                    ))
                except (ValueError, IndexError):
                    continue
        
        if self.poses:
            self.timestamps = np.array([p.timestamp for p in self.poses])
            print(f"  Loaded {len(self.poses)} poses, t=[{self.timestamps.min():.2f}, {self.timestamps.max():.2f}]s")
    
    def _build_interpolators(self):
        if len(self.poses) < 2:
            return
        
        rotations = Rotation.from_quat([p.quaternion for p in self.poses])
        translations = np.array([p.get_position_meters() for p in self.poses])
        
        self._rot_interp = Slerp(self.timestamps, rotations)
        self._trans_interp = interp1d(self.timestamps, translations, axis=0,
                                    kind='linear', bounds_error=False, fill_value='extrapolate')
    
    def get_pose_at_time(self, t: float) -> Optional[OptiTrackPose]:
        if self._rot_interp is None:
            return None
        
        tol = 0.01
        if t < self.timestamps.min() - tol or t > self.timestamps.max() + tol:
            return None
        
        t = np.clip(t, self.timestamps.min(), self.timestamps.max())
        
        rot = self._rot_interp(t)
        trans = self._trans_interp(t)
        
        return OptiTrackPose(
            timestamp=t,
            position=trans * 1000.0,
            quaternion=rot.as_quat()
        )
    
    def get_pose_interpolated(self, t: float) -> Optional[OptiTrackPose]:
        return self.get_pose_at_time(t)
    
    def get_pose_nearest(self, t: float) -> Optional[OptiTrackPose]:
        if not self.poses:
            return None
        idx = np.argmin(np.abs(self.timestamps - t))
        return self.poses[idx]


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        parser = OptiTrackParserSE3(sys.argv[1])