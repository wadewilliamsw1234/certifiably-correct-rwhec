#!/usr/bin/env python3
"""AprilTag detection and pose estimation for RWHEC calibration."""

import cv2
import numpy as np
from pupil_apriltags import Detector
from pathlib import Path
from scipy.spatial.transform import Rotation
import csv
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, List
import json
import re


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: np.ndarray
    width: int
    height: int

    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, 'r') as f:
            content = f.read()

        width_match = re.search(r'image_width:\s*(\d+)', content)
        height_match = re.search(r'image_height:\s*(\d+)', content)
        width = int(width_match.group(1)) if width_match else 1280
        height = int(height_match.group(1)) if height_match else 720

        cam_match = re.search(r'camera_matrix:.*?data:\s*\[([\d.,\s\-]+)\]', content, re.DOTALL)
        if not cam_match:
            raise ValueError(f"Could not parse camera matrix from {yaml_path}")

        cam_data = [float(x.strip()) for x in cam_match.group(1).split(',')]
        fx, fy, cx, cy = cam_data[0], cam_data[4], cam_data[2], cam_data[5]

        dist_match = re.search(r'distortion_coefficients:.*?data:\s*\[([\d.,\s\-]+)\]', content, re.DOTALL)
        dist_data = [float(x.strip()) for x in dist_match.group(1).split(',')] if dist_match else [0]*5
        
        return cls(fx=fx, fy=fy, cx=cx, cy=cy, dist_coeffs=np.array(dist_data), 
                   width=width, height=height)

    def get_camera_matrix(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])


@dataclass
class TagDetection:
    tag_id: int
    tag_family: str
    corners: np.ndarray
    center: np.ndarray
    pose_R: np.ndarray
    pose_t: np.ndarray
    pose_err: float
    frame_idx: int
    timestamp: float

    def get_quaternion_wxyz(self):
        R = self.pose_R.copy()
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1
        U, _, Vt = np.linalg.svd(R)
        R_corrected = U @ Vt
        if np.linalg.det(R_corrected) < 0:
            U[:, -1] *= -1
            R_corrected = U @ Vt
        q = Rotation.from_matrix(R_corrected).as_quat()
        return np.array([q[3], q[0], q[1], q[2]])

    def get_transform_matrix(self):
        T = np.eye(4)
        T[:3, :3] = self.pose_R
        T[:3, 3] = self.pose_t.flatten()
        return T


class AprilTagDetector:
    TAG_FAMILIES = ['tag36h11', 'tag25h9', 'tag16h5', 'tagStandard41h12', 'tagCircle21h7']

    def __init__(self, camera_intrinsics: CameraIntrinsics, tag_size_meters: float = 0.150,
                 tag_family: Optional[str] = None):
        self.intrinsics = camera_intrinsics
        self.tag_size = tag_size_meters
        self.tag_family = tag_family
        self.camera_params = (camera_intrinsics.fx, camera_intrinsics.fy,
                              camera_intrinsics.cx, camera_intrinsics.cy)

        if tag_family:
            self.detectors = {tag_family: Detector(families=tag_family)}
        else:
            self.detectors = {}
            for family in self.TAG_FAMILIES:
                try:
                    self.detectors[family] = Detector(families=family)
                except Exception:
                    pass

    def undistort_image(self, image: np.ndarray):
        K = self.intrinsics.get_camera_matrix()
        return cv2.undistort(image, K, self.intrinsics.dist_coeffs)

    def detect(self, image: np.ndarray, frame_idx: int = 0, timestamp: float = 0.0,
               undistort: bool = True) -> List[TagDetection]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        if undistort:
            gray = self.undistort_image(gray)

        detections = []
        for family, detector in self.detectors.items():
            results = detector.detect(gray, estimate_tag_pose=True,
                                      camera_params=self.camera_params, tag_size=self.tag_size)
            for r in results:
                if r.pose_R is not None and r.pose_t is not None:
                    detections.append(TagDetection(
                        tag_id=r.tag_id, tag_family=family, corners=r.corners,
                        center=r.center, pose_R=r.pose_R, pose_t=r.pose_t,
                        pose_err=getattr(r, 'pose_err', 0.0),
                        frame_idx=frame_idx, timestamp=timestamp
                    ))
        return detections

    def auto_detect_family(self, video_path: str, num_frames: int = 30):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        family_counts = {family: 0 for family in self.detectors.keys()}

        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            gray = self.undistort_image(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            for family, detector in self.detectors.items():
                family_counts[family] += len(detector.detect(gray, estimate_tag_pose=False))

        cap.release()
        best_family = max(family_counts, key=family_counts.get)
        
        if family_counts[best_family] > 0:
            print(f"Detected: {best_family} ({family_counts[best_family]} tags)")
            self.tag_family = best_family
            self.detectors = {best_family: self.detectors[best_family]}
            return best_family
        return None


def process_video(video_path: str, detector: AprilTagDetector, output_dir: str,
                  camera_name: str, fps: Optional[float] = None, start_frame: int = 0,
                  end_frame: Optional[int] = None, visualize: bool = False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame is None:
        end_frame = total_frames

    print(f"Processing {video_path}: frames {start_frame}-{end_frame}, fps={fps}")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_detections: Dict[int, List[TagDetection]] = {}
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    det_count = 0

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame, frame_idx, frame_idx / fps)
        if detections:
            all_detections[frame_idx] = detections
            det_count += len(detections)

        if frame_idx % 500 == 0:
            print(f"  {frame_idx}/{end_frame} ({det_count} detections)")

        if visualize and detections:
            vis = frame.copy()
            for det in detections:
                cv2.polylines(vis, [det.corners.astype(int)], True, (0, 255, 0), 2)
                c = tuple(det.center.astype(int))
                cv2.putText(vis, str(det.tag_id), c, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.imshow(f'{camera_name}', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1

    cap.release()
    if visualize:
        cv2.destroyAllWindows()

    print(f"  Done: {det_count} detections in {len(all_detections)} frames")
    save_detections_to_csv(all_detections, output_path, camera_name)
    save_detections_to_json(all_detections, output_path, camera_name)
    return all_detections


def save_detections_to_csv(detections: Dict[int, List[TagDetection]], output_dir: Path, camera_name: str):
    by_tag: Dict[int, List[TagDetection]] = {}
    for frame_dets in detections.values():
        for det in frame_dets:
            by_tag.setdefault(det.tag_id, []).append(det)

    for tag_id, tag_dets in by_tag.items():
        tag_dets.sort(key=lambda d: d.frame_idx)
        csv_path = output_dir / f"tag_{tag_id}_{camera_name}_B.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for det in tag_dets:
                q = det.get_quaternion_wxyz()
                t = det.pose_t.flatten()
                writer.writerow([q[0], q[1], q[2], q[3], t[0], t[1], t[2]])
        print(f"  tag_{tag_id}: {len(tag_dets)} poses")

    idx_path = output_dir / f"{camera_name}_frame_index.csv"
    with open(idx_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_idx', 'timestamp', 'tag_ids'])
        for frame_idx in sorted(detections.keys()):
            dets = detections[frame_idx]
            writer.writerow([frame_idx, dets[0].timestamp if dets else 0,
                           ','.join(str(d.tag_id) for d in dets)])


def save_detections_to_json(detections: Dict[int, List[TagDetection]], output_dir: Path, camera_name: str):
    data = {}
    for frame_idx, dets in detections.items():
        data[frame_idx] = [{
            'tag_id': det.tag_id, 'tag_family': det.tag_family,
            'center': det.center.tolist(), 'corners': det.corners.tolist(),
            'quaternion_wxyz': det.get_quaternion_wxyz().tolist(),
            'translation': det.pose_t.flatten().tolist(),
            'pose_error': det.pose_err, 'timestamp': det.timestamp
        } for det in dets]

    with open(output_dir / f"{camera_name}_detections.json", 'w') as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str)
    parser.add_argument('--camera', type=str, choices=['left', 'right'], default='left')
    parser.add_argument('--intrinsics', type=str)
    parser.add_argument('--tag-size', type=float, default=0.150)
    parser.add_argument('--tag-family', type=str, default=None)
    parser.add_argument('--output', type=str, default='output/detections')
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--end-frame', type=int, default=None)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    root = Path(__file__).parent.parent

    if args.video is None:
        args.video = str(root / f'data/session_008/{args.camera}_synced.mp4')
    if args.intrinsics is None:
        cam_num = '1' if args.camera == 'left' else '2'
        args.intrinsics = str(root / f'config/camera_instrinsics/cam_{cam_num}_intrinsics.yaml')
    if args.output == 'output/detections':
        args.output = str(root / 'output/detections')

    print(f"Intrinsics: {args.intrinsics}")
    intrinsics = CameraIntrinsics.from_yaml(args.intrinsics)

    detector = AprilTagDetector(intrinsics, args.tag_size, args.tag_family)
    if args.tag_family is None:
        detector.auto_detect_family(args.video)

    process_video(args.video, detector, args.output, args.camera,
                  start_frame=args.start_frame, end_frame=args.end_frame,
                  visualize=args.visualize)


if __name__ == '__main__':
    main()