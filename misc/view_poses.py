from matplotlib import pyplot as plt
import torch
import sys
sys.path.append("../")
from mpl_toolkits.mplot3d import Axes3D
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.model_components import ray_samplers
from nerfstudio.utils import plotly_utils as vis
# import open3d as o3d
import numpy as np
import json
from PIL import Image
import os
import rerun as rr


class PoseViewer:
    def __init__(self, JsonPath):
        self.JsonPath = JsonPath
        self.poses = []
        self.depth_paths = []
        self.rgb_paths = []
        self.K : np.ndarray
        self.load() # world to camera transformation matrices

    def load(self):
        json_metadata = json.load(open(self.JsonPath))
        base_path = os.path.dirname(self.JsonPath)
        self.cx = json_metadata['cx']
        self.cy = json_metadata['cy']
        self.fx = json_metadata['fl_x']
        self.fy = json_metadata['fl_y']
        self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
        frames = json_metadata['frames']
        for frame in frames:
            pose = frame['transform_matrix']
            pose = torch.tensor(pose).reshape(4, 4)
            self.poses.append(pose)
            self.depth_paths.append(os.path.join(base_path,frame['depth_file_path']))
            self.rgb_paths.append(os.path.join(base_path,frame['file_path']))

    def plot(self, is_scale_pose=False, aabb_scale=1.0, near=0.05, far=10.0):
        # rr.init("ycb_box", spawn = True)
        rr.init(
            "nerf_samples",
            recording_id="7a12aa85-8d31-40b5-9e67-599c4e164c93"
        )
        rr.connect()
        scale_factor = 1.0
        if is_scale_pose:
            scale_factor = 1.0 / float(np.max(np.abs(np.array(self.poses)[:, :3, 3])))
        print(f"{scale_factor=}")
        # draw a red point for the origin
        rr.log("scene_box", rr.Clear(recursive=True))
        rr.log("world", rr.Clear(recursive=True))
        rr.log("scene_box", rr.Boxes3D(centers=[0, 0, 0], sizes=[aabb_scale, aabb_scale, aabb_scale]), static=True)
        
        for i, cam_to_world in enumerate(self.poses):
            # which is already in OpenGL convention
            pose = cam_to_world.numpy()
            pose[:3, 3] *= scale_factor
            rr.set_time_sequence("stable_time",i)
            """Logs a point cloud and a perspective camera looking at it."""
            rgb = np.array(Image.open(self.rgb_paths[i]))
            rr.log("world/cam", rr.Pinhole(image_from_camera=self.K,
                                           camera_xyz=rr.ViewCoordinates.RUB
            ))
            rr.log("world/cam", rr.Transform3D(
                translation=pose[:3, 3].squeeze(),
                mat3x3=pose[:3, :3].squeeze(),
            ))
            rr.log("world/cam", rr.Image(rgb))
            depth_path = self.depth_paths[i]
            depth = np.array(Image.open(depth_path))
            depth = depth / 1000.0
            # Filter out the invalid depth values
            valid_mask = depth > 0.0
            d = depth[valid_mask]
            # Get the 3D points u,v,d
            v, u = np.where(valid_mask)
            # DownSampling
            u = u[::4]
            v = v[::4]
            d = d[::4]
            # get the 3D points xyz
            pcl_xyz_in_cam = np.linalg.inv(self.K) @ (d * np.stack([u, v, np.ones_like(u)], axis=0)) # (3, N)
            # OpenCV to OpenGL, RDF to RUB
            pcl_xyz_in_cam = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]) @ pcl_xyz_in_cam
            pcl_xyz_in_world = pose[:3, :3] @ pcl_xyz_in_cam + pose[:3, 3][:, None]
            rr.log("world/pcl", rr.Points3D(pcl_xyz_in_world.T, colors=rgb[v, u], radii=0.001 * np.ones_like(d)))
    
    def invert_pose_and_save(self, save_path):
        json_metadata = json.load(open(self.JsonPath))
        base_path = os.path.dirname(self.JsonPath)
        frames = json_metadata['frames']
        for i, frame in enumerate(frames):
            pose = self.poses[i].numpy()
            pose = np.linalg.inv(pose)
            # from OpenGL to OpenCV: RUB to RDF
            # https://docs.nerf.studio/quickstart/data_conventions.html#
            pose = pose @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            frame['transform_matrix'] = pose.tolist()
        with open(os.path.join(base_path,save_path), 'w') as f:
            json.dump(json_metadata, f, indent=4)
        print(f"Saved to {save_path}")
        

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--is_scale_pose", type=bool, default=False)
    parser.add_argument("--JsonPath", type=str, required=True)
    parser.add_argument("--aabb_scale", type=float, default=1.0)
    parser.add_argument("--near", type=float, default=0.05)
    parser.add_argument("--far", type=float, default=10.0)
    args = parser.parse_args()
    pose_viewer = PoseViewer(args.JsonPath)
    print(f"total poses: {len(pose_viewer.poses)}")
    pose_viewer.plot(is_scale_pose=args.is_scale_pose, aabb_scale=args.aabb_scale,near=args.near,far=args.far)
    # pose_viewer.invert_pose_and_save("box_21_c2w.json")
    
        

    
