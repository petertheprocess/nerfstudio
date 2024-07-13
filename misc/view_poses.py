from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import json

class PoseViewer:
    def __init__(self, JsonPath):
        self.JsonPath = JsonPath
        self.poses = self.load_poses() # world to camera transformation matrices
    
    def load_poses(self):
        json_metadata = json.load(open(self.JsonPath))
        frames = json_metadata['frames']
        poses = []
        depth_paths = []
        for frame in frames:
            pose = frame['transform_matrix']
            pose = np.array(pose).reshape(4, 4)
            poses.append(pose)

        return poses
    
    def plot_poses(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        scale_factor = 1.0 / float(np.max(np.abs(np.array(self.poses)[:, :3, 3])))
        print(f"{scale_factor=}")

        for pose in self.poses:
            pose[:3, 3] *= scale_factor
            origin = pose[:3, 3]
            x_axis = pose[:3, 0]
            y_axis = pose[:3, 1]
            z_axis = pose[:3, 2]
            ax.quiver(origin[0], origin[1], origin[2], x_axis[0], x_axis[1], x_axis[2], color='r')
            ax.quiver(origin[0], origin[1], origin[2], y_axis[0], y_axis[1], y_axis[2], color='g')
            ax.quiver(origin[0], origin[1], origin[2], z_axis[0], z_axis[1], z_axis[2], color='b')
        
        
        # draw a unit cube in origin
        r = [-0.5, 0.5]
        points = np.array([[x, y, z] for x in r for y in r for z in r])
        for s, e in [[0, 1], [1, 3], [3, 2], [2, 0], [4, 5], [5, 7], [7, 6], [6, 4], [0, 4], [1, 5], [2, 6], [3, 7]]:
            ax.plot3D(*points[[s, e]].T, color='k')
        
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()
        

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--JsonPath", type=str, required=True)
    args = parser.parse_args()
    pose_viewer = PoseViewer(args.JsonPath)
    pose_viewer.plot_poses()

        
        
