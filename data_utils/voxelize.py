import glob
import pdb
import time

import numpy as np
# import open3d as o3d
import trimesh
from pdb import set_trace as bp

from tqdm import tqdm
# import pandas as pd
import os

import argparse

class Voxel():

    def __init__(self, root='/home/yuefanw/scratch/version1',normalize=False):

        self.want_normalize = normalize

        self.root = root


        self.resolution = 128
        self.b_min = np.array([-2, -2, -2])
        self.b_max = np.array([5, 5, 5])
        self.transform = None
        self.num_samples = 2097152
        self.max_score = -100

        self.time_count = 0

    def create_grid(self, resX, resY, resZ, b_min=np.array([0, 0, 0]), b_max=np.array([1, 1, 1]), transform=None):
        '''
        Create a dense grid of given resolution and bounding box
        :param resX: resolution along X axis
        :param resY: resolution along Y axis
        :param resZ: resolution along Z axis
        :param b_min: vec3 (x_min, y_min, z_min) bounding box corner
        :param b_max: vec3 (x_max, y_max, z_max) bounding box corner
        :return: [3, resX, resY, resZ] coordinates of the grid, and transform matrix from mesh index
        '''
        coords = np.mgrid[:resX, :resY, :resZ]
        coords = coords.reshape(3, -1)
        coords_matrix = np.eye(4)
        length = b_max - b_min
        coords_matrix[0, 0] = length[0] / resX
        coords_matrix[1, 1] = length[1] / resY
        coords_matrix[2, 2] = length[2] / resZ
        coords_matrix[0:3, 3] = b_min
        coords = np.matmul(coords_matrix[:3, :3], coords) + coords_matrix[:3, 3:4]
        if transform is not None:
            coords = np.matmul(transform[:3, :3], coords) + transform[:3, 3:4]
            coords_matrix = np.matmul(transform, coords_matrix)
        coords = coords.reshape(3, resX, resY, resZ)
        return coords, coords_matrix

    def normalize(self, mesh):
        scale = 4 / np.max(mesh.extents)
        matrix = np.eye(4)
        matrix[:3, :3] *= scale
        mesh.apply_transform(matrix)

        return mesh


    def get_voxel(self,mesh_gt):

        if type(mesh_gt) == str:
            mesh_gt = trimesh.load(mesh_gt)
        if self.want_normalize:
            mesh_gt = self.normalize(mesh_gt)

        coords, mat = self.create_grid(self.resolution, self.resolution, self.resolution,
                            self.b_min, self.b_max, transform=self.transform)

        coords = coords.reshape([3, -1])
        coords = coords.T
        voxel_array = mesh_gt.contains(coords)
        voxel_array = voxel_array.reshape((self.resolution,self.resolution,self.resolution))
        return voxel_array


if __name__ =='__main__':
    vv = Voxel()
    # test_mesh = '/home/yuefanw/scratch/version1/african_elephant_female/frame_000001.obj'
    root_dir = '/home/yuefanw/scratch/version1'
    # frame_list = glob.glob(os.path.join(root_dir,'*/*.obj'))
    # result = vv.get_voxel(test_mesh)

    out_dir = '/home/yuefanw/scratch/voxel1'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # for animal in tqdm(os.listdir(root_dir)):
    animal = 'dingo_male'
    if not os.path.exists(os.path.join(out_dir,animal)):
        os.makedirs(os.path.join(out_dir,animal))
    frame_list = glob.glob(os.path.join(root_dir,animal,'*.obj'))
    for frame in tqdm(frame_list):
        result = vv.get_voxel(frame)
        # pdb.set_trace()
        np.save(os.path.join(out_dir,animal,frame.split('/')[-1].replace('obj','npy')),result)

    # pdb.set_trace()