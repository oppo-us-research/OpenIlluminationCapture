"""
Author: Haian Jin 5/24/22
Feature: DataLoader for Pure TensoRF
"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from dataLoader.ray_utils import *
from models.relight_utils import gen_light_xyz


class ShapeBufferDataset(Dataset):
    def __init__(self,
                 shape_buffer_dir,  
                 split='train',
                 is_stack=False,
                 is_jitter = False,
                 sub=0,
                 ):
        """
        @param root_dir: str | Root path of dataset folder
        @param shape_buffer_dir: str | path of the folder that containes visibility file, surface file and so on
        @param split: str | e.g. 'train' / 'test'
        @param is_stack: bool | Whether stack all rays / rgbs from different frames, if yes [frames*h*w, 6]
        else [frames, h*w, 6]
        @param downsample: float | Downsample ratio for input rgb images
        @param n_views: int | Number of neighbor views used to build MVS cost volume
        """
        self.shape_buffer_dir = Path(shape_buffer_dir)
        self.split = split
        self.split_list = [x for x in self.shape_buffer_dir.iterdir() if x.stem.startswith(self.split)]
        if sub > 0:
            self.split_list = self.split_list[:sub]
        self.is_stack = is_stack
        self.img_wh = (800, 800)
        self.scan = self.shape_buffer_dir.stem # hotdog for this experiment

        self.light_xyz = None
        self.light_areas = None
        self.lights_probes = {}
        self.read_lights()



        # .npy file is too large, so can only be loaded when used
        
        # self.all_surface_points = []
        # self.all_normals = []
        # self.all_visibility = []
        # self.all_acc = []


        # self.dataset_items = []

        # for idx in range(len(self.split_list)):
        #     item_path = self.split_list[idx]

        #     # read normals
        #     # not used now
            
        #     # read acc map
        #     acc_path = item_path / f'acc.npy'
        #     acc_map = torch.from_numpy(np.load(acc_path))
        #     acc_map = acc_map.view(-1) # (H * W)
        #     self.all_acc.append(acc_map)

        #     # read surface points
        #     surface_path = item_path / f'surface_xyz.npy'
        #     surface_points = torch.from_numpy(np.load(surface_path))
        #     surface_points = surface_points.view(-1, 3) # (H * W, 3)
        #     self.all_surface_points.append(surface_points)
            
        #     # read viisibility

        #     visibility_path = item_path / f'visibility.npy'
        #     visibility = torch.from_numpy(np.load(visibility_path))
        #     visibility = visibility.view(-1, self.light_xyz.shape[0]) # (H * W, envH * envW)




    def read_lights(self):
        envmap_h, envmap_w = 16, 32  # TODO mergeg to config
        nlights = envmap_h * envmap_w
        light_xyz, light_areas, _ = gen_light_xyz(envmap_h, envmap_w)
        self.light_xyz = torch.from_numpy(light_xyz.reshape(nlights, 3)).float()  # [envH * envW, 3]
        self.light_areas = torch.from_numpy(light_areas.reshape(nlights)).float()  # [envH * envW,]


  

    def __len__(self):
        return len(self.split_list)

    def __getitem__(self, idx):
        
        item_path = self.split_list[idx]

        # read normals
        # not used now
        
        # read acc map
        acc_path = item_path / f'acc.npy'
        acc_map = torch.from_numpy(np.load(acc_path))
        acc_map = acc_map.view(-1) # (H * W)


        # read surface points
        surface_path = item_path / f'surface_xyz.npy'
        surface_points = torch.from_numpy(np.load(surface_path))
        surface_points = surface_points.view(-1, 3) # (H * W, 3)

        # read viisibility

        visibility_path = item_path / f'visibility.npy'
        visibility = torch.from_numpy(np.load(visibility_path))
        visibility = visibility.view(-1, self.light_xyz.shape[0]) # (H * W, envH * envW)


        return acc_map, surface_points, visibility

# used for test
if __name__ == "__main__":
    from opt import config_parser

    args = config_parser()

    dataset = ShapeBufferDataset(
        shape_buffer_dir='/home/haian/Dataset/NeRF_DATA/tensorfactor/hotdog_geo_buffer/',
        split='train'
    )

    # Test 1: Get single item
    acc_map, surface_points, visibility = dataset.__getitem__(0)
    print(acc_map.shape)
    print(surface_points.shape)
    print(visibility.shape)


    # Test 2: Iteration
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1, drop_last=True, shuffle=True)
    train_iter = iter(train_dataloader)
    for i in range(20):
        try:
            acc_map, surface_points, visibility = next(train_iter)
            print(acc_map.shape)
            print(surface_points.shape)
            print(visibility.shape)

        except StopIteration:
            print('Start a new iteration from the dataloader')
            train_iter = iter(train_dataloader)


