"""
Author: Haian Jin 5/24/22
Feature: DataLoader for Pure TensoRF
"""

import os, random
import json
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from dataLoader.ray_utils import *
from models.relight_utils import gen_light_xyz


class TensoRF_Init_Dataset(Dataset):
    def __init__(self,
                 root_dir,
                 split='train',
                 is_stack=False,
                 N_vis=-1,
                 downsample=1.0,
                 sub=0,
                 light_name = "sunset",
                 n_views=3
                 ):
        """
        @param root_dir: str | Root path of dataset folder
        @param split: str | e.g. 'train' / 'test'
        @param is_stack: bool | Whether stack all rays / rgbs from different frames, if yes [frames*h*w, 6]
        else [frames, h*w, 6]
        @param N_vis: int | If N_vis > 0, select N_vis frames from the dataset, else (-1) import entire dataset
        @param downsample: float | Downsample ratio for input rgb images
        @light_name: str | Name of the light environment
        @param n_views: int | Number of neighbor views used to build MVS cost volume
        """
        self.N_vis = N_vis
        self.root_dir = Path(root_dir)
        self.split = split
        self.split_list = [x for x in self.root_dir.iterdir() if x.stem.startswith(self.split)]
        if sub > 0:
            self.split_list = self.split_list[:sub]
        self.is_stack = is_stack
        self.img_wh = (int(800/downsample),int(800/downsample))
        self.white_bg = True
        self.downsample = downsample
        self.transform = self.define_transforms()
        self.light_name = light_name # what lighting condition the pretrained tensorf will be
        self.scan = self.root_dir.stem # hotdog for this experiment
        ## TODO
        self.near_far = [2.0, 6.0]  # TODO
        self.scene_bbox = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]]) * self.downsample
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


        self.light_xyz = None
        self.light_areas = None
        self.lights_probes = {}
        self.read_lights()

        self.all_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []
        self.all_depth = []
        
        self.dataset_items = []

        for idx in range(len(self.split_list)):
            item_path = self.split_list[idx]

            item_meta_path:Path = item_path / 'metadata.json'
            self.all_paths.append(os.fspath(item_path))

            with open(item_meta_path, 'r') as f:
                meta = json.load(f)
            img_wh = (int(meta['imw'] / self.downsample), int(meta['imh'] / self.downsample))

            # Get ray directions for all pixels, same for all images (with same H, W, focal)
            focal = 0.5 * int(meta['imw']) / np.tan(0.5 * meta['cam_angle_x'])  # fov -> focal length
            focal *= img_wh[0] / meta['imw']
            directions = get_ray_directions(img_wh[1], img_wh[0], [focal, focal])  # [H, W, 3]
            directions = directions / torch.norm(directions, dim=-1, keepdim=True)
            intrinsics = torch.tensor([[focal, 0, img_wh[0] / 2], [0, focal, img_wh[1] / 2], [0, 0, 1]]).float()  # [3, 3]
            # TODO should change if update metadata.json cam_trans
            cam_trans = np.array(list(map(float, meta["cam_transform_mat"].split(',')))).reshape(4, 4)
            pose = cam_trans @ self.blender2opencv
            c2w = torch.FloatTensor(pose)  # [4, 4]
            self.poses += [c2w]

            w2c = torch.linalg.inv(c2w)  # [4, 4]

            # Read RGB data
            img_path = item_path / f'rgba_{self.light_name}.png'
            img = Image.open(img_path)
            if self.downsample != 1.0:
                img = img.resize(img_wh, Image.Resampling.LANCZOS)
            img = self.transform(img)  # [4, H, W]
            img = img.view(4, -1).permute(1, 0)  # [H*W, 4]
            ## Blend A to RGB
            rgbs = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # [H*W, 3]
            self.all_rgbs += [rgbs]

            rgbs_mask = ~(rgbs[:, -1:] == 0)

            # Read ray data
            rays_o, rays_d = get_rays(directions, c2w)
            rays = torch.cat([rays_o, rays_d], 1)  # [H*W, 6]
            self.all_rays += [rays]  # (h*w, 6)

            # item = {
            #     'img_wh': img_wh,  # (int, int)
            #     # 'near_far': near_far,  # [float, float]
            #     'rgbs': rgbs,  # [H*W, 3],
            #     'rgbs_mask': rgbs_mask,  # [H*W, 1]
            #     'rays': rays,  # [H*W, 6]
            #     'c2w': c2w,  # [4, 4]
            #     'w2c': w2c  # [4, 4]
            # }
            # self.dataset_items.append(item)
        

        self.poses = torch.stack(self.poses)
        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)
#             self.all_depth = torch.cat(self.all_depth, 0)  # (len(self.meta['frames])*h*w, 3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
            # self.all_masks = torch.stack(self.all_masks, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)



    def define_transforms(self):
        transforms = T.Compose([
            T.ToTensor(),
        ])
        return transforms

    def read_lights(self):
        envmap_h, envmap_w = 16, 32  # TODO mergeg to config
        nlights = envmap_h * envmap_w
        light_xyz, light_areas = gen_light_xyz(envmap_h, envmap_w)
        self.light_xyz = torch.from_numpy(light_xyz.reshape(nlights, 3)).float()  # [envH * envW, 3]
        self.light_areas = torch.from_numpy(light_areas.reshape(nlights)).float()  # [envH * envW,]


    def read_mvs(self):
        pairs_file = self.root_dir / f'pairs_{self.split}.txt'
        n_ref_views = len(open(pairs_file).readlines())

        self.mvs_views = []
        with open(pairs_file) as f:
            for _ in range(n_ref_views):
                cur_views = [int(x) for x in f.readline().rstrip().split()]
                if len(cur_views) == 0:
                    break
                ref_view = cur_views[0]
                src_views = cur_views[1:]
                self.mvs_views += [(self.scan, ref_view, src_views)]

    def world2ndc(self, points, lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)

    def read_stack(self):
        for idx in range(self.__len__()):
            item = self.__getitem__(idx)
            rays = item['rays']
            rgbs = item['rgbs']
            self.all_rays += [rays]
            self.all_rgbs += [rgbs]
        self.all_rays = torch.stack(self.all_rays, 0)  # [len(self), H*W, 6]
        self.all_rgbs = torch.stack(self.all_rgbs, 0)  # [len(self), H*W, 3]

    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):
        # item = self.dataset_items[idx]

        # # item = {
        # #     'img_wh': img_wh,  # (int, int)
        # #     'rgbs': rgbs,  # [H*W, 3],
        # #     'rgbs_mask': rgbs_mask,  # [H*W, 1]
        # #     'rays': rays,  # [H*W, 6]
        # #     'c2w': c2w,  # [4, 4]
        # #     'w2c': w2c  # [4, 4]
        # # }

        # return item

        path_idx = (idx // (self.img_wh[0] * self.img_wh[0])) if not self.is_stack else idx
        item_path = self.all_paths[path_idx]
        normal_path = os.path.join(item_path, 'normal.png')
        normal_img = Image.open(normal_path)
        normal = np.array(normal_img)[..., :3] / 255  # [H, W, 3] in range [0, 1]
        normal = (normal - 0.5) * 2.0  # [H, W, 3] in range (-1, 1)
        normal = torch.from_numpy(normal).float().reshape(-1, 3)  # [H, W, 3]
        if self.split == 'train':  # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx],
                      'paths': self.all_paths[path_idx]
                      }

        else:  # create data for each image separately

            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            # mask = self.all_masks[idx] # for quantity evaluation

            sample = {'rays': rays,
                      'rgbs': img,
                      'gt_normals': normal,
                      'paths': self.all_paths[path_idx]
                    #   'mask': mask
                      }
        return sample


if __name__ == "__main__":
    from opt import config_parser

    args = config_parser()

    dataset = TensoRF_Init_Dataset(
        root_dir='/home/haian/Dataset/NeRF_DATA/hotdog/',
        split='train',
        downsample=2.0,
        is_stack=False
    )

    # Test 1: Get single item
    item = dataset.__getitem__(0)
    print(dataset.img_wh)
    print(item['rgbs'].shape)

    # Test 2: Iteration
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1, drop_last=True, shuffle=True)
    train_iter = iter(train_dataloader)
    for i in range(20):
        try:
            item = next(train_iter)
            print(item.keys())
            print(item['rays'].shape)
            print(item['paths'])
        except StopIteration:
            print('Start a new iteration from the dataloader')
            train_iter = iter(train_dataloader)

    # Test 3: Test dataset all stack
    # test_dataset = TensoRFactorDataset(
    #     root_dir='/code/MVSNeRFactor/data/nerfactor_synthesis/hotdog',
    #     hdr_dir='/code/MVSNeRFactor/data/low_res_envmaps_32_16',
    #     split='test',
    #     downsample=1.0,
    #     is_stack=True
    # )
    # print(test_dataset.all_rays.shape)  # [4, 640000, 6]
    # print(test_dataset.all_rgbs.shape)  # [4, 640000, 3]
