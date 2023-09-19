# Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import glob
import json
import cv2

import torch
import numpy as np

from render import util

# from dataset import Dataset
from .dataset_openillum_base import DatasetOpenIllumBase

###############################################################################
# NERF image based dataset (synthetic)
###############################################################################

def _load_img(path, ds=1.0):
    files = glob.glob(path + '.*')
    assert len(files) > 0, "Tried to find image file for: %s, but found 0 files" % (path)
    img = util.load_image_raw(files[0])
    if ds != 1.0:
        img_hw_full = img.shape[:2]
        img_wh_ds = (int(img_hw_full[1] * ds), int(img_hw_full[0] * ds))
        img = cv2.resize(img, img_wh_ds, interpolation = cv2.INTER_AREA)
    if img.dtype != np.float32: # LDR image
        img = torch.tensor(img / 255, dtype=torch.float32)
        img[..., 0:3] = util.srgb_to_rgb(img[..., 0:3])
    else:
        img = torch.tensor(img, dtype=torch.float32)
    return img

def _load_mask(path, ds=1.0):
    files = glob.glob(path + '.*')
    assert len(files) > 0, "Tried to find image file for: %s, but found 0 files" % (path)
    img = util.load_image_raw(files[0])
    if ds != 1.0:
        img_hw_full = img.shape[:2]
        img_wh_ds = (int(img_hw_full[1] * ds), int(img_hw_full[0] * ds))
        img = cv2.resize(img, img_wh_ds, interpolation = cv2.INTER_AREA)
    img = torch.tensor(img / 255, dtype=torch.bool)
    return img

def scaled_perspective(fovy=0.7854, aspect=1.0, n=0.1, f=1000.0, downsample=1, cx=0, cy=0, h=1, w=1, device=None):
    # y = np.tan(fovy / 2)/downsample
    y = np.tan(fovy / 2)
    x0 = 2*cx/w - downsample
    y0 = 2*cy/h - downsample
    return torch.tensor([[1/(y*aspect),    0,            x0,              0], 
                         [           0, 1/-y,            y0,              0], 
                         [           0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)], 
                         [           0,    0,           -1,              0]], dtype=torch.float32, device=device)

class DatasetOpenIllum(DatasetOpenIllumBase):
    def __init__(self, cfg_path, FLAGS, examples=None):
        self.FLAGS = FLAGS
        self.examples = examples
        self.base_dir = os.path.dirname(os.path.dirname(cfg_path))

        # Load config / transforms
        self.cfg = json.load(open(cfg_path, 'r'))
        self.n_images = len(self.cfg['frames'])

        # Determine resolution & aspect ratio
        item_idx = list(self.cfg['frames'].keys())[0]
        item_path = os.path.join(self.base_dir, 'Lights/013/raw_undistorted', item_idx)
        self.resolution = _load_img(item_path).shape[0:2]
        self.resolution_to_use = (1200, 800)  # TODO
        # self.resolution_to_use = self.resolution
        self.ds_ratio = self.resolution_to_use[0] / self.resolution[0]
        self.aspect = self.resolution[1] / self.resolution[0]
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        print("DatasetOpenIllum: %d images with shape [%d, %d]" % (self.n_images, self.resolution_to_use[0], self.resolution_to_use[1]))

        # Pre-load from disc to avoid slow png parsing
        if self.FLAGS.pre_load:
            self.preloaded_data = []
            for i in range(self.n_images):
                self.preloaded_data += [self._parse_frame(self.cfg, i)]

    def _parse_frame(self, cfg, idx):
        item_idx = list(cfg['frames'].keys())[idx]
        
        # Config projection matrix (static, so could be precomputed)
        cam_angle_x = cfg['frames'][item_idx]['camera_angle_x']
        # Resize fov
        # cam_angle_x /= self.ds_ratio
        fovy   = util.fovx_to_fovy(cam_angle_x, self.aspect)
        # proj   = util.perspective(fovy, self.aspect, self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])
        proj   = scaled_perspective(fovy, self.aspect, 
                                    self.FLAGS.cam_near_far[0], 
                                    self.FLAGS.cam_near_far[1], 
                                    downsample=1/self.ds_ratio, 
                                    h=self.resolution_to_use[0],
                                    w=self.resolution_to_use[1],
                                    cx=self.resolution[1] / 2,
                                    cy=self.resolution[0] / 2)

        # Load image data and modelview matrix
        img_path = os.path.join(self.base_dir, 'Lights/013/raw_undistorted', item_idx)
        img    = _load_img(img_path, ds=self.ds_ratio)
        # Load common mask and set background as white
        com_mask_path = os.path.join(self.base_dir, 'output/com_masks', item_idx)
        com_mask   = _load_mask(com_mask_path, ds=self.ds_ratio)  # [1200, 800]
        img[~com_mask] = 1.0
        # Compose alpha image
        img = torch.cat([img, com_mask[..., None]], dim=-1)
        
        # Load object mask
        obj_mask_path = os.path.join(self.base_dir, 'output/obj_masks', item_idx)
        obj_mask = _load_mask(obj_mask_path, ds=self.ds_ratio)  # [1200, 800]
        
        pose = np.array(cfg['frames'][item_idx]['transform_matrix'], dtype=np.float32) @ self.blender2opencv
        mv     = torch.linalg.inv(torch.tensor(pose, dtype=torch.float32))
        mv     = mv @ util.rotate_x(-np.pi / 2)

        campos = torch.linalg.inv(mv)[:3, 3]
        mvp    = proj @ mv

        return img[None, ...], mv[None, ...], mvp[None, ...], campos[None, ...], com_mask[None, ...], obj_mask[None, ...] # Add batch dimension

    def getMesh(self):
        return None # There is no mesh

    def __len__(self):
        return self.n_images if self.examples is None else self.examples

    def __getitem__(self, itr):
        img      = []
        # fovy     = util.fovx_to_fovy(self.cfg['camera_angle_x'], self.aspect)

        if self.FLAGS.pre_load:
            img, mv, mvp, campos, com_mask, obj_mask = self.preloaded_data[itr % self.n_images]
        else:
            img, mv, mvp, campos, com_mask, obj_mask = self._parse_frame(self.cfg, itr % self.n_images)

        return {
            'mv' : mv,
            'mvp' : mvp,
            'campos' : campos,
            'resolution' : self.FLAGS.train_res,
            'spp' : self.FLAGS.spp,
            'img' : img,
            'com_mask' : com_mask,
            'obj_mask' : obj_mask,
        }


if __name__ == '__main__':
    ref_mesh = '/home/isabella/Lab/OpenIllumination/data/OpenIllumination/20230524-13_08_01_obj_2_egg'
    iter = 5000
    batch = 8
    FLAGS = dict()
    FLAGS['pre_load'] = True
    dataset_train = DatasetOpenIllum(
        os.path.join(ref_mesh, 'transforms_train.json'), 
        FLAGS, 
        examples=(iter+1)*batch
        )