import os
import torch
import numpy as np

import utils.general as utils
from utils import rend_util
import json
import imageio


class OpenIllumDataset(torch.utils.data.Dataset):
    def __init__(self,
                 instance_dir,
                 frame_skip,
                 split='train',
                 img_hw=(1200, 800),
                 ):
        self.instance_dir = instance_dir
        print('Creating dataset from: ', self.instance_dir)
        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.split = split

        json_path = os.path.join(self.instance_dir, 'output', 'transforms_{}.json'.format(split))
        print('Read cam from {}'.format(json_path))
        with open(json_path, 'r') as fp:
            meta = json.load(fp)
        
        image_paths = []
        com_mask_paths = []
        obj_mask_paths = []

        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        poses = []
        intrinsics = []
        
        # Get image downsize ratio
        item_id = list(meta['frames'].keys())[0]
        img_h_full, img_w_full = rend_util.load_rgb(os.path.join(self.instance_dir, 'Lights', '013/raw_undistorted', f'{item_id}.JPG')).shape[:2]
        ds_ratio = img_hw[0] / img_h_full
        
        for frame_id in meta['frames']:
            frame = meta['frames'][frame_id]
            
            # Read camera poses
            poses.append(np.array(frame['transform_matrix']) @ self.blender2opencv)
            
            # Read camera intrinsics
            camera_angle_x = float(frame['camera_angle_x'])
            focal = .5 * img_w_full / np.tan(.5 * camera_angle_x)
            focal *= ds_ratio  # modify focal length to match size of downsampled image
            intrinsics.append(np.array([[focal, 0, img_hw[1]/2], [0, focal, img_hw[0]/2], [0, 0, 1]]))
            
            # Read images
            image_paths.append(os.path.join(self.instance_dir, 'Lights', '013/raw_undistorted', f'{frame_id}.JPG'))
            com_mask_paths.append(os.path.join(self.instance_dir, 'output', 'com_masks', f'{frame_id}.png'))
            obj_mask_paths.append(os.path.join(self.instance_dir, 'output', 'obj_masks', f'{frame_id}.png'))

        # skip for training
        image_paths = image_paths[::frame_skip]
        com_mask_paths = com_mask_paths[::frame_skip]
        obj_mask_paths = obj_mask_paths[::frame_skip]
        poses = poses[::frame_skip]
        intrinsics = intrinsics[::frame_skip]
        
        print('Training image: {}'.format(len(image_paths)))
        self.image_paths = image_paths
        self.com_mask_paths = com_mask_paths
        self.obj_mask_paths = obj_mask_paths

        self.single_imgname = None
        self.single_imgname_idx = None
        self.sampling_idx = None

        self.intrinsics_all = []
        self.pose_all = []
        self.n_cameras = len(image_paths)
        for i in range(self.n_cameras):
            self.intrinsics_all.append(torch.from_numpy(intrinsics[i]).float())
            self.pose_all.append(torch.from_numpy(poses[i]).float())

        self.rgb_images = []
        self.object_masks = []

        # H, W = rend_util.load_rgb(image_paths[0]).shape[:2]
        self.img_res = [img_hw[0], img_hw[1]]
        self.total_pixels = self.img_res[0] * self.img_res[1]

        # read images
        for path in image_paths:
            rgb = rend_util.load_rgb(path, ds=ds_ratio).reshape(-1, 3)
            self.rgb_images.append(torch.from_numpy(rgb).float())
            self.has_groundtruth = True

        # read mask images
        if split == 'train':
            mask_paths = com_mask_paths[::frame_skip]
        elif split == 'test':
            mask_paths = obj_mask_paths[::frame_skip]
        for path in mask_paths:
            print('Loaded mask: ', path)
            object_mask = rend_util.load_mask(path, ds=ds_ratio)
            object_mask = object_mask.reshape(-1)
            self.object_masks.append(torch.from_numpy(object_mask).bool())

    def __len__(self):
        return (self.n_cameras)

    def __getitem__(self, idx):
        if self.single_imgname_idx is not None:
            idx = self.single_imgname_idx
        
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx],
            "object_mask": self.object_masks[idx],
        }
        ground_truth = {
            "rgb": self.rgb_images[idx]
        }

        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            sample["object_mask"] = self.object_masks[idx][self.sampling_idx]
            sample["uv"] = uv[self.sampling_idx, :]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, 
        # ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]


if __name__ == "__main__":
    dataset = OpenIllumDataset(instance_dir='/home/isabella/Lab/OpenIllumination/data/OpenIllumination/20230524-13_08_01_obj_2_egg',
                         frame_skip=1,
                         split='train',)
    print()