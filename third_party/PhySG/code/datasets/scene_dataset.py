import os
import torch
import numpy as np

import utils.general as utils
from utils import rend_util
import json

rotation_matrix = np.array([
        [1, 0, 0, 0],
        [ 0, -1, 0, 0],
        [ 0, 0,-1, 0],
        [ 0, 0, 0, 1]
    ])

def read_cam_dict(cam_dict_file):
    with open(cam_dict_file) as fp:
        cam_dict = json.load(fp)
        intrinsics_all = []
        pose_all = []
        image_paths = []
        mask_paths = []
        for x in sorted(cam_dict.keys()):
            K = np.array(cam_dict[x]['K']).reshape((4, 4)) * 2
            W2C = np.array(cam_dict[x]['W2C']).reshape((4, 4))
            C2W = np.linalg.inv(W2C)

            intrinsics = K.astype(np.float32)
            intrinsics_all.append(torch.from_numpy(intrinsics).float())
            pose = C2W.astype(np.float32)
            pose_all.append(torch.from_numpy(pose).float())
            image_paths.append(os.path.join(x+".JPG"))
            mask_paths.append(os.path.join(x+".png"))


    return intrinsics_all,pose_all,image_paths,mask_paths


class SceneDataset(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self,
                 gamma,
                 instance_dir,
                 illumination_idx,
                 output_img_size,
                 train_cameras,
                 is_test=False
                 ):
        self.instance_dir = os.path.join(instance_dir) 
        print('Creating dataset from: ', self.instance_dir)
        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.gamma = gamma
        self.train_cameras = train_cameras

        camera_json = 'transforms_train.json' if not is_test else 'transforms_test.json'        
        self.intrinsics_all,self.pose_all,image_paths,mask_paths = read_cam_dict(os.path.join(self.instance_dir,"output", camera_json))

        print('Found # images, # masks', len(image_paths), len(mask_paths))
        self.n_cameras = len(image_paths)
        self.image_paths = image_paths

        self.single_imgname = None
        self.single_imgname_idx = None
        self.sampling_idx = None
        if len(mask_paths) > 0:
            assert (len(mask_paths) == self.n_cameras)
            self.object_masks = []
            self.object_mask_float = []
            for path in mask_paths:
                path = os.path.join(self.instance_dir,"output","com_masks",path)

                object_mask = rend_util.load_mask(path, output_img_size)
                object_mask_float = rend_util.load_mask_float(path, output_img_size)
                print('Loaded mask: ', path)
                object_mask = object_mask.reshape(-1)
                object_mask_float = object_mask_float.reshape(-1)
                self.object_masks.append(torch.from_numpy(object_mask).bool())
                self.object_mask_float.append(torch.from_numpy(object_mask_float).float())
        else:
            self.object_masks = [torch.ones((self.total_pixels, )).bool(), ] * self.n_cameras
        if len(image_paths) > 0:
            assert (len(image_paths) == self.n_cameras)
            self.has_groundtruth = True
            self.rgb_images = []
            print('Applying inverse gamma correction: ', self.gamma)
            for path in image_paths:
                path = os.path.join(self.instance_dir,"Lights",illumination_idx,"raw_undistorted", path)
                rgb = rend_util.load_rgb(path, output_img_size)
                rgb = np.power(rgb, self.gamma)

                H, W = rgb.shape[1:3]
                self.img_res = [H, W]
                self.total_pixels = self.img_res[0] * self.img_res[1]

                rgb = rgb.reshape(3, -1).transpose(1, 0)
                self.rgb_images.append(torch.from_numpy(rgb).float())

        def apply_mask_to_rgb(rgb, mask):
            expanded_mask = mask.unsqueeze(-1).expand_as(rgb)
            masked_img = rgb * expanded_mask + (1 - expanded_mask)
            return masked_img
        for idx, (image, mask) in enumerate(zip(self.rgb_images, self.object_mask_float)):
            self.rgb_images[idx] = apply_mask_to_rgb(image, mask)
        

    def __len__(self):
        return self.n_cameras

    def return_single_img(self, img_name):
        self.single_imgname = img_name
        for idx in range(len(self.image_paths)):
            if os.path.basename(self.image_paths[idx]) == self.single_imgname:
                self.single_imgname_idx = idx
                break
        print('Always return: ', self.single_imgname, self.single_imgname_idx)

    def __getitem__(self, idx):
        if self.single_imgname_idx is not None:
            idx = self.single_imgname_idx

        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "object_mask": self.object_masks[idx],
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
        }

        ground_truth = {
            "rgb": self.rgb_images[idx]
        }

        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            sample["object_mask"] = self.object_masks[idx][self.sampling_idx]
            sample["uv"] = uv[self.sampling_idx, :]

        if not self.train_cameras:
            sample["pose"] = self.pose_all[idx]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
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

    def change_sampling_idx_patch(self, N_patch, r_patch=1):
        '''
        :param N_patch: number of patches to be sampled
        :param r_patch: patch size will be (2*r_patch)*(2*r_patch)
        :return:
        '''
        if N_patch == -1:
            self.sampling_idx = None
        else:
            # offsets to center pixels
            H, W = self.img_res
            u, v = np.meshgrid(np.arange(-r_patch, r_patch),
                               np.arange(-r_patch, r_patch))
            u = u.reshape(-1)
            v = v.reshape(-1)
            offsets = v * W + u
            # center pixel coordinates
            u, v = np.meshgrid(np.arange(r_patch, W - r_patch),
                               np.arange(r_patch, H - r_patch))
            u = u.reshape(-1)
            v = v.reshape(-1)
            select_inds = np.random.choice(u.shape[0], size=(N_patch,), replace=False)
            # convert back to original image
            select_inds = v[select_inds] * W + u[select_inds]
            # pick patches
            select_inds = np.stack([select_inds + shift for shift in offsets], axis=1)
            select_inds = select_inds.reshape(-1)
            self.sampling_idx = torch.from_numpy(select_inds).long()

    def get_pose_init(self):
        init_pose = torch.cat([pose.clone().float().unsqueeze(0) for pose in self.pose_all], 0).cuda()
        init_quat = rend_util.rot_to_quat(init_pose[:, :3, :3])
        init_quat = torch.cat([init_quat, init_pose[:, :3, 3]], 1)

        return init_quat
