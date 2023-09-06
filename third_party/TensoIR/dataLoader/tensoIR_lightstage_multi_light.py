
import os
import os.path as osp
import cv2
import json
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from dataLoader.ray_utils import *
from models.relight_utils import read_hdr
import torch.nn as nn


def generate_camera_poses(center, radius, num_poses, z_shift=0.1):
    # Extract the x, y, and z coordinates of the object center
    x, y, z = center
    z += z_shift

    # Set up an initial camera pose matrix
    T = np.array([[1, 0, 0, x],
                  [0, 1, 0, y],
                  [0, 0, 1, z],
                  [0, 0, 0, 1]])

    # Define the angle of rotation around the z-axis
    theta = 2 * np.pi / num_poses

    # Generate a sequence of camera poses
    camera_poses = []
    for i in range(num_poses):
        # Compute the position of the camera along the trajectory
        pos_x = x + radius * np.cos(i * theta)
        pos_y = y + radius * np.sin(i * theta)
        pos_z = z

        # Set up a translation matrix for the camera position
        T_cam = np.array([[1, 0, 0, pos_x],
                          [0, 1, 0, pos_y],
                          [0, 0, 1, pos_z],
                          [0, 0, 0, 1]])

        # Compute the vector pointing from the camera to the object center
        look_at = center - np.array([pos_x, pos_y, pos_z])
        look_at /= np.linalg.norm(look_at)
        look_at *= -1

        # Compute the cross product of the up vector and the look-at vector to get the right vector
        up = np.array([0, 0, 1])
        right = np.cross(up, look_at)
        right /= np.linalg.norm(right)
        # right *= -1

        # Compute the cross product of the look-at vector and the right vector to get the up vector
        up = np.cross(look_at, right)
        up /= np.linalg.norm(up)
        up *= -1

        # Compute the rotation matrix that aligns the camera with the look-at, right, and up vectors
        R = np.array([[right[0], up[0], -look_at[0], 0],
                      [right[1], up[1], -look_at[1], 0],
                      [right[2], up[2], -look_at[2], 0],
                      [0,        0,        0,          1]])

        # Combine the translation and rotation matrices to get the camera pose matrix
        M = T_cam.dot(R)
        

        # Store the camera pose matrix in the list of camera poses
        camera_poses.append(M)

    return camera_poses


class TensoIR_Dataset_lightstage_multi_light(Dataset):
    def __init__(self,
                 root_dir,
                 hdr_dir,
                 split='train',
                 random_test = True,
                 light_names=[],
                 N_vis=-1,
                 downsample=1.0,
                 sub=0,
                 light_positions_train=['000', '001', '002'],
                 light_positions_test=['002'],
                 light_name="sunset",
                 scene_bbox=[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
                 img_height=900,
                 img_width=1200,
                 near=1.0,
                 far=12.0,
                 bbox_scale=1.0,
                 test_new_pose=False,
                 *temp
                 ):
        
        assert split in ['train', 'test']
        self.N_vis = N_vis
        self.root_dir = Path(root_dir)
        
        transforms_file_path = os.path.join(self.root_dir, f'transforms_{split}.json')
        with open(transforms_file_path, 'r') as f:
            self.transforms_json = json.load(f)
        self.split = split
        light_positions = light_positions_train if split == 'train' else light_positions_test
        self.light_positions = light_positions
        self.light_positions_train = light_positions_train
        self.light_positions_test_idx_in_train = [light_positions_train.index(x) for x in light_positions_test]
        self.light_names = light_names
        self.light_num = len(self.light_positions)
        self.split_list = []
        
        # Frame choosing
        self.chosen_frame_idx = []
        for idx, x in enumerate(self.transforms_json['frames']):
            if self.transforms_json['frames'][x]['light_idx'] < self.light_num:
                
                # For multi light, the folder organization is a bit different
                for light_idx in range(self.light_num):
                    new_light_file_path = f'../Lights/{light_positions[light_idx]}/raw_undistorted/{x}'
                    self.split_list.append(new_light_file_path)
                    self.chosen_frame_idx.append(x)

        if not random_test:
            # sort split_list and chosen_frame_idx according to the file name
            sorted_idx = np.argsort(self.split_list)
            self.split_list = [self.split_list[i] for i in sorted_idx]
            self.chosen_frame_idx = [self.chosen_frame_idx[i] for i in sorted_idx]

        # Only train subset, for debugging
        if sub > 0:
            self.split_list = self.split_list[:sub]

        self.img_wh = (int(int(img_width) / downsample), int(int(img_height) / downsample)) 
        self.white_bg = True
        self.downsample = downsample
        self.transform = self.define_transforms()
        self.light_name = light_name
        self.near_far = [near, far]  
        # scene_bbox = [eval(item) for item in scene_bbox]
        scale = bbox_scale
        scene_bbox=[[-scale, -scale, -scale], [scale, scale, scale]]
        
        self.scene_bbox = torch.tensor(scene_bbox)
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)

    

        ## Load light data
        if hdr_dir != '':
            self.hdr_dir = Path(os.path.join(hdr_dir, 'hdrs'))
            self.sg_dir = Path(os.path.join(hdr_dir, 'sgs'))
            self.read_lights()
            self.read_lights_sgs()
        else:
            self.lights_probes = None

        if split == 'train' and not test_new_pose:
            self.read_stack()
            
        elif split == 'test' and not test_new_pose:
            self.read_stack_test()
            
        if split == 'test' and test_new_pose:
            
            print("=================== test new pose ===================")

            def normalize(x: np.ndarray) -> np.ndarray:
                """Normalization helper function."""
                return x / np.linalg.norm(x)
            
            def average_poses(poses):
                """
                Calculate the average pose, which is then used to center all poses
                using @center_poses. Its computation is as follows:
                1. Compute the center: the average of pose centers.
                2. Compute the z axis: the normalized average z axis.
                3. Compute axis y': the average y axis.
                4. Compute x' = y' cross product z, then normalize it as the x axis.
                5. Compute the y axis: z cross product x.
                
                Note that at step 3, we cannot directly use y' as y axis since it's
                not necessarily orthogonal to z axis. We need to pass from x to y.
                Inputs:
                    poses: (N_images, 3, 4)
                Outputs:
                    pose_avg: (3, 4) the average pose
                """
                # 1. Compute the center
                center = poses[..., 3].mean(0)  # (3)

                # 2. Compute the z axis
                z = normalize(poses[..., 2].mean(0))  # (3)

                # 3. Compute axis y' (no need to normalize as it's not the final output)
                y_ = poses[..., 1].mean(0)  # (3)

                # 4. Compute the x axis
                x = normalize(np.cross(y_, z))  # (3)

                # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
                y = np.cross(z, x)  # (3)

                pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

                return pose_avg
            
            def center_poses(poses):
                """
                Center the poses so that we can use NDC.
                See https://github.com/bmild/nerf/issues/34
                Inputs:
                    poses: (N_images, 3, 4)
                Outputs:
                    poses_centered: (N_images, 3, 4) the centered poses
                    pose_avg: (3, 4) the average pose
                """

                pose_avg = average_poses(poses)  # (3, 4)
                pose_avg_homo = np.eye(4)
                pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
                # by simply adding 0, 0, 0, 1 as the last row
                last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
                poses_homo = np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

                poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
                poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

                return poses_centered, np.linalg.inv(pose_avg_homo)

            self.read_stack()
            
            # Calculate average poses
            poses = self.all_poses.numpy()  # [N, 4, 4]
            
            
            avg_pose = average_poses(poses[:, :3, :])  # [N, 3, 4]
            avg_pose = np.vstack([avg_pose, np.array([0, 0, 0, 1]).reshape(1, 4)])
            
            centered_pose, centered_pose_T = center_poses(poses[:, :3, :])  # [N, 3, 4]
            
            
            # import pdb; pdb.set_trace()
            
            # centroid = poses[:,:3,3].mean(0)
            
            # Get Delta trsansformation from origin to average pose
            
            # poses = self.all_poses.numpy()
            # centroid = poses[:,:3,3].mean(0)
            # centroid = np.array([0.02, 0.05, 0.1])
            # centroid = np.array([0.24104036, 0.21734161, 0.08039788])
            # centroid = np.array([-1.5220, 1.1360, 0.5853])
            centroid = np.array([0.0, 0.0, 0.0])
            
            render_poses = generate_camera_poses(centroid, radius=4, num_poses=80, z_shift=2.0)
            
            render_poses = np.stack(render_poses, axis=0)  # [N, 4, 4]
            
            
            
            # tmp 
            mirror_z = np.array([[-1, 0, 0, 0],
                                 [0, -1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]])
            # render_poses = render_poses @ np.linalg.inv(avg_pose) @ mirror_z
            
            tsfm=np.array([[ 0.2062411 , 0.50268688,-0.83950611, 0.51466776],
                            [ 0.97749609,-0.1447207 , 0.1534839 , 0.79926193],
                            [-0.04433957,-0.85226863,-0.52122182, 2.76169091],
                            [ 0.        , 0.        , 0.        , 1.        ]])
            
            render_poses = np.linalg.inv(tsfm) @ render_poses
            
            
            
          
        


            # # unified foclal
            img_wh = self.img_wh
            frame_idx = self.chosen_frame_idx[0]
            fov = self.transforms_json['frames'][str(frame_idx)]["camera_angle_x"]
            calib_imgw = self.transforms_json['frames'][str(frame_idx)]["calib_imgw"]
            
            focal = 0.5 * int(calib_imgw) / np.tan(0.5 * fov)  # fov -> focal length
            focal *= self.img_wh[0] / calib_imgw  # modify focal length to match size self.img_wh

            # directions = get_ray_directions_blender(img_wh[1], img_wh[0], [focal, focal])  # [H, W, 3]
            directions = get_ray_directions(img_wh[1], img_wh[0], [focal, focal])  # [H, W, 3]
            directions = directions / torch.norm(directions, dim=-1, keepdim=True)


            self.test_rays = []
            self.test_w2c = []
            
            for pose_idx in tqdm(range(render_poses.shape[0])):
                pose = render_poses[pose_idx]
                pose = torch.from_numpy(pose).float()
                # import pdb; pdb.set_trace()
                # c2w = torch.cat([pose, torch.tensor([[0, 0, 0, 1]])], dim=0)
                # c2w = torch.from_numpy(pose)
                c2w = pose

                # import ipdb; ipdb.set_trace()
                rays_o, rays_d = get_rays(directions, c2w)
                rays = torch.cat([rays_o, rays_d], 1)  # [H*W, 6]
                self.test_rays.append(rays)
                # w2c = torch.inverse(c2w)
                w2c = c2w
                
                self.test_w2c.append(w2c)
                
    
            self.test_rays = torch.stack(self.test_rays, dim=0)
            self.test_w2c = torch.stack(self.test_w2c, dim=0)
            
        
            del self.all_rays, self.all_rgbs, self.all_light_idx, self.all_masks, self.all_poses
    

    def define_transforms(self):
        transforms = T.Compose([
            T.ToTensor(),
        ])
        return transforms

    def read_lights(self):
        """
        Read hdr file from local path
        """
        self.lights_probes = []
        for light_name in self.light_positions_train:
            hdr_path = self.hdr_dir / f'{light_name}.hdr'
            if os.path.exists(hdr_path):
                light_rgb = read_hdr(hdr_path)
                self.envir_map_h, self.envir_map_w = light_rgb.shape[:2]
                light_rgb = light_rgb.reshape(-1, 3)
                light_rgb = torch.from_numpy(light_rgb).float()
                self.lights_probes.append(light_rgb)
        
    def read_lights_sgs(self):
        self.lgtSGs_list = []
        for light_name in self.light_positions_train:
            sg_path = self.sg_dir / f'{light_name}.npy'
            if os.path.exists(sg_path):
                lgt_sgs = np.load(sg_path)
                lgt_sgs = torch.from_numpy(lgt_sgs).float()
                self.lgtSGs_list.append(lgt_sgs)

    def world2ndc(self, points, lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)

    def read_stack(self):
        self.all_rays = []
        self.all_rgbs = []
        self.all_light_idx = []
        self.all_masks = []
        self.all_poses = []
        self.all_w2c = []
        for idx in tqdm(range(self.__len__())):
            item = self.__getitem__(idx)
            rays = item['rays']
            rgbs = item['rgbs']
            light_idx = item['light_idx']
            self.all_rays += [rays]
            self.all_rgbs += [rgbs.squeeze(0)]
            self.all_light_idx += [light_idx.squeeze(0)]
            self.all_masks += [item['rgbs_mask'].squeeze(0)]
            self.all_poses += [item['c2w'].squeeze(0)]
            self.all_w2c += [item['w2c'].squeeze(0)]

        self.all_rays = torch.cat(self.all_rays, dim=0)  # [N*H*W, 6]
        self.all_rgbs = torch.cat(self.all_rgbs, dim=0)  # [N*H*W, 3]
        self.all_light_idx = torch.cat(self.all_light_idx, dim=0)  # [N*H*W, 1]
        self.all_masks = torch.cat(self.all_masks, dim=0)  # [N*H*W, 1]
        self.all_poses = torch.stack(self.all_poses, dim=0)  # [N, 4, 4]
        self.all_w2c = torch.stack(self.all_w2c, dim=0)  # [N, 4, 4]
    
    def read_stack_test(self):
        self.all_rays = []
        self.all_rgbs = []
        self.all_light_idx = []
        self.all_masks = []
        self.all_poses = []
        self.all_w2c = []
        for idx in tqdm(range(self.__len__())):
            item = self.__getitem__(idx)
            rays = item['rays']
            rgbs = item['rgbs']
            light_idx = item['light_idx']
            self.all_rays += [rays]
            self.all_rgbs += [rgbs.squeeze(0)]
            self.all_light_idx += [light_idx.squeeze(0)]
            self.all_masks += [item['rgbs_mask'].squeeze(0)]
            self.all_poses += [item['c2w'].squeeze(0)]
            self.all_w2c += [item['w2c'].squeeze(0)]

        self.all_rays = torch.stack(self.all_rays, dim=0)  # [N*H*W, 6]
        self.all_rgbs = torch.cat(self.all_rgbs, dim=0)  # [N*H*W, 3]
        self.all_light_idx = torch.cat(self.all_light_idx, dim=0)  # [N*H*W, 1]
        self.all_masks = torch.cat(self.all_masks, dim=0)  # [N*H*W, 1]
        self.all_poses = torch.stack(self.all_poses, dim=0)  # [N, 4, 4]
        self.all_w2c = torch.stack(self.all_w2c, dim=0)  # [N, 4, 4]
        
    def __len__(self):
        return len(self.split_list)

    def __getitem__(self, idx):
        # if self.split == 'train':
        item_path = self.split_list[idx]
        if item_path.startswith('./'):
            item_path = item_path[2:]
        frame_idx = self.chosen_frame_idx[idx]
        img_wh = self.img_wh
        # Get ray directions for all pixels, same for all images (with same H, W, focal)
        fov = self.transforms_json['frames'][str(frame_idx)]["camera_angle_x"]
        calib_imgw = self.transforms_json['frames'][str(frame_idx)]["calib_imgw"]

        
        focal = 0.5 * int(calib_imgw) / np.tan(0.5 * fov)  # fov -> focal length
        focal *= img_wh[0] / calib_imgw  # modify focal length to match size self.img_wh
        
        directions = get_ray_directions(img_wh[1], img_wh[0], [focal, focal])  # [H, W, 3]
        # directions = get_ray_directions(img_wh[1], img_wh[0], [focal, focal])  # [H, W, 3]
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)

        cam_trans = self.transforms_json['frames'][str(frame_idx)]["transform_matrix"]
        cam_trans = np.array(cam_trans).reshape(4, 4)
        c2w = torch.FloatTensor(cam_trans)  # [4, 4]
        w2c = torch.linalg.inv(c2w)  # [4, 4]


        # light_idx = self.transforms_json['frames'][str(frame_idx)]["light_idx"]
        # TODO
        if self.split == 'train':
            light_idx = self.light_positions.index(item_path.split('/')[-3])
        elif self.split == 'test':
            light_idx = self.light_positions_train.index(item_path.split('/')[-3])
            
        light_idx = torch.tensor(light_idx, dtype=torch.int).repeat((img_wh[0] * img_wh[1], 1)) # [H*W, 1]
        
        
        # For JPG format images
        # Read image with background
        img_path = os.path.join(self.root_dir, item_path) + '.JPG'
        img = Image.open(img_path)
        if self.downsample!=1.0:
            img = img.resize(self.img_wh, Image.LANCZOS)
        img = self.transform(img)  # [3, H, W]
        img = img.view(3, -1).permute(1, 0)  # [H*W, 3]
        
        # img_orig_vis = img.numpy().reshape(img_wh[1], img_wh[0], 3)
        
        img_rgbs = img
        
        # Read mask
        if self.split == 'train':
            img_mask_path = os.path.join(self.root_dir, 'com_masks', f'{frame_idx}.png')
        else:
            img_mask_path = os.path.join(self.root_dir, 'obj_masks', f'{frame_idx}.png')
        
            
        img_mask = cv2.imread(img_mask_path, cv2.IMREAD_UNCHANGED) > 0 # [H, W]
        img_mask = img_mask.astype(np.uint8)
        if self.downsample!=1.0:
            img_mask = cv2.resize(img_mask, self.img_wh, interpolation=cv2.INTER_NEAREST)
        img_mask = img_mask.astype(dtype=bool).reshape(-1, 1)
        img_mask = torch.from_numpy(img_mask)
        # img_mask = ~(torch.all(img_rgbs == 0, dim=-1, keepdim=True))  # [H*W, 1], NOT WORKING!!
        img_rgbs[~img_mask[..., 0]] = 1.0  # [H*W, 3]  # NOTE background color is white
        
        # img_vis = img_rgbs.numpy().reshape(img_wh[1], img_wh[0], 3)
        # img_mask_vis = img_mask.numpy().reshape(img_wh[1], img_wh[0], 1)
        # img_orig_vis = img.numpy().reshape(img_wh[1], img_wh[0], 3)
        
        rays_o, rays_d = get_rays(directions, c2w)
        rays = torch.cat([rays_o, rays_d], 1)  # [H*W, 6]
        
        item = {
            'img_wh': img_wh,  # (int, int)
            'light_idx': light_idx.view(1, -1, 1),  # [1, H*W, 1]
            'rgbs': img_rgbs.view(1, -1, 3),  # [1, H*W, 3]
            'rgbs_mask': img_mask,  # [H*W, 1]
            'rays': rays,  # [H*W, 6]
            'c2w': c2w,  # [4, 4]
            'w2c': w2c,  # [4, 4],
            'id': str(frame_idx)
        }
        return item
        
            

if __name__ == "__main__":
    from opt import config_parser

    args = config_parser()

    dataset = TensoIR_Dataset_lightstage_multi_light(
        root_dir='/mnt/data/lightstage_dataset/20230524-17_27_05_obj_26_pumpkin/output',
        hdr_dir='/home/isabella/NeurIPS2023/datasets/generate_light_gt_sg',
        split='test',
        downsample=4.0,
        light_positions_train=['001', '003', '008'],
        light_positions_test=['008'],
        img_height=3984,
        img_width=2656,
        near=0.1,
        far=2.0,
        bbox_scale=0.2
    )
    item = dataset[0]
        