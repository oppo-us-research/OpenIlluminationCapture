#  created by Isabella Liu (lal005@ucsd.edu) at 2023/06/03 21:31.
#  
#  Relight trained TensoIR model under calibrated light SGs.

import os
from tqdm import tqdm
import numpy as np
import imageio
import torch
import torch.nn as nn
from torchvision import transforms as T
import json
from PIL import Image

from opt import config_parser
from dataLoader import dataset_dict
from models.tensoRF_general_multi_lights import TensorVMSplit
from models.tensorBase_general_multi_lights import compute_energy, render_envmap_sg
from models.relight_utils import *
from dataLoader.ray_utils import *
from utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

JSON_PATH = '/mnt/data/lightstage_dataset/20230524-17_28_58_obj_27_pumpkin2/output/transforms_test.json'
ROOT_PATH = '/mnt/data/lightstage_dataset/20230524-17_28_58_obj_27_pumpkin2/output'
SGS_PATH = '/home/isabella/NeurIPS2023/datasets/generate_light_gt_sg/sgs'
RES_PATH = '/home/isabella/NeurIPS2023/Novel_relighting/relighting_001'

# envW = 512

class Environment_Light():
    def __init__(self, hdr_path):
        # transverse the hdr image to get the environment light
        files = os.listdir(hdr_path)
        self.hdr_rgbs = dict()
        self.hdr_pdf = dict()
        
        for file in files:
            if file.endswith(".hdr"):
                self.hdr_path = os.path.join(hdr_path, file)
                light_name = file.split(".")[0]
                light_rgbs = read_hdr(self.hdr_path) * 60.
                light_rgbs = torch.from_numpy(light_rgbs)
                self.hdr_rgbs[light_name] = light_rgbs.to(device)
                # compute the pdf of importance sampling of the environment map
                light_intensity = torch.sum(light_rgbs, dim=2, keepdim=True)
                env_map_h, env_map_w, _ = light_intensity.shape
                h_interval = 1.0 / env_map_h
                sin_theta = torch.sin(torch.linspace(0 + 0.5 * h_interval, np.pi - 0.5 * h_interval, env_map_h))
                pdf = light_intensity * sin_theta.view(-1, 1, 1) 
                pdf = pdf / torch.sum(pdf)
                self.hdr_pdf[light_name] = pdf.to(device)


    def sample_light(self, light_name,bs, num_samples):
        '''
        - Args:
            - light_name: the name of the light
            - bs: batch size
            - num_samples: the number of samples
        - Returns:
            - light_dir: the direction of the light [bs, num_samples, 3]
            - light_rgb: the rgb of the light [bs, num_samples, 3]
            - light_pdf: the pdf of the light [bs, num_samples, 1]
        '''

        # sample the environment light
        environment_map = self.hdr_rgbs[light_name]
        environment_map_pdf:torch.tensor = self.hdr_pdf[light_name].view(-1).expand(bs, -1) # [bs, env_map_h * env_map_w]
        env_map_h, env_map_w, _ = environment_map.shape
        h_interval = 1.0 / env_map_h
        w_interval = 1.0 / env_map_w
        # sample the light direction
        light_dir_idx = torch.multinomial(environment_map_pdf , num_samples, replacement=True) # [bs, num_samples]
        light_u = (light_dir_idx % env_map_w).float() * w_interval + 0.5 * w_interval
        light_v = (light_dir_idx // env_map_w).float() * h_interval + 0.5 * h_interval
        light_dir = torch.stack((torch.cos(2 * np.pi * light_u) * torch.sqrt(1 - light_v ** 2), torch.sin(2 * np.pi * light_u) * torch.sqrt(1 - light_v ** 2), light_v), dim=2) # [bs, num_samples, 3]
        # sample the light rgb
        light_rgb = torch.stack([environment_map[light_dir_idx[:, i] // env_map_w, light_dir_idx[:, i] % env_map_w, :] for i in range(num_samples)], dim=1) # [bs, num_samples, 3]
        light_pdf = environment_map_pdf.gather(1, light_dir_idx).unsqueeze(-1) # [bs, num_samples, 1]

        return light_dir, light_rgb, light_pdf


    def get_light(self, light_name, incident_dir):

        envir_map = self.hdr_rgbs[light_name]
        envir_map = envir_map.permute(2, 0, 1).unsqueeze(0) # [1, 3, H, W]
        phi = torch.arccos(incident_dir[:, 2]).reshape(-1) - 1e-6
        theta = torch.atan2(incident_dir[:, 1], incident_dir[:, 0]).reshape(-1)
        # normalize to [-1, 1]
        query_y = (phi / np.pi) * 2 - 1
        query_x = - theta / np.pi
        grid = torch.stack((query_x, query_y)).permute(1, 0).unsqueeze(0).unsqueeze(0)
        light_rgbs = F.grid_sample(envir_map, grid, align_corners=False).squeeze().permute(1, 0).reshape(-1, 3)
    
        return light_rgbs


def get_item(key='CE3', img_wh=(2656, 3984)):  # TODO downsample
    json_path = JSON_PATH
    root_dir = ROOT_PATH
    with open(json_path, 'r') as f:
        transforms = json.load(f)
    item = transforms['frames'][key]
    
    # read rgbs and poses
    fov = item['camera_angle_x']
    calib_imgw = item['calib_imgw']
    focal = 0.5 * int(calib_imgw) / np.tan(0.5 * fov)  # fov -> focal length
    focal *= img_wh[0] / calib_imgw  # modify focal length to match size self.img_wh
    
    # Get rays
    directions = get_ray_directions(img_wh[1], img_wh[0], [focal, focal])  # [H, W, 3]
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    cam_trans = item["transform_matrix"]
    cam_trans = np.array(cam_trans).reshape(4, 4)
    c2w = torch.FloatTensor(cam_trans)  # [4, 4]
    w2c = torch.linalg.inv(c2w)  # [4, 4]
    
    # Get image
    img_path = os.path.join(root_dir, item['file_path']) + '.png'
    img = Image.open(img_path)
    transforms = T.Compose([
            T.ToTensor(),
        ])
    img = transforms(img)
    img = img.view(4, -1).permute(1, 0)  # [H*W, 4]
    img_rgbs = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # [H*W, 3]
    img_mask = ~(img[:, -1:] == 0)  # [H*W, 1]
    rays_o, rays_d = get_rays(directions, c2w)
    rays = torch.cat([rays_o, rays_d], 1)  # [H*W, 6]
    
    item = {
        'img_wh': img_wh,  # (int, int)
        'rgbs': img_rgbs.view(1, -1, 3),  # [1, H*W, 3]
        'rgbs_mask': img_mask,  # [H*W, 1]
        'rays': rays,  # [H*W, 6]
        'c2w': c2w,  # [4, 4]
        'w2c': w2c,  # [4, 4],
    }
    return item


def load_SGs(sg_path, light_names):
    lgt_sgs = []
    for light in light_names:
        lgt_sg_path = os.path.join(sg_path, f'{light}.npy')
        lgt_sg = torch.from_numpy(np.load(lgt_sg_path)).float().to(device)  # [LIGHT_NUM, 3]
        
        # normalize TODO
        energy = compute_energy(lgt_sg)
        
        lgt_sg[:, 4:] = torch.abs(lgt_sg[:, 4:]) / torch.sum(energy, dim=0, keepdim=True) * 2. * torch.pi * 0.8
        energy = compute_energy(lgt_sg)
        lgt_sgs.append(lgt_sg)
    return lgt_sgs


@torch.no_grad()
def relight(args):
    if not os.path.exists(args.ckpt):
        print('the checkpoint path for tensorfactor does not exists!!')
        return
        
    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorfactor = eval(args.model_name)(**kwargs)
    tensorfactor.load(ckpt)

    if args.ckpt_visibility is not None:
        visibility_net:nn.Module = eval(args.vis_model_name)().to(device)
        visibility_net.load_state_dict(torch.load(args.ckpt_visibility))
        visibility_net.requires_grad_(False) # freeze the visibility net
        print("load visibility network succcessfully")
    else:
        print('Not using visibility network')
    W, H = args.imgw, args.imgh
    near_far = args.near, args.far
    
    # export mesh
    alpha, _ = tensorfactor.getDenseAlpha()
    convert_sdf_samples_to_ply(alpha.cpu(), f'{args.ckpt[:-3]}.ply', bbox=tensorfactor.aabb.cpu(), level=0.005)

    # load light sgs
    light_names = ['001', '009', '013']
    lgt_sgs = load_SGs(SGS_PATH, light_names)  # [LIGHT_NUM, 3]
    
    relight_pred_img = dict()
    for light_name in light_names:
        relight_pred_img[light_name] = []
    
    # Get item
    item = get_item('CE3', img_wh=(W, H))
    frame_rays = item['rays'].squeeze(0).to(device) # [H*W, 6]
    light_idx = torch.zeros((frame_rays.shape[0], 1), dtype=torch.int).to(device).fill_(0)
    
    rgb_map, depth_map, normal_map, albedo_map, roughness_map, fresnel_map, relight_rgb_map, normals_diff_map, normals_orientation_loss_map = [], [], [], [], [], [], [], [], []
    acc_map = []
    albedo_rescale_factor = torch.tensor([1.0, 1.0, 1.0]).to(device) * 1.0
    

    chunk_idxs = torch.split(torch.arange(frame_rays.shape[0]), args.batch_size)
    for chunk_idx in tqdm(chunk_idxs):
        # Query materials
        rgb_chunk, depth_chunk, normal_chunk, albedo_chunk, roughness_chunk, \
                    fresnel_chunk, acc_chunk, *temp \
                    = tensorfactor(frame_rays[chunk_idx], light_idx[chunk_idx], is_train=False, white_bg=True, ndc_ray=False, N_samples=-1)

        # Relighting
        
        relight_rgb_chunk = torch.ones_like(rgb_chunk)
        acc_chunk_mask = (acc_chunk > args.acc_mask_threshold)
        rays_o_chunk, rays_d_chunk = frame_rays[chunk_idx][:, :3], frame_rays[chunk_idx][:, 3:]
        surface_xyz_chunk = rays_o_chunk + depth_chunk.unsqueeze(-1) * rays_d_chunk  # [bs, 3]
        masked_surface_xyz = surface_xyz_chunk[acc_chunk_mask] # [surface_point_num, 3]
        masked_normal_chunk = normal_chunk[acc_chunk_mask] # [surface_point_num, 3]
        masked_albedo_chunk = albedo_chunk[acc_chunk_mask] # [surface_point_num, 3]
        masked_roughness_chunk = roughness_chunk[acc_chunk_mask] # [surface_point_num, 1]
        masked_fresnel_chunk = fresnel_chunk[acc_chunk_mask] # [surface_point_num, 1]
        ## Get incident light direction
        light_area_weight = tensorfactor.light_area_weight.to(device) # [envW * envH, ]
        # incident_light_dirs = tensorfactor.gen_light_incident_dirs(method=args.light_sample_train).to(device)
        incident_light_dirs = tensorfactor.gen_light_incident_dirs(method='fixed_envirmap').to(device)
        surf2l = incident_light_dirs.reshape(1, -1, 3).repeat(masked_surface_xyz.shape[0], 1, 1)  # [bs, envW * envH, 3]
        surf2c = -rays_d_chunk[acc_chunk_mask]  # [bs, 3]
        surf2c = safe_l2_normalize(surf2c, dim=-1)  # [bs, 3]
        
        cosine = torch.einsum("ijk,ik->ij", surf2l, masked_normal_chunk)    # surf2l:[surface_point_num, envW * envH, 3] * masked_normal_chunk:[surface_point_num, 3] -> cosine:[surface_point_num, envW * envH]
        cosine_mask = (cosine > 1e-6)  # [surface_point_num, envW * envH] mask half of the incident light that is behind the surface
        visibility = torch.zeros((*cosine_mask.shape, 1), device=device)    # [surface_point_num, envW * envH, 1]
        masked_surface_xyz = masked_surface_xyz[:, None, :].expand((*cosine_mask.shape, 3))  # [surface_point_num, envW * envH, 3]
        cosine_masked_surface_pts = masked_surface_xyz[cosine_mask] # [num_of_vis_to_get, 3]
        cosine_masked_surf2l = surf2l[cosine_mask] # [num_of_vis_to_get, 3]
        cosine_masked_visibility = torch.zeros(cosine_masked_surf2l.shape[0], 1, device=device) # [num_of_vis_to_get, 1]

        chunk_idxs_vis = torch.split(torch.arange(cosine_masked_surface_pts.shape[0]), 150000)  

        for chunk_vis_idx in chunk_idxs_vis:
            chunk_surface_pts = cosine_masked_surface_pts[chunk_vis_idx]  # [chunk_size, 3]
            chunk_surf2light = cosine_masked_surf2l[chunk_vis_idx]    # [chunk_size, 3]
            if args.if_predict_single_view_visibility:
                cosine_masked_visibility[chunk_vis_idx] = visibility_net(chunk_surface_pts, chunk_surf2light) # [chunk_size, 1]
            else :
                nerv_vis, nerfactor_vis = compute_transmittance(tensoIR=tensorfactor, 
                                                                surf_pts=chunk_surface_pts, 
                                                                light_in_dir=chunk_surf2light, 
                                                                nSample=96, 
                                                                vis_near=0.03,
                                                                vis_far=1.5
                                                                ) # [chunk_size, 1]
                if args.vis_equation == 'nerfactor':
                    cosine_masked_visibility[chunk_vis_idx] = nerfactor_vis.unsqueeze(-1)
                elif args.vis_equation == 'nerv':
                    cosine_masked_visibility[chunk_vis_idx] = nerv_vis.unsqueeze(-1)
            visibility[cosine_mask] = cosine_masked_visibility
        ## Get BRDF specs
        nlights = surf2l.shape[1]
        specular_relighting = brdf_specular(masked_normal_chunk, surf2c, surf2l, masked_roughness_chunk, masked_fresnel_chunk)  # [surface_point_num, envW * envH, 3]
        masked_albedo_chunk_rescaled = masked_albedo_chunk * albedo_rescale_factor
        surface_brdf_relighting = masked_albedo_chunk_rescaled.unsqueeze(1).expand(-1, nlights, -1) / np.pi + specular_relighting # [surface_point_num, envW * envH, 3]
        ## Relighting for each light
        for idx, cur_light_name in enumerate(light_names):
            relight_rgb_chunk.fill_(1.0)
            # Get direct light rgb 
            cur_light_idx = light_names.index(cur_light_name)
            cur_lgt_sg = lgt_sgs[cur_light_idx]
            # Get view dir 
            _, view_dirs = tensorfactor.generate_envir_map_dir(16, 32, is_jittor=False)
            cur_light_rgbs = render_envmap_sg(cur_lgt_sg.to(device), view_dirs).reshape(-1, 3) # [sample_number, 3]
            direct_light_rgbs = cur_light_rgbs
            
            # direct_light_rgbs = envir_light.get_light(cur_light_name, incident_light_dirs)
            light_rgbs = visibility * direct_light_rgbs  # [bs, envW * envH, 3]
            light_pix_contrib = surface_brdf_relighting * light_rgbs * cosine[:, :, None] * light_area_weight[None,:, None]   # [bs, envW * envH, 3]
            surface_relight_rgb_chunk  = torch.sum(light_pix_contrib, dim=1)  # [bs, 3]
            ### Tonemapping
            surface_relight_rgb_chunk = torch.clamp(surface_relight_rgb_chunk, min=0.0, max=1.0)  
            ### Colorspace transform
            if surface_relight_rgb_chunk.shape[0] > 0:
                surface_relight_rgb_chunk = linear2srgb_torch(surface_relight_rgb_chunk)
            relight_rgb_chunk[acc_chunk_mask] = surface_relight_rgb_chunk
            relight_pred_img[cur_light_name].append(relight_rgb_chunk.detach().clone().cpu())
        
        # For original network output
        rgb_map.append(rgb_chunk.cpu().detach())
        depth_map.append(depth_chunk.cpu().detach())
        acc_map.append(acc_chunk.cpu().detach())
        normal_map.append(normal_chunk.cpu().detach())
        albedo_map.append(albedo_chunk.cpu().detach())
        roughness_map.append(roughness_chunk.cpu().detach())
        
    rgb_map = torch.cat(rgb_map, dim=0)
    depth_map = torch.cat(depth_map, dim=0)
    acc_map = torch.cat(acc_map, dim=0)
    normal_map = torch.cat(normal_map, dim=0)
    acc_map_mask = (acc_map > args.acc_mask_threshold)
    albedo_map = torch.cat(albedo_map, dim=0)
    roughness_map = torch.cat(roughness_map, dim=0)
    
    # Save relighting results
    os.makedirs(os.path.join(RES_PATH, 'relighting'), exist_ok=True)
    for light_name_idx, cur_light_name in enumerate(light_names):
        relight_rgb_map = torch.cat(relight_pred_img[cur_light_name], dim=0).reshape(H, W, 3).numpy()
        imageio.imwrite(os.path.join(RES_PATH, 'relighting', f'{cur_light_name}.png'), (relight_rgb_map * 255).astype('uint8'))
    
    # Save original network output
    os.makedirs(os.path.join(RES_PATH, 'original_output'), exist_ok=True)
    albedo_map = albedo_map.reshape(H, W, 3).numpy()
    imageio.imwrite(os.path.join(RES_PATH, 'original_output', f'albedo.png'), (albedo_map * 255).astype('uint8'))
    normal_map = F.normalize(normal_map, dim=-1)
    normal_map = normal_map * 0.5 + 0.5
    normal_map = normal_map.reshape(H, W, 3).numpy()
    imageio.imwrite(os.path.join(RES_PATH, 'original_output', f'normal.png'), (normal_map * 255).astype('uint8'))
    rgb_map = rgb_map.reshape(H, W, 3).numpy()
    imageio.imwrite(os.path.join(RES_PATH, 'original_output', f'rgb.png'), (rgb_map * 255).astype('uint8'))
    
def main():
    args = config_parser()
    print(args)
    print("*" * 80)
    print('The result will be saved in {}'.format(os.path.abspath(args.geo_buffer_path)))
    
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20230101)
    torch.cuda.manual_seed_all(20230101)
    np.random.seed(20230101)

    # The following args are not defined in opt.py
    args.if_save_point_cloud = False
    args.if_save_rgb = False
    args.if_save_depth = False
    args.if_save_acc = True
    args.if_save_rgb_video = False
    args.if_save_relight_rgb = True
    args.if_save_albedo = True
    args.if_save_albedo_gamma_corrected = True
    args.acc_mask_threshold = 0.1
    args.if_predict_single_view_visibility = False # single view, with rotated OLAT, predict visibility from vis-net
    args.if_compute_single_view_visibility = False # single view, with rotated OLAT, compute the transimittance in NeRV as visibility 
    args.if_render_normal = False
    args.vis_equation = 'nerfactor'
    args.if_material_editing = False
    args.batch_size = 24576


    relight(args)


if __name__ == '__main__':
    main()
