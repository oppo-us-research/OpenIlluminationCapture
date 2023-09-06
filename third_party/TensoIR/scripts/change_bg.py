'''
Author: Haian-Jin 3190106083@zju.edu.cn
Date: 2022-08-03 20:37:43
LastEditors: Haian-Jin 3190106083@zju.edu.cn
LastEditTime: 2022-08-27 00:10:21
FilePath: /TensoRFactor/scripts/see_visibilty.py
Description: see the visibilty of the model and export a video to visualize the visibilty
'''


import os
from tqdm import tqdm
import imageio
import numpy as np

from opt import config_parser
from models.shapeBuffer import ShapeModel
import torch
from datetime import datetime
import torch.nn as nn
from utils import visualize_depth_numpy
from models.tensoRF_rotated_lights import raw2alpha, TensorVMSplit, AlphaGridMask
from dataLoader.ray_utils import safe_l2_normalize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from dataLoader import dataset_dict
from models.relight_utils import *
brdf_specular = specular_pipeline_render_multilight_new
from utils import rgb_ssim, rgb_lpips


class Environment_Light():
    def __init__(self, hdr_path):
        # transverse the hdr image to get the environment light
        files = os.listdir(hdr_path)
        self.hdr_rgbs = dict()
        self.hdr_pdf_sample = dict()
        self.hdr_pdf_return = dict()
        self.hdr_dir = dict()
        for file in files:
            if file.endswith(".hdr"):
                self.hdr_path = os.path.join(hdr_path, file)
                light_name = file.split(".")[0]
                light_rgbs = read_hdr(self.hdr_path)
                light_rgbs = torch.from_numpy(light_rgbs)
                self.hdr_rgbs[light_name] = light_rgbs.to(device)
                # compute the pdf of importance sampling of the environment map
                light_intensity = torch.sum(light_rgbs, dim=2, keepdim=True) # [H, W, 1]
                env_map_h, env_map_w, _ = light_intensity.shape
                h_interval = 1.0 / env_map_h
                sin_theta = torch.sin(torch.linspace(0 + 0.5 * h_interval, np.pi - 0.5 * h_interval, env_map_h))
                pdf = light_intensity * sin_theta.view(-1, 1, 1) # [H, W, 1]
                pdf = pdf / torch.sum(pdf)
                pdf_return = pdf * env_map_h * env_map_w / (2 * np.pi * np.pi * sin_theta.view(-1, 1, 1)) 
                self.hdr_pdf_sample[light_name] = pdf.to(device)
                self.hdr_pdf_return[light_name] = pdf_return.to(device)

                lat_step_size = np.pi / env_map_h
                lng_step_size = 2 * np.pi / env_map_w
                phi, theta = torch.meshgrid([torch.linspace(np.pi / 2 - 0.5 * lat_step_size, -np.pi / 2 + 0.5 * lat_step_size, env_map_h), 
                                    torch.linspace(np.pi - 0.5 * lng_step_size, -np.pi + 0.5 * lng_step_size, env_map_w)], indexing='ij')


                view_dirs = torch.stack([  torch.cos(theta) * torch.cos(phi), 
                                        torch.sin(theta) * torch.cos(phi), 
                                        torch.sin(phi)], dim=-1).view(env_map_h, env_map_w, 3)    # [envH, envW, 3]
                self.hdr_dir[light_name] = view_dirs.to(device)

    @torch.no_grad()
    def sample_light(self, light_name, bs, num_samples):
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
        environment_map_pdf_sample = self.hdr_pdf_sample[light_name].view(-1).expand(bs, -1) # [bs, env_map_h * env_map_w]
        environment_map_pdf_return = self.hdr_pdf_return[light_name].view(-1).expand(bs, -1) # [bs, env_map_h * env_map_w]
        environment_map_dir = self.hdr_dir[light_name].view(-1, 3).expand(bs, -1, -1) # [bs, env_map_h * env_map_w, 3]
        environment_map_rgb = environment_map.view(-1, 3).expand(bs, -1, -1) # [bs, env_map_h * env_map_w, 3]

        # sample the light direction
        light_dir_idx = torch.multinomial(environment_map_pdf_sample, num_samples, replacement=True) # [bs, num_samples]
        light_dir = environment_map_dir.gather(1, light_dir_idx.unsqueeze(-1).expand(-1, -1, 3)).view(bs, num_samples, 3) # [bs, num_samples, 3]
        # sample the light rgbs
        light_rgb = environment_map_rgb.gather(1, light_dir_idx.unsqueeze(-1).expand(-1, -1, 3)).view(bs, num_samples, 3) # [bs, num_samples, 3]
        light_pdf = environment_map_pdf_return.gather(1, light_dir_idx).unsqueeze(-1) # [bs, num_samples, 1]

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


@torch.no_grad()
def compute_rescale_ratio(tensoIR, dataset, sampled_num=20):
    '''compute three channel rescale ratio for albedo by sampling some views
    - Args:
        tensoIR: model
        dataset: dataset containing the G.T albedo
    - Returns:
        rescale_ratio: [3]
    '''
    W, H = dataset.img_wh
    data_num = len(dataset)
    interval = data_num // sampled_num
    idx_list = [i * interval for i in range(sampled_num)]
    ratio_list = list()
    gt_albedo_list = []
    reconstructed_albedo_list = []
    for idx in tqdm(idx_list, desc="compute rescale ratio"):
        item = dataset[idx]
        frame_rays = item['rays'].squeeze(0).to(device) # [H*W, 6]
        gt_mask = item['rgbs_mask'].squeeze(0).squeeze(-1).cpu() # [H*W]
        gt_albedo = item['albedo'].squeeze(0).to(device) # [H*W, 3]
        light_idx = torch.zeros((frame_rays.shape[0], 1), dtype=torch.int).to(device).fill_(0)
        albedo_map = list()
        chunk_idxs = torch.split(torch.arange(frame_rays.shape[0]), 4096) 
        for chunk_idx in chunk_idxs:
            with torch.enable_grad():
                rgb_chunk, depth_chunk, normal_chunk, albedo_chunk, roughness_chunk, \
                    fresnel_chunk, acc_chunk, *temp \
                    = tensoIR(frame_rays[chunk_idx], light_idx[chunk_idx], is_train=False, white_bg=True, ndc_ray=False, N_samples=-1)
            albedo_map.append(albedo_chunk.detach())
        albedo_map = torch.cat(albedo_map, dim=0).reshape(H, W, 3)
        gt_albedo = gt_albedo.reshape(H, W, 3)
        gt_mask = gt_mask.reshape(H, W)
        # ratio_value = torch.sum(albedo_map[gt_mask] * gt_albedo[gt_mask], dim=0) / torch.sum(albedo_map[gt_mask] * albedo_map[gt_mask], dim=0)
        # ratio_value, _ = (gt_albedo[gt_mask]/ albedo_map[gt_mask].clamp(min=1e-6)).median(dim=0)
        # ratio_list.append(ratio_value.detach())
        gt_albedo_list.append(gt_albedo[gt_mask])
        reconstructed_albedo_list.append(albedo_map[gt_mask])
    # ratio = torch.stack(ratio_list, dim=0).mean(dim=0)
    gt_albedo_all = torch.cat(gt_albedo_list, dim=0)
    albedo_map_all = torch.cat(reconstructed_albedo_list, dim=0)
    ratio , _ = (gt_albedo_all/ albedo_map_all.clamp(min=1e-6)).median(dim=0)
    print("rescale ratio: ", ratio)
    return ratio

@torch.no_grad()
def relight(dataset, args):

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
    W, H = dataset.img_wh
    near_far = dataset.near_far
    light_area_weight = tensorfactor.light_area_weight.to(device)  # [envH * envW, ]
    
    rgb_frames_list = []
    gt_normal_list = []
    optimized_normal_list = []
    material_editing_list = []
    aligned_albedo_list = []
    roughness_list = []

    envir_light = Environment_Light(args.hdrdir)

    #### 
    light_rotation_idx = 0
    ####

    # rescale_value = compute_rescale_ratio(tensorfactor, dataset)
    # rescale_value = torch.tensor([1, 1, 1], device=device)
    rescale_value = torch.tensor([2.7391, 2.1476, 1.5051], device='cuda:0')
    relight_psnr, relight_pred_img, relight_gt_img = dict(), dict(), dict()
    relight_l_alex, relight_l_vgg, relight_ssim = dict(), dict(), dict() 
    for cur_light_name in dataset.light_names:
        relight_psnr[f'{cur_light_name}'] = []
        relight_pred_img[f'{cur_light_name}'] = []
        relight_gt_img[f'{cur_light_name}'] = []
        relight_l_alex[f'{cur_light_name}'] = []
        relight_l_vgg[f'{cur_light_name}'] = []
        relight_ssim[f'{cur_light_name}'] = []

    test_idx = [i * 20 for i in range(10)]
    for idx in test_idx:
    # for idx in tqdm(range(len(dataset))):
    # for idx in tqdm(range(1)):
        relight_pred_img, relight_gt_img = dict(), dict()
        for cur_light_name in dataset.light_names:
            relight_pred_img[f'{cur_light_name}'] = []
            relight_gt_img[f'{cur_light_name}'] = []

        cur_dir_path = os.path.join(args.geo_buffer_path, f'{dataset.split}_{idx:0>3d}')
        os.makedirs(cur_dir_path, exist_ok=True)
        item = dataset[idx]
        frame_rays = item['rays'].squeeze(0).to(device) # [H*W, 6]
        gt_normal = item['normals'].squeeze(0).cpu() # [H*W, 3]
        gt_mask = item['rgbs_mask'].squeeze(0).squeeze(-1).cpu() # [H*W]
        gt_rgb = item['rgbs'].squeeze(0).reshape(len(light_name_list), H, W, 3).cpu()  # [N, H, W, 3]

        gt_albedo = item['albedo'].squeeze(0).to(device) # [H*W, 3]
        light_idx = torch.zeros((frame_rays.shape[0], 1), dtype=torch.int).to(device).fill_(light_rotation_idx)

        rgb_map, depth_map, normal_map, albedo_map, roughness_map, fresnel_map, relight_rgb_map, normals_diff_map, normals_orientation_loss_map = [], [], [], [], [], [], [], [], []
        acc_map = []


        chunk_idxs = torch.split(torch.arange(frame_rays.shape[0]), args.batch_size) # choose the first light idx
        for chunk_idx in chunk_idxs:
            with torch.enable_grad():
                rgb_chunk, depth_chunk, normal_chunk, albedo_chunk, roughness_chunk, \
                    fresnel_chunk, acc_chunk, *temp \
                    = tensorfactor(frame_rays[chunk_idx], light_idx[chunk_idx], is_train=False, white_bg=True, ndc_ray=False, N_samples=-1)

            # # use gt normal to test
            # normal_chunk = gt_normal[chunk_idx].to(device)

            relight_rgb_chunk = torch.ones_like(rgb_chunk)
            material_editing_rgb_chunk = torch.ones_like(rgb_chunk) 
            # albedo_chunk = gt_albedo[chunk_idx] # use GT to debug
            acc_chunk_mask = (acc_chunk > args.acc_mask_threshold)
            rays_o_chunk, rays_d_chunk = frame_rays[chunk_idx][:, :3], frame_rays[chunk_idx][:, 3:]
            surface_xyz_chunk = rays_o_chunk + depth_chunk.unsqueeze(-1) * rays_d_chunk  # [bs, 3]
            masked_surface_pts = surface_xyz_chunk[acc_chunk_mask] # [surface_point_num, 3]
            
            masked_normal_chunk = normal_chunk[acc_chunk_mask] # [surface_point_num, 3]
            masked_albedo_chunk = albedo_chunk[acc_chunk_mask] # [surface_point_num, 3]
            masked_roughness_chunk = roughness_chunk[acc_chunk_mask] # [surface_point_num, 1]
            masked_fresnel_chunk = fresnel_chunk[acc_chunk_mask] # [surface_point_num, 1]
            masked_light_idx_chunk = light_idx[chunk_idx][acc_chunk_mask] # [surface_point_num, 1]

            ## Get incident light direction
            light_area_weight = tensorfactor.light_area_weight.to(device) # [envW * envH, ]

            for idx, cur_light_name in enumerate(dataset.light_names):
                if os.path.exists(os.path.join(cur_dir_path, 'relighting', f'{cur_light_name}.png')):
                    continue
                relight_rgb_chunk.fill_(1.0)
                masked_light_dir, masked_light_rgb, masked_light_pdf = envir_light.sample_light(cur_light_name, masked_normal_chunk.shape[0], 512) # [bs, envW * envH, 3]
                surf2l = masked_light_dir # [surface_point_num, envW * envH, 3]
                surf2c = -rays_d_chunk[acc_chunk_mask]  # [surface_point_num, 3]
                surf2c = safe_l2_normalize(surf2c, dim=-1)  # [surface_point_num, 3]


                ## get visibilty map from visibility network or compute it using density
                cosine = torch.einsum("ijk,ik->ij", surf2l, masked_normal_chunk)    # surf2l:[surface_point_num, envW * envH, 3] * masked_normal_chunk:[surface_point_num, 3] -> cosine:[surface_point_num, envW * envH]
                cosine_mask = (cosine > 1e-6)  # [surface_point_num, envW * envH] mask half of the incident light that is behind the surface
                visibility = torch.zeros((*cosine_mask.shape, 1), device=device)    # [surface_point_num, envW * envH, 1]
                try:
                    masked_surface_xyz = masked_surface_pts[:, None, :].expand((*cosine_mask.shape, 3))  # [surface_point_num, envW * envH, 3]
                except:
                    import ipdb; ipdb.set_trace()
                cosine_masked_surface_pts = masked_surface_xyz[cosine_mask] # [num_of_vis_to_get, 3]
                cosine_masked_surf2l = surf2l[cosine_mask] # [num_of_vis_to_get, 3]
                cosine_masked_visibility = torch.zeros(cosine_masked_surf2l.shape[0], 1, device=device) # [num_of_vis_to_get, 1]

                chunk_idxs_vis = torch.split(torch.arange(cosine_masked_surface_pts.shape[0]), 100000)  

                for chunk_vis_idx in chunk_idxs_vis:
                    chunk_surface_pts = cosine_masked_surface_pts[chunk_vis_idx]  # [chunk_size, 3]
                    chunk_surf2light = cosine_masked_surf2l[chunk_vis_idx]    # [chunk_size, 3]
                    if args.if_predict_single_view_visibility:
                        cosine_masked_visibility[chunk_vis_idx] = visibility_net(chunk_surface_pts, chunk_surf2light) # [chunk_size, 1]
                    else :
                        nerv_vis, nerfactor_vis = compute_transmittance(tensorfactor=tensorfactor, 
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
                

                # relighting
                specular_relighting = brdf_specular(masked_normal_chunk, surf2c, surf2l, masked_roughness_chunk, masked_fresnel_chunk)  # [surface_point_num, envW * envH, 3]
                masked_albedo_chunk_rescaled = masked_albedo_chunk * rescale_value
                surface_brdf_relighting = masked_albedo_chunk_rescaled.unsqueeze(1).expand(-1, nlights, -1) / np.pi + specular_relighting # [surface_point_num, envW * envH, 3]
                direct_light = masked_light_rgb
                light_rgbs = visibility * direct_light  # [bs, envW * envH, 3]

                light_pix_contrib = surface_brdf_relighting * light_rgbs * cosine[:, :, None] / masked_light_pdf
                surface_relight_rgb_chunk  = torch.mean(light_pix_contrib, dim=1)  # [bs, 3]

                ### Tonemapping
                surface_relight_rgb_chunk = torch.clamp(surface_relight_rgb_chunk, min=0.0, max=1.0)  
                ### Colorspace transform
                if surface_relight_rgb_chunk.shape[0] > 0:
                    surface_relight_rgb_chunk = linear2srgb_torch(surface_relight_rgb_chunk)
                relight_rgb_chunk[acc_chunk_mask] = surface_relight_rgb_chunk

                envir_map_color = envir_light.get_light(cur_light_name, rays_d_chunk) # [bs, 3]
                envir_map_color = torch.clamp(envir_map_color, min=0.0, max=1.0)
                envir_map_color = linear2srgb_torch(envir_map_color)
                final_color = relight_rgb_chunk * acc_chunk[..., None] + envir_map_color * (1.0 - acc_chunk[..., None])


                relight_pred_img[cur_light_name].append(final_color.detach().clone().cpu())

            # # material editing
            # if idx < 50:
            #     masked_fresnel_chunk = torch.ones_like(masked_fresnel_chunk) * torch.tensor([0.3, 0.3, 0.3]).cuda()
            #     masked_albedo_chunk = masked_albedo_chunk * torch.tensor([0.6, 0.6, 0.6]).cuda()
            #     # masked_albedo_chunk = torch.clamp(torch.tensor([0.75, 0.45, 0.35]).to(device) - masked_albedo_chunk, 0, 1)
            #     masked_roughness_chunk = 0.3 * torch.ones_like(masked_roughness_chunk)

            # elif idx < 100:
            #     masked_fresnel_chunk = torch.ones_like(masked_fresnel_chunk) * 1
            #     masked_albedo_chunk = 0 * masked_albedo_chunk
            #     masked_roughness_chunk = 0.3 * torch.ones_like(masked_roughness_chunk)

            # elif idx < 150:
            #     masked_fresnel_chunk = masked_fresnel_chunk * 3
            #     masked_albedo_chunk = torch.clamp(torch.tensor([0.80, 0.50, 0.40]).to(device) - masked_albedo_chunk, 0, 1)
            #     # masked_roughness_chunk = 0.2 * torch.ones_like(masked_roughness_chunk)
            # else:
            #     masked_fresnel_chunk = torch.ones_like(masked_fresnel_chunk) * torch.tensor([0.4, 0.6, 0.4]).cuda()
            #     masked_albedo_chunk = 0 * masked_albedo_chunk
            #     masked_roughness_chunk = 0.2 * torch.ones_like(masked_roughness_chunk)

            # specular_material_editing = brdf_specular(masked_normal_chunk, surf2c, surf2l, masked_roughness_chunk, masked_fresnel_chunk)  # [surface_point_num, envW * envH, 3]
            # surface_brdf_material_editing = masked_albedo_chunk.unsqueeze(1).expand(-1, nlights, -1) / np.pi + specular_material_editing # [surface_point_num, envW * envH, 3]


            # ## Material editing
            # rotated_light_rgbs = tensorfactor.get_light_rgbs(incident_light_dirs, device=device).to(device) # [rotation_num, envW * envH, 3]
            # direct_light_rgbs = torch.index_select(rotated_light_rgbs, dim=0, index=masked_light_idx_chunk.squeeze(-1)).to(device) # [bs, envW * envH, 3]
            # light_rgbs = visibility * direct_light_rgbs  # [bs, envW * envH, 3]

            
            
            # light_pix_contrib = surface_brdf_material_editing * light_rgbs * cosine[:, :, None] * light_area_weight[None,:, None]   # [bs, envW * envH, 3]
            # surface_relight_rgb_chunk  = torch.sum(light_pix_contrib, dim=1)  # [bs, 3]

            # ### Tonemapping
            # surface_relight_rgb_chunk = torch.clamp(surface_relight_rgb_chunk, min=0.0, max=1.0)  # NOTE
            # ### Colorspace transform
            # if surface_relight_rgb_chunk.shape[0] > 0:
            #     surface_relight_rgb_chunk = linear2srgb_torch(surface_relight_rgb_chunk)
            # relight_rgb_chunk[acc_chunk_mask] = surface_relight_rgb_chunk

            rgb_map.append(rgb_chunk.cpu().detach())
            depth_map.append(depth_chunk.cpu().detach())
            acc_map.append(acc_chunk.cpu().detach())
            normal_map.append(normal_chunk.cpu().detach())
            # relight_rgb_map.append(relight_rgb_chunk.cpu().detach())
            albedo_map.append(albedo_chunk.cpu().detach())
            roughness_map.append(roughness_chunk.cpu().detach())

        rgb_map = torch.cat(rgb_map, dim=0)
        depth_map = torch.cat(depth_map, dim=0)
        acc_map = torch.cat(acc_map, dim=0)
        normal_map = torch.cat(normal_map, dim=0)
        acc_map_mask = (acc_map > args.acc_mask_threshold)
        albedo_map = torch.cat(albedo_map, dim=0)
        roughness_map = torch.cat(roughness_map, dim=0)
        os.makedirs(os.path.join(cur_dir_path, 'relighting'), exist_ok=True)

        for light_name_idx, cur_light_name in enumerate(dataset.light_names):
            if os.path.exists(os.path.join(cur_dir_path, 'relighting', f'{cur_light_name}.png')):
                relight_rgb_map = imageio.imread(os.path.join(cur_dir_path, 'relighting', f'{cur_light_name}.png'))
                relight_rgb_map = relight_rgb_map.astype(np.float32) / 255.0
            else:
                relight_rgb_map = torch.cat(relight_pred_img[cur_light_name], dim=0).reshape(H, W, 3).numpy()
        #     gt_img_map = gt_rgb[light_name_idx].numpy()
        #     loss_relight = np.mean((relight_rgb_map  - gt_img_map) ** 2)
        #     cur_psnr = -10.0 * np.log(loss_relight) / np.log(10.0)

        #     ssim_relight = rgb_ssim(relight_rgb_map, gt_img_map, 1)
        #     l_a_relight = rgb_lpips(gt_img_map, relight_rgb_map, 'alex', tensorfactor.device)
        #     l_v_relight = rgb_lpips(gt_img_map, relight_rgb_map, 'vgg', tensorfactor.device)

        #     relight_psnr[cur_light_name].append(cur_psnr)
        #     relight_ssim[cur_light_name].append(ssim_relight)
        #     relight_l_alex[cur_light_name].append(l_a_relight)
        #     relight_l_vgg[cur_light_name].append(l_v_relight)

            # relight_rgb_map = torch.cat(relight_pred_img[cur_light_name], dim=0).reshape(H, W, 3).numpy()
            if args.if_save_relight_rgb:
                imageio.imwrite(os.path.join(cur_dir_path, 'relighting', f'{cur_light_name}.png'), (relight_rgb_map * 255).astype('uint8'))

        # # write relight image psnr to a txt file
        # with open(os.path.join(cur_dir_path, 'relighting', 'relight_psnr.txt'), 'w') as f:
        #     for cur_light_name in dataset.light_names:
        #         f.write(f'{cur_light_name}: PNSR {relight_psnr[cur_light_name][-1]}; SSIM {relight_ssim[cur_light_name][-1]}; L_Alex {relight_l_alex[cur_light_name][-1]}; L_VGG {relight_l_vgg[cur_light_name][-1]}\n')

        rgb_map = (rgb_map.reshape(H, W, 3).numpy() * 255).astype('uint8')
        rgb_frames_list.append(rgb_map)
        depth_map, _ = visualize_depth_numpy(depth_map.reshape(H, W, 1).numpy(), near_far)
        acc_map = (acc_map.reshape(H, W, 1).numpy() * 255).astype('uint8')

        if args.if_save_rgb:
            imageio.imwrite(os.path.join(cur_dir_path, 'rgb.png'), rgb_map)
        # if args.if_save_relight_rgb:
        #     relight_rgb_map = relight_rgb_map.reshape(H, W, 3)
        #     material_editing_list.append(relight_rgb_map)
        #     # gt_rgb_mask = gt_mask.reshape(H, W)
        #     # ratio = (gt_rgb[gt_rgb_mask][:, 0] / relight_rgb_map[gt_rgb_mask][:, 0]).median()
        #     # relight_rgb_map = ratio * relight_rgb_map
        #     imageio.imwrite(os.path.join(cur_dir_path, 'relight_rgb.png'), (relight_rgb_map * 255).numpy().astype('uint8'))

        if args.if_save_depth:
            imageio.imwrite(os.path.join(cur_dir_path, 'depth.png'), depth_map)
        if args.if_save_acc:
            imageio.imwrite(os.path.join(cur_dir_path, 'acc.png'), acc_map)
        if args.if_save_albedo:
            gt_albedo_reshaped = gt_albedo.reshape(H, W, 3).cpu()
            albedo_map = albedo_map.reshape(H, W, 3)
            # three channels rescale
            gt_albedo_mask = gt_mask.reshape(H, W)
            ratio_value, _ = (gt_albedo_reshaped[gt_albedo_mask]/ albedo_map[gt_albedo_mask].clamp(min=1e-6)).median(dim=0)
            # ratio_value = gt_albedo_reshaped[gt_albedo_mask].median(dim=0)[0] / albedo_map[gt_albedo_mask].median(dim=0)[0] 
            albedo_map[gt_albedo_mask] = (ratio_value * albedo_map[gt_albedo_mask]).clamp(min=0.0, max=1.0)

            # #single channel rescale
            # gt_albedo_mask = gt_mask.reshape(H, W)
            # ratio_value = (gt_albedo_reshaped[gt_albedo_mask]/ albedo_map[gt_albedo_mask])[..., 0].median()
            # ratio_value = (gt_albedo_reshaped[gt_albedo_mask]/ albedo_map[gt_albedo_mask])[..., 0].median()
            # albedo_map[gt_albedo_mask] = (ratio_value * albedo_map[gt_albedo_mask]).clamp(min=0.0, max=1.0)
            
            imageio.imwrite(os.path.join(cur_dir_path, 'albedo.png'), (albedo_map * 255).numpy().astype('uint8'))
            if args.if_save_albedo_gamma_corrected:
                to_save_albedo = (albedo_map ** (1/2.2) * 255).numpy().astype('uint8')
                to_save_albedo = np.concatenate([to_save_albedo, acc_map], axis=2)
                # gamma cororection
                imageio.imwrite(os.path.join(cur_dir_path, 'albedo_gamma_corrected.png'), to_save_albedo)

            # save GT gamma corrected albedo
            gt_albedo_reshaped = (gt_albedo_reshaped ** (1/2.2) * 255).numpy().astype('uint8')
            gt_albedo_reshaped = np.concatenate([gt_albedo_reshaped, acc_map], axis=2)
            imageio.imwrite(os.path.join(cur_dir_path, 'gt_albedo_gamma_corrected.png'), gt_albedo_reshaped)

            aligned_albedo_list.append(((albedo_map ** (1.0/2.2)) * 255).numpy().astype('uint8'))

            roughness_map = roughness_map.reshape(H, W, 1)
            # expand to three channels
            roughness_map = (roughness_map.expand(-1, -1, 3) * 255)
            roughness_map = np.concatenate([roughness_map, acc_map], axis=2)
            imageio.imwrite(os.path.join(cur_dir_path, 'roughness.png'), (roughness_map).astype('uint8'))
            roughness_list.append((roughness_map).astype('uint8'))
        if args.if_render_normal:
            normal_map = F.normalize(normal_map, dim=-1)
            normal_rgb_map = normal_map * 0.5 + 0.5
            normal_rgb_map = (normal_rgb_map.reshape(H, W, 3).numpy() * 255).astype('uint8')
            normal_rgb_map = np.concatenate([normal_rgb_map, acc_map], axis=2)
            imageio.imwrite(os.path.join(cur_dir_path, 'normal.png'), normal_rgb_map)
            gt_normal = F.normalize(gt_normal, dim=-1)
            gt_normal_rgb_map = gt_normal * 0.5 + 0.5
            gt_normal_rgb_map = (gt_normal_rgb_map.reshape(H, W, 3).numpy() * 255).astype('uint8')
            gt_normal_list.append(gt_normal_rgb_map)
            optimized_normal_list.append(normal_rgb_map)

    # write relight image psnr to a txt file
    with open(os.path.join(args.geo_buffer_path, 'relight_psnr.txt'), 'w') as f:
        for cur_light_name in dataset.light_names:
            f.write(f'{cur_light_name}:  PSNR {np.mean(relight_psnr[cur_light_name])}; SSIM {np.mean(relight_ssim[cur_light_name])}; L_Alex {np.mean(relight_l_alex[cur_light_name])}; L_VGG {np.mean(relight_l_vgg[cur_light_name])}\n')

    if args.if_save_rgb_video:
        video_path = os.path.join(args.geo_buffer_path,'video')
        os.makedirs(video_path, exist_ok=True)
        imageio.mimsave(os.path.join(video_path, 'rgb_video.mp4'), np.stack(rgb_frames_list), fps=24, macro_block_size=1)

    if args.if_render_normal:
        video_path = os.path.join(args.geo_buffer_path,'video')
        os.makedirs(video_path, exist_ok=True)
        imageio.mimsave(os.path.join(video_path, 'gt_normal_video.mp4'), np.stack(gt_normal_list), fps=24, macro_block_size=1)
        imageio.mimsave(os.path.join(video_path, 'render_normal_video.mp4'), np.stack(optimized_normal_list), fps=24, macro_block_size=1)

    if args.if_save_albedo:
        video_path = os.path.join(args.geo_buffer_path,'video')
        os.makedirs(video_path, exist_ok=True)
        imageio.mimsave(os.path.join(video_path, 'aligned_albedo_video.mp4'), np.stack(aligned_albedo_list), fps=24, macro_block_size=1)
        imageio.mimsave(os.path.join(video_path, 'roughness_video.mp4'), np.stack(roughness_list), fps=24, macro_block_size=1)
    if args.if_material_editing: 
        video_path = os.path.join(args.geo_buffer_path,'video')
        os.makedirs(video_path, exist_ok=True)
        imageio.mimsave(os.path.join(video_path, 'material_editing.mp4'), np.stack(material_editing_list), fps=24, macro_block_size=1)
    if args.render_video:
        video_path = os.path.join(args.geo_buffer_path,'video')
        os.makedirs(video_path, exist_ok=True)
        
        for cur_light_name in dataset.light_names:
            frame_list = []

            for render_idx in range(len(dataset)):
                cur_dir_path = os.path.join(args.geo_buffer_path, f'{dataset.split}_{render_idx:0>3d}', 'relighting')
                frame_list.append(imageio.imread(os.path.join(cur_dir_path, f'{cur_light_name}.png')))

            imageio.mimsave(os.path.join(video_path, f'{cur_light_name}_video.mp4'), np.stack(frame_list), fps=24, macro_block_size=1)

if __name__ == "__main__":
    args = config_parser()
    print(args)
    print("*" * 80)
    print('The result will be saved in {}'.format(os.path.abspath(args.geo_buffer_path)))


    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    torch.cuda.manual_seed_all(20211202)
    np.random.seed(20211202)


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
    args.if_render_normal = True
    args.vis_equation = 'nerfactor'
    args.if_material_editing = False

    args.render_video = True

    dataset = dataset_dict[args.dataset_name]

    # light_name_list= ['bridge','city', 'courtyard', 'forest', 'fireplace', 'interior', 'museum', 'night', 'snow', 'square', 'studio',
    #                         'sunrise', 'sunset', 'tunnel']
    light_name_list= ['bridge', 'city', 'fireplace', 'forest', 'night']

    test_dataset = dataset(                            
                            args.datadir, 
                            args.hdrdir, 
                            split='test', 
                            random_test=False,
                            downsample=args.downsample_test,
                            light_names=light_name_list,
                            light_rotation=args.light_rotation
                            )
    relight(test_dataset , args)

    