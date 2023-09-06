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
                light_rgbs = read_hdr(self.hdr_path)
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
        chunk_idxs = torch.split(torch.arange(frame_rays.shape[0]), args.batch_size) 
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

    global_rescale_value_single, global_rescale_value_three = compute_rescale_ratio(tensoIR, dataset)
    rescale_value = global_rescale_value_three
    rescale_value = torch.tensor([1., 1., 1.], device=device)

    relight_psnr, relight_pred_img, relight_gt_img = dict(), dict(), dict()
    for cur_light_name in dataset.light_names:
        relight_psnr[f'{cur_light_name}'] = []
        relight_pred_img[f'{cur_light_name}'] = []
        relight_gt_img[f'{cur_light_name}'] = []



    for idx in tqdm(range(len(dataset))):
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
        relight_rgb_map_only_direct = []
        relight_rgb_map_only_indirect = []
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
            relight_rgb_chunk_only_direct = torch.ones_like(rgb_chunk)
            relight_rgb_chunk_only_indirect = torch.ones_like(rgb_chunk)
            material_editing_rgb_chunk = torch.ones_like(rgb_chunk) 
            # albedo_chunk = gt_albedo[chunk_idx] # use GT to debug
            acc_chunk_mask = (acc_chunk > args.acc_mask_threshold)
            rays_o_chunk, rays_d_chunk = frame_rays[chunk_idx][:, :3], frame_rays[chunk_idx][:, 3:]
            surface_xyz_chunk = rays_o_chunk + depth_chunk.unsqueeze(-1) * rays_d_chunk  # [bs, 3]
            masked_surface_xyz = surface_xyz_chunk[acc_chunk_mask] # [surface_point_num, 3]
            
            masked_normal_chunk = normal_chunk[acc_chunk_mask] # [surface_point_num, 3]
            masked_albedo_chunk = albedo_chunk[acc_chunk_mask] # [surface_point_num, 3]
            masked_roughness_chunk = roughness_chunk[acc_chunk_mask] # [surface_point_num, 1]
            masked_fresnel_chunk = fresnel_chunk[acc_chunk_mask] # [surface_point_num, 1]
            masked_light_idx_chunk = light_idx[chunk_idx][acc_chunk_mask] # [surface_point_num, 1]

            ## Get incident light direction
            light_area_weight = tensorfactor.light_area_weight.to(device) # [envW * envH, ]

            incident_light_dirs = tensorfactor.gen_light_incident_dirs(method='fixed_envirmap').to(device)  # [envW * envH, 3]
            surf2l = incident_light_dirs.reshape(1, -1, 3).repeat(masked_surface_xyz.shape[0], 1, 1)  # [bs, envW * envH, 3]


            surf2c = -rays_d_chunk[acc_chunk_mask]  # [bs, 3]
            surf2c = safe_l2_normalize(surf2c, dim=-1)  # [bs, 3]

            try:
                ## get visibilty map from visibility network or compute it using density
                cosine = torch.einsum("ijk,ik->ij", surf2l, masked_normal_chunk)    # surf2l:[surface_point_num, envW * envH, 3] * masked_normal_chunk:[surface_point_num, 3] -> cosine:[surface_point_num, envW * envH]
                cosine_mask = (cosine > 1e-6)  # [surface_point_num, envW * envH] mask half of the incident light that is behind the surface
                visibility = torch.zeros((*cosine_mask.shape, 1), device=device)    # [surface_point_num, envW * envH, 1]
                indirect_light = torch.zeros((*cosine_mask.shape, 3), device=device)    # [surface_point_num, envW * envH, 1]
                masked_surface_xyz = masked_surface_xyz[:, None, :].expand((*cosine_mask.shape, 3))  # [surface_point_num, envW * envH, 3]
                masked_light_idx_chunk = masked_light_idx_chunk[:, None].expand((*cosine_mask.shape, 1))  # [surface_point_num, envW * envH, 1]
                cosine_masked_surface_pts = masked_surface_xyz[cosine_mask] # [num_of_vis_to_get, 3]
                cosine_masked_surf2l = surf2l[cosine_mask] # [num_of_vis_to_get, 3]
                cosine_masked_visibility = torch.zeros(cosine_masked_surf2l.shape[0], 1, device=device) # [num_of_vis_to_get, 1]
                cosine_masked_indirect_light =  torch.zeros(cosine_masked_surf2l.shape[0], 3, device=device)  # [num_of_vis_to_get, 1]
                
                cosine_masked_light_idx = masked_light_idx_chunk[cosine_mask] # [num_of_vis_to_get, 1]
            except:
                import ipdb; ipdb.set_trace()

            chunk_idxs_vis = torch.split(torch.arange(cosine_masked_surface_pts.shape[0]), 100000)  

            for chunk_vis_idx in chunk_idxs_vis:
                try:
                    chunk_surface_pts = cosine_masked_surface_pts[chunk_vis_idx]  # [chunk_size, 3]
                    chunk_surf2light = cosine_masked_surf2l[chunk_vis_idx]    # [chunk_size, 3]
                    chunk_light_idx = cosine_masked_light_idx[chunk_vis_idx]      # [chunk_size, 1]
                except:
                    import ipdb; ipdb.set_trace()
                if args.if_predict_single_view_visibility:
                    cosine_masked_visibility[chunk_vis_idx] = visibility_net(chunk_surface_pts, chunk_surf2light) # [chunk_size, 1]
                else :
                    nerv_vis, nerfactor_vis, indirect_light_chunk = compute_radiance(
                                                                    tensorfactor=tensorfactor, 
                                                                    surf_pts=chunk_surface_pts, 
                                                                    light_in_dir=chunk_surf2light,
                                                                    light_idx=chunk_light_idx,  
                                                                    nSample=96, 
                                                                    vis_near=0.05,
                                                                    vis_far=1.5
                                                                    ) # [chunk_size, 1]
                    if args.vis_equation == 'nerfactor':
                        cosine_masked_visibility[chunk_vis_idx] = nerfactor_vis.unsqueeze(-1)
                    elif args.vis_equation == 'nerv':
                        cosine_masked_visibility[chunk_vis_idx] = nerv_vis.unsqueeze(-1)
                    cosine_masked_indirect_light[chunk_vis_idx] = indirect_light_chunk
                visibility[cosine_mask] = cosine_masked_visibility
                indirect_light[cosine_mask] = cosine_masked_indirect_light
            ## Get BRDF specs
            nlights = surf2l.shape[1]
            

            # relighting
            specular_relighting = brdf_specular(masked_normal_chunk, surf2c, surf2l, masked_roughness_chunk, masked_fresnel_chunk)  # [surface_point_num, envW * envH, 3]
            masked_albedo_chunk_rescaled = masked_albedo_chunk
            surface_brdf_relighting = masked_albedo_chunk_rescaled.unsqueeze(1).expand(-1, nlights, -1) / np.pi + specular_relighting # [surface_point_num, envW * envH, 3]

            try:
                rotated_light_rgbs = tensorfactor.get_light_rgbs(incident_light_dirs, device=device).to(device) # [rotation_num, envW * envH, 3]
                direct_light_rgbs = torch.index_select(rotated_light_rgbs, dim=0, index=light_idx[chunk_idx][acc_chunk_mask].squeeze()).to(device) # [bs, envW * envH, 3]

                # only direct light
                light_rgbs = visibility * direct_light_rgbs  # [bs, envW * envH, 3]
                light_pix_contrib = surface_brdf_relighting * light_rgbs * cosine[:, :, None] * light_area_weight[None,:, None]   # [bs, envW * envH, 3]
                surface_relight_rgb_chunk  = torch.sum(light_pix_contrib, dim=1)  # [bs, 3]
            except:
                import ipdb; ipdb.set_trace()

            ### Tonemapping
            surface_relight_rgb_chunk = torch.clamp(surface_relight_rgb_chunk, min=0.0, max=1.0)  # NOTE
            ### Colorspace transform
            if surface_relight_rgb_chunk.shape[0] > 0:
                surface_relight_rgb_chunk = linear2srgb_torch(surface_relight_rgb_chunk)
            relight_rgb_chunk_only_direct[acc_chunk_mask] = surface_relight_rgb_chunk

            # only indirect light
            light_rgbs = indirect_light
            light_pix_contrib = surface_brdf_relighting * light_rgbs * cosine[:, :, None] * light_area_weight[None,:, None]   # [bs, envW * envH, 3]
            surface_relight_rgb_chunk  = torch.sum(light_pix_contrib, dim=1)  # [bs, 3]
            ### Tonemapping
            surface_relight_rgb_chunk = torch.clamp(surface_relight_rgb_chunk, min=0.0, max=1.0)  # NOTE
            ### Colorspace transform
            if surface_relight_rgb_chunk.shape[0] > 0:
                surface_relight_rgb_chunk = linear2srgb_torch(surface_relight_rgb_chunk)
            relight_rgb_chunk_only_indirect[acc_chunk_mask] = surface_relight_rgb_chunk


            # direct + indirect light
            light_rgbs = visibility * direct_light_rgbs + indirect_light
            light_pix_contrib = surface_brdf_relighting * light_rgbs * cosine[:, :, None] * light_area_weight[None,:, None]   # [bs, envW * envH, 3]
            surface_relight_rgb_chunk  = torch.sum(light_pix_contrib, dim=1)  # [bs, 3]
            ### Tonemapping
            surface_relight_rgb_chunk = torch.clamp(surface_relight_rgb_chunk, min=0.0, max=1.0)  # NOTE
            ### Colorspace transform
            if surface_relight_rgb_chunk.shape[0] > 0:
                surface_relight_rgb_chunk = linear2srgb_torch(surface_relight_rgb_chunk)
            relight_rgb_chunk[acc_chunk_mask] = surface_relight_rgb_chunk

            rgb_map.append(rgb_chunk.cpu().detach())
            depth_map.append(depth_chunk.cpu().detach())
            acc_map.append(acc_chunk.cpu().detach())
            normal_map.append(normal_chunk.cpu().detach())
            relight_rgb_map.append(relight_rgb_chunk.cpu().detach())
            relight_rgb_map_only_direct.append(relight_rgb_chunk_only_direct.cpu().detach())
            relight_rgb_map_only_indirect.append(relight_rgb_chunk_only_indirect.cpu().detach())
            albedo_map.append(albedo_chunk.cpu().detach())
            roughness_map.append(roughness_chunk.cpu().detach())

        rgb_map = torch.cat(rgb_map, dim=0)
        depth_map = torch.cat(depth_map, dim=0)
        acc_map = torch.cat(acc_map, dim=0)
        normal_map = torch.cat(normal_map, dim=0)

        albedo_map = torch.cat(albedo_map, dim=0)
        roughness_map = torch.cat(roughness_map, dim=0)
        relight_rgb_map = torch.cat(relight_rgb_map, dim=0)
        relight_rgb_map_only_direct = torch.cat(relight_rgb_map_only_direct, dim=0)
        relight_rgb_map_only_indirect = torch.cat(relight_rgb_map_only_indirect, dim=0)

        rgb_map = (rgb_map.reshape(H, W, 3).numpy() * 255).astype('uint8')
        rgb_frames_list.append(rgb_map)
        depth_map, _ = visualize_depth_numpy(depth_map.reshape(H, W, 1).numpy(), near_far)
        acc_map = (acc_map.reshape(H, W, 1).numpy() * 255).astype('uint8')

        if args.if_save_rgb:
            imageio.imwrite(os.path.join(cur_dir_path, 'rgb.png'), rgb_map)
        if args.if_save_relight_rgb:
            relight_rgb_map = relight_rgb_map.reshape(H, W, 3)
            imageio.imwrite(os.path.join(cur_dir_path, 'relight_rgb.png'), (relight_rgb_map * 255).numpy().astype('uint8'))
            relight_rgb_map_only_direct = relight_rgb_map_only_direct.reshape(H, W, 3)
            imageio.imwrite(os.path.join(cur_dir_path, 'relight_rgb_only_direct.png'), (relight_rgb_map_only_direct * 255).numpy().astype('uint8'))
            relight_rgb_map_only_indirect = relight_rgb_map_only_indirect.reshape(H, W, 3)
            imageio.imwrite(os.path.join(cur_dir_path, 'relight_rgb_only_indirect.png'), (relight_rgb_map_only_indirect * 255).numpy().astype('uint8'))

        if args.if_save_depth:
            imageio.imwrite(os.path.join(cur_dir_path, 'depth.png'), depth_map)
        if args.if_save_acc:
            imageio.imwrite(os.path.join(cur_dir_path, 'acc.png'), acc_map)
        if args.if_save_albedo:
            gt_albedo_reshaped = gt_albedo.reshape(H, W, 3).cpu()
            albedo_map = albedo_map.reshape(H, W, 3)
            # three channels rescale
            gt_albedo_mask = gt_mask.reshape(H, W)
            ratio_value, _ = (gt_albedo_reshaped[gt_albedo_mask]/ albedo_map[gt_albedo_mask]).median(dim=0)
            # ratio_value = gt_albedo_reshaped[gt_albedo_mask].median(dim=0)[0] / albedo_map[gt_albedo_mask].median(dim=0)[0] 
            albedo_map[gt_albedo_mask] = (ratio_value * albedo_map[gt_albedo_mask]).clamp(min=0.0, max=1.0)

            # #single channel rescale
            # gt_albedo_mask = gt_mask.reshape(H, W)
            # ratio_value = (gt_albedo_reshaped[gt_albedo_mask]/ albedo_map[gt_albedo_mask])[..., 0].median()
            # ratio_value = (gt_albedo_reshaped[gt_albedo_mask]/ albedo_map[gt_albedo_mask])[..., 0].median()
            # albedo_map[gt_albedo_mask] = (ratio_value * albedo_map[gt_albedo_mask]).clamp(min=0.0, max=1.0)

            imageio.imwrite(os.path.join(cur_dir_path, 'albedo.png'), (albedo_map * 255).numpy().astype('uint8'))
            if args.if_save_albedo_gamma_corrected:
                # gamma cororection
                imageio.imwrite(os.path.join(cur_dir_path, 'albedo_gamma_corrected.png'), ((albedo_map ** (1/2.2)) * 255).numpy().astype('uint8'))
        
            aligned_albedo_list.append(((albedo_map ** (1.0/2.2)) * 255).numpy().astype('uint8'))

            roughness_map = roughness_map.reshape(H, W, 1)
            imageio.imwrite(os.path.join(cur_dir_path, 'roughness.png'), (roughness_map * 255).numpy().astype('uint8'))
            roughness_list.append((roughness_map  * 255).numpy().astype('uint8'))
        if args.if_render_normal:
            normal_map = F.normalize(normal_map, dim=-1)
            normal_rgb_map = normal_map * 0.5 + 0.5
            normal_rgb_map = (normal_rgb_map.reshape(H, W, 3).numpy() * 255).astype('uint8')
            imageio.imwrite(os.path.join(cur_dir_path, 'normal.png'), normal_rgb_map)
            gt_normal = F.normalize(gt_normal, dim=-1)
            gt_normal_rgb_map = gt_normal * 0.5 + 0.5
            gt_normal_rgb_map = (gt_normal_rgb_map.reshape(H, W, 3).numpy() * 255).astype('uint8')
            gt_normal_list.append(gt_normal_rgb_map)
            optimized_normal_list.append(normal_rgb_map)


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
    args.if_compute_single_view_visibility = True # single view, with rotated OLAT, compute the transimittance in NeRV as visibility 
    args.if_render_normal = False
    args.vis_equation = 'nerfactor'
    args.if_material_editing = False

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

    