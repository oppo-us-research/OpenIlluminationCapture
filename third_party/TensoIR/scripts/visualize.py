'''
Author: Haian-Jin 3190106083@zju.edu.cn
Date: 2022-08-03 20:37:43
LastEditors: Haian-Jin 3190106083@zju.edu.cn
LastEditTime: 2022-09-05 14:43:27
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



@torch.no_grad()
def visualize(dataset, args):

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
        print('Not use the visibility network')
    W, H = dataset.img_wh
    near_far = dataset.near_far
    light_xyz, light_area, light_pdf = gen_light_xyz(16, 32)
    light_xyz_to_predicted = torch.from_numpy(light_xyz[light_xyz[:,:,-1] > 0]).to(device, dtype=torch.float32) # [predicted_light_num, 3], find the light position that is not behind the surface
    # light_xyz_to_predicted = torch.from_numpy(light_xyz[light_xyz[:,:,-1] < 0]).to(device, dtype=torch.float32) # [predicted_light_num, 3], find the light position that is not behind the surface
    # light_xyz_to_predicted = torch.from_numpy(light_xyz.reshape(-1, 3)).to(device, dtype=torch.float32) # [predicted_light_num, 3], find the light position that is not behind the surface


    rgb_frames_list = []
    gt_normal_list, gt_normal_rgb_list, gt_albedo_list = [], [], []
    optimized_normal_list, optimized_normal_rgb_list, optimized_albedo_one_list,  optimized_albedo_three_list= [], [], [], []
    # for idx in tqdm(range(len(dataset))):
    for idx in tqdm(range(1)):
        cur_dir_path = os.path.join(args.geo_buffer_path, f'{dataset.split}_{idx:0>3d}')
        os.makedirs(cur_dir_path, exist_ok=True)
        # item = dataset[199] 
        idx = 103
        item = dataset[idx]

        frame_rays = item['rays'].squeeze(0).to(device) # [H*W, 6]
        gt_normal = item['normals'].squeeze(0).cpu() # [H*W, 3]
        gt_albedo = item['albedo'].squeeze(0).to(device) # [H*W, 3]
        gt_mask = item['rgbs_mask'].squeeze(0).cpu() # [H*W]
        light_idx = torch.zeros((frame_rays.shape[0], 1), dtype=torch.int).to(device).fill_(0)

        step1 = 0
        step2 = 0
        step3 = 0
        step4 = 0
        step5 = 0
        rgb_map, depth_map, normal_map, albedo_map, roughness_map, fresnel_map, relight_rgb_map, normals_diff_map, normals_orientation_loss_map = [], [], [], [], [], [], [], [], []
        acc_map = []
        visibility_map_net, visibility_map_from_density = [], []

        visibility_generated_with_indirect_light, indirect_light, indirect_light_cosine_masked = [], [], []
        mix_visibility_preditct, mix_indirect_light, mix_invisibile_to_direct_light_mask, mix_visibility_masked = [], [], [], []
        
        rays_o, rays_d = frame_rays[:, :3].cpu(), frame_rays[:, 3:].cpu()
        chunk_idxs = torch.split(torch.arange(frame_rays.shape[0]), args.batch_size_test) # choose the first light idx
        
        for chunk_idx in chunk_idxs:
            checkpoint1 = datetime.now()
            with torch.enable_grad():
                rgb_chunk, depth_chunk, normal_chunk, albedo_chunk, roughness_chunk, _, acc_chunk, *temp = \
                    tensorfactor(frame_rays[chunk_idx], light_idx[chunk_idx], is_train=False, white_bg=True, ndc_ray=False, N_samples=-1)
            
            

            
            checkpoint2 = datetime.now()
            step1 += (checkpoint2 - checkpoint1).total_seconds()
            if args.if_predict_single_view_visibility: # get visibility using visibility net
                visibility_chunk = torch.zeros((chunk_idx.shape[0], light_xyz_to_predicted.shape[0], 1), dtype=torch.float32).to(device)
                acc_chunk_mask = (acc_chunk > args.acc_mask_threshold)
                rays_o_chunk, rays_d_chunk = frame_rays[chunk_idx][:, :3], frame_rays[chunk_idx][:, 3:]
                surface_xyz_chunk = rays_o_chunk + depth_chunk.unsqueeze(-1) * rays_d_chunk  # [bs, 3]
                masked_surface_xyz = surface_xyz_chunk[acc_chunk_mask] # [surface_point_num, 3]
                visibility_chunk_mask = predict_visibility(visibility_net, masked_surface_xyz, light_xyz_to_predicted) # [surface_point_num, predicted_light_num]
                visibility_chunk[acc_chunk_mask] = visibility_chunk_mask
                visibility_map_net.append(visibility_chunk.cpu().detach())
            checkpoint3 = datetime.now()
            if args.if_compute_single_view_visibility: # get visibility by accumulating the density
                # visibility_chunk = torch.zeros((chunk_idx.shape[0], light_xyz_to_predicted.shape[0], 1), dtype=torch.float32).to(device)
                visibility_chunk = torch.ones((chunk_idx.shape[0], light_xyz_to_predicted.shape[0], 1), dtype=torch.float32).to(device)
                acc_chunk_mask = (acc_chunk > args.acc_mask_threshold)
                rays_o_chunk, rays_d_chunk = frame_rays[chunk_idx][:, :3], frame_rays[chunk_idx][:, 3:]
                surface_xyz_chunk = rays_o_chunk + (depth_chunk - 0.01).unsqueeze(-1) * rays_d_chunk  # [bs, 3]
                masked_surface_xyz = surface_xyz_chunk[acc_chunk_mask] # [surface_point_num, 3]
                visibility_chunk_mask = compute_visibility( tensorfactor, 
                                                            masked_surface_xyz, 
                                                            light_xyz_to_predicted, 
                                                            96, 
                                                            0.05, 
                                                            1.5, 
                                                            args) 
                visibility_chunk[acc_chunk_mask] = visibility_chunk_mask
                visibility_map_from_density.append(visibility_chunk.cpu().detach())
            checkpoint4 = datetime.now()
            
            if args.if_compute_indirect_light:
                # visibility_chunk = torch.zeros((chunk_idx.shape[0], light_xyz_to_predicted.shape[0], 1), dtype=torch.float32).to(device)
                visibility_chunk = torch.ones((chunk_idx.shape[0], light_xyz_to_predicted.shape[0], 1), dtype=torch.float32).to(device)
                indirect_light_chunk = torch.zeros((chunk_idx.shape[0], light_xyz_to_predicted.shape[0], 3), dtype=torch.float32).to(device)
                indirect_light_cosine_chunk = torch.zeros((chunk_idx.shape[0], light_xyz_to_predicted.shape[0], 3), dtype=torch.float32).to(device)
                acc_chunk_mask = (acc_chunk > args.acc_mask_threshold)
                rays_o_chunk, rays_d_chunk = frame_rays[chunk_idx][:, :3], frame_rays[chunk_idx][:, 3:]
                surface_xyz_chunk = rays_o_chunk + (depth_chunk).unsqueeze(-1) * rays_d_chunk  # [H*W, 3]
                masked_surface_xyz = surface_xyz_chunk[acc_chunk_mask] # [surface_point_num, 3]
                visibility_chunk_mask, indirect_light_mask = compute_visibility_and_indirect_light( tensorfactor, 
                                                                                                    masked_surface_xyz, 
                                                                                                    light_xyz_to_predicted, 
                                                                                                    light_idx[chunk_idx], 
                                                                                                    96, 
                                                                                                    0.03, 
                                                                                                    1.5,
                                                                                                    args
                                                                                                    ) # [surface_point_num, predicted_light_num]
                visibility_chunk[acc_chunk_mask] = visibility_chunk_mask
                indirect_light_chunk[acc_chunk_mask] = indirect_light_mask

                surf2light = light_xyz_to_predicted[None, :, :] - masked_surface_xyz[:, None, :] # [N, preditected_light_num, 3]
                surf2light = safe_l2_normalize(surf2light, dim=-1)  # [N, preditected_light_num, 3]
                cosine = torch.einsum("ijk,ik->ij", surf2light , normal_chunk[acc_chunk_mask])  # surf2light:[bs, predicted_light_num, 3] * normal_chunk:[bs, 3] -> cosine:[bs, predicted_light_num]
                cosine_mask = (cosine > 1e-6)  # [bs, predicted_light_num], mask half of the incident light that is behind the surface
                cosine[~cosine_mask] = 0.0
                indirect_light_cosine_chunk[acc_chunk_mask] = cosine[:, :, None] * indirect_light_mask # [bs, predicted_light_num, 1]

                visibility_generated_with_indirect_light.append(visibility_chunk.cpu().detach())
                indirect_light.append(indirect_light_chunk.cpu().detach())
                indirect_light_cosine_masked.append(indirect_light_cosine_chunk.cpu().detach())
            checkpoint5 = datetime.now()

            if args.if_mix_compute_indirect_light:
                indirect_light_chunk = torch.zeros((chunk_idx.shape[0], light_xyz_to_predicted.shape[0], 3), dtype=torch.float32).to(device)
                acc_chunk_mask = (acc_chunk > args.acc_mask_threshold)
                rays_o_chunk, rays_d_chunk = frame_rays[chunk_idx][:, :3], frame_rays[chunk_idx][:, 3:]
                surface_xyz_chunk = rays_o_chunk + (depth_chunk - 0.01).unsqueeze(-1) * rays_d_chunk  # [bs, 3]
                masked_surface_xyz = surface_xyz_chunk[acc_chunk_mask] # [surface_point_num, 3]
                masked_light_idx = light_idx[chunk_idx][acc_chunk_mask] # [surface_point_num, 1]
                _, _, indirect_light_mask , _ = \
                    mix_compute_visibility_and_indirect_light(  
                                                                visibility_net,
                                                                tensorfactor, 
                                                                masked_surface_xyz, 
                                                                light_xyz_to_predicted, 
                                                                masked_light_idx, 
                                                                96, 
                                                                0.03, 
                                                                1.5,
                                                                args
                                                            ) 
                indirect_light_chunk[acc_chunk_mask] = indirect_light_mask

                mix_indirect_light.append(indirect_light_chunk.cpu().detach())
            checkpoint6 = datetime.now()
            step2 += (checkpoint3 - checkpoint2).total_seconds()
            step3 += (checkpoint4 - checkpoint3).total_seconds()
            step4 += (checkpoint5 - checkpoint4).total_seconds()
            step5 += (checkpoint6 - checkpoint5).total_seconds()

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
        albedo_map = torch.cat(albedo_map, dim=0)
        acc_map_mask = (acc_map > args.acc_mask_threshold)
        # acc_map_mask = (gt_mask > 0)
        # import ipdb; ipdb.set_trace()
        surface_xyz = rays_o + depth_map.unsqueeze(-1) * rays_d  # [bs, 3]
        masked_surface_xyz = surface_xyz[acc_map_mask] # [surface_point_num, 3]
        masked_surface_rgb = rgb_map[acc_map_mask] # [surface_point_num, 3]


        print("step1:", step1)
        print("step2:", step2)
        print("step3:", step3)
        print("step4:", step4)
        print("step5:", step5)
        if args.if_predict_single_view_visibility:
            visibility_map = torch.cat(visibility_map_net, dim=0).reshape(H, W, -1, 1).permute(2, 0, 1, 3) # [predicted_light_num, H, W, 1]


            imageio.mimsave(os.path.join(cur_dir_path, 'visibility_from_net.mp4'), (visibility_map*255).numpy().astype(np.uint8), fps=24, macro_block_size=1)

        if args.if_compute_single_view_visibility:
            visibility_map = torch.cat(visibility_map_from_density, dim=0).reshape(H, W, -1, 1).permute(2, 0, 1, 3)
            # save every frame in images
            for i in range(visibility_map.shape[0]):
                imageio.imwrite(os.path.join(cur_dir_path, 'visibility_from_density_'+str(i)+'.png'), (visibility_map[i]*255).numpy().astype(np.uint8))
            

            os.makedirs(os.path.join(cur_dir_path, args.vis_equation), exist_ok=True)
            np.save(os.path.join(cur_dir_path, args.vis_equation, 'visibility_from_density.npy'), (visibility_map).numpy())
            imageio.mimsave(os.path.join(cur_dir_path, 'visibility_from_density.mp4'), (visibility_map*255).numpy().astype(np.uint8), fps=24, macro_block_size=1)
        
        # if True: # render OLAT environment map
        #     envir_map = []
        #     for i in range(light_xyz_to_predicted.shape[0]):
        #         temp_image = np.zeros((16*8, 32*8, 3), dtype=np.float32)
        #         row = i//32
        #         col = i%32
        #         temp_image[row*8:row*8+8, col*8:col*8+8] = 1.0
        #         envir_map.append(temp_image)
        #     envir_map = np.stack(envir_map, axis=0)
        #     imageio.mimsave(os.path.join(cur_dir_path, 'envir_map.mp4'), (envir_map*255).astype(np.uint8), fps=24, macro_block_size=1)


        # if True:
        #     def render_envmap_sg(lgtSGs, viewdirs):
        #         viewdirs = viewdirs.to(lgtSGs.device)
        #         viewdirs = viewdirs.unsqueeze(-2)  # [..., 1, 3]

        #         # [M, 7] ---> [..., M, 7]
        #         dots_sh = list(viewdirs.shape[:-2])
        #         M = lgtSGs.shape[0]
        #         lgtSGs = lgtSGs.view([1,] * len(dots_sh) + [M, 7]).expand(dots_sh + [M, 7])
                
        #         lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True))
        #         lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])
        #         lgtSGMus = torch.abs(lgtSGs[..., -3:]) 
        #         # [..., M, 3]
        #         rgb = lgtSGMus * torch.exp(lgtSGLambdas * \
        #             (torch.sum(viewdirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
        #         rgb = torch.sum(rgb, dim=-2)  # [..., 3]
        #         return rgb

        # nlights = 512 * 256
        # lat_step_size = np.pi / 256
        # lng_step_size = 2 * np.pi / 512
        # phi, theta = torch.meshgrid([torch.linspace(np.pi / 2 - 0.5 * lat_step_size, -np.pi / 2 + 0.5 * lat_step_size, 256), 
        #                             torch.linspace(np.pi - 0.5 * lng_step_size, -np.pi + 0.5 * lng_step_size, 512)], indexing='ij')

        # sin_phi = torch.sin(torch.pi / 2 - phi)  # [envH, envW]
        # light_area_weight = 4 * torch.pi * sin_phi / torch.sum(sin_phi)  # [envH, envW]
        # assert 0 not in light_area_weight, "There shouldn't be light pixel that doesn't contribute"
        # light_area_weight = light_area_weight.to(torch.float32).reshape(-1) # [envH * envW, ]

        # fixed_viewdirs = torch.stack([ torch.cos(theta) * torch.cos(phi), 
        #                                     torch.sin(theta) * torch.cos(phi), 
        #                                     torch.sin(phi)], dim=-1)    # [envH, envW, 3]
        # fixed_viewdirs = fixed_viewdirs.reshape(-1, 3).to(device) # [envH * envW, 3]
        # light_rgbs = render_envmap_sg(tensorfactor.lgtSGs.to(device), fixed_viewdirs).reshape(256, 512, 3) 
        # light_rgbs = torch.clamp(light_rgbs, 0, 1) ** (1.0 / 2.2)
        # imageio.imwrite(os.path.join(cur_dir_path, 'envmap.png'), (light_rgbs * 255).detach().cpu().numpy().astype(np.uint8))

        if args.if_compute_indirect_light:
            visibility_map = torch.cat(visibility_generated_with_indirect_light, dim=0).reshape(H, W, -1, 1).permute(2, 0, 1, 3)
            imageio.mimsave(os.path.join(cur_dir_path, 'visibility_generated_with_indirect_light.mp4'), (visibility_map*255).numpy().astype(np.uint8), fps=24, macro_block_size=1)
            imageio.imsave(os.path.join(cur_dir_path, 'visibility.png'), (torch.mean(visibility_map, dim=0)*255).detach().cpu().numpy().astype(np.uint8))
            indirect_light_map = torch.cat(indirect_light, dim=0).reshape(H, W, -1, 3).permute(2, 0, 1, 3)
            imageio.mimsave(os.path.join(cur_dir_path, 'indirect_light.mp4'), (indirect_light_map*255).numpy().astype(np.uint8), fps=24, macro_block_size=1)
            
            if args.if_compute_mean_received_indirect_light:
                indirect_light_cosine_masked_map = torch.cat(indirect_light_cosine_masked, dim=0).reshape(H, W, -1, 3)
                indirect_light_cosine_masked_map = indirect_light_cosine_masked_map.mean(dim=-2)
                # gamma correction
                imageio.imwrite(os.path.join(cur_dir_path, 'indirect_light_cosine_masked.png'), (indirect_light_cosine_masked_map*255).numpy().astype(np.uint8))
        if args.if_mix_compute_indirect_light:
            indirect_light_map = torch.cat(mix_indirect_light, dim=0).reshape(H, W, -1, 3).permute(2, 0, 1, 3)
            imageio.mimsave(os.path.join(cur_dir_path, 'mix_indirect_light.mp4'), (indirect_light_map*255).numpy().astype(np.uint8), fps=24, macro_block_size=1)



        # save surface_points as point cloud
        if args.if_save_point_cloud:
            import open3d as o3d
            point_cloud_surface = o3d.geometry.PointCloud()
            point_cloud_surface.points = o3d.utility.Vector3dVector(masked_surface_xyz.cpu().detach().numpy())
            point_cloud_surface.colors = o3d.utility.Vector3dVector(masked_surface_rgb.cpu().detach().numpy())
            o3d.io.write_point_cloud(os.path.join(cur_dir_path, 'surface.ply'), point_cloud_surface)

        depth_map, _ = visualize_depth_numpy(depth_map.reshape(H, W, 1).numpy(), near_far)
        acc_map = (acc_map.reshape(H, W, 1).numpy() * 255).astype('uint8')
        rgb_map = (rgb_map.reshape(H, W, 3).numpy() * 255).astype('uint8')
        rgba_map = np.concatenate([rgb_map, acc_map], axis=-1)
        
        rgb_frames_list.append(rgba_map)


        if args.if_save_rgb:
            imageio.imwrite(os.path.join(cur_dir_path, 'rgb.png'), rgba_map)
        if args.if_save_depth:
            imageio.imwrite(os.path.join(cur_dir_path, 'depth.png'), depth_map)
        if args.if_save_acc:
            imageio.imwrite(os.path.join(cur_dir_path, 'acc.png'), acc_map)

        if args.if_save_albedo:
            os.makedirs(os.path.join(cur_dir_path, 'albedo'), exist_ok=True)
            gt_albedo_reshaped = gt_albedo.reshape(H, W, 3).cpu()
            albedo_map = albedo_map.reshape(H, W, 3)
            single_aligned_albedo_map = albedo_map.clone()
            three_aligned_albedo_map = albedo_map.clone()
            # three channels rescale
            gt_albedo_mask = gt_mask.reshape(H, W)
            # PhySG's way of rescaling
            ratio_value, _ = (gt_albedo_reshaped[gt_albedo_mask]/ albedo_map[gt_albedo_mask].clamp(min=1e-6)).median(dim=0)
            ## NeRFactor's way of rescaling
            # ratio_value = torch.sum(albedo_map[gt_albedo_mask] * gt_albedo_reshaped[gt_albedo_mask], dim=0) / torch.sum(albedo_map[gt_albedo_mask] * albedo_map[gt_albedo_mask], dim=0)

            three_aligned_albedo_map[gt_albedo_mask] = (ratio_value * three_aligned_albedo_map[gt_albedo_mask]).clamp(min=0.0, max=1.0)
            
            # single channel rescale
            gt_albedo_mask = gt_mask.reshape(H, W)
            ratio_value = (gt_albedo_reshaped[gt_albedo_mask]/ albedo_map[gt_albedo_mask].clamp(min=1e-6))[..., 0].median()
            # ratio_value = ratio_value[0]
            single_aligned_albedo_map[gt_albedo_mask] = (ratio_value * single_aligned_albedo_map[gt_albedo_mask]).clamp(min=0.0, max=1.0)
            # imageio.imwrite(os.path.join(cur_dir_path, 'albedo','gt_albedo.png'), (gt_albedo_reshaped * 255).numpy().astype('uint8'))
            # imageio.imwrite(os.path.join(cur_dir_path, 'albedo','three_aligned_albedo.png'), (three_aligned_albedo_map * 255).numpy().astype('uint8'))
            imageio.imwrite(os.path.join(cur_dir_path, 'albedo','single_aligned_albedo.png'), (single_aligned_albedo_map * 255).numpy().astype('uint8'))
            
            # if args.if_save_albedo_gamma_corrected:
            #     # gamma cororection
            #     imageio.imwrite(os.path.join(cur_dir_path, 'albedo','gt_albedo_gamma.png'), (gt_albedo_reshaped ** (1/2.2)) * 255).numpy().astype('uint8')
            #     imageio.imwrite(os.path.join(cur_dir_path, 'albedo','single_aligned_albedo_gamma.png'), (single_aligned_albedo_map ** (1/2.2)) * 255).numpy().astype('uint8')



            # gamma cororection
            gt_albedo_gamma = (((gt_albedo_reshaped ** (1/2.2))) * 255).numpy().astype('uint8')
            gt_albedo_gamma = np.concatenate([gt_albedo_gamma, acc_map], axis=-1)
            imageio.imwrite(os.path.join(cur_dir_path, 'albedo','gt_albedo_gamma.png'), gt_albedo_gamma)
            imageio.imwrite(os.path.join(cur_dir_path, 'albedo','single_aligned_albedo_gamma.png'), (((single_aligned_albedo_map ** (1/2.2))) * 255).numpy().astype('uint8'))
            
            three_aligned_albedo_map = (((three_aligned_albedo_map ** (1/2.2))) * 255).numpy().astype('uint8')
            three_aligned_albedo_map_to_save = np.concatenate([three_aligned_albedo_map, acc_map], axis=-1)
            imageio.imwrite(os.path.join(cur_dir_path, 'albedo','three_aligned_albedo_gamma.png'), three_aligned_albedo_map_to_save)

            gt_albedo_list.append(gt_albedo_reshaped ** (1/2.2))
            optimized_albedo_one_list.append(single_aligned_albedo_map ** (1/2.2))
            optimized_albedo_three_list.append(three_aligned_albedo_map ** (1/2.2))

        if args.if_render_normal:
            normal_map = F.normalize(normal_map, dim=-1)
            optimized_normal_list.append(normal_map.reshape(H, W, 3).numpy())
            normal_rgb_map = normal_map * 0.5 + 0.5
            normal_rgb_map = (normal_rgb_map.reshape(H, W, 3).numpy() * 255).astype('uint8')
            normal_rgba_map = np.concatenate([normal_rgb_map, acc_map], axis=-1)
            imageio.imwrite(os.path.join(cur_dir_path, 'normal.png'), normal_rgba_map)

            gt_normal = F.normalize(gt_normal, dim=-1)
            gt_normal_list.append(gt_normal.reshape(H, W, 3).numpy())
            gt_normal_rgb_map = gt_normal * 0.5 + 0.5
            gt_normal_rgb_map = (gt_normal_rgb_map.reshape(H, W, 3).numpy() * 255).astype('uint8')
            gt_normal_rgb_list.append(gt_normal_rgb_map)
            optimized_normal_rgb_list.append(normal_rgb_map)

    if args.if_save_rgb_video:
        video_path = os.path.join(args.geo_buffer_path,'video')
        os.makedirs(video_path, exist_ok=True)
        imageio.mimsave(os.path.join(video_path, 'rgb_video.mp4'), np.stack(rgb_frames_list), fps=24, macro_block_size=1)

    if args.if_save_albedo_video:
        video_path = os.path.join(args.geo_buffer_path,'video')
        os.makedirs(video_path, exist_ok=True)

        imageio.mimsave(os.path.join(video_path, 'albedo_one_video.mp4'), np.stack(optimized_albedo_one_list), fps=24, macro_block_size=1)
        imageio.mimsave(os.path.join(video_path, 'albedo_three_video.mp4'), np.stack(optimized_albedo_three_list), fps=24, macro_block_size=1)
        imageio.mimsave(os.path.join(video_path, 'gt_albedo_video.mp4'), np.stack(gt_albedo_list), fps=24, macro_block_size=1)

    if args.if_render_normal:
        video_path = os.path.join(args.geo_buffer_path,'video')
        os.makedirs(video_path, exist_ok=True)
        imageio.mimsave(os.path.join(video_path, 'gt_normal_video.mp4'), np.stack(gt_normal_rgb_list), fps=24, macro_block_size=1)
        imageio.mimsave(os.path.join(video_path, 'render_normal_video.mp4'), np.stack(optimized_normal_rgb_list), fps=24, macro_block_size=1)
    if args.compute_MAE:
        gt_normal_stack = np.stack(gt_normal_list)
        render_normal_stack = np.stack(optimized_normal_list)
        # compute mean angular error
        MAE = np.mean(np.arccos(np.clip(np.sum(gt_normal_stack * render_normal_stack, axis=-1), -1, 1)) * 180 / np.pi)
        # mae = np.mean(np.arccos(np.dot(gt_normal_stack, render_normal_stack))) / np.pi * 180
        # gt = gt_normal_list[0]
        # predict = optimized_normal_list[0]
        # error_map = np.linalg.norm((gt - predict), axis=-1).reshape(H, W, 1)
        # error_map = (error_map / np.max(error_map) * 255).astype('uint8')
        # imageio.imwrite(os.path.join(video_path, 'error_map.png'), error_map)
        print('MAE: ', MAE)
    
    if args.compute_albedo_PSNR:
        gt_albedo_stack = np.stack(gt_albedo_list)
        render_albedo_one_stack = np.stack(optimized_albedo_one_list)
        render_albedo_three_stack = np.stack(optimized_albedo_three_list)
        # compute pnsr
        loss_albedo_one = np.mean((gt_albedo_stack - render_albedo_one_stack) ** 2)
        loss_albedo_three = np.mean((gt_albedo_stack - render_albedo_three_stack) ** 2)
        PSNR_one = -10.0 * np.log(loss_albedo_one.item()) / np.log(10.0)
        PSNR_three = -10.0 * np.log(loss_albedo_three.item()) / np.log(10.0)
        print('albedo PSNR one channel: ', PSNR_one)
        print('albedo PSNR three channel: ', PSNR_three)

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
    args.if_save_point_cloud = True
    args.if_save_rgb = True
    args.if_save_depth = False
    args.if_save_acc = True
    args.if_save_rgb_video = True
    args.acc_mask_threshold = 0.1
    args.if_predict_single_view_visibility = False # single view, with rotated OLAT, predict visibility from vis-net
    args.if_compute_single_view_visibility = True # single view, with rotated OLAT, compute the transimittance in NeRV as visibility 
    args.if_render_normal = True
    args.if_compute_indirect_light = False
    args.if_compute_mean_received_indirect_light = True

    args.if_mix_compute_indirect_light = False
    args.vis_equation = 'nerfactor'

    args.compute_MAE = True
    args.compute_albedo_PSNR = True
    args.if_save_albedo = True
    args.if_save_albedo_video = True

    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset( 
                            args.datadir, 
                            args.hdrdir, 
                            split='test', 
                            random_test=False,
                            downsample=args.downsample_test,
                            light_name=args.light_name,
                            light_rotation=args.light_rotation
                            # light_rotation=['000', '120', '240']

                            )
    visualize(test_dataset , args)