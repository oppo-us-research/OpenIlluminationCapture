
import os
from tqdm import tqdm
import imageio
import numpy as np
import time

from opt import config_parser
from models.shapeBuffer import ShapeModel
import torch
import torch.nn as nn
from utils import visualize_depth_numpy
from utils import *
# from models.tensoRF_rotated_lights import raw2alpha, TensorVMSplit, AlphaGridMask
from models.tensoRF_general_multi_lights import raw2alpha, TensorVMSplit, AlphaGridMask
from dataLoader.ray_utils import safe_l2_normalize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from dataLoader import dataset_dict
from models.relight_utils import *
brdf_specular = specular_pipeline_render_multilight_new
from models.relight_utils import Environment_Light





# @torch.no_grad()
# def test_on_synthesized_poses(dataset, args):

#     if not os.path.exists(args.ckpt):
#         print('the checkpoint path for tensoIR does not exists!!')
#         return
        

#     ckpt = torch.load(args.ckpt, map_location=device)
#     kwargs = ckpt['kwargs']
#     kwargs.update({'device': device})
#     tensoIR = eval(args.model_name)(**kwargs)
#     tensoIR.load(ckpt)

#     if args.ckpt_visibility is not None:
#         visibility_net:nn.Module = eval(args.vis_model_name)().to(device)
#         visibility_net.load_state_dict(torch.load(args.ckpt_visibility))
#         visibility_net.requires_grad_(False) # freeze the visibility net
#         print("load visibility network succcessfully")
#     else:
#         print('Not using visibility network')
#     W, H = dataset.img_wh

    
#     rgb_frames_list = []
#     envir_light = Environment_Light(args.hdrdir)
    
#     # use default light
#     # envir_light = None

#     #### 
#     light_rotation_idx = 0
#     ####


#     rescale_value = torch.tensor([[1.8, 1.97, 1]], device=device) * 0.6


#     test_rays = dataset.test_rays
#     test_w2c= dataset.test_w2c
#     os.makedirs(os.path.join(args.geo_buffer_path, f'rgb'), exist_ok=True)
#     os.makedirs(os.path.join(args.geo_buffer_path, f'albedo'), exist_ok=True)
#     os.makedirs(os.path.join(args.geo_buffer_path, f'albedo_gamma'), exist_ok=True)
#     os.makedirs(os.path.join(args.geo_buffer_path, f'normal'), exist_ok=True)
#     os.makedirs(os.path.join(args.geo_buffer_path, f'roughness'), exist_ok=True)
#     os.makedirs(os.path.join(args.geo_buffer_path, f'normal_camera'), exist_ok=True)
#     os.makedirs(os.path.join(args.geo_buffer_path, f'relighting'), exist_ok=True)

#     for idx in tqdm(range(test_rays.shape[0]), desc="relighting on synthesized poses"):
#         relight_pred_img_with_bg, relight_pred_img_without_bg = dict(), dict()
#         for cur_light_name in dataset.light_names:
#             relight_pred_img_with_bg[f'{cur_light_name}'] = []
#             relight_pred_img_without_bg[f'{cur_light_name}'] = []

#         # os.makedirs(cur_dir_path, exist_ok=True)
#         cur_dir_path = os.path.join(args.geo_buffer_path, 'relighting', f'{dataset.split}_{idx:0>3d}')
#         os.makedirs(cur_dir_path, exist_ok=True)

#         frame_rays = test_rays[idx].to(device) # [H*W, 6]
#         w2c = test_w2c[idx]
#         light_idx = torch.zeros((frame_rays.shape[0], 1), dtype=torch.int).to(device).fill_(light_rotation_idx)
        
#         rgb_map, depth_map, normal_map, albedo_map, roughness_map, fresnel_map, relight_rgb_map, normals_diff_map, normals_orientation_loss_map = [], [], [], [], [], [], [], [], []
#         acc_map_new = []
#         acc_map = []

#         chunk_idxs = torch.split(torch.arange(frame_rays.shape[0]), args.batch_size) # choose the first light idx
#         for chunk_idx in chunk_idxs:
#             with torch.enable_grad():
#                 rgb_chunk, depth_chunk, normal_chunk, albedo_chunk, roughness_chunk, \
#                     fresnel_chunk, acc_chunk, *temp \
#                     = tensoIR(frame_rays[chunk_idx], light_idx[chunk_idx], is_train=False, white_bg=True, ndc_ray=False, N_samples=-1)



#             relight_rgb_chunk = torch.ones_like(rgb_chunk) 
#             acc_map_chunk = acc_chunk
#             acc_chunk_mask = (acc_chunk > args.acc_mask_threshold)
            
#             rays_o_chunk, rays_d_chunk = frame_rays[chunk_idx][:, :3], frame_rays[chunk_idx][:, 3:]
#             surface_xyz_chunk = rays_o_chunk + depth_chunk.unsqueeze(-1) * rays_d_chunk  # [bs, 3]
#             masked_surface_pts = surface_xyz_chunk[acc_chunk_mask]      # [surface_point_num, 3]
            
#             masked_normal_chunk = normal_chunk[acc_chunk_mask]          # [surface_point_num, 3]
#             masked_albedo_chunk = albedo_chunk[acc_chunk_mask]          # [surface_point_num, 3]
#             masked_roughness_chunk = roughness_chunk[acc_chunk_mask]    # [surface_point_num, 1]
#             masked_fresnel_chunk = fresnel_chunk[acc_chunk_mask]        # [surface_point_num, 1]
        

#             ## Get incident light directions
#             for light_name_idx, cur_light_name in enumerate(dataset.light_names):

#                 relight_rgb_chunk.fill_(1.0)
#                 masked_light_dir, masked_light_rgb, masked_light_pdf = envir_light.sample_light(cur_light_name, masked_normal_chunk.shape[0], 512) # [bs, envW * envH, 3]
#                 surf2l = masked_light_dir # [surface_point_num, envW * envH, 3]
#                 surf2c = -rays_d_chunk[acc_chunk_mask]  # [surface_point_num, 3]
#                 surf2c = safe_l2_normalize(surf2c, dim=-1)  # [surface_point_num, 3]


#                 ## get visibilty map from visibility network or compute it using density
#                 cosine = torch.einsum("ijk,ik->ij", surf2l, masked_normal_chunk)    # surf2l:[surface_point_num, envW * envH, 3] * masked_normal_chunk:[surface_point_num, 3] -> cosine:[surface_point_num, envW * envH]
#                 cosine_mask = (cosine > 1e-6)  # [surface_point_num, envW * envH] mask half of the incident light that is behind the surface
#                 visibility = torch.zeros((*cosine_mask.shape, 1), device=device)    # [surface_point_num, envW * envH, 1]
#                 masked_surface_xyz = masked_surface_pts[:, None, :].expand((*cosine_mask.shape, 3))  # [surface_point_num, envW * envH, 3]

#                 cosine_masked_surface_pts = masked_surface_xyz[cosine_mask] # [num_of_vis_to_get, 3]
#                 cosine_masked_surf2l = surf2l[cosine_mask] # [num_of_vis_to_get, 3]
#                 cosine_masked_visibility = torch.zeros(cosine_masked_surf2l.shape[0], 1, device=device) # [num_of_vis_to_get, 1]

#                 chunk_idxs_vis = torch.split(torch.arange(cosine_masked_surface_pts.shape[0]), 100000)  

#                 for chunk_vis_idx in chunk_idxs_vis:
#                     chunk_surface_pts = cosine_masked_surface_pts[chunk_vis_idx]  # [chunk_size, 3]
#                     chunk_surf2light = cosine_masked_surf2l[chunk_vis_idx]    # [chunk_size, 3]
#                     if args.if_predict_single_view_visibility:
#                         cosine_masked_visibility[chunk_vis_idx] = visibility_net(chunk_surface_pts, chunk_surf2light) # [chunk_size, 1]
#                     else :
#                         nerv_vis, nerfactor_vis = compute_transmittance(tensoIR=tensoIR, 
#                                                                         surf_pts=chunk_surface_pts, 
#                                                                         light_in_dir=chunk_surf2light, 
#                                                                         nSample=96, 
#                                                                         vis_near=0.05,
#                                                                         vis_far=1.5
#                                                                         ) # [chunk_size, 1]
#                         if args.vis_equation == 'nerfactor':
#                             cosine_masked_visibility[chunk_vis_idx] = nerfactor_vis.unsqueeze(-1)
#                         elif args.vis_equation == 'nerv':
#                             cosine_masked_visibility[chunk_vis_idx] = nerv_vis.unsqueeze(-1)
#                     visibility[cosine_mask] = cosine_masked_visibility

#                 ## Get BRDF specs
#                 nlights = surf2l.shape[1]
                
#                 # relighting
#                 specular_relighting = brdf_specular(masked_normal_chunk, surf2c, surf2l, masked_roughness_chunk, masked_fresnel_chunk)  # [surface_point_num, envW * envH, 3]
#                 masked_albedo_chunk_rescaled = masked_albedo_chunk * rescale_value
#                 surface_brdf_relighting = masked_albedo_chunk_rescaled.unsqueeze(1).expand(-1, nlights, -1) / np.pi + specular_relighting # [surface_point_num, envW * envH, 3]
#                 direct_light = masked_light_rgb
#                 light_rgbs = visibility * direct_light  # [bs, envW * envH, 3]

                
#                 light_pix_contrib = surface_brdf_relighting * light_rgbs * cosine[:, :, None] / masked_light_pdf
#                 surface_relight_rgb_chunk  = torch.mean(light_pix_contrib, dim=1)  # [bs, 3]

#                 ### Tonemapping
#                 surface_relight_rgb_chunk = torch.clamp(surface_relight_rgb_chunk, min=0.0, max=1.0)  
#                 ### Colorspace transform
#                 if surface_relight_rgb_chunk.shape[0] > 0:
#                     surface_relight_rgb_chunk = linear2srgb_torch(surface_relight_rgb_chunk)
#                 relight_rgb_chunk[acc_chunk_mask] = surface_relight_rgb_chunk


#                 bg_color = envir_light.get_light(cur_light_name, rays_d_chunk) # [bs, 3]
#                 bg_color = torch.clamp(bg_color, min=0.0, max=1.0)
#                 bg_color = linear2srgb_torch(bg_color)
#                 relight_without_bg = torch.ones_like(bg_color)
#                 relight_with_bg = torch.ones_like(bg_color)
#                 relight_without_bg[acc_chunk_mask] = relight_rgb_chunk[acc_chunk_mask]
#                 acc_temp = acc_chunk[..., None]
#                 acc_temp[acc_temp <= 0.9] = 0.0
#                 relight_with_bg = acc_temp * relight_without_bg + (1.0 - acc_temp) * bg_color

#                 relight_pred_img_with_bg[cur_light_name].append(relight_with_bg.detach().clone().cpu())
#                 relight_pred_img_without_bg[cur_light_name].append(relight_without_bg.detach().clone().cpu())


#             acc_map_new.append(acc_map_chunk.cpu().detach())
#             rgb_map.append(rgb_chunk.cpu().detach())
#             depth_map.append(depth_chunk.cpu().detach())
#             acc_map.append(acc_chunk.cpu().detach())
#             normal_map.append(normal_chunk.cpu().detach())
#             # relight_rgb_map.append(relight_rgb_chunk.cpu().detach())
#             albedo_map.append(albedo_chunk.cpu().detach())
#             roughness_map.append(roughness_chunk.cpu().detach())

#         rgb_map = torch.cat(rgb_map, dim=0)
#         depth_map = torch.cat(depth_map, dim=0)
#         acc_map = torch.cat(acc_map, dim=0)
#         acc_map_new = torch.cat(acc_map_new, dim=0)
#         normal_map = torch.cat(normal_map, dim=0)
#         acc_map_mask = (acc_map > args.acc_mask_threshold)
#         albedo_map = torch.cat(albedo_map, dim=0)
#         roughness_map = torch.cat(roughness_map, dim=0)
#         acc_map_new = acc_map_new.reshape(H, W, 1).numpy()

#         os.makedirs(os.path.join(cur_dir_path, 'relighting_with_bg'), exist_ok=True)
#         os.makedirs(os.path.join(cur_dir_path, 'relighting_without_bg'), exist_ok=True)
#         os.makedirs(cur_dir_path, exist_ok=True)
#         for light_name_idx, cur_light_name in enumerate(dataset.light_names):
#             relight_map_with_bg = torch.cat(relight_pred_img_with_bg[cur_light_name], dim=0).reshape(H, W, 3).numpy()
#             relight_map_without_bg = torch.cat(relight_pred_img_without_bg[cur_light_name], dim=0).reshape(H, W, 3).numpy()

#             if args.if_save_relight_rgb:
#                 imageio.imwrite(os.path.join(cur_dir_path, 'relighting_with_bg', f'{cur_light_name}.png'), (relight_map_with_bg * 255).astype('uint8'))
#                 imageio.imwrite(os.path.join(cur_dir_path, 'relighting_without_bg', f'{cur_light_name}.png'), (relight_map_without_bg * 255).astype('uint8'))


#         rgb_map = (rgb_map.reshape(H, W, 3).numpy() * 255).astype('uint8')
#         rgb_frames_list.append(rgb_map)


#         if args.if_save_rgb:
#             cur_dir_path = os.path.join(args.geo_buffer_path, f'rgb')
#             imageio.imwrite(os.path.join(cur_dir_path, 'rgb_{:03d}.png'.format(idx)), rgb_map)


#             cur_dir_path = os.path.join(args.geo_buffer_path, f'albedo')
#             albedo_map = albedo_map * rescale_value.cpu()
#             albedo_map = albedo_map.clamp(0, 1)
#             albedo_map = albedo_map.reshape(H, W, 3)
#             # change to RGBA 4 channel with GT mask
#             albedo_map_to_save = np.concatenate([albedo_map, acc_map_new.reshape(H, W, 1)], axis=-1)
#             imageio.imwrite(os.path.join(cur_dir_path, 'albedo_{:03d}.png'.format(idx)), (albedo_map_to_save * 255).astype('uint8'))


#             normal_map_camera = normal_map @ w2c[:3, :3].T
#             normal_map_camera = normal_map_camera.reshape(H, W, 3)
#             normal_map_camera = normal_map_camera * 0.5 + 0.5
#             normal_map_camera = np.concatenate([normal_map_camera, acc_map_new.reshape(H, W, 1)], axis=-1)
#             cur_dir_path = os.path.join(args.geo_buffer_path, f'normal_camera')
#             imageio.imwrite(os.path.join(cur_dir_path, 'normal_camera_{:03d}.png'.format(idx)), (normal_map_camera * 255).astype('uint8'))

#             normal_map = normal_map.reshape(H, W, 3)
#             normal_map = normal_map * 0.5 + 0.5
#             normal_map_to_save = np.concatenate([normal_map, acc_map_new.reshape(H, W, 1)], axis=-1)
#             cur_dir_path = os.path.join(args.geo_buffer_path, f'normal')
#             imageio.imwrite(os.path.join(cur_dir_path, 'normal_{:03d}.png'.format(idx)), (normal_map_to_save * 255).astype('uint8'))
#             if args.if_save_albedo_gamma_corrected:
#                 # gamma cororection
#                 albedo_gamma = (albedo_map ** (1/2.2))
#                 albedo_gamma = np.concatenate([albedo_gamma, acc_map_new.reshape(H, W, 1)], axis=-1)
#                 cur_dir_path = os.path.join(args.geo_buffer_path, f'albedo_gamma')
#                 imageio.imwrite(os.path.join(cur_dir_path, 'albedo_gamma_{:03d}.png'.format(idx)), ( albedo_gamma * 255).astype('uint8'))
        
#             # save roughness map
#             roughness_map = roughness_map.reshape(H, W, 1)
#             # three channel
#             roughness_map_three = np.concatenate([roughness_map, roughness_map, roughness_map], axis=-1)
#             roughness_map = np.concatenate([roughness_map_three, acc_map_new.reshape(H, W, 1)], axis=-1)
#             cur_dir_path = os.path.join(args.geo_buffer_path, f'roughness')
#             imageio.imwrite(os.path.join(cur_dir_path, 'roughness_{:03d}.png'.format(idx)), (roughness_map * 255).astype('uint8'))
        

#     if args.render_video:
#         video_path = os.path.join(args.geo_buffer_path,'video')
#         os.makedirs(video_path, exist_ok=True)
        
#         to_render_list = ['albedo', 'normal', 'normal_camera', 'roughness', 'rgb', 'albedo_gamma']
        
#         for cur_render in to_render_list:
#             frame_list = []
#             for render_idx in range(test_rays.shape[0]):
#                 cur_dir_path = os.path.join(args.geo_buffer_path, f'{cur_render}')
#                 cur_image = imageio.v2.imread(os.path.join(cur_dir_path, f'{cur_render}_{render_idx:0>3d}.png'))
#                 if cur_image.shape[-1] == 4:
#                     alpha = cur_image[:, :, 3:] / 255
#                     cur_image = (cur_image[:, :, :3] * alpha + (1 - alpha) * 255).astype('uint8')
#                 frame_list.append(cur_image)
#             imageio.mimsave(os.path.join(video_path, f'{cur_render}_video.mp4'), np.stack(frame_list), fps=24, macro_block_size=1)

#         # imageio.mimsave(os.path.join(video_path, f'{cur_light_name}_video.mp4'), np.stack(frame_list), fps=24, macro_block_size=1)

#         video_path = os.path.join(args.geo_buffer_path,'video_without_bg')
#         os.makedirs(video_path, exist_ok=True)
        
#         for cur_light_name in dataset.light_names:
#             frame_list = []

#             for render_idx in range(test_rays.shape[0]):
#                 cur_dir_path = os.path.join(args.geo_buffer_path, "relighting", f'{dataset.split}_{render_idx:0>3d}', 'relighting_without_bg')
#                 frame_list.append(imageio.v2.imread(os.path.join(cur_dir_path, f'{cur_light_name}.png')))

#             imageio.mimsave(os.path.join(video_path, f'{cur_light_name}_video.mp4'), np.stack(frame_list), fps=24, macro_block_size=1)

#         video_path = os.path.join(args.geo_buffer_path,'video_with_bg')
#         os.makedirs(video_path, exist_ok=True)
        
#         for cur_light_name in dataset.light_names:
#             frame_list = []

#             for render_idx in range(test_rays.shape[0]):
#                 cur_dir_path = os.path.join(args.geo_buffer_path, "relighting", f'{dataset.split}_{render_idx:0>3d}', 'relighting_with_bg')
#                 frame_list.append(imageio.v2.imread(os.path.join(cur_dir_path, f'{cur_light_name}.png')))

#             imageio.mimsave(os.path.join(video_path, f'{cur_light_name}_video.mp4'), np.stack(frame_list), fps=24, macro_block_size=1)


@torch.no_grad()
def export_mesh(args):
    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensoIR = eval(args.model_name)(**kwargs)
    tensoIR.load(ckpt)

    alpha, _ = tensoIR.getDenseAlpha()
    convert_sdf_samples_to_ply(alpha.cpu(), f'{args.ckpt[:-3]}.ply', bbox=tensoIR.aabb.cpu(), level=0.005)



@torch.no_grad()
def test_on_synthesized_poses_real(dataset, args):

    if not os.path.exists(args.ckpt):
        print('the checkpoint path for tensoIR does not exists!!')
        return
        

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensoIR = eval(args.model_name)(**kwargs)
    tensoIR.load(ckpt)

    if args.ckpt_visibility is not None:
        visibility_net:nn.Module = eval(args.vis_model_name)().to(device)
        visibility_net.load_state_dict(torch.load(args.ckpt_visibility))
        visibility_net.requires_grad_(False) # freeze the visibility net
        print("load visibility network succcessfully")
    else:
        print('Not using visibility network')
    W, H = dataset.img_wh

    
    rgb_frames_list = []
    envir_light = Environment_Light(args.hdrdir)
    
    # use default light
    # envir_light = None

    #### 
    light_rotation_idx = 0
    ####


    rescale_value = torch.tensor([[1.8, 1.97, 1]], device=device) * 0.6
    
    # temp change output directory
    # args.geo_buffer_path = os.path.join(args.basedir, 'render_new_pose')
    datetime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    args.geo_buffer_path = os.path.join(os.path.dirname(args.ckpt), f'render_new_pose_{datetime}')
    os.makedirs(args.geo_buffer_path, exist_ok=True)
    
    print("*" * 80)
    print('The result will be saved in {}'.format(os.path.abspath(args.geo_buffer_path)))
    
    test_rays = dataset.test_rays   # [120, 96000, 6]
    test_w2c= dataset.test_w2c  # [120, 4, 4]

    # test_rays = dataset.all_rays
    # test_w2c= dataset.all_w2c
        
    # print(f'test_rays.shape: {test_rays.shape} test_w2c.shape {test_w2c.shape}')

    
    os.makedirs(os.path.join(args.geo_buffer_path, f'rgb'), exist_ok=True)
    os.makedirs(os.path.join(args.geo_buffer_path, f'albedo'), exist_ok=True)
    os.makedirs(os.path.join(args.geo_buffer_path, f'albedo_gamma'), exist_ok=True)
    os.makedirs(os.path.join(args.geo_buffer_path, f'normal'), exist_ok=True)
    os.makedirs(os.path.join(args.geo_buffer_path, f'depth'), exist_ok=True)
    os.makedirs(os.path.join(args.geo_buffer_path, f'roughness'), exist_ok=True)
    os.makedirs(os.path.join(args.geo_buffer_path, f'normal_camera'), exist_ok=True)
    os.makedirs(os.path.join(args.geo_buffer_path, f'relighting'), exist_ok=True)

    for idx in tqdm(range(test_rays.shape[0]), desc="relighting on synthesized poses"):
        relight_pred_img_with_bg, relight_pred_img_without_bg = dict(), dict()
        for cur_light_name in dataset.light_names:
            relight_pred_img_with_bg[f'{cur_light_name}'] = []
            relight_pred_img_without_bg[f'{cur_light_name}'] = []

        # os.makedirs(cur_dir_path, exist_ok=True)
        cur_dir_path = os.path.join(args.geo_buffer_path, 'relighting', f'{dataset.split}_{idx:0>3d}')
        os.makedirs(cur_dir_path, exist_ok=True)

        frame_rays = test_rays[idx].to(device) # [H*W, 6]
        w2c = test_w2c[idx]
        light_idx = torch.zeros((frame_rays.shape[0], 1), dtype=torch.int).to(device).fill_(light_rotation_idx)
        
        rgb_map, depth_map, normal_map, albedo_map, roughness_map, fresnel_map, relight_rgb_map, normals_diff_map, normals_orientation_loss_map = [], [], [], [], [], [], [], [], []
        acc_map_new = []
        acc_map = []

        chunk_idxs = torch.split(torch.arange(frame_rays.shape[0]), args.batch_size) # choose the first light idx
        for chunk_idx in chunk_idxs:
            with torch.enable_grad():
                rgb_chunk, depth_chunk, normal_chunk, albedo_chunk, roughness_chunk, \
                    fresnel_chunk, acc_chunk, *temp \
                    = tensoIR(frame_rays[chunk_idx], light_idx[chunk_idx], is_train=False, white_bg=True, ndc_ray=False, N_samples=-1)



            relight_rgb_chunk = torch.ones_like(rgb_chunk) 
            acc_map_chunk = acc_chunk
            acc_chunk_mask = (acc_chunk > args.acc_mask_threshold)
            
            rays_o_chunk, rays_d_chunk = frame_rays[chunk_idx][:, :3], frame_rays[chunk_idx][:, 3:]
            surface_xyz_chunk = rays_o_chunk + depth_chunk.unsqueeze(-1) * rays_d_chunk  # [bs, 3]
            masked_surface_pts = surface_xyz_chunk[acc_chunk_mask]      # [surface_point_num, 3]
            
            masked_normal_chunk = normal_chunk[acc_chunk_mask]          # [surface_point_num, 3]
            masked_albedo_chunk = albedo_chunk[acc_chunk_mask]          # [surface_point_num, 3]
            masked_roughness_chunk = roughness_chunk[acc_chunk_mask]    # [surface_point_num, 1]
            masked_fresnel_chunk = fresnel_chunk[acc_chunk_mask]        # [surface_point_num, 1]
            
            
            masked_fresnel_chunk = torch.ones_like(masked_fresnel_chunk) * torch.tensor([0.3, 0.3, 0.3]).cuda()
            masked_albedo_chunk = masked_albedo_chunk * torch.tensor([0.6, 0.6, 0.6]).cuda()
            # masked_albedo_chunk = torch.clamp(torch.tensor([0.75, 0.45, 0.35]).to(device) - masked_albedo_chunk, 0, 1)
            masked_roughness_chunk = 0.3 * torch.ones_like(masked_roughness_chunk)
            
            # # material editing
            # if idx < 20:
            #     masked_fresnel_chunk = torch.ones_like(masked_fresnel_chunk) * torch.tensor([0.3, 0.3, 0.3]).cuda()
            #     masked_albedo_chunk = masked_albedo_chunk * torch.tensor([0.6, 0.6, 0.6]).cuda()
            #     # masked_albedo_chunk = torch.clamp(torch.tensor([0.75, 0.45, 0.35]).to(device) - masked_albedo_chunk, 0, 1)
            #     masked_roughness_chunk = 0.3 * torch.ones_like(masked_roughness_chunk)

            # elif idx < 40:
            #     masked_fresnel_chunk = torch.ones_like(masked_fresnel_chunk) * 1
            #     masked_albedo_chunk = 0 * masked_albedo_chunk
            #     masked_roughness_chunk = 0.3 * torch.ones_like(masked_roughness_chunk)

            # elif idx < 60:
            #     masked_fresnel_chunk = masked_fresnel_chunk * 3
            #     masked_albedo_chunk = torch.clamp(torch.tensor([0.80, 0.50, 0.40]).to(device) - masked_albedo_chunk, 0, 1)
            #     # masked_roughness_chunk = 0.2 * torch.ones_like(masked_roughness_chunk)
            # else:
            #     masked_fresnel_chunk = torch.ones_like(masked_fresnel_chunk) * torch.tensor([0.4, 0.6, 0.4]).cuda()
            #     masked_albedo_chunk = 0 * masked_albedo_chunk
            #     masked_roughness_chunk = 0.2 * torch.ones_like(masked_roughness_chunk)
        

            ## Get incident light directions
            for light_name_idx, cur_light_name in enumerate(dataset.light_names):

                relight_rgb_chunk.fill_(1.0)
                masked_light_dir, masked_light_rgb, masked_light_pdf = envir_light.sample_light(cur_light_name, masked_normal_chunk.shape[0], 512) # [bs, envW * envH, 3]
                surf2l = masked_light_dir # [surface_point_num, envW * envH, 3]
                surf2c = -rays_d_chunk[acc_chunk_mask]  # [surface_point_num, 3]
                surf2c = safe_l2_normalize(surf2c, dim=-1)  # [surface_point_num, 3]


                ## get visibilty map from visibility network or compute it using density
                cosine = torch.einsum("ijk,ik->ij", surf2l, masked_normal_chunk)    # surf2l:[surface_point_num, envW * envH, 3] * masked_normal_chunk:[surface_point_num, 3] -> cosine:[surface_point_num, envW * envH]
                cosine_mask = (cosine > 1e-6)  # [surface_point_num, envW * envH] mask half of the incident light that is behind the surface
                visibility = torch.zeros((*cosine_mask.shape, 1), device=device)    # [surface_point_num, envW * envH, 1]
                masked_surface_xyz = masked_surface_pts[:, None, :].expand((*cosine_mask.shape, 3))  # [surface_point_num, envW * envH, 3]

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
                        nerv_vis, nerfactor_vis = compute_transmittance(tensoIR=tensoIR, 
                                                                        surf_pts=chunk_surface_pts, 
                                                                        light_in_dir=chunk_surf2light, 
                                                                        nSample=96, 
                                                                        vis_near=0.05,
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


                bg_color = envir_light.get_light(cur_light_name, rays_d_chunk) # [bs, 3]
                bg_color = torch.clamp(bg_color, min=0.0, max=1.0)
                bg_color = linear2srgb_torch(bg_color)
                relight_without_bg = torch.ones_like(bg_color)
                relight_with_bg = torch.ones_like(bg_color)
                relight_without_bg[acc_chunk_mask] = relight_rgb_chunk[acc_chunk_mask]
                acc_temp = acc_chunk[..., None]
                acc_temp[acc_temp <= 0.9] = 0.0
                relight_with_bg = acc_temp * relight_without_bg + (1.0 - acc_temp) * bg_color

                relight_pred_img_with_bg[cur_light_name].append(relight_with_bg.detach().clone().cpu())
                relight_pred_img_without_bg[cur_light_name].append(relight_without_bg.detach().clone().cpu())


            acc_map_new.append(acc_map_chunk.cpu().detach())
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
        acc_map_new = torch.cat(acc_map_new, dim=0)
        normal_map = torch.cat(normal_map, dim=0)
        acc_map_mask = (acc_map > args.acc_mask_threshold)
        albedo_map = torch.cat(albedo_map, dim=0)
        roughness_map = torch.cat(roughness_map, dim=0)
        acc_map_new = acc_map_new.reshape(H, W, 1).numpy()

        os.makedirs(os.path.join(cur_dir_path, 'relighting_with_bg'), exist_ok=True)
        os.makedirs(os.path.join(cur_dir_path, 'relighting_without_bg'), exist_ok=True)
        os.makedirs(cur_dir_path, exist_ok=True)
        for light_name_idx, cur_light_name in enumerate(dataset.light_names):
            relight_map_with_bg = torch.cat(relight_pred_img_with_bg[cur_light_name], dim=0).reshape(H, W, 3).numpy()
            relight_map_without_bg = torch.cat(relight_pred_img_without_bg[cur_light_name], dim=0).reshape(H, W, 3).numpy()

            if args.if_save_relight_rgb:
                imageio.imwrite(os.path.join(cur_dir_path, 'relighting_with_bg', f'{cur_light_name}.png'), (relight_map_with_bg * 255).astype('uint8'))
                imageio.imwrite(os.path.join(cur_dir_path, 'relighting_without_bg', f'{cur_light_name}.png'), (relight_map_without_bg * 255).astype('uint8'))


        rgb_map = (rgb_map.reshape(H, W, 3).numpy() * 255).astype('uint8')
        acc_map_mask = acc_map_mask.reshape(H, W).numpy()
        
        rgb_frames_list.append(rgb_map)


        if args.if_save_rgb:
            cur_dir_path = os.path.join(args.geo_buffer_path, f'rgb')
            # rgb_map[~acc_map_mask] = 255.
            imageio.imwrite(os.path.join(cur_dir_path, 'rgb_{:03d}.png'.format(idx)), rgb_map)

            # Save albedo
            cur_dir_path = os.path.join(args.geo_buffer_path, f'albedo')
            albedo_map = albedo_map * rescale_value.cpu()
            albedo_map = albedo_map.clamp(0, 1)
            albedo_map = albedo_map.reshape(H, W, 3)
            # change to RGBA 4 channel with GT mask
            albedo_map_to_save = np.concatenate([albedo_map, acc_map_new.reshape(H, W, 1)], axis=-1)
            # albedo_map_to_save[~acc_map_mask] = 1.0
            imageio.imwrite(os.path.join(cur_dir_path, 'albedo_{:03d}.png'.format(idx)), (albedo_map_to_save * 255).astype('uint8'))


            # Save depth
            cur_dir_path = os.path.join(args.geo_buffer_path, f'depth')
            depth_map = depth_map.reshape(H, W).cpu().numpy() * 1000
            cv2.imwrite(os.path.join(cur_dir_path, 'depth_{:03d}.png'.format(idx)), depth_map.astype('uint16'))
            
            normal_map_camera = normal_map @ w2c[:3, :3].T
            normal_map_camera = normal_map_camera.reshape(H, W, 3)
            normal_map_camera = normal_map_camera * 0.5 + 0.5
            normal_map_camera = np.concatenate([normal_map_camera, acc_map_new.reshape(H, W, 1)], axis=-1)
            cur_dir_path = os.path.join(args.geo_buffer_path, f'normal_camera')
            imageio.imwrite(os.path.join(cur_dir_path, 'normal_camera_{:03d}.png'.format(idx)), (normal_map_camera * 255).astype('uint8'))

            normal_map = normal_map.reshape(H, W, 3)
            normal_map = normal_map * 0.5 + 0.5
            normal_map_to_save = np.concatenate([normal_map, acc_map_new.reshape(H, W, 1)], axis=-1)
            cur_dir_path = os.path.join(args.geo_buffer_path, f'normal')
            imageio.imwrite(os.path.join(cur_dir_path, 'normal_{:03d}.png'.format(idx)), (normal_map_to_save * 255).astype('uint8'))
            if args.if_save_albedo_gamma_corrected:
                # gamma cororection
                albedo_gamma = (albedo_map ** (1/2.2))
                albedo_gamma = np.concatenate([albedo_gamma, acc_map_new.reshape(H, W, 1)], axis=-1)
                cur_dir_path = os.path.join(args.geo_buffer_path, f'albedo_gamma')
                imageio.imwrite(os.path.join(cur_dir_path, 'albedo_gamma_{:03d}.png'.format(idx)), ( albedo_gamma * 255).astype('uint8'))
        
            # save roughness map
            roughness_map = roughness_map.reshape(H, W, 1)
            # three channel
            roughness_map_three = np.concatenate([roughness_map, roughness_map, roughness_map], axis=-1)
            roughness_map = np.concatenate([roughness_map_three, acc_map_new.reshape(H, W, 1)], axis=-1)
            cur_dir_path = os.path.join(args.geo_buffer_path, f'roughness')
            imageio.imwrite(os.path.join(cur_dir_path, 'roughness_{:03d}.png'.format(idx)), (roughness_map * 255).astype('uint8'))
        

    if args.render_video:
        video_path = os.path.join(args.geo_buffer_path,'video')
        os.makedirs(video_path, exist_ok=True)
        
        to_render_list = ['albedo', 'normal', 'normal_camera', 'roughness', 'rgb', 'albedo_gamma']
        
        for cur_render in to_render_list:
            frame_list = []
            for render_idx in range(test_rays.shape[0]):
                cur_dir_path = os.path.join(args.geo_buffer_path, f'{cur_render}')
                cur_image = imageio.v2.imread(os.path.join(cur_dir_path, f'{cur_render}_{render_idx:0>3d}.png'))
                if cur_image.shape[-1] == 4:
                    alpha = cur_image[:, :, 3:] / 255
                    cur_image = (cur_image[:, :, :3] * alpha + (1 - alpha) * 255).astype('uint8')
                frame_list.append(cur_image)
            imageio.mimsave(os.path.join(video_path, f'{cur_render}_video.mp4'), np.stack(frame_list), fps=24, macro_block_size=1)

        # imageio.mimsave(os.path.join(video_path, f'{cur_light_name}_video.mp4'), np.stack(frame_list), fps=24, macro_block_size=1)

        video_path = os.path.join(args.geo_buffer_path,'video_without_bg')
        os.makedirs(video_path, exist_ok=True)
        
        for cur_light_name in dataset.light_names:
            frame_list = []

            for render_idx in range(test_rays.shape[0]):
                cur_dir_path = os.path.join(args.geo_buffer_path, "relighting", f'{dataset.split}_{render_idx:0>3d}', 'relighting_without_bg')
                frame_list.append(imageio.v2.imread(os.path.join(cur_dir_path, f'{cur_light_name}.png')))

            imageio.mimsave(os.path.join(video_path, f'{cur_light_name}_video.mp4'), np.stack(frame_list), fps=24, macro_block_size=1)

        video_path = os.path.join(args.geo_buffer_path,'video_with_bg')
        os.makedirs(video_path, exist_ok=True)
        
        for cur_light_name in dataset.light_names:
            frame_list = []

            for render_idx in range(test_rays.shape[0]):
                cur_dir_path = os.path.join(args.geo_buffer_path, "relighting", f'{dataset.split}_{render_idx:0>3d}', 'relighting_with_bg')
                frame_list.append(imageio.v2.imread(os.path.join(cur_dir_path, f'{cur_light_name}.png')))

            imageio.mimsave(os.path.join(video_path, f'{cur_light_name}_video.mp4'), np.stack(frame_list), fps=24, macro_block_size=1)
                
if __name__ == "__main__":
    args = config_parser()
    print(args)
    # print("*" * 80)
    # print('The result will be saved in {}'.format(os.path.abspath(args.geo_buffer_path)))


    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    torch.cuda.manual_seed_all(20211202)
    np.random.seed(20211202)


    # The following args are not defined in opt.py
    args.if_save_point_cloud = False
    args.if_save_rgb = True
    args.if_save_depth = False
    args.if_save_acc = True
    args.if_save_rgb_video = True
    args.if_save_relight_rgb = True
    args.if_save_albedo = True
    args.if_save_albedo_gamma_corrected = True
    args.acc_mask_threshold = 0.5
    args.if_predict_single_view_visibility = False # single view, with rotated OLAT, predict visibility from vis-net
    args.if_compute_single_view_visibility = False # single view, with rotated OLAT, compute the transimittance in NeRV as visibility 
    args.if_render_normal = False
    args.vis_equation = 'nerfactor'
    args.if_material_editing = True

    args.render_video = True

    dataset = dataset_dict[args.dataset_name]

    # light_name_list= ['bridge','city', 'courtyard', 'forest', 'fireplace', 'interior', 'museum', 'night', 'snow', 'square', 'studio',
    #                         'sunrise', 'sunset', 'tunnel']
    # light_name_list= ['bridge', 'city', 'fireplace', 'forest', 'night']

    # light_name_list= ['city', 'forest']
    # test_dataset = dataset(                            
    #                         args.datadir, 
    #                         args.hdrdir, 
    #                         split='test', 
    #                         random_test=False,
    #                         downsample=args.downsample_test,
    #                         light_names=light_name_list,
    #                         light_rotation=args.light_rotation,
    #                         scene_bbox=args.scene_bbox,
    #                         test_new_pose=True
    #                         )
    
    test_dataset = dataset(
                    args.datadir, 
                    args.hdrdir, 
                    split='train', 
                    downsample=args.downsample_test, 
                    light_name=args.light_name,
                    light_names=args.light_name_list,
                    light_rotation=args.light_rotation,
                    # scene_bbox=args.scene_bbox,
                    img_height=args.imgh,
                    img_width=args.imgw,
                    near=args.near,
                    far=args.far,
                    test_new_pose=True,
                    )
        
    # # export mesh
    # export_mesh(args)
    
    # import trimesh
    
    # mesh = trimesh.load('./mesh.ply')
    # print(np.mean(mesh.vertices, axis=0))
    
    
    test_on_synthesized_poses_real(test_dataset, args)
