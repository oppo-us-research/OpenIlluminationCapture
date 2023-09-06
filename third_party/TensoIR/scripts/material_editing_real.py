

import os
from tqdm import tqdm
import imageio
import numpy as np

from opt import config_parser
import torch
import torch.nn as nn
from utils import visualize_depth_numpy
from models.tensoRF_rotated_lights import raw2alpha, TensorVMSplit, AlphaGridMask
from dataLoader.ray_utils import safe_l2_normalize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from dataLoader import dataset_dict
from models.relight_utils import *
brdf_specular = specular_pipeline_render_multilight_new
from utils import rgb_ssim, rgb_lpips
from models.relight_utils import Environment_Light
from renderer import compute_rescale_ratio



@torch.no_grad()
def material_editing(dataset, args):

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

    #### 
    light_rotation_idx = 0
    ####
    # global_rescale_value_single, global_rescale_value_three = compute_rescale_ratio(tensoIR, dataset)
    # rescale_value = global_rescale_value_three
    rescale_value = torch.tensor([[1., 1., 1.]], device=device)
    rescale_value = torch.tensor([[1.8, 1.97, 1]], device=device) * 0.6
    test_rays = dataset.test_rays
    test_w2c= dataset.test_w2c

    for idx in tqdm(range(test_rays.shape[0]), desc="materials editing on synthesized poses"):
        material_editing_img = dict()
        for cur_light_name in dataset.light_names:
            material_editing_img[f'{cur_light_name}'] = []

        cur_dir_path = os.path.join(args.geo_buffer_path, f'{dataset.split}_{idx:0>3d}')
        os.makedirs(cur_dir_path, exist_ok=True)

        frame_rays = test_rays[idx].to(device) # [H*W, 6]
        w2c = test_w2c[idx]

        light_idx = torch.zeros((frame_rays.shape[0], 1), dtype=torch.int).to(device).fill_(light_rotation_idx)

        rgb_map, depth_map, normal_map, albedo_map, roughness_map, fresnel_map, relight_rgb_map, normals_diff_map, normals_orientation_loss_map = [], [], [], [], [], [], [], [], []
        acc_map = []


        chunk_idxs = torch.split(torch.arange(frame_rays.shape[0]), args.batch_size) # choose the first light idx
        for chunk_idx in chunk_idxs:
            with torch.enable_grad():
                rgb_chunk, depth_chunk, normal_chunk, albedo_chunk, roughness_chunk, \
                    fresnel_chunk, acc_chunk, *temp \
                    = tensoIR(frame_rays[chunk_idx], light_idx[chunk_idx], is_train=False, white_bg=True, ndc_ray=False, N_samples=-1)

            # # use gt normal to test
            # normal_chunk = gt_normal[chunk_idx].to(device)

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
        
            # material editing
            if idx < 50:
                masked_fresnel_chunk = torch.ones_like(masked_fresnel_chunk) * torch.tensor([0.3, 0.3, 0.3]).cuda()
                masked_albedo_chunk = masked_albedo_chunk * torch.tensor([0.6, 0.6, 0.6]).cuda()
                # masked_albedo_chunk = torch.clamp(torch.tensor([0.75, 0.45, 0.35]).to(device) - masked_albedo_chunk, 0, 1)
                masked_roughness_chunk = 0.3 * torch.ones_like(masked_roughness_chunk)

            elif idx < 100:
                masked_fresnel_chunk = torch.ones_like(masked_fresnel_chunk) * 1
                masked_albedo_chunk = 0 * masked_albedo_chunk
                masked_roughness_chunk = 0.3 * torch.ones_like(masked_roughness_chunk)

            elif idx < 150:
                masked_fresnel_chunk = masked_fresnel_chunk * 3
                masked_albedo_chunk = torch.clamp(torch.tensor([0.80, 0.50, 0.40]).to(device) - masked_albedo_chunk, 0, 1)
                # masked_roughness_chunk = 0.2 * torch.ones_like(masked_roughness_chunk)
            else:
                masked_fresnel_chunk = torch.ones_like(masked_fresnel_chunk) * torch.tensor([0.4, 0.6, 0.4]).cuda()
                masked_albedo_chunk = 0 * masked_albedo_chunk
                masked_roughness_chunk = 0.2 * torch.ones_like(masked_roughness_chunk)


            ## Get incident light directions
            for light_name_idx, cur_light_name in enumerate(dataset.light_names):
                material_editing_rgb_chunk.fill_(1.0)
                masked_light_dir, masked_light_rgb, masked_light_pdf = envir_light.sample_light(cur_light_name, masked_normal_chunk.shape[0], 512 * 10, sample_type='importance') # [bs, envW * envH, 3]
                surf2l = masked_light_dir # [surface_point_num, sampled_number, 3]
                surf2c = -rays_d_chunk[acc_chunk_mask]  # [surface_point_num, 3]
                surf2c = safe_l2_normalize(surf2c, dim=-1)  # [surface_point_num, 3]


                ## get visibilty map from visibility network or compute it using density
                cosine = torch.einsum("ijk,ik->ij", surf2l, masked_normal_chunk)    # surf2l:[surface_point_num, sampled_number, 3] * masked_normal_chunk:[surface_point_num, 3] -> cosine:[surface_point_num, envW * envH]
                cosine_mask = (cosine > 1e-6)  # [surface_point_num, sampled_number] mask half of the incident light that is behind the surface
                visibility = torch.zeros((*cosine_mask.shape, 1), device=device)    # [surface_point_num, sampled_number, 1]

                masked_surface_xyz = masked_surface_pts[:, None, :].expand((*cosine_mask.shape, 3))  # [surface_point_num, sampled_number, 3]

                cosine_masked_surface_pts = masked_surface_xyz[cosine_mask] # [num_of_vis_to_get, 3]
                cosine_masked_surf2l = surf2l[cosine_mask] # [num_of_vis_to_get, 3]
                cosine_masked_visibility = torch.zeros(cosine_masked_surf2l.shape[0], 1, device=device) # [num_of_vis_to_get, 1]

                chunk_idxs_vis = torch.split(torch.arange(cosine_masked_surface_pts.shape[0]), 160000)  

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

                specular_material_editing = brdf_specular(masked_normal_chunk, surf2c, surf2l, masked_roughness_chunk, masked_fresnel_chunk)  # [surface_point_num, envW * envH, 3]
                masked_albedo_chunk_rescaled = masked_albedo_chunk * rescale_value
                surface_brdf_material_editing = masked_albedo_chunk_rescaled.unsqueeze(1).expand(-1, nlights, -1) / np.pi + specular_material_editing # [surface_point_num, envW * envH, 3]
                direct_light = masked_light_rgb
                light_rgbs = visibility * direct_light  # [bs, sampled_number, 3]

                light_pix_contrib = surface_brdf_material_editing * light_rgbs * cosine[:, :, None] / masked_light_pdf
                surface_material_editiing_rgb_chunk  = torch.mean(light_pix_contrib, dim=1)  # [bs, 3]

                ### Tonemapping
                surface_material_editiing_rgb_chunk = torch.clamp(surface_material_editiing_rgb_chunk, min=0.0, max=1.0)  
                ### Colorspace transform
                if surface_material_editiing_rgb_chunk.shape[0] > 0:
                    surface_material_editiing_rgb_chunk= linear2srgb_torch(surface_material_editiing_rgb_chunk)
                material_editing_rgb_chunk[acc_chunk_mask] = surface_material_editiing_rgb_chunk
                material_editing_img[cur_light_name].append(material_editing_rgb_chunk.detach().clone().cpu())



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
        os.makedirs(os.path.join(cur_dir_path, 'material_editing'), exist_ok=True)
        for light_name_idx, cur_light_name in enumerate(dataset.light_names):
            material_editing_img_map = torch.cat(material_editing_img[cur_light_name], dim=0).reshape(H, W, 3).numpy()
            if args.if_save_relight_rgb:
                imageio.imwrite(os.path.join(cur_dir_path, 'material_editing', f'{cur_light_name}.png'), (material_editing_img_map * 255).astype('uint8'))


        rgb_map = (rgb_map.reshape(H, W, 3).numpy() * 255).astype('uint8')
        rgb_frames_list.append(rgb_map)


        if args.if_save_rgb:
            imageio.imwrite(os.path.join(cur_dir_path, 'rgb.png'), rgb_map)

    if args.if_save_rgb_video:
        video_path = os.path.join(args.geo_buffer_path,'video')
        os.makedirs(video_path, exist_ok=True)
        imageio.mimsave(os.path.join(video_path, 'rgb_video.mp4'), np.stack(rgb_frames_list), fps=24, quality=8)

        for cur_light_name in dataset.light_names:
            frame_list = []

            for render_idx in range(len(dataset)):
                cur_dir_path = os.path.join(args.geo_buffer_path, f'{dataset.split}_{render_idx:0>3d}', 'material_editing')
                frame_list.append(imageio.imread(os.path.join(cur_dir_path, f'{cur_light_name}.png')))

            imageio.mimsave(os.path.join(video_path, f'{cur_light_name}_video.mp4'), np.stack(frame_list), fps=24, quality=8)

    if args.render_video:
        video_path = os.path.join(args.geo_buffer_path,'video')
        os.makedirs(video_path, exist_ok=True)
        
        for cur_light_name in dataset.light_names:
            frame_list = []

            for render_idx in range(len(dataset)):
                cur_dir_path = os.path.join(args.geo_buffer_path, f'{dataset.split}_{render_idx:0>3d}', 'material_editing')
                frame_list.append(imageio.imread(os.path.join(cur_dir_path, f'{cur_light_name}.png')))

            imageio.mimsave(os.path.join(video_path, f'{cur_light_name}_video.mp4'), np.stack(frame_list), fps=24, quality=8)


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
    args.acc_mask_threshold = 0.5
    args.if_predict_single_view_visibility = False # single view, with rotated OLAT, predict visibility from vis-net
    args.if_compute_single_view_visibility = False # single view, with rotated OLAT, compute the transimittance in NeRV as visibility 
    args.if_render_normal = False
    args.vis_equation = 'nerfactor'
    args.if_material_editing = False

    args.render_video = True

    dataset = dataset_dict[args.dataset_name]

    # light_name_list= ['bridge','city', 'courtyard', 'forest', 'fireplace', 'interior', 'museum', 'night', 'snow', 'square', 'studio',
    #                         'sunrise', 'sunset', 'tunnel']
    light_name_list= ['sunset']

    test_dataset = dataset(                            
                            args.datadir, 
                            args.hdrdir, 
                            split='test', 
                            random_test=False,
                            downsample=args.downsample_test,
                            light_names=light_name_list,
                            light_rotation=args.light_rotation,
                            scene_bbox=args.scene_bbox,
                            test_new_pose=True
                            )
    material_editing(test_dataset , args)

    