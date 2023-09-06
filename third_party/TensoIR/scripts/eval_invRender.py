'''
Author: Haian-Jin 3190106083@zju.edu.cn
Date: 2022-08-03 20:37:43
LastEditors: Haian-Jin 3190106083@zju.edu.cn
LastEditTime: 2022-09-05 14:43:27
FilePath: /TensoRFactor/scripts/see_visibilty.py
Description: see the visibilty of the model and export a video to visualize the visibilty
'''

from utils import rgb_ssim, rgb_lpips
import os
from tqdm import tqdm
import imageio
import numpy as np
import cv2
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

from PIL import Image




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
def eval_nerfactor(dataset, args):

    envir_light = Environment_Light(args.hdrdir)
    rgb_frames_list = []
    MAE_list = []
    albedo_psnr_list, albedo_ssim_list, albedo_l_vgg_list, albedo_l_alex_list= [], [], [], []
    rgb_psnr_list, rgb_ssim_list, rgb_l_vgg_list, rgb_l_alex_list= [], [], [], []
    
    relight_psnr_dict = {}
    relight_l_alex_dict, relight_l_vgg_dict, relight_ssim_dict = dict(), dict(), dict() 
    for cur_light_name in dataset.light_names:
        relight_psnr_dict[f'{cur_light_name}'] = []
        relight_l_alex_dict[f'{cur_light_name}'] = []
        relight_l_vgg_dict[f'{cur_light_name}'] = []
        relight_ssim_dict[f'{cur_light_name}'] = []
 
    for idx in tqdm(range(len(dataset))):
    # for idx in tqdm(range(2)):
        cur_nerfactor_save_path = os.path.join(result_cache_path, 'test_{:0>3d}'.format(idx))
        if not os.path.exists(cur_nerfactor_save_path):
            os.makedirs(cur_nerfactor_save_path)
        item = dataset[idx]
        gt_rgb_path = os.path.join(GT_rgb_root_path,'test_{:0>3d}'.format(idx), 'rgba_sunset_000.png')
        # read in RGB
        gt_rgb = dataset.transform(Image.open(gt_rgb_path)).view(4, -1).permute(1, 0)  # [H*W, 4]
        gt_alpha = gt_rgb[:, 3].unsqueeze(-1) # [H*W, 1] / 255
        gt_alpha = gt_alpha.reshape(800, 800, 1).numpy()
        gt_rgb = gt_rgb[..., :3] * gt_rgb[..., -1:] + (1 - gt_rgb[..., -1:]) 
        gt_rgb = gt_rgb.reshape(800, 800, 3)
        imageio.imsave(os.path.join(cur_nerfactor_save_path, 'gt_rgb.png'), gt_rgb)



        gt_normal = item['normals'].squeeze(0).cpu().reshape(800, 800, 3) # [H*W, 3]
        gt_normal_white = item['normals_white'].squeeze(0).cpu().reshape(800, 800, 3) # [H*W, 3]
        gt_normal_white = (gt_normal_white + 1.0) / 2.0
        gt_albedo = item['albedo'].squeeze(0).cpu().reshape(800, 800, 3)  # [H*W, 3]
        gt_mask = item['rgbs_mask'].squeeze(0).cpu().reshape(800, 800) # [H*W]
        gt_rays = item['rays'].squeeze(0).cpu().reshape(800, 800, 6) # [H*W, 3]
        rayd = gt_rays[...,3:]
        # gt_rgb_all = item['rgbs'].cpu() # [N, H*W, 3]
        gt_rgb_all = item['rgbs'].squeeze(0).cpu() # [N, H*W, 3]

        gt_albedo_gamma_corrected = torch.pow(gt_albedo, 1/2.2)
        imageio.imsave(os.path.join(cur_nerfactor_save_path, 'gt_albedo.png'), np.array(gt_albedo_gamma_corrected * 255, dtype=np.uint8))
        imageio.imsave(os.path.join(cur_nerfactor_save_path, 'gt_normal.png'), np.array(gt_normal_white * 255, dtype=np.uint8))

        nerfactor_item_path = 'batch000000{:0>3d}'.format(idx)
        nerfactor_albedo_path = os.path.join(args.nerfactor_path, nerfactor_item_path, 'pred_albedo.png')
        nerfactor_normal_path = os.path.join(args.nerfactor_path, nerfactor_item_path, 'pred_normal.png')
        nerfactor_rgb_path = os.path.join(args.nerfactor_path, nerfactor_item_path, 'pred_rgb.png')
        nerfactor_alpha_path = os.path.join(args.nerfactor_path, nerfactor_item_path, 'gt_alpha.png')
        nerfactor_brdf_path = os.path.join(args.nerfactor_path, nerfactor_item_path, 'pred_brdf.png')
        # read albedo, rgb and normal; resize to 800*800; and save
        nerfactor_albedo = imageio.v2.imread(nerfactor_albedo_path)
        nerfactor_albedo = cv2.resize(nerfactor_albedo, (800, 800))
        nerfactor_rgb = imageio.v2.imread(nerfactor_rgb_path)
        nerfactor_rgb = cv2.resize(nerfactor_rgb, (800, 800))
        
        nerfactor_normal = imageio.v2.imread(nerfactor_normal_path)
        nerfactor_normal = cv2.resize(nerfactor_normal, (800, 800))
        nerfactor_alpha = imageio.v2.imread(nerfactor_alpha_path)
        nerfactor_alpha = cv2.resize(nerfactor_alpha, (800, 800)) / 255.0

        nerfactor_brdf = imageio.v2.imread(nerfactor_brdf_path)
        nerfactor_brdf = cv2.resize(nerfactor_brdf, (800, 800))

        nerfactor_rgb = np.concatenate([nerfactor_rgb, nerfactor_alpha[..., None]], axis=-1)
        nerfactor_albedo = np.concatenate([nerfactor_albedo, nerfactor_alpha[..., None]], axis=-1)
        imageio.imsave(os.path.join(cur_nerfactor_save_path, 'nerfactor_albedo.png'), nerfactor_albedo)
        imageio.imsave(os.path.join(cur_nerfactor_save_path, 'nerfactor_rgb.png'), nerfactor_rgb)
        imageio.imsave(os.path.join(cur_nerfactor_save_path, 'nerfactor_normal.png'), nerfactor_normal)
        imageio.imsave(os.path.join(cur_nerfactor_save_path, 'nerfactor_alpha.png'), nerfactor_alpha)
        imageio.imsave(os.path.join(cur_nerfactor_save_path, 'nerfactor_brdf.png'), nerfactor_brdf)
        nerfactor_albedo = nerfactor_albedo.astype(np.float32) / 255.0
        nerfactor_albedo[gt_mask == 0] = 1
        nerfactor_rgb = nerfactor_rgb.astype(np.float32) / 255.0
        nerfactor_normal = torch.from_numpy(nerfactor_normal.astype(np.float32) / 255.0 * 2 - 1)
        nerfactor_normal[nerfactor_alpha == 0] = torch.tensor([0, 0, 1.0])
        visualized_normal = nerfactor_normal * 0.5 + 0.5
        visualized_normal = np.concatenate((visualized_normal.numpy() * 255, nerfactor_alpha[..., None]), axis=-1)
        imageio.imsave(os.path.join(cur_nerfactor_save_path, 'nerfactor_normal_visualized.png'), visualized_normal)
        
        MAE = np.mean(np.arccos(np.clip(np.sum(nerfactor_normal.numpy() * gt_normal.numpy(), axis=-1), -1, 1)) * 180 / np.pi)
        MAE_list.append(MAE)

        # # albedo metrics
        # loss_albedo = torch.mean((gt_albedo_gamma_corrected - nerfactor_albedo) ** 2).item()
        # cur_psnr_albedo = -10.0 * np.log(loss_albedo) / np.log(10.0)
        # # device='cpu'
        # ssim_albedo = rgb_ssim(gt_albedo_gamma_corrected, nerfactor_albedo, 1)
        # l_a_albedo= rgb_lpips(gt_albedo_gamma_corrected.numpy(), nerfactor_albedo, 'alex', device=device)
        # l_v_albedo = rgb_lpips(gt_albedo_gamma_corrected.numpy(), nerfactor_albedo, 'vgg', device=device)
        # albedo_psnr_list.append(cur_psnr_albedo)
        # albedo_ssim_list.append(ssim_albedo)
        # albedo_l_alex_list.append(l_a_albedo)
        # albedo_l_vgg_list.append(l_v_albedo)
        
        # # rgb metrics
        # loss_rgb = torch.mean((gt_rgb - nerfactor_rgb) ** 2).item()
        # cur_psnr_rgb = -10.0 * np.log(loss_rgb) / np.log(10.0)
        # ssim_rgb = rgb_ssim(gt_rgb, nerfactor_rgb, 1)
        # l_a_rgb = rgb_lpips(gt_rgb.numpy(), nerfactor_rgb, 'alex', device=device)
        # l_v_rgb = rgb_lpips(gt_rgb.numpy(), nerfactor_rgb, 'vgg', device=device)
        # rgb_psnr_list.append(cur_psnr_rgb)
        # rgb_ssim_list.append(ssim_rgb)
        # rgb_l_alex_list.append(l_a_rgb)
        # rgb_l_vgg_list.append(l_v_rgb)
        # print('albedo psnr: {:.4f}, ssim: {:.4f}, l_a: {:.4f}, l_v: {:.4f}'.format(cur_psnr_albedo, ssim_albedo, l_a_albedo, l_v_albedo))
        # print('rgb psnr: {:.4f}, ssim: {:.4f}, l_a: {:.4f}, l_v: {:.4f}'.format(cur_psnr_rgb, ssim_rgb, l_a_rgb, l_v_rgb))
        # print('MAE: {:.4f}'.format(MAE))

        for light_idx, cur_light_name in enumerate(dataset.light_names):
            nerfactor_relight_path = os.path.join(args.nerfactor_path, nerfactor_item_path, 'pred_rgb_probes_{}.png'.format(cur_light_name))
            nerfactor_relight = imageio.v2.imread(nerfactor_relight_path)
            nerfactor_relight = cv2.resize(nerfactor_relight, (800, 800))
      
            nerfactor_relight = nerfactor_relight.astype(np.float32) / 255.0
            nerfactor_relight[nerfactor_alpha == 0] = 1.0
            gt_relight = gt_rgb_all[light_idx].numpy().reshape(800, 800, 3)
            
            background_color = envir_light.get_light(cur_light_name, rayd.reshape(-1, 3).cuda()) # [H, W, 3]
            background_color = background_color.cpu().numpy().reshape(800, 800, 3).clip(0, 1)
            background_color = background_color ** (1 / 2.2)
            nerfactor_relight = nerfactor_relight * nerfactor_alpha[..., None] + background_color * (1 - nerfactor_alpha[..., None])
            gt_relight = gt_relight * gt_alpha + background_color * (1 - gt_alpha)
            imageio.imsave(os.path.join(cur_nerfactor_save_path, 'nerfactor_relight_{}.png'.format(cur_light_name)), nerfactor_relight)
            imageio.imsave(os.path.join(cur_nerfactor_save_path, 'gt_relight_{}.png'.format(cur_light_name)), gt_relight)
            # # relight metrics
            # loss_relight = np.mean((gt_relight - nerfactor_relight) ** 2)
            # cur_psnr_relight = -10.0 * np.log(loss_relight) / np.log(10.0)
            # ssim_relight = rgb_ssim(gt_relight, nerfactor_relight, 1)
            # l_a_relight = rgb_lpips(gt_relight, nerfactor_relight, 'alex', device=device)
            # l_v_relight = rgb_lpips(gt_relight, nerfactor_relight, 'vgg', device=device)
            # relight_psnr_dict[cur_light_name].append(cur_psnr_relight)
            # relight_ssim_dict[cur_light_name].append(ssim_relight)
            # relight_l_alex_dict[cur_light_name].append(l_a_relight)
            # relight_l_vgg_dict[cur_light_name].append(l_v_relight)
            # print('relight {} psnr: {:.4f}, ssim: {:.4f}, l_a: {:.4f}, l_v: {:.4f}'.format(cur_light_name, cur_psnr_relight, ssim_relight, l_a_relight, l_v_relight))
    # write avg metrics to txt
    # with open(os.path.join(result_cache_path, 'metrics.txt'), 'w') as f:
    #     f.write('mae: {:.4f}\n'.format(np.mean(MAE_list)))
    #     f.write('albedo psnr: {:.4f}, ssim: {:.4f}, l_a: {:.4f}, l_v: {:.4f}\n'.format(np.mean(albedo_psnr_list), np.mean(albedo_ssim_list), np.mean(albedo_l_alex_list), np.mean(albedo_l_vgg_list)))
    #     f.write('rgb psnr: {:.4f}, ssim: {:.4f}, l_a: {:.4f}, l_v: {:.4f}\n'.format(np.mean(rgb_psnr_list), np.mean(rgb_ssim_list), np.mean(rgb_l_alex_list), np.mean(rgb_l_vgg_list)))
    #     for light_idx, cur_light_name in enumerate(dataset.light_names):
    #         f.write('relight {} psnr: {:.4f}, ssim: {:.4f}, l_a: {:.4f}, l_v: {:.4f}\n'.format(cur_light_name, np.mean(relight_psnr_dict[cur_light_name]), np.mean(relight_ssim_dict[cur_light_name]), np.mean(relight_l_alex_dict[cur_light_name]), np.mean(relight_l_vgg_dict[cur_light_name])))


    # # render video
    # # to_render = ['nerfactor_brdf', 'nerfactor_albedo', 'nerfactor_normal', 'nerfactor_relight_bridge', 'nerfactor_relight_fireplace', 'nerfactor_relight_forest', 'nerfactor_relight_night', 'nerfactor_relight_city']
    # # for name in to_render:
    # #     video_path = os.path.join(result_cache_path, '{}.mp4'.format(name))
    # #     frame_list = []
    # #     for idx in tqdm(range(len(dataset))):
    # #         cur_nerfactor_save_path = os.path.join(result_cache_path, 'test_{:0>3d}'.format(idx))
    # #         item_path = os.path.join(cur_nerfactor_save_path, '{}.png'.format(name))
    # #         image = imageio.v2.imread(item_path)
    # #         frame_list.append(image)
    # #     imageio.mimsave(video_path, frame_list, fps=24) 
    # # to_render = ['gt_albedo', 'gt_normal', 'gt_relight_bridge', 'gt_relight_fireplace', 'gt_relight_forest', 'gt_relight_night', 'gt_relight_city']
    
    # to_render = ['gt_rgb', 'nerfactor_rgb']

    # for name in to_render:
    #     video_path = os.path.join(result_cache_path, '{}.mp4'.format(name))
    #     frame_list = []
    #     for idx in tqdm(range(len(dataset))):
    #         cur_nerfactor_save_path = os.path.join(result_cache_path, 'test_{:0>3d}'.format(idx))
    #         item_path = os.path.join(cur_nerfactor_save_path, '{}.png'.format(name))
    #         image = imageio.v2.imread(item_path)
    #         frame_list.append(image)
    #     imageio.mimsave(video_path, frame_list, fps=24) 

if __name__ == "__main__":
    args = config_parser()
    print(args)
    print("*" * 80)
    print('The result will be saved in {}'.format(os.path.abspath(args.geo_buffer_path)))

    # sunset_path 
    args.nerfactor_path = '/home/haian/research/nerfactor/output/train/lego_nerfactor/lr5e-3/vis_test/ckpt-10'
    result_cache_path = './nerfactor_cache4/lego_rotate'
    if not os.path.exists(result_cache_path):
        os.makedirs(result_cache_path)
    GT_rgb_root_path = '/home/haian/research/blender/our_rendered_data/lego'
    dataset = dataset_dict[args.dataset_name]
    light_name_list= ['bridge', 'city', 'fireplace', 'forest', 'night']
    light_name_list = [f'bridge_{idx:03d}' for idx in range(128)]
    args.light_name_list = light_name_list
    # light_name_list= ['city']
    test_dataset = dataset(                            
                            args.datadir, 
                            args.hdrdir, 
                            split='test', 
                            random_test=False,
                            downsample=args.downsample_test,
                            light_names=light_name_list,
                            light_rotation=args.light_rotation,
                            sub=1
                            )
    eval_nerfactor(test_dataset , args)

    