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
import torch
from opt import config_parser
import cv2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from dataLoader import dataset_dict

from PIL import Image
from glob import glob

@torch.no_grad()
def eval_visibility(args):
    visibility_list = []
    # find file start with pred_lvis_olat_ in args.visibiliy_path
    pred_lvis_olat_files = glob(os.path.join(args.visibiliy_path, 'pred_lvis_olat_*'))
    pred_lvis_olat_files.sort()
    for file_path in pred_lvis_olat_files:
        # gray image
        input_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        # resize to 800 * 800
        input_img = cv2.resize(input_img, (800, 800))
        visibility_list.append(input_img)
    # save to video
    imageio.mimsave(os.path.join(args.save_path, 'pred_lvis_olat.mp4'), visibility_list, fps=24)


def generate_video(args):
    frame_list = []
   
    # pred_lvis_olat_files = glob(os.path.join(args.to_generate_path, 'nerfactor_relight_*'))
    pred_lvis_olat_files = glob(os.path.join(args.to_generate_path, 'normal*'))

    # pred_lvis_olat_files = glob(os.path.join(args.to_generate_path, '*', 'relight_rgb_only_indirect.png'))


    pred_lvis_olat_files.sort()
    pred_lvis_olat_files = pred_lvis_olat_files[:165]
    for file_path in pred_lvis_olat_files:
        input_img = imageio.imread(file_path)

        frame_list.append(input_img)
    # save to video
    imageio.mimsave(os.path.join(args.save_path, 'generated_video.mp4'), frame_list, fps=24)

def synthesize_video(args):

    final_frames = []
    video_list = []
    video_paths = os.listdir(args.to_synthesize_path).sort()

    for video_path in video_paths:
        cur_frames = []
        # read all the frames of the video
        video = imageio.get_reader(os.path.join(args.to_synthesize_path, video_path), 'ffmpeg')
        for frame in video:
            cur_frames.append(frame)
        video_list.append(cur_frames)
    frame_num = len(video_list[0])

    frame_interval = frame_num // len(video_list)

    for i in range(frame_num):
        cur_frame = video_list[i // frame_interval][i]
        final_frames.append(cur_frame)
    imageio.mimsave(os.path.join(args.save_path, 'synthesized_video.mp4'), final_frames, fps=24, macro_block_size=1)


def synthesize_video2(args):

    final_frames = []
    video_list = []
    video_paths = os.listdir(args.to_synthesize_path).sort()
    to_render = ['city_video', 'bridge_video', 'fireplace_video', 'forest_video', 'night_video']
    # read rotated video
    rotated_video = imageio.get_reader(os.path.join(args.to_synthesize_path, 'rotate_light.mp4'), 'ffmpeg')
    for frame in rotated_video:
        final_frames.append(frame)

    for video_path in to_render:
        cur_frames = []
        # read all the frames of the video
        video = imageio.get_reader(os.path.join(args.to_synthesize_path, video_path + '.mp4'), 'ffmpeg')
        for frame in video:
            cur_frames.append(frame)
        video_list.append(cur_frames)
    frame_num = len(video_list[0])

    frame_interval = frame_num // len(video_list)

    for i in range(frame_num):
        cur_frame = video_list[i // frame_interval][i]
        final_frames.append(cur_frame)
    imageio.mimsave(os.path.join(args.save_path, 'synthesized_video.mp4'), final_frames, fps=24, macro_block_size=1)



def generate_envir_map_video(args):
    frame_num = 200
    envir_map_list = os.listdir(args.envir_map_path)
    envir_map_list.sort()
    envir_map_num = len(envir_map_list)
    interval = frame_num / envir_map_num 
    frame_list = []
    for i in range(frame_num):
        envir_map_idx = int(i // interval)
        envir_map = imageio.imread(os.path.join(args.envir_map_path, envir_map_list[envir_map_idx]))
        envir_map = envir_map.clip(0, 1)
        # gamma correction
        envir_map = np.power(envir_map, 1/2.2)
        envir_map = (envir_map * 255).astype(np.uint8)
        # resize to 1024 * 512
        envir_map = cv2.resize(envir_map, (1024, 512))
        frame_list.append(envir_map)
    imageio.mimsave(os.path.join(args.save_path, 'envir_map_video.mp4'), frame_list, fps=24)

def generate_envir_map_video2(args):
    envir_map_list = os.listdir(args.envir_map_path)
    envir_map_list.sort()
    envir_map_num = len(envir_map_list)

    frame_list = []
    for i in range(envir_map_num):

        envir_map = imageio.imread(os.path.join(args.envir_map_path, envir_map_list[i]))
        envir_map = envir_map.clip(0, 1)
        # gamma correction
        envir_map = np.power(envir_map, 1/2.2)
        envir_map = (envir_map * 255).astype(np.uint8)
        # resize to 1024 * 512
        envir_map = cv2.resize(envir_map, (1024, 512))
        frame_list.append(envir_map)
    imageio.mimsave(os.path.join(args.save_path, 'envir_map_video.mp4'), frame_list, fps=24)


def generate_gt_video(args):
    frame_num = 200
    post_fix_list = ['bridge', 'city', 'fireplace', 'forest', 'night']
    # post_fix_list = ['normal']
    videos_list = []
    for post_fix in post_fix_list:
        cur_frame_list = []
        for image_idx in tqdm(range(frame_num)):
            # test_img_path = os.path.join(args.gt_rgb_path, f'test_{image_idx:03d}', f'normal.png')
            test_img_path = os.path.join(args.gt_rgb_path, f'test_{image_idx:03d}', f'rgba_{post_fix}.png')
            img = imageio.imread_v2(test_img_path )
            img_alpha = img[..., -1:] / 255.0
            relight_rgbs = img[..., :3] * img_alpha + (255 - img[..., -1:]) 
            relight_rgbs = relight_rgbs.clip(0, 255)
            relight_rgbs = relight_rgbs.astype(np.uint8)
            cur_frame_list.append(relight_rgbs)
        videos_list.append(cur_frame_list)
    
    for i, post_fix in enumerate(post_fix_list):
        imageio.mimsave(os.path.join(args.save_path, f'gt_video_{post_fix}.mp4'), videos_list[i], fps=24, macro_block_size=1)
            


if __name__ == "__main__":
    args = config_parser()
    print(args)
    args.visibiliy_path = '/home/haian/research/nerfactor/output/train/hotdog_nerfactor/lr5e-3/vis_test/ckpt-10/batch000000004'
    args.save_path = './'
    # eval_visibility(args)
    args.to_generate_path = '/home/haian/research/volsdf/evals/bmvs/rendering_1200'
    # generate_video(args)
    
    # args.to_synthesize_path = 'relighting'
    # synthesize_video2(args)
    
    args.envir_map_path = '/home/haian/research/blender/light_probes/high_res_envmaps_1k_128_bridge/'
    # generate_envir_map_video(args)
    # generate_envir_map_video2(args)
    args.gt_rgb_path = '/home/haian/research/blender/our_rendered_data_transparent/mouse/'
    # args.gt_rgb_path = '/home/haian/research/blender/our_rendered_data/lego'
    generate_gt_video(args)