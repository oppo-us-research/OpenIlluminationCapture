# Author: Isabella Liu
# Date: 2023-06-08
# Description: Calculate albedo from all MVC captures 

import argparse
import glob
import os
import os.path as osp
from multiprocessing import Pool

import cv2
import numpy as np
import rawpy
import tqdm
from tqdm import tqdm

# # Rotate list for MVC cameras
# ROTATE90_CLOCKWISE_LIST = [
#     'A1', 'A2', 'A4', 'A5', 'A6',
#     'B1', 'B2', 'B3', 'B4', 'B5', 
#     'C1', 
# ]

# ROTATE90_COUNTERCLOCKWISE_LIST = [
#     'A3',
#     'B6',
#     'C2', 'C3', 'C4', 'C5', 'C6',
#     'D1', 'D3', 'D4', 'D5', 'D6',
# ]

# Rotate list for MVC cameras
ROTATE90_CLOCKWISE_LIST = [
    'A1', 'A2', 'A4', 'A5', 'A6',
    'B2', 'B3', 'B4', 'B5',
    'C1',
]

ROTATE90_COUNTERCLOCKWISE_LIST = [
    'A3',
    'B6',
    'C2', 'C3', 'C4', 'C5', 'C6',
    'D1', 'D3', 'D4', 'D5', 'D6',
]

old_H, old_W = 4096, 3000


def read_raw(file_path):
    with rawpy.imread(file_path) as raw:
        image = raw.raw_image.copy()
        white_balance = np.array(raw.daylight_whitebalance, dtype=np.float32)
        black = np.reshape(np.array(raw.black_level_per_channel, dtype=image.dtype), (2, 2))
        black = np.tile(black, (image.shape[0] // 2, image.shape[1] // 2))
        # white = np.quantile(image, 0.9)
        white = raw.white_level
        image = np.maximum(image, black) - black
        image = cv2.demosaicing(image, code=cv2.COLOR_BAYER_BG2BGR)  # COLOR_BAYER_BG2BGR)
        image = image / (white - black)[..., np.newaxis].astype(np.float32)
        # image = image / (raw.white_level - black)[...,np.newaxis].astype(np.float32)
        # image = image / (white - black).astype(np.float32)
        # image = image / (raw.white_level - black)[...,np.newaxis].astype(np.float32)
        image = cv2.resize(image, (H, W))
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        # image = cv2.resize(image, (H, W)).transpose(1,0,2)

    image = image * white_balance[:3]

    # image_vis = (image * 5)**(1/2.2)
    # image_vis = image_vis[..., ::-1]

    return image[None]


def read_jpeg(file_path):
    jpg_image = cv2.imread(file_path)
    # jpg_image = cv2.resize(jpg_image, (1000,1500))
    # jpg_image = cv2.resize(jpg_image, (1500,1000)) # MVC
    jpg_image = cv2.resize(jpg_image, (old_H // 4, old_W // 4))  # MVC

    # jpg_img_vis = jpg_image[..., ::-1]
    # jpg_img_vis_gamma = jpg_img_vis**(2.2)
    return jpg_image[None]


def calc_albedo_normal(cam_id, root_dir, use_raw, out_dir, light_dirs):
    all_images = []
    datas = []
    obj_path = os.path.join(root_dir, cam_id)

    img_ignore_list = ['142']
    for file in tqdm(glob.glob(obj_path + '/*.jpg')):
        if file.split('/')[-1].split('.')[0] in img_ignore_list:
            continue
        datas.append(file)
    light_dirs = np.vstack(light_dirs)
    read_func = read_raw if use_raw else read_jpeg
    with Pool() as p:
        all_images = list(tqdm(p.imap(read_func, datas), total=len(datas)))
    all_images = np.vstack(all_images)  # [N, H, W, 3]

    H, W = all_images.shape[1], all_images.shape[2]

    Albedo_img_res = np.zeros((H, W, 3))  # [H, W, 3]
    N_res = np.zeros((H * W, 3))  # [H * W, 3]

    for channel in range(3):
        print(f'Processing channel {channel}')

        img_array_r = all_images[:, :, :, channel]  # [N, H, W]
        img_array_flatten_r = img_array_r.reshape((img_array_r.shape[0], -1))  # [N, H * W]

        N = np.linalg.lstsq(light_dirs, img_array_flatten_r, rcond=None)[0].T  # [H * W, 3]

        albedo_cur = np.linalg.norm(N, axis=1)
        zero_mask = np.all(N == 0, axis=-1)  # [H * W, ]
        albedo_cur[zero_mask] = 0
        albedo_cur = albedo_cur.reshape((H, W))  # [H, W, ]
        Albedo_img_res[:, :, channel] = albedo_cur

        N_real = N / np.linalg.norm(N, axis=-1, keepdims=True)  # normalize to account for diffuse reflectance
        N_res[~zero_mask] += N_real[~zero_mask]

    N_res = N_res / 3.0

    # # Save normal results
    # N_img_res = (N_res.reshape(H, W, 3) + 1.0)/ 2.0
    # N_img_res *= 255.0
    # # cv2.imwrite(os.path.join(RES_PATH, 'normal.png'), cv2.cvtColor(N_img_res.astype(np.uint8), cv2.COLOR_BGR2RGB))
    # cv2.imwrite(os.path.join(out_dir, 'normal.png'), N_img_res.astype(np.uint8))
    # np.save(os.path.join(out_dir, 'normal.npy'), N_res.reshape((H, W, 3)))

    # Save albedo results
    Albedo_img_res = (Albedo_img_res - Albedo_img_res.min()) / (Albedo_img_res.max() - Albedo_img_res.min())
    if use_raw:
        albedo_scale = 2.0
        Albedo_img_res = (Albedo_img_res * albedo_scale) ** (1.0 / 2.2)  # gamma correction
        Albedo_img_res = np.clip(Albedo_img_res, 0, 1)
    Albedo_img_res *= 255
    # np.save(os.path.join(out_dir, 'albedo.npy'), Albedo_img_res)
    # cv2.imwrite(os.path.join(RES_PATH, 'albedo.png'), cv2.cvtColor(Albedo_img_res.astype(np.uint8), cv2.COLOR_BGR2RGB))

    if cam_id in ROTATE90_CLOCKWISE_LIST:
        Albedo_img_res = cv2.rotate(Albedo_img_res, cv2.ROTATE_90_CLOCKWISE)
    elif cam_id in ROTATE90_COUNTERCLOCKWISE_LIST:
        Albedo_img_res = cv2.rotate(Albedo_img_res, cv2.ROTATE_90_COUNTERCLOCKWISE)

    cv2.imwrite(os.path.join(out_dir, f'{cam_id}.png'), Albedo_img_res.astype(np.uint8))


def calc_albedo_normal_pool(datas):
    cam_id, root_dir, use_raw, out_dir, light_dir = datas
    all_images = []
    datas = []
    obj_path = os.path.join(root_dir, cam_id)

    img_ignore_list = ['142']
    light_dirs = []
    for file in glob.glob(obj_path + '/*.jpg'):
        if file.split('/')[-1].split('.')[0] in img_ignore_list:
            continue
        datas.append(file)
        light_idx = int(file.split('/')[-1].split('.')[0])
        light_dirs += [light_dir[light_idx]]
    read_func = read_raw if use_raw else read_jpeg

    for file in datas:
        all_images.append(read_func(file))

    all_images = np.vstack(all_images)  # [N, H, W, 3]

    H, W = all_images.shape[1], all_images.shape[2]

    Albedo_img_res = np.zeros((H, W, 3))  # [H, W, 3]
    N_res = np.zeros((H * W, 3))  # [H * W, 3]

    for channel in range(3):
        print(f'Processing channel {channel}')

        img_array_r = all_images[:, :, :, channel]  # [N, H, W]
        img_array_flatten_r = img_array_r.reshape((img_array_r.shape[0], -1))  # [N, H * W]

        N = np.linalg.lstsq(light_dirs, img_array_flatten_r, rcond=None)[0].T  # [H * W, 3]

        albedo_cur = np.linalg.norm(N, axis=1)
        zero_mask = np.all(N == 0, axis=-1)  # [H * W, ]
        albedo_cur[zero_mask] = 0
        albedo_cur = albedo_cur.reshape((H, W))  # [H, W, ]
        Albedo_img_res[:, :, channel] = albedo_cur

        N_real = N / np.linalg.norm(N, axis=-1, keepdims=True)  # normalize to account for diffuse reflectance
        N_res[~zero_mask] += N_real[~zero_mask]

    N_res = N_res / 3.0

    # Save normal results
    N_img_res = (N_res.reshape(H, W, 3) + 1.0) / 2.0
    N_img_res = cv2.resize(N_img_res, (old_H, old_W), interpolation=cv2.INTER_CUBIC)
    N_img_res *= 255.0

    # Save albedo results
    Albedo_img_res = (Albedo_img_res - Albedo_img_res.min()) / (Albedo_img_res.max() - Albedo_img_res.min())
    Albedo_img_res = cv2.resize(Albedo_img_res, (old_H, old_W), interpolation=cv2.INTER_CUBIC)
    if use_raw:
        albedo_scale = 2.0
        Albedo_img_res = (Albedo_img_res * albedo_scale) ** (1.0 / 2.2)  # gamma correction
        Albedo_img_res = np.clip(Albedo_img_res, 0, 1)
    Albedo_img_res *= 255

    if cam_id in ROTATE90_CLOCKWISE_LIST:
        Albedo_img_res = cv2.rotate(Albedo_img_res, cv2.ROTATE_90_CLOCKWISE)
        N_img_res = cv2.rotate(N_img_res, cv2.ROTATE_90_CLOCKWISE)

    elif cam_id in ROTATE90_COUNTERCLOCKWISE_LIST:
        Albedo_img_res = cv2.rotate(Albedo_img_res, cv2.ROTATE_90_COUNTERCLOCKWISE)
        N_img_res = cv2.rotate(N_img_res, cv2.ROTATE_90_COUNTERCLOCKWISE)

    np.save(os.path.join(out_dir, 'Normal_npy', f'{cam_id}.npy'), N_res.reshape((H, W, 3)))
    # Upsample to original resolution
    cv2.imwrite(os.path.join(out_dir, 'Normal_png', f'{cam_id}.png'), N_img_res.astype(np.uint8))
    cv2.imwrite(os.path.join(out_dir, 'Albedo_png', f'{cam_id}.png'), Albedo_img_res.astype(np.uint8))


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--root_dir', type=str, default=None, required=True, help='Root directory to the images')
    argparser.add_argument('--light_dir', type=str, default='/nfs/STG/HumanAction/lab150/UCSD_other_data/NeuRIPS_2023_OpenIlumination/light_pos.npy', help='Path to light direction file')
    argparser.add_argument('--use_raw', type=bool, default=False, help='Whether to use raw images')
    argparser.add_argument('--out_dir', type=str, default='ps_results', help='Path to reconstruction results under the root dir')
    args = argparser.parse_args()

    all_cam_list = ROTATE90_CLOCKWISE_LIST + ROTATE90_COUNTERCLOCKWISE_LIST

    # Read light directions from numpy file
    light_dirs = np.load(args.light_dir)

    # folder_list =  [glob.glob(osp.join(args.root_dir, '*_obj_*'))]
    folder_list = [sorted(glob.glob(osp.join(args.root_dir, '*_obj*')))[-1]]

    for folder in folder_list:
        print(f'Processing {folder}')

        out_dir = osp.join(folder, 'ps_results')
        out_dir += '_raw' if args.use_raw else '_jpg'
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(osp.join(out_dir, 'Normal_npy'), exist_ok=True)
        os.makedirs(osp.join(out_dir, 'Normal_png'), exist_ok=True)
        os.makedirs(osp.join(out_dir, 'Albedo_png'), exist_ok=True)

        datas = [(cam_id, folder, args.use_raw, out_dir, light_dirs) for cam_id in all_cam_list]

        with Pool() as p:
            all_images = list(tqdm(p.imap(calc_albedo_normal_pool, datas), total=len(datas)))

        # calc_albedo_normal_pool(datas[0])

    # for cam_id in tqdm(all_cam_list):
    #     calc_albedo_normal(cam_id, args.root_dir, args.use_raw, args.out_dir, light_dirs)


if __name__ == '__main__':
    main()
