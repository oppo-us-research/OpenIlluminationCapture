#  created by Isabella Liu (lal005@ucsd.edu) at 2023/03/13 20:21.
#
#  Util functions
import os
import os
import os.path as osp

import os, sys
import random

import loguru
import numpy as np
import json
from tqdm import tqdm
from ltsg.utils.camera_io import read_cameras


def remove_exif_transpose(img_path):
    """ Read image from img_path and remove the exif rotation and save other exif parameters

    Args:
        img_path (str): Path of the image
    """
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(Image.open(img_path))
    img_exif = img.getexif()
    img.save(img_path, exif=img_exif)


def rm_exif_transpose_folder(folder):
    """ Remove EXIF transpose tag from all images in the folder
    Args:
        folder (str): Path to the folder
    """
    for file in os.listdir(folder):
        if file.lower().endswith('.jpg') or file.lower().endswith('.png'):
            remove_exif_transpose(os.path.join(folder, file))


def create_folder_for_images(folder):
    """ Create a folder for each image in the folder
    Args:
        folder (str): Path to the folder
    """
    for file in tqdm(os.listdir(folder)):
        if file.endswith('.jpg') or file.endswith('.png'):
            img_path = os.path.join(folder, file)
            img_name = file.split('.')[0]
            img_folder = os.path.join(folder, img_name)
            os.mkdir(img_folder)
            os.rename(img_path, os.path.join(img_folder, file))


def do_system(arg):
    """ Run a system command

    Args:
        arg (str): String command
    """
    loguru.logger.info(f"==== running: {arg}")
    err = os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)


def recenter_and_rescale_cam_poses(train_json, test_json):
    cam_poses = []
    for j in [train_json, test_json]:
        for k, v in j.items():
            cam_P = np.array(v['transform_matrix'])
            cam_poses.append(cam_P)
    cam_poses = np.array(cam_poses)
    center = (cam_poses[:, :3, 3].min(0) + cam_poses[:, :3, 3].max(0)) / 2
    cam_poses[:, :3, 3] -= center
    scale = np.linalg.norm(cam_poses[:, :3, 3], axis=-1).mean()
    scale_factor = 2 / scale

    for j in [train_json, test_json]:
        for k, v in j.items():
            cam_P = np.array(v['transform_matrix'])
            cam_P[:3, 3] -= center
            cam_P[:3, 3] *= scale_factor
            v['transform_matrix'] = cam_P.tolist()
    return train_json, test_json


def visualize_cam_json(train_json, test_json):
    train_cam_poses, test_cam_poses = [], []
    train_idx, test_idx = [], []
    for k, v in train_json.items():
        cam_P = np.array(v['transform_matrix'])
        train_idx.append(k)
        train_cam_poses.append(cam_P)
    for k, v in test_json.items():
        cam_P = np.array(v['transform_matrix'])
        test_idx.append(k)
        test_cam_poses.append(cam_P)
    train_cam_poses = np.array(train_cam_poses)
    test_cam_poses = np.array(test_cam_poses)
    all_cam_poses = np.concatenate([train_cam_poses, test_cam_poses], axis=0)
    all_idxs = train_idx + test_idx


def cam_reformat(calib_dir, train_ratio=0.8):
    """ Reformat camera parameters from COLMAP to NeRF format

    Args:
        calib_dir (str): Path to the undistored camera directory
        train_ratio (float, optional): Ratio of images used for training. Defaults to 0.8.
    """
    # Create output folder
    # todo
    camera_file = os.path.join(calib_dir, 'cameras.json')
    cameras = read_cameras(camera_file)

    # Create split file
    total_cam_num = len(cameras)
    total_cam_num_train = int(total_cam_num * train_ratio)
    cam_train = random.sample(cameras.keys(), total_cam_num_train)
    cam_test = list(set(cameras.keys()) - set(cam_train))

    # Create split json file
    train_json = {}
    test_json = {}

    for (split_json, cams) in zip([train_json, test_json], [cam_train, cam_test]):
        for imgid in cams:
            camera = cameras[imgid]
            file_path = f'./images/{imgid}'

            rotation = 0  # TODO
            light_idx = 0  # TODO

            transform_matrix = camera.pose.tolist()
            img_w = camera.width
            camera_angle_x = 2 * np.arctan(img_w * 0.5 / camera.K[0, 0])  # TODO customize img size

            sample = {'file_path': file_path, 'rotation': rotation, 'transform_matrix': transform_matrix,
                      'camera_angle_x': camera_angle_x, 'calib_imgw': img_w, 'light_idx': light_idx}

            split_json.update({imgid: sample})
    # train_json, test_json = recenter_and_rescale_cam_poses(train_json, test_json)
    # visualize_cam_json(train_json, test_json)
    train_json_final = {'frames': train_json}
    test_json_final = {'frames': test_json}

    with open(osp.join(calib_dir, 'transforms_train.json'), 'w') as f:
        json.dump(train_json_final, f)

    with open(osp.join(calib_dir, 'transforms_test.json'), 'w') as f:
        json.dump(test_json_final, f)

    # TODO
    with open(osp.join(calib_dir, 'transforms_val.json'), 'w') as f:
        json.dump(test_json_final, f)


if __name__ == "__main__":
    # Test
    cam_reformat('../lightstage-data/lightstage-vivo-debug/undistorted_cameras/', 1080)
