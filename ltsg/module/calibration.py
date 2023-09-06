import argparse
import glob
import json
import os
import os.path as osp
import shutil
from typing import Dict

import loguru
import numpy as np
from PIL import Image

from ltsg.structures.camera import Camera
from ltsg.utils.camera_io import write_cameras
from ltsg.utils.util_funcs import do_system, rm_exif_transpose_folder
from ltsg.utils.utils_3d import sphereFit
from easyhec.utils.utils_3d import Rt_to_pose
from ltsg.utils.colmap_utils import do_read_colmap



def run_colmap(data_dir, images, colmap_matcher, normalize_camera_poses=False, dbg=False,
               rm_exif_transpose=True):
    """ Run COLMAP on the given images

    Args:
        data_dir (str): Path to the data directory
        # colmap_db (str): Name of the colmap database
        images (str): Name of the images folder
        # text (str): Name of the text folder
        colmap_matcher (str): Type of colmap matcher to use
    """
    db_path = os.path.join(data_dir, "colmap.db")
    images_dir = os.path.join(data_dir, images)
    # colmap_cam_text_dir = os.path.join(data_dir, "colmap_text")
    sparse_dir = os.path.join(data_dir, 'sparse')

    # Check if db_path exists
    if os.path.exists(db_path):
        # return
        loguru.logger.info('Database already exists. Deleting the old one.')
        os.remove(db_path)

    # Remove all transposed EXIF tags
    if rm_exif_transpose:
        loguru.logger.info('Removing EXIF transpose tags')
        rm_exif_transpose_folder(images_dir)

    # Run colmap command
    cmd = f'colmap feature_extractor --database_path {db_path} --image_path {images_dir}'
    if not dbg: cmd = cmd + " > /dev/null 2> /dev/null"
    do_system(cmd)

    cmd = f'colmap {colmap_matcher}_matcher --database_path {db_path}'
    if not dbg: cmd = cmd + " > /dev/null 2> /dev/null"
    do_system(cmd)

    # Check if sparse folder exists
    if osp.exists(sparse_dir):
        shutil.rmtree(sparse_dir)

    do_system(f'mkdir {sparse_dir}')
    cmd = f"colmap mapper --database_path {db_path} --image_path {images_dir} --output_path {sparse_dir}"
    if not dbg: cmd = cmd + " > /dev/null 2> /dev/null"
    do_system(cmd)

    cmd = f"colmap bundle_adjuster --input_path {sparse_dir}/0 --output_path {sparse_dir}/0 --BundleAdjustment.refine_principal_point 1"
    if not dbg: cmd = cmd + " > /dev/null 2> /dev/null"
    do_system(cmd)

    # try:
    #     shutil.rmtree(colmap_cam_text_dir)
    # except:
    #     pass
    # do_system(f"mkdir {colmap_cam_text_dir}")
    # cmd = f"colmap model_converter --input_path {sparse}/0 --output_path {colmap_cam_text_dir} --output_type TXT"
    # if not dbg: cmd = cmd + " > /dev/null 2> /dev/null"
    # do_system(cmd)

    cameras = do_read_colmap(osp.join(sparse_dir, "0"), '.bin')
    img_paths = sorted(glob.glob(osp.join(images_dir, '*')))
    assert len(cameras) == len(img_paths), f'Number of cameras ({len(cameras)}) != Number of images ({len(img_paths)})'
    for img_path in img_paths:
        img = Image.open(img_path)
        key = img_path.split('/')[-1].split('.')[0]
        cameras[key]['width'] = img.size[0]
        cameras[key]['height'] = img.size[1]
    for imgid, cam in cameras.items():
        cam = Camera(cam['K'], cam['dist'], cam['width'], cam['height'],
                     np.linalg.inv(Rt_to_pose(cam['R'], cam['T'][:, 0])))
        cameras[imgid] = cam

    if normalize_camera_poses:
        loguru.logger.info('Normalizing camera poses...')
        cameras = normalize_cameras(cameras)
    write_cameras(cameras, osp.join(data_dir, 'cameras.json'))
    return cameras


def normalize_cameras(cameras: Dict[str, Camera]):
    poses = [cam.pose for imgid, cam in cameras.items()]
    poses = np.stack(poses, axis=0)
    poses = poses.reshape(-1, 4, 4)
    cam_positions = poses[:, :3, 3]
    radius, cx, cy, cz = sphereFit(cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2])
    for imgid, cam in cameras.items():
        cam.pose[:3, 3] = cam.pose[:3, 3] - np.array([cx, cy, cz]).reshape(3)
        cam.pose[:3, 3] = cam.pose[:3, 3] / radius
        cameras[imgid] = cam
    return cameras


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_dir', type=str, required=True)
    argparser.add_argument('--images', type=str, default='images')
    argparser.add_argument("--colmap_matcher", default="exhaustive",
                           choices=["exhaustive", "sequential", "spatial", "transitive",
                                    "vocab_tree"],
                           help="select which matcher colmap should use. sequential for videos, exhaustive for adhoc images")
    argparser.add_argument("--normalize_camera_poses", default=False, action='store_true')
    argparser.add_argument("--dbg", default=False, action='store_true')
    argparser.add_argument("--no_rm_exif_transpose", default=False, action='store_true')
    args = argparser.parse_args()

    run_colmap(args.data_dir, args.images, args.colmap_matcher, args.normalize_camera_poses,
               args.dbg, not args.no_rm_exif_transpose)


if __name__ == '__main__':
    main()
