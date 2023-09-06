import imageio
from multiprocessing import Pool

import tqdm
import os
import os.path as osp

import numpy as np
from PIL import Image, ImageOps

from ltsg.utils.camera_io import read_cameras


def undistort_one_image(data):
    img_path, camera, img_type, output_path, imgid = data
    # print('force', force)
    output_path = os.path.join(output_path, imgid + "." + img_type)
    # if osp.exists(output_path) and not force: return
    img = ImageOps.exif_transpose(Image.open(img_path))
    img_exif = img.getexif()
    img = np.array(img)
    # Undistort image
    dst = camera.undistort_image(img)

    dst = dst.astype('uint8')
    if len(img_exif) > 0:
        if img_type.lower() == 'jpg':
            img_dst = Image.fromarray(dst, 'RGB')
        elif img_type.lower() == 'png':
            img_dst = Image.fromarray(dst, 'RGBA')
        else:
            raise NotImplementedError()
        img_dst.save(output_path, exif=img_exif)
    else:
        imageio.imsave(output_path, dst)


def undistort_images(cameras, img_dir, output_path=None, img_type='jpg', scale=1.0, force=False):
    """
    Undistort images using camera parameters

    Args:
        cameras (dict): Dict of camera parameters
        img_dir (str): Path to images
        output_path (str, optional): Path to output folder
        img_type (str, optional): Image type. Defaults to 'jpg'.
        scale (float, optional): Scale factor for camera intrinsics. Defaults to 1.0.
                                 [Example: If the img resolution is 1/2 of the resolution used in calibration, scale=0.5]
    Returns:
        cameras (dict): Dict of undistorted camera parameters
    """
    # Create output folder
    if output_path is None:
        folder_name = osp.basename(img_dir.rstrip("/"))
        output_path = os.path.join(os.path.dirname(img_dir), folder_name + '_undistorted')

    os.makedirs(output_path, exist_ok=True)
    datas = []
    for imgid, camera in tqdm.tqdm(cameras.items(), desc='Prepare undistorting images'):
        img_path = os.path.join(img_dir, imgid + '.' + img_type)
        data = img_path, camera, img_type, output_path, imgid
        if not osp.exists(os.path.join(output_path, imgid + "." + img_type)) or force:
            datas.append(data)
    if len(datas) > 0:
        with Pool() as p:
            results = list(tqdm.tqdm(p.imap(undistort_one_image, datas), total=len(datas)))


def do_image_undistortion(cam_dir, img_dir, force=False):
    camera_file = osp.join(cam_dir, 'cameras.json')
    cameras = read_cameras(camera_file)
    img_type = os.listdir(img_dir)[0].split('.')[-1]
    undistort_images(cameras, img_dir, img_type=img_type, force=force)


def main():
    # Unit test
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--cam_dir', type=str, required=True)
    argparser.add_argument('--img_dir', type=str, required=True)
    argparser.add_argument('--force', default=False, action='store_true')
    args = argparser.parse_args()

    do_image_undistortion(args.cam_dir, args.img_dir, args.force)


if __name__ == '__main__':
    main()
