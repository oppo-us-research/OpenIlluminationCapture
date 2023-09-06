#  created by Isabella Liu (lal005@ucsd.edu) at 2023/03/30 13:17.
#  
#  An example of how to process single-light data captured by light stage
import glob
import os.path as osp

import os

import loguru

from ltsg.module.calibration import run_colmap, normalize_cameras
from ltsg.module.segmentation import do_object_segmentation, do_object_segmentation_sam, do_object_segmentation_sam_bbox_prior, fake_object_segmentation
from ltsg.module.undistortion import do_image_undistortion
from ltsg.utils.camera_io import read_cameras
from ltsg.utils.sam_api import SAMAPI
from opt import config_parser, setup
from ltsg.utils.util_funcs import cam_reformat, do_system


def main():
    # Get arguments and setup
    args = config_parser()
    setup(args)

    basedir = osp.expanduser(args.basedir)

    # Calibration
    calib_dir = os.path.join(basedir, 'calibration')
    if args.force_calib:
        loguru.logger.info('Running COLMAP...')
        run_colmap(calib_dir, args.colmap_images, args.colmap_matcher, args.normalize_camera_poses, args.dbg)

    light_img_dirs = sorted(glob.glob(os.path.join(basedir, args.images, '*/raw')))
    for light_img_dir in light_img_dirs:
        do_image_undistortion(calib_dir, light_img_dir)

    # Segmentation
    seg_path = os.path.join(basedir, 'segmentation')
    do_image_undistortion(calib_dir, os.path.join(seg_path, 'images'))
    # Do segmentation on undistorted images
    # if not args.use_sam:
    masks = None
    if args.use_matting:
        do_image_undistortion(calib_dir, os.path.join(seg_path, 'backgrounds'))
        masks = do_object_segmentation(args.seg_model_dir, args.seg_backbone, seg_path, 'output')
    if args.use_sam:
        if args.draw_sam:
            do_object_segmentation_sam(seg_path, 'output', masks,multi_box=args.multi_box,
                                       use_prior_box=args.use_prior_box)
        else:
            camera_file = osp.join(calib_dir, 'cameras.json')
            cameras = read_cameras(camera_file)
            do_object_segmentation_sam_bbox_prior(seg_path, 'output', 0.15, cameras)

    # Camera parameters re-formatting
    cam_reformat(calib_dir, args.train_ratio)

    # Store processed images and camera parameters
    final_output = os.path.join(basedir, args.output_name)
    imgs_final_output = os.path.join(final_output, 'images')
    masks_final_output = os.path.join(final_output, 'masks')
    alphas_final_output = os.path.join(final_output, 'alphas')
    os.makedirs(imgs_final_output, exist_ok=True)
    os.makedirs(masks_final_output, exist_ok=True)
    os.makedirs(alphas_final_output, exist_ok=True)
    # Move processed images, masks, and camera parameters
    do_system(f'mv {calib_dir}/transforms*.json {final_output}')
    do_system(f'mv {os.path.join(seg_path, "output", "images_processed")}/* {imgs_final_output}')
    do_system(f'mv {os.path.join(seg_path, "output", "masks")}/* {masks_final_output}')
    do_system(f'mv {os.path.join(seg_path, "output", "alphas")}/* {alphas_final_output}')


if __name__ == '__main__':
    main()
