#  created by Isabella Liu (lal005@ucsd.edu) at 2023/03/13 19:48.
#
#  Define configuration parameters for the project
import os.path as osp
import os.path as osp

import configargparse

from ltsg.utils.log_utils import log_args
from easyhec.utils.logger import setup_logger


def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', '-c', is_config_file=True, help='config file path')

    # File paths
    parser.add_argument('--basedir', type=str, help='path to base directory')
    # parser.add_argument('--outputdir', type=str,
    #                     help='path to store output data')
    parser.add_argument('--images', type=str, default='raw_images')
    parser.add_argument("--images_ds", type=int, default=1, help="downsample ratio for images")
    parser.add_argument('--output_name', type=str, default='output')

    # Lights configuration
    parser.add_argument("--lightnum", type=int, default=1)

    # Calibration
    # parser.add_argument('--colmap_db', type=str,
    #                     default='colmap.db', help='Name of the colmap database')
    # parser.add_argument("--colmap_text", default="colmap_text",
    #                     help="input path to the colmap text files (set automatically if run_colmIap is used)")
    parser.add_argument('--force_calib', default=False, action='store_true')
    parser.add_argument('--colmap_images', type=str, default='images')
    parser.add_argument("--colmap_matcher", default="exhaustive",
                        choices=["exhaustive", "sequential", "spatial", "transitive",
                                 "vocab_tree"],
                        help="select which matcher colmap should use. sequential for videos, exhaustive for adhoc images")
    parser.add_argument("--normalize_camera_poses", default=False, action='store_true',
                        help="normalize the camera poses")

    # Undistortion

    # Segmentation
    parser.add_argument('--use_matting', default=False, action='store_true')
    parser.add_argument("--seg_backbone", type=str, default='mobilenetv2',
                        help='backbone for segmentation', choices=['mobilenetv2', 'resnet50', 'resnet101'])
    parser.add_argument('--seg_model_dir', type=str, help='path to segmentation model',
                        default='third_party/BackgroundMattingV2/bgmattingv2_models')

    parser.add_argument("--use_sam", default=False, action='store_true')
    parser.add_argument("--draw_sam", default=False, action='store_true')
    parser.add_argument("--multi_box", default=False, action='store_true')
    parser.add_argument("--use_prior_box", default=False, action='store_true')

    # Camera re-formating
    parser.add_argument("--calib_imgw", type=int, default=1600)
    parser.add_argument("--train_ratio", type=float, default=0.8)

    parser.add_argument("--dbg", type=bool, default=False)
    parser.add_argument("--backup_source", type=bool, default=False)

    parser.add_argument("opts", default=[], nargs=configargparse.REMAINDER)

    # Parse arguments
    if cmd is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd)

    # Print out arguments
    # print_args(args)
    args = merge_opts(args)
    return args


def merge_opts(args):
    opts = args.opts
    assert len(opts) % 2 == 0
    for i in range(0, len(opts), 2):
        opt, val = opts[i], eval(opts[i + 1])
        if opt.startswith('--'):
            opt = opt[2:]
        if opt in vars(args):
            setattr(args, opt, val)
        else:
            raise ValueError(f'Unknown option {opt}')
    delattr(args, 'opts')
    return args


def setup(args):
    output_dir = osp.join(osp.expanduser(args.basedir), args.output_name)
    logger = setup_logger(output_dir)
    log_args(args)
    