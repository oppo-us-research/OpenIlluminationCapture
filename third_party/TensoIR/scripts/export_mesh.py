
import os, gc
from re import sub
import sys

import torch
from tqdm.auto import tqdm
from opt import config_parser
import json, random
import datetime

from models.shapeBuffer import ShapeModel
from renderer import *
# from models.tensoRF import TensorVM, TensorCP, raw2alpha, TensorVMSplit, AlphaGridMask # used for pure tensorf
# from models.tensoRF_init import TensorVM, TensorCP, raw2alpha, TensorVMSplit, AlphaGridMask # used for pure tensorf
# from models.tensoRF_init_factorize import TensorVM, TensorCP, raw2alpha, TensorVMSplit, AlphaGridMask # used for pure tensorf
from models.tensoRF_rotated_lights import raw2alpha, TensorVMSplit, AlphaGridMask
from utils import *
from dataLoader import dataset_dict
from dataLoader.ray_utils import safe_l2_normalize
from models.relight_utils import relight
from models.brdf import specular_pipeline_render_multilight

args = config_parser()
print(args)

device = torch.device("cuda:{}".format(args.local_rank) if torch.cuda.is_available() else "cpu")




@torch.no_grad()
def export_mesh(args):
    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    alpha, _ = tensorf.getDenseAlpha()
    convert_sdf_samples_to_ply(alpha.cpu(), f'{args.ckpt[:-3]}.ply', bbox=tensorf.aabb.cpu(), level=0.005)




if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    torch.cuda.manual_seed_all(20211202)
    np.random.seed(20211202)

    export_mesh(args)


