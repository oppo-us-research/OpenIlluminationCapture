from argparse import ArgumentParser
from pathlib import Path
import os
import numpy as np
import imageio 
parser = ArgumentParser()
from tqdm import tqdm

parser.add_argument("--gt_file_path", default=".", type=str, help="ground truth file path")
parser.add_argument("--src_file_path", default=".", type=str, help="image file folder that contains the images to be transformed")
parser.add_argument("--dst_file_path", default=".", type=str, help="image file folder that contains the transformed images")

args = parser.parse_args()

gt_file_path = Path(args.gt_file_path)
folder_path = [path for path in gt_file_path.iterdir() if path.stem.startswith("test")]
folder_path.sort()
os.makedirs(args.dst_file_path, exist_ok=True)

image_list = []

# for gt_path in tqdm(folder_path):
gt_path = folder_path[24]
view_folder_name = str(gt_path).split("/")[-1]
# import ipdb; ipdb.set_trace()
img_path = os.path.join(gt_path, "rgba.png")
img_to_transform = np.array(imageio.v2.imread(os.path.join(args.src_file_path, view_folder_name, "normal.png")))
img_folder_to_save = os.path.join(args.dst_file_path, view_folder_name)
os.makedirs(img_folder_to_save, exist_ok=True)
img = np.array(imageio.imread(img_path))
img_bg_mask = (img[..., -1:] == 0).squeeze()
img_to_transform[img_bg_mask] = 255
imageio.imwrite(os.path.join(img_folder_to_save, "normal.png"), img_to_transform)
image_list.append(img_to_transform)

# for gt_path in tqdm(folder_path):
#     view_folder_name = str(gt_path).split("/")[-1]
#     # import ipdb; ipdb.set_trace()
#     img_path = os.path.join(gt_path, "rgba.png")
#     img_to_transform = np.array(imageio.v2.imread(os.path.join(gt_path, "normal.png")))
#     img_folder_to_save = os.path.join(args.dst_file_path, view_folder_name)
#     os.makedirs(img_folder_to_save, exist_ok=True)
#     img = np.array(imageio.imread(img_path))
#     img_bg_mask = (img[..., -1:] == 0).squeeze()
#     img_to_transform[img_bg_mask] = 255
#     imageio.imwrite(os.path.join(img_folder_to_save, "normal.png"), img_to_transform)
#     image_list.append(img_to_transform)


# # save the result as a video
# video_path = os.path.join(args.dst_file_path, "video.mp4")
# imageio.mimsave(os.path.join(video_path), np.stack(image_list), fps=24, macro_block_size=1)


