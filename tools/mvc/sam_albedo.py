from multiprocessing import Pool

import cv2
import tqdm
import os
import glob
import os.path as osp

from ltsg.module.segmentation import read_one_image
from easyhec.utils.prompt_drawer import PromptDrawer


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=osp.expanduser("~/Datasets/lightstage_dataset/20230524-13_43_31_obj_14_red_bucket/output_sam1/images"))
    parser.add_argument("--output_dir", default=osp.expanduser("~/Datasets/lightstage_dataset/20230524-13_43_31_obj_14_red_bucket/output_sam2"))
    args = parser.parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    img_paths = sorted(glob.glob(osp.join(data_dir, f'*.png')))
    print(f"Found {len(img_paths)} images")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(osp.join(output_dir, "masks"), exist_ok=True)
    os.makedirs(osp.join(output_dir, "images"), exist_ok=True)
    with Pool() as p:
        imgs = list(tqdm.tqdm(p.imap(read_one_image, img_paths), total=len(img_paths)))
    point_drawer = PromptDrawer(screen_scale=2.0, sam_checkpoint="third_party/segment_anything/sam_vit_h_4b8939.pth")
    i = 0
    while i < len(img_paths):
        img_path = img_paths[i]
        imgid = img_path.split('/')[-1].split('.')[0]
        img = imgs[i][..., :3]
        # origin_mask = imgs[i][..., 3]
        # img = img + (255 - origin_mask[..., None])
        mask, flag = point_drawer.run(img.copy(), f"{imgid}_{i}/{len(img_paths)}")
        if flag == -1:
            print("redo last image")
            i = max(0, i - 1)
        elif flag == 1 and i < len(img_paths) - 1:
            print("skip this image")
            i = min(i + 1, len(img_paths) - 1)
        else:
            print("Done")
            output_path = osp.join(output_dir, "masks", f"{imgid}.png")
            # mask = np.logical_and(origin_mask, ~mask)
            cv2.imwrite(output_path, (mask * 255).astype('uint8'))
            # img_ = img.copy()
            # img_[mask == 0] = 255
            # img_rgba = np.concatenate([img_, (mask * 255).astype('uint8')[..., None]], axis=-1)
            # img_rgba = Image.fromarray(img_rgba)
            # img_rgba.save(osp.join(output_dir, "images", f"{imgid}.png"))
            i += 1
        point_drawer.reset()

if __name__ == '__main__':
    main()