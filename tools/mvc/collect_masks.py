import glob
import os
import os.path as osp
from multiprocessing import Pool

import cv2
import imageio
import numpy as np
import tqdm


def f(data):
    img_path, obj_mask_path, com_mask_path, light_index, data_dir = data
    img_obj_path = osp.join(data_dir, f"Lights/{light_index:03d}/obj_masked/{osp.basename(img_path)[:-4]}.png")
    img_com_path = osp.join(data_dir, f"Lights/{light_index:03d}/com_masked/{osp.basename(img_path)[:-4]}.png")
    obj_mask = cv2.imread(obj_mask_path, 2) > 0
    com_mask = cv2.imread(com_mask_path, 2) > 0
    img = cv2.imread(img_path)[:, :, ::-1]
    img_obj = img.copy()
    img_obj[obj_mask == 0] = 255
    img_obj_rgba = np.concatenate([img_obj, ((obj_mask > 0) * 255).astype('uint8')[..., None]], axis=-1)
    img_com = img.copy()
    img_com[com_mask == 0] = 255
    img_com_rgba = np.concatenate([img_com, ((com_mask > 0) * 255).astype('uint8')[..., None]], axis=-1)
    imageio.imwrite(img_obj_path, img_obj_rgba)
    imageio.imwrite(img_com_path, img_com_rgba)


def verify_img(img_path):
    try:
        img = imageio.imread_v2(img_path)
    except Exception as e:
        print(e, img_path, "failed")


def verify_imgs(data_dirs):
    datas = []
    for data_dir in tqdm.tqdm(data_dirs):
        for light_index in tqdm.trange(1, 14, leave=False):
            img_paths = sorted(glob.glob(osp.join(data_dir, f"Lights/{light_index:03d}/raw_undistorted/*")))
            datas.extend(img_paths)
    with Pool() as p:
        results = list(tqdm.tqdm(p.imap(verify_img, datas), total=len(datas)))


def combine_mask(data):
    smp, mp, data_dir, obj_mask_path, com_mask_path = data
    obj_mask = imageio.imread_v2(mp)
    obj_mask = (obj_mask > 0).astype('uint8') * 255
    imageio.imwrite(obj_mask_path, obj_mask)
    support_mask = imageio.imread_v2(smp)
    com_mask = np.logical_or(obj_mask, support_mask)
    com_mask = (com_mask > 0).astype('uint8') * 255
    imageio.imwrite(com_mask_path, com_mask)


def main():
    root = osp.expanduser("~/Datasets/lightstage_dataset2/UCSD_OLAT2/UCSD_OLAT")
    data_dirs = sorted(glob.glob(osp.join(root, "*obj*")))[:-1]
    for data_dir in tqdm.tqdm(data_dirs):
        mask_paths = sorted(glob.glob(osp.join(data_dir, "output/masks/*")))
        support_mask_paths = sorted(glob.glob(osp.join(data_dir, "output_support/masks/*")))
        assert len(mask_paths) == len(support_mask_paths), (len(mask_paths), len(support_mask_paths), data_dir)
        obj_mask_paths, com_mask_paths = [], []
        os.makedirs(osp.join(data_dir, "output/obj_masks"), exist_ok=True)
        os.makedirs(osp.join(data_dir, "output/com_masks"), exist_ok=True)
        datas = []
        for mp, smp in zip(mask_paths, support_mask_paths):
            obj_mask_path = osp.join(data_dir, "output/obj_masks", osp.basename(mp)[:-4] + ".png")
            com_mask_path = osp.join(data_dir, "output/com_masks", osp.basename(mp)[:-4] + ".png")
            obj_mask_paths.append(obj_mask_path)
            com_mask_paths.append(com_mask_path)
            data = [smp, mp, data_dir, obj_mask_path, com_mask_path]
            datas.append(data)
        for data in datas:
            combine_mask(data)
        # break


if __name__ == '__main__':
    main()
