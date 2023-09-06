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
    support_mask = imageio.imread_v2(smp)
    if data_dir.endswith("20230524-20_04_27_obj_52_hair"):
        com_mask = imageio.imread_v2(mp)
        obj_mask = np.logical_and(com_mask, ~support_mask)
    else:
        obj_mask = imageio.imread_v2(mp)
        com_mask = np.logical_or(obj_mask, support_mask)
    obj_mask = (obj_mask > 0).astype('uint8') * 255
    com_mask = (com_mask > 0).astype('uint8') * 255
    imageio.imwrite(obj_mask_path, obj_mask)
    imageio.imwrite(com_mask_path, com_mask)


def main():
    root = osp.expanduser("~/Datasets/lightstage_dataset")
    data_dirs = sorted(glob.glob(osp.join(root, "*_obj*")))
    # data_dirs = data_dirs[:52] + data_dirs[53:-2] + [data_dirs[-1]]
    # verify_imgs(data_dirs)
    # exit(0)
    for data_dir in tqdm.tqdm(data_dirs):
        if not data_dir.endswith("20230524-20_04_27_obj_52_hair"): continue
        # if not data_dir.endswith("20230601-21_54_04_obj-63-fabric-friends-cup"): continue
        # if not data_dir.endswith("20230601-21_16_39_obj-55-pumpkin5"): continue
        if data_dir.endswith("20230524-13_43_31_obj_14_red_bucket"):
            mask_paths = sorted(glob.glob(osp.join(data_dir, "output_sam2/masks/*")))
        else:
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
            # if not osp.exists(obj_mask_path) or not osp.exists(com_mask_path):
            data = [smp, mp, data_dir, obj_mask_path, com_mask_path]
            datas.append(data)
        for data in datas:
            combine_mask(data)
        # with Pool() as p:
        #     results = list(tqdm.tqdm(p.imap(combine_mask, datas), total=len(datas)))

        # datas = []
        # for light_index in tqdm.trange(1, 14, leave=False):
        #     # for light_index in tqdm.trange(13, 14, leave=False):
        #     img_paths = sorted(glob.glob(osp.join(data_dir, f"Lights/{light_index:03d}/raw_undistorted/*")))
        #     os.makedirs(osp.join(data_dir, f"Lights/{light_index:03d}/obj_masked"), exist_ok=True)
        #     os.makedirs(osp.join(data_dir, f"Lights/{light_index:03d}/com_masked"), exist_ok=True)
        #     for obj_mask_path, com_mask_path, img_path in tqdm.tqdm(safe_zip(obj_mask_paths, com_mask_paths, img_paths), leave=False, total=len(img_paths)):
        #         datas.append([img_path, obj_mask_path, com_mask_path, light_index, data_dir])
        # with Pool() as p:
        #     results = list(tqdm.tqdm(p.imap(f, datas), total=len(datas), leave=False))


if __name__ == '__main__':
    main()
