import numpy as np
import cv2
import imageio
import tqdm
import os
import glob
import os.path as osp

import PIL.Image


def f(data):
    img_path, mask_path, mask_type = data
    if mask_path != "NA":
        out_path = img_path.replace("raw_undistorted", f"{mask_type}_masked_thumbnail")[:-4] + ".png"
    else:
        out_path = img_path.replace("raw_undistorted", f"origin_thumbnail")[:-4] + ".png"
    # if osp.exists(out_path):
    #     return
    img = cv2.imread(img_path)[:, :, ::-1]
    if mask_path != "NA":
        mask = imageio.imread_v2(mask_path) > 0
        img[mask == 0] = 255
        img_rgba = np.concatenate([img, ((mask > 0) * 255).astype('uint8')[..., None]], axis=-1)
        img = PIL.Image.fromarray(img_rgba)
    else:
        img = PIL.Image.fromarray(img)
    img.thumbnail((400, 600))
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    img.save(out_path)


def main():
    root = osp.expanduser("~/Datasets/lightstage_dataset")
    data_dirs = sorted(glob.glob(osp.join(root, "*_obj*")))
    data_dirs = [dd for dd in data_dirs if "background" not in dd]
    # data_dirs = data_dirs[:52] + data_dirs[53:-2] + [data_dirs[-1]]
    datas = []
    for data_dir in tqdm.tqdm(data_dirs):
        # if not data_dir.endswith("20230524-20_04_27_obj_52_hair"): continue
        for mask_type in ["obj", "com", "origin"]:
            if mask_type != "origin":
                mask_paths = sorted(glob.glob(osp.join(data_dir, f"output/{mask_type}_masks/*")))
            else:
                mask_paths = ["NA"] * 48
            for lightidx in tqdm.trange(1, 14, leave=False):
                img_paths = sorted(glob.glob(osp.join(data_dir, f"Lights/{lightidx:03d}/raw_undistorted/*")))
                os.makedirs(osp.join(data_dir, f"Lights/{lightidx:03d}/{mask_type}_thumbnail"), exist_ok=True)
                for img_path, mask_path in tqdm.tqdm(zip(img_paths, mask_paths), leave=False, total=len(img_paths)):
                    datas.append((img_path, mask_path, mask_type))
    from multiprocessing import Pool
    with Pool() as p:
        results = list(tqdm.tqdm(p.imap(f, datas), total=len(datas)))
    # for data in datas:
    #     f(data)


if __name__ == '__main__':
    main()
