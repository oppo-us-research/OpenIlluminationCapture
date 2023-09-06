from multiprocessing import Pool

import tqdm
import cv2
import os
import os.path as osp
import glob
import argparse

ROTATE90_CLOCKWISE_LIST = [
    'A1', 'A2', 'A4', 'A5', 'A6',
    'B2', 'B3', 'B4', 'B5',
    'C1',
]

ROTATE90_COUNTERCLOCKWISE_LIST = [
    'A3',
    'B6',
    'C2', 'C3', 'C4', 'C5', 'C6',
    'D1', 'D3', 'D4', 'D5', 'D6',
]


def rotate(img_path):
    viewid = img_path.split("/")[-1].split(".")[0]
    if viewid in ROTATE90_CLOCKWISE_LIST:
        img = cv2.rotate(cv2.imread(img_path), cv2.ROTATE_90_CLOCKWISE)
    elif viewid in ROTATE90_COUNTERCLOCKWISE_LIST:
        img = cv2.rotate(cv2.imread(img_path), cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        print(img_path, "failed to rotate")
        raise NotImplementedError()
    cv2.imwrite(img_path, img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",default="/nfs/STG/HumanAction/lab150/UCSD_other_data/UCSD_OLAT",type=str)
    parser.add_argument("--calibration_dir",default="",type=str)

    args=parser.parse_args()
    root = args.root
    calibration_dir=args.calibration_dir
    data_dirs = sorted(glob.glob(osp.join(root, "*")))
    for data_dir in tqdm.tqdm(data_dirs):
        os.makedirs(osp.join(data_dir, "segmentation"), exist_ok=True)
        os.system(f"mv {data_dir}/ps_results_jpg {data_dir}/segmentation/images")
        os.system(f"mv {data_dir}/segmentation/images/Albedo_png/* {data_dir}/segmentation")
        os.system(f"rm {data_dir}/segmentation/images/* -rf")
        os.system(f"mv {data_dir}/segmentation/* {data_dir}/segmentation/images")
        os.makedirs(osp.join(data_dir, "Lights"), exist_ok=True)
        os.system(f"mv {data_dir}/* {data_dir}/Lights")
        os.system(f"mv {data_dir}/Lights/segmentation {data_dir}/")
        views = sorted(os.listdir(osp.join(data_dir, "Lights")))
        light_indices = range(143)
        datas = []
        for light in tqdm.tqdm(light_indices, leave=False):
            os.makedirs(osp.join(data_dir, "Lights", f"{light:03d}/raw"), exist_ok=True)
            for view in tqdm.tqdm(views, leave=False):
                img_path = osp.join(data_dir, "Lights", view, str(light) + ".jpg")
                tgt_path = f"{data_dir}/Lights/{light:03d}/raw/{view}.jpg"
                os.system(f"mv {img_path} {tgt_path}")
                # cv2.imwrite(tgt_path, cv2.rotate(cv2.imread(tgt_path), cv2.ROTATE_90_CLOCKWISE))
                datas.append(tgt_path)
        # for data in datas:
        #     rotate(data)
        with Pool() as p:
            results = list(tqdm.tqdm(p.imap(rotate, datas), total=len(datas)))
        for view in views:
            os.system(f"rm {data_dir}/Lights/{view} -rf")
        os.system(f"cp -r {calibration_dir} {data_dir}")
        # break
        # print()


if __name__ == '__main__':
    main()
