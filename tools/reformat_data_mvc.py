import cv2
import os
import glob
import os.path as osp

from PIL import Image

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

root="/nfs/STG/HumanAction/lab150/UCSD_other_data/UCSD_OLAT"


def reformat_greenhead():
    subdirs = sorted(glob.glob(osp.join(root, "*")))
    for subdir in subdirs:
        src_path = osp.join(subdir, "0.jpg")
        dst_path = osp.join(root, subdir.split("/")[-1] + ".jpg")
        os.system(f"mv {src_path} {dst_path}")
        os.system(f"rm -rf {subdir}")


def rotate_greenhead():
    for imgid in ROTATE90_CLOCKWISE_LIST:
        src_path = osp.join(root, "calibration/images", imgid + ".jpg")
        dst_path = osp.join(root, "calibration/images", imgid + "rot.jpg")
        img = cv2.rotate(cv2.imread(src_path), cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(src_path, img)
        # Image.open(src_path).rotate(-90).save(dst_path)
    for imgid in ROTATE90_COUNTERCLOCKWISE_LIST:
        src_path = osp.join(root, "calibration/images", imgid + ".jpg")
        dst_path = osp.join(root, "calibration/images", imgid + "rot.jpg")
        # Image.open(src_path).rotate(90).save(dst_path)
        img = cv2.rotate(cv2.imread(src_path), cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(src_path, img)


def main():
    reformat_greenhead()
    rotate_greenhead()


if __name__ == '__main__':
    main()
