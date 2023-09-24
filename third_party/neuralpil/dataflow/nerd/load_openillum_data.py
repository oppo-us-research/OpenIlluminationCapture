import json
import os
from traceback import print_tb
import cv2

import imageio
import numpy as np
import tensorflow as tf


def trans_t(t):
    return tf.convert_to_tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1],], dtype=tf.float32,
    )


def rot_phi(phi):
    return tf.convert_to_tensor(
        [
            [1, 0, 0, 0],
            [0, tf.cos(phi), -tf.sin(phi), 0],
            [0, tf.sin(phi), tf.cos(phi), 0],
            [0, 0, 0, 1],
        ],
        dtype=tf.float32,
    )


def rot_theta(th):
    return tf.convert_to_tensor(
        [
            [tf.cos(th), 0, -tf.sin(th), 0],
            [0, 1, 0, 0],
            [tf.sin(th), 0, tf.cos(th), 0],
            [0, 0, 0, 1],
        ],
        dtype=tf.float32,
    )


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w

rotation_matrix = np.array([
        [1, 0, 0, 0],
        [ 0, -1, 0, 0],
        [ 0, 0,-1, 0],
        [ 0, 0, 0, 1]
    ])

def load_openillum_data(basedir,output_img_size=[600,400], illumination_idx=[], trainskip=1, testskip=1, valskip=1,is_single = False,scale = 4):
    splits = ["train", "val", "test"]
    metas = {}
    for s in splits:
        with open(os.path.join(basedir,"output", "transforms_{}.json".format(s)), "r") as fp:
            metas[s] = json.load(fp)
    camera_key = []
    all_imgs = []
    all_masks = []
    all_poses = []
    all_ev100 = []
    all_focal = []
    all_light_idx = []
    counts = [0]
    meta = None
    for s in splits:
        meta = metas[s]
        imgs = []
        masks = []
        poses = []
        focal = []
        light_idx = []
        if s == "train":
            skip = max(trainskip, 1)
        elif s == "val":
            skip = max(valskip, 1)
        else:
            skip = max(testskip, 1)
        tmp_counts = 0
        for key, frame in (meta['frames'].items()):
            camera_key.append(key)
            tmp_counts += 1
            if tmp_counts % skip == 0:
                camera_name = os.path.basename(frame["file_path"])
                if s != "test" :
                    mname = os.path.join(basedir, "output","com_masks",camera_name + ".png")
                else:
                    mname = os.path.join(basedir, "output","masks",camera_name + ".png")
                nmask =  (imageio.imread(mname) / 255).astype(np.float32)
                nmask = np.expand_dims(nmask, axis=-1)
                nmask = tf.image.resize(nmask, output_img_size, method="area").numpy()
                if is_single:
                    single_light_idx = illumination_idx[0]
                    fname_light = os.path.join(basedir, "Lights",single_light_idx, 'raw_undistorted',camera_name + ".JPG")
                    img_file = (imageio.imread(fname_light) / 255).astype(np.float32)
                    img_file = tf.image.resize(img_file, output_img_size, method="area").numpy()
                    img_masked = np.ones_like(img_file)
                    img_masked = img_file * nmask + (1 - nmask) * img_masked
                    imgs.append(img_masked)
                    masks.append(nmask)
                    focal.append(frame["camera_angle_x"])
                    # Read the poses
                    new_transform_matrix = np.matmul(frame["transform_matrix"],rotation_matrix)
                    camera_position = scale * new_transform_matrix[:3, 3]
                    new_transform_matrix[:3, 3] =  camera_position
                    poses.append(np.array(new_transform_matrix))
                    all_ev100.append(8)
                else:
                    list_light_dir = illumination_idx
                    for i, l_idx in enumerate(list_light_dir):
                        l_idx = l_idx.replace("'", "")
                        fname_light = os.path.join(basedir, "Lights",l_idx, 'raw_undistorted',camera_name + ".JPG")
                        img_file = (imageio.imread(fname_light) / 255).astype(np.float32)
                        img_file = tf.image.resize(img_file, output_img_size, method="area").numpy()
                        img_masked = np.ones_like(img_file)                        
                        img_masked = img_file * nmask + (1 - nmask) * img_masked
                        imgs.append(img_masked)
                        masks.append(nmask)
                        if s != "test":
                            light_idx.append(i)
                        else:
                            light_idx.append(0)
                        focal.append(frame["camera_angle_x"])
                        # Read the poses
                        new_transform_matrix = np.matmul(frame["transform_matrix"],rotation_matrix)
                        camera_position = 4 * new_transform_matrix[:3, 3]
                        new_transform_matrix[:3, 3] =  camera_position
                        poses.append(np.array(new_transform_matrix))
                        all_ev100.append(8)


        imgs = np.array(imgs).astype(np.float32)
        # Continue with the masks.
        # They only require values to be between 0 and 1
        # Clip to be sure
        masks = np.clip(np.array(masks).astype(np.float32), 0, 1)

        poses = np.array(poses).astype(np.float32)

        counts.append(counts[-1] + imgs.shape[0])

        all_imgs.append(imgs)
        all_masks.append(masks)
        all_poses.append(poses)
        all_focal.append(focal)
        all_light_idx.append(light_idx)
    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0).astype(np.float32)
    masks = np.concatenate(all_masks, 0).astype(np.float32)
    focal = np.concatenate(all_focal, 0).astype(np.float32)
    poses = np.concatenate(all_poses, 0)
    if is_single:
        light_idxs = None
    else:
        light_idxs = np.concatenate(all_light_idx, 0).astype(np.int32)
    ev100s = np.stack(all_ev100, 0).astype(np.float32)
    H, W = imgs[0].shape[:2]
    focal = 0.5 * W / np.tan(0.5 * focal)

    render_poses = tf.stack(
        [
            pose_spherical(angle, -30.0, 4.0)
            for angle in np.linspace(-180, 180, 40 + 1)[:-1]
        ],
        0,
    )

    return imgs, masks, poses, ev100s, render_poses, [H, W, focal], i_split,light_idxs
