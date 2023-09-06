#  created by Isabella Liu (lal005@ucsd.edu).

import numpy as np
import cv2
import open3d as o3d
from third_party.colmap.read_write_model import read_model, qvec2rotmat

def do_read_colmap(path, ext):
    cameras, images, points3D = read_model(path=path, ext=ext)
    cameras_out = {}
    for key in cameras.keys():
        p = cameras[key].params
        if cameras[key].model == 'SIMPLE_RADIAL':
            f, cx, cy, k = p
            K = np.array([f, 0, cx, 0, f, cy, 0, 0, 1]).reshape(3, 3)
            dist = np.array([[k, 0, 0, 0, 0]])
        elif cameras[key].model == 'PINHOLE':
            fx, fy, cx, cy = p
            K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)
            dist = np.array([[0, 0, 0, 0, 0]])
        else:
            K = np.array([[p[0], 0, p[2], 0, p[1], p[3], 0, 0, 1]]).reshape(3, 3)
            dist = np.array([[p[4], p[5], p[6], p[7], 0.]])
        cameras_out[key] = {'K': K, 'dist': dist}
    cameras_new = {}
    for key, val in images.items():
        cam = cameras_out[val.camera_id].copy()
        t = val.tvec.reshape(3, 1)
        R = qvec2rotmat(val.qvec)
        cam['R'] = R
        cam['Rvec'] = cv2.Rodrigues(R)[0]
        cam['T'] = t
        # mapkey[val.name.split('.')[0]] = val.camera_id
        cameras_new[val.name.split('.')[0]] = cam
        # cameras_new[val.name.split('.')[0].split('/')[0]] = cam
    keys = sorted(list(cameras_new.keys()))
    cameras_new = {key: cameras_new[key] for key in keys}
    print("num_cameras: {}/{}".format(len(cameras), len(cameras_new)))
    print("num_images:", len(images))
    print("num_points3D:", len(points3D))
    if len(points3D) > 0:
        keys = list(points3D.keys())
        xyz = np.stack([points3D[k].xyz for k in keys])
        rgb = np.stack([points3D[k].rgb for k in keys])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb / 255.)
        from os.path import join
        pcdname = join(path, 'sparse.ply')
        o3d.io.write_point_cloud(pcdname, pcd)
    return cameras_new
