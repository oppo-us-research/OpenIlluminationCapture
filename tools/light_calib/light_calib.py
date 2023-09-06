import glob
import json
import os.path as osp
from multiprocessing import Pool

import cv2
import imageio
import numpy as np
import tqdm
import trimesh.primitives
from scipy.optimize import least_squares

from easyhec.utils import utils_3d
from ltsg.utils import render_api

radius = 0.08973365203905238
ball_pos = np.array([0.0022671485908238084, -0.02663361382465225, -0.030429725434514568])


def detect(data):
    img, cp, K, mask = data
    img[mask == 0] = 0
    if (img == 255).sum() == 0:
        return np.array([0, 0])
    else:
        y, x = np.stack((img == 255).nonzero(), axis=-1).mean(0)
        return np.array([x, y])


def get_rays(x, y, K, c2w, requires_neg):
    assert isinstance(requires_neg, bool)
    pp = -1 if requires_neg is True else 1
    dirs = np.stack([(x - K[0, 2]) / K[0, 0],
                     pp * (y - K[1, 2]) / K[1, 1],
                     pp], -1)
    rays_d = np.sum(dirs[..., None, :] * c2w[:3, :3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_o = c2w[:3, -1]
    return rays_o, rays_d


def compute_gt_pts_2d(gt_light, camera_poses, K):
    light = gt_light / np.linalg.norm(gt_light)
    pts_2d = []
    for cp in camera_poses:
        pos = cp[:3, 3]
        pos = pos / np.linalg.norm(pos)
        mean = (pos + light) / 2
        normal = mean / np.linalg.norm(mean)
        intersection_pt = normal / np.linalg.norm(normal)
        pt_2d = utils_3d.rect_to_img(K, utils_3d.transform_points(intersection_pt[None, :], np.linalg.inv(cp)))
        pts_2d.append(pt_2d[0])
    return np.stack(pts_2d)


def objective_function(point, rays):
    residuals = []
    theta, phi = point
    point = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    for ray in rays:
        origin = ray[0]
        end = ray[1]
        direction = end - origin

        point_vector = point - origin

        dot_product = np.dot(direction, point_vector)
        magnitude_squared = np.dot(direction, direction)

        projection_factor = dot_product / magnitude_squared
        projection_vector = projection_factor * direction

        projected_point = origin + projection_vector
        residuals.append(np.linalg.norm(projected_point - point))

    return residuals


def find_closest_point(rays, init_guess):
    """
    Find the 3D point closest to multiple rays in 3D space using least-squares optimization.

    Args:
        rays (ndarray): Array of rays, each defined by a starting point and a direction vector.
                        Shape: (num_rays, 2, 3)

    Returns:
        ndarray: Optimal point closest to the rays.
                 Shape: (3,)
    """
    num_rays = rays.shape[0]

    # Initial guess for the optimal point (centered at the origin)
    # initial_point = np.zeros(3)
    # initial_point = np.mean(rays[:, 0], axis=0).reshape(3)

    # Perform least-squares optimization
    # bounds = (0, np.inf)  # Bounds for positive direction5
    init_guess = init_guess / np.linalg.norm(init_guess)
    theta = np.arccos(init_guess[2])
    phi = np.arctan2(init_guess[1], init_guess[0])
    init_guess = np.array([theta, phi])
    result = least_squares(objective_function, init_guess, args=(rays,))
    optimal_point = result.x
    optimal_point = np.array([np.sin(optimal_point[0]) * np.cos(optimal_point[1]),
                              np.sin(optimal_point[0]) * np.sin(optimal_point[1]),
                              np.cos(optimal_point[0])])

    return optimal_point


def read_img(img_path):
    return cv2.cvtColor(np.array(imageio.imread_v2(img_path))[:, :, :3], cv2.COLOR_RGB2GRAY)


def main():
    data_root = osp.expanduser("~/Datasets/lightstage_dataset/light_calib/20230524-20_43_12_mirrorball_OLAT")
    solutions = []
    cameras = json.load(open(osp.expanduser("~/Datasets/lightstage_dataset/20230524-13_08_01_obj_2_egg/calibration/cameras.json")))
    camids = list(cameras.keys())
    camera_poses, Ks, masks = [], [], []
    mesh = trimesh.primitives.Sphere(radius=radius)

    for camid in tqdm.tqdm(camids):
        camera_poses.append(np.array(cameras[camid]['pose']))
        Ks.append(np.array(cameras[camid]['newK']))
        # img_paths.append(osp.join(data_dir, f"{camid}.JPG"))
        H, W = 3984, 2656
        mask = render_api.pyrender_render_mesh_api(mesh, np.linalg.inv(np.array(cameras[camid]['pose'])),
                                                   H, W, np.array(cameras[camid]['newK']))
        masks.append(mask)
    all_camera_poses = np.stack(camera_poses)
    all_camera_poses[:, :3, 3] -= ball_pos
    all_Ks = np.stack(Ks)
    for light_index in tqdm.trange(1, 143):
        # for light_index in tqdm.trange(80, 81):
        # light_index = 80
        data_dir = glob.glob(osp.join(data_root, f"{light_index:03d}*undistorted"))[0]
        img_paths = []
        for camid in tqdm.tqdm(camids):
            img_paths.append(osp.join(data_dir, f"{camid}.JPG"))
        with Pool() as p:
            imgs = list(tqdm.tqdm(p.imap(read_img, img_paths), total=len(img_paths), desc='read imgs'))
        imgs = np.stack(imgs)

        datas = []
        for img, cp, K, mask in zip(imgs, all_camera_poses, all_Ks, masks):
            datas.append([img, cp, K, mask])
        with Pool() as p:
            pts_2d = np.array(list(tqdm.tqdm(p.imap(detect, datas), total=len(datas))))
        keep = ~np.all(pts_2d == 0, axis=1)
        pts_2d = pts_2d[keep]
        imgs = imgs[keep]
        Ks = all_Ks[keep]
        camera_poses = all_camera_poses[keep]

        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
        rays_to_solve = []
        init_guesses = []
        for i in range(len(imgs)):
            ro, rd = get_rays(pts_2d[i, 0], pts_2d[i, 1], Ks[i], camera_poses[i], requires_neg=False)
            # ro = rays_o[pts_2d[i, 1], pts_2d[i, 0]]
            # rd = rays_d[pts_2d[i, 1], pts_2d[i, 0]]
            intersection_pts = intersector.intersects_location(ro[None], rd[None])[0]
            if len(intersection_pts) > 0:
                cam_pos = camera_poses[i][:3, 3]
                intersection_pt = intersection_pts[np.argmin(np.linalg.norm(intersection_pts - cam_pos, axis=-1))]
                normal_at_intersection_pt = intersection_pt / np.linalg.norm(intersection_pt)
                I = intersection_pt - cam_pos
                T = I - 2 * (I.dot(normal_at_intersection_pt)) * normal_at_intersection_pt
                rays_to_solve.append(np.stack([intersection_pt, intersection_pt + T], axis=0))
                init_guesses.append(intersection_pt + T - radius)
        init_guess = np.stack(init_guesses).mean(0)
        rays_to_solve = np.stack(rays_to_solve)
        solution = find_closest_point(rays_to_solve, init_guess)
        solutions.append(solution + ball_pos)
        # print()
    solutions = np.stack(solutions)
    np.savetxt('data/solutions.txt', solutions)


if __name__ == '__main__':
    main()
