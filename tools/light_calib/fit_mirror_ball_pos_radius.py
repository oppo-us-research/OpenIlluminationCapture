import cv2
import matplotlib.pyplot as plt
import imageio
import numpy as np
import os.path as osp
import json

import trimesh

from ltsg.utils import utils_3d


def main():
    mesh_path = osp.expanduser("~/PycharmProjects/instant-nsr-pl/exp/neus-oppo_mirrorball_r0.5/@20230527-160411/light_calib.obj")
    mesh = trimesh.load(mesh_path)
    pts = mesh.sample(10000)
    radius, x, y, z = utils_3d.sphereFit(pts[:, 0], pts[:, 1], pts[:, 2])
    radius, x, y, z = radius[0], x[0], y[0], z[0]
    print(radius, x, y, z)


if __name__ == '__main__':
    main()
