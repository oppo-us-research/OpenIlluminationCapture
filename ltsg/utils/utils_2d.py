#  created by Isabella Liu (lal005@ucsd.edu)

import numpy as np
from scipy.optimize import least_squares


def circle_equation(params, x):
    cx, cy, r = params
    return (x[:, 0] - cx) ** 2 + (x[:, 1] - cy) ** 2 - r ** 2


def residuals(params, x):
    return circle_equation(params, x)


def mask_to_points(mask):
    points = np.argwhere(mask)
    return points[:, ::-1]  # Swap x and y coordinates


def fit_circle(mask):
    points = mask_to_points(mask)
    x0 = [mask.shape[1] / 2, mask.shape[0] / 2, min(mask.shape) / 2]  # Initial guess for circle parameters
    result = least_squares(residuals, x0, args=(points,))
    cx, cy, r = result.x
    return cx, cy, r
