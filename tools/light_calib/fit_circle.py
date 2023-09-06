import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np

from ltsg.utils.utils_2d import fit_circle

mask_path, rgb_path = "data/CB5_mask.JPG", "data/CB5.JPG"
mask = imageio.imread_v2(mask_path)[:, :, 0]
contours, _ = cv2.findContours(
    mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
tmp = cv2.drawContours(np.zeros_like(mask), contours, -1, [255, 255, 255], 1, cv2.LINE_AA)
tmp = tmp > 0
# plt.imshow(tmp)
# plt.show()
cx, cy, r = fit_circle(tmp)
print("Center coordinates: ({}, {})".format(cx, cy))
print("Radius: {}".format(r))
# r += 20
plt.imshow(tmp)
plt.gca().add_patch(plt.Circle((cx, cy), r, fill=False))
plt.show()
olat_img = np.array(imageio.imread_v2(rgb_path))
# plt.imshow(olat_img)
# plt.show()
left, top, right, bottom = int(cx - r), int(cy - r), int(cx + r), int(cy + r)
olat_img = cv2.rectangle(olat_img.astype(np.uint8).copy(), (left, top), (right, bottom), (0, 255, 0), 5)
plt.imshow(olat_img)
plt.show()
imageio.imsave(mask_path.replace("mask", "crop"), olat_img)

tmp = cv2.circle(np.zeros_like(olat_img), (int(cx), int(cy)), int(r), [255, 255, 255], -1)
plt.imshow(tmp)
plt.show()
print()
imageio.imsave(mask_path.replace("mask", "crop_mask"), tmp)
