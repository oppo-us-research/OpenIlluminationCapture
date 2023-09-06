import tqdm
import imageio
import torch
import numpy as np
import imageio
import cv2
import os
import matplotlib.pyplot as plt
import trimesh

TINY_NUMBER = 1e-8


def parse_raw_sg(sg):
    SGLobes = sg[..., :3] / (torch.norm(sg[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)  # [..., M, 3]
    SGLambdas = torch.abs(sg[..., 3:4])
    SGMus = torch.abs(sg[..., -3:])
    return SGLobes, SGLambdas, SGMus


#######################################################################################################
# compute envmap from SG
#######################################################################################################
def SG2Envmap(lgtSGs, H, W, upper_hemi=False):
    # exactly same convetion as Mitsuba, check envmap_convention.png
    if upper_hemi:
        phi, theta = torch.meshgrid([torch.linspace(0., np.pi / 2., H), torch.linspace(-0.5 * np.pi, 1.5 * np.pi, W)])
    else:
        phi, theta = torch.meshgrid([torch.linspace(0., np.pi, H), torch.linspace(-0.5 * np.pi, 1.5 * np.pi, W)])

    viewdirs = torch.stack([torch.cos(theta) * torch.sin(phi), torch.cos(phi), torch.sin(theta) * torch.sin(phi)],
                           dim=-1)  # [H, W, 3]
    # print(viewdirs[0, 0, :], viewdirs[0, W//2, :], viewdirs[0, -1, :])
    # print(viewdirs[H//2, 0, :], viewdirs[H//2, W//2, :], viewdirs[H//2, -1, :])
    # print(viewdirs[-1, 0, :], viewdirs[-1, W//2, :], viewdirs[-1, -1, :])

    # lgtSGs = lgtSGs.clone().detach()
    viewdirs = viewdirs.to(lgtSGs.device)
    viewdirs = viewdirs.unsqueeze(-2)  # [..., 1, 3]
    # [M, 7] ---> [..., M, 7]
    dots_sh = list(viewdirs.shape[:-2])
    M = lgtSGs.shape[0]
    lgtSGs = lgtSGs.view([1, ] * len(dots_sh) + [M, 7]).expand(dots_sh + [M, 7])
    # sanity
    # [..., M, 3]
    lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)
    lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])
    lgtSGMus = torch.abs(lgtSGs[..., -3:])  # positive values
    # [..., M, 3]
    rgb = lgtSGMus * torch.exp(lgtSGLambdas * (torch.sum(viewdirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
    rgb = torch.sum(rgb, dim=-2)  # [..., 3]
    envmap = rgb.reshape((H, W, 3))
    return envmap


filename = 'NC1_envmap_crop.jpg'
filename = os.path.abspath(filename)
gt_envmap = imageio.imread(filename)[:, :, :3]
gt_envmap = cv2.resize(gt_envmap, (512, 256), interpolation=cv2.INTER_AREA)
x_gt, y_gt = cv2.minMaxLoc(gt_envmap[..., 0])[-1]
gt_envmap_tensor = torch.from_numpy(gt_envmap).cuda() / 255.0
plt.title("gt")
plt.imshow(gt_envmap)
plt.show()
H, W = gt_envmap.shape[:2]
print(H, W)
pts = np.array(trimesh.primitives.Sphere().sample(1000))
numLgtSGs = 1
lgtSGs = torch.randn(numLgtSGs, 7)
lgtSGs.data[..., 3:4] *= 100.
xypreds = []
for pt in tqdm.tqdm(pts):
    this_lgtSGs = lgtSGs.clone()
    this_lgtSGs.data[..., :3] = torch.from_numpy(pt)
    pred_envmap = SG2Envmap(this_lgtSGs, H, W)
    x_pred, y_pred = cv2.minMaxLoc((pred_envmap.cpu().numpy() * 255).astype(np.uint8)[..., 0])[-1]
    xypreds.append([x_pred, y_pred])
    # plt.imshow(pred_envmap)
    # plt.show()
    # print()
xypreds = np.array(xypreds)
minidx = np.abs(xypreds - np.array([x_gt, y_gt])).sum(1).argmin(0)
print(minidx)
print(xypreds[minidx])
print(pts[minidx])

this_lgtSGs = lgtSGs.clone()
this_lgtSGs.data[..., :3] = torch.from_numpy(pts[minidx])
pred_envmap = SG2Envmap(this_lgtSGs, H, W)
import matplotlib.pyplot as plt

plt.imshow(pred_envmap.cpu().numpy());
plt.show()
