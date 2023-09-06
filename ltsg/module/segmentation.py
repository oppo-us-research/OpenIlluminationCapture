#  created by Isabella Liu (lal005@ucsd.edu) at 2023/03/13 20:45.
#
#  Image segmentation
import tqdm
from multiprocessing import Pool

import imageio
import glob
import os
import os.path as osp

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms.functional import to_tensor

from ltsg.utils import sam_api
from easyhec.utils import utils_3d
# Define device and precision
from easyhec.utils.prompt_drawer import PromptDrawer
from third_party.BackgroundMattingV2.model import MattingRefine

device = torch.device('cuda')
precision = torch.float32


def remove_component(mask):
    """Remove small components in the binary mask

    Args:
        mask (numpy.ndarray): HxW. Mask to be bianrized

    Returns:
        new_mask (numpy.ndarray): HxW. Binary mask with small components removed
    """
    mask = np.array(mask)  # [H, W, 1]
    retval, labels, stats, cent = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8))

    maxcomp = np.argmax(stats[1:, 4]) + 1
    new_mask = labels == maxcomp

    return new_mask


def rm_bg(bgr, src, model, mask_threshold=0.2):
    """ Remove background from the image

    Args:
        bgr (PIL.Image): Background image
        src (PIL.Image): Source image
        model (MattingRefine): Model for image segmentation 
        mask_threshold (float, optional): Threshold on alpha value to obtain the binary mask. Defaults to 0.2.

    Returns:
        fgr_pic (numpy.ndarray): Foreground image with background removed
        obj_mask (numpy.ndarray): Binary object mask
        rgba (numpy.ndarray): Alpha blended RGBA image
        alpha (numpy.ndarray): Alpha of the image
    """
    src = to_tensor(src).to(device).unsqueeze(0).to(precision)  # [1, 3, W, H]
    bgr = to_tensor(bgr).to(device).unsqueeze(0).to(precision)  # [1, 3, W, H]

    with torch.no_grad():
        pha, fgr = model(src, bgr)[:2]

    # Obtain RGBA
    rgba = torch.cat([fgr, pha], dim=1)  # [1, 4, W, H]
    rgba = rgba[0].cpu().permute(1, 2, 0).detach().numpy()  # [H, W, 4]

    # Obtain object mask
    obj_mask = (pha[0] > mask_threshold).permute(
        1, 2, 0).cpu().detach().numpy()  # [H, W, 1]

    # Obtain foreground using binary mask
    fgr_pic = fgr[0].permute(1, 2, 0).cpu().detach().numpy()  # [H, W, 3]
    fgr_pic[~obj_mask[..., 0]] = 0

    # Obtain alpha
    alpha = pha[0].permute(1, 2, 0).cpu().detach().numpy()  # [H, W, 1]

    return fgr_pic, obj_mask, rgba, alpha


def rm_bg_folder(bg_dir, img_dir, model, out_dir):
    """ Remove background from all images in the folder

    Args:
        bg_dir (str): Directory to the background images
        img_dir (str): Directory to the source images
        model (_type_): _description_
        out_dir (str): _description_
        img_type (str, optional): Image suffix. Default: 'jpg'.
    """
    masks = []
    img_type = os.listdir(img_dir)[0].split('.')[-1]
    bg_img_paths = sorted(glob.glob(osp.join(bg_dir, f'*.{img_type}')))
    img_paths = sorted(glob.glob(osp.join(img_dir, f'*.{img_type}')))

    assert len(bg_img_paths) == len(img_paths), 'Number of background images and source images do not match'
    for bg_path, img_path in tqdm.tqdm(zip(bg_img_paths, img_paths), total=len(bg_img_paths), desc="Removing background"):
        imgid = img_path.split('/')[-1].split('.')[0]
        # if osp.exists(os.path.join(out_dir, 'masks', imgid + '.png')):
        #     img_mask = imageio.imread_v2(os.path.join(out_dir, 'masks', imgid + '.png'))
        # else:
        bg = ImageOps.exif_transpose(Image.open(bg_path))
        img = ImageOps.exif_transpose(Image.open(img_path))
        img_fgr, img_mask, img_rgba, img_alpha = rm_bg(bg, img, model)

        # Remove small components
        img_mask = remove_component(img_mask)
        img_rgba[~img_mask] = 0.
        img_alpha[~img_mask] = 0.

        img_fgr = (img_fgr * 255).astype('uint8')
        img_mask = (img_mask * 255).astype('uint8')
        img_rgba = (img_rgba * 255).astype('uint8')
        img_alpha = (img_alpha * 255).astype('uint8')

        # cv2.imwrite(os.path.join(out_path, 'foreground', idx + '.png'), img_fgr)
        cv2.imwrite(os.path.join(out_dir, 'masks', imgid + '.png'), img_mask)
        cv2.imwrite(os.path.join(out_dir, 'alphas', imgid + '.png'), img_alpha)

        # cv2.imwrite(os.path.join(out_path, 'images_processed', idx + '.png'), img_rgba)

        img_fgr = Image.fromarray(img_fgr)
        img_fgr.save(os.path.join(out_dir, 'foreground', imgid + '.png'))

        img_rgba = Image.fromarray(img_rgba)
        img_rgba.save(os.path.join(out_dir, 'images_processed', imgid + '.png'))
        masks.append(img_mask > 0)
    return masks


def do_object_segmentation(seg_model_dir, seg_backbone, data_dir,
                           output_name, fg_dir_name="images_undistorted",
                           bg_dir_name="backgrounds_undistorted"):
    """ Remove background from all images in the folder

    Args:
        seg_model_dir (str): Directory to the segmentation models
        seg_backbone (str): Segmentation backbone
        data_dir (str): Directory containing images with and without the object. Default: 'data'
        output (str): Name of the output folder, will be stored in the same directory as the input folder
    """
    output_dir = os.path.join(data_dir, output_name)
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = MattingRefine(
        backbone=seg_backbone,
        backbone_scale=1 / 2,
        refine_mode='full',
    )
    seg_model_path = os.path.join(seg_model_dir, "pytorch_" + seg_backbone + ".pth")
    if not osp.exists(seg_model_path):
        print("Cannot find the segmentation model, please download it.")
    model.load_state_dict(torch.load(seg_model_path, "cpu"))
    model = model.eval().to(precision).to(device)

    # Create folders
    os.makedirs(osp.join(output_dir, 'foreground'), exist_ok=True)
    os.makedirs(osp.join(output_dir, 'masks'), exist_ok=True)
    os.makedirs(osp.join(output_dir, 'alphas'), exist_ok=True)
    os.makedirs(osp.join(output_dir, 'images_processed'), exist_ok=True)

    # Check if the image has been undistorted, if not, undistort first
    undist_bg_path = osp.join(data_dir, bg_dir_name)
    undist_img_path = osp.join(data_dir, fg_dir_name)

    if not os.path.exists(undist_bg_path) or not os.path.exists(undist_img_path):
        raise RuntimeError(f'Undistorted images not found, please undistort the images first!')

    masks = rm_bg_folder(undist_bg_path, undist_img_path, model, output_dir)
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return masks


def read_one_image(img_path):
    return np.array(ImageOps.exif_transpose(Image.open(img_path)))


def do_object_segmentation_sam(data_dir, output_name, masks=None,
                               multi_box=False, use_prior_box=False):
    """

    Parameters
    ----------
    data_dir
    output_name
    masks: may be masks from matting as prior

    Returns
    -------

    """
    point_drawer = PromptDrawer(screen_scale=2.0,
                                sam_checkpoint="third_party/segment_anything/sam_vit_h_4b8939.pth")
    output_dir = os.path.join(data_dir, output_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(osp.join(output_dir, 'foreground'), exist_ok=True)
    os.makedirs(osp.join(output_dir, 'masks'), exist_ok=True)
    os.makedirs(osp.join(output_dir, 'alphas'), exist_ok=True)
    os.makedirs(osp.join(output_dir, 'images_processed'), exist_ok=True)
    
    # undist_bg_path = osp.join(data_dir, 'backgrounds_undistorted')
    undist_img_dir = osp.join(data_dir, 'images_undistorted')

    if not os.path.exists(undist_img_dir):
        raise RuntimeError(f'Undistorted images not found, please undistort the images first!')
    img_paths = sorted(glob.glob(osp.join(undist_img_dir, f'*.jpg'))) + sorted(glob.glob(osp.join(undist_img_dir, f'*.JPG'))) + \
                sorted(glob.glob(osp.join(undist_img_dir, f'*.png'))) + sorted(glob.glob(osp.join(undist_img_dir, f'*.PNG')))
    with Pool() as p:
        imgs = list(tqdm.tqdm(p.imap(read_one_image, img_paths), total=len(img_paths)))
    i = 0
    while i < len(img_paths):
        img_path = img_paths[i]
        imgid = img_path.split('/')[-1].split('.')[0]
        img = imgs[i]

        if use_prior_box:
            box = np.loadtxt(osp.expanduser(osp.join(f"~/Datasets//lightstage_dataset2/UCSD_OLAT2/UCSD_OLAT/23.06.02_12.42.57_obj-5-fabric-hedgehog-olat/segmentation/output/boxes/{imgid}.txt")))
            direct_return = False
        else:
            box = None
            direct_return = False
        
        mask, flag = point_drawer.run(img.copy(),
                                        f"{imgid}_{i}/{len(img_paths)}",
                                        mask=masks[i] if masks is not None else None,
                                        box=box, direct_return=direct_return
                                        )
        
        if flag == -1:
            print("redo last image")
            i = max(0, i - 1)
        elif flag == 1 and i < len(img_paths) - 1:
            print("skip this image")
            i = min(i + 1, len(img_paths) - 1)
        else:
            print("Done")
            cv2.imwrite(os.path.join(output_dir, 'alphas', imgid + '.png'), (mask * 255).astype('uint8'))
            cv2.imwrite(os.path.join(output_dir, 'masks', imgid + '.png'), (mask * 255).astype('uint8'))
            img_ = img.copy()
            img_[mask == 0] = 255
            img_rgba = np.concatenate([img_, (mask * 255).astype('uint8')[..., None]], axis=-1)
            img_rgba = Image.fromarray(img_rgba)
            img_rgba.save(os.path.join(output_dir, 'images_processed', imgid + '.png'))
            i += 1
        point_drawer.reset()


def fake_object_segmentation(data_dir, output_name):
    output_dir = os.path.join(data_dir, output_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(osp.join(output_dir, 'foreground'), exist_ok=True)
    os.makedirs(osp.join(output_dir, 'masks'), exist_ok=True)
    os.makedirs(osp.join(output_dir, 'alphas'), exist_ok=True)
    os.makedirs(osp.join(output_dir, 'images_processed'), exist_ok=True)
    # undist_bg_path = osp.join(data_dir, 'backgrounds_undistorted')
    undist_img_dir = osp.join(data_dir, 'images_undistorted')

    if not os.path.exists(undist_img_dir):
        raise RuntimeError(f'Undistorted images not found, please undistort the images first!')
    img_paths = sorted(glob.glob(osp.join(undist_img_dir, f'*.jpg'))) + sorted(glob.glob(osp.join(undist_img_dir, f'*.JPG')))
    for img_path in img_paths:
        imgid = img_path.split('/')[-1].split('.')[0]
        img = ImageOps.exif_transpose(Image.open(img_path))
        img = np.array(img)
        alpha = np.ones_like(img[..., 0])
        cv2.imwrite(os.path.join(output_dir, 'alphas', imgid + '.png'), (alpha * 255).astype('uint8'))

        img[alpha < 0.5] = 255
        img_rgba = np.concatenate([img, (alpha * 255).astype('uint8')[..., None]], axis=-1)

        img_rgba = Image.fromarray(img_rgba)
        img_rgba.save(os.path.join(output_dir, 'images_processed', imgid + '.png'))


def do_object_segmentation_sam_bbox_prior(data_dir, output_name, bbox_size, cameras):
    output_dir = os.path.join(data_dir, output_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(osp.join(output_dir, 'foreground'), exist_ok=True)
    os.makedirs(osp.join(output_dir, 'masks'), exist_ok=True)
    os.makedirs(osp.join(output_dir, 'alphas'), exist_ok=True)
    os.makedirs(osp.join(output_dir, 'images_processed'), exist_ok=True)
    undist_img_dir = osp.join(data_dir, 'images_undistorted')

    if not os.path.exists(undist_img_dir):
        raise RuntimeError(f'Undistorted images not found, please undistort the images first!')
    img_paths = sorted(glob.glob(osp.join(undist_img_dir, f'*.jpg'))) + sorted(glob.glob(osp.join(undist_img_dir, f'*.JPG')))
    box3d_corners = np.array([[-bbox_size / 2, -bbox_size / 2, 0],
                              [-bbox_size / 2, bbox_size / 2, 0],
                              [bbox_size / 2, bbox_size / 2, 0],
                              [bbox_size / 2, -bbox_size / 2, 0],
                              [-bbox_size / 2, -bbox_size / 2, bbox_size],
                              [-bbox_size / 2, bbox_size / 2, bbox_size],
                              [bbox_size / 2, bbox_size / 2, bbox_size],
                              [bbox_size / 2, -bbox_size / 2, bbox_size]])
    for i, img_path in enumerate(img_paths):
        imgid = img_path.split('/')[-1].split('.')[0]
        img = ImageOps.exif_transpose(Image.open(img_path))
        img = np.array(img)
        camera = cameras[imgid]
        obj_pose = np.linalg.inv(camera.pose)
        ptsimg = utils_3d.rect_to_img(camera.newK, utils_3d.transform_points(box3d_corners, obj_pose))
        x1, x2, y1, y2 = ptsimg[:, 0].min(), ptsimg[:, 0].max(), ptsimg[:, 1].min(), ptsimg[:, 1].max()
        plt.imshow(img)
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='r', linewidth=3))
        plt.show()
        alpha = sam_api.SAMAPI.segment_api(img, None, None)
        # print("alpha.min,max", alpha.min(), alpha.max())
        # point_drawer.reset()
        cv2.imwrite(os.path.join(output_dir, 'alphas', imgid + '.png'), (alpha * 255).astype('uint8'))

        # img_fgr = img * mask[..., None]

        img[alpha < 0.5] = 255
        img_rgba = np.concatenate([img, (alpha * 255).astype('uint8')[..., None]], axis=-1)

        img_rgba = Image.fromarray(img_rgba)
        img_rgba.save(os.path.join(output_dir, 'images_processed', imgid + '.png'))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=osp.expanduser("~/Datasets/lightstage_dataset/20230524-13_43_31_obj_14_red_bucket/output_sam1/images"))
    parser.add_argument("--output_dir", default=osp.expanduser("~/Datasets/lightstage_dataset/20230524-13_43_31_obj_14_red_bucket/output_sam2"))
    args = parser.parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    img_paths = sorted(glob.glob(osp.join(data_dir, f'*.png')))
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(osp.join(output_dir, "masks"), exist_ok=True)
    os.makedirs(osp.join(output_dir, "images"), exist_ok=True)
    with Pool() as p:
        imgs = list(tqdm.tqdm(p.imap(read_one_image, img_paths), total=len(img_paths)))
    point_drawer = PromptDrawer(screen_scale=2.0, sam_checkpoint="third_party/segment_anything/sam_vit_h_4b8939.pth")
    i = 0
    while i < len(img_paths):
        img_path = img_paths[i]
        imgid = img_path.split('/')[-1].split('.')[0]
        img = imgs[i][..., :3]
        # origin_mask = imgs[i][..., 3]
        origin_mask = np.ones_like(img[..., 0])
        img = img + (255 - origin_mask[..., None])
        mask, flag = point_drawer.run(img.copy(), f"{imgid}_{i}/{len(img_paths)}")
        if flag == -1:
            print("redo last image")
            i = max(0, i - 1)
        elif flag == 1 and i < len(img_paths) - 1:
            print("skip this image")
            i = min(i + 1, len(img_paths) - 1)
        else:
            print("Done")
            output_path = osp.join(output_dir, "masks", f"{imgid}.png")
            mask = np.logical_and(origin_mask, ~mask)
            cv2.imwrite(output_path, (mask * 255).astype('uint8'))
            img_ = img.copy()
            img_[mask == 0] = 255
            img_rgba = np.concatenate([img_, (mask * 255).astype('uint8')[..., None]], axis=-1)
            img_rgba = Image.fromarray(img_rgba)
            img_rgba.save(osp.join(output_dir, "images", f"{imgid}.png"))
            i += 1
        point_drawer.reset()


if __name__ == '__main__':
    main()
