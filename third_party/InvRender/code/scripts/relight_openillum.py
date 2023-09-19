import sys
sys.path.append('../code')
import argparse
import GPUtil
import os, datetime
from pyhocon import ConfigFactory
import torch
import torch.nn.functional as F

import numpy as np
from PIL import Image
import math
from tqdm import trange

import utils.general as utils
import utils.plots as plt
from utils import rend_util
from model.sg_render import compute_envmap, render_envmap
import imageio


tonemap_img = lambda x: np.power(x, 1./2.2)
clip_img = lambda x: np.clip(x, 0., 1.)
mse2psnr = lambda x: -10. * np.log(x + 1e-8) / np.log(10.)

def decode_img(img, batch_size, total_pixels, img_res, is_tonemap=False):
    img = img.reshape(batch_size, total_pixels, 3)
    img = plt.lin2img(img, img_res).detach().cpu().numpy()[0]
    img = img.transpose(1, 2, 0)
    if is_tonemap:
        img = tonemap_img(img)
    img = clip_img(img)
    return img


def relit_with_light(model, relit_dataloader, images_dir, 
                total_pixels, img_res, albedo_ratio=None, light_type='origin', device='cuda'):

    all_frames_normal = []
    all_frames_gt_normal = []
    
    all_frames_roughness = []
    
    all_frames_rgb = []
    all_frames_gt_rgb = []

    all_relit_psnr = []
    all_relit_ssim = []
    all_relit_lpip_a = []
    all_relit_lpip_v = []


    all_normal_mae = []
        
    for data_index, (indices, model_input, ground_truth) in enumerate(relit_dataloader):
        print('relighting data_index: ', data_index, len(relit_dataloader))
        for key in model_input.keys():
            model_input[key] = model_input[key].cuda()

        split = utils.split_input(model_input, total_pixels)
        res = []
        for s in split:
            s['albedo_ratio'] = None
            out = model(s, trainstage="Material")
            res.append({
                'normals': out['normals'].detach(),
                'network_object_mask': out['network_object_mask'].detach(),
                'object_mask': out['object_mask'].detach(),
                'roughness':out['roughness'].detach(),
                'diffuse_albedo': out['diffuse_albedo'].detach(),
                'sg_diffuse_rgb': out['sg_diffuse_rgb'].detach(),
                'sg_specular_rgb': out['sg_specular_rgb'].detach(),
                'indir_rgb':out['indir_rgb'].detach(),
                'sg_rgb': out['sg_rgb'].detach(),
                'bg_rgb': out['bg_rgb'].detach(),
            })
        
        out_img_name = '{}'.format(indices[0])
        batch_size = ground_truth['rgb'].shape[0]
        model_outputs = utils.merge_output(res, total_pixels, batch_size)

        assert (batch_size == 1)

        # GT mask
        gt_object_mask = model_input['object_mask']
        gt_object_mask = plt.lin2img(gt_object_mask.unsqueeze(-1), img_res)[0].permute(1, 2, 0)
        gt_bg_mask = ~gt_object_mask.expand(-1,-1,3).cpu().numpy()

        # render background
        bg_rgb = model_outputs['bg_rgb']
        bg_rgb = decode_img(bg_rgb, batch_size, total_pixels, img_res, is_tonemap=True)
        object_mask = model_outputs['network_object_mask'].unsqueeze(0)
        object_mask = plt.lin2img(object_mask.unsqueeze(-1), img_res)[0].permute(1, 2, 0)
        bg_mask = ~object_mask.expand(-1,-1,3).cpu().numpy()
    
        ### save sg
        if light_type == 'origin':
            rgb_relit = model_outputs['sg_rgb'] + model_outputs['indir_rgb']
        else:
            rgb_relit = model_outputs['sg_rgb']
        rgb_relit = decode_img(rgb_relit, batch_size, total_pixels, img_res, is_tonemap=True)

        # envmap background
        rgb_relit[gt_bg_mask] = bg_rgb[gt_bg_mask]
        rgb_relit_env_bg = Image.fromarray((rgb_relit * 255).astype(np.uint8))
        rgb_relit_env_bg.save('{0}/sg_rgb_bg_{1}.png'.format(images_dir, out_img_name))
        all_frames_rgb.append(np.array(rgb_relit))

        if light_type == 'origin':
            # save ground truth, all in sRGB, all metrics also calculated in sRGB
            ## save gt rgb
            gt_rgb_masked = decode_img(ground_truth['rgb'], batch_size, total_pixels, img_res, is_tonemap=True).reshape(rgb_relit.shape)
            gt_rgb = np.ones_like(gt_rgb_masked)
            gt_rgb[~gt_bg_mask] = gt_rgb_masked[~gt_bg_mask]
            gt_rgb_img = Image.fromarray((gt_rgb * 255).astype(np.uint8))
            gt_rgb_img.save('{0}/gt_{1}.png'.format(images_dir, out_img_name))
            all_frames_gt_rgb.append(np.array(gt_rgb))
            
            gt_bg_mask_img = Image.fromarray((gt_bg_mask * 255).astype(np.uint8))
            gt_bg_mask_img.save('{0}/gt_bg_mask_{1}.png'.format(images_dir, out_img_name))
           
            ## Calculate PSNR for relighting
            relight_psnr = mse2psnr(np.mean((gt_rgb - rgb_relit) ** 2))
            all_relit_psnr.append(np.array(relight_psnr))
            ### Obtain SSIM and LPIPS
            relight_ssim = utils.rgb_ssim(rgb_relit, gt_rgb, 1)
            relight_l_a = utils.rgb_lpips(gt_rgb, rgb_relit, 'alex', device)
            relight_l_v = utils.rgb_lpips(gt_rgb, rgb_relit, 'vgg', device)
            all_relit_ssim.append(np.array(relight_ssim))
            all_relit_lpip_a.append(np.array(relight_l_a))
            all_relit_lpip_v.append(np.array(relight_l_v))
            print(f'Relighting psnr {relight_psnr} ssim {relight_ssim} lpips_a {relight_l_a} lpips_v {relight_l_v}')

            ### save roughness
            roughness_relit = model_outputs['roughness']
            roughness_relit = decode_img(roughness_relit, batch_size, total_pixels, img_res, is_tonemap=False)
            roughness_relit[gt_bg_mask[..., 0]] = 1.0
            roughness_relit_img = Image.fromarray((roughness_relit * 255).astype(np.uint8))
            roughness_relit_img.save('{0}/roughness_{1}.png'.format(images_dir, out_img_name))
            all_frames_roughness.append(roughness_relit)

            ### save normals
            output_normal = model_outputs['normals'].reshape(rgb_relit.shape)  # [H, W, 3]
            output_normal = F.normalize(output_normal, dim=2)
            output_normal = output_normal.cpu().numpy()
            output_normal[gt_bg_mask[..., 0]] = np.array([0., 0., 1.])
            normal = (output_normal + 1.) / 2.
            normal[bg_mask[..., 0]] = 1.
            normal[gt_bg_mask[..., 0]] = 1.
            normal_img = Image.fromarray((normal * 255).astype(np.uint8))
            normal_img.save('{0}/normal_{1}.png'.format(images_dir, out_img_name))
            all_frames_normal.append(normal)

            ### save indirect rendering
            # indir_rgb = model_outputs['indir_rgb']
            # indir_rgb = decode_img(indir_rgb, batch_size, total_pixels, img_res, is_tonemap=True)
            # indir_rgb = Image.fromarray((indir_rgb * 255).astype(np.uint8))
            # indir_rgb.save('{0}/sg_indir_rgb_{1}.png'.format(images_dir, out_img_name))
        
            # save albedo 
            albedo_relit = model_outputs['diffuse_albedo']
            albedo_relit = decode_img(albedo_relit, batch_size, total_pixels, img_res, is_tonemap=False)
            albedo_relit[gt_bg_mask[..., 0]] = 1.0
            albedo_relit_img = Image.fromarray((albedo_relit * 255).astype(np.uint8))
            albedo_relit_img.save('{0}/albedo_{1}.png'.format(images_dir, out_img_name))
            
            # diffuse rgb
            diffuse_rgb_relit = model_outputs['sg_diffuse_rgb']
            diffuse_rgb_relit = decode_img(diffuse_rgb_relit, batch_size, total_pixels, img_res, is_tonemap=True)
            diffuse_rgb_relit[gt_bg_mask[..., 0]] = 1.0
            diffuse_rgb_relit_img = Image.fromarray((diffuse_rgb_relit * 255).astype(np.uint8))
            diffuse_rgb_relit_img.save('{0}/sg_diffuse_rgb_{1}.png'.format(images_dir, out_img_name))
                
            ### save mask
            bg_mask = Image.fromarray((bg_mask * 255).astype(np.uint8))
            bg_mask.save('{0}/bg_mask_{1}.png'.format(images_dir, out_img_name))
            gt_bg_mask = Image.fromarray((gt_bg_mask * 255).astype(np.uint8))
            gt_bg_mask.save('{0}/gt_bg_mask_{1}.png'.format(images_dir, out_img_name))
        
        
        else:
            # obtain ground truth relit
            gt_rgb_masked = decode_img(ground_truth['relit_rgb'], batch_size, total_pixels, img_res, is_tonemap=True).reshape(rgb_relit.shape)
            gt_rgb = np.ones_like(gt_rgb_masked)
            gt_rgb[~gt_bg_mask] = gt_rgb_masked[~gt_bg_mask]
            # gt_rgb[gt_bg_mask] = bg_rgb[gt_bg_mask]
            gt_rgb_img = Image.fromarray((gt_rgb * 255).astype(np.uint8))
            gt_rgb_img.save('{0}/gt_{1}.png'.format(images_dir, out_img_name))
            
            # calculate relit psnr
            rgb_relit[bg_mask] = 1
            rgb_relit_white_bg = Image.fromarray((rgb_relit * 255).astype(np.uint8))
            rgb_relit_white_bg.save('{0}/sg_rgb_bg_{1}_white_bg.png'.format(images_dir, out_img_name))
        
            relight_psnr = mse2psnr(np.mean((gt_rgb - rgb_relit) ** 2))
            all_relit_psnr.append(np.array(relight_psnr))
            ### Obtain SSIM and LPIPS
            relight_ssim = utils.rgb_ssim(rgb_relit, gt_rgb, 1)
            relight_l_a = utils.rgb_lpips(gt_rgb, rgb_relit, 'alex', device)
            relight_l_v = utils.rgb_lpips(gt_rgb, rgb_relit, 'vgg', device)
            all_relit_ssim.append(np.array(relight_ssim))
            all_relit_lpip_a.append(np.array(relight_l_a))
            all_relit_lpip_v.append(np.array(relight_l_v))
            print(f'Relighting psnr {relight_psnr} ssim {relight_ssim} lpips_a {relight_l_a} lpips_v {relight_l_v}')


    if light_type == 'origin':
        # Calculate final relight psnr
        final_relit_psnr = np.mean(all_relit_psnr)
        final_relit_ssim = np.mean(all_relit_ssim)
        final_relit_lpip_a = np.mean(all_relit_lpip_a)
        final_relit_lpip_v = np.mean(all_relit_lpip_v)


        final_res = f'Final results: \n ' + \
            f'relighting PSNR:  {final_relit_psnr} SSIM {final_relit_ssim} LPIPS_a {final_relit_lpip_a} LPIPS_v {final_relit_lpip_v} \n ' + \
            f'for {len(relit_dataloader)} test instance'
    else:
        final_relit_psnr = np.mean(all_relit_psnr)
        final_relit_ssim = np.mean(all_relit_ssim)
        final_relit_lpip_a = np.mean(all_relit_lpip_a)
        final_relit_lpip_v = np.mean(all_relit_lpip_v)
        
        final_res = f'Final results: \n ' + \
            f'relighting PSNR:  {final_relit_psnr} SSIM {final_relit_ssim} LPIPS_a {final_relit_lpip_a} LPIPS_v {final_relit_lpip_v} \n '

    print(final_res)
    with open(f'{images_dir}/final_metrics.txt', 'w') as f:
        f.write(final_res)
    
    
    # save videos
    os.makedirs(f'{images_dir}/videos', exist_ok=True)
    # save videos
    imageio.mimwrite(os.path.join(images_dir, 'videos', 'rgb.mp4'), all_frames_rgb, fps=24, quality=9)
    imageio.mimwrite(os.path.join(images_dir, 'videos', 'rgb_gt.mp4'), all_frames_gt_rgb, fps=24, quality=9)

    imageio.mimwrite(os.path.join(images_dir, 'videos', 'normal.mp4'), all_frames_normal, fps=24, quality=9)
    imageio.mimwrite(os.path.join(images_dir, 'videos', 'normal_gt.mp4'), all_frames_gt_normal, fps=24, quality=9)
    
    imageio.mimwrite(os.path.join(images_dir, 'videos', 'roughness.mp4'), all_frames_roughness, fps=24, quality=9)
    
    print('Done rendering', images_dir)


def relit_with_light_rotation(model, relit_dataloader, images_dir, 
                total_pixels, img_res, device='cuda'):

    all_frames_rgb = []
    
    item_idx = 0
    _, model_input, ground_truth = next(iter(relit_dataloader))
    for key in model_input.keys():
        model_input[key] = model_input[key].cuda()
        
    frame_num = 128
    frame_rate = 24
    rotate_angle = 2 * np.pi / frame_num
    
    for frame_idx in trange(frame_num):
        print(f'Processing frame {frame_idx}')
        # rotate light
        model.envmap_material_network.rotate_light(frame_idx * rotate_angle)
        model.envmap_material_network.rotate_envmap(frame_idx / frame_num)
        
        # Select one instance to obtain albedo ratio, NOTE: only do albedo alignment on novel lightings
        if frame_idx == 0:
            split = utils.split_input(model_input, total_pixels)
            res = []
            for s in split:
                s['albedo_ratio'] = None
                out = model(s, trainstage="Material")
                res.append({
                    'normals': out['normals'].detach(),
                    'network_object_mask': out['network_object_mask'].detach(),
                    'object_mask': out['object_mask'].detach(),
                    'roughness':out['roughness'].detach(),
                    'diffuse_albedo': out['diffuse_albedo'].detach(),
                    'sg_diffuse_rgb': out['sg_diffuse_rgb'].detach(),
                    'sg_specular_rgb': out['sg_specular_rgb'].detach(),
                    'indir_rgb':out['indir_rgb'].detach(),
                    'sg_rgb': out['sg_rgb'].detach(),
                    'bg_rgb': out['bg_rgb'].detach(),
                })
            batch_size = ground_truth['rgb'].shape[0]
            model_outputs = utils.merge_output(res, total_pixels, batch_size)

            assert (batch_size == 1)
            
            # obtain mask
            bg_rgb = model_outputs['bg_rgb']
            bg_rgb = decode_img(bg_rgb, batch_size, total_pixels, img_res, is_tonemap=False)
            object_mask = model_outputs['network_object_mask'].unsqueeze(0)
            object_mask = plt.lin2img(object_mask.unsqueeze(-1), img_res)[0].permute(1, 2, 0)
            bg_mask = ~object_mask.expand(-1,-1,3).cpu().numpy()
            
            # obtain gt mask
            gt_object_mask = model_outputs['object_mask'].unsqueeze(0)
            gt_object_mask = plt.lin2img(gt_object_mask.unsqueeze(-1), img_res)[0].permute(1, 2, 0)
            gt_bg_mask = ~gt_object_mask.expand(-1,-1,3).cpu().numpy()
            
            # obtain gt albedo and albeo to calculate ratio
            gt_albedo_linear_masked = decode_img(ground_truth['albedo'], batch_size, total_pixels, img_res, is_tonemap=False)
            gt_albedo_linear = np.ones_like(gt_albedo_linear_masked)
            gt_albedo_linear[~gt_bg_mask] = gt_albedo_linear_masked[~gt_bg_mask]
            
            albedo_ratio = (gt_albedo_linear[~gt_bg_mask] / np.clip(albedo_relit_linear[~gt_bg_mask], 1e-6, 1.)).reshape(-1, 3)  # [H*W, 3]
            albedo_ratio = np.median(albedo_ratio, axis=0, keepdims=True).reshape(1, 1, -1)  # [1,/ 1, 3]
        
        albedo_ratio_to_use = albedo_ratio
        albedo_ratio_torch = torch.from_numpy(albedo_ratio_to_use.reshape(-1, 3)).to(device) if albedo_ratio_to_use is not None else None
        
        # query results
        split = utils.split_input(model_input, total_pixels)
        res = []
        for s in split:
            s['albedo_ratio'] = albedo_ratio_torch
            out = model(s, trainstage="Material")
            res.append({
                'normals': out['normals'].detach(),
                'network_object_mask': out['network_object_mask'].detach(),
                'object_mask': out['object_mask'].detach(),
                'roughness':out['roughness'].detach(),
                'diffuse_albedo': out['diffuse_albedo'].detach(),
                'sg_diffuse_rgb': out['sg_diffuse_rgb'].detach(),
                'sg_specular_rgb': out['sg_specular_rgb'].detach(),
                'indir_rgb':out['indir_rgb'].detach(),
                'sg_rgb': out['sg_rgb'].detach(),
                'bg_rgb': out['bg_rgb'].detach(),
            })
        batch_size = ground_truth['rgb'].shape[0]
        model_outputs = utils.merge_output(res, total_pixels, batch_size)
        assert (batch_size == 1)
        
        # render background
        bg_rgb = model_outputs['bg_rgb']
        bg_rgb = decode_img(bg_rgb, batch_size, total_pixels, img_res, is_tonemap=True)
        object_mask = model_outputs['network_object_mask'].unsqueeze(0)
        object_mask = plt.lin2img(object_mask.unsqueeze(-1), img_res)[0].permute(1, 2, 0)
        bg_mask = ~object_mask.expand(-1,-1,3).cpu().numpy()
        
        # save sg_rgb
        rgb_relit = model_outputs['sg_rgb']
        rgb_relit = decode_img(rgb_relit, batch_size, total_pixels, img_res, is_tonemap=True)
        # envmap background
        rgb_relit[bg_mask] = bg_rgb[bg_mask]
        rgb_relit_env_bg = Image.fromarray((rgb_relit * 255).astype(np.uint8))
        rgb_relit_env_bg.save('{0}/frame_{1}.png'.format(images_dir, frame_idx))
        all_frames_rgb.append(np.array(rgb_relit))
        
        # save albedo 
        albedo_relit = model_outputs['diffuse_albedo']
        albedo_relit = decode_img(albedo_relit, batch_size, total_pixels, img_res, is_tonemap=True)
        albedo_relit[bg_mask] = 1.0
        albedo_relit_img = Image.fromarray((albedo_relit * 255).astype(np.uint8))
        albedo_relit_img.save('{0}/albedo_{1}.png'.format(images_dir, frame_idx))
        
    # save videos
    imageio.mimwrite(os.path.join(images_dir, 'rgb.mp4'), all_frames_rgb, fps=frame_rate, quality=9)
    

def relight_obj(**kwargs):
    torch.set_default_dtype(torch.float32)

    conf = ConfigFactory.parse_file(kwargs['conf'])
    exps_folder_name = kwargs['exps_folder_name']
    relits_folder_name = kwargs['relits_folder_name']

    expname = 'Mat-' + kwargs['expname']

    if kwargs['timestamp'] == 'latest':
        if os.path.exists(os.path.join('../', kwargs['exps_folder_name'], expname)):
            timestamps = os.listdir(os.path.join('../', kwargs['exps_folder_name'], expname))
            if (len(timestamps)) == 0:
                print('WRONG EXP FOLDER')
                exit()
            else:
                timestamp = sorted(timestamps)[-1]
        else:
            print('WRONG EXP FOLDER')
            exit()
    else:
        timestamp = kwargs['timestamp']

    utils.mkdir_ifnotexists(os.path.join('../', relits_folder_name))
    expdir = os.path.join('../', exps_folder_name, expname)
    relitdir = os.path.join('../', relits_folder_name, expname, os.path.basename(kwargs['data_split_dir']))

    model = utils.get_class(conf.get_string('train.model_class'))(conf=conf.get_config('model'))
    if torch.cuda.is_available():
        model.cuda()
    
    # load data
    if  kwargs['light_path'] == '':
        relit_dataset = utils.get_class(conf.get_string('train.dataset_class'))(
                                        kwargs['data_split_dir'], kwargs['frame_skip'], split='test')
    
    else:
        envmap_path = kwargs['light_path']
        light_name = os.path.basename(envmap_path).split('.')[0]
        relit_dataset = utils.get_class(conf.get_string('train.dataset_class'))(
                                        kwargs['data_split_dir'], kwargs['frame_skip'], split='test', relight_name=light_name)    
    
    relit_dataloader = torch.utils.data.DataLoader(relit_dataset, batch_size=1,
                                    shuffle=False, collate_fn=relit_dataset.collate_fn)

    total_pixels = relit_dataset.total_pixels
    img_res = relit_dataset.img_res

    # load trained model
    old_checkpnts_dir = os.path.join(expdir, timestamp, 'checkpoints')
    ckpt_path = os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth")
    print('Loading checkpoint: ', ckpt_path)
    saved_model_state = torch.load(ckpt_path)
    model.load_state_dict(saved_model_state["model_state_dict"])

    print("start render...")
    model.eval()
    
    if kwargs['light_path'] == '':
        # Test on original light
        images_dir = relitdir + datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")
        utils.mkdir_ifnotexists(images_dir)
        print('Output directory is: ', images_dir)
        # with open(os.path.join(relitdir, 'ckpt_path.txt'), 'w') as fp:
        #     fp.write(ckpt_path + '\n')

        relit_with_light(model, relit_dataloader, images_dir, 
                        total_pixels, img_res, albedo_ratio=None, light_type='origin')
    elif kwargs['light_path'] != '' and kwargs['rotate_light'] == False:
        # Test on novel light and fixed light rotation
        print(f'Loading light {light_name} from: {envmap_path}')
        model.envmap_material_network.load_light(envmap_path)
        images_dir = relitdir + f'_{light_name}_relit' + datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")
        utils.mkdir_ifnotexists(images_dir)
        print('Output directory is: ', images_dir)

        relit_with_light(model, relit_dataloader, images_dir, 
                        total_pixels, img_res, albedo_ratio=None, light_type='novel')
    else:
        # Test on novel light and light rotation
        print(f'Loading light {light_name} from: {envmap_path}')
        model.envmap_material_network.load_light(envmap_path)
        images_dir = relitdir + f'_{light_name}_relit' + datetime.datetime.now().strftime("-%Y%m%d-%H%M%S") + '_rotate_with_bg'
        utils.mkdir_ifnotexists(images_dir)
        print('Output directory is: ', images_dir)
        relit_with_light_rotation(model, relit_dataloader, images_dir, total_pixels, img_res, device='cuda')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/default.conf')
    parser.add_argument('--expname', type=str, default='', help='The experiment name to be relituated.')
    parser.add_argument('--light_path', type=str, default='', help='Path to relight envmap')
    parser.add_argument('--exps_folder', type=str, default='exps', help='The experiments folder name.')

    parser.add_argument('--data_split_dir', type=str, default='')
    parser.add_argument('--frame_skip', type=int, default=1, help='skip frame when test')
    parser.add_argument('--rotate_light', type=int, default=0, help='whether rotate light')

    parser.add_argument('--timestamp', default='latest', type=str, help='The experiemnt timestamp to test.')
    parser.add_argument('--checkpoint', default='latest',type=str,help='The trained model checkpoint to test')

    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')

    opt = parser.parse_args()

    gpu = opt.gpu
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")

    if (not gpu == 'ignore'):
        os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(gpu)

    relight_obj(conf=opt.conf,
                relits_folder_name='relits',
                data_split_dir=opt.data_split_dir,
                expname=opt.expname,
                light_path=opt.light_path,
                exps_folder_name=opt.exps_folder,
                timestamp=opt.timestamp,
                checkpoint=opt.checkpoint,
                frame_skip=opt.frame_skip,
                rotate_light=opt.rotate_light,
                device=device
                )
