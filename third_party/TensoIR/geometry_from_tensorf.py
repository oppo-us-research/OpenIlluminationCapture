import os
import imageio
from matplotlib import image
from numpy import einsum
from tqdm.auto import tqdm
from opt import config_parser
from models.tensoRF_init import TensorVM, TensorCP, raw2alpha, TensorVMSplit, AlphaGridMask
from renderer import OctreeRender_trilinear_fast_init
from dataLoader.ray_utils import safe_l2_normalize

from utils import *
from torch.utils.tensorboard import SummaryWriter
from dataLoader import dataset_dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

renderer = OctreeRender_trilinear_fast_init


@torch.no_grad()
def export_mesh(args):
    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    alpha,_ = tensorf.getDenseAlpha()
    convert_sdf_samples_to_ply(alpha.cpu(), f'{args.ckpt[:-3]}.ply',bbox=tensorf.aabb.cpu(), level=0.005)



def store_geometry(dataset, tensorf, args, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    width, height = dataset.img_wh
    for idx in tqdm(range(len(dataset))):
        try:
            item = dataset[idx]
            # output paths
            frame_path = item['paths'].split('/')[-1] # such as "train_052"
            base_dir = os.path.join(args.geo_buffer_path, frame_path)
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)
            normals_path = os.path.join(base_dir, 'normals_gen.npy')
            normals_vis_path = os.path.join(base_dir, 'normals_gen.png')
            normals_surface_path = os.path.join(base_dir, 'normals_surface_gen.npy')
            normals_surface_vis_path = os.path.join(base_dir, 'normals_surface_gen.png')

            surface_xyz_path = os.path.join(base_dir, 'surface_xyz.npy')
            surface_xyz_vis_path = os.path.join(base_dir, 'surface_xyz.png')
            acc_path = os.path.join(base_dir, 'acc.npy')
            acc_vis_path = os.path.join(base_dir, 'acc.png')
            visibility_path = os.path.join(base_dir, 'visibility.npy') #(H, W, 32*16)
            visibility_vis_path = os.path.join(base_dir, 'visibility.png') # average over all samples

            indirect_light_vis_path = os.path.join(base_dir, 'indirect_light.png')
            indirect_light_vis_raw_path = os.path.join(base_dir, 'indirect_light_raw.png')

            frame_rays = item['rays'].squeeze(0).to(device) # (H*W, 6)
            frame_gt_normals = item['gt_normals'].to(device) # (H*W, 3)


            rays_mask = in_box_filter(tensorf, frame_rays)
            acc_map_valid, depth_map_valid, normals_map_valid = compute_acc_and_depth_and_normals( 
                                                                    tensorf=tensorf, 
                                                                    frame_rays=frame_rays[rays_mask],
                                                                    N_samples=N_samples,
                                                                    ndc_ray=ndc_ray,
                                                                    chunk=512,
                                                                    device=device
                                                                    )
            acc_map = torch.zeros((frame_rays.shape[0])).to(device)
            xyz_map = torch.zeros((frame_rays.shape[0]), 3).to(device)
            normals_map = torch.zeros((frame_rays.shape[0], 3)).to(device)

            # clip small acc_map value to zero
            acc_map_valid[acc_map_valid<args.acc_thre] = 0
            acc_map[rays_mask] = acc_map_valid

            acc_map = torch.clamp(acc_map, 0, 1)
            # cumpute XYZ map
            rays_o_valid, rays_d_valid = frame_rays[rays_mask][..., :3].to(device), frame_rays[rays_mask][..., 3:].to(device)  # [bs, 3]
            surface_xyz_valid = rays_o_valid + depth_map_valid.unsqueeze(-1) * rays_d_valid  # [bs, 3]
            xyz_map[rays_mask] = surface_xyz_valid.detach()
            xyz_map[acc_map == 0] = 0

            is_hit = (acc_map > 0)

            # compute normals map by volume rendering
            normals_map[rays_mask] = normals_map_valid
            normals_map[acc_map == 0] = torch.Tensor([0, 0, 1]).to(device)
            # compute normals map by only caculating the surface normal
            normals_surface_map = torch.zeros((frame_rays.shape[0], 3)).to(device)
            normals_surface_map[torch.logical_not(is_hit)] = torch.Tensor([0, 0, 1]).to(device)
            normals_surface_map[is_hit] = compute_normals_on_surface(tensorf, xyz_map[is_hit], chunk=4096, device=device)

            # # compute visibility map
            # visibility_map = torch.zeros(xyz_map.shape[0], dataset.light_xyz.shape[0], device=device) # [N_rays, 32 * 16]
            # visibility_map[is_hit] = compute_visibilty( dataset,
            #                                             tensorf,
            #                                             xyz_map[is_hit],
            #                                             N_samples,
            #                                             args
            #                                             )


            # compute visibility map and indirect light map
            visibility_map = torch.zeros(xyz_map.shape[0], dataset.light_xyz.shape[0], device=device) # [N_rays, 32 * 16]
            indirect_light_map = torch.zeros(xyz_map.shape[0], dataset.light_xyz.shape[0], 3, device=device) # [N_rays, 32 * 16, 3]
            
            visibility_map[is_hit], indirect_light_map[is_hit] = compute_visibilty_and_indirect_light(  dataset,
                                                                                                        tensorf,
                                                                                                        xyz_map[is_hit],
                                                                                                        frame_gt_normals[is_hit],
                                                                                                        N_samples,
                                                                                                        args
                                                                                                        )

            acc_map = acc_map.view(width, height, -1).detach().cpu().numpy()
            xyz_map = xyz_map.view(width, height, -1).detach().cpu().numpy()
            visibility_map_avg = torch.mean(visibility_map.view(width, height, -1), dim=-1).detach().cpu().numpy() # used for visualization
            visibility_map = visibility_map.view(width, height, -1).detach().cpu().numpy()
            normals_map = normals_map.view(width, height, -1).detach().cpu().numpy()
            normals_surface_map = normals_surface_map.view(width, height, -1).detach().cpu().numpy()
            indirect_light_map_accumulated = torch.sum(indirect_light_map, dim=1).view(width, height, -1).detach().cpu().numpy()
            indirect_light_map = indirect_light_map.view(width, height, -1).detach().cpu().numpy()


            # write raw data
            np.save(acc_path,  acc_map)
            np.save(surface_xyz_path,  xyz_map)
            np.save(visibility_path,  visibility_map)
            np.save(normals_path,  normals_map)
            np.save(normals_surface_path,  normals_surface_map)

            # write visualization data
            imageio.imwrite(acc_vis_path,  (acc_map - acc_map.min()) / (acc_map.max() - acc_map.min()))
            imageio.imwrite(surface_xyz_vis_path,  (xyz_map - xyz_map.min()) / (xyz_map.max() - xyz_map.min()))
            imageio.imwrite(visibility_vis_path,  (visibility_map_avg - visibility_map_avg.min()) / (visibility_map_avg.max() - visibility_map_avg.min()))
            imageio.imwrite(normals_vis_path,  (normals_map - normals_map.min()) / (normals_map.max() - normals_map.min()))
            imageio.imwrite(normals_surface_vis_path,  (normals_surface_map - normals_surface_map.min()) / (normals_surface_map.max() - normals_surface_map.min()))
            imageio.imwrite(indirect_light_vis_raw_path, indirect_light_map_accumulated)
            imageio.imwrite(indirect_light_vis_path,  (indirect_light_map_accumulated - indirect_light_map_accumulated.min()) / (indirect_light_map_accumulated.max() - indirect_light_map_accumulated.min()))
            tensorf.zero_grad()

        except StopIteration:
            print('Unexpected error of dataloader')


def compute_acc_and_depth_and_normals(tensorf, 
                                    frame_rays,
                                    N_samples, 
                                    ndc_ray,
                                    chunk=4096, 
                                    device=device
                                    ):
    acc_maps, depth_maps, normals_maps = [], [], []
    viewdirs = frame_rays[:, 3:6].to(device)
    
    N_rays_all = frame_rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = frame_rays[chunk_idx * chunk: (chunk_idx + 1) * chunk]
        viewdirs_chunk = viewdirs[chunk_idx * chunk: (chunk_idx + 1) * chunk]

        if ndc_ray:
            xyz_sampled, z_vals, ray_valid = tensorf.sample_ray_ndc(rays_chunk[:, :3], viewdirs_chunk, is_train=False,N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
            rays_norm = torch.norm(viewdirs_chunk, dim=-1, keepdim=True)
            dists = dists * rays_norm
            viewdirs_chunk = viewdirs_chunk / rays_norm
        else:
            xyz_sampled, z_vals, ray_valid = tensorf.sample_ray(rays_chunk[:, :3], viewdirs_chunk, is_train=False,N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
            viewdirs_chunk = viewdirs_chunk.view(-1, 1, 3).expand(xyz_sampled.shape)
        
        if tensorf.alphaMask is not None:
            alphas = tensorf.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= (~alpha_mask)
            ray_valid = ~ray_invalid

        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)


        if ray_valid.any():
            xyz_sampled = tensorf.normalize_coord(xyz_sampled) # TODO:check if we should use normalize_coord to derive normals
            xyz_sampled.requires_grad_(True) # used for derive gradient as normals

            # The follow is equivalent to "sigma_feature = tensorf.compute_densityfeature(xyz_sampled[ray_valid])", !!delete the detach!!
            # It now can only support TensorVMSplit TODO
            # beigin
            coordinate_plane = torch.stack((xyz_sampled[ray_valid][..., tensorf.matMode[0]], xyz_sampled[ray_valid][..., tensorf.matMode[1]], xyz_sampled[ray_valid][..., tensorf.matMode[2]])).view(3, -1, 1, 2)
            coordinate_line = torch.stack((xyz_sampled[ray_valid][..., tensorf.vecMode[0]], xyz_sampled[ray_valid][..., tensorf.vecMode[1]], xyz_sampled[ray_valid][..., tensorf.vecMode[2]]))
            coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).view(3, -1, 1, 2)

            sigma_feature = torch.zeros((xyz_sampled[ray_valid].shape[0],), device=xyz_sampled[ray_valid].device)
            for idx_plane in range(len(tensorf.density_plane)):
                plane_coef_point = F.grid_sample(tensorf.density_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                    align_corners=True).view(-1, *xyz_sampled[ray_valid].shape[:1])
                line_coef_point = F.grid_sample(tensorf.density_line[idx_plane], coordinate_line[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled[ray_valid].shape[:1])
                sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)
            # end

            validsigma = tensorf.feature2density(sigma_feature)
            sigma[ray_valid] = validsigma

            d_output = torch.ones_like(validsigma, requires_grad=False, device=sigma[ray_valid].device)

            gradients = torch.autograd.grad(
                                    outputs=validsigma,
                                    inputs=xyz_sampled,
                                    grad_outputs=d_output,
                                    create_graph=True,
                                    retain_graph=True,
                                    only_inputs=True
                                    )[0]


            gradients[torch.logical_not(ray_valid)] = torch.Tensor([0, 0, 1]).to(gradients.device)
   
            normals = - safe_l2_normalize(gradients, dim=-1)
            normals = normals.view(-1, z_vals.shape[1], 3)

        alpha, weight, bg_weight = raw2alpha(sigma, dists * tensorf.distance_scale)
        acc_map = torch.sum(weight, -1)
        depth_map = torch.sum(weight * z_vals, -1)
        normals_map = torch.sum(weight[..., None] * normals, -2)
        normals_map = safe_l2_normalize(normals_map, dim=-1)
        acc_maps.append(acc_map.detach())
        depth_maps.append(depth_map.detach())
        normals_maps.append(normals_map.detach())

    return torch.cat(acc_maps) , torch.cat(depth_maps), torch.cat(normals_maps)

# choose the rays that is in the bounding box
@torch.no_grad()
def in_box_filter(tensorf, all_rays, N_samples=256, chunk=10240*5, bbox_only=False):
    print('========> filtering rays ...')

    N = torch.tensor(all_rays.shape[:-1]).prod()

    mask_filtered = []
    idx_chunks = torch.split(torch.arange(N), chunk)
    for idx_chunk in idx_chunks:
        rays_chunk = all_rays[idx_chunk].to(tensorf.device)

        rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
        if bbox_only:
            vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
            rate_a = (tensorf.aabb[1] - rays_o) / vec
            rate_b = (tensorf.aabb[0] - rays_o) / vec
            t_min = torch.minimum(rate_a, rate_b).amax(-1)#.clamp(min=near, max=far)
            t_max = torch.maximum(rate_a, rate_b).amin(-1)#.clamp(min=near, max=far)
            mask_inbbox = t_max > t_min

        else:
            xyz_sampled, _,_ = tensorf.sample_ray(rays_o, rays_d, N_samples=N_samples, is_train=False)
            mask_inbbox= (tensorf.alphaMask.sample_alpha(xyz_sampled).view(xyz_sampled.shape[:-1]) > 0).any(-1)

        mask_filtered.append(mask_inbbox)

    mask_filtered = torch.cat(mask_filtered).view(all_rays.shape[:-1])

    return mask_filtered

@torch.no_grad()
def compute_visibilty(dataset, tensorf, surface_point, N_samples, args, points_chunk=64, device='cuda'):
    # TODO: can introduce normals to save GPU memory and computation time
    # TODO: handle ndc_rays
    acc_maps = []
    # N_samples = N_samples if N_samples > 0 else tensorf.nSamples
    N_samples = 128 # TODO: add this to config file
    lxyz = dataset.light_xyz.to(device) # [envH * envW, 3]
    lareas = dataset.light_areas.to(device) # [envH * envW,]
    lxyz_flat = lxyz.view(1, -1, 3) # [1, envH * envW, 3]
    n_lights = lxyz_flat.shape[1]

    # lvis_hit = torch.zeros((surface_point.shape[0], n_lights), dtype=torch.float32) # (n_surf_pts, n_lights)
    points_num = surface_point.shape[0]
    idx_chunks = torch.split(torch.arange(points_num), points_chunk)
    for idx_chunk in idx_chunks:
        surf_pts_chunk = surface_point[idx_chunk].to(device) # [points_chunk, 3]
        lxyz_chunk = lxyz_flat.repeat(surf_pts_chunk.shape[0], 1, 1) # [points_chunk, envH * envW, 3]

        # surface to lights
        surf2l = lxyz_chunk - surf_pts_chunk[:, None, :] # [points_chunk, envH * envW, 3]
        surf2l_norm = torch.norm(surf2l, dim=-1, keepdim=True) # [points_chunk, envH * envW, 1]
        surf2l = surf2l / surf2l_norm # [points_chunk, envH * envW, 3]
        
        t = torch.linspace(0., 1., N_samples, device=device) # [N_samples,]
        vis_near = 0.03
        vis_far = 2. # TODO : need to be tuned and write the args into the config file

        z_vals = (vis_near * (1. - t) + vis_far * t).unsqueeze(0) # [1, N_samples]

        xyz_sampled = surf_pts_chunk[:, None,None,:] + surf2l[:,:,None,:] * z_vals.view(1, 1, -1, 1) # [points_chunk, envH * envW, N_samples, 3]
        xyz_sampled = xyz_sampled.view(-1,xyz_sampled.shape[-2], 3)  # [points_chunk * envH * envW, N_samples, 3]
        dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)

        sigma = torch.zeros(xyz_sampled.shape[:-1],  device=xyz_sampled.device)

        xyz_sampled = tensorf.normalize_coord(xyz_sampled)
        sigma_feature = tensorf.compute_densityfeature(xyz_sampled.view(-1, 3))

        validsigma = tensorf.feature2density(sigma_feature)
        sigma[:,:] = validsigma.view(-1, N_samples)
        alpha, weight, bg_weight = raw2alpha(sigma, dists * tensorf.distance_scale)
        acc_map = torch.sum(weight, -1) # [points_chunk * envH * envW, ]
        acc_map[acc_map<0.6] = 0

        acc_maps.append(acc_map.detach())

    acc_maps = torch.cat(acc_maps) # # [points_surface * envH * envW, ]

    acc_maps = acc_maps.view(-1, n_lights) # [points_surface, envH * envW]
    visibility_maps = 1 - acc_maps
    return visibility_maps

def compute_normals_on_surface( tensorf, 
                                surface_points,
                                chunk=4096, 
                                device=device
                                ):
    '''
    Given the surface points obtained from pure TensorF, return the normals on the surface.
    '''
    normals_maps = []
    
    N_points_all = surface_points.shape[0]
    for chunk_idx in range(N_points_all // chunk + int(N_points_all % chunk > 0)):
        points_chunk = surface_points[chunk_idx * chunk: (chunk_idx + 1) * chunk]

        # TODO: may need consider NDC rays  

        sigma = torch.zeros(points_chunk.shape[0], device=points_chunk.device)

        points_chunk = tensorf.normalize_coord(points_chunk) # TODO:check if we should use normalize_coord to derive normals
        points_chunk.requires_grad_(True) # used for derive gradient as normals

        # The follow is equivalent to "sigma_feature = tensorf.compute_densityfeature(points_chunk)", !!delete the detach!!
        # It now can only support TensorVMSplit TODO
        # beigin
        coordinate_plane = torch.stack((points_chunk[..., tensorf.matMode[0]], points_chunk[..., tensorf.matMode[1]], points_chunk[..., tensorf.matMode[2]])).view(3, -1, 1, 2)
        coordinate_line = torch.stack((points_chunk[..., tensorf.vecMode[0]], points_chunk[..., tensorf.vecMode[1]], points_chunk[..., tensorf.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).view(3, -1, 1, 2)

        sigma_feature = torch.zeros((points_chunk.shape[0],), device=points_chunk.device)
        for idx_plane in range(len(tensorf.density_plane)):
            plane_coef_point = F.grid_sample(tensorf.density_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *points_chunk.shape[:1])
            line_coef_point = F.grid_sample(tensorf.density_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *points_chunk.shape[:1])
            sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)
        # # end

        sigma = tensorf.feature2density(sigma_feature)

        d_output = torch.ones_like(sigma, requires_grad=False, device=sigma.device)

        gradients = torch.autograd.grad(
                                    outputs=sigma,
                                    inputs=points_chunk,
                                    grad_outputs=d_output,
                                    create_graph=True,
                                    retain_graph=True,
                                    only_inputs=True
                                    )[0]

        normals = - safe_l2_normalize(gradients, dim=-1)
        normals = normals.view(-1, 3)
        normals_maps.append(normals.detach())
    return torch.cat(normals_maps)



def export_geometry(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)


    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    if args.geo_buffer_train:
        store_geometry(train_dataset, tensorf, args, renderer)
    if args.geo_buffer_test:
        test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_test, is_stack=True)
        store_geometry(test_dataset, tensorf, args, renderer)


@torch.no_grad()
def compute_visibilty_and_indirect_light(dataset, tensorf, surface_point, surface_normal, N_samples, args, points_chunk=64, device='cuda'):
    '''
    args:
        dataset: the dataset object
        tensorf: the tensorf object
        surface_point: [N, 3] the surface point location on the surface
        surface_normal: [N, 3] the surface normal on the surface
        N_samples: the number of samples on each ray from surface to light location
    '''
    
    # TODO: can introduce normals to save GPU memory and computation time
    # TODO: handle ndc_rays
    acc_maps = []
    indirect_light_maps = []
    # N_samples = N_samples if N_samples > 0 else tensorf.nSamples
    N_samples = 128 # TODO: add this to config file
    lxyz = dataset.light_xyz.to(device) # [envH * envW, 3]
    lareas = dataset.light_areas.to(device) # [envH * envW,]
    lxyz_flat = lxyz.view(1, -1, 3) # [1, envH * envW, 3]
    n_lights = lxyz_flat.shape[1]

    # lvis_hit = torch.zeros((surface_point.shape[0], n_lights), dtype=torch.float32) # (n_surf_pts, n_lights)
    points_num = surface_point.shape[0]
    idx_chunks = torch.split(torch.arange(points_num), points_chunk)
    for idx_chunk in idx_chunks:
        surf_pts_chunk = surface_point[idx_chunk].to(device) # [points_chunk, 3]
        lxyz_chunk = lxyz_flat.repeat(surf_pts_chunk.shape[0], 1, 1) # [points_chunk, envH * envW, 3]
        normals_chunk = surface_normal[idx_chunk].to(device) # [points_chunk, 3]
        # normals_chunk = normals_chunk.view(-1, 1, 3).repeat(1, n_lights, 1) # [points_chunk, envH * envW, 3]
        # surface to lights
        surf2l = lxyz_chunk - surf_pts_chunk[:, None, :] # [points_chunk, envH * envW, 3]
        surf2l_norm = torch.norm(surf2l, dim=-1, keepdim=True) # [points_chunk, envH * envW, 1]
        surf2l = surf2l / surf2l_norm # [points_chunk, envH * envW, 3]
        
        t = torch.linspace(0., 1., N_samples, device=device) # [N_samples,]
        vis_near = 0.03
        vis_far = 2. # TODO : need to be tuned and write the args into the config file

        z_vals = (vis_near * (1. - t) + vis_far * t).unsqueeze(0) # [1, N_samples]

        xyz_sampled = surf_pts_chunk[:, None,None,:] + surf2l[:,:,None,:] * z_vals.view(1, 1, -1, 1) # [points_chunk, envH * envW, N_samples, 3]
        xyz_sampled = xyz_sampled.view(-1,xyz_sampled.shape[-2], 3)  # [points_chunk * envH * envW, N_samples, 3]
        dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)

        sigma = torch.zeros(xyz_sampled.shape[:-1],  device=xyz_sampled.device)
        indirect_light = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)

        xyz_sampled = tensorf.normalize_coord(xyz_sampled)
        sigma_feature = tensorf.compute_densityfeature(xyz_sampled.view(-1, 3))

        validsigma = tensorf.feature2density(sigma_feature)
        sigma[:,:] = validsigma.view(-1, N_samples)
        alpha, weight, bg_weight = raw2alpha(sigma, dists * tensorf.distance_scale)
        

        app_mask = weight > tensorf.rayMarch_weight_thres

        if app_mask.any():
            viewdirs = surf2l.unsqueeze(2).repeat(1, 1, N_samples, 1)  # [points_chunk, envH * envW, N_samples, 3]
            viewdirs = viewdirs.view(-1, N_samples, 3)  # [points_chunk * envH * envW, N_samples, 3]
            app_features = tensorf.compute_appfeature(xyz_sampled[app_mask])
            valid_rgbs = tensorf.renderModule(xyz_sampled[app_mask], viewdirs[app_mask], app_features)
            indirect_light[app_mask] = valid_rgbs

        acc_map = torch.sum(weight, -1)
        indirect_light_map = torch.sum(weight[..., None] * indirect_light, -2)  # [points_chunk * envH * envW, 3]

        acc_map = torch.sum(weight, -1)
        acc_map[acc_map < 0.6] = 0    
        acc_maps.append(acc_map.detach())
        # surf2l:[points_chunk, envH * envW, 3], normals_chunk:[points_chunk, 3] -> cos:[points_chunk, envH * envW]
        cos = torch.einsum("ijk,ik->ij", surf2l, normals_chunk) 
        cos[cos < 1e-6] = 0 # [points_chunk, envH * envW]
        light_area = lareas.view(1, -1) # [1, envH * envW]
        light_contribution = cos * light_area # [points_chunk, envH * envW]
        light_contribution = light_contribution.view(-1, 1) # [points_chunk * envH * envW, 1]
        indirect_light_map = indirect_light_map * light_contribution  # [points_chunk * envH * envW, 3]
        indirect_light_maps.append(indirect_light_map.detach())
    acc_maps = torch.cat(acc_maps) # # [points_surface * envH * envW, ]
    indirect_light_maps = torch.cat(indirect_light_maps) # [points_surface * envH * envW, 3]
    acc_maps = acc_maps.view(-1, n_lights) # [points_surface, envH * envW]
    indirect_light_maps = indirect_light_maps.view(*acc_maps.shape[:2], 3) # [points_surface, envH * envW, 3]
    visibility_maps = 1 - acc_maps
    return visibility_maps, indirect_light_maps

if __name__ == "__main__":
    args = config_parser()
    print(args)
    export_geometry(args)
