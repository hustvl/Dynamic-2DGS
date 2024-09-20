#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import time
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui, render_flow
import sys
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from train import training_report
import math
from cam_utils import OrbitCamera
import numpy as np
import dearpygui.dearpygui as dpg
import imageio
import datetime
from PIL import Image
from train_gui_utils import DeformKeypoints
from scipy.spatial.transform import Rotation as R
from utils.system_utils import load_config_from_file, merge_config

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 1 / tanHalfFovX
    P[1, 1] = 1 / tanHalfFovY
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def landmark_interpolate(landmarks, steps, step, interpolation='log'):
    stage = (step >= np.array(steps)).sum()
    if stage == len(steps):
        return max(0, landmarks[-1])
    elif stage == 0:
        return 0
    else:
        ldm1, ldm2 = landmarks[stage-1], landmarks[stage]
        if ldm2 <= 0:
            return 0
        step1, step2 = steps[stage-1], steps[stage]
        ratio = (step - step1) / (step2 - step1)
        if interpolation == 'log':
            return np.exp(np.log(ldm1) * (1 - ratio) + np.log(ldm2) * ratio)
        elif interpolation == 'linear':
            return ldm1 * (1 - ratio) + ldm2 * ratio
        else:
            print(f'Unknown interpolation type: {interpolation}')
            raise NotImplementedError

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar, fid):
        # c2w (pose) should be in NeRF convention.

        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.fid = fid
        self.c2w = c2w

        w2c = np.linalg.inv(c2w)

        # rectify...
        w2c[1:3, :3] *= -1
        w2c[:3, 3] *= -1

        self.world_view_transform = torch.tensor(w2c).transpose(0, 1).cuda().float()
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda().float()
        )
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = -torch.tensor(c2w[:3, 3]).cuda()

    def reset_extrinsic(self, R, T):
        self.world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class GUI:
    def __init__(self, args, dataset, opt, pipe, testing_iterations, saving_iterations) -> None:
        self.dataset = dataset
        self.args = args
        self.opt = opt
        self.pipe = pipe
        self.testing_iterations = testing_iterations
        self.saving_iterations = saving_iterations

        if self.opt.progressive_train:
            self.opt.iterations_node_sampling = max(self.opt.iterations_node_sampling, int(self.opt.progressive_stage_steps / self.opt.progressive_stage_ratio))
            self.opt.iterations_node_rendering = max(self.opt.iterations_node_rendering, self.opt.iterations_node_sampling + 2000)
            print(f'Progressive trian is on. Adjusting the iterations node sampling to {self.opt.iterations_node_sampling} and iterations node rendering {self.opt.iterations_node_rendering}')

        self.tb_writer = prepare_output_and_logger(dataset)
        self.deform = DeformModel(K=self.dataset.K, deform_type=self.dataset.deform_type, is_blender=self.dataset.is_blender, skinning=self.args.skinning, hyper_dim=self.dataset.hyper_dim, node_num=self.dataset.node_num, pred_opacity=self.dataset.pred_opacity, pred_color=self.dataset.pred_color, use_hash=self.dataset.use_hash, hash_time=self.dataset.hash_time, d_rot_as_res=self.dataset.d_rot_as_res and not self.dataset.d_rot_as_rotmat, local_frame=self.dataset.local_frame, progressive_brand_time=self.dataset.progressive_brand_time, with_arap_loss=not self.opt.no_arap_loss, max_d_scale=self.dataset.max_d_scale, enable_densify_prune=self.opt.node_enable_densify_prune, is_scene_static=dataset.is_scene_static)
        deform_loaded = self.deform.load_weights(dataset.model_path, iteration=-1)
        self.deform.train_setting(opt)

        gs_fea_dim = self.deform.deform.node_num if args.skinning and self.deform.name == 'node' else self.dataset.hyper_dim
        self.gaussians = GaussianModel(dataset.sh_degree, fea_dim=gs_fea_dim, with_motion_mask=self.dataset.gs_with_motion_mask)

        self.scene = Scene(dataset, self.gaussians, load_iteration=-1)
        self.gaussians.training_setup(opt)
        if self.deform.name == 'node' and not deform_loaded:
            if not self.dataset.is_blender:
                if self.opt.random_init_deform_gs:
                    num_pts = 100_000
                    print(f"Generating random point cloud ({num_pts})...")
                    xyz = torch.rand((num_pts, 3)).float().cuda() * 2 - 1
                    mean, scale = self.gaussians.get_xyz.mean(dim=0), self.gaussians.get_xyz.std(dim=0).mean() * 3
                    xyz = xyz * scale + mean
                    self.deform.deform.init(init_pcl=xyz, force_init=True, opt=self.opt, as_gs_force_with_motion_mask=self.dataset.as_gs_force_with_motion_mask, force_gs_keep_all=True)
                else:
                    print('Initialize nodes with COLMAP point cloud.')
                    self.deform.deform.init(init_pcl=self.gaussians.get_xyz, force_init=True, opt=self.opt, as_gs_force_with_motion_mask=self.dataset.as_gs_force_with_motion_mask, force_gs_keep_all=self.dataset.init_isotropic_gs_with_all_colmap_pcl)
            else:
                print('Initialize nodes with Random point cloud.')
                self.deform.deform.init(init_pcl=self.gaussians.get_xyz, force_init=True, opt=self.opt, as_gs_force_with_motion_mask=False, force_gs_keep_all=args.skinning)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.iter_start = torch.cuda.Event(enable_timing=True)
        self.iter_end = torch.cuda.Event(enable_timing=True)
        self.iteration = 1 if self.scene.loaded_iter is None else self.scene.loaded_iter
        self.iteration_node_rendering = 1 if self.scene.loaded_iter is None else self.opt.iterations_node_rendering

        self.viewpoint_stack = None
        self.ema_loss_for_log = 0.0
        self.best_psnr = 0.0
        self.best_ssim = 0.0
        self.best_ms_ssim = 0.0
        self.best_lpips = np.inf
        self.best_alex_lpips = np.inf
        self.best_iteration = 0
        self.progress_bar = tqdm.tqdm(range(opt.iterations), desc="Training progress")
        self.smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)

        # For UI
        self.visualization_mode = 'RGB'

        #self.gui = args.gui # enable gui
        self.W = args.W
        self.H = args.H
        self.cam = OrbitCamera(args.W, args.H, r=args.radius, fovy=args.fovy)
        self.mode = "render"
        self.seed = "random"
        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.training = False



    
    # no gui mode
    def train(self, iters=5000):
        if iters > 0:
            for i in tqdm.trange(iters):
                if self.deform.name == 'node' and self.iteration_node_rendering < self.opt.iterations_node_rendering:
                    self.train_node_rendering_step()
                else:
                    self.train_step()
    
    def train_step(self):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, self.pipe.do_shs_python, self.pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, self.gaussians, self.pipe, self.background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, self.dataset.source_path)
                if do_training and ((self.iteration < int(self.opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        self.iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if self.iteration % self.opt.oneupSHdegree_step == 0:
            self.gaussians.oneupSHdegree()

        # Pick a random Camera
        if not self.viewpoint_stack:
            if self.opt.progressive_train and self.iteration < int(self.opt.progressive_stage_steps / self.opt.progressive_stage_ratio):
                cameras_to_train_idx = int(min(((self.iteration) / self.opt.progressive_stage_steps + 1) * self.opt.progressive_stage_ratio, 1.) * len(self.scene.getTrainCameras()))
                cameras_to_train_idx = max(cameras_to_train_idx, 1)
                interval_len = int(len(self.scene.getTrainCameras()) * self.opt.progressive_stage_ratio)
                min_idx = max(0, cameras_to_train_idx - interval_len)
                sorted_train_cams = sorted(self.scene.getTrainCameras().copy(), key=lambda x: x.fid)
                viewpoint_stack = sorted_train_cams[min_idx: cameras_to_train_idx]
                out_domain_idx = np.arange(min_idx)
                if len(out_domain_idx) >= interval_len:
                    out_domain_idx = np.random.choice(out_domain_idx, [interval_len], replace=False)
                    out_domain_stack = [sorted_train_cams[idx] for idx in out_domain_idx]
                    viewpoint_stack = viewpoint_stack + out_domain_stack
            else:
                viewpoint_stack = self.scene.getTrainCameras().copy()
            self.viewpoint_stack = viewpoint_stack
        
        total_frame = len(self.scene.getTrainCameras())
        time_interval = 1 / total_frame

        viewpoint_cam = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack) - 1))
        if self.dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid

        if self.deform.name == 'mlp' or self.deform.name == 'static':
            if self.iteration < self.opt.warm_up:
                d_xyz, d_rotation, d_scaling, d_opacity, d_color = 0.0, 0.0, 0.0, 0.0, 0.0
            else:
                N = self.gaussians.get_xyz.shape[0]
                time_input = fid.unsqueeze(0).expand(N, -1)
                ast_noise = 0 if self.dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * self.smooth_term(self.iteration)
                d_values = self.deform.step(self.gaussians.get_xyz.detach(), time_input + ast_noise, iteration=self.iteration, feature=self.gaussians.feature, camera_center=viewpoint_cam.camera_center)
                d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], d_values['d_scaling'], d_values['d_opacity'], d_values['d_color']
        elif self.deform.name == 'node':
            if not self.deform.deform.inited:
                print('Notice that warping nodes are initialized with Gaussians!!!')
                self.deform.deform.init(self.opt, self.gaussians.get_xyz.detach(), feature=self.gaussians.feature)
            time_input = self.deform.deform.expand_time(fid)
            N = time_input.shape[0]
            ast_noise = 0 if self.dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * self.smooth_term(self.iteration)
            d_values = self.deform.step(self.gaussians.get_xyz.detach(), time_input + ast_noise, iteration=self.iteration, feature=self.gaussians.feature, motion_mask=self.gaussians.motion_mask, camera_center=viewpoint_cam.camera_center, time_interval=time_interval)
            d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], d_values['d_scaling'], d_values['d_opacity'], d_values['d_color']
            if self.iteration < self.opt.warm_up:
                d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_xyz.detach(), d_rotation.detach(), d_scaling.detach(), d_opacity.detach() if d_opacity is not None else None, d_color.detach() if d_color is not None else None
            elif self.iteration < self.opt.dynamic_color_warm_up:
                d_color = d_color.detach() if d_color is not None else None

        # Render
        random_bg_color = (not self.dataset.white_background and self.opt.random_bg_color) and self.opt.gt_alpha_mask_as_scene_mask and viewpoint_cam.gt_alpha_mask is not None
        render_pkg_re = render(viewpoint_cam, self.gaussians, self.pipe, self.background, d_xyz, d_rotation, d_scaling, random_bg_color=random_bg_color, d_opacity=d_opacity, d_color=d_color, d_rot_as_res=self.deform.d_rot_as_res)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re["viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]
        

        lambda_normal = 0.02 if self.iteration > 8000 else 0.0
        lambda_dist = 1000 if self.iteration > 8000 else 0.0
        rend_dist = render_pkg_re["rend_dist"]
        rend_normal  = render_pkg_re['rend_normal']
        surf_normal = render_pkg_re['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()
        
        
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        if random_bg_color:
            gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
            gt_image = gt_alpha_mask * gt_image + (1 - gt_alpha_mask) * render_pkg_re['bg_color'][:, None, None]
        elif self.dataset.white_background and viewpoint_cam.gt_alpha_mask is not None and self.opt.gt_alpha_mask_as_scene_mask:
            gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
            gt_image = gt_alpha_mask * gt_image + (1 - gt_alpha_mask) * self.background[:, None, None]

        Ll1 = l1_loss(image, gt_image)
        loss_img = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss = loss_img + normal_loss + dist_loss

        if self.iteration > self.opt.warm_up:
            loss = loss + self.deform.reg_loss

        # Flow loss
        flow_id2_candidates = viewpoint_cam.flow_dirs
        lambda_optical = landmark_interpolate(self.opt.lambda_optical_landmarks, self.opt.lambda_optical_steps, self.iteration)
        if flow_id2_candidates != [] and lambda_optical > 0 and self.iteration >= self.opt.warm_up:
            # Pick flow file and read it
            flow_id2_dir = np.random.choice(flow_id2_candidates)
            flow = np.load(flow_id2_dir)
            mask_id2_dir = flow_id2_dir.replace('raft_neighbouring', 'raft_masks').replace('.npy', '.png')
            masks = imageio.imread(mask_id2_dir) / 255.
            flow = torch.from_numpy(flow).float().cuda()
            masks = torch.from_numpy(masks).float().cuda()
            if flow.shape[0] != image.shape[1] or flow.shape[1] != image.shape[2]:
                flow = torch.nn.functional.interpolate(flow.permute([2, 0, 1])[None], (image.shape[1], image.shape[2]))[0].permute(1, 2, 0)
                masks = torch.nn.functional.interpolate(masks.permute([2, 0, 1])[None], (image.shape[1], image.shape[2]))[0].permute(1, 2, 0)
            fid1 = viewpoint_cam.fid
            cam2_id = os.path.basename(flow_id2_dir).split('_')[-1].split('.')[0]
            if not hasattr(self, 'img2cam'):
                self.img2cam = {cam.image_name: idx for idx, cam in enumerate(self.scene.getTrainCameras().copy())}
            if cam2_id in self.img2cam:  # Only considering the case with existing files
                cam2_id = self.img2cam[cam2_id]
                viewpoint_cam2 = self.scene.getTrainCameras().copy()[cam2_id]
                fid2 = viewpoint_cam2.fid
                # Calculate the GT flow, weight, and mask
                coor1to2_flow = flow / torch.tensor(flow.shape[:2][::-1], dtype=torch.float32).cuda() * 2
                cycle_consistency_mask = masks[..., 0] > 0
                occlusion_mask = masks[..., 1] > 0
                mask_flow = cycle_consistency_mask | occlusion_mask
                pair_weight = torch.clamp(torch.cos((fid1 - fid2).abs() * np.pi / 2), 0.2, 1)
                # Calculate the motion at t2
                time_input2 = self.deform.deform.expand_time(fid2)
                ast_noise = 0 if self.dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * self.smooth_term(self.iteration)
                d_xyz2 = self.deform.step(self.gaussians.get_xyz.detach(), time_input2 + ast_noise, iteration=self.iteration, feature=self.gaussians.feature, motion_mask=self.gaussians.motion_mask, camera_center=viewpoint_cam2.camera_center)['d_xyz']
                # Render the flow image
                render_pkg2 = render_flow(pc=self.gaussians, viewpoint_camera1=viewpoint_cam, viewpoint_camera2=viewpoint_cam2, d_xyz1=d_xyz, d_xyz2=d_xyz2, d_rotation1=d_rotation, d_scaling1=d_scaling, scale_const=None)
                coor1to2_motion = render_pkg2["render"][:2].permute(1, 2, 0)
                mask_motion = (render_pkg2['alpha'][0] > .9).detach()  # Only optimizing the space with solid points to avoid dilation
                mask = (mask_motion & mask_flow)[..., None] * pair_weight
                # Flow loss based on pixel rgb loss
                l1_loss_weight = (image.detach() - gt_image).abs().mean(dim=0)
                l1_loss_weight = torch.cos(l1_loss_weight * torch.pi / 2)
                mask = mask * l1_loss_weight[..., None]
                # Flow mask
                optical_flow_loss = l1_loss(mask * coor1to2_flow, mask * coor1to2_motion)
                loss = loss + lambda_optical * optical_flow_loss

        # Motion Mask Loss
        lambda_motion_mask = landmark_interpolate(self.opt.lambda_motion_mask_landmarks, self.opt.lambda_motion_mask_steps, self.iteration)
        if not self.opt.no_motion_mask_loss and self.deform.name == 'node' and self.opt.gt_alpha_mask_as_dynamic_mask and viewpoint_cam.gt_alpha_mask is not None and lambda_motion_mask > 0:
            gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
            render_pkg_motion = render(viewpoint_cam, self.gaussians, self.pipe, self.background, d_xyz, d_rotation, d_scaling, random_bg_color=random_bg_color, render_motion=True, detach_xyz=True, detach_rot=True, detach_scale=True, detach_opacity=True, d_opacity=d_opacity, d_color=d_color, d_rot_as_res=self.deform.d_rot_as_res)
            motion_image = render_pkg_motion["render"][0]
            L_motion = l1_loss(gt_alpha_mask, motion_image)
            loss = loss + L_motion * lambda_motion_mask

        loss.backward()

        self.iter_end.record()

        if self.dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        with torch.no_grad():
            # Progress bar
            self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
            if self.iteration % 10 == 0:
                self.progress_bar.set_postfix({"Loss": f"{self.ema_loss_for_log:.{7}f}"})
                self.progress_bar.update(10)
            if self.iteration == self.opt.iterations:
                self.progress_bar.close()

            # Keep track of max radii in image-space for pruning
            if self.gaussians.max_radii2D.shape[0] == 0:
                self.gaussians.max_radii2D = torch.zeros_like(radii)
            self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

            # Log and save
            cur_psnr, cur_ssim, cur_lpips, cur_ms_ssim, cur_alex_lpips = training_report(self.tb_writer, self.iteration, Ll1, loss, l1_loss, self.iter_start.elapsed_time(self.iter_end), self.testing_iterations, self.scene, render, (self.pipe, self.background), self.deform, self.dataset.load2gpu_on_the_fly, self.progress_bar)
            if self.iteration in self.testing_iterations:
                if cur_psnr.item() > self.best_psnr:
                    self.best_psnr = cur_psnr.item()
                    self.best_iteration = self.iteration
                    self.best_ssim = cur_ssim.item()
                    self.best_ms_ssim = cur_ms_ssim.item()
                    self.best_lpips = cur_lpips.item()
                    self.best_alex_lpips = cur_alex_lpips.item()

            if self.iteration in self.saving_iterations or self.iteration == self.best_iteration or self.iteration == self.opt.warm_up-1:
                print("\n[ITER {}] Saving Gaussians".format(self.iteration))
                self.scene.save(self.iteration)
                self.deform.save_weights(self.args.model_path, self.iteration)

            # Densification
            if self.iteration < self.opt.densify_until_iter:
                self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if self.iteration > self.opt.node_densify_from_iter and self.iteration % self.opt.node_densification_interval == 0 and self.iteration < self.opt.node_densify_until_iter and self.iteration > self.opt.warm_up or self.iteration == self.opt.node_force_densify_prune_step:
                    # Nodes densify
                    self.deform.densify(max_grad=self.opt.densify_grad_threshold, x=self.gaussians.get_xyz, x_grad=self.gaussians.xyz_gradient_accum / self.gaussians.denom, feature=self.gaussians.feature, force_dp=(self.iteration == self.opt.node_force_densify_prune_step))

                if self.iteration > self.opt.densify_from_iter and self.iteration % self.opt.densification_interval == 0:
                    size_threshold = 20 if self.iteration > self.opt.opacity_reset_interval else None
                    self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, 0.01, self.scene.cameras_extent, size_threshold)

                if self.iteration % self.opt.opacity_reset_interval == 0 or (
                        self.dataset.white_background and self.iteration == self.opt.densify_from_iter):
                    self.gaussians.reset_opacity()

            # Optimizer step
            if self.iteration < self.opt.iterations:
                self.gaussians.optimizer.step()
                self.gaussians.update_learning_rate(self.iteration)
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.deform.optimizer.step()
                self.deform.optimizer.zero_grad()
                self.deform.update_learning_rate(self.iteration)
                
        self.deform.update(max(0, self.iteration - self.opt.warm_up))


        self.progress_bar.set_description("Best PSNR={} in Iteration {}, SSIM={}, LPIPS={}, MS-SSIM={}, ALex-LPIPS={}".format('%.5f' % self.best_psnr, self.best_iteration, '%.5f' % self.best_ssim, '%.5f' % self.best_lpips, '%.5f' % self.best_ms_ssim, '%.5f' % self.best_alex_lpips))
        self.iteration += 1

   
    def train_node_rendering_step(self):
        # Pick a random Camera
        if not self.viewpoint_stack:
            if self.opt.progressive_train_node and self.iteration_node_rendering < int(self.opt.progressive_stage_steps / self.opt.progressive_stage_ratio) + self.opt.node_warm_up:
                if self.iteration_node_rendering < self.opt.node_warm_up:
                    sorted_train_cams = sorted(self.scene.getTrainCameras().copy(), key=lambda x: x.fid)
                    max_cam_num = max(30, int(0.01 * len(sorted_train_cams)))
                    viewpoint_stack = sorted_train_cams[0: max_cam_num]
                else:
                    cameras_to_train_idx = int(min(((self.iteration_node_rendering - self.opt.node_warm_up) / self.opt.progressive_stage_steps + 1) * self.opt.progressive_stage_ratio, 1.) * len(self.scene.getTrainCameras()))
                    cameras_to_train_idx = max(cameras_to_train_idx, 1)
                    interval_len = int(len(self.scene.getTrainCameras()) * self.opt.progressive_stage_ratio)
                    min_idx = max(0, cameras_to_train_idx - interval_len)
                    sorted_train_cams = sorted(self.scene.getTrainCameras().copy(), key=lambda x: x.fid)
                    viewpoint_stack = sorted_train_cams[min_idx: cameras_to_train_idx]
                    out_domain_idx = np.concatenate([np.arange(min_idx), np.arange(cameras_to_train_idx, min(len(self.scene.getTrainCameras()), cameras_to_train_idx+interval_len))])
                    if len(out_domain_idx) >= interval_len:
                        out_domain_len = min(interval_len*5, len(out_domain_idx))
                        out_domain_idx = np.random.choice(out_domain_idx, [out_domain_len], replace=False)
                        out_domain_stack = [sorted_train_cams[idx] for idx in out_domain_idx]
                        viewpoint_stack = viewpoint_stack + out_domain_stack
            else:
                viewpoint_stack = self.scene.getTrainCameras().copy()
            self.viewpoint_stack = viewpoint_stack

        viewpoint_cam = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack) - 1))
        if self.dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid
        
        time_input = fid.unsqueeze(0).expand(self.deform.deform.as_gaussians.get_xyz.shape[0], -1)
        N = time_input.shape[0]

        total_frame = len(self.scene.getTrainCameras())
        time_interval = 1 / total_frame

        ast_noise = 0 if self.dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * self.smooth_term(self.iteration_node_rendering)
        d_values = self.deform.deform.query_network(x=self.deform.deform.as_gaussians.get_xyz.detach(), t=time_input + ast_noise)
        d_xyz, d_opacity, d_color = d_values['d_xyz'] * self.deform.deform.as_gaussians.motion_mask, d_values['d_opacity'] * self.deform.deform.as_gaussians.motion_mask if d_values['d_opacity'] is not None else None, d_values['d_color'] * self.deform.deform.as_gaussians.motion_mask if d_values['d_color'] is not None else None
        d_rot, d_scale = 0., 0.
        if self.iteration_node_rendering < self.opt.node_warm_up:
            d_xyz = d_xyz.detach()
        d_color = d_color.detach() if d_color is not None else None
        d_opacity = d_opacity.detach() if d_opacity is not None else None

        # Render
        random_bg_color = (self.opt.gt_alpha_mask_as_scene_mask or (self.opt.gt_alpha_mask_as_dynamic_mask and not self.deform.deform.as_gaussians.with_motion_mask)) and viewpoint_cam.gt_alpha_mask is not None
        render_pkg_re = render(viewpoint_cam, self.deform.deform.as_gaussians, self.pipe, self.background, d_xyz, d_rot, d_scale, random_bg_color=random_bg_color, d_opacity=d_opacity, d_color=d_color, d_rot_as_res=self.deform.d_rot_as_res)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re["viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        if random_bg_color:
            gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
            gt_image = gt_image * gt_alpha_mask + render_pkg_re['bg_color'][:, None, None] * (1 - gt_alpha_mask)
        Ll1 = l1_loss(image, gt_image)
        loss_img = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss = loss_img

        if self.iteration_node_rendering > self.opt.node_warm_up:
            if not self.deform.deform.use_hash:
                elastic_loss = 1e-3 * self.deform.deform.elastic_loss(t=fid, delta_t=time_interval)
                loss_acc = 1e-5 * self.deform.deform.acc_loss(t=fid, delta_t=3*time_interval)
                loss = loss + elastic_loss + loss_acc
            if not self.opt.no_arap_loss:
                loss_opt_trans = 1e-2 * self.deform.deform.arap_loss()
                loss = loss + loss_opt_trans

        # Motion Mask Loss
        if self.opt.gt_alpha_mask_as_dynamic_mask and self.deform.deform.as_gaussians.with_motion_mask and viewpoint_cam.gt_alpha_mask is not None and self.iteration_node_rendering > self.opt.node_warm_up:
            gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()[0]
            render_pkg_motion = render(viewpoint_cam, self.deform.deform.as_gaussians, self.pipe, self.background, d_xyz, d_rot, d_scale, render_motion=True, detach_xyz=True, detach_rot=True, detach_scale=True, detach_opacity=self.deform.deform.as_gaussians.with_motion_mask, d_opacity=d_opacity, d_color=d_color, d_rot_as_res=self.deform.d_rot_as_res)
            motion_image = render_pkg_motion["render"][0]
            L_motion = l1_loss(gt_alpha_mask, motion_image)
            loss = loss + L_motion

        loss.backward()
        with torch.no_grad():
            if self.iteration_node_rendering < self.opt.iterations_node_sampling:
                # Densification
                self.deform.deform.as_gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if self.iteration_node_rendering % self.opt.densification_interval == 0 or self.iteration_node_rendering == self.opt.node_warm_up - 1:
                    size_threshold = 20 if self.iteration_node_rendering > self.opt.opacity_reset_interval else None
                    if self.dataset.is_blender:
                        grad_max = self.opt.densify_grad_threshold
                    else:
                        if self.deform.deform.as_gaussians.get_xyz.shape[0] > self.deform.deform.node_num * self.opt.node_max_num_ratio_during_init:
                            grad_max = torch.inf
                        else:
                            grad_max = self.opt.densify_grad_threshold
                    self.deform.deform.as_gaussians.densify_and_prune(grad_max, 0.005, self.scene.cameras_extent, size_threshold)
                if self.iteration_node_rendering % self.opt.opacity_reset_interval == 0 or (
                        self.dataset.white_background and self.iteration_node_rendering == self.opt.densify_from_iter):
                    self.deform.deform.as_gaussians.reset_opacity()
            elif self.iteration_node_rendering == self.opt.iterations_node_sampling:
                # Downsampling nodes for sparse control
                # Strategy 1: Directly use the original gs as nodes
                # Strategy 2: Sampling in the hyper space across times
                strategy = self.opt.deform_downsamp_strategy
                if strategy == 'direct':
                    original_gaussians: GaussianModel = self.deform.deform.as_gaussians
                    self.deform.deform.init(opt=self.opt, init_pcl=original_gaussians.get_xyz, keep_all=True, force_init=True, reset_bbox=False, feature=self.gaussians.feature)
                    gaussians: GaussianModel = self.deform.deform.as_gaussians
                    gaussians._features_dc = torch.nn.Parameter(original_gaussians._features_dc)
                    gaussians._features_rest = torch.nn.Parameter(original_gaussians._features_rest)
                    gaussians._scaling = torch.nn.Parameter(original_gaussians._scaling)
                    gaussians._opacity = torch.nn.Parameter(original_gaussians._opacity)
                    gaussians._rotation = torch.nn.Parameter(original_gaussians._rotation)
                    if gaussians.fea_dim > 0:
                        gaussians.feature = torch.nn.Parameter(original_gaussians.feature)
                    print('Reset the optimizer of the deform model.')
                    self.deform.train_setting(self.opt)
                elif strategy == 'samp_hyper':
                    original_gaussians: GaussianModel = self.deform.deform.as_gaussians
                    time_num = 16
                    t_samp = torch.linspace(0, 1, time_num).cuda()
                    x = original_gaussians.get_xyz.detach()
                    trans_samp = []
                    for i in range(time_num):
                        time_input = t_samp[i:i+1, None].expand_as(x[..., :1])
                        trans_samp.append(self.deform.deform.query_network(x=x, t=time_input)['d_xyz'] * original_gaussians.motion_mask)
                    trans_samp = torch.stack(trans_samp, dim=1)
                    hyper_pcl = (trans_samp + original_gaussians.get_xyz[:, None]).reshape([original_gaussians.get_xyz.shape[0], -1])
                    dynamic_mask = self.deform.deform.as_gaussians.motion_mask[..., 0] > .5
                    if not self.opt.deform_downsamp_with_dynamic_mask:
                        dynamic_mask = torch.ones_like(dynamic_mask)
                    idx = self.deform.deform.init(init_pcl=original_gaussians.get_xyz[dynamic_mask], hyper_pcl=hyper_pcl[dynamic_mask], force_init=True, opt=self.opt, reset_bbox=False, feature=self.gaussians.feature)
                    gaussians: GaussianModel = self.deform.deform.as_gaussians
                    gaussians._features_dc = torch.nn.Parameter(original_gaussians._features_dc[dynamic_mask][idx])
                    gaussians._features_rest = torch.nn.Parameter(original_gaussians._features_rest[dynamic_mask][idx])
                    gaussians._scaling = torch.nn.Parameter(original_gaussians._scaling[dynamic_mask][idx])
                    gaussians._opacity = torch.nn.Parameter(original_gaussians._opacity[dynamic_mask][idx])
                    gaussians._rotation = torch.nn.Parameter(original_gaussians._rotation[dynamic_mask][idx])
                    if gaussians.fea_dim > 0:
                        gaussians.feature = torch.nn.Parameter(original_gaussians.feature[dynamic_mask][idx])
                    gaussians.training_setup(self.opt)
                # No update at the step
                self.deform.deform.as_gaussians.optimizer.zero_grad(set_to_none=True)
                self.deform.optimizer.zero_grad()

            if self.iteration_node_rendering == self.opt.iterations_node_rendering - 1 and self.iteration_node_rendering > self.opt.iterations_node_sampling:
                # Just finish node training and has down sampled control nodes
                self.deform.deform.nodes.data[..., :3] = self.deform.deform.as_gaussians._xyz

            if not self.iteration_node_rendering == self.opt.iterations_node_sampling and not self.iteration_node_rendering == self.opt.iterations_node_rendering - 1:
                # Optimizer step
                self.deform.deform.as_gaussians.optimizer.step()
                self.deform.deform.as_gaussians.update_learning_rate(self.iteration_node_rendering)
                self.deform.deform.as_gaussians.optimizer.zero_grad(set_to_none=True)
                self.deform.update_learning_rate(self.iteration_node_rendering)
                self.deform.optimizer.step()
                self.deform.optimizer.zero_grad()
        
        self.deform.update(max(0, self.iteration_node_rendering - self.opt.node_warm_up))

        if self.dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        self.iteration_node_rendering += 1


    

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--elevation', type=float, default=0, help="default GUI camera elevation")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[5000, 6000, 7_000] + list(range(8000, 100_0001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 40000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--deform-type", type=str, default='mlp')
    parser.add_argument("--white_background2", default=False, type=bool, help='Mesh: resolution for unbounded mesh extraction')
    parser.add_argument("--config", type=str, default=None)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)


    if not args.model_path.endswith(args.deform_type):
        args.model_path = os.path.join(os.path.dirname(os.path.normpath(args.model_path)), os.path.basename(os.path.normpath(args.model_path)) + f'_{args.deform_type}')
    
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    gui = GUI(args=args, dataset=lp.extract(args), opt=op.extract(args), pipe=pp.extract(args),testing_iterations=args.test_iterations, saving_iterations=args.save_iterations)


    gui.train(args.iterations)
    
    # All done
    print("\nTraining complete.")
