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

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scipy.spatial.transform import Rotation as R
from utils.pose_utils import rotation2quad, get_tensor_from_camera, get_tensor_from_camera_torch, get_camera_from_tensor
from utils.graphics_utils import getWorld2View2
from scene.per_point_adam import PerPointAdam
#from scene.camtrans import MLPLip as MLP

def save_pose(path, quat_pose, train_cams, llffhold=2):
    # Get camera IDs and convert quaternion poses to camera matrices
    camera_ids = [cam.colmap_id for cam in train_cams]
    world_to_camera = [get_camera_from_tensor(quat) for quat in quat_pose]
    
    # Reorder poses according to colmap IDs
    colmap_poses = []
    for i in range(len(camera_ids)):
        idx = camera_ids.index(i + 1)  # Find position of camera i+1
        pose = world_to_camera[idx]
        colmap_poses.append(pose)
    
    # Convert to numpy array and save
    colmap_poses = torch.stack(colmap_poses).detach().cpu().numpy()
    np.save(path, colmap_poses)

def load_pose(path, num_poses=None):
    colmap_poses = []
    quat_pose = []
    
    loaded_poses = np.load(path)
    #breakpoint()
    pose_optimized = {name: loaded_poses[name] for name in loaded_poses.files}
    sorted_pose_optimized = dict(sorted(pose_optimized.items(), key=lambda x: x[0]))

    if num_poses is not None:
        sorted_pose_optimized = dict(list(sorted_pose_optimized.items())[:num_poses])
        
    for img_name,pose in sorted_pose_optimized.items():
        pose_final = pose
        R = pose[:3, :3]
        pose_final[:3, :3] = R.T
        q_pose = get_tensor_from_camera(pose_final)
        quat_pose.append(q_pose)
        
    poses = torch.stack(quat_pose)
    #breakpoint()
    return poses

# def load_pose(path):
#     colmap_poses = []
#     quat_pose = []
    
#     loaded_poses = np.load(path)
#     #breakpoint()
#     pose_optimized = {name: loaded_poses[name] for name in loaded_poses.files}
#     sorted_pose_optimized = dict(sorted(pose_optimized.items(), key=lambda x: x[0]))
#     #breakpoint()
#     #sorted_pose_optimized = 
#     for img_name,pose in sorted_pose_optimized.items():
#         q_pose = get_tensor_from_camera(pose.transpose(0, 1))#
#         quat_pose.append(q_pose)
#     #breakpoint()
#     poses = torch.stack(quat_pose)
#     return poses



class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self._delta_x = torch.empty(0)
        self._delta_c = torch.empty(0)
        #deta_s, deta_c, deta_o
        self._delta_s = torch.empty(0)
        self._delta_r = torch.empty(0)
        self._delta_o = torch.empty(0)
        self.optimizer = None
        self.nn_optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        # self.bg_color = torch.empty(0)
        # self.confidence = torch.empty(0)

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self._delta_x,
            self._delta_c,
            #deta_s, deta_c, deta_o
            self._delta_s,
            self._delta_r,
            self._delta_o,
            #self._cam_scale,
            self.optimizer.state_dict(),
            #self._mlp.state_dict(),
            #self.mlp_optimizer.state_dict(),
            self.spatial_lr_scale,
            self.P,
            self.relative_pose
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
        self._xyz,
        self._features_dc,
        self._features_rest,
        self._scaling,
        self._rotation,
        self._opacity,
        self.max_radii2D,
        xyz_gradient_accum,
        denom,
        self._delta_x,
        self._delta_c,
        #deta_s, deta_c, deta_o
        self._delta_s,
        self._delta_r,
        self._delta_o,
        #self._cam_scale,
        opt_dict,
        self.spatial_lr_scale,
        self.P,
        self.relative_pose
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        #self._mlp.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_delta_x(self):
        return self._delta_x
    
    @property
    def get_delta_c(self):
        return self._delta_c
    
    @property
    def get_delta_s(self):
        return self._delta_s
    
    @property
    def get_delta_r(self):
        return self._delta_r
    
    @property
    def get_delta_o(self):
        return self._delta_o
    
    # @property
    # def get_cam_scale(self):
    #     return self._cam_scale
    
    # @property
    # def get_relative_pose(self):
    #     return self.relative_pose
    
    # @property
    # def get_cam_offest(self):
    #     return self._cam_offset
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def init_RT_seq(self, cam_list):
        poses =[]
        sorted_cam_list = sorted(cam_list[1.0], key=lambda x: x.image_name)
        for cam in sorted_cam_list:
            #breakpoint()
            if cam.image_name.startswith('u') and not cam.image_name.startswith('w'):
                p = get_tensor_from_camera(cam.world_view_transform.transpose(0, 1)) # R^T t -> quat t
                poses.append(p)
        poses = torch.stack(poses)
        self.P = poses.cuda().requires_grad_(True)


    def get_RT(self, idx):
        pose = self.P[idx]
        return pose
    
    def get_RT_test(self, idx):
        pose = self.test_P[idx]
        return pose

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        delta_x = torch.zeros_like(fused_point_cloud, device="cuda")
        #應該是跟隨color
        delta_c = torch.zeros_like(features, device="cuda")
        #print("delta_c.shape",delta_c.shape)
        
        delta_s = torch.zeros_like(scales, device="cuda")
        delta_r = torch.zeros_like(rots, device="cuda")
        delta_o = torch.zeros_like(opacities, device="cuda")
        #調整pose的scale
        #cam_scale = torch.tensor([0.0008]).cuda().requires_grad_(True)
        #cam_offset = torch.tensor([0.08]).cuda().requires_grad_(True)
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._delta_x = nn.Parameter(delta_x.requires_grad_(True))
        self._delta_c = nn.Parameter(delta_c.requires_grad_(True))
        self._delta_s = nn.Parameter(delta_s.requires_grad_(True))
        self._delta_r = nn.Parameter(delta_r.requires_grad_(True))
        
        self._delta_o = nn.Parameter(delta_o.requires_grad_(True))
        
        #self._mlp = MLP(58 + 1, 3).to("cuda")
        #self._cam_scale = nn.Parameter(cam_scale.requires_grad_(True))
        #self._cam_offset = nn.Parameter(cam_offset.requires_grad_(True))
        
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        #self.confidence = torch.ones_like(opacities, device="cuda")
        # if self.args.train_bg:
        #     self.bg_color = nn.Parameter((torch.zeros(3, 1, 1) + 0.).cuda().requires_grad_(True))

    def training_setup(self, training_args, stage, Q_rel):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._delta_x], 'lr': training_args.delta_x_lr, "name": "delta_x"},
            {'params': [self._delta_c], 'lr': training_args.delta_c_lr, "name": "delta_c"},
            {'params': [self._delta_s], 'lr': training_args.delta_s_lr, "name": "delta_s"},
            {'params': [self._delta_r], 'lr': training_args.delta_r_lr, "name": "delta_r"},
            {'params': [self._delta_o], 'lr': training_args.delta_o_lr, "name": "delta_o"}
        ]
        # if self.args.train_bg:
        #     l.append({'params': [self.bg_color], 'lr': 0.001, "name": "bg_color"})
        l_cam = [{'params': [self.P],'lr': 1e-4, "name": "pose"}]
        l += l_cam
        #grads = list(self._mlp.parameters())
        #l_cam_scale = [{'params': [self._cam_scale],'lr': 1e-2, "name": "cam_scale"}]
        
        
        #breakpoint()
        
        # self._cam_scale = torch.tensor([0.008]).cuda().requires_grad_(True)
        #if stage == "uw2wide":
        self.relative_pose = nn.Parameter(torch.as_tensor(Q_rel, device='cuda'))
        #breakpoint()
        l_cam_relative = [{'params': [self.relative_pose],'lr': 1e-4, "name": "relative_pose"}]
        l += l_cam_relative


        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        #training_args.rotation_lr*0.1,
        #self.mlp_optimizer = torch.optim.Adam(grads, lr=1e-3)
        #self.mlp_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.mlp_optimizer, milestones=[10000, 15000, 20000], gamma=0.1)
        #self.mlp_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.mlp_optimizer, milestones=[2500, 5000, 7500, 10000], gamma=0.1)
        #training_args.rotation_lr*0.01
        self.cam_scheduler_args = get_expon_lr_func(lr_init=1e-4,
                                                    lr_final=1e-6,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.iterations)
    # per-point optimizer
    def training_setup_pp(self, training_args, stage, Q_rel, confidence_lr=None):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.per_point_lr = confidence_lr

        l = [
            {'params': [self._xyz], 'per_point_lr': self.per_point_lr, 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr * 10, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0 * 10, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr * 10, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr * 10, "name": "rotation"}
        ]

        l_cam = [{'params': [self.P],'lr': training_args.rotation_lr*0.1, "name": "pose"},]
        l += l_cam
        self.relative_pose = nn.Parameter(torch.as_tensor(Q_rel, device='cuda'))
        #breakpoint()
        l_cam_relative = [{'params': [self.relative_pose],'lr': training_args.rotation_lr*0.1, "name": "relative_pose"}]
        l += l_cam_relative
        self.optimizer = PerPointAdam(l, lr=0, betas=(0.9, 0.999), eps=1e-15, weight_decay=0.0)

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

        self.cam_scheduler_args = get_expon_lr_func(
                                                    lr_init=training_args.rotation_lr*0.1,
                                                    lr_final=training_args.rotation_lr*0.001,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "pose" or param_group["name"] == "pose_relative":
                lr = self.cam_scheduler_args(iteration)
                # print("pose learning rate", iteration, lr)
                param_group['lr'] = lr
            # if param_group["name"] == "relative_pose":
            #     lr = self.cam_scheduler_args(iteration)
            #     param_group['lr'] = lr
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
        # return lr


    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        # delta
        for i in range(self._delta_x.shape[1]):
            l.append('delta_x_{}'.format(i))
        for i in range(self._delta_c.shape[1]*self._delta_c.shape[2]):
            l.append('delta_c_{}'.format(i))
        #print("delta_c_field:",self._delta_c.shape[1]*self._delta_c.shape[2])
        for i in range(self._delta_s.shape[1]):
            l.append('delta_s_{}'.format(i))
        for i in range(self._delta_r.shape[1]):
            l.append('delta_r_{}'.format(i))
        for i in range(self._delta_o.shape[1]):
            l.append('delta_o_{}'.format(i))
        # for i in range(self._cam_scale.shape[0]):
        #     l.append('cam_scale_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        #delta
        delta_x = self._delta_x.detach().cpu().numpy()
        #delta_c = self._delta_c.detach().detach().flatten(start_dim=1).contiguous().cpu().numpy()
        delta_c = self._delta_c.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        delta_s = self._delta_s.detach().cpu().numpy()
        delta_r = self._delta_r.detach().cpu().numpy()
        delta_o = self._delta_o.detach().cpu().numpy()
        #cam_scale = self._cam_scale.detach().cpu().numpy()
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, delta_x, delta_c, delta_s, delta_r, delta_o), axis=1)
        #, delta_s, delta_r, delta_o#, delta_x, delta_c
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        delta_x_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("delta_x")]
        delta_x_names = sorted(delta_x_names, key = lambda x: int(x.split('_')[-1]))
        delta_xs = np.zeros((xyz.shape[0], len(delta_x_names)))
        for idx, attr_name in enumerate(delta_x_names):
            delta_xs[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        delta_c_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("delta_c")]
        delta_c_names = sorted(delta_c_names, key = lambda x: int(x.split('_')[-1]))
        assert len(delta_c_names)==3*(self.max_sh_degree + 1) ** 2
        delta_cs = np.zeros((xyz.shape[0], len(delta_c_names)))
        for idx, attr_name in enumerate(delta_c_names):
            delta_cs[:, idx] = np.asarray(plydata.elements[0][attr_name])
        delta_cs = delta_cs.reshape((delta_cs.shape[0], 3, (self.max_sh_degree + 1) ** 2))
        
        delta_s_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("delta_s")]
        delta_s_names = sorted(delta_s_names, key = lambda x: int(x.split('_')[-1]))
        delta_ss = np.zeros((xyz.shape[0], len(delta_s_names)))
        for idx, attr_name in enumerate(delta_s_names):
            delta_ss[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        delta_r_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("delta_r")]
        delta_r_names = sorted(delta_r_names, key = lambda x: int(x.split('_')[-1]))
        delta_rs = np.zeros((xyz.shape[0], len(delta_r_names)))
        for idx, attr_name in enumerate(delta_r_names):
            delta_rs[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        delta_o_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("delta_o")]
        delta_o_names = sorted(delta_o_names, key = lambda x: int(x.split('_')[-1]))
        delta_os = np.zeros((xyz.shape[0], len(delta_o_names)))
        for idx, attr_name in enumerate(delta_o_names):
            delta_os[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # cam_scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("cam_scale")]
        # cam_scale_names = sorted(cam_scale_names, key = lambda x: int(x.split('_')[-1]))
        # cam_scales = np.zeros((xyz.shape[0], len(cam_scale_names)))
        # for idx, attr_name in enumerate(cam_scale_names):
        #     cam_scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        #delta
        self._delta_x = nn.Parameter(torch.tensor(delta_xs, dtype=torch.float, device="cuda").requires_grad_(True))
        self._delta_c = nn.Parameter(torch.tensor(delta_cs, dtype=torch.float, device="cuda").requires_grad_(True))
        self._delta_s = nn.Parameter(torch.tensor(delta_ss, dtype=torch.float, device="cuda").requires_grad_(True))
        self._delta_r = nn.Parameter(torch.tensor(delta_rs, dtype=torch.float, device="cuda").requires_grad_(True))
        self._delta_o = nn.Parameter(torch.tensor(delta_os, dtype=torch.float, device="cuda").requires_grad_(True))
        #self._cam_scale = nn.Parameter(torch.tensor(cam_scales, dtype=torch.float, device="cuda").requires_grad_(True))
        
        self.active_sh_degree = self.max_sh_degree

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        #self._mlp = MLP(58 + 3 + 1, 3).to("cuda")

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        #breakpoint()
        for group in self.optimizer.param_groups:
            #breakpoint()
            if "name" in group and group["name"] == "pose":
                continue 
            elif "name" in group and group["name"] == "relative_pose":
                continue
            # elif "name" in group and group["name"] == "":
            #     continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        #delta
        self._delta_x = optimizable_tensors["delta_x"]
        self._delta_c = optimizable_tensors["delta_c"]
        self._delta_s = optimizable_tensors["delta_s"]
        self._delta_r = optimizable_tensors["delta_r"]
        self._delta_o = optimizable_tensors["delta_o"]
        #self._cam_scale = optimizable_tensors["cam_scale"]
        
        
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        #self.confidence = self.confidence[valid_points_mask]
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        #breakpoint()
        for group in self.optimizer.param_groups:
            if group["name"] == "pose":
                continue  # 跳過pose相關操作
            elif group["name"] == "relative_pose":
                continue
            elif group["name"] == 'bg_color':
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        delta_features = torch.cat((new_features_dc, new_features_rest), dim=1)
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "delta_x": torch.zeros_like(new_xyz, device="cuda"),
        "delta_c": torch.zeros_like(delta_features, device="cuda").transpose(1, 2),
        "delta_s": torch.zeros_like(new_scaling, device="cuda"), 
        "delta_r": torch.zeros_like(new_rotation, device="cuda"),
        "delta_o": torch.zeros_like(new_opacities, device="cuda")}
        #"cam_scale": torch.ones(1, device="cuda")
        

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self._delta_x = optimizable_tensors["delta_x"]
        self._delta_c = optimizable_tensors["delta_c"]
        self._delta_s = optimizable_tensors["delta_s"]
        self._delta_r = optimizable_tensors["delta_r"] 
        self._delta_o = optimizable_tensors["delta_o"]
        #self._cam_scale = optimizable_tensors["cam_scale"]
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        #self.confidence = torch.cat([self.confidence, torch.ones(new_opacities.shape, device="cuda")], 0)
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]

        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1