# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Main class for the implementation of the global alignment
# --------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
import cv2
from dust3r.cloud_opt.base_opt import BasePCOptimizer
from dust3r.utils.geometry import xy_grid, geotrf
from dust3r.utils.device import to_cpu, to_numpy
from dust3r.cloud_opt.commons import edge_str, signed_log1p
from pathlib import Path
from functools import lru_cache as cache
from dust3r.utils.geometry import inv, geotrf
import roma

@cache
def pixel_grid(H, W):
    return np.mgrid[:W, :H].T.astype(np.float32)
    
class PointCloudOptimizer(BasePCOptimizer):
    """ Optimize a global scene, given a list of pairwise observations.
    Graph node: images
    Graph edges: observations = (pred1, pred2)
    """

    def __init__(self, *args, optimize_pp=False, focal_break=20, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_im_poses = True  # by definition of this class
        self.imshape = self.imshapes[0]
        self.focal_break_wide = focal_break
        self.focal_break_uw = focal_break
        H = self.imshape[0]
        W = self.imshape[1]
        #breakpoint()
        focal_uw = nn.Parameter(torch.tensor(self.focal_break_uw * np.log(max(H, W)), dtype=torch.float))
        focal_wide = nn.Parameter(torch.tensor(self.focal_break_wide * np.log(max(H, W)), dtype=torch.float))

        # 存儲這些 Parameter 以便後續使用
        self.im_focal_uw = focal_uw
        self.im_focal_wide = focal_wide
        self.im_focals = [self.im_focal_uw if i in self.nodes_uw else self.im_focal_wide for i in range(self.n_imgs)]

        self.im_depthmaps = nn.ParameterList(torch.randn(H, W)/10-3 for H, W in self.imshapes)  # log(depth)   
        self.im_poses = [nn.Parameter(self.rand_pose(self.POSE_DIM)) if next(iter(self.view_mapping[i])).startswith('uw') 
            else torch.ones(self.POSE_DIM) for i in range(self.n_imgs)] # camera poses

        self.im_pp = nn.ParameterList(torch.zeros((2,)) for _ in range(self.n_imgs))  # camera intrinsics
        self.im_pp.requires_grad_(optimize_pp)

        
        self.im_areas = [h*w for h, w in self.imshapes]
        self.max_area = max(self.im_areas)

        self.im_depthmaps = ParameterStack(self.im_depthmaps, is_param=True, fill=self.max_area)
        self.im_pp = ParameterStack(self.im_pp, is_param=True)
        self.register_buffer('_pp', torch.tensor([(w/2, h/2) for h, w in self.imshapes]))
        self.register_buffer('_grid', ParameterStack(
            [xy_grid(W, H, device=self.device) for H, W in self.imshapes], fill=self.max_area))

        # pre-compute pixel weights
        self.register_buffer('_weight_i', ParameterStack(
            [self.conf_trf(self.conf_i[i_j]) for i_j in self.str_edges], fill=self.max_area))
        self.register_buffer('_weight_j', ParameterStack(
            [self.conf_trf(self.conf_j[i_j]) for i_j in self.str_edges], fill=self.max_area))

        # precompute aa
        self.register_buffer('_stacked_pred_i', ParameterStack(self.pred_i, self.str_edges, fill=self.max_area))
        self.register_buffer('_stacked_pred_j', ParameterStack(self.pred_j, self.str_edges, fill=self.max_area))
        self.register_buffer('_ei', torch.tensor([i for i, j in self.edges]))
        self.register_buffer('_ej', torch.tensor([j for i, j in self.edges]))
        self.total_area_i = sum([self.im_areas[i] for i, j in self.edges])
        self.total_area_j = sum([self.im_areas[j] for i, j in self.edges])

        #relative_pose weight
        self.rel_pose_weight = 0.1

    def _check_all_imgs_are_selected(self, msk):
        assert np.all(self._get_msk_indices(msk) == np.arange(self.n_imgs)), 'incomplete mask!'

    # def preset_depth_scale(self, known_focal_uw, known_focal_wide):
    #     param = self.depth_scale
    #     new_depth_scale = np.log(known_focal_wide / known_focal_uw)
    #     param.data[:] = new_depth_scale
        
        
    def preset_pose(self, known_poses, pose_msk=None):  # cam-to-world
        self._check_all_imgs_are_selected(pose_msk)

        if isinstance(known_poses, torch.Tensor) and known_poses.ndim == 2:
            known_poses = [known_poses]
        for idx, pose in zip(self._get_msk_indices(pose_msk), known_poses):
            if self.verbose:
                print(f' (setting pose #{idx} = {pose[:3,3]})')
            self._no_grad(self._set_pose(self.im_poses, idx, torch.tensor(pose)))

        # normalize scale if there's less than 1 known pose
        n_known_poses = sum((p.requires_grad is False) for p in self.im_poses)
        self.norm_pw_scale = (n_known_poses <= 1)

        self.im_poses.requires_grad_(False)
        self.norm_pw_scale = False


    def preset_focal_wide(self, known_focal_wide, msk=None):
        self._check_all_imgs_are_selected(msk)
        self.im_focal_wide.requires_grad_(True)
        for idx in range(self.n_imgs):
            if idx in self.nodes_wide:
                print(f' (setting focal #{idx} = {known_focal_wide})')
                self._no_grad(self._set_focal(idx, known_focal_wide))

        self.im_focal_wide.requires_grad_(False)

    def preset_focal_uw(self, known_focal_uw, msk=None):
        self._check_all_imgs_are_selected(msk)

        self.im_focal_uw.requires_grad_(True)
        for idx in range(self.n_imgs):
            if idx in self.nodes_uw:
                print(f' (setting focal_uw #{idx} = {known_focal_uw})')
                self._no_grad(self._set_focal(idx, known_focal_uw))
        self.im_focal_uw.requires_grad_(False)
    def preset_principal_point(self, known_pp, msk=None):
        self._check_all_imgs_are_selected(msk)

        for idx, pp in zip(self._get_msk_indices(msk), known_pp):
            if self.verbose:
                print(f' (setting principal point #{idx} = {pp})')
            self._no_grad(self._set_principal_point(idx, pp))

        self.im_pp.requires_grad_(False)

    def _get_msk_indices(self, msk):
        if msk is None:
            return range(self.n_imgs)
        elif isinstance(msk, int):
            return [msk]
        elif isinstance(msk, (tuple, list)):
            return self._get_msk_indices(np.array(msk))
        elif msk.dtype in (bool, torch.bool, np.bool_):
            assert len(msk) == self.n_imgs
            return np.where(msk)[0]
        elif np.issubdtype(msk.dtype, np.integer):
            return msk
        else:
            raise ValueError(f'bad {msk=}')

    def _no_grad(self, tensor):
        assert tensor.requires_grad, 'it must be True at this point, otherwise no modification occurs'

    def _set_focal(self, idx, focal, force=False):
        param = self.im_focals[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            #breakpoint()
            for i in range(self.n_imgs):
                if i in self.nodes_uw:
                    param.data = torch.tensor(self.focal_break_uw * np.log(focal))
                elif i in self.nodes_wide:
                    param.data = torch.tensor(self.focal_break_wide * np.log(focal))
        return param

    def get_focals(self):
        log_focals = torch.stack(list(self.im_focals), dim=0).cuda()
        new_focals = log_focals.clone()
        
        for i in range(len(log_focals)):
            if i in self.nodes_uw:
                new_focals[i] = log_focals[i] / self.focal_break_uw
            elif i in self.nodes_wide:
                new_focals[i] = log_focals[i] / self.focal_break_wide
        #breakpoint()
        return new_focals.exp()
    
    def get_known_focal_mask_wide(self):
        return torch.tensor([not (p.requires_grad) for p in self.im_focals_wide])

    def get_known_focal_mask_uw(self):
        return torch.tensor([not (p.requires_grad) for p in self.im_focals_uw])

    def _set_principal_point(self, idx, pp, force=False):
        param = self.im_pp[idx]
        H, W = self.imshapes[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = to_cpu(to_numpy(pp) - (W/2, H/2)) / 10
        return param

    def get_principal_points(self):
        return self._pp + 10 * self.im_pp

    def get_intrinsics(self):
        K = torch.zeros((self.n_imgs, 3, 3), device=self.device)
        focals = self.get_focals().flatten()
        K[:, 0, 0] = K[:, 1, 1] = focals
        K[:, :2, 2] = self.get_principal_points()
        K[:, 2, 2] = 1
        return K

    
    def verify_rel_poses_C2W(self,poses):
        rel_poses = []
        for (uw_idx, wide_idx), _ in self.UW_W_pair.items():
            wide_pose_C2W = poses[wide_idx]
            uw_pose_C2W = poses[uw_idx]
            rel_pose = inv(wide_pose_C2W) @ uw_pose_C2W
            rel_poses.append(rel_pose)
        
        assert len(rel_poses) == self.n_imgs//2
        for i in range(len(rel_poses) - 1):
            if i+1 < len(rel_poses):
                #breakpoint()
                assert torch.allclose(rel_poses[i], rel_poses[i+1], rtol=1e-4, atol=1e-5)
        
        return rel_poses
    
    def verify_rel_poses_W2C(self,poses):
        rel_poses = []
        for (uw_idx, wide_idx), _ in self.UW_W_pair.items():
            wide_pose_W2C = poses[wide_idx]
            uw_pose_W2C = poses[uw_idx]
            rel_pose = wide_pose_W2C @ inv(uw_pose_W2C)
            rel_poses.append(rel_pose)
        
        assert len(rel_poses) == self.n_imgs//2
        for i in range(len(rel_poses) - 1):
            if i+1 < len(rel_poses):
                #breakpoint()
                assert torch.allclose(rel_poses[i], rel_poses[i+1], rtol=1e-4, atol=1e-5)
        
        return rel_poses
        
    def get_im_poses(self):
        all_poses = [torch.ones(4,4, dtype=torch.float32) for i in range(self.n_imgs)]
        # 對於wide相機，基於UW姿態和相對姿態計算姿態
        for i in range(self.n_imgs):
            if i in self.nodes_wide:
                for (uw_idx, wide_idx), _ in self.UW_W_pair.items():
                    if i == wide_idx:
                        # 從UW姿態和相對姿態計算wide姿態
                        wide_pose = self.get_corresponding_wide_pose(uw_idx).to(dtype=torch.float32)
                        all_poses[i] = wide_pose.cuda()
                        
            elif i in self.nodes_uw:
                QT_uw = self.im_poses[i]
                uw_pose = self._get_one_pose(QT_uw)
                all_poses[i] = uw_pose.cuda()

        poses_tensor = torch.stack([param for param in all_poses])
        rel_poses_check = self.verify_rel_poses_C2W(poses_tensor)
        return poses_tensor
    
    
    def _set_depthmap(self, idx, depth, force=False):
        depth = _ravel_hw(depth, self.max_area)

        param = self.im_depthmaps[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = depth.log().nan_to_num(neginf=0)
        return param

    def get_depthmaps(self, raw=False):
        res = self.im_depthmaps.exp()
        if not raw:
            res = [dm[:h*w].view(h, w) for dm, (h, w) in zip(res, self.imshapes)]
        return res

    def depth_to_pts3d(self, iteration, output_path):
        # Get depths and  projection params if not provided
        focals = self.get_focals()
        pp = self.get_principal_points()
        im_poses = self.get_im_poses()
        depth = self.get_depthmaps(raw=True)
        depth_num = depth.shape[0]
        rel_ptmaps = _fast_depthmap_to_pts3d(depth, self._grid, focals, pp=pp)
        return geotrf(im_poses, rel_ptmaps)
    
    def get_pts3d(self, iteration, output_path, raw=False):
        res = self.depth_to_pts3d(iteration, output_path)
        if not raw:
            res = [dm[:h*w].view(h, w, 3) for dm, (h, w) in zip(res, self.imshapes)]
        return res
    
    
    def cal_relative_pose_loss(self, uw_idx, w_idx, niter_PnP=10):
        "R*, T* = argmin Ci^1,1 Ci^1,2 ||scale * (R * Xi^1,1 + t) - Xi^1,2 ||"
        "我們要找一個relative pose R, T 使得"
        #UW_W_pair:  {(1, 0): {'w_001', 'uw_001'}, (2, 3): {'uw_008', 'w_008'}, (5, 4): {'w_015', 'uw_015'}}
        RT = self.get_pw_poses_rel()  # 改用global scale
        #scale = self.rel_scale.clone() #不確定
        
        i_j = edge_str(uw_idx, w_idx)
         
        # 乘起來
        conf = self.conf_i[i_j] * self.conf_j[i_j]  
        valid_mask = conf > self.min_conf_thr
        
        #自己的點雲
        X1_1i = self.pred_i[i_j]
        
        # ||σ(RX1,1i + t) - X1,2i||²
        X1_2i = self.pred_j[i_j] 
        
        
        
        #用RANSAC+PnP拿inliers
        H, W, _ = X1_2i.shape
        pixels = pixel_grid(H, W)
        pixels = torch.from_numpy(pixels).cuda()
        pp = self.im_pp[uw_idx].clone().detach().cpu().numpy()
        # #這邊按照公式focal是去找uw
        focal = float(self.im_focal_uw.clone())
        # # focal = focal.cpu().numpy()
        K = np.float32([
            [focal, 0, pp[0]], 
            [0, focal, pp[1]], 
            [0, 0, 1]
        ])
        # #看一下有沒有這是trainable
        
        try:
            
            X1_2i_npy = X1_2i[valid_mask].cpu().numpy()
            pixels_npy = pixels[valid_mask].cpu().numpy()
            
            success, _, _, X1_2i_inliers = cv2.solvePnPRansac(X1_2i_npy, pixels_npy, K, None,
                                                        iterationsCount=niter_PnP, reprojectionError=5, flags=cv2.SOLVEPNP_SQPNP)
        
            if success and X1_2i_inliers is not None:
                # 處理mask
                inlier_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
                valid_indices = valid_mask.nonzero().squeeze()
                inlier_mask[valid_indices[X1_2i_inliers]] = True
                
                final_mask = valid_mask & inlier_mask
                # 算loss
            else:
                final_mask = valid_mask
        
        except Exception as e:
            print(f"PnP RANSAC failed: {e}")
            final_mask = valid_mask
        
        X1_1i = self.pred_i[i_j]
        X1_2i = self.pred_j[i_j]
        aligned_pred = geotrf(RT, X1_1i)#

        loss = self.dist(X1_2i, aligned_pred, weight=conf*final_mask).sum()/self.max_area
        
        return loss

    def forward(self, iteration, output_path):

        #這邊在forward
        pw_poses = self.get_pw_poses()  # cam-to-world
        pw_adapt = self.get_adaptors().unsqueeze(1)
        proj_pts3d = self.get_pts3d(iteration, output_path, raw=True)

        #rotate pairwise prediction according to pw_poses=>把他們轉到common coordinate
        aligned_pred_i = geotrf(pw_poses, pw_adapt * self._stacked_pred_i)
        aligned_pred_j = geotrf(pw_poses, pw_adapt * self._stacked_pred_j)

        # compute the loss
        li = self.dist(proj_pts3d[self._ei], aligned_pred_i, weight=self._weight_i).sum() / self.total_area_i
        lj = self.dist(proj_pts3d[self._ej], aligned_pred_j, weight=self._weight_j).sum() / self.total_area_j
        
        loss_rel_pose = 0.0
        cnt = 0
        for (uw_idx, w_idx), _ in self.UW_W_pair.items():
            loss_rel_pose += self.cal_relative_pose_loss(uw_idx, w_idx)
            cnt += 1
        
        if cnt > 0:
            loss_rel_pose = loss_rel_pose / cnt
            return li + lj + self.rel_pose_weight * loss_rel_pose
        else:
            return li + lj
    

def _fast_depthmap_to_pts3d(depth, pixel_grid, focal, pp):
    pp = pp.unsqueeze(1)
    focal = focal.unsqueeze(1).unsqueeze(1)
    #breakpoint()
    assert focal.shape == (len(depth), 1, 1)
    assert pp.shape == (len(depth), 1, 2)
    assert pixel_grid.shape == depth.shape + (2,)
    depth = depth.unsqueeze(-1)
    #breakpoint()
    return torch.cat((depth * (pixel_grid - pp) / focal, depth), dim=-1)


def ParameterStack(params, keys=None, is_param=None, fill=0):
    if keys is not None:
        params = [params[k] for k in keys]

    if fill > 0:
        params = [_ravel_hw(p, fill) for p in params]

    requires_grad = params[0].requires_grad
    assert all(p.requires_grad == requires_grad for p in params)

    params = torch.stack(list(params)).float().detach()
    if is_param or requires_grad:
        params = nn.Parameter(params)
        params.requires_grad_(requires_grad)
    return params


def _ravel_hw(tensor, fill=0):
    # ravel H,W
    tensor = tensor.view((tensor.shape[0] * tensor.shape[1],) + tensor.shape[2:])

    if len(tensor) < fill:
        tensor = torch.cat((tensor, tensor.new_zeros((fill - len(tensor),)+tensor.shape[1:])))
    return tensor


def acceptable_focal_range(H, W, minf=0.5, maxf=3.5):
    focal_base = max(H, W) / (2 * np.tan(np.deg2rad(60) / 2))  # size / 1.1547005383792515
    return minf*focal_base, maxf*focal_base


def apply_mask(img, msk):
    img = img.copy()
    img[msk] = 0
    return img