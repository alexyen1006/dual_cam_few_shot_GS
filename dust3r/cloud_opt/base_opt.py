# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Base class for the global alignement procedure
# --------------------------------------------------------
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import roma
from copy import deepcopy
import tqdm
import cv2
from functools import lru_cache as cache
from dust3r.utils.geometry import inv, geotrf
from dust3r.utils.device import to_numpy
from dust3r.utils.image import rgb
from dust3r.viz import SceneViz, segment_sky, auto_cam_size
from dust3r.optim_factory import adjust_learning_rate_by_lr

from dust3r.cloud_opt.commons import (edge_str, ALL_DISTS, NoGradParamDict, get_imshapes, signed_expm1, signed_log1p,
                                      cosine_schedule, linear_schedule, get_conf_trf, get_nodes_from_edges)
import dust3r.cloud_opt.init_im_poses as init_fun

import os

class BasePCOptimizer (nn.Module):
    """ Optimize a global scene, given a list of pairwise observations.
    Graph node: images
    Graph edges: observations = (pred1, pred2)
    """

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            other = deepcopy(args[0])
            attrs = '''edges is_symmetrized dist n_imgs pred_i pred_j imshapes 
                        min_conf_thr conf_thr conf_i conf_j im_conf
                        base_scale norm_pw_scale POSE_DIM pw_poses 
                        pw_adaptors pw_adaptors has_im_poses rand_pose imgs verbose'''.split()
            self.__dict__.update({k: other[k] for k in attrs})
        else:
            self._init_from_views(*args, **kwargs)

    def _init_from_views(self, view1, view2, pred1, pred2, view_mapping,
                         dist='l1',
                         conf='log',
                         min_conf_thr=3,
                         base_scale=0.5,
                         allow_pw_adaptors=True,
                         pw_break=20,
                         rand_pose=torch.randn,
                         iterationsCount=None,
                         verbose=True):
        super().__init__()
        if not isinstance(view1['idx'], list):
            view1['idx'] = view1['idx'].tolist()
        if not isinstance(view2['idx'], list):
            view2['idx'] = view2['idx'].tolist()
        self.uw_indices = []
        self.w_indices = []
        # view_mapping 是字典，所以需要用 .items()
        for index, image_set in view_mapping.items():
            # image_set 是一個集合，需要取出第一個（也是唯一的）元素
            image_name = next(iter(image_set))
            if image_name.startswith('w') and not image_name.startswith('uw'):
                self.w_indices.append(index)
            elif image_name.startswith('uw') and not image_name.startswith('w'):
                self.uw_indices.append(index)

        self.edges = [(int(i), int(j)) for i, j in zip(view1['idx'], view2['idx'])]
        self.edges_wide = [(int(i), int(j)) for i, j in self.edges 
                        if i in self.w_indices and j in self.w_indices]
        self.edges_uw = [(int(i), int(j)) for i, j in self.edges 
                        if i in self.uw_indices and j in self.uw_indices]
        self.nodes_uw = get_nodes_from_edges(self.edges_uw)
        self.nodes_wide = get_nodes_from_edges(self.edges_wide)
        self.is_symmetrized = set(self.edges) == {(j, i) for i, j in self.edges}
        self.dist = ALL_DISTS[dist]
        self.verbose = verbose

        self.n_imgs = self._check_edges()

        # input data
        pred1_pts = pred1['pts3d']#是在相機 i 坐標系中的點雲
        pred2_pts = pred2['pts3d_in_other_view']#是同樣的點，但在相機 j 坐標系中表示
        self.pred_i = NoGradParamDict({ij: pred1_pts[n] for n, ij in enumerate(self.str_edges)})
        self.pred_j = NoGradParamDict({ij: pred2_pts[n] for n, ij in enumerate(self.str_edges)})
        self.imshapes = get_imshapes(self.edges, pred1_pts, pred2_pts)

        # work in log-scale with conf
        pred1_conf = pred1['conf']
        pred2_conf = pred2['conf']
        self.min_conf_thr = min_conf_thr
        self.conf_trf = get_conf_trf(conf)

        self.conf_i = NoGradParamDict({ij: pred1_conf[n] for n, ij in enumerate(self.str_edges)})
        self.conf_j = NoGradParamDict({ij: pred2_conf[n] for n, ij in enumerate(self.str_edges)})
        self.im_conf = self._compute_img_conf(pred1_conf, pred2_conf)
        for i in range(len(self.im_conf)):
            self.im_conf[i].requires_grad = False

        # pairwise pose parameters
        self.base_scale = base_scale
        self.norm_pw_scale = True
        self.pw_break = pw_break
        self.POSE_DIM = 7
        self.pw_poses = nn.Parameter(rand_pose((self.n_edges, 1+self.POSE_DIM)))  # pairwise poses
        self.pw_adaptors = nn.Parameter(torch.zeros((self.n_edges, 2)))  # slight xy/z adaptation
        self.pw_adaptors.requires_grad_(allow_pw_adaptors)

        self.has_im_poses = False
        self.rand_pose = rand_pose

        # possibly store images for show_pointcloud
        self.imgs = None
        if 'img' in view1 and 'img' in view2:
            imgs = [torch.zeros((3,)+hw) for hw in self.imshapes]
            for v in range(len(self.edges)):
                idx = view1['idx'][v]
                imgs[idx] = view1['img'][v]
                idx = view2['idx'][v]
                imgs[idx] = view2['img'][v]
            self.imgs = rgb(imgs)

        
        self.view_mapping = view_mapping
        self.UW_W_pair = self.pairwise_view_mapping(view_mapping)
        self.estimate_relative_pose_or_not = True
        # relative pose parameter:
        self.relative_pose = nn.Parameter(self.rand_pose(self.POSE_DIM)) #relative_pose 其實是cam_uw to cam_wide 
        
    @property
    def n_edges(self):
        return len(self.edges)
    @property
    def str_edges(self):
        return [edge_str(i, j) for i, j in self.edges]

    @property
    def imsizes(self):
        return [(w, h) for h, w in self.imshapes]

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def custom_sort(self,item):
        value = list(item[1])[0]  # 从集合中获取值
        prefix = value[:2]  # 获取前缀 (uw 或 w)
        num = int(value.split('_')[1])  # 获取数字部分
        
        # 优先按前缀排序（uw 在前），其次按数字排序
        return (0 if prefix == "uw" else 1, num)
    
    def pairwise_view_mapping(self, view_mapping):
        # 排序並取得(index, set)對
        sorted_view_mapping = sorted(view_mapping.items(), key=self.custom_sort)
        UW_W_view_mapping = {}
        
        for index, image_set in sorted_view_mapping:
            # 從集合取出圖片名稱
            image_name = next(iter(image_set))
            if image_name.startswith('uw'):
                # 取出id號碼
                
                img_id = image_name.split('_')[1]
                wide_img_name = 'w_' + img_id
                
                # 找出對應的wide圖片的index
                wide_index = next(idx for idx, img_set in sorted_view_mapping 
                                if next(iter(img_set)) == wide_img_name)
                
                # 儲存配對: {(uw_index, w_index): {uw圖片, w圖片}}
                UW_W_view_mapping[(index, wide_index)] = (image_name, wide_img_name)
                #breakpoint()
        return UW_W_view_mapping     

    
    def state_dict(self, trainable=True):
        all_params = super().state_dict()
        return {k: v for k, v in all_params.items() if k.startswith(('_', 'pred_i.', 'pred_j.', 'conf_i.', 'conf_j.')) != trainable}

    def load_state_dict(self, data):
        return super().load_state_dict(self.state_dict(trainable=False) | data)

    def _check_edges(self):
        indices = sorted({i for edge in self.edges for i in edge})
        assert indices == list(range(len(indices))), 'bad pair indices: missing values '
        return len(indices)

    @torch.no_grad()
    def _compute_img_conf(self, pred1_conf, pred2_conf):
        im_conf = nn.ParameterList([torch.zeros(hw, device=self.device) for hw in self.imshapes])
        for e, (i, j) in enumerate(self.edges):
            im_conf[i] = torch.maximum(im_conf[i], pred1_conf[e])
            im_conf[j] = torch.maximum(im_conf[j], pred2_conf[e])
        return im_conf

    def get_adaptors(self):
        adapt = self.pw_adaptors
        adapt = torch.cat((adapt[:, 0:1], adapt), dim=-1)  # (scale_xy, scale_xy, scale_z)
        if self.norm_pw_scale:  # normalize so that the product == 1
            adapt = adapt - adapt.mean(dim=1, keepdim=True)
        return (adapt / self.pw_break).exp()

    def _get_wide_pose(self):
        for idx,name in self.view_mapping:
            pose_uw = self.im_poses[idx]
    
    def _get_poses(self, poses):
        # normalize rotation
        Q = poses[:, :4]
        T = signed_expm1(poses[:, 4:7])
        RT = roma.RigidUnitQuat(Q, T).normalize().to_homogeneous()
        return RT
    
    def _get_one_pose(self, pose):
        Q = pose[:4]
        T = signed_expm1(pose[4:7])  
        RT = roma.RigidUnitQuat(Q, T).normalize().to_homogeneous()
        return RT

    def _set_pose(self, poses, idx, R, T=None, scale=None, force=False):
        # all poses == cam-to-world
        pose = poses[idx]
        if idx in self.nodes_uw:
            if not (pose.requires_grad or force):
                return pose

        if R.shape == (4, 4):
            assert T is None
            T = R[:3, 3]
            R = R[:3, :3]

        if R is not None:
            pose.data[0:4] = roma.rotmat_to_unitquat(R)
        if T is not None:
            pose.data[4:7] = signed_log1p(T / (scale or 1))  # translation is function of scale

        if scale is not None:
            assert poses.shape[-1] in (8, 13)
            pose.data[-1] = np.log(float(scale))
        return pose

    def _set_one_pose(self, R, T, scale=None):
        pose = torch.zeros(7)
        if R is not None:
            pose[0:4] = roma.rotmat_to_unitquat(R)
        if T is not None:
            pose[4:7] = signed_log1p(T / (scale or 1))  # translation is function of scale

        return pose
    
    def compute_one_wide_pose_C2W(self, E_uw):
        #cam_uw to cam_wide
        QT_rel = self.relative_pose.clone()
        E_uw_to_wide = self._get_one_pose(QT_rel)
    
        E_wide = E_uw @ inv(E_uw_to_wide)

        R_wide = E_wide[:3, :3]
        T_wide_init = E_wide[:3, 3]

        Q_wide = roma.rotmat_to_unitquat(R_wide)
        T_wide = signed_log1p(T_wide_init)
        
        QT_wide = torch.cat([Q_wide, T_wide], dim=0)
        
        return QT_wide
    def compute_pairwise_relative_pose(self, uw_idx, wide_idx):
        # 獲取相機到世界(C2W)的變換矩陣
        # 相對相機位姿已經是UW_to_wide
        E_uw = self._get_one_pose(self.im_poses[uw_idx])
        E_wide = self._get_one_pose(self.im_poses[wide_idx])
        
        # 計算從 UW 相機到 Wide 相機的相對變換
        # Wide 到世界的逆變換（即世界到 Wide）
        E_wide_inv = torch.inverse(E_wide)
        
        # 相對變換 = (世界到 Wide) @ (UW 到世界)
        E_rel_uw_to_wide = torch.matmul(E_wide_inv, E_uw)
        
        return E_rel_uw_to_wide
    
    def get_corresponding_wide_pose(self, uw_idx):
        """
            input : index (QT)
            output : 對應wide pose的4X4
        """
        E_uw = self._get_one_pose(self.im_poses[uw_idx])
        E_uw_tensor = E_uw.to(device='cuda', dtype=torch.float32)
        QT_wide_computed = self.compute_one_wide_pose_C2W(E_uw_tensor)
        E_wide_computed = self._get_one_pose(QT_wide_computed)
        return E_wide_computed

    def get_pw_norm_scale_factor(self):
        if self.norm_pw_scale:
            # normalize scales so that things cannot go south
            # we want that exp(scale) ~= self.base_scale
            return (np.log(self.base_scale) - self.pw_poses[:, -1].mean()).exp()
        else:
            return 1  # don't norm scale for known poses
    
    def get_pw_norm_scale_factor_rel(self):
        if self.norm_pw_scale:
            # normalize scales so that things cannot go south
            # we want that exp(scale) ~= self.base_scale_rel
            return (np.log(self.base_scale) - self.relative_pose[-1]).exp()
        else:
            return 1  # don't norm scale for known poses
    
    def get_pw_scale(self):
        scale = self.pw_poses[:, -1].exp()  # (n_edges,)
        scale = scale * self.get_pw_norm_scale_factor()
        return scale

    
    def get_pw_scale_rel(self):
        scale = self.relative_pose[-1].exp()  # (n_edges,)
        scale = scale * self.get_pw_norm_scale_factor_rel()
        return scale
         

    def get_pw_poses(self):  # cam to world
        RT = self._get_poses(self.pw_poses)
        scaled_RT = RT.clone()
        scaled_RT[:, :3] *= self.get_pw_scale().view(-1, 1, 1)  # scale the rotation AND translation
        return scaled_RT

    
    def get_pw_poses_rel(self):
        RT = self._get_one_pose(self.relative_pose)
        scaled_RT = RT.clone()
        #breakpoint()
        scaled_RT[:, :3] *= self.get_pw_scale_rel().view(-1, 1)  # scale the rotation AND translation
        return RT

    def get_masks(self):
        return [(conf > self.min_conf_thr) for conf in self.im_conf]

    def depth_to_pts3d(self):
        raise NotImplementedError()

    def get_pts3d(self, raw=False):
        res = self.depth_to_pts3d()
        if not raw:
            res = [dm[:h*w].view(h, w, 3) for dm, (h, w) in zip(res, self.imshapes)]
        return res

    def _set_focal(self, idx, focal, force=False):
        raise NotImplementedError()

    def get_focals(self):
        raise NotImplementedError()

    def get_known_focal_mask(self):
        raise NotImplementedError()

    def get_principal_points(self):
        raise NotImplementedError()

    def get_conf(self, mode=None):
        trf = self.conf_trf if mode is None else get_conf_trf(mode)
        return [trf(c) for c in self.im_conf]

    def get_im_poses(self):
        raise NotImplementedError()

    def _set_depthmap(self, idx, depth, force=False):
        raise NotImplementedError()

    def get_depthmaps(self, raw=False):
        raise NotImplementedError()

    def clean_pointcloud(self, **kw):
        cams = inv(self.get_im_poses())
        K = self.get_intrinsics()
        depthmaps = self.get_depthmaps()
        all_pts3d = self.get_pts3d()

        new_im_confs = clean_pointcloud(self.im_conf, K, cams, depthmaps, all_pts3d, **kw)

        for i, new_conf in enumerate(new_im_confs):
            self.im_conf[i].data[:] = new_conf
        return self
    
    def forward(self, ret_details=False):
        #不是在這邊forward 是在optimizer 那邊forward
        pw_poses = self.get_pw_poses()  # cam-to-world
        pw_adapt = self.get_adaptors()
        proj_pts3d = self.get_pts3d()
        # pre-compute pixel weights
        weight_i = {i_j: self.conf_trf(c) for i_j, c in self.conf_i.items()}
        weight_j = {i_j: self.conf_trf(c) for i_j, c in self.conf_j.items()}

        loss = 0
        if ret_details:
            details = -torch.ones((self.n_imgs, self.n_imgs))

        for e, (i, j) in enumerate(self.edges):
            i_j = edge_str(i, j)
            # distance in image i and j
            aligned_pred_i = geotrf(pw_poses[e], pw_adapt[e] * self.pred_i[i_j])
            aligned_pred_j = geotrf(pw_poses[e], pw_adapt[e] * self.pred_j[i_j])
            li = self.dist(proj_pts3d[i], aligned_pred_i, weight=weight_i[i_j]).mean()
            lj = self.dist(proj_pts3d[j], aligned_pred_j, weight=weight_j[i_j]).mean()
            loss = loss + li + lj

            if ret_details:
                details[i, j] = li + lj
        loss /= self.n_edges  # average over all pairs

        if ret_details:
            return loss, details
        return loss

            
            
    @torch.cuda.amp.autocast(enabled=False)
    def compute_global_alignment(self, init=None, niter_PnP=10, focal_avg=False, known_focal_uw=None, known_focal_wide=None, view_mapping=None, output_path= None, **kw):
        if init is None:
            pass
        elif init == 'msp' or init == 'mst':
            #breakpoint()
            init_fun.init_minimum_spanning_tree(self, niter_PnP=niter_PnP, focal_avg=focal_avg, known_focal_uw=known_focal_uw, known_focal_wide=known_focal_wide, view_mapping=view_mapping, output_path = output_path)
        elif init == 'known_poses':
            init_fun.init_from_known_poses(self, min_conf_thr=self.min_conf_thr,
                                           niter_PnP=niter_PnP)
        else:
            raise ValueError(f'bad value for {init=}')
        #breakpoint()
        return global_alignment_loop(self, **kw, output_path = output_path)
    

    @torch.no_grad()
    def mask_sky(self):
        res = deepcopy(self)
        for i in range(self.n_imgs):
            sky = segment_sky(self.imgs[i])
            res.im_conf[i][sky] = 0
        return res

    def show(self, show_pw_cams=False, show_pw_pts3d=False, cam_size=None, **kw):
        viz = SceneViz()
        if self.imgs is None:
            colors = np.random.randint(0, 256, size=(self.n_imgs, 3))
            colors = list(map(tuple, colors.tolist()))
            for n in range(self.n_imgs):
                viz.add_pointcloud(self.get_pts3d()[n], colors[n], self.get_masks()[n])
        else:
            viz.add_pointcloud(self.get_pts3d(), self.imgs, self.get_masks())
            colors = np.random.randint(256, size=(self.n_imgs, 3))

        # camera poses
        im_poses = to_numpy(self.get_im_poses())
        if cam_size is None:
            cam_size = auto_cam_size(im_poses)
        viz.add_cameras(im_poses, self.get_focals(), colors=colors,
                        images=self.imgs, imsizes=self.imsizes, cam_size=cam_size)
        if show_pw_cams:
            pw_poses = self.get_pw_poses()
            viz.add_cameras(pw_poses, color=(192, 0, 192), cam_size=cam_size)

            if show_pw_pts3d:
                pts = [geotrf(pw_poses[e], self.pred_i[edge_str(i, j)]) for e, (i, j) in enumerate(self.edges)]
                viz.add_pointcloud(pts, (128, 0, 128))

        viz.show(**kw)
        return viz


def global_alignment_loop(net, lr=0.01, niter=300, schedule='cosine', lr_min=1e-6, output_path = None):
    params = [p for p in net.parameters() if p.requires_grad]

    if not params:
        return net

    verbose = net.verbose
    if verbose:
        print('Global alignement - optimizing for:')
        print("Before GA :")
        print([name for name, value in net.named_parameters() if value.requires_grad])
        #breakpoint()
    lr_base = lr
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.9))

    loss = float('inf')
    if verbose:
        with tqdm.tqdm(total=niter) as bar:
            while bar.n < bar.total:
                
                loss, lr = global_alignment_iter(net, bar.n, niter, lr_base, lr_min, optimizer, schedule, output_path)
                bar.set_postfix_str(f'{lr=:g} loss={loss:g}')
                bar.update()
    else:
        for n in range(niter):
            loss, _ = global_alignment_iter(net, n, niter, lr_base, lr_min, optimizer, schedule)
    return loss


def global_alignment_iter(net, cur_iter, niter, lr_base, lr_min, optimizer, schedule, output_path):
    t = cur_iter / niter
    if schedule == 'cosine':
        lr = cosine_schedule(t, lr_base, lr_min)
    elif schedule == 'linear':
        lr = linear_schedule(t, lr_base, lr_min)
    else:
        raise ValueError(f'bad lr {schedule=}')
    adjust_learning_rate_by_lr(optimizer, lr)
    optimizer.zero_grad()

    loss = net(cur_iter, output_path)

    loss.backward()

    grad_file_path = os.path.join(output_path, 'gradient_info.txt')

    save_gradients(cur_iter, net, log_file_path=grad_file_path)
    optimizer.step()

    return float(loss), lr

def save_gradients(iteration, net, log_file_path=None):
    # 如果沒有指定log文件路徑，創建一個包含時間戳的文件名
    if log_file_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = f"gradient_log_{timestamp}.txt"
    
    # 每50次迭代檢查梯度
    if iteration % 50 == 0:
        # 初始化 log_text
        log_text = f"\nIteration {iteration} Gradient Check:\n"
        
        # 獲取相對位姿的梯度
        relative_pose_grad = net.relative_pose.grad if net.relative_pose.grad is not None else "No gradient"
        log_text += f"relative_pose gradient: {relative_pose_grad}\n"
        
        # 檢查每個相機位姿的梯度
        for i in range(len(net.im_poses)):
            log_text += f"camera {i} requires_grad: {net.im_poses[i].requires_grad}"
            log_text += f"camera {i} grad: {net.im_poses[i].grad}"
        
        log_text += "-" * 50 + "\n"  # 分隔線
        
        # 以追加模式寫入文件
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(log_text)

@torch.no_grad()
def clean_pointcloud(im_confs, K, cams, depthmaps, all_pts3d, 
                      tol=0.001, bad_conf=0, dbg=()):
    """ Method: 
    1) express all 3d points in each camera coordinate frame
    2) if they're in front of a depthmap --> then lower their confidence
    """
    assert len(im_confs) == len(cams) == len(K) == len(depthmaps) == len(all_pts3d)
    assert 0 <= tol < 1
    res = [c.clone() for c in im_confs]

    # reshape appropriately
    all_pts3d = [p.view(*c.shape,3) for p,c in zip(all_pts3d, im_confs)]
    depthmaps = [d.view(*c.shape) for d,c in zip(depthmaps, im_confs)]
    
    for i, pts3d in enumerate(all_pts3d):
        for j in range(len(all_pts3d)):
            if i == j: continue

            # project 3dpts in other view
            proj = geotrf(cams[j], pts3d)
            proj_depth = proj[:,:,2]
            u,v = geotrf(K[j], proj, norm=1, ncol=2).round().long().unbind(-1)

            # check which points are actually in the visible cone
            H, W = im_confs[j].shape
            msk_i = (proj_depth > 0) & (0 <= u) & (u < W) & (0 <= v) & (v < H)
            msk_j = v[msk_i], u[msk_i]

            # find bad points = those in front but less confident
            bad_points = (proj_depth[msk_i] < (1-tol) * depthmaps[j][msk_j]) & (res[i][msk_i] < res[j][msk_j])

            bad_msk_i = msk_i.clone()
            bad_msk_i[msk_i] = bad_points
            res[i][bad_msk_i] = res[i][bad_msk_i].clip_(max=bad_conf)

    return res