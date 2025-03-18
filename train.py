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
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torchvision
import sys
from random import randint
from utils.loss_utils import l1_loss, l1_loss_mask, l2_loss, ssim, SmoothLoss, l1_loss＿binocular_loss, SmoothLossWeightScheduler
from gaussian_renderer import render, network_gui
from time import time

from datetime import datetime

from tqdm import tqdm
from utils.depth_utils import estimate_depth
from utils.image_utils import psnr
from utils.graphics_utils import inverse_warp_images
from argparse import ArgumentParser, Namespace

from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene, GaussianModel
from scene.cameras import Camera
from utils.camera_utils import generate_interpolated_path
from utils.general_utils import safe_state
from utils.graphics_utils import getWorld2View2, getWorld2View2_torch
from metrics import convert_poses_for_evaluation, convert_poses_for_evaluation_rel, evaluate_pose
from utils.utils_poses.comp_ate import compute_rpe, compute_ATE

from scene.colmap_loader import qvec2rotmat,read_intrinsics_binary, read_extrinsics_binary
from utils.pose_utils import get_camera_from_tensor, get_tensor_from_camera, get_tensor_from_camera_torch, rotation2quad, quad2rotation, compute_relative_world_to_camera, compute_relative_world_to_camera_mod 
from utils.sfm_utils import save_time, align_pose, read_colmap_gt_pose
from lpipsPyTorch import lpips
from torchmetrics.functional.regression import pearson_corrcoef
import torch.nn as nn
from scene.gaussian_model import load_pose
from pathlib import Path
from render import render_set_optimize
import random
import copy
from scene.dataset_readers import loadCameras
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

from utils.utils_poses.relative_pose import compute_relative_poses, compute_one_wide_pose_W2C, split_poses, combine_poses, verify_poses
from metrics import evaluate_pose_together, convert_dict_to_tensor, evaluate_pose_new_mod
from utils.utils_poses.vis_pose_utils import draw_poses_pair



def get_tensor_from_camera_R_transpose(Extr):
    """
        我先默認要轉成Q^T, t
    """
    assert Extr.shape == (4, 4), f"Expected 4x4 matrix, got {Extr.shape}"
    R = Extr[:3, :3].transpose(0,1)
    t = Extr[:3, 3]
    #breakpoint()
    Q = rotation2quad(R.unsqueeze(0)).squeeze(0)
    return torch.cat([Q, t], dim=0)


def convert_Quad_to_pose_dict(Quad):
    """
        我先默認進來的都是Q^T, t
    """
    
    Q_transpose = Quad[:4]
    t = Quad[4:]
    #breakpoint()
    R = quad2rotation(Q_transpose.unsqueeze(0)).squeeze(0) #=> R^T
    
    #這邊save_all_poses 先設定 這樣才不會transpose
    #return {'R': R, 't': t}#{R, t}
    return {'R': R, 't': t}



# def convert_Quad_to_pose_dict(Quad):
#     """
#         我先默認進來的都是Q^T, t
#     """
    
#     Q_transpose = Quad[:4]
#     t = Quad[4:]
#     #breakpoint()
#     R = quad2rotation(Q_transpose.unsqueeze(0)).squeeze(0) #=> R^T
    
#     #這邊save_all_poses 先設定 這樣才不會transpose
#     #return {'R': R, 't': t}#{R, t}
#     return {'R': R, 't': t}

def obtain_cor_uw_cam(viewpoint_stack_uw, viewpoint_cam_wide):
    wide_name = viewpoint_cam_wide.image_name
    uw_name = 'uw' + wide_name[1:]
    
    uw_cam = None
    for cam in viewpoint_stack_uw:
        if cam.image_name == uw_name:
            uw_cam = cam
            break
    
    if uw_cam is None:
        print(f"Warning: Could not find matching ultra-wide camera for {wide_name}")
        return None
    
    #breakpoint()
    return uw_cam






        

# def compute_cur_wide_pose(uw_pose, relative_transform):
#     #breakpoint()
#     device = uw_pose['R'].device
#     zero_row = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32, device=device)
#     # E_uw = torch.cat([uw_pose['R'], -uw_pose['R'] @ uw_pose['t'].reshape(-1, 1)], dim=1)
#     # E_uw = torch.cat([E_uw, zero_row], dim=0)
#     E_uw = torch.cat([uw_pose['R'], uw_pose['t'].reshape(-1, 1)], dim=1)
#     E_uw = torch.cat([E_uw, zero_row], dim=0)
#     # 應用相對變換
#     E_wide = relative_transform @ E_uw
     
#     # 提取旋轉和平移
#     wide_poses = {
#         'R': E_wide[:3, :3],
#         't': -E_wide[:3, :3].T @ E_wide[:3, 3]
#     }
    
#     R = wide_poses['R'].transpose(0,1)
#     t = wide_poses['t']
#     #breakpoint()
#     Q = rotation2quad(R)
    
#     QT_wide = torch.cat([Q, t], dim=0)
    
#     return wide_poses, QT_wide

def save_all_poses(gaussians, viewpoint_stack_uw, viewpoint_stack_wide, QT_rel, output_path, stage, iteration):
    uw_poses = {}
    w_poses = {}
    for cam in viewpoint_stack_uw:
        #[Q^T, T]
        Q_and_T = gaussians.get_RT(cam.uid)
        new_img_name = cam.image_name+'.jpg'
        pose_uw_dict = convert_Quad_to_pose_dict(Q_and_T.detach())
        pose_uw_dict['R'] = pose_uw_dict['R'].T
        uw_poses[new_img_name] = pose_uw_dict
    
    for viewpoint_cam_wide in viewpoint_stack_wide:
        #cor_uw_cam = obtain_cor_uw_cam(viewpoint_stack_uw, viewpoint_cam_wide)
        new_wide_img_name = viewpoint_cam_wide.image_name+'.jpg'
        QT_wide = compute_and_check_QT_wide(viewpoint_cam_wide, viewpoint_stack_uw, gaussians, QT_rel)
        wide_pose_dict = convert_Quad_to_pose_dict(QT_wide.detach())
        wide_pose_dict['R'] = wide_pose_dict['R'].T
        
        #wide_pose_dict['R'] = wide_pose_dict['R'].T
        # if stage == "uwpretrain" and iteration == 0:
        #     diff_norm = np.linalg.norm(wide_pose_dict.R - viewpoint_cam_wide.R)
        #     threshold = 1e-5  # Adjust this based on your requirements
        #     # Assert that the difference is below the threshold
        #     assert diff_norm < threshold, f"Matrices differ too much: difference norm = {diff_norm}"
        w_poses[new_wide_img_name] = wide_pose_dict
        
    #w_poses_rel = compute_wide_poses(uw_poses, relative_pose)
    #breakpoint()
    train_poses_rel = combine_poses(uw_poses, w_poses)
    # os.makedirs(output_path, exist_ok=True)
    #breakpoint()
    np.savez(output_path, **train_poses_rel)


def finalize_pose(gaussians, viewpoint_stack_uw, viewpoint_stack_wide, relative_pose_tensor, opt_base_path, source_path, stage, iteration):

        #input: viewpoint_stack_uw, relative_pose_tensor
        #output: 
    os.makedirs(opt_base_path, exist_ok=True)
    opt_pose_path = os.path.join(opt_base_path, 'pose_optimized.npz')
    #dust3r_pose = np.load()
    #breakpoint()
    save_all_poses(gaussians, viewpoint_stack_uw, viewpoint_stack_wide, relative_pose_tensor, opt_pose_path, stage, iteration)
    #breakpoint()
    print("init_pose_path:,", opt_pose_path)
    assert os.path.exists(opt_pose_path)
    loaded_init_poses = np.load(opt_pose_path)
    
    train_img_names = ['uw_001','uw_008','uw_015', 'w_001','w_008','w_015']
    
    #breakpoint()
    pose_colmap = read_colmap_gt_pose(source_path)#args.source_path
    #breakpoint()
    # uw_poses = collect_poses(viewpoint_stack_uw)
    # w_poses = collect_poses(viewpoint_stack_wide)
    # gt_pose = combine_poses(uw_poses, w_poses)
    
    gt_train_pose_dict = {name.replace(".jpg", ""): pose for name, pose in pose_colmap.items() if name.replace(".jpg", "") in train_img_names}
    pose_init_dict = {name.replace(".jpg", ""): loaded_init_poses[name] for name in loaded_init_poses.files if name.replace(".jpg", "") in train_img_names}
    
    pose_init_dict_tensor =  convert_dict_to_tensor(pose_init_dict)
    gt_train_pose_tensor = convert_dict_to_tensor(gt_train_pose_dict)
    print("====================================================")
    #breakpoint()
    #breakpoint()
    #draw_poses_pair(gt_train_pose_tensor, pose_init_dict_tensor, opt_base_path) 
    results = evaluate_pose_new_mod(gt_train_pose_tensor, pose_init_dict_tensor, opt_base_path, train_img_names)
    
    #測試pose
    total_view = viewpoint_stack_uw + viewpoint_stack_wide
    optimized_pose = load_pose(opt_pose_path)
    viewpoint_stack = loadCameras(optimized_pose, total_view)
    
    # render_QT_dict = {}
    # for idx, view in enumerate(viewpoint_stack):
    #     camera_pose = get_tensor_from_camera(view.world_view_transform.transpose(0, 1))
    #     render_QT_dict[view.image_name] = camera_pose
    # breakpoint()
    # for name, QT in render_QT_dict.items():
    #     actual_QT = Total_QT[name]
    #     predicted_QT = render_QT_dict[name]
    #     assert torch.all_close(actual_QT, predicted_QT)
    # return results
    
def collect_poses(viewpoint_stack):
    pose_dict={}
    for cam in viewpoint_stack:
        new_mod_name = cam.image_name + '.jpg'
        pose_dict[new_mod_name] = {'R' : torch.tensor(cam.R, dtype=torch.float32, device="cuda"), 't':torch.tensor(cam.T, dtype=torch.float32, device="cuda")}
    return pose_dict


def verify_camera_centers(viewpoint_stack_uw, viewpoint_stack_wide):
    relative_pose_dict={}
    for viewpoint_cam_wide in viewpoint_stack_wide:
        cor_uw_cam = obtain_cor_uw_cam(viewpoint_stack_uw, viewpoint_cam_wide) 
          
        assert viewpoint_cam_wide.image_name.split('_')[1] == cor_uw_cam.image_name.split('_')[1]
        idx = viewpoint_cam_wide.image_name.split('_')[1]
        #breakpoint()
        # 直接使用 camera_center
        C1_w = cor_uw_cam.camera_center
        C2_w = viewpoint_cam_wide.camera_center
        
        # 計算真實的基線向量（世界坐標系下）
        baseline_w = C2_w - C1_w
        cor_uw_cam_R_tensor = torch.tensor(cor_uw_cam.R, dtype = torch.float32, device = baseline_w.device)
        device = baseline_w.device
        dtype = torch.float32
        # 將基線向量轉換到第一個相機坐標系
        baseline_c1 = cor_uw_cam_R_tensor @ baseline_w  # 假設 R 是 W2C 旋轉矩陣
        R1 = torch.tensor(cor_uw_cam.R, dtype=dtype, device=device) if not torch.is_tensor(cor_uw_cam.R) else cor_uw_cam.R.to(dtype=dtype, device=device)
        T1 = torch.tensor(cor_uw_cam.T, dtype=dtype, device=device) if not torch.is_tensor(cor_uw_cam.T) else cor_uw_cam.T.to(dtype=dtype, device=device)
        
        # 轉換 viewpoint_cam_wide 的 R 和 T
        R2 = torch.tensor(viewpoint_cam_wide.R, dtype=dtype, device=device) if not torch.is_tensor(viewpoint_cam_wide.R) else viewpoint_cam_wide.R.to(dtype=dtype, device=device)
        T2 = torch.tensor(viewpoint_cam_wide.T, dtype=dtype, device=device) if not torch.is_tensor(viewpoint_cam_wide.T) else viewpoint_cam_wide.T.to(dtype=dtype, device=device)
        
        # 獲取計算出的相對位姿
        E_rel = compute_relative_world_to_camera_mod(R1, T1, R2, T2)
        computed_t = E_rel[:3, 3]
        computed_R = E_rel[:3, :3]
        relative_pose_dict[idx] = {
            'true_baseline_in_c1': baseline_c1,
            'computed_translation': computed_t,
            'computed_rotation' : computed_R,
            'difference': torch.norm(baseline_c1 - computed_t),
            'baseline_length': torch.norm(baseline_w)
        }
    
    return relative_pose_dict

def save_gradients(iteration, gaussians, log_file_path=None):
    # 如果沒有指定log文件路徑，創建一個包含時間戳的文件名
    if log_file_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = f"gradient_log_{timestamp}.txt"
    
    # 每500次迭代檢查梯度
    if iteration % 50 == 0:
        # 獲取梯度（如果存在）
        relative_pose_grad = gaussians.relative_pose.grad if gaussians.relative_pose.grad is not None else "No gradient"
        P_grad = gaussians.P.grad if gaussians.P.grad is not None else "No gradient"
        
        # 準備寫入的文本
        log_text = f"\nIteration {iteration} Gradient Check:\n"
        log_text += f"relative_pose gradient: {relative_pose_grad}\n"
        log_text += f"P gradient: {P_grad}\n"
        log_text += "-" * 50 + "\n"  # 分隔線
        
        # 以追加模式寫入文件
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(log_text)




# def compute_cur_wide_pose(viewpoint_cam_wide, viewpoint_stack_uw, gaussians):
#     # 檢查一下gradient也有沒有段
#     # 找到對應的 ultra-wide camera
#     #breakpoint()
#     wide_name = viewpoint_cam_wide.image_name
#     uw_name = 'uw' + wide_name[1:]
    
#     uw_cam = None
#     for cam in viewpoint_stack_uw:
#         if cam.image_name == uw_name:
#             uw_cam = cam
#             break
    
#     if uw_cam is None:
#         print(f"Warning: Could not find matching ultra-wide camera for {wide_name}")
#         return None
    
#     #整理矩陣
#     uw_cam_pose = gaussians.get_RT(uw_cam.uid)#[Q^T, t]
#     QT_rel_mod = gaussians.relative_pose.clone()#[Q^T, t]
    
    
#     uw_pose = convert_Quad_to_pose_dict(uw_cam_pose)
    
#     rel_pose = convert_Quad_to_pose_dict(QT_rel_mod)
#     device = uw_cam_pose.device
#     #breakpoint()
#     zero_row = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32, device=device)  
#     E_uw = torch.cat([uw_pose['R'], -uw_pose['R'] @ uw_pose['t'].reshape(-1, 1)], dim=1)
#     E_uw = torch.cat([E_uw, zero_row], dim=0)
    
#     relative_transform = torch.cat([rel_pose['R'], rel_pose['t'].reshape(-1, 1)], dim=1)
#     relative_transform = torch.cat([relative_transform, zero_row], dim=0)
#     #breakpoint()
#     # 應用相對變換
    
#     E_wide = relative_transform @ E_uw
    
#     #轉成[R O]
#     #    [T I]
#     R = E_wide[:3,:3].transpose(0,1)
#     viewpoint_cam_wide.R = R
#     T = E_wide[:3, 3]
#     viewpoint_cam_wide.T = T
#     R_npy = R.detach().cpu().numpy()
#     T_npy = T.detach().cpu().numpy()
#     tmp_world_transform_test = torch.tensor(getWorld2View2(R_npy, T_npy)).transpose(0, 1).cuda()
#     tmp_world_transform = getWorld2View2_torch(R, T).float().transpose(0, 1).cuda()
#     # world_transform = tmp_world_transform.clone()
#     #breakpoint()
#     QT_wide = get_tensor_from_camera_R_transpose(tmp_world_transform)#
#     #breakpoint()
#     #修改viewpoint_cam_wide
#     return QT_wide#這邊要render 所以你要return [Q^T, t]


# def compute_wide_poses_all(uw_poses, relative_transform):
#     """
#     使用相對位姿計算wide view的poses
#     """
#     wide_poses = {}
#     device = next(iter(uw_poses.values()))['R'].device
#     for name, uw_pose in uw_poses.items():
#         #breakpoint()
#         # 構建UW的4x4變換矩陣
        
#         zero_row = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32, device=device)
        
#         E_uw = torch.cat([uw_pose['R'], -uw_pose['R'] @ uw_pose['t'].reshape(-1, 1)], dim=1)
#         E_uw = torch.cat([E_uw, zero_row], dim=0)
        
#         # 應用相對變換
#         E_wide = relative_transform @ E_uw
        
#         uw_num = name.split('_')[1].split('.')[0]
#         wide_num = f'w_{uw_num}.jpg' 
#         # 提取旋轉和平移
#         wide_poses[wide_num] = {
#             'R': E_wide[:3, :3],
#             't': -torch.linalg.solve(E_wide[:3, :3], E_wide[:3, 3])
#         }
    
#     return wide_poses



def load_and_prepare_confidence(confidence_path, device='cuda', scale=(0.1, 1.0)):
    """
    Loads, normalizes, inverts, and scales confidence values to obtain learning rate modifiers.
    
    Args:
        confidence_path (str): Path to the .npy confidence file.
        device (str): Device to load the tensor onto.
        scale (tuple): Desired range for the learning rate modifiers.
    
    Returns:
        torch.Tensor: Learning rate modifiers.
    """
    # Load and normalize
    confidence_np = np.load(confidence_path)
    confidence_tensor = torch.from_numpy(confidence_np).float().to(device)
    normalized_confidence = torch.sigmoid(confidence_tensor)

    # Invert confidence and scale to desired range
    inverted_confidence = 1.0 - normalized_confidence
    min_scale, max_scale = scale
    lr_modifiers = inverted_confidence * (max_scale - min_scale) + min_scale
    
    return lr_modifiers

# def load_pose_dict(path):
#     colmap_poses = []
#     quat_pose = []
    
#     loaded_poses = np.load(path)

#     pose_optimized = {name: loaded_poses[name] for name in loaded_poses.files}
#     sorted_pose_optimized = dict(sorted(pose_optimized.items(), key=lambda x: x[0]))
#     dict_pose={}

#     for img_name,pose in sorted_pose_optimized.items():
#         q_pose = get_tensor_from_camera(pose.transpose(0, 1))
#         #quat_pose.append(q_pose)
#         dict_pose[img_name] = q_pose

#     #poses = torch.stack(quat_pose)
#     return dict_pose

# def compute_rel_pose_and_evaluate(src_path, output_path):
#     pose_colmap = read_colmap_gt_pose(src_path)#args.source_path
#     train_img_names = ['uw_001', 'uw_008', 'uw_015','w_001', 'w_008','w_015']
#     #breakpoint()
#     gt_train_pose = {name: pose for name, pose in pose_colmap.items() if name.replace(".jpg", "") in train_img_names}
#     #[R T]
#     #[0 I]
#     gt_train_pose_tensor = {name: torch.tensor(pose, dtype=torch.float32) for name, pose in pose_colmap.items() if name.replace(".jpg", "") in train_img_names}
    

#     uw_poses, w_poses = split_poses(gt_train_pose)
#     #breakpoint()
#     relative_pose = compute_relative_poses(uw_poses, w_poses)
#     print("After computing_relative_pose:", relative_pose)
#     rel_pose_path=os.path.join(src_path, "sparse/0")
#     os.makedirs(rel_pose_path, exist_ok=True)
#     np.save(rel_pose_path + "/relative_pose_ori", relative_pose)
#     #[R T]
#     #[0 I]
#     print("relative pose saving in :", rel_pose_path + "/relative_pose_ori")
    
#     device = next(iter(uw_poses.values()))['R'].device
#     relative_pose = torch.tensor(relative_pose, dtype=torch.float32, device= device)
#     #breakpoint()
#     w_poses_rel = compute_wide_poses(uw_poses, relative_pose)
#     #breakpoint()
#     train_poses_rel = combine_poses(uw_poses, w_poses_rel)
#     init_pose_path = os.path.join(output_path + f'/pose/ours_0')
#     #def evaluate_pose_new(gt_train_pose, pose_optimized, pose_path, name_list, encoding, rel_or_not):
    
#     sorted_train_poses_E_rel = dict(sorted(train_poses_rel.items(), key=lambda x: x[0]))
#     sorted_train_poses_gt = dict(sorted(gt_train_pose.items(), key=lambda x: x[0]))
#     #breakpoint()

#     wide_name_list = ['w_001','w_008','w_015']
    
#     pose_optimized_aligned_tensor = convert_dict_to_tensor(sorted_train_poses_E_rel)
#     poses_gt_tensor = convert_dict_to_tensor(sorted_train_poses_gt)
    
#     #breakpoint()
#     draw_poses_pair(poses_gt_tensor, pose_optimized_aligned_tensor, init_pose_path) 
#     results = evaluate_pose_new_mod(sorted_train_poses_gt, sorted_train_poses_E_rel, init_pose_path, train_img_names)



def compute_and_check_QT_wide(viewpoint_cam_wide, viewpoint_stack_uw, gaussians, QT_rel, tol=1e-5):
    
    #QT_wide_ori = gaussians.get_RT(viewpoint_cam_wide.uid)
    

    cor_uw_cam = obtain_cor_uw_cam(viewpoint_stack_uw, viewpoint_cam_wide)
    cor_pose_uw = gaussians.get_RT(cor_uw_cam.uid)
    uw_pose_dict = convert_Quad_to_pose_dict(cor_pose_uw)

    #breakpoint()
    assert viewpoint_cam_wide.image_name.split('_')[1] == cor_uw_cam.image_name.split('_')[1]

    #QT_rel = gaussians.relative_pose.clone()
    Rel_pose_4X4 = get_camera_from_tensor(QT_rel)

    
    W2C_wide, QT_wide = compute_one_wide_pose_W2C(uw_pose_dict, Rel_pose_4X4)
    
    #breakpoint()
    # viewpoint_cam_wide_R_tensor = torch.tensor(viewpoint_cam_wide.R, dtype=torch.float32, device=W2C_wide['R'].device)
    # viewpoint_cam_wide_T_tensor = torch.tensor(viewpoint_cam_wide.T, dtype=torch.float32, device=W2C_wide['t'].device)

    # assert torch.allclose(W2C_wide['R'], viewpoint_cam_wide_R_tensor, atol=tol), \
    #     f"Rotation matrices are too different: {W2C_wide['R']} vs {viewpoint_cam_wide.R}"
    # assert torch.allclose(W2C_wide['t'], viewpoint_cam_wide_T_tensor, atol=tol), \
    #     f"Translation vectors are too different: {W2C_wide['t']} vs {viewpoint_cam_wide.T}"
    
    # #breakpoint()
    # assert torch.allclose(QT_wide_ori, QT_wide, atol=tol), \
    #     f"QT_wide vectors are too different: {QT_wide_ori} vs {QT_wide}"
    

    return QT_wide



def training(dataset, opt, pipe, args):
    testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from = args.test_iterations, \
            args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from

    warmup_iter = 4999
    #breakpoint()
    if args.stage == "uw_pretrain":
        first_iter = 0
        tb_writer = prepare_output_and_logger(dataset)
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(args, gaussians, shuffle=False, get_all_cam="all")
        test_args = copy.deepcopy(args)
        test_args.eval = True
        test_scene = Scene(test_args, gaussians,shuffle=False)
         # per-point-optimizer
        confidence_path = os.path.join(dataset.source_path, f"sparse_{dataset.n_views}/0", "confidence_dsp.npy")
        confidence_lr = load_and_prepare_confidence(confidence_path, device='cuda', scale=(1, 100))
        
        #relative pose
        new_src_path = os.path.join(args.source_path, f"sparse_{args.n_views}","1")
        rel_pose_path = os.path.join(new_src_path, "relative_pose_from_dust3R.npy")
        Quad_rel = np.load(rel_pose_path)
        #breakpoint()
        output_pose_eval_path = scene.model_path + f'/pose/ours_{first_iter}'
        
        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_stack_uw = viewpoint_stack[0:3]
        viewpoint_stack_wide = viewpoint_stack[3:6]
        
        #breakpoint()
        finalize_pose(gaussians, viewpoint_stack_uw, viewpoint_stack_wide, Quad_rel, output_pose_eval_path, args.source_path, args.stage, first_iter)
        
        if opt.pp_optimizer:
            #def training_setup_pp(self, training_args, stage, Q_rel, confidence_lr=None)
            gaussians.training_setup_pp(opt, args.stage, Quad_rel, confidence_lr)                          
        else:
            gaussians.training_setup(opt, args.stage, Quad_rel)
        #relative_poses = load_and_process_scene(args.source_path)
        
        
        
        #testing
        #compute_rel_pose_and_evaluate(args.source_path, scene.model_path)
        
        # 
        # if not os.path.exists(rel_pose_path):
        #     print(f"File not found: {rel_pose_path}")
        # relative_pose = np.load(rel_pose_path)

        # R_rel = relative_pose[:3,:3]
        # T_rel = relative_pose[:3,3]
        # #這邊是[R T]
        # #     [O I]
        # relative_pose_tensor = torch.tensor(relative_pose).float().cuda()
        # Quad_rel = get_tensor_from_camera_R_transpose(relative_pose_tensor)
        #print("Quad_rel:",Quad_rel)
        uw_poses = collect_poses(viewpoint_stack_uw)
        w_poses = collect_poses(viewpoint_stack_wide)
        ################################ 計算colmap_GT relative pose######################################
        # relative_pose_4X4, E_rel_all = compute_relative_poses(uw_poses, w_poses, viewpoint_stack_wide)
        # rel_pose_path = os.path.join(args.source_path, "sparse", "0")
        # os.makedirs(rel_pose_path, exist_ok=True)
        # np.save(os.path.join(rel_pose_path, "relative_pose_ori"), relative_pose_4X4)
        # Quad_rel_ori = get_tensor_from_camera(relative_pose_4X4)
        ##################################################################################################################
        
        #breakpoint()
        if checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)
            #breakpoint()
            
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        iter_start = torch.cuda.Event(enable_timing = True)
        iter_end = torch.cuda.Event(enable_timing = True)
        progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

        viewpoint_stack = scene.getTrainCameras().copy()
        #breakpoint()
        viewpoint_stack_uw = viewpoint_stack[0 : 3]
        viewpoint_stack_wide = viewpoint_stack[3 : 6]

        ema_loss_for_log = 0.0

        progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")    
        first_iter += 1
        start = time()
        

        depth_loss = torch.tensor(0.0).cuda()
        depth_wide_loss = torch.tensor(0.0).cuda()
        depth_loss_pseudo_wide_final = torch.tensor(0.0).cuda()
        deta_x_loss = torch.tensor(0.0).cuda()
        deta_c_loss = torch.tensor(0.0).cuda()
        
        for iteration in range(first_iter, warmup_iter + 1):
       
            iter_start.record()
            gaussians.update_learning_rate(iteration)

            if opt.optim_pose==False:
                gaussians.P.requires_grad_(False)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True
            
            # Pick a random Camera
            viewpoint_cam = viewpoint_stack_uw[(randint(0, len(viewpoint_stack_uw)-1))]
            pose_uw = scene.gaussians.get_RT(viewpoint_cam.uid)
            #breakpoint()
            
            
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, info=None, camera_pose = pose_uw)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()

            Ll1 =  l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)
            loss1 = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
            loss = loss1
            loss.backward()
            iter_end.record()

            
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, depth_loss, depth_wide_loss, depth_loss_pseudo_wide_final, deta_x_loss, deta_c_loss,
                                testing_iterations ,scene, test_scene, render, (pipe, background), args)#, E_rel
                
            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                if (iteration == warmup_iter):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)
                    warmup_opt_pose_path = scene.model_path + f'/pose/ours_{iteration}'
                    
                    #===============================================================================================================================#
                    finalize_pose(gaussians, viewpoint_stack_uw, viewpoint_stack_wide, Quad_rel, warmup_opt_pose_path, args.source_path, args.stage, iteration)
                    #===============================================================================================================================#

                # if iteration < opt.densify_until_iter:  
                #     gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                #     gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                #     if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                #         size_threshold = None
                #         gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                #     if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                #         gaussians.reset_opacity()
                
                
                
                if iteration < opt.iterations:
                    
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                # Log and save
    elif args.stage == "uw2wide":
        # test gradient information
        grad_info_path = os.path.join(args.source_path, "grad_info")
        os.makedirs(grad_info_path, exist_ok = True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        grad_file_path = os.path.join(grad_info_path, f"gradient_log_{timestamp}.txt")
        
        
        # 寫入初始信息
        with open(grad_file_path, 'w', encoding='utf-8') as f:
            f.write(f"Gradient Log Started at {datetime.now()}\n")
            f.write("=" * 50 + "\n")
        
        
        first_iter = 0
        tb_writer = prepare_output_and_logger(dataset)
        gaussians = GaussianModel(args.sh_degree)

        scene = Scene(args, gaussians, load_iteration=warmup_iter, shuffle=False, get_all_cam="all")
        test_args = copy.deepcopy(args)
        test_args.eval = True
        test_scene = Scene(test_args, gaussians, load_iteration=warmup_iter, shuffle=False)
        ###########################################################################################
        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_stack_uw = viewpoint_stack[0 : 3]
        viewpoint_stack_wide = viewpoint_stack[3 : 6]
        
        if args.fsgs_loss:
            pseudo_stack_wide = scene.getPseudoCameras_wide().copy()
            pseudo_stack_uw = scene.getPseudoCameras_uw().copy()
            assert len(pseudo_stack_wide) > 0
            assert len(pseudo_stack_uw) > 0
        
        #breakpoint()
        #rel_pose_dict = verify_camera_centers(viewpoint_stack_uw, viewpoint_stack_wide)
        
        #breakpoint()
        #Quad_rel_test = get_tensor_from_camera_R_transpose(relative_pose)
        
        ###############################計算ground truth pose比較############################################################
        # pose_colmap = read_colmap_gt_pose(args.source_path)#args.source_path
        # train_img_names = ['uw_001', 'uw_008', 'uw_015','w_001', 'w_008','w_015']
        # gt_train_pose = {name: pose for name, pose in pose_colmap.items() if name.replace(".jpg", "") in train_img_names}
        # # #[R T]
        # # #[0 I]
        # # gt_train_pose_tensor = {name: torch.tensor(pose, dtype=torch.float32) for name, pose in pose_colmap.items() if name.replace(".jpg", "") in train_img_names}
        

        # uw_poses_test, w_poses_test = split_poses(gt_train_pose)
        # relative_pose_test, E_rel_all_test = compute_relative_poses(uw_poses_test, w_poses_test, viewpoint_stack_wide)
        # uw_poses = collect_poses(viewpoint_stack_uw)
        # w_poses = collect_poses(viewpoint_stack_wide)
        
        # relative_pose_4X4, E_rel_all = compute_relative_poses(uw_poses, w_poses, viewpoint_stack_wide)
        # Quad_rel = get_tensor_from_camera(relative_pose_4X4)
        # poses_gt = combine_poses(uw_poses, w_poses)
        # init_pose_path = os.path.join(scene.model_path ,'pose/ours_0')
        # sorted_pose_gt_E_rel = dict(sorted(poses_gt.items(), key=lambda x: x[0]))
        # poses_gt_tensor = convert_dict_to_tensor(sorted_pose_gt_E_rel)
        # draw_poses(poses_gt_tensor, init_pose_path) 
        ###########################################################################################
        # QT_wide_rel_dict = {}
        # QT_wide_gt_dict = {}
        # for viewpoint_cam_wide in viewpoint_stack_wide:
        #     QT_wide_gt_dict[viewpoint_cam_wide.image_name] = gaussians.get_RT(viewpoint_cam_wide.uid)
        #     QT_wide_rel_dict[viewpoint_cam_wide.image_name] = compute_and_check_QT_wide(viewpoint_cam_wide, viewpoint_stack_uw, gaussians, Quad_rel)
        #check_QT_wide_gt(gaussians, viewpoint_stack_uw, viewpoint_stack_wide, relative_pose_4x4)
        #relative_pose_4X4_tensor = torch.tensor(relative_pose_4X4, dtype=torch.float32, device = E_rel_all[0].device)
        #check_dict = verify_poses(E_rel_all, relative_pose_4X4_tensor, uw_poses,  w_poses)
        #breakpoint()
        
        #breakpoint()
        ###########################################################################################
        #draw_poses
        
        #baseline_dict = compute_baesline(viewpoint_stack_uw, viewpoint_stack_wide)
        #breakpoint()
        new_src_path = os.path.join(args.source_path, f"sparse_{args.n_views}","1")
        rel_pose_path = os.path.join(new_src_path, "relative_pose_from_dust3R.npy")
        Quad_rel = np.load(rel_pose_path)
        #gaussians.training_setup(opt, args.stage, Quad_rel)#
        confidence_path = os.path.join(dataset.source_path, f"sparse_{dataset.n_views}/0", "confidence_dsp.npy")
        confidence_lr = load_and_prepare_confidence(confidence_path, device='cuda', scale=(1, 100))
        
        if opt.pp_optimizer:
            #def training_setup_pp(self, training_args, stage, Q_rel, confidence_lr=None)
            gaussians.training_setup_pp(opt, args.stage, Quad_rel, confidence_lr)                          
        else:
            gaussians.training_setup(opt, args.stage, Quad_rel)
            
        if checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            #gaussians.P = np.load()
            gaussians.restore(model_params, opt)
            
        

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)
        progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

        

        
        image_height = viewpoint_stack[0].image_height
        image_width = viewpoint_stack[0].image_width
        row_indices = torch.arange(0, image_height).view(-1, 1).repeat(1, image_width).cuda()
        column_indices = torch.arange(0, image_width).repeat(image_height, 1).cuda()
        mask = torch.ones((1, image_height, image_width), dtype=torch.float32).cuda()
        mask_wide = mask
        ema_loss_for_log = 0.0
        first_iter += 1

        # pose_iter = 2000 #2000
        depth_loss = torch.tensor(0.0).cuda()
        depth_wide_loss = torch.tensor(0.0).cuda()
        depth_loss_pseudo_uw = torch.tensor(0.0).cuda()
        depth_loss_pseudo_wide_final = torch.tensor(0.0).cuda()
        deta_x_loss = torch.tensor(0.0).cuda()
        deta_c_loss = torch.tensor(0.0).cuda()
        start = time()
        for iteration in range(first_iter, opt.iterations + 1):
            iter_start.record()
            gaussians.update_learning_rate(iteration)

            if opt.optim_pose==False:
                gaussians.P.requires_grad_(False)            
                
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            viewpoint_cam = viewpoint_stack_uw[(randint(0, len(viewpoint_stack_uw)-1))]
            viewpoint_cam_wide = viewpoint_stack_wide[(randint(0, len(viewpoint_stack_wide)-1))]
            
            if (iteration - 1) == debug_from:
                pipe.debug = True

            # uw reconstruction loss
            pose_uw = gaussians.get_RT(viewpoint_cam.uid)
            #get_tensor_from_camera(viewpoint_cam.world_view_transform.transpose(0, 1)) != pose_uw
            #
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, info=None, camera_pose=pose_uw)
            image = render_pkg["render"]
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 =  l1_loss(image, gt_image)
            loss1 = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)))  
        
            #w reconstruction loss
            QT_wide = compute_and_check_QT_wide(viewpoint_cam_wide, viewpoint_stack_uw, gaussians, gaussians.relative_pose)
            #QT_wide_test = gaussians.get_RT(viewpoint_cam_wide.uid)
            #breakpoint()
            render_pkg_wide = render(viewpoint_cam_wide, gaussians, pipe, background, info={"c":1., "target":"cx"}, camera_pose=QT_wide)

            image_wide = render_pkg_wide["render"] 
            viewspace_point_tensor, visibility_filter, radii = render_pkg_wide["viewspace_points"], render_pkg_wide["visibility_filter"], render_pkg_wide["radii"]
            gt_image_wide = viewpoint_cam_wide.original_image.cuda()
            loss2 = ((1.0 - opt.lambda_dssim) * l1_loss(image_wide, gt_image_wide) + opt.lambda_dssim * (1.0 - ssim(image_wide, gt_image_wide)))

            loss = 0.5*loss1 + 0.5*loss2
            # identity regulation
            # gt_image = viewpoint_cam.original_image.cuda()
            # render_pkg_uwc0 = render(viewpoint_cam, gaussians, pipe, background, info={"c":0., "target":"cx"}, camera_pose=pose_uw)
            # deta_x = render_pkg_uwc0["deta_x"]
            # deta_c = render_pkg_uwc0["deta_c"]
            
            # loss3 = torch.abs(deta_x).mean() + torch.abs(deta_c).mean() #+ torch.abs(deta_o).mean()
            # loss = loss + 1.0*loss3
            if args.with_identity_loss:
                #print("training with identity loss")
                #breakpoint()
                delta_x = render_pkg_wide["deta_x"]
                delta_c = render_pkg_wide["deta_c"]
                deta_x_loss = torch.abs(delta_x).mean()
                deta_c_loss = torch.abs(delta_c).mean()
                # if args.with_distortion_loss:
                #     #print("training with distortion loss")
                #     deta_x_loss = distortion_loss(gaussians, delta_x, viewpoint_cam_wide)
                # else:
                #     deta_x_loss = torch.abs(delta_x).mean()
                loss3 = deta_x_loss + deta_c_loss
                loss = loss + args.identity_loss_weight * loss3 
            disparity_loss=0.0
            disparity_loss_wide=0.0
            current_smooth_weight=0.0
            if args.fsgs_loss:
                #print("Training with FSGS Loss")
                #==================wide depth loss=====================
                rendered_wide_depth = render_pkg_wide["depth"][0]
                #breakpoint()
                midas_wide_depth = torch.tensor(viewpoint_cam_wide.depth_image).cuda()
                rendered_wide_depth = rendered_wide_depth.reshape(-1, 1)
                midas_wide_depth = midas_wide_depth.reshape(-1, 1)

                depth_wide_loss = min(
                                (1 - pearson_corrcoef( - midas_wide_depth, rendered_wide_depth)),
                                (1 - pearson_corrcoef(1 / (midas_wide_depth + 200.), rendered_wide_depth))
                )
                loss += args.depth_weight * depth_wide_loss
                #==================ultra-wide depth loss=====================
                rendered_depth = render_pkg["depth"][0]
                #breakpoint()
                midas_depth = torch.tensor(viewpoint_cam.depth_image).cuda()
                rendered_depth = rendered_depth.reshape(-1, 1)
                midas_depth = midas_depth.reshape(-1, 1)

                depth_loss = min(
                                (1 - pearson_corrcoef( - midas_depth, rendered_depth)),
                                (1 - pearson_corrcoef(1 / (midas_depth + 200.), rendered_depth))
                )
                loss += args.depth_weight * depth_loss
                #==================pseudo training view======================
                # if iteration > args.end_sample_pseudo:
                #     args.depth_weight = 0.001
                # if iteration % args.sample_pseudo_interval == 0 and iteration > args.start_sample_pseudo and iteration < args.end_sample_pseudo:
                #     if not pseudo_stack_wide:
                #         pseudo_stack_wide = scene.getPseudoCameras_wide().copy()
                        
                    
                #     pseudo_cam_wide = pseudo_stack_wide.pop(randint(0, len(pseudo_stack_wide) - 1))
                #     # device = "cuda"
                #     pseudo_cam_wide_R_tensor = torch.tensor(pseudo_cam_wide.R, dtype = torch.float32).cuda()
                #     pseudo_cam_wide_T_tensor = torch.tensor(pseudo_cam_wide.T, dtype = torch.float32).cuda()
                #     zero_row = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32).cuda()
                #     Extr_pseudo_cam = torch.cat([pseudo_cam_wide_R_tensor, pseudo_cam_wide_T_tensor.reshape(-1, 1)], dim=1)
                #     Extr_pseudo_cam = torch.cat([Extr_pseudo_cam, zero_row], dim=0)
                #     QT_pseudo_cam = get_tensor_from_camera_R_transpose(Extr_pseudo_cam)
                #     #pseudo_cam.image_name.startswith('w') and not pseudo_cam.image_name.startswith('u'):
                #     render_pkg_pseudo_wide = render(pseudo_cam_wide, gaussians, pipe, background, info={"c":1., "target":"cx"}, camera_pose= QT_pseudo_cam)
                #     #render_pkg_pseudo_wide = render(pseudo_cam, gaussians, pipe, background, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, info={"c":1., "target":"cx"})
                #     rendered_depth_pseudo_wide = render_pkg_pseudo_wide["depth"][0]
                #     midas_depth_pseudo_wide = estimate_depth(render_pkg_pseudo_wide["render"], mode='train')

                #     rendered_depth_pseudo_wide = rendered_depth_pseudo_wide.reshape(-1, 1)
                #     midas_depth_pseudo_wide = midas_depth_pseudo_wide.reshape(-1, 1)
                #     depth_loss_pseudo_wide = (1 - pearson_corrcoef(rendered_depth_pseudo_wide, -midas_depth_pseudo_wide)).mean()
                    
                #     if torch.isnan(depth_loss_pseudo_wide).sum() == 0:
                #         loss_scale = min((iteration - args.start_sample_pseudo) / 500., 1)
                #         depth_loss_pseudo_wide_final = loss_scale * args.depth_pseudo_weight * depth_loss_pseudo_wide
                #         loss += depth_loss_pseudo_wide_final
            loss.backward()
            save_gradients(iteration, gaussians, grad_file_path)

            
            
            with torch.no_grad():
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)
                    
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, depth_loss, depth_wide_loss, depth_loss_pseudo_wide_final, deta_x_loss, deta_c_loss,
                                testing_iterations, scene, test_scene, render, (pipe, background), args)#,E_rel

                # if iteration < opt.densify_until_iter:  
                #     gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                #     gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                #     if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                #         size_threshold = None
                #         gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                #     if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                #         gaussians.reset_opacity()
                        
                if iteration < opt.iterations:

                    gaussians.optimizer.step()
                    # gaussians.mlp_optimizer.step()
                    # gaussians.mlp_optimizer.zero_grad(set_to_none = True)
                    # gaussians.mlp_scheduler.step() 
                    gaussians.optimizer.zero_grad(set_to_none = True)
                                         
                if iteration == opt.iterations:
                    end = time()
                    train_time_wo_log = end - start
                    save_time(scene.model_path, '[2] train_joint_TrainTime', train_time_wo_log)
                    
                    finish_opt_pose_path = scene.model_path + f'/pose/ours_{iteration}'
                    os.makedirs(finish_opt_pose_path , exist_ok=True)
                    
                    # Total_QT={}
                    # for cam in viewpoint_stack_uw:
                    #     QT_uw = gaussians.get_RT(cam.uid)
                    #     Total_QT[cam.image_name] = QT_uw
                    
                    # for viewpoint_cam_wide in viewpoint_stack_wide:
                    #     wide_img_name = cam.image_name.replace('u', '')
                    #     Total_QT[wide_img_name] = compute_and_check_QT_wide(viewpoint_cam_wide, viewpoint_stack_uw, gaussians, gaussians.relative_pose)
                    #breakpoint()
                    finalize_pose(gaussians, viewpoint_stack_uw, viewpoint_stack_wide, gaussians.relative_pose, finish_opt_pose_path, args.source_path, args.stage, iteration)
                
            
            
            
    end = time()
    train_time = end - start
    save_time(scene.model_path, '[2] train_joint', train_time)


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer



def training_report(tb_writer, iteration, Ll1, loss, l1_loss, depth_loss, depth_wide_loss, depth_loss_pseudo_wide_final, deta_x_loss, deta_c_loss, testing_iterations, scene : Scene, test_scene : Scene, renderFunc, renderArgs, args):#, E_rel
    #breakpoint()
    #if iteration % 2000 == 0 and args.stage == "uw2wide":
        # test_img_list = ['w_003', 'w_006', 'w_009']
        # test_cam = [cam for cam in test_scene.getTestCameras() if cam.image_name in test_img_list]
        # viewpoint_stack = scene.getTrainCameras()
        # pipeline = renderArgs[0]
        # background = renderArgs[1]
        # render_set_optimize(scene.model_path, "test", iteration, test_cam, scene.gaussians, pipeline, background, args)
        # render_set(
        #     scene.model_path,
        #     "train",
        #     iteration,
        #     viewpoint_stack,
        #     scene.gaussians,
        #     pipeline,
        #     background,
        # )

    #with torch.no_grad():
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/depth_loss', depth_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/depth_wide_loss', depth_wide_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/depth_pseudo_wide_loss', depth_loss_pseudo_wide_final.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/deta_x_loss', deta_x_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/deta_c_loss', deta_c_loss.item(), iteration)
        tb_writer.add_scalar('iter_time', iteration)
        
    

    # Report test and samples of training set
    if iteration in testing_iterations or iteration % 5000 == 0:
        #torch.cuda.empty_cache()
        #選w_006.jpg
        
        #breakpoint()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                            {'name': 'train', 'cameras' : scene.getTrainCameras()})
        #validation_configs = ({'name': 'train', 'cameras' : scene.getTrainCameras()})
        viewpoint_stack = scene.getTrainCameras()
        #viewpoint_stack_wide = viewpoint_stack[3 : 6]
        viewpoint_stack_uw = viewpoint_stack[0 : 3]

        for config in validation_configs:
            #breakpoint()
            if config['cameras']:
                l1_train_wide = 0.0
                psnr_train_wide = 0.0
                l1_train_uw = 0.0
                psnr_train_uw = 0.0
                l1_test = 0.0
                psnr_test = 0.0
                

                for idx, viewpoint in enumerate(config['cameras']):
                    if config['name']=="train":
                        if viewpoint.image_name.startswith('u') and not viewpoint.image_name.startswith('w'):
                            #QT_wide = compute_wide_pose(viewpoint, viewpoint_stack_uw, scene.gaussians, E_rel)
                            pose_uw = scene.gaussians.get_RT(viewpoint.uid)
                            image_uw = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, info=None, camera_pose = pose_uw)["render"], 0.0, 1.0)
                            gt_image_uw = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                            l1_train_uw += l1_loss(image_uw, gt_image_uw).mean().double()
                            psnr_train_uw += psnr(image_uw, gt_image_uw).mean().double()
                            if tb_writer and (idx < 8):
                                tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image_uw[None], global_step=iteration)
                                if iteration == testing_iterations[0]:
                                    tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image_uw[None], global_step=iteration)
                            
                            
                        elif viewpoint.image_name.startswith('w') and not viewpoint.image_name.startswith('u') and args.stage == "uw2wide":
                            #QT_wide= compute_cur_wide_pose(viewpoint, viewpoint_stack_uw, scene.gaussians)  
                            QT_wide = compute_and_check_QT_wide(viewpoint, viewpoint_stack_uw, scene.gaussians, scene.gaussians.relative_pose)
                            image_wide = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, info={"c":1., "target":"cx"}, camera_pose = QT_wide)["render"], 0.0, 1.0)
                            gt_image_wide = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                            l1_train_wide += l1_loss(image_wide, gt_image_wide).mean().double()
                            psnr_train_wide += psnr(image_wide, gt_image_wide).mean().double()    
                            if tb_writer and (idx < 8):
                                tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image_wide[None], global_step=iteration)
                                if iteration == testing_iterations[0]:
                                    tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image_wide[None], global_step=iteration)
                            
                    # elif config['name']=="test" and stage == "uw2wide":
                    #     #只有wide
                    #     #breakpoint()
                        
                    #     QT_test = get_tensor_from_camera(viewpoint.world_view_transform.transpose(0, 1))
                    #     #breakpoint()
                    #     image_test = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, info={"c":1., "target":"cx"}, camera_pose = QT_test)["render"], 0.0, 1.0)
                    #     gt_image_test = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    #     l1_test += l1_loss(image_test, gt_image_test).mean().double()
                    #     psnr_test += psnr(image_test, gt_image_test).mean().double() 
                    #     if tb_writer and (idx < 8):
                    #         tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image_test[None], global_step=iteration)
                    #     # pose_colmap = read_colmap_gt_pose(gt_pose_path)
                        
                        
                    # if tb_writer and (idx < 5):
                    #     tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                    #     if iteration == testing_iterations[0]:
                    #         tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    
                
                
                
                n_wide = sum(1 for v in config['cameras'] if v.image_name.startswith('w_') and config['name']=="train")
                if args.stage != "uw2wide":
                    n_wide = 0
                n_uw = sum(1 for v in config['cameras'] if v.image_name.startswith('uw_'))
                n_wide_test = sum(1 for v in config['cameras'] if v.image_name.startswith('w_') and config['name']=="test")

                if n_wide > 0:
                    psnr_train_wide /= n_wide
                    l1_train_wide /= n_wide
                if n_uw > 0:
                    psnr_train_uw /= n_uw
                    l1_train_uw /= n_uw
                    
                if n_wide_test > 0:
                    psnr_test /= n_wide_test
                    l1_test /= n_wide_test
                
                if config['name'] == "train":
                    print("\n[ITER {}] Evaluating for UW {}: L1 {} PSNR {}".format(iteration, config['name'], l1_train_uw, psnr_train_uw))
                    if args.stage == "uw2wide":
                        print("\n[ITER {}] Evaluating for wide {}: L1 {} PSNR {}".format(iteration, config['name'], l1_train_wide, psnr_train_wide))
                    
                elif config['name'] == "test":
                    print("\n[ITER {}] Evaluating for wide {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                #breakpoint()
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss_wide', l1_train_wide, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr_wide', psnr_train_wide, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss_uw', l1_train_uw, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr_uw', psnr_train_uw, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr_test', psnr_test, iteration)
                    #tb_writer.add_scalar(config['name'] + '/')

        
        
        # if tb_writer and stage == "uw2wide":
        #     #pose
        #     #做pose error
        #     relative_pose_tensor = scene.gaussians.relative_pose.detach()
        #     #breakpoint()
        #     finish_opt_pose_path = Path(scene.model_path + f'/pose/ours_{iteration}')
        #     os.makedirs(finish_opt_pose_path , exist_ok=True)
        #     rel_pose = convert_Quad_to_pose_dict(relative_pose_tensor)
        #     #breakpoint()
        #     device = relative_pose_tensor.device
        #     zero_row = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32, device=device)
        #     relative_transform = torch.cat([rel_pose['R'], rel_pose['t'].reshape(-1, 1)], dim=1)
        #     relative_transform = torch.cat([relative_transform, zero_row], dim=0)
        #     #def finalize_pose(gaussians, viewpoint_stack_uw, relative_pose_tensor, opt_base_path, gt_pose_path):
        #     results = finalize_pose(scene.gaussians, viewpoint_stack_uw, relative_transform, finish_opt_pose_path, gt_pose_path)
        #     #all
        #     tb_writer.add_scalar("Pose/All - RPE_t", results["all"]["RPE_t"], iteration)
        #     tb_writer.add_scalar("Pose/All - RPE_r", results["all"]["RPE_r"], iteration)
        #     tb_writer.add_scalar("Pose/All - ATE", results["all"]["ATE"], iteration )
            
        #     #wide
        #     tb_writer.add_scalar("Pose/Wide - RPE_t", results["wide"]["RPE_t"], iteration)
        #     tb_writer.add_scalar("Pose/Wide - RPE_r", results["wide"]["RPE_r"], iteration)
        #     tb_writer.add_scalar("Pose/Wide - ATE", results["wide"]["ATE"], iteration)
        #     #uw
        #     tb_writer.add_scalar("Pose/UW - RPE_t", results["uw"]["RPE_t"], iteration)
        #     tb_writer.add_scalar("Pose/UW - RPE_r", results["uw"]["RPE_r"], iteration)
        #     tb_writer.add_scalar("Pose/UW - ATE", results["uw"]["ATE"], iteration)

        #     #opacity & histogram
        #     tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        #     tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

        torch.cuda.empty_cache()
        #render_set_optimize(model_path, name, iteration, views, gaussians, pipeline, background)
        

if __name__ == "__main__":
    # Set up command line argument parser
    print("pretraining")
    parser = ArgumentParser(description="Training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)

    parser.add_argument("--test_iterations", nargs="+", type=int, default=[2, 500, 800, 1_000, 2_000, 3_000, 4_000, 5_000, 7_000, 9_000, 10_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[5_000, 10_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[10_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--stage", type=str, default = None)
    parser.add_argument("--with_identity_loss", action="store_true", default=False)
    parser.add_argument("--identity_loss_weight", type=float, default = 1.0)
    parser.add_argument("--binocular_consistency", action="store_true", default=False)
    parser.add_argument("--shift_cam_start", type=int, default=1)
    parser.add_argument("--fsgs_loss", action="store_true", default=False)
    parser.add_argument("--test_fps", action="store_true")
    parser.add_argument("--optim_test_pose_iter", default=500, type=int)
    # parserdepth_weight
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print(args.test_iterations)
    os.makedirs(args.model_path, exist_ok=True)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # if not args.disable_viewer:
    #     network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args)

    # All done
    print("\nTraining complete.")
