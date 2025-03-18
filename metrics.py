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

# Standard imports
from pathlib import Path
import os
import json
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from argparse import ArgumentParser

# Metrics and image processing
import torchvision.transforms.functional as tf
from lpipsPyTorch import lpips
from utils.loss_utils import ssim
from utils.image_utils import psnr

# Pose and SfM utilities
from utils.utils_poses.align_traj import align_ate_c2b_use_a2b
from utils.utils_poses.comp_ate import compute_rpe, compute_ATE
from utils.utils_poses.vis_pose_utils import plot_pose, draw_poses_pair
from utils.sfm_utils import readImages, align_pose, read_colmap_gt_pose
from scene.colmap_loader import qvec2rotmat,read_intrinsics_binary, read_extrinsics_binary
from utils.pose_utils import get_camera_from_tensor

from pytorch3d.transforms import so3_relative_angle

def revaluate_pose_optimized(initialized_pose, gt_train_pose, pose_optimized, pose_path):
    #先對齊
    #先把gt_pose轉成有序的
    if list(gt_train_pose.keys())[0].endswith(".jpg"):
        gt_train_pose_mod = {name.replace(".jpg",""): pose for name, pose in gt_train_pose.items()}
   
    print("align poses for gt and initialized poses...")
    sorted_names = sorted(gt_train_pose_mod.keys())
    ref_poses = {name: gt_train_pose_mod[name] for name in sorted_names}
    optimized_poses = {name: pose_optimized[name] for name in sorted_names}
    pose_optimized_copy = optimized_poses
    init_pose_only = {name: initialized_pose[name] for name in sorted_names}
    # ref_poses = {name: pose for name, pose in gt_train_pose.items()}
    # optimized_poses ={name: pose for name, pose in pose_optimized_copy.items()}
    # pose_init = {name: pose for name, pose in initialized_pose.items()}
    #算scale
    look_for_scale = True
    print("calculating scale...")
    #這邊scale是指init_pose 對齊gt_pose的scale
    poses_gt_init, pose_init_aligned, scale = convert_poses_for_evaluation(ref_poses, init_pose_only, look_for_scale)
    print("scale:", scale)
    #再把scale乘回去pose_optimized=再跟gt_pose對齊
    for name, pose in pose_optimized_copy.items():
        #乘在translation上
        # breakpoint()
        #是4X4矩陣 => 記得乘在UW 的translation上
        if name.startswith("uw"):
            pose_optimized_copy[name][:3,3] /= scale
    
    print("align poses for gt and optimized poses...")
    #跟pose_GT 對齊
    pose_optimized_copy = {name: pose_optimized_copy[name] for name in sorted_names}
    poses_gt_opt, pose_optimized_aligned, _ =  convert_poses_for_evaluation(ref_poses, pose_optimized_copy, False)
    
    #開始畫圖
    #再sort一次
    
    poses_gt_opt = {name: poses_gt_opt[name] for name in sorted_names}
    pose_optimized_aligned = {name: pose_optimized_aligned[name] for name in sorted_names}
    
    #先宣告值
    with_GT = True
    with_Mast3R =True
    with_only_one_cam = False
    print("plotting poses for gt and optimized poses...")
    plot_pose(poses_gt＿opt, pose_optimized_aligned, pose_path, with_GT, with_Mast3R, with_only_one_cam) 
    print("computing metrics...")
    #再sort一次
    poses_gt_np =[poses_gt_opt[name] for name in sorted_names]
    pose_optimized_aligned_np = [pose_optimized_aligned[name] for name in sorted_names]
    
    
    ate = compute_ATE(poses_gt_np, pose_optimized_aligned_np)
    rpe_trans, rpe_rot = compute_rpe(poses_gt_np, pose_optimized_aligned_np)
    
    print(f"camera RPE")
    print(f"  RPE_t: {rpe_trans*100:>12.7f}")
    print(f"  RPE_r: {rpe_rot * 180 / np.pi:>12.7f}")
    print(f"  ATE  : {ate:>12.7f}")
    print("")
    
    


def evaluate_pose(gt_train_pose, pose_optimized, pose_path, name_list, encoding, rel_or_not):
    
    # print("\n=== Before Conversion ===")
    # print("GT train pose:", gt_train_pose)
    # print("Pose optimized:", pose_optimized)
    
    img_name = name_list[0]
    #breakpoint()
    gt_name = list(gt_train_pose.keys())[0]
    with_GT = True
    if gt_name.endswith(".jpg"):
        #帶表示有GT
        ref_poses = {name.replace(".jpg",""): pose for name, pose in gt_train_pose.items() if name.replace(".jpg","") in name_list}
    else:
        #with_GT = False
        ref_poses = {name: pose for name, pose in gt_train_pose.items() if name in name_list}
    optimized_poses ={name: pose for name, pose in pose_optimized.items() if name in name_list}
    #print("align poses for gt and optimized poses...")
    if rel_or_not:
        print("\n=== Using relative evaluation ===")
        poses_gt, pose_optimized_aligned, scale = convert_poses_for_evaluation_rel(ref_poses, optimized_poses)
    else:
        poses_gt, pose_optimized_aligned, scale = convert_poses_for_evaluation(ref_poses, optimized_poses, False)
    #breakpoint()
    with_colmap = True
    # if rel_or_not:
    #     with_Mast3R = False
    if encoding == "optimized":
        with_colmap = False
    # if scale != 0.0:
    #     with_GT = True
    #     if hasattr(scale, 'numpy'):
    #         scale_np = scale.numpy()
    #     else:
    #         scale_np = scale
    #     os.makedirs(pose_path, exist_ok=True)
    #     print("====================saving scale====================")
    #     np.save(os.path.join(pose_path, 'pose_scale.npy'), scale_np)
    
    # breakpoint()
    
    with_only_one_cam = True
    if_wide = all(key.startswith('w') and not key.startswith('u') for key in pose_optimized_aligned.keys())
    if_uw = all(key.startswith('u') and not key.startswith('w') for key in pose_optimized_aligned.keys())
    
    if not if_uw and not if_wide:
        with_only_one_cam = False
    plot_pose(poses_gt, pose_optimized_aligned, pose_path, with_GT, with_colmap, with_only_one_cam)  
    print("computing metrics...")
    poses_gt_np = [poses_gt[name] for name in name_list]
    pose_optimized_aligned_np = [pose_optimized_aligned[name] for name in name_list]
    
    #for name in name_list:
    #   poses_gt_np.append(poses_gt[name])
    #   pose_optimized_aligned_np.append(pose_optimized_aligned[name])
    # print("Number of GT poses:", len(poses_gt_np))
    # print("Number of optimized poses:", len(pose_optimized_aligned_np))
    # print("GT poses:", poses_gt_np)
    # print("Optimized poses:", pose_optimized_aligned_np)
    ate = compute_ATE(poses_gt_np, pose_optimized_aligned_np)
    rpe_trans, rpe_rot = compute_rpe(poses_gt_np, pose_optimized_aligned_np)
    
    if img_name.startswith("w") and not img_name.startswith("uw"):
        print(f" Wide camera RPE")
    else:
        print(f" Ultrawide camera RPE")
    # print(f"  RPE_t: {rpe_trans*100:>12.7f}")
    # print(f"  RPE_r: {rpe_rot * 180 / np.pi:>12.7f}")
    # print(f"  ATE  : {ate:>12.7f}")
    # print("")
    # full_dict[scene_dir][method].update({"RPE_t": rpe_trans*100,
    #                                     "RPE_r": rpe_rot * 180 / np.pi,
    #                                     "ATE": ate})
    if encoding == "optimized":
        if img_name.startswith("w") and not img_name.startswith("uw"):
            print("111111111111111111")
            with open(pose_path / f"pose_eval_optimized_and_GT_wide.txt", 'w') as f:
                f.write("RPE_t: {:.04f}, RPE_r: {:.04f}, ATE: {:.04f}".format(
                    rpe_trans*100,
                    rpe_rot * 180 / np.pi,
                    ate))
        else:
            print("2222222222222222222")
            with open(pose_path / f"pose_eval_optimized_and_GT_uw.txt", 'w') as f:
                f.write("RPE_t: {:.04f}, RPE_r: {:.04f}, ATE: {:.04f}".format(
                    rpe_trans*100,
                    rpe_rot * 180 / np.pi,
                    ate))
    elif encoding == "initialized":
        if img_name.startswith("w") and not img_name.startswith("uw"):
            print("333333333333333333")
            with open(pose_path / f"pose_eval_initalized_and_GT_wide.txt", 'w') as f:
                f.write("RPE_t: {:.04f}, RPE_r: {:.04f}, ATE: {:.04f}".format(
                    rpe_trans*100,
                    rpe_rot * 180 / np.pi,
                    ate))
        else:
            print("444444444444444444")
            with open(pose_path / f"pose_eval_initalized_and_GT_uw.txt", 'w') as f:
                f.write("RPE_t: {:.04f}, RPE_r: {:.04f}, ATE: {:.04f}".format(
                    rpe_trans*100,
                    rpe_rot * 180 / np.pi,
                    ate))
                
    # 檢查對齊後的結果
    # breakpoint()
    # print("\n=== After Alignment ===")
    # print("Poses GT:", poses_gt)
    # print("Pose optimized aligned:", pose_optimized_aligned)
    
def evaluate_pose_together(gt_train_pose, pose_optimized, pose_path, encoding):
    #img_name = name_list[0]
    gt_name = list(gt_train_pose.keys())[0]
    ref_poses = {name: pose for name, pose in gt_train_pose.items()}
    optimized_poses ={name: pose for name, pose in pose_optimized.items()}
    with_GT = True
    if gt_name.endswith(".jpg"):
        #帶表示有GT
        #with_GT = True
        ref_poses = {name.replace(".jpg",""): pose for name, pose in gt_train_pose.items()}
    else:
        with_GT = False
        ref_poses = {name: pose for name, pose in gt_train_pose.items()}
      
    with_Mast3R = False  
    if encoding == "optimized":
        with_Mast3R = False
    elif encoding == "initialized":
        with_Mast3R = True
        
    print("align poses for gt and optimized poses...")
    
    poses_gt, pose_optimized_aligned, scale = convert_poses_for_evaluation(ref_poses, optimized_poses, False)
    with_only_one_cam = False
    plot_pose(poses_gt, pose_optimized_aligned, pose_path, with_GT, with_Mast3R, with_only_one_cam)
    print("computing metrics...")
    poses_gt_np =[pose for pose in poses_gt.values()]
    pose_optimized_aligned_np = [pose for pose in pose_optimized_aligned.values()]
    ate = compute_ATE(poses_gt_np, pose_optimized_aligned_np)
    rpe_trans, rpe_rot = compute_rpe(poses_gt_np, pose_optimized_aligned_np)
    

    print(f"  RPE_t: {rpe_trans*100:>12.7f}")
    print(f"  RPE_r: {rpe_rot * 180 / np.pi:>12.7f}")
    print(f"  ATE  : {ate:>12.7f}")
    print("")
    # full_dict[scene_dir][method].update({"RPE_t": rpe_trans*100,
    #                                     "RPE_r": rpe_rot * 180 / np.pi,
    #                                     "ATE": ate})
    if encoding == "optimized":
        with open(pose_path / f"pose_eval_optimized_and_GT_all.txt", 'w') as f:
            f.write("RPE_t: {:.04f}, RPE_r: {:.04f}, ATE: {:.04f}".format(
                rpe_trans*100,
                rpe_rot * 180 / np.pi,
                ate))
    elif encoding == "initialized":
        with open(pose_path / f"pose_eval_initalized_and_GT_all.txt", 'w') as f:
            f.write("RPE_t: {:.04f}, RPE_r: {:.04f}, ATE: {:.04f}".format(
                rpe_trans*100,
                rpe_rot * 180 / np.pi,
                ate))

# 首先，修改pose轉換和對齊的部分
def convert_poses_for_evaluation(train_poses_dict, optimized_poses, look_for_scale):
    # 將ground truth poses轉換為有序列表
    # GT_train名字有.jpg
    # optimized 沒有 .jpg
    #breakpoint()
    train_poses_mod = {name.replace('.jpg', ''): pose for name, pose in train_poses_dict.items()}
    sorted_names = sorted(train_poses_mod.keys())
    gt_poses_list = [train_poses_mod[name] for name in sorted_names]
    optimized_poses = [optimized_poses[name] for name in sorted_names]
    #breakpoint()
    
    # 轉換為tensor
    poses_gt = torch.from_numpy(np.stack(gt_poses_list, axis=0)).float()
    pose_optimized = torch.from_numpy(np.stack(optimized_poses, axis=0)).float()
    
    # 對齊
    #breakpoint()
    scale = 0.0
    trans_gt_align, trans_est_align, _, scale = align_pose(poses_gt[:, :3, -1].numpy(), 
                                                pose_optimized[:, :3, -1].numpy())
    poses_gt[:, :3, -1] = torch.from_numpy(trans_gt_align)
    pose_optimized[:, :3, -1] = torch.from_numpy(trans_est_align)
    #breakpoint()
    # 進行ATE對齊
    c2ws_est_aligned = align_ate_c2b_use_a2b(pose_optimized, poses_gt)
    
    # 將對齊後的optimized poses轉換為字典形式
    aligned_poses_dict = {name: c2ws_est_aligned[i].cpu().numpy() 
                         for i, name in enumerate(sorted_names)}
    
    pose_gt_dict = {name: poses_gt[i].cpu().numpy() 
                         for i, name in enumerate(sorted_names)}
    return pose_gt_dict, aligned_poses_dict, scale

def convert_poses_for_evaluation_rel(train_poses_dict, optimized_poses):
    # 將ground truth poses轉換為有序列表
    # GT_train名字有.jpg
    # optimized 沒有 .jpg
    #breakpoint()
    train_poses_mod = {name.replace('.jpg', ''): pose for name, pose in train_poses_dict.items()}
    sorted_names = sorted(train_poses_mod.keys())
    gt_poses_list = [train_poses_mod[name] for name in sorted_names]
    optimized_poses = [optimized_poses[name] for name in sorted_names]
    #breakpoint()
    
    # 轉換為tensor
    poses_gt = torch.from_numpy(np.stack(gt_poses_list, axis=0)).float()
    pose_optimized = torch.from_numpy(np.stack(optimized_poses, axis=0)).float()
    
    # 對齊
    scale = 0.0
    trans_gt_align, trans_est_align, _, scale = align_pose(poses_gt[:, :3, -1].numpy(), 
                                                pose_optimized[:, :3, -1].numpy())
    poses_gt[:, :3, -1] = torch.from_numpy(trans_gt_align)
    pose_optimized[:, :3, -1] = torch.from_numpy(trans_est_align)

    # 進行ATE對齊
    #c2ws_est_aligned = align_ate_c2b_use_a2b(pose_optimized, poses_gt)
    
    # 將對齊後的optimized poses轉換為字典形式
    aligned_poses_dict = {name: pose_optimized[i].cpu().numpy() 
                         for i, name in enumerate(sorted_names)}
    
    pose_gt_dict = {name: poses_gt[i].cpu().numpy() 
                         for i, name in enumerate(sorted_names)}
    # aligned_poses_dict = {name: c2ws_est_aligned[i].cpu().numpy() 
    #                      for i, name in enumerate(sorted_names)}
    
    # pose_gt_dict = {name: poses_gt[i].cpu().numpy() 
    #                      for i, name in enumerate(sorted_names)}
    #scale = 0.0
    return pose_gt_dict, aligned_poses_dict, scale


def evaluate(args):

    full_dict = {}
    per_view_dict = {}
    print("")

    for scene_dir in args.model_paths:
        
        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}

        test_dir = Path(scene_dir) / "test"

        for method in os.listdir(test_dir):
            print("Method:", method)

            full_dict[scene_dir][method] = {}
            per_view_dict[scene_dir][method] = {}

            # ------------------------------ (1) image evaluation ------------------------------ #
            method_dir = test_dir / method
            out_f = open(method_dir / 'metrics.txt', 'w') 
            gt_dir = method_dir/ "gt"
            renders_dir = method_dir / "renders"
            renders, gts, image_names = readImages(renders_dir, gt_dir)

            ssims = []
            psnrs = []
            lpipss = []

            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                s=ssim(renders[idx], gts[idx])
                p=psnr(renders[idx], gts[idx])
                l=lpips(renders[idx], gts[idx], net_type='vgg')
                out_f.write(f"image name{image_names[idx]}, image idx: {idx}, PSNR: {p.item():.2f}, SSIM: {s:.4f}, LPIPS: {l.item():.4f}\n")
                ssims.append(s)
                psnrs.append(p)
                lpipss.append(l)

            print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
            print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
            print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))

            full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                    "PSNR": torch.tensor(psnrs).mean().item(),
                                                    "LPIPS": torch.tensor(lpipss).mean().item()})
            per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                        "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                        "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})
            
            # ------------------------------ (2) pose evaluation ------------------------------ #
            # load GT Colmap poses
            train_img_names = ['w_001','w_008','w_015','uw_001','uw_008','uw_015']
            train_img_names_wide = ['w_001','w_008','w_015']
            train_img_names_uw = ['uw_001','uw_008','uw_015']
            pose_dir = Path(scene_dir) / "pose"
            pose_path = pose_dir / method
            #breakpoint()
            print("reading optimized poses...")
            #breakpoint()
            loaded_poses = np.load(os.path.join(pose_path, 'pose_optimized.npz'))
            pose_optimized = {name: loaded_poses[name] for name in loaded_poses.files}
            pose_optimized_wide = {name: pose for name, pose in pose_optimized.items() if name in train_img_names_wide}
            pose_optimized_uw = {name: pose for name, pose in pose_optimized.items() if name in train_img_names_uw}
            #breakpoint()
            #load initial poses
            org_pose_path = os.path.join(args.source_path, "sparse_6/0", "images.bin")
            org_cams = read_extrinsics_binary(org_pose_path)
            
            pose_org_dict={}
            for cam in org_cams.values():
                if cam.name.replace(".jpg","") in train_img_names:
                    idx_name = cam.name.replace(".jpg","")
                    #get_camera_from_tensor
                    qvec = cam.qvec
                    R = torch.tensor(qvec2rotmat(qvec))
                    T = torch.tensor(cam.tvec)
                    pose = torch.eye(4)
                    pose[:3,:3] = R
                    pose[:3,3] = T
                    pose_org_dict[idx_name] = pose
            pose_org_dict_wide={}
            pose_org_dict_uw={}
            for cam in org_cams.values():
                if cam.name.replace(".jpg","") in train_img_names_wide:
                    idx_name = cam.name.replace(".jpg","")
                    qvec = cam.qvec
                    R = torch.tensor(qvec2rotmat(qvec))
                    T = torch.tensor(cam.tvec)
                    pose = torch.eye(4)
                    pose[:3,:3] = R
                    pose[:3,3] = T
                    pose_org_dict_wide[idx_name] = pose 
                elif cam.name.replace(".jpg","") in train_img_names_uw:
                    idx_name = cam.name.replace(".jpg","")
                    qvec = cam.qvec
                    R = torch.tensor(qvec2rotmat(qvec))
                    T = torch.tensor(cam.tvec)
                    pose = torch.eye(4)
                    pose[:3,:3] = R
                    pose[:3,3] = T
                    pose_org_dict_uw[idx_name] = pose
                    
            print("reading GT Pose...")
            pose_colmap = read_colmap_gt_pose(args.source_path)
            gt_train_pose = {name: pose for name, pose in pose_colmap.items() if name.replace(".jpg", "") in train_img_names}
            #idx有.jpg
            
            gt_train_pose_wide = {name: pose for name, pose in pose_colmap.items() if name.replace(".jpg", "") in train_img_names_wide}
            gt_train_pose_uw = {name: pose for name, pose in pose_colmap.items() if name.replace(".jpg", "") in train_img_names_uw}
            
            #breakpoint()
            Rel_or_not = False
            evaluate_pose(gt_train_pose_wide, pose_optimized_wide, pose_path, train_img_names_wide, "optimized", Rel_or_not)
            evaluate_pose(gt_train_pose_uw, pose_optimized_uw, pose_path, train_img_names_uw, "optimized", Rel_or_not)
            #breakpoint()
            evaluate_pose(gt_train_pose_wide, pose_org_dict_wide, pose_path, train_img_names_wide, "initialized", Rel_or_not)
            evaluate_pose(gt_train_pose_uw, pose_org_dict_uw, pose_path, train_img_names_uw, "initialized", Rel_or_not)
    
            #
            evaluate_pose_together(gt_train_pose, pose_optimized, pose_path, "optimized")
            evaluate_pose_together(gt_train_pose, pose_org_dict, pose_path, "initialized")

            #revaluate_pose_optimized(pose_org_dict, gt_train_pose, pose_optimized, pose_path)
            
        with open(scene_dir + "/results.json", 'w') as fp:
            json.dump(full_dict[scene_dir], fp, indent=True)
        with open(scene_dir + "/per_view.json", 'w') as fp:
            json.dump(per_view_dict[scene_dir], fp, indent=True)

        # except:
        #     print("Unable to compute metrics for model", scene_dir)

def evaluate_pose_new_mod(gt_train_pose, pose_optimized, pose_path, name_list):
    gt_name = list(gt_train_pose.keys())[0]
    
    # 处理输入数据
    if gt_name.endswith(".jpg"):
        optimized_poses = {name.replace(".jpg",""): pose for name, pose in pose_optimized.items() if name.replace(".jpg","") in name_list}
        ref_poses = {name.replace(".jpg",""): pose for name, pose in gt_train_pose.items() if name.replace(".jpg","") in name_list}
    else:
        optimized_poses = {name: pose for name, pose in pose_optimized.items() if name in name_list}
        ref_poses = {name: pose for name, pose in gt_train_pose.items() if name in name_list}
    
    # 分别获取UW和Wide的name_list
    uw_names = [name for name in name_list if name.startswith('u')]
    wide_names = [name for name in name_list if name.startswith('w')]
    
    # 对所有poses进行对齐
    poses_gt = torch.from_numpy(np.stack([ref_poses[name] for name in name_list]))
    pose_optimized = torch.from_numpy(np.stack([optimized_poses[name] for name in name_list]))
    
    # 对齐操作
    trans_gt_align, trans_est_align, *_ = align_pose(poses_gt[:, :3, -1].numpy(), 
                                                    pose_optimized[:, :3, -1].numpy())
    poses_gt[:, :3, -1] = torch.from_numpy(trans_gt_align)
    pose_optimized[:, :3, -1] = torch.from_numpy(trans_est_align)
    c2ws_est_aligned = align_ate_c2b_use_a2b(pose_optimized, poses_gt)
    
    # # 将对齐后的poses转回字典形式用于绘图
    aligned_gt_dict = {name: poses_gt[i].float() for i, name in enumerate(name_list)}
    aligned_est_dict = {name: c2ws_est_aligned[i].float() for i, name in enumerate(name_list)}
    
    # # 使用对齐后的poses进行绘图
    draw_poses_pair(aligned_gt_dict, aligned_est_dict, pose_path)
    
    # 计算整体指标
    ate_all = compute_ATE(poses_gt.cpu().numpy(), c2ws_est_aligned.cpu().numpy())
    rpe_trans_all, rpe_rot_all = compute_rpe(poses_gt.cpu().numpy(), c2ws_est_aligned.cpu().numpy())
    
    # 分别计算UW和Wide相机的指标
    if uw_names:
        poses_gt_uw = poses_gt[[i for i, name in enumerate(name_list) if name in uw_names]]
        poses_opt_uw = c2ws_est_aligned[[i for i, name in enumerate(name_list) if name in uw_names]]
        ate_uw = compute_ATE(poses_gt_uw.cpu().numpy(), poses_opt_uw.cpu().numpy())
        rpe_trans_uw, rpe_rot_uw = compute_rpe(poses_gt_uw.cpu().numpy(), poses_opt_uw.cpu().numpy())
        
        txt_path = os.path.join(pose_path, "pose_eval_optimized_uw.txt")
        with open(txt_path, 'w') as f:
            f.write(f"RPE_t: {rpe_trans_uw*100:.4f}, RPE_r: {rpe_rot_uw * 180 / np.pi:.4f}, ATE: {ate_uw:.4f}")
    
    if wide_names:
        poses_gt_wide = poses_gt[[i for i, name in enumerate(name_list) if name in wide_names]]
        poses_opt_wide = c2ws_est_aligned[[i for i, name in enumerate(name_list) if name in wide_names]]
        ate_wide = compute_ATE(poses_gt_wide.cpu().numpy(), poses_opt_wide.cpu().numpy())
        rpe_trans_wide, rpe_rot_wide = compute_rpe(poses_gt_wide.cpu().numpy(), poses_opt_wide.cpu().numpy())
        
        txt_path = os.path.join(pose_path, "pose_eval_optimized_wide.txt")
        with open(txt_path, 'w') as f:
            f.write(f"RPE_t: {rpe_trans_wide*100:.4f}, RPE_r: {rpe_rot_wide * 180 / np.pi:.4f}, ATE: {ate_wide:.4f}")
    
    # 保存总体结果
    txt_path = os.path.join(pose_path, "pose_eval_optimized_all.txt")
    with open(txt_path, 'w') as f:
        f.write(f"RPE_t: {rpe_trans_all*100:.4f}, RPE_r: {rpe_rot_all * 180 / np.pi:.4f}, ATE: {ate_all:.4f}")
    
    # 打印结果
    results = {
        "all": {
            "RPE_t": rpe_trans_all*100,
            "RPE_r": rpe_rot_all * 180 / np.pi,
            "ATE": ate_all
        },
        "wide": {
            "RPE_t": rpe_trans_wide*100,
            "RPE_r": rpe_rot_wide * 180 / np.pi,
            "ATE": ate_wide
        },
        "uw": {
            "RPE_t": rpe_trans_uw*100,
            "RPE_r": rpe_rot_uw * 180 / np.pi,
            "ATE": ate_uw
        }
    }
        
    print(f"All  => RPE_t: {rpe_trans_all*100:.4f}, RPE_r: {rpe_rot_all * 180 / np.pi:.4f}, ATE: {ate_all:.4f}")
    if wide_names:
        print(f"Wide => RPE_t: {rpe_trans_wide*100:.4f}, RPE_r: {rpe_rot_wide * 180 / np.pi:.4f}, ATE: {ate_wide:.4f}")
    if uw_names:
        print(f"UW   => RPE_t: {rpe_trans_uw*100:.4f}, RPE_r: {rpe_rot_uw * 180 / np.pi:.4f}, ATE: {ate_uw:.4f}")
    
    return results   

def convert_dict_to_tensor(poses_dict):
    converted_dict = {}
    for name, pose in poses_dict.items():
        if isinstance(pose, np.ndarray):
            converted_dict[name] = torch.from_numpy(pose).float()
        else:
            converted_dict[name] = pose.float()  # 如果已經是tensor就保持不變
    return converted_dict

def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e6):
    """Normalize the translation vectors and compute the angle between them."""
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t

def rotation_angle(rot_gt, rot_pred, batch_size=None):
    # rot_gt, rot_pred (B, 3, 3)
    rel_angle_cos = so3_relative_angle(rot_gt, rot_pred, eps=1e-4)
    rel_rangle_deg = rel_angle_cos * 180 / np.pi

def translation_angle(tvec_gt, tvec_pred, batch_size=None):
    # tvec_gt, tvec_pred (B, 3,)
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi

    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

    return rel_tangle_deg

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--source_path', '-s', required=True, type=str, default=None)
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument("--n_views", default=None, type=int)
    parser.add_argument("--fsgs_loss", action="store_true", default=False)
    args = parser.parse_args()
    evaluate(args)