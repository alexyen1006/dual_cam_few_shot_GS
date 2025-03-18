import os
import argparse
import torch
import numpy as np
from pathlib import Path
from time import time

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from icecream import ic
ic(torch.cuda.is_available())  # Check if CUDA is available
ic(torch.cuda.device_count())

from mast3r.model import AsymmetricMASt3R
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.utils.device import to_numpy
from dust3r.utils.geometry import inv
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from utils.sfm_utils import (save_intrinsics, save_extrinsic, save_points3D, save_time, save_images_and_masks,
                             init_filestructure, get_sorted_image_files, split_train_test, load_images, compute_co_vis_masks)
from utils.camera_utils import generate_interpolated_path
from utils.utils_poses.relative_pose import load_and_process_scene, load_one_pose, verify_formula
from scene.colmap_loader import read_intrinsics_binary, read_extrinsics_binary
from scene.dataset_readers import readColmapCameras
from utils.pose_utils import get_camera_from_tensor, get_tensor_from_camera
from utils.sfm_utils import read_colmap_gt_pose
from utils.utils_poses.relative_pose import pose_evaluation, compute_one_wide_pose_W2C
import matplotlib.pyplot as plt
import os


def main(source_path, model_path, ckpt_path, device, batch_size, image_size, schedule, lr, niter, 
         min_conf_thr, llffhold, n_views, co_vis_dsp, depth_thre, conf_aware_ranking=False, focal_avg=False, infer_video=False):

    # ---------------- (1) Load model and images ----------------  
    save_path, sparse_0_path, sparse_1_path = init_filestructure(Path(source_path), n_views)
    model = AsymmetricMASt3R.from_pretrained(ckpt_path).to(device)
    image_dir = Path(source_path) / 'images'
    image_files, image_suffix = get_sorted_image_files(image_dir)
    cameras_intrinsic_file = os.path.join(source_path, "sparse/0", "cameras.bin")
    known_focal_uw = None
    known_focal_wide = None
    
    #load known focals if needed
    if args.with_GT_focal:
        intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        uw_indices = [1, 8, 15]  
        wide_indices = [16, 23, 30]  

        uw_intrinsics = [intrinsics[i] for i in intrinsics.keys() if i in uw_indices]
        wide_intrinsics = [intrinsics[i] for i in intrinsics.keys() if i in wide_indices]

        known_focal_uw = np.mean([intrinsic.params[0] for intrinsic in uw_intrinsics])
        known_focal_wide = np.mean([intrinsic.params[0] for intrinsic in wide_intrinsics])
        
    
    if infer_video:
        train_img_files = image_files
    else:
        train_img_files, test_img_files = split_train_test(image_files, llffhold, n_views, verbose=True)
    
    # when geometry init, only use train images
    image_files = train_img_files
    images, org_imgs_shape = load_images(image_files, size=image_size)

    start_time = time()
    print(f'>> Making pairs...')
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    print(f'>> Inference...')
    output = inference(pairs, model, device, batch_size=1, verbose=True)
    print(f'>> Global alignment...')
    scene = global_aligner(output, device=args.device, mode=GlobalAlignerMode.PointCloudOptimizer)


    loss = scene.compute_global_alignment(init="mst", niter=300, schedule=schedule, lr=0.01, focal_avg=args.focal_avg, known_focal_uw=known_focal_uw, known_focal_wide=known_focal_wide, view_mapping = scene.view_mapping, output_path = sparse_1_path)
    extrinsics_c2w = scene.get_im_poses()

    extrinsics_w2c_tensor = inv(extrinsics_c2w)
    #check_rel_poses_w2c = scene.verify_rel_poses_W2C(extrinsics_w2c_tensor)

    extrinsics_w2c = to_numpy(extrinsics_w2c_tensor)
    results = pose_evaluation(scene, source_path, model_path, extrinsics_w2c)

    intrinsics = to_numpy(scene.get_intrinsics())
    focals = to_numpy(scene.get_focals())

    imgs = np.array(scene.imgs)
    pts3d = to_numpy(scene.get_pts3d(-1, sparse_1_path))
    pts3d = np.array(pts3d)
    depthmaps = to_numpy(scene.im_depthmaps.detach().cpu().numpy())
    values = [param.detach().cpu().numpy() for param in scene.im_conf]
    confs = np.array(values)
    
    if conf_aware_ranking:
        print(f'>> Confiden-aware Ranking...')
        avg_conf_scores = confs.mean(axis=(1, 2))
        sorted_conf_indices = np.argsort(avg_conf_scores)[::-1]
        sorted_conf_avg_conf_scores = avg_conf_scores[sorted_conf_indices]
        print("Sorted indices:", sorted_conf_indices)
        print("Sorted average confidence scores:", sorted_conf_avg_conf_scores)
    else:
        sorted_conf_indices = np.arange(n_views)
        print("Sorted indices:", sorted_conf_indices)

    # Calculate the co-visibility mask
    print(f'>> Calculate the co-visibility mask...')
    if depth_thre > 0:
        overlapping_masks = compute_co_vis_masks(sorted_conf_indices, depthmaps, pts3d, intrinsics, extrinsics_w2c, imgs.shape, depth_threshold=depth_thre)
        overlapping_masks = ~overlapping_masks
    else:
        co_vis_dsp = False
        overlapping_masks = None
    end_time = time()
    Train_Time = end_time - start_time
    print(f"Time taken for {n_views} views: {Train_Time} seconds")
    save_time(model_path, '[1] coarse_init_TrainTime', Train_Time)

    # ---------------- (2) Interpolate training pose to get initial testing pose ----------------
    if not infer_video:
        #breakpoint()
        n_train = len(train_img_files)
        n_test = len(test_img_files)
        #只取wide
        
        # # 1. 首先找出所有Wide相機的索引
        if n_train < n_test:
            wide_indices = []
            for i in range(n_train):
                if i in scene.nodes_wide:
                    wide_indices.append(i)
            
            # 2. 只選取Wide相機的poses
            wide_poses = extrinsics_w2c[wide_indices]
            n_wide = len(wide_indices)
            assert n_train//2 == n_wide
            
            # 3. 計算需要的interpolation數量
            n_interp = (n_test // (n_wide-1)) + 1
            
            # 4. 在Wide相機之間做interpolation
            all_inter_pose = []
            for i in range(n_wide-1):
                tmp_inter_pose = generate_interpolated_path(
                    poses=wide_poses[i:i+2], 
                    n_interp=n_interp
                )
                all_inter_pose.append(tmp_inter_pose)
            
            # 5. 組合所有interpolated poses
            all_inter_pose = np.concatenate(all_inter_pose, axis=0)
            all_inter_pose = np.concatenate([
                all_inter_pose, 
                wide_poses[-1][:3, :].reshape(1, 3, 4)
            ], axis=0)
            # 6. 均勻採樣得到需要數量的poses
            indices = np.linspace(0, all_inter_pose.shape[0] - 1, n_test, dtype=int)
            sampled_poses = all_inter_pose[indices]
            sampled_poses = np.array(sampled_poses).reshape(-1, 3, 4)
            assert sampled_poses.shape[0] == n_test
            inter_pose_list = []
            for p in sampled_poses:
                tmp_view = np.eye(4)
                tmp_view[:3, :3] = p[:3, :3]
                tmp_view[:3, 3] = p[:3, 3]
                inter_pose_list.append(tmp_view)
            pose_test_init = np.stack(inter_pose_list, 0)
        else:
            indices = np.linspace(0, extrinsics_w2c.shape[0] - 1, n_test, dtype=int)
            pose_test_init = extrinsics_w2c[indices]

        save_extrinsic(sparse_1_path, pose_test_init, test_img_files, image_suffix)
        #抓wide focal 取平均
        focal_wide = []
        for i in scene.view_mapping.keys():
            view_name = next(iter(scene.view_mapping[i]))
            if view_name.startswith('w') and not view_name.startswith('uw'):
                focal_wide.append(focals[i])
                
        focal_wide_avg = np.mean(focal_wide)
        test_focals = np.repeat(focal_wide_avg, n_test)
        save_intrinsics(sparse_1_path, test_focals, org_imgs_shape, imgs.shape, save_focals=False, with_GT_focal = args.with_GT_focal)
    # -----------------------------------------------------------------------------------------

    
    print(f'>> Saving results...')
    end_time = time()
    save_time(model_path, '[1] init_geo', end_time - start_time)
    
    
    focals = to_numpy(scene.get_focals())

    #breakpoint()
    save_extrinsic(sparse_0_path, extrinsics_w2c, image_files, image_suffix)
    save_intrinsics(sparse_0_path, focals, org_imgs_shape, imgs.shape, save_focals=True)

    rel_path = os.path.join(sparse_1_path ,"relative_pose_from_dust3R.npy")
    E_rel_uw_to_wide = scene._get_one_pose(scene.relative_pose.detach())
    QT_rel_test = get_tensor_from_camera(E_rel_uw_to_wide)

    wide_pose_dict = verify_formula(E_rel_uw_to_wide, extrinsics_w2c, scene.UW_W_pair)
 
    np.save(rel_path, QT_rel_test.cpu().numpy())
    breakpoint()
    
    pts_num = save_points3D(sparse_0_path, imgs, pts3d, confs.reshape(pts3d.shape[0], -1), overlapping_masks, use_masks=co_vis_dsp, save_all_pts=True, save_txt_path=model_path, depth_threshold=depth_thre)
    save_images_and_masks(sparse_0_path, n_views, imgs, overlapping_masks, image_files, image_suffix)
    print(f'[INFO] MASt3R Reconstruction is successfully converted to COLMAP files in: {str(sparse_0_path)}')
    print(f'[INFO] Number of points: {pts3d.reshape(-1, 3).shape[0]}')    
    print(f'[INFO] Number of points after downsampling: {pts_num}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images and save results.')
    parser.add_argument('--source_path', '-s', type=str, required=True, help='Directory containing images')
    parser.add_argument('--model_path', '-m', type=str, required=True, help='Directory to save the results')
    parser.add_argument('--ckpt_path', type=str,
        default='./mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth', help='Path to the model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for inference')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing images')
    parser.add_argument('--image_size', type=int, default=512, help='Size to resize images')
    parser.add_argument('--schedule', type=str, default='cosine', help='Learning rate schedule')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--niter', type=int, default=300, help='Number of iterations')
    parser.add_argument('--min_conf_thr', type=float, default=5, help='Minimum confidence threshold')
    parser.add_argument('--llffhold', type=int, default=8, help='')
    parser.add_argument('--n_views', type=int, default=3, help='')
    # parser.add_argument('--focal_avg', type=bool, default=False, help='')
    parser.add_argument('--with_GT_focal', action="store_true", default=False)
    parser.add_argument('--focal_avg', action="store_true")
    parser.add_argument('--conf_aware_ranking', action="store_true")
    parser.add_argument('--co_vis_dsp', action="store_true")
    parser.add_argument('--depth_thre', type=float, default=0.01, help='Depth threshold')
    parser.add_argument('--infer_video', action="store_true")
    
    args = parser.parse_args()
    main(args.source_path, args.model_path, args.ckpt_path, args.device, args.batch_size, args.image_size, args.schedule, args.lr, args.niter,         
          args.min_conf_thr, args.llffhold, args.n_views, args.co_vis_dsp, args.depth_thre, args.conf_aware_ranking, args.focal_avg, args.infer_video)
