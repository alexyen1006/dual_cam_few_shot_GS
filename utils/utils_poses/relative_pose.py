import torch
import os
import numpy as np
import roma
from utils.sfm_utils import read_colmap_gt_pose
from utils.utils_poses.vis_pose_utils import draw_poses_pair 
from utils.utils_poses.lie_group_helper import SO3_to_quat, quat_to_SO3, quat_to_SO3_tensor
from utils.pose_utils import compute_relative_world_to_camera, get_image_number, quad2rotation, rotation2quad, compute_relative_world_to_camera_mod
from dust3r.utils.geometry import inv
from dust3r.utils.device import to_cpu, to_numpy
from scene.colmap_loader import qvec2rotmat, read_intrinsics_binary, read_extrinsics_binary
from metrics import evaluate_pose_new_mod
from scipy.spatial.transform import Rotation as Rot_scipy
from dust3r.cloud_opt.commons import signed_log1p

def load_one_pose(scene_path, image_id):
    """載入單個相機的pose"""

    images = read_extrinsics_binary(os.path.join(scene_path, "sparse/0/images.bin"))
    image_name_uw = f"uw_{image_id:03d}.jpg"
    image_name_w = f"w_{image_id:03d}.jpg"
    for image_id, image in images.items():
        if image.name == image_name_uw:
            image_uw = image
            R_uw_init = qvec2rotmat(image_uw.qvec)
            R_uw = torch.tensor(R_uw_init, dtype=torch.float32, device="cuda")
            t_uw_init = image_uw.tvec
            T_uw = torch.tensor(t_uw_init, dtype=torch.float32, device="cuda")
        if image.name == image_name_w:
            image_w = image
            R_w_init = qvec2rotmat(image_w.qvec)
            R_w = torch.tensor(R_w_init, dtype=torch.float32, device="cuda")
            t_w_init = image_w.tvec
            T_w = torch.tensor(t_w_init, dtype=torch.float32, device="cuda")
    
    E_rel = compute_relative_world_to_camera(
        R_uw, T_uw,
        R_w, T_w
    )
    E_rel = E_rel.detach().cpu().numpy()
    return E_rel
  
def load_and_process_scene(scene_path):
    """載入單個場景的相機參數並計算relative poses"""
    # 讀取相機參數
    cameras = read_intrinsics_binary(os.path.join(scene_path, "sparse/0/cameras.bin"))
    images = read_extrinsics_binary(os.path.join(scene_path, "sparse/0/images.bin"))
    
    # 建立編號到pose的映射
    uw_poses = {}  # key: image number, value: pose
    w_poses = {}   # key: image number, value: pose
    
    for image_id, image in images.items():
        R = qvec2rotmat(image.qvec)
        t = image.tvec
        
        pose = {
            'R': torch.tensor(R, dtype=torch.float32, device="cuda"),
            't': torch.tensor(t, dtype=torch.float32, device="cuda"),
            'camera_id': image.camera_id,
            'name': image.name
        }
        
        image_num = get_image_number(image.name)
        if image_num is None:
            continue
            
        if 'uw_' in image.name:
            uw_poses[image_num] = pose
        elif 'w_' in image.name:
            w_poses[image_num] = pose
        #breakpoint()
        #relative_pose_path = os.path.join(scene_path, "relative_poses")
    #save_relative_translations_to_txt(uw_poses, w_poses, relative_pose_path)
    return compute_relative_poses(uw_poses, w_poses)


def normalize_rotation_matrix(rotation):
    """確保旋轉矩陣是正交的"""
    U, _, Vt = np.linalg.svd(rotation)
    return U @ Vt

def normalize_quaternion(quat):
    """確保四元數是單位長度"""
    return quat / np.linalg.norm(quat)

def save_relative_translations_to_txt(uw_poses, w_poses, output_path):
    """
    保存15對W/UW相機的relative translations和平均值到txt文件
    """
    translations = []  # 用來儲存所有translation以計算平均值
    
    with open(output_path, 'w') as f:
        f.write("# Relative translations between Wide and Ultra-wide cameras\n")
        f.write("# Format: camera_id tx ty tz\n")
        f.write("#----------------------------------------------------------\n")
        
        common_numbers = set(uw_poses.keys()) & set(w_poses.keys())
        
        for num in sorted(common_numbers):
            uw_pose = uw_poses[num]
            w_pose = w_poses[num]
            
            E_rel = compute_relative_world_to_camera(
                uw_pose['R'], uw_pose['t'],
                w_pose['R'], w_pose['t']
            )
            
            t = E_rel[:3, 3]
            translations.append(t.cpu().numpy())  # 儲存每個translation
            
            # 寫入相機ID和平移向量
            f.write(f"{num:02d} {t[0].item():.6f} {t[1].item():.6f} {t[2].item():.6f}\n")
        
        # 計算並寫入平均值
        #breakpoint()
        avg_translation = np.mean(np.stack(translations), axis=0)
        f.write("#----------------------------------------------------------\n")
        f.write(f"AVG {avg_translation[0]:.6f} {avg_translation[1]:.6f} {avg_translation[2]:.6f}\n")
            
    print(f"Relative translations and average have been saved to {output_path}")


def split_poses(gt_train_pose):
    """
    將gt_train_pose分成uw_poses和w_poses
    每個pose包含R和t
    """
    uw_poses = {}
    w_poses = {}
    
    for name, pose in gt_train_pose.items():
        # 轉換成tensor
        pose_tensor = torch.tensor(pose, dtype=torch.float32).cuda()
        
        # 提取R和t
        R = pose_tensor[:3, :3]
        t = pose_tensor[:3, 3]
        
        # 根據檔名分類
        if name.startswith('uw_'):
            uw_poses[name] = {'R': R, 't': t}
        elif name.startswith('w_'):
            w_poses[name] = {'R': R, 't': t}
            
    return uw_poses, w_poses

def combine_poses(uw_poses, w_poses):
    # 初始化空字典
    #breakpoint()
    combined_dict = {}
    device = next(iter(uw_poses.values()))['R'].device
    last_row = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)
    
    # 先找出所有共同的數字
    uw_numbers = set(name.split('_')[1].split('.')[0] for name in uw_poses.keys())
    w_numbers = set(name.split('_')[1].split('.')[0] for name in w_poses.keys())
    common_numbers = uw_numbers & w_numbers  # 這樣會得到 {'001', '008', '015'}
    #breakpoint()
    # 對每個共同的數字處理對應的poses
    
    for num in sorted(common_numbers):
        uw_name = f'uw_{num}.jpg'
        w_name = f'w_{num}.jpg'
        
        # 處理UW pose
        uw_pose = uw_poses[uw_name]
        R = uw_pose['R']
        t = uw_pose['t']
        if len(t.shape) == 1:
            t = t.unsqueeze(1)
        upper_mat = torch.cat([R, t], dim=1)
        transform_mat = torch.cat([upper_mat, last_row], dim=0)
        combined_dict[uw_name] = transform_mat.detach().cpu().numpy()
        
        # 處理W pose
        #breakpoint()
        w_pose = w_poses[w_name]
        R = w_pose['R']
        t = w_pose['t']
        if len(t.shape) == 1:
            t = t.unsqueeze(1)
        upper_mat = torch.cat([R, t], dim=1)
        transform_mat = torch.cat([upper_mat, last_row], dim=0)
        combined_dict[w_name] = transform_mat.detach().cpu().numpy()
    #breakpoint()
    return combined_dict

def compute_one_wide_pose_W2C(uw_pose, E_uw_to_wide):
    #breakpoint()
    device = uw_pose['R'].device
    zero_row = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32, device=device)
    E_uw = torch.cat([uw_pose['R'], uw_pose['t'].reshape(-1, 1)], dim=1)
    E_uw = torch.cat([E_uw, zero_row], dim=0)
    # 應用相對變換
    #breakpoint()
    E_wide = E_uw_to_wide @ E_uw
    #E_wide =  relative_transform @ E_uw
    #E_wide = E_uw @ torch.inverse(relative_transform)
    # 提取旋轉和平移
    #breakpoint()
    wide_pose = {
        'R': E_wide[:3, :3],
        't': E_wide[:3, 3]
        #'t': -E_wide[:3, :3].T @ E_wide[:3, 3]
    }
    
    R = wide_pose['R'].transpose(0,1)
    t = wide_pose['t']
    #breakpoint()
    #Q = rotation2quad(R)
    Q_wide = roma.rotmat_to_unitquat(R)
    T_wide_log = signed_log1p(t)
    
    QT_wide = torch.cat([Q_wide, T_wide_log], dim=0)
    
    return wide_pose, QT_wide

def verify_poses(E_rel_all, averaged_pose, uw_poses, gt_wide_poses):
    """比較各種方法得到的 wide poses 和 GT 的差異"""
    
    results = {}
    for uw_name in uw_poses.keys():
        w_name = f'w_{uw_name.split("_")[1].split(".")[0]}.jpg'
        if w_name not in gt_wide_poses:
            continue
            
        uw_pose = uw_poses[uw_name]
        gt_pose = gt_wide_poses[w_name]
        
        # 1. 用每個單獨的 E_rel 計算
        individual_poses = []
        for E_rel in E_rel_all:
            wide_pose, _ = compute_one_wide_pose_W2C(uw_pose, E_rel)
            individual_poses.append(wide_pose)
            
        # 2. 用平均後的 E_rel 計算
        avg_wide_pose, _ = compute_one_wide_pose_W2C(uw_pose, averaged_pose)
        
        # 計算與 GT 的差異
        errors = {
            'individual_R_errors': [torch.norm(pose['R'] - gt_pose['R']).item() for pose in individual_poses],
            'individual_t_errors': [torch.norm(pose['t'] - gt_pose['t']).item() for pose in individual_poses],
            'avg_R_error': torch.norm(avg_wide_pose['R'] - gt_pose['R']).item(),
            'avg_t_error': torch.norm(avg_wide_pose['t'] - gt_pose['t']).item()
        }
        
        results[w_name] = errors
        
    return results

def compute_relative_poses(uw_poses, w_poses, viewpoint_stack_wide, tol=1e-6):
    """計算對應編號的相機對之間的relative poses"""
    rotations = []
    translations = []
    
    common_numbers = []
    # breakpoint()
    for uw_name in uw_poses.keys():
        # 從uw_name中提取數字 (例如從'uw_008.jpg'提取'008')
        uw_num = uw_name.split('_')[1].split('.')[0]
        
        # 檢查對應的w檔案是否存在
        w_name = f'w_{uw_num}.jpg'
        if w_name in w_poses:
            common_numbers.append((uw_name, w_name))
    
    # 只處理在兩個字典中都存在的編號
    #common_numbers = set(uw_poses.keys()) & set(w_poses.keys())
    E_rel_all = []
    #breakpoint()
    for uw_name, w_name in sorted(common_numbers):
        uw_pose = uw_poses[uw_name]
        w_pose = w_poses[w_name]
        
        # 計算relative transformation
        #breakpoint()
        E_rel = compute_relative_world_to_camera_mod(
            uw_pose['R'], uw_pose['t'],
            w_pose['R'], w_pose['t']
        )
        
        #測試
        # for cam in viewpoint_stack_wide:
        #     mod_omg_name = cam.image_name + '.jpg'
        #     if mod_omg_name == w_name:
        #breakpoint()
        w_pose_rel, QT_wide = compute_one_wide_pose_W2C(uw_pose, E_rel)

        #breakpoint()
        # assert torch.allclose(w_pose_rel['R'], w_pose['R'], atol=tol), \
        #     f"Rotation matrices are too different: {w_pose_rel['R']} vs {w_pose['R']}"
        # assert torch.allclose(w_pose_rel['t'], w_pose['t'], atol=tol), \
        #     f"Translation vectors are too different: {w_pose_rel['t']} vs {w_pose['t']}"
        
        
        E_rel_all.append(E_rel)

        R = E_rel[:3, :3]
        if not isinstance(R, torch.Tensor):
            R = torch.tensor(R, dtype=torch.float32).cuda()
            

        Q_rel = rotation2quad(R)

        SO3_rel = quat_to_SO3(Q_rel.cpu().numpy())
        rotations.append(SO3_rel)

        t = E_rel[:3, 3]
        translations.append(t.cpu().numpy())
    
    
    stacked_rotation = np.stack(rotations)
    avg_rotation_SO3 = Rot_scipy.from_matrix(stacked_rotation).mean().as_matrix()
    norm_avg_rotation_SO3 = normalize_rotation_matrix(avg_rotation_SO3)
    
    avg_rotation_quat = SO3_to_quat(norm_avg_rotation_SO3)
    avg_rotation_quat = normalize_quaternion(avg_rotation_quat)  
    #breakpoint()
    avg_rotation_quat_mod= torch.from_numpy(avg_rotation_quat).unsqueeze(0)
    avg_rotation_R = quad2rotation(avg_rotation_quat_mod).squeeze(0)
    avg_rotation_R_npy = avg_rotation_R.cpu().numpy()
    # 平均平移向量
    
    avg_translation_npy = np.mean(np.stack(translations), axis=0)
    pose = np.block([[avg_rotation_R_npy, avg_translation_npy.reshape(3, 1)], [np.zeros((1, 3)), 1]])
    #avg_transform_final = getWorld2View2(avg_rotation_R_npy, avg_translation_npy).T
    #breakpoint()
    return pose, E_rel_all

def process_all_scenes(dataset_path):
    """處理整個數據集"""
    all_relative_poses = []
    all_baselines = []
    scene_stats = []
    
    # 遍歷所有場景
    scene_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    for scene_dir in scene_dirs:
        scene_path = os.path.join(dataset_path, scene_dir)
        relative_poses = load_and_process_scene(scene_path)
        
        # 計算該場景的統計數據
        scene_baselines = [pose['baseline'] for pose in relative_poses]
        scene_stats.append({
            'scene_name': scene_dir,
            'avg_baseline': np.mean(scene_baselines) if scene_baselines else 0,
            'std_baseline': np.std(scene_baselines) if scene_baselines else 0,
            'num_pairs': len(relative_poses)
        })
        
        all_relative_poses.extend(relative_poses)
        all_baselines.extend(scene_baselines)
    
    # 計算整體統計數據
    avg_baseline = np.mean(all_baselines) if all_baselines else 0
    std_baseline = np.std(all_baselines) if all_baselines else 0
    
    # 計算平均rotation和translation
    if all_relative_poses:
        avg_R = np.mean([pose['R'] for pose in all_relative_poses], axis=0)
        # 正交化平均旋轉矩陣
        U, _, Vh = np.linalg.svd(avg_R)
        avg_R = U @ Vh
        
        avg_t = np.mean([pose['t'] for pose in all_relative_poses], axis=0)
    else:
        avg_R = np.eye(3)
        avg_t = np.zeros(3)
    
    return {
        'average_baseline': avg_baseline,
        'std_baseline': std_baseline,
        'average_rotation': avg_R,
        'average_translation': avg_t,
        'scene_stats': scene_stats,
        'total_pairs': len(all_relative_poses),
        'total_scenes': len(scene_dirs)
    }



def convert_dict_to_tensor(poses_dict):
    converted_dict = {}
    for name, pose in poses_dict.items():
        if isinstance(pose, np.ndarray):
            converted_dict[name] = torch.from_numpy(pose).float()
        else:
            converted_dict[name] = pose.float()  # 如果已經是tensor就保持不變
    return converted_dict

def pose_evaluation(scene,source_path, model_path, extrinsics_w2c):
    
    pose_colmap = read_colmap_gt_pose(source_path)#args.source_path
    train_img_names = ['uw_001', 'uw_008', 'uw_015',
                         'w_001', 'w_008', 'w_015']#
    #breakpoint()
    gt_train_pose = {name: pose for name, pose in pose_colmap.items() if name.replace(".jpg", "") in train_img_names}
    gt_train_pose_tensor = convert_dict_to_tensor(gt_train_pose)
    
    pose_init_dict={}
    for index, img_names_set in scene.view_mapping.items():
        for img_name in img_names_set:
            pose_init_dict[img_name] = extrinsics_w2c[index]
    
    opt_base_path = os.path.join(model_path, "dust3R_pose_eval")
    os.makedirs(opt_base_path, exist_ok=True)
    pose_init_dict_tensor = convert_dict_to_tensor(pose_init_dict)
    print("======================================")
    breakpoint()
    #draw_poses_pair(gt_train_pose_tensor, pose_init_dict_tensor, opt_base_path) 
    results = evaluate_pose_new_mod(gt_train_pose_tensor, pose_init_dict_tensor, opt_base_path, train_img_names)
    
    return results

def verify_formula(E_rel_uw_to_wide, extrinsics_w2c, UW_W_pair):
    wide_pose_dict = {}
    for (uw_idx,wide_idx), (uw_name, wide_name) in UW_W_pair.items():
        assert uw_name.split("_")[1] == wide_name.split("_")[1]
        uw_pose = extrinsics_w2c[uw_idx]
        R_uw = uw_pose[:3, :3]
        T_uw = uw_pose[:3, 3]
        R_uw_tensor = torch.tensor(R_uw, dtype=torch.float32).cuda()
        T_uw_tensor = torch.tensor(T_uw, dtype=torch.float32).cuda()
        uw_pose_dict = {'R' : R_uw_tensor, 't' : T_uw_tensor}
        #breakpoint()
        wide_pose, _ = compute_one_wide_pose_W2C(uw_pose_dict, E_rel_uw_to_wide)
        
        wide_pose_dict[wide_name] = wide_pose
        #print("wide img name:", wide_name)
        E_wide_w2c = extrinsics_w2c[wide_idx]
        actual_R_w2c = E_wide_w2c[:3, :3]
        actual_T_w2c = E_wide_w2c[:3, 3]
        #breakpoint()
        actual_R_w2c_tensor = torch.tensor(actual_R_w2c, dtype=torch.float32).to(wide_pose['R'].device)
        actual_T_w2c_tensor = torch.tensor(actual_T_w2c, dtype=torch.float32).to(wide_pose['t'].device)
        
        #print("extrinsics_w2c:", E_wide_w2c)
        assert torch.allclose(wide_pose['R'], actual_R_w2c_tensor, rtol=1e-4, atol=1e-5)
        assert torch.allclose(wide_pose['t'], actual_T_w2c_tensor, rtol=1e-4, atol=1e-5)
        #R_W2C = extrinsics_w2c[wide_idx][:3, :3]
        #assert wide_pose['R'] - R_W2C
        #print("wide_pose:", wide_pose)
        E_uw_w2c = extrinsics_w2c[uw_idx]
        actual_E_rel_uw_to_wide = E_wide_w2c @ inv(E_uw_w2c)
        #print("actual_E_rel_uw_to_wide:", actual_E_rel_uw_to_wide )
    return wide_pose_dict
    # breakpoint()