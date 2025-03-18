import os
import matplotlib
import matplotlib.pyplot as plt
import copy
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.main_ape import ape
from evo.tools import plot
from evo.core import sync
from evo.tools import file_interface
from evo.core import metrics
import evo
import torch
import numpy as np
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
import scipy.interpolate as si


def interp_poses(c2ws, N_views):
    N_inputs = c2ws.shape[0]
    trans = c2ws[:, :3, 3:].permute(2, 1, 0)
    rots = c2ws[:, :3, :3]
    render_poses = []
    rots = R.from_matrix(rots)
    slerp = Slerp(np.linspace(0, 1, N_inputs), rots)
    interp_rots = torch.tensor(
        slerp(np.linspace(0, 1, N_views)).as_matrix().astype(np.float32))
    interp_trans = torch.nn.functional.interpolate(
        trans, size=N_views, mode='linear').permute(2, 1, 0)
    render_poses = torch.cat([interp_rots, interp_trans], dim=2)
    render_poses = convert3x4_4x4(render_poses)
    return render_poses


def interp_poses_bspline(c2ws, N_novel_imgs, input_times, degree):
    target_trans = torch.tensor(scipy_bspline(
        c2ws[:, :3, 3], n=N_novel_imgs, degree=degree, periodic=False).astype(np.float32)).unsqueeze(2)
    rots = R.from_matrix(c2ws[:, :3, :3])
    slerp = Slerp(input_times, rots)
    target_times = np.linspace(input_times[0], input_times[-1], N_novel_imgs)
    target_rots = torch.tensor(
        slerp(target_times).as_matrix().astype(np.float32))
    target_poses = torch.cat([target_rots, target_trans], dim=2)
    target_poses = convert3x4_4x4(target_poses)
    return target_poses


def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:, 4:5]

    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        # c = np.dot(c2w[:3,:4], np.array([0.7*np.cos(theta) , -0.3*np.sin(theta) , -np.sin(theta*zrate) *0.1, 1.]) * rads)
        # c = np.dot(c2w[:3,:4], np.array([0.3*np.cos(theta) , -0.3*np.sin(theta) , -np.sin(theta*zrate) *0.01, 1.]) * rads)
        c = np.dot(c2w[:3, :4], np.array(
            [0.2*np.cos(theta), -0.2*np.sin(theta), -np.sin(theta*zrate) * 0.1, 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses


def scipy_bspline(cv, n=100, degree=3, periodic=False):
    """ Calculate n samples on a bspline

        cv :      Array ov control vertices
        n  :      Number of samples to return
        degree:   Curve degree
        periodic: True - Curve is closed
    """
    cv = np.asarray(cv)
    count = cv.shape[0]

    # Closed curve
    if periodic:
        kv = np.arange(-degree, count+degree+1)
        factor, fraction = divmod(count+degree+1, count)
        cv = np.roll(np.concatenate(
            (cv,) * factor + (cv[:fraction],)), -1, axis=0)
        degree = np.clip(degree, 1, degree)

    # Opened curve
    else:
        degree = np.clip(degree, 1, count-1)
        kv = np.clip(np.arange(count+degree+1)-degree, 0, count-degree)

    # Return samples
    max_param = count - (degree * (1-periodic))
    spl = si.BSpline(kv, cv, degree)
    return spl(np.linspace(0, max_param, n))


def generate_spiral_nerf(learned_poses, bds, N_novel_views, hwf):
    learned_poses_ = np.concatenate((learned_poses[:, :3, :4].detach(
    ).cpu().numpy(), hwf[:len(learned_poses)]), axis=-1)
    c2w = poses_avg(learned_poses_)
    print('recentered', c2w.shape)
    # Get spiral
    # Get average pose
    up = normalize(learned_poses_[:, :3, 1].sum(0))
    # Find a reasonable "focus depth" for this dataset

    close_depth, inf_depth = bds.min()*.9, bds.max()*5.
    dt = .75
    mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
    focal = mean_dz

    # Get radii for spiral path
    shrink_factor = .8
    zdelta = close_depth * .2
    tt = learned_poses_[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
    rads = np.percentile(np.abs(tt), 90, 0)
    c2w_path = c2w
    N_rots = 2
    c2ws = render_path_spiral(
        c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_novel_views)
    c2ws = torch.tensor(np.stack(c2ws).astype(np.float32))
    c2ws = c2ws[:, :3, :4]
    c2ws = convert3x4_4x4(c2ws)
    return c2ws


def convert3x4_4x4(input):
    """
    :param input:  (N, 3, 4) or (3, 4) torch or np
    :return:       (N, 4, 4) or (4, 4) torch or np
    """
    if torch.is_tensor(input):
        if len(input.shape) == 3:
            output = torch.cat([input, torch.zeros_like(
                input[:, 0:1])], dim=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = torch.cat([input, torch.tensor(
                [[0, 0, 0, 1]], dtype=input.dtype, device=input.device)], dim=0)  # (4, 4)
    else:
        if len(input.shape) == 3:
            output = np.concatenate(
                [input, np.zeros_like(input[:, 0:1])], axis=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = np.concatenate(
                [input, np.array([[0, 0, 0, 1]], dtype=input.dtype)], axis=0)  # (4, 4)
            output[3, 3] = 1.0
    return output


plt.rc('legend', fontsize=20)  # using a named size


def plot_pose_ori(ref_poses, est_poses, output_path, args, vid=False):
    ref_poses = [pose for pose in ref_poses]
    if isinstance(est_poses, dict):
        est_poses = [pose for k, pose in est_poses.items()]
    else:
        est_poses = [pose for pose in est_poses]
    traj_ref = PosePath3D(poses_se3=ref_poses)
    traj_est = PosePath3D(poses_se3=est_poses)
    traj_est_aligned = copy.deepcopy(traj_est)
    traj_est_aligned.align(traj_ref, correct_scale=True,
                           correct_only_scale=False)
    if vid:
        for p_idx in range(len(ref_poses)):
            fig = plt.figure()
            current_est_aligned = traj_est_aligned.poses_se3[:p_idx+1]
            current_ref = traj_ref.poses_se3[:p_idx+1]
            current_est_aligned = PosePath3D(poses_se3=current_est_aligned)
            current_ref = PosePath3D(poses_se3=current_ref)
            traj_by_label = {
                # "estimate (not aligned)": traj_est,
                "Ours (aligned)": current_est_aligned,
                "Ground-truth": current_ref
            }
            plot_mode = plot.PlotMode.xyz
            # ax = plot.prepare_axis(fig, plot_mode, 111)
            ax = fig.add_subplot(111, projection="3d")
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)
            ax.zaxis.set_tick_params(labelleft=False)
            colors = ['r', 'b']
            styles = ['-', '--']

            for idx, (label, traj) in enumerate(traj_by_label.items()):
                plot.traj(ax, plot_mode, traj,
                          styles[idx], colors[idx], label)
                # break
            # plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
            ax.view_init(elev=10., azim=45)
            plt.tight_layout()
            os.makedirs(os.path.join(os.path.dirname(
                output_path), 'pose_vid'), exist_ok=True)
            pose_vis_path = os.path.join(os.path.dirname(
                output_path), 'pose_vid', 'pose_vis_{:03d}.png'.format(p_idx))
            print(pose_vis_path)
            fig.savefig(pose_vis_path)


    fig = plt.figure()
    fig.patch.set_facecolor('white')                   # Set background to pure white
    traj_by_label = {
        # "estimate (not aligned)": traj_est,
        "Ours (aligned)": traj_est_aligned,
        "COLMAP (GT)": traj_ref
    }
    plot_mode = plot.PlotMode.xyz
    # ax = plot.prepare_axis(fig, plot_mode, 111)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor('white')                           # Set subplot to pure white
    ax.xaxis.set_tick_params(labelbottom=True)
    ax.yaxis.set_tick_params(labelleft=True)
    ax.zaxis.set_tick_params(labelleft=True)
    colors = ['#2c9e38', '#d12920']
    styles = ['s-', 's-.']

    for idx, (label, traj) in enumerate(traj_by_label.items()):
        plot.traj(ax, plot_mode, traj,
                  styles[idx], colors[idx], label)
    # plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=1)
    ax.view_init(elev=30., azim=45)
    plt.tight_layout()
    pose_vis_path = output_path / f'pose_vis.png'
    fig.savefig(pose_vis_path , transparent=False)

def plot_pose(ref_poses_dict, est_poses_dict, output_path, with_GT, with_Mast3R, with_only_one_cam):
    sorted_names = sorted(ref_poses_dict.keys())
    ref_poses = [ref_poses_dict[name] for name in sorted_names]
    est_poses = [est_poses_dict[name] for name in sorted_names]
    
    traj_ref = PosePath3D(poses_se3=ref_poses)
    traj_est = PosePath3D(poses_se3=est_poses)
    
    fig = plt.figure(figsize=(15, 12))
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor('white')
    
    colors = ['#2c9e38', '#d12920']  # GT和estimated的顏色
    markers = ['o', 's']  # GT和estimated的標記
    marker_size = 100
    
    positions_ref = traj_ref.positions_xyz
    positions_est = traj_est.positions_xyz
    
    # 畫GT點
    if with_GT:
        if with_Mast3R:
            print("estimation with Mast3R")
            print("compare with colmap GT")
            ax.scatter(positions_ref[:, 0], positions_ref[:, 1], positions_ref[:, 2],
                color=colors[0], marker=markers[0], label="COLMAP (GT)", s=marker_size)
            ax.scatter(positions_est[:, 0], positions_est[:, 1], positions_est[:, 2],
              color=colors[1], marker=markers[1], label="initialized (Mast3R)", s=marker_size)
        else:
            print("estimation with Ours")
            print("compare with colmap GT")
            ax.scatter(positions_ref[:, 0], positions_ref[:, 1], positions_ref[:, 2],
                color=colors[0], marker=markers[0], label="COLMAP (GT)", s=marker_size)
            ax.scatter(positions_est[:, 0], positions_est[:, 1], positions_est[:, 2],
              color=colors[1], marker=markers[1], label="Ours", s=marker_size)
    else:
        if with_Mast3R:
            print("compare with initialized pose")
            ax.scatter(positions_ref[:, 0], positions_ref[:, 1], positions_ref[:, 2],
                color=colors[0], marker=markers[0], label="initialized (Mast3R)", s=marker_size)
            ax.scatter(positions_est[:, 0], positions_est[:, 1], positions_est[:, 2],
              color=colors[1], marker=markers[1], label="Ours", s=marker_size)
            
        else:
            print("might be something wrong")
        
        
        # 設置更小的軸範圍來凸顯差異
        margin = 0.2  # 給點額外空間
        max_range = max(
            positions_ref.max() - positions_ref.min(),
            positions_est.max() - positions_est.min()
        )
        mid_x = (positions_ref[:, 0].mean() + positions_est[:, 0].mean()) / 2
        mid_y = (positions_ref[:, 1].mean() + positions_est[:, 1].mean()) / 2
        mid_z = (positions_ref[:, 2].mean() + positions_est[:, 2].mean()) / 2
        
        ax.set_xlim(mid_x - max_range/2 - margin, mid_x + max_range/2 + margin)
        ax.set_ylim(mid_y - max_range/2 - margin, mid_y + max_range/2 + margin)
        ax.set_zlim(mid_z - max_range/2 - margin, mid_z + max_range/2 + margin)
        
        # 更細的網格
        tick_spacing = max_range / 10  # 10個主要刻度
        ax.set_xticks(np.arange(ax.get_xlim()[0], ax.get_xlim()[1], tick_spacing))
        ax.set_yticks(np.arange(ax.get_ylim()[0], ax.get_ylim()[1], tick_spacing))
        ax.set_zticks(np.arange(ax.get_zlim()[0], ax.get_zlim()[1], tick_spacing))
        
        ax.grid(True, linestyle='--', alpha=0.3)
        
    # 畫estimated點
    
    
    # 連接對應點並添加標籤
    for i, name in enumerate(sorted_names):
        # 連接相應的點
        ax.plot([positions_ref[i, 0], positions_est[i, 0]],
                [positions_ref[i, 1], positions_est[i, 1]],
                [positions_ref[i, 2], positions_est[i, 2]],
                'k--', alpha=0.5, linewidth=2)
        
        # 在GT點附近標記完整名稱
        ax.text(positions_ref[i, 0], positions_ref[i, 1], positions_ref[i, 2], 
               name, fontsize=20, fontweight='bold')
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
    # 設置軸標籤字體大小
    ax.set_xlabel('X', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y', fontsize=14, fontweight='bold')
    ax.set_zlabel('Z', fontsize=14, fontweight='bold')
    
    ax.view_init(elev=30., azim=45)
    plt.tight_layout()
    
    if with_only_one_cam:
        if with_GT:
            if with_Mast3R:
                pose_vis_path = output_path / f'pose_vis_init_one_cam.png'
            else:
                #
                pose_vis_path = output_path / f'pose_vis_trained_pose_one_cam.png'
        else:
            if with_Mast3R:
                pose_vis_path = output_path / f'pose_vis_with_init_vs_trained_pose_one_cam.png'
            else:
                print("might be something wrong")
    else:
        if with_GT:
            if with_Mast3R:
                pose_vis_path = output_path / f'pose_vis_init_together.png'
            else:
                #
                pose_vis_path = output_path / f'pose_vis_trained_pose_together.png'
        else:
            if with_Mast3R:
                pose_vis_path = output_path / f'pose_vis_with_init_vs_trained_pose_together.png'
            else:
                print("might be something wrong")
        
    #fig.savefig(pose_vis_path, transparent=False)
    fig.savefig(pose_vis_path, transparent=False, dpi=300)
    print("Finishing plotting poses!")
    
def get_camera_mesh(pose, depth=1):
    vertices = (
        torch.tensor(
            [[-0.5, -0.5, 1], [0.5, -0.5, 1], [0.5, 0.5, 1], [-0.5, 0.5, 1], [0, 0, 0]]
        )
        * depth
    )
    faces = torch.tensor(
        [[0, 1, 2], [0, 2, 3], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]
    )
    # vertices = cam2world(vertices[None],pose)
    vertices = vertices @ pose[:, :3, :3].transpose(-1, -2)
    vertices += pose[:, None, :3, 3]
    wireframe = vertices[:, [0, 1, 2, 3, 0, 4, 1, 2, 4, 3]]
    return vertices, faces, wireframe
    
def merge_wireframes(wireframe):
    wireframe_merged = [[], [], []]
    for w in wireframe:
        wireframe_merged[0] += [float(n) for n in w[:, 0]]
        wireframe_merged[1] += [float(n) for n in w[:, 1]]
        wireframe_merged[2] += [float(n) for n in w[:, 2]]
    return wireframe_merged
    
def draw_poses_pair(poses_dict1, poses_dict2, output_path, title1="Ground Truth", title2="Predicted"):
    """
    畫出兩組相機軌跡
    
    Args:
        poses_dict1: 第一組poses字典
        poses_dict2: 第二組poses字典
        output_path: 輸出路徑
        title1: 第一組軌跡的標題
        title2: 第二組軌跡的標題
    """
    
    # 處理兩組數據
    sorted_names1 = sorted(poses_dict1.keys())
    sorted_names2 = sorted(poses_dict2.keys())
    
    poses1 = torch.stack([poses_dict1[name] for name in sorted_names1])
    #breakpoint()
    poses2 = torch.stack([poses_dict2[name] for name in sorted_names2])
    
    # 定義兩組軌跡的顏色方案
    colors = {
        'traj1': {
            'uw': 'navy',        # 深藍色用於第一組UW相機
            'w': 'royalblue',    # 淺藍色用於第一組W相機
            'pair': 'dodgerblue' # 用於第一組相機對的連線
        },
        'traj2': {
            'uw': 'darkred',     # 深紅色用於第二組UW相機
            'w': 'crimson',      # 淺紅色用於第二組W相機
            'pair': 'lightcoral' # 用於第二組相機對的連線
        }
    }
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(projection='3d')
    
    # 轉換矩陣
    R_convert = torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]], 
                            device=poses1.device, 
                            dtype=poses1.dtype)
    
    # 處理兩組poses
    centered_poses1 = poses1.clone()
    centered_poses1[:, :3, :3] = centered_poses1[:, :3, :3] @ R_convert
    
    centered_poses2 = poses2.clone()
    centered_poses2[:, :3, :3] = centered_poses2[:, :3, :3] @ R_convert
    
    # 獲取相機mesh
    vertices1, faces1, wireframe1 = get_camera_mesh(centered_poses1, 0.5)
    vertices2, faces2, wireframe2 = get_camera_mesh(centered_poses2, 0.5)
    
    # 找到所有頂點的範圍
    all_vertices = torch.cat([vertices1, vertices2], dim=0)
    max_val = all_vertices.abs().max().item()
    margin = max_val * 0.1
    limit = max_val + margin
    
    # 設置顯示範圍
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_zlim(-4, 4)
    
    # 合併wireframe
    wireframe1_merged = merge_wireframes(wireframe1)
    wireframe2_merged = merge_wireframes(wireframe2)
    
    # 獲取相機中心點
    center1 = vertices1[:, -1]
    center2 = vertices2[:, -1]
    
    # 畫出第一組相機
    drawn_types1 = set()  # 追踪已經畫過的相機類型
    for c in range(center1.shape[0]):
        cam_type = sorted_names1[c].split('_')[0]
        is_uw = cam_type.startswith('u')
        color = colors['traj1']['uw'] if is_uw else colors['traj1']['w']
        
        # 只為每種類型的相機添加一次標籤
        should_label = cam_type not in drawn_types1
        label = f"{title1} - {'UW' if is_uw else 'W'} Camera" if should_label else None
        
        ax.plot(
            wireframe1_merged[0][c * 10 : (c + 1) * 10],
            wireframe1_merged[1][c * 10 : (c + 1) * 10],
            wireframe1_merged[2][c * 10 : (c + 1) * 10],
            color=color,
            linewidth=1.5,
            label=label
        )
        drawn_types1.add(cam_type)
    
    # 畫出第二組相機
    drawn_types2 = set()  # 追踪已經畫過的相機類型
    for c in range(center2.shape[0]):
        cam_type = sorted_names2[c].split('_')[0]
        is_uw = cam_type.startswith('u')
        color = colors['traj2']['uw'] if is_uw else colors['traj2']['w']
        
        # 只為每種類型的相機添加一次標籤
        should_label = cam_type not in drawn_types2
        label = f"{title2} - {'UW' if is_uw else 'W'} Camera" if should_label else None
        
        ax.plot(
            wireframe2_merged[0][c * 10 : (c + 1) * 10],
            wireframe2_merged[1][c * 10 : (c + 1) * 10],
            wireframe2_merged[2][c * 10 : (c + 1) * 10],
            color=color,
            linewidth=1.5,
            label=label
        )
        drawn_types2.add(cam_type)
    
    # 處理相機對連線
    def process_pairs(names, centers, title, line_color):
        pairs = {}
        for name in names:
            base_name = name.split('_')[1].split('.')[0]
            if base_name not in pairs:
                pairs[base_name] = []
            pairs[base_name].append(name)
        
        # 只為第一對添加標籤
        first_pair = True
        for base_name, pair in pairs.items():
            if len(pair) == 2:
                idx1 = names.index(pair[0])
                idx2 = names.index(pair[1])
                
                pos1 = centers[idx1]
                pos2 = centers[idx2]
                
                label = f'{title} - Camera Pair' if first_pair else None
                
                ax.plot([pos1[0], pos2[0]], 
                       [pos1[1], pos2[1]], 
                       [pos1[2], pos2[2]], 
                       color=line_color,
                       linewidth=2,
                       linestyle='-' if title == title1 else '--',
                       label=label)
                first_pair = False
    
    # 處理兩組軌跡的相機對
    process_pairs(sorted_names1, center1, title1, colors['traj1']['pair'])
    process_pairs(sorted_names2, center2, title2, colors['traj2']['pair'])
    
    # 設置圖例
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.view_init(elev=30, azim=45)
    
    # 添加軸標籤
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.tight_layout()
    
    # 根據相機類型設置輸出路徑
    only_wide = all(key.startswith('w') and not key.startswith('u') for key in poses_dict1.keys())
    only_uw = all(key.startswith('u') and not key.startswith('w') for key in poses_dict1.keys())
    
    if not only_wide and not only_uw:
        pose_vis_path = os.path.join(output_path, 'camera_poses_vis_two_trajectories.png')
    else:
        if only_wide:
            pose_vis_path = os.path.join(output_path, 'camera_poses_vis_two_trajectories_wide.png')
        elif only_uw:
            pose_vis_path = os.path.join(output_path, 'camera_poses_vis_two_trajectories_uw.png')
            
    os.makedirs(os.path.dirname(pose_vis_path), exist_ok=True)
    fig.savefig(pose_vis_path, transparent=False, dpi=300, bbox_inches='tight')
    plt.close(fig)