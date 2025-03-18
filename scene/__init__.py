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
import random
import json
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel,load_pose
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import torch
from utils.pose_utils import generate_random_poses_360
from scene.cameras import Camera
from scene.cameras import PseudoCamera
from torch import load, save

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, opt=None, shuffle=True, resolution_scales=[1.0], get_all_cam=False, load_mlp=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        #Scene(dataset, gaussians, load_iteration=iteration, opt=args, shuffle=False)
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.pseudo_cameras_wide = {}
        self.pseudo_cameras_uw = {}
        if os.path.exists(os.path.join(args.source_path, f"sparse_{args.n_views}")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print(self.cameras_extent, 'cameras_extent')

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print('train_camera_num: ', len(self.train_cameras[resolution_scale]))
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            print('train_camera_num: ', len(self.test_cameras[resolution_scale]))
            pseudo_cams_wide = []
            pseudo_cams_uw = []
            # if args.source_path.find('llff'):
            #     pseudo_poses = generate_random_poses_llff(self.train_cameras[resolution_scale])
            # elif args.source_path.find('360'):
            # input_cam_list = self.train_cameras[resolution_scale]
            # input_cam_list = [cam for cam in input_cam_list if cam.image_name.startswith('w')]
            
            
            #if view.image_name.startswith('w') and not view.image_name.startswith('u'):
            #breakpoint()
            #Wide camera
            # if args.fsgs_loss:
            #     train_wide_cam = [cam for cam in self.train_cameras[resolution_scale] if cam.image_name.startswith('w') and not cam.image_name.startswith('u')]
            #     assert all(cam.image_name.startswith('w') for cam in train_wide_cam), "Found non-w image"
                
                
            #     pseudo_poses_wide = generate_random_poses_360(train_wide_cam)
            #     length = len(self.train_cameras[resolution_scale])
            #     self.train_cameras[resolution_scale] = sorted(self.train_cameras[resolution_scale].copy(), key = lambda x : x.image_name)
            #     #view = [cam for cam in self.train_cameras[resolution_scale] if cam.image_name.startsiwth('w') and not cam.image_name.startsiwth('u')]
                
            #     view = self.train_cameras[resolution_scale][length//2+1]
            #     assert view.image_name.startswith('w') and not view.image_name.startswith('u')
            #     for pose in pseudo_poses_wide:
            #         pseudo_cams_wide.append(PseudoCamera(
            #             R=pose[:3, :3].T, T=pose[:3, 3], FoVx=view.FoVx, FoVy=view.FoVy,
            #             width=view.image_width, height=view.image_height
            #         ))
            #     self.pseudo_cameras_wide[resolution_scale] = pseudo_cams_wide
            #     #UW camera
            #     train_uw_cam = [cam for cam in self.train_cameras[resolution_scale] if cam.image_name.startswith('u') and not cam.image_name.startswith('w')]
            #     assert all(cam.image_name.startswith('u') for cam in train_uw_cam), "Found non-w image"
            #     pseudo_poses_uw = generate_random_poses_360(train_uw_cam)
            #     length = len(self.train_cameras[resolution_scale])
            #     self.train_cameras[resolution_scale] = sorted(self.train_cameras[resolution_scale].copy(), key = lambda x : x.image_name)
            #     #view = [cam for cam in self.train_cameras[resolution_scale] if cam.image_name.startsiwth('w') and not cam.image_name.startsiwth('u')]
                
            #     view = self.train_cameras[resolution_scale][1]
            #     assert view.image_name.startswith('u') and not view.image_name.startswith('w')
            #     for pose in pseudo_poses_uw:
            #         pseudo_cams_uw.append(PseudoCamera(
            #             R=pose[:3, :3].T, T=pose[:3, 3], FoVx=view.FoVx, FoVy=view.FoVy,
            #             width=view.image_width, height=view.image_height
            #         ))
            #     self.pseudo_cameras_uw[resolution_scale] = pseudo_cams_uw
                
            
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            pose_path = os.path.join(self.model_path,"pose","ours_" + str(self.loaded_iter),"pose_optimized.npz")
            #breakpoint()
            # assert os.path.exists(pose_path)
            poses = load_pose(pose_path, num_poses = 3)
            #breakpoint()
            self.gaussians.P = poses.cuda().requires_grad_(True)
            # all_cam_scales=[]
            # for i in range(len(poses)):
            #     tmp_cam_scale = torch.rand(1)
            #     all_cam_scales.append(tmp_cam_scale)
            # all_cam_scales = torch.stack(all_cam_scales)
            # self.gaussians.cam_scale = all_cam_scales.cuda().requires_grad_(True)
            

        else:
            #breakpoint()
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            if not args.eval:
                self.gaussians.init_RT_seq(self.train_cameras)
        # if args.stage == 'uw2wide':
        #     poses = load_pose(os.path.join(self.model_path,"pose","ours_1" ,"pose_optimized.npz"), num_poses=3)
        #     self.gaussians.P = poses.cuda().requires_grad_(True)
            # self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            #poses = load_pose(os.path.join(self.model_path,"pose","ours_" + str(self.loaded_iter),"pose_optimized.npz"), num_poses=3)
            # self.gaussians.P = poses.cuda().requires_grad_(True)
        
        
        
        # self.gaussians.init_exposure_seq(self.train_cameras)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        #self.gaussians.save_mlp_checkpoints(point_cloud_path)
        #save_pose(scene.model_path + f'/pose/ours_{iteration}/pose_optimized.npy', self.gaussians.P, train_cams_init)
        #save(self.gaussians._mlp, os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration), "mlp_model.pt"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getPseudoCameras_wide(self, scale=1.0):
        if len(self.pseudo_cameras_wide) == 0:
            return [None]
        else:
            return self.pseudo_cameras_wide[scale]
        
    def getPseudoCameras_uw(self, scale=1.0):
        if len(self.pseudo_cameras_uw) == 0:
            return [None]
        else:
            return self.pseudo_cameras_uw[scale]
        
    def getShiftedCamera(self, camera, trans_dist=0.1):
        intrinsic, extrinsic = camera.get_camera_matrix()
        point = torch.tensor([trans_dist, 0.0, 0.0, 1.0], device="cuda")
        point_world = torch.inverse(extrinsic) @ point
        point_world = point_world[:3]
        camera_center_trans = (point_world - camera.camera_center).cpu().numpy()
        camera = Camera(
            colmap_id=camera.colmap_id,
            R=camera.R,
            T=camera.T,
            FoVx=camera.FoVx,
            FoVy=camera.FoVy,
            image=torch.ones_like(camera.original_image),
            gt_alpha_mask=None,
            image_name=None,
            uid=camera.uid,
            trans=camera_center_trans,
            data_device=camera.data_device
        )
        return camera