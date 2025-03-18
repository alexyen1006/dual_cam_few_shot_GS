def depth_to_pts3d(self, iteration, output_path):
        # Get depths and  projection params if not provided
        focals = self.get_focals()
        pp = self.get_principal_points()

        im_poses = self.get_im_poses()

        depth = self.get_depthmaps(raw=True)
        depth_num = depth.shape[0]
        for i in range(depth_num):
            if iteration % 50 == 0:
                img_name = self.view_mapping[i]
                assert len(img_name) == 1
                #breakpoint()
                img_name_str = next(iter(img_name))
                depth_img = depth[i]
                depth_img_copy = depth_img.clone()
                depth_np = depth_img_copy.detach().cpu().numpy()
                h,w = self.imshapes[i]
                depth_np = depth_np.reshape(h, w)
                # 正規化到 0-255 範圍
                depth_normalized = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX)
                depth_normalized = depth_normalized.astype(np.uint8)

                # 使用 cv2 的色彩映射
                depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_VIRIDIS)

                #breakpoint()
                output_colmap_path = Path(output_path)
                depth_img_path = output_colmap_path / 'depth_images'  # 使用 Path 的路徑連接方式

                # 創建目錄，exist_ok=True 表示如果目錄已存在不會報錯
                depth_img_path.mkdir(parents=True, exist_ok=True)

                # 使用 Path 來正確連接路徑，並使用 f-string 做字符串格式化
                output_file = depth_img_path / f'depthmap_for_{img_name_str}_{iteration}.png'
                cv2.imwrite(str(output_file), depth_colormap)
        # get pointmaps in camera frame
        rel_ptmaps = _fast_depthmap_to_pts3d(depth, self._grid, focals, pp=pp)

        return geotrf(im_poses, rel_ptmaps)
    
    
    #計算relative_pose:
    # 
    # 
    #breakpoint()
    #self.relative_pose = self._set_one_pose(R_avg_rel, T_avg_rel)
    #error :  "TypeError: cannot assign 'torch.FloatTensor' as parameter 'relative_pose' (torch.nn.Parameter or None expected)""
    



# def verify_relative_pose(self, uw_idx, wide_idx):


#     # 獲取地面真實的 Wide 相機位姿（用於比較）
#     E_wide_gt = self._get_one_pose(self.im_poses[wide_idx])
#     E_wide_computed, E_rel_wide_to_uw = self.get_corresponding_wide_pose(uw_idx)
    
#     # 計算位姿差異
#     pose_diff = torch.norm(E_wide_computed - E_wide_gt)
#     print(f"Pose difference: {pose_diff.item()}")
#     #breakpoint()
#     # 使用點雲驗證
#     i_j = edge_str(uw_idx, wide_idx)
#     j_i = edge_str(wide_idx, uw_idx)
#     pts_uw = self.pred_i[i_j]  # UW 相機觀察到的點雲 => 座標是uw
#     pts_wide = self.pred_i[j_i] # UW 相機觀察到的點雲 => 座標是wide
#     pts_uw_gt = self.pred_j[i_j] # Wide 相機觀察到的點雲 => 座標是uw
#     pts_wide_gt = self.pred_j[j_i]  # Wide 相機觀察到的點雲 => 座標是wide
     
    
#     # 獲取尺度因子
#     s_factor = self.get_pw_norm_scale_factor()
#     print(f"Scale factor: {s_factor.item()}")

#     # 方法1：應用尺度因子到相對位姿的平移部分
#     E_rel_wide_to_uw_scaled = E_rel_wide_to_uw.clone()
#     E_rel_wide_to_uw_scaled[:3, 3] *= s_factor
#     pts_uw_computed1 = geotrf(E_rel_wide_to_uw_scaled, pts_wide) 
#     #把 "UW 相機觀察到的點雲 => 座標是uw" 轉到 "UW 相機觀察到的點雲 => 座標是wide"
#     diff1 = torch.norm(pts_uw_computed1 - pts_uw, dim=-1).mean()
#     print(f"Method 1 - Scale transform translation: {diff1.item()}")
#     pts_uw_computed2 = geotrf(E_rel_wide_to_uw, pts_wide)
#     diff2 = torch.norm(pts_uw_computed2 - pts_uw, dim=-1).mean()
#     print(f"Method 2 - direct transform translation: {diff2.item()}")
    
    
#     # 檢查幾個特定點
#     for idx in range(5):
#         pt_flat_idx = idx * 100  # 隨機選擇點
#         pt_uw = pts_uw.reshape(-1, 3)[pt_flat_idx]
#         pt_wide = pts_wide.reshape(-1, 3)[pt_flat_idx]
#         pt_wide_computed = geotrf(E_rel_wide_to_uw, pt_wide.unsqueeze(0)).squeeze(0)
        
#         print(f"Point {idx}:")
#         print(f"  UW:            {pt_uw}")
#         print(f"  Wide (GT):     {pt_wide}")
#         print(f"  Wide (Comp):   {pt_wide_computed}")
#         print(f"  Difference:    {torch.norm(pt_wide - pt_wide_computed).item()}")
#         print()
    
#     #breakpoint()
#     return {
#         "pose_diff": pose_diff.item(),
#         "pts_diff1": diff1.item(),
#         "pts_diff2": diff2.item(),
#         "E_wide_gt": E_wide_gt,
#         "E_wide_computed": E_wide_computed
#     }