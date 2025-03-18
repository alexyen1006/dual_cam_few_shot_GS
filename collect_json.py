import os
import json
import csv

root_path = "/home/yenalex/dual_cam_few_shot_v0/output/ZoomGS"
scenes = os.listdir(root_path)
#把training-times.txt 拔掉
scenes = [s for s in scenes if s.isdigit()]  # 只保留数字文件夹名
print(scenes)
scenes = sorted(scenes, key=lambda x: int(x))
# 創建並開啟 CSV 檔案
csv_file = 'scene_metrics.csv'
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    # 寫入標題行
    writer.writerow(['Scene', 'PSNR', 'SSIM', 'LPIPS'])
    
    # 遍歷每個場景
    for scene in scenes:
        json_file = os.path.join(root_path, scene,'6_views' ,'results.json')
        if os.path.exists(json_file):
            # 讀取 JSON 檔案
            with open(json_file, 'r') as jf:
                data = json.load(jf)
                
                # 獲取 ours_15000 的指標數據
                metrics = data['ours_10000']
                psnr = metrics['PSNR']
                ssim = metrics['SSIM']
                lpips = metrics['LPIPS']
                
                # 寫入該場景的數據
                writer.writerow([scene, psnr, ssim, lpips])

print(f"Results have been saved to {csv_file}")